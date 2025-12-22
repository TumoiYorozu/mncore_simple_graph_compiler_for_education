#!/usr/bin/env python3

import torch
import torch.nn as nn
import numpy as np
import os
import shutil
import inspect
import onnx
import onnx.helper as helper
from onnx import shape_inference, TensorProto
from dataclasses import dataclass
from typing import Dict, List, Any, Tuple, Optional
from collections import OrderedDict, defaultdict
from fx_export.operators import get_operator

# ============= Export関連の関数 =============

def process_ignore_outputs(graph, ignore_outputs):
    """
    ignore_outputsに基づいてグラフから不要な出力を削除し、保持する出力を返す
    """
    outputs_to_keep = None
    
    if ignore_outputs:
        print(f"\n指定された出力を削除中: {ignore_outputs}")
        
        # 削除する出力を保存しているsaveノードを削除
        removed_nodes = set()
        for node_id, node in list(graph.nodes.items()):
            if node.op_type == 'save' and len(node.inputs) > 0:
                saved_var = node.inputs[0]
                if saved_var in ignore_outputs:
                    print(f"  出力 '{saved_var}' を保存するノード '{node_id}' を削除")
                    removed_nodes.add(node_id)
                    del graph.nodes[node_id]
        
        # 保持したい出力を特定
        outputs_to_keep = set()
        for node_id, node in graph.nodes.items():
            if node.op_type == 'save':
                outputs_to_keep.update(node.inputs)
        
        print(f"  保持する出力: {outputs_to_keep}")
    else:
        # ignore_outputsが指定されていない場合も、saveノードから出力を特定
        outputs_to_keep = set()
        for node_id, node in graph.nodes.items():
            if node.op_type == 'save':
                outputs_to_keep.update(node.inputs)
    
    return outputs_to_keep

def build_onnx_and_outputs(func, dummy_inputs, output_prefix="/tmp", ignore_outputs=None, decompose=True):
    """
    関数からONNXモデルとダミー出力を生成する（モデル付き・モデルフリー両対応）
    
    Args:
        func: エクスポートする関数
        dummy_inputs: ダミー入力
        output_prefix: 出力ディレクトリのプレフィックス
        ignore_outputs: 無視する出力
        decompose: 複雑なオペレーターを基本的な演算に分解するかどうか
    
    Returns:
        tuple: (onnx_path, expected_outputs_list, all_inputs_list)
        - expected_outputs_list: [(name, tensor), ...] の順序付きリスト
        - all_inputs_list: [(name, tensor), ...] の順序付きリスト
    """
    # 関数のクロージャにモデルがあるかチェック
    closure_vars = inspect.getclosurevars(func)
    has_model = False
    
    for _, var_value in closure_vars.nonlocals.items():
        if isinstance(var_value, nn.Module):
            has_model = True
            break
    
    # グローバル変数もチェック
    if not has_model:
        for _, var_value in closure_vars.globals.items():
            if isinstance(var_value, nn.Module):
                has_model = True
                break
    
    # グローバル変数をチェック
    for var_name, var_value in closure_vars.globals.items():
        if var_name == 'model' and isinstance(var_value, nn.Module):
            has_model = True
            break
    
    # 出力ディレクトリを作成
    os.makedirs(output_prefix, exist_ok=True)
    onnx_path = os.path.join(output_prefix, "model.onnx")
    
    if has_model:
        # モデル付き関数の場合
        print("モデル付き関数を検出しました。")
        
        # モデル参照を取得
        model_ref = None
        for _, var_value in closure_vars.nonlocals.items():
            if isinstance(var_value, nn.Module):
                model_ref = var_value
                break
        for var_name, var_value in closure_vars.globals.items():
            if isinstance(var_value, nn.Module) and var_name == 'model':
                model_ref = var_value
                break
        
        if model_ref is None:
            raise ValueError("関数クロージャ内にモデルが見つかりませんでした")
        
        # パラメータを保存
        param_before = {}
        for name, param in model_ref.named_parameters():
            param_before[name] = param.data.clone()
        
        # 自動微分ベースのグラフ構築（訓練・推論の両方で使用）
        print("自動微分ベースのグラフエクスポートを使用します。")
        
        debug = os.environ.get('DEBUG_AUTOGRAD', '').lower() == 'true'
        graph, metadata = build_computation_graph_autograd(func, dummy_inputs, model_ref, debug=debug)
        
        # パラメータを復元
        with torch.no_grad():
            for name, param in model_ref.named_parameters():
                param.data.copy_(param_before[name])
        
        # ignore_outputsの処理
        outputs_to_keep = process_ignore_outputs(graph, ignore_outputs)
        
        # ONNXにエクスポート
        skip_nodes = set()
        export_graph_to_onnx(graph, metadata, onnx_path, skip_nodes, decompose=decompose)
        
        # 実際の入力データを生成（これを保存して使い回す）
        actual_inputs = {}
        for key, tensor in dummy_inputs.items():
            # ランダムなデータを生成
            if key in ['target', 't']:
                actual_inputs[key] = torch.randint_like(tensor, 0, 10)
            else:
                actual_inputs[key] = torch.randn_like(tensor)
        
        # 実際の入力で期待される出力を計算
        # 訓練関数では勾配計算が必要なため、no_gradを使わない
        expected_outputs = func(actual_inputs)
        
        # train_stepの場合、勾配と更新されたパラメータを個別の出力として追加
        if func.__name__ == 'train_step':
            # 逆伝播後にモデルから勾配を取得
            for name, param in model_ref.named_parameters():
                if param.grad is not None:
                    grad_name = f'grad_{name.replace(".", "_")}'
                    expected_outputs[grad_name] = param.grad.clone()
            
            # 更新されたパラメータを取得（optimizer.step()の後）
            for name, param in model_ref.named_parameters():
                updated_name = f'updated_{name.replace(".", "_")}'
                expected_outputs[updated_name] = param.data.clone()
        
        # すべての入力を収集（実際の入力 + 実行前のモデルパラメータ）
        # ONNXの入力順序に従ってリストを構築
        all_inputs_list = []
        
        # まず実際の入力を追加（ONNXの期待する順序で）
        # autograd builderは決まった順序で入力を作成: input, target (存在する場合), パラメータ
        if 'x' in actual_inputs:
            all_inputs_list.append(('input', actual_inputs['x']))
        if 't' in actual_inputs:
            all_inputs_list.append(('target', actual_inputs['t']))
        
        # 次にモデルパラメータを追加（順序はnamed_parametersの順序）
        for name, param_value in param_before.items():
            onnx_name = name.replace('.', '_')
            all_inputs_list.append((onnx_name, param_value))
        
        # 期待される出力もリストに変換
        # ONNXの出力順序に合わせる必要がある
        # ONNXから順序を読み取る
        model = onnx.load(onnx_path)
        onnx_output_order = [out.name for out in model.graph.output]
        
        # ONNXの順序に従って出力リストを構築
        expected_outputs_list = []
        for name in onnx_output_order:
            if name in expected_outputs:
                expected_outputs_list.append((name, expected_outputs[name]))
    else:
        # モデルフリー関数の場合
        print("モデルフリー関数を検出しました。シンプルなトレースを使用します。")
        
        # 実際の入力データを生成
        actual_inputs = {}
        for key, tensor in dummy_inputs.items():
            # ランダムなデータを生成
            if key in ['target', 't']:
                actual_inputs[key] = torch.randint_like(tensor, 0, 10)
            else:
                actual_inputs[key] = torch.randn_like(tensor)
        
        # 関数を実行して出力を取得
        with torch.no_grad():
            expected_outputs = func(actual_inputs)
        
        # 入力と出力の情報を取得（ソート済みの順序を使用）
        input_names = list(sorted(dummy_inputs.keys()))
        output_names = list(sorted(expected_outputs.keys()))
        
        # モデルフリーの場合、実際の入力をリストとして作成
        all_inputs_list = [(name, actual_inputs[name]) for name in input_names]
        
        # 期待される出力もリストに変換
        expected_outputs_list = [(name, expected_outputs[name]) for name in output_names]
        
        # タプル入出力のラッパー関数を作成
        def tuple_wrapper(*args):
            inputs_dict = {name: args[i] for i, name in enumerate(input_names)}
            outputs_dict = func(inputs_dict)
            return tuple(outputs_dict[name] for name in output_names)
        
        # 実際の入力をタプルに変換
        actual_input_tuple = tuple(actual_inputs[k] for k in input_names)
        
        # torch.jit.traceを使用
        traced = torch.jit.trace(tuple_wrapper, actual_input_tuple)
        
        # 入力形状を取得
        input_shapes = {}
        for name, tensor in dummy_inputs.items():
            input_shapes[name] = list(tensor.shape)
        
        # ONNXにエクスポート（動的軸を使わず、具体的な形状を保持）
        torch.onnx.export(
            traced,
            actual_input_tuple,
            onnx_path,
            input_names=input_names,
            output_names=output_names,
            # dynamic_axesを削除して具体的な形状を保持
            opset_version=11
        )
        
        # MatMulノードをGemmに置換（モデルフリーの場合）
        model = onnx.load(onnx_path)
        new_nodes = []
        for node in model.graph.node:
            if node.op_type == 'MatMul':
                # MatMulをGemmに変換
                gemm_node = helper.make_node(
                    'Gemm',
                    inputs=list(node.input),
                    outputs=list(node.output),
                    name=node.name if node.name else f'Gemm_{len(new_nodes)}',
                    alpha=1.0,
                    beta=0.0,
                    transA=0,
                    transB=0
                )
                new_nodes.append(gemm_node)
            else:
                new_nodes.append(node)
        
        # 新しいグラフを作成
        new_graph = helper.make_graph(
            new_nodes,
            model.graph.name,
            model.graph.input,
            model.graph.output,
            model.graph.initializer,
            value_info=model.graph.value_info
        )
        
        # 新しいモデルを作成
        new_model = helper.make_model(new_graph)
        for opset in model.opset_import:
            new_opset = new_model.opset_import.add()
            new_opset.CopyFrom(opset)
        
        # 置換されたモデルを保存
        onnx.save(new_model, onnx_path)
        
    return onnx_path, expected_outputs_list, all_inputs_list

def export_function(func, dummy_inputs, output_dir, ignore_outputs=None, decompose=True):
    """
    PyTorch関数をONNXとテストデータにエクスポートする。
    
    Args:
        func: エクスポートする関数
        dummy_inputs: 入力テンソルの辞書
        output_dir: 出力ディレクトリ
        ignore_outputs: エクスポートから除外する出力名のリスト
        decompose: 複雑なオペレーターを基本的な演算に分解するかどうか
    
    Returns:
        tuple: (output_dir, all_inputs_list, expected_outputs_list)
    """
    
    # モデルをチェックする前にグローバル変数を初期化
    if 'init_globals' in globals():
        init_globals()
    
    # /tmp以下のディレクトリの場合、既存ファイルをクリーンアップ
    if output_dir.startswith('/tmp/') and os.path.exists(output_dir):
        # ディレクトリ内のファイルとサブディレクトリを削除
        for filename in os.listdir(output_dir):
            file_path = os.path.join(output_dir, filename)
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        print(f"既存ファイルをクリーンアップしました: {output_dir}")
    
    # ディレクトリが存在しない場合は作成
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. ONNXとダミー出力を生成
    print("=== ONNXエクスポート ===")
    onnx_path, expected_outputs_list, all_inputs_list = build_onnx_and_outputs(
        func, dummy_inputs, output_dir, ignore_outputs, decompose
    )
    
    # 入力と出力データを保存
    for i, (name, tensor) in enumerate(all_inputs_list):
        np.save(os.path.join(output_dir, f"input_{i}.npy"), tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy())
    for i, (name, tensor) in enumerate(expected_outputs_list):
        np.save(os.path.join(output_dir, f"output_{i}.npy"), tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy())
    
    print(f"\nエクスポート完了: {output_dir}")
    print(f"  - model.onnx")
    print(f"  - input_*.npy ({len(all_inputs_list)} files)")
    print(f"  - output_*.npy ({len(expected_outputs_list)} files)")
    print(f"\nコード生成するには以下を実行:")
    print(f"  ./haribote_graph_compiler.py compile {os.path.join(output_dir, 'model.onnx')}")
    print(f"\nユニットテストを生成するには以下を実行:")
    print(f"  ./haribote_graph_compiler.py build_unit_tests {output_dir}")
    print(f"\nコード生成とテストを行うには以下を実行:")
    print(f"  ./haribote_graph_compiler.py test {output_dir}")
    
    return output_dir, all_inputs_list, expected_outputs_list

@dataclass
class ComputeNode:
    """計算グラフ内のノードを表す"""
    node_id: str
    op_type: str
    inputs: List[str]
    outputs: List[str]
    params: Dict[str, Any]

class ComputationGraph:
    """計算全体をDAG（有向非巡回グラフ）として表す"""

    def __init__(self):
        self.nodes: OrderedDict[str, ComputeNode] = OrderedDict()
        self.node_counter = 0

    def add_node(self, op_type: str, inputs: List[str], outputs: List[str], params: Optional[Dict[str, Any]] = None) -> str:
        """グラフにノードを追加してそのIDを返す"""
        node_id = f"{op_type}_{self.node_counter}"
        self.node_counter += 1

        node = ComputeNode(
            node_id=node_id,
            op_type=op_type,
            inputs=inputs,
            outputs=outputs,
            params=params or {}
        )
        self.nodes[node_id] = node
        return node_id

    def topological_sort(self) -> List[ComputeNode]:
        """ノードをトポロジカル順序で返す"""
        # 隣接リストと入次数を作成
        in_degree = {}
        adj_list = defaultdict(list)
        output_to_node = {}  # 出力名から生成ノードへのマップ
        
        # 1回目: output_to_node を構築
        for node_id, node in self.nodes.items():
            for output in node.outputs:
                output_to_node[output] = node_id
        
        # 2回目: グラフ構築と入次数カウント
        for node_id, node in self.nodes.items():
            in_degree[node_id] = 0
            
        for node_id, node in self.nodes.items():
            for input_name in node.inputs:
                if input_name in output_to_node:
                    producer_id = output_to_node[input_name]
                    adj_list[producer_id].append(node_id)
                    in_degree[node_id] = in_degree.get(node_id, 0) + 1
        
        # トポロジカルソートはKahn's algorithmを使用
        queue = [node_id for node_id, degree in in_degree.items() if degree == 0]
        result = []
        
        while queue:
            node_id = queue.pop(0)
            result.append(self.nodes[node_id])
            
            for neighbor_id in adj_list[node_id]:
                in_degree[neighbor_id] -= 1
                if in_degree[neighbor_id] == 0:
                    queue.append(neighbor_id)
        
        # 全ノードを訪問できていない場合は残り（非連結成分）を追加
        if len(result) < len(self.nodes):
            visited_ids = {node.node_id for node in result}
            for node_id, node in self.nodes.items():
                if node_id not in visited_ids:
                    result.append(node)
        
        return result

class AutogradGraphBuilder:
    """PyTorchのautogradを分析して計算グラフを構築する"""

    def __init__(self, model_ref, debug=False):
        self.model_ref = model_ref
        self.graph = ComputationGraph()
        self.var_counter = defaultdict(int)
        self.tensor_to_var = {}  # テンソルIDから変数名へのマッピング
        self.param_to_var = {}  # パラメータ名から変数名へのマッピング
        self.saved_tensors = {}  # 逆伝播に必要なテンソルを保存
        self.debug = debug  # デバッグ出力を制御

    def get_var_name(self, base_name):
        """一意の変数名を生成する"""
        count = self.var_counter[base_name]
        self.var_counter[base_name] += 1
        if count == 0:
            return base_name
        return f"{base_name}_{count}"

    def build_graph(self, func, dummy_inputs):
        """順伝播を実行して逆伝播を分析することで完全な計算グラフを構築する"""

        # model_refを検証
        if self.model_ref is None:
            raise ValueError("model_ref is None - cannot build computation graph")
        
        # 次元を取得
        input_shape = list(dummy_inputs['x'].shape)
        batch_size = input_shape[0]
        input_dim = torch.prod(torch.tensor(input_shape[1:])).item()

        # 層情報を抽出
        layers = []
        for name, module in self.model_ref.named_modules():
            if isinstance(module, nn.Linear):
                layers.append({
                    'name': name,
                    'module': module,
                    'weight_shape': list(module.weight.shape),
                    'has_bias': module.bias is not None,
                    'bias_shape': list(module.bias.shape) if module.bias is not None else None
                })

        if not layers:
            raise ValueError("モデルに層が見つかりません。計算グラフの構築に失敗しました。")
        num_classes = layers[-1]['weight_shape'][0]

        # loadノードを追加
        self.graph.add_node('load', [], ['input'], {'shape': (batch_size, input_dim), 'ptr': 'input_ptr'})
        
        # 入力に含まれる場合のみターゲットノードを追加（訓練用）
        has_target = "t" in dummy_inputs
        if has_target:
            self.graph.add_node('load', [], ['target'], {'shape': (batch_size,), 'ptr': 'target_ptr', 'dtype': 'int'})

        # パラメータを読み込む
        for layer in layers:
            name = layer['name']
            self.graph.add_node('load', [], [f'{name}_weight'], {
                'shape': tuple(layer['weight_shape']),
                'ptr': f'{name}_weight_ptr'
            })
            self.param_to_var[f"{name}.weight"] = f'{name}_weight'

            if layer['has_bias']:
                self.graph.add_node('load', [], [f'{name}_bias'], {
                    'shape': tuple(layer['bias_shape']),
                    'ptr': f'{name}_bias_ptr'
                })
                self.param_to_var[f"{name}.bias"] = f'{name}_bias'

        # テンソルの流れを追跡するために順伝播にフック
        handles = []
        self.forward_trace = []  # 順伝播操作を追跡

        def make_pre_hook(layer_name):
            def hook(_, input):
                # 入力テンソルを保存
                if len(input) > 0:
                    self.saved_tensors[f'{layer_name}_input'] = input[0]
            return hook

        def make_post_hook(layer_name):
            def hook(module, _, output):
                # 出力テンソルとそのgrad_fnを追跡
                var_name = f"{layer_name}_pre"
                self.tensor_to_var[id(output)] = var_name
                self.saved_tensors[var_name] = output

                # 操作を記録
                self.forward_trace.append(('linear', layer_name, module))
            return hook

        # 線形層用のフックを登録
        for layer in layers:
            name = layer['name']
            module = layer['module']

            handle = module.register_forward_pre_hook(make_pre_hook(name))
            handles.append(handle)
            handle = module.register_forward_hook(make_post_hook(name))
            handles.append(handle)

        # ReLU操作用のフック
        if hasattr(self.model_ref, 'relu'):
            def relu_hook(_, input, output):
                # このReLUがどの層に属しているかを見つける
                input_tensor = input[0]
                for var_id, var_name in self.tensor_to_var.items():
                    if var_id == id(input_tensor) and var_name.endswith('_pre'):
                        layer_name = var_name.replace('_pre', '')
                        output_var = layer_name

                        self.tensor_to_var[id(output)] = output_var
                        self.saved_tensors[output_var] = output

                        self.forward_trace.append(('relu', layer_name, module))
                        break

            handle = self.model_ref.relu.register_forward_hook(relu_hook)
            handles.append(handle)

        # 入力を準備
        x = dummy_inputs["x"].requires_grad_(True)
        t = dummy_inputs.get("t", None)  # ターゲットは推論ではオプショナル
        input_tensor = x.view(batch_size, -1)
        self.tensor_to_var[id(input_tensor)] = 'input'
        self.saved_tensors['input'] = input_tensor

        # 実行前のパラメータ状態を取得してgrad_fnを追跡
        param_before = {}
        for name, param in self.model_ref.named_parameters():
            param_before[name] = param.data.clone()

        # 順伝播を実行
        func_inputs = {"x": x}
        if t is not None:
            func_inputs["t"] = t
        outputs = func(func_inputs)
        
        # 推論ではlossはオプショナル
        loss = outputs.get("loss", None)

        # トレースから順伝播グラフを構築
        prev_output = 'input'
        for op_type, layer_name, _ in self.forward_trace:
            if op_type == 'linear':
                layer = next(layer for layer in layers if layer['name'] == layer_name)
                inputs = [prev_output, f'{layer_name}_weight']
                if layer['has_bias']:
                    inputs.append(f'{layer_name}_bias')

                output_var = f'{layer_name}_pre'
                self.graph.add_node('linear', inputs, [output_var], {
                    'in_features': layer['weight_shape'][1],
                    'out_features': layer['weight_shape'][0],
                    'has_bias': layer['has_bias']
                })
                prev_output = output_var

            elif op_type == 'relu':
                input_var = f'{layer_name}_pre'
                output_var = layer_name
                self.graph.add_node('relu', [input_var], [output_var], {})
                prev_output = output_var

        # 出力用のエイリアスを追加
        self.graph.add_node('alias', [f'{layers[-1]["name"]}_pre'], ['output'], {
            'shape': (batch_size, num_classes)
        })
        
        # 推論の場合、要求されればsoftmax確率を計算
        if not has_target and "probs" in outputs:
            self.graph.add_node('softmax', ['output'], ['probs'], {
                'dim': 1
            })

        # ターゲットがある場合のみloss計算を追加（訓練モード）
        if has_target and loss is not None:
            self.graph.add_node('cross_entropy_forward', ['output', 'target'], ['loss', 'probs'], {
                'batch_size': batch_size,
                'num_classes': num_classes
            })
            
            # grad_fnを辿って逆伝播グラフを分析
            self._analyze_backward_autograd(loss, layers, batch_size, num_classes)

        # オプティマイザの更新をチェック
        optimizer_detected = False
        learning_rate = None

        for name, param in self.model_ref.named_parameters():
            if not torch.allclose(param_before[name], param.data):
                optimizer_detected = True
                if learning_rate is None and param.grad is not None:
                    update = param_before[name] - param.data
                    lr_estimate = (update / param.grad).abs().mean().item()
                    learning_rate = lr_estimate

        # 検出された場合はオプティマイザノードを追加
        if optimizer_detected and learning_rate:
            for layer in layers:
                name = layer['name']
                # 重みの更新
                self.graph.add_node('sgd_update', [f'{name}_weight', f'grad_{name}_weight'],
                                    [f'updated_{name}_weight'], {'learning_rate': learning_rate})
                # バイアスの更新
                if layer['has_bias']:
                    self.graph.add_node('sgd_update', [f'{name}_bias', f'grad_{name}_bias'],
                                        [f'updated_{name}_bias'], {'learning_rate': learning_rate})

        # saveノードを追加
        self.graph.add_node('save', ['output'], [], {'ptr': 'output_ptr'})
        
        # 推論用にprobsを保存
        if not has_target and "probs" in outputs:
            self.graph.add_node('save', ['probs'], [], {'ptr': 'probs_ptr'})
        
        # 計算した場合のみlossを保存（訓練モード）
        if has_target and loss is not None:
            self.graph.add_node('save', ['loss'], [], {'ptr': 'loss_ptr', 'scalar': True})

        # 逆伝播を実行した場合のみ勾配を保存（訓練モード）
        if has_target and loss is not None:
            for layer in layers:
                name = layer['name']
                self.graph.add_node('save', [f'grad_{name}_weight'], [], {'ptr': f'grad_{name}_weight_ptr'})
                if layer['has_bias']:
                    self.graph.add_node('save', [f'grad_{name}_bias'], [], {'ptr': f'grad_{name}_bias_ptr'})

        # 更新されたパラメータを保存
        if optimizer_detected:
            for layer in layers:
                name = layer['name']
                self.graph.add_node('save', [f'updated_{name}_weight'], [], {'ptr': f'updated_{name}_weight_ptr'})
                if layer['has_bias']:
                    self.graph.add_node('save', [f'updated_{name}_bias'], [], {'ptr': f'updated_{name}_bias_ptr'})

        # フックをクリーンアップ
        for handle in handles:
            handle.remove()

        # 実行前の状態にモデルパラメータを復元
        # train_stepがoptimizer.step()を呼び出してパラメータを変更するため、これは重要
        with torch.no_grad():
            for name, param in self.model_ref.named_parameters():
                param.data.copy_(param_before[name])

        return {
            'batch_size': batch_size,
            'input_dim': input_dim,
            'num_classes': num_classes,
            'layers': layers,
            'optimizer_detected': optimizer_detected,
            'learning_rate': learning_rate
        }

    def _analyze_backward_autograd(self, loss, layers, batch_size, num_classes):
        """PyTorchのautogradグラフを辿って逆伝播グラフを分析する"""

        # grad_fnを生成する必要がある操作にマッピング
        grad_fn_to_op = {}
        param_to_grad_fn = {}  # パラメータ名をAccumulateGradノードにマッピング

        # 最初のパス: すべてのgrad関数をマッピング
        def map_grad_fns(tensor, visited=None):
            if visited is None:
                visited = set()

            if not hasattr(tensor, 'grad_fn') or tensor.grad_fn is None:
                return

            grad_fn = tensor.grad_fn
            if id(grad_fn) in visited:
                return

            visited.add(id(grad_fn))
            grad_fn_type = type(grad_fn).__name__

            # タイプとコンテキストに基づいてgrad_fnをマッピング
            if 'NllLoss' in grad_fn_type:
                grad_fn_to_op[id(grad_fn)] = 'nll_loss_backward'
            elif 'LogSoftmax' in grad_fn_type:
                grad_fn_to_op[id(grad_fn)] = 'log_softmax_backward'
            elif 'Addmm' in grad_fn_type or 'Linear' in grad_fn_type:
                # すべてのAddmmをlinear backwardとしてマーク
                grad_fn_to_op[id(grad_fn)] = 'linear_backward'
            elif 'Relu' in grad_fn_type:
                grad_fn_to_op[id(grad_fn)] = 'relu_backward'
            elif 'AccumulateGrad' in grad_fn_type:
                # これがどのパラメータに属しているかを見つける
                for name, param in self.model_ref.named_parameters():
                    if hasattr(param, 'grad_fn') and param.grad_fn is grad_fn:
                        param_to_grad_fn[name] = grad_fn
                        break

            # 次の関数を辿る
            if hasattr(grad_fn, 'next_functions'):
                for next_fn, _ in grad_fn.next_functions:
                    if next_fn is not None:
                        # 辿るためのダミーテンソルを作成
                        map_grad_fns(type('dummy', (), {'grad_fn': next_fn})(), visited)

        # lossから始めてすべてのgrad関数をマッピング
        map_grad_fns(loss)

        # デバッグ: 見つかったものを出力
        if self.debug:
            print("\nGrad functionのマッピング:")
            for grad_id, op in grad_fn_to_op.items():
                print(f"  {grad_id}: {op}")

        # 2番目のパス: 順番に辿って逆伝播グラフを構築
        visited_backward = set()
        grad_output_mapping = {}  # grad_fnをその出力勾配変数にマッピング

        def traverse_and_build(grad_fn, grad_output_var=None):
            if grad_fn is None:
                return
            if id(grad_fn) in visited_backward:
                return

            grad_fn_id = id(grad_fn)
            visited_backward.add(grad_fn_id)
            grad_fn_type = type(grad_fn).__name__
            if self.debug:
                print(f"\nトラバース中: {grad_fn_type} (id={grad_fn_id})")

            # 操作タイプに基づいて処理
            if 'NllLoss' in grad_fn_type:
                # これはloss - cross entropy backwardを開始
                # Cross entropy backwardはNllLossとLogSoftmaxを組み合わせる
                self.graph.add_node('cross_entropy_backward', ['probs', 'target'],
                                    ['grad_output'], {
                                        'batch_size': batch_size,
                                        'num_classes': num_classes
                                    })

                # 次の関数はLogSoftmaxであるべき、勾配を通す
                if hasattr(grad_fn, 'next_functions'):
                    for next_fn, _ in grad_fn.next_functions:
                        if next_fn and 'LogSoftmax' in type(next_fn).__name__:
                            grad_output_mapping[id(next_fn)] = 'grad_output'
                            traverse_and_build(next_fn, 'grad_output')

            elif 'LogSoftmax' in grad_fn_type:
                # すでにcross_entropyで処理済み、勾配を通すだけ
                if hasattr(grad_fn, 'next_functions'):
                    for next_fn, _ in grad_fn.next_functions:
                        if next_fn:
                            grad_output_mapping[id(next_fn)] = grad_output_var
                            traverse_and_build(next_fn, grad_output_var)

            elif 'Addmm' in grad_fn_type or (grad_fn_id in grad_fn_to_op and grad_fn_to_op[grad_fn_id] == 'linear_backward'):
                # Linear backward - どの層かを決定
                # これまでに見たAddmm操作の数をカウント
                layer_idx = None
                layer_name = None

                # すでに処理したlinear_backward操作の数をカウント
                # 注意: 現在のgrad_fnはすでにvisited_backwardに含まれている
                addmm_count = sum(1 for visited_id in visited_backward
                                  if visited_id in grad_fn_to_op
                                  and grad_fn_to_op[visited_id] == 'linear_backward'
                                  and visited_id != grad_fn_id)

                if self.debug:
                    print(f"  addmm_count (excluding current)={addmm_count}, len(layers)={len(layers)}")
                    print(f"  layers: {[layer['name'] for layer in layers]}")

                # Addmm操作を逆順で層にマッピング
                # 逆伝播では最後の層が最初に処理される
                if addmm_count < len(layers):
                    layer_idx = len(layers) - addmm_count - 1
                    layer_name = layers[layer_idx]['name']

                if layer_idx is not None:
                    layer = layers[layer_idx]
                    if self.debug:
                        print(f"  layerとして識別: {layer_name}")

                    # 入力勾配変数を取得
                    input_grad = grad_output_mapping.get(grad_fn_id, grad_output_var)
                    if not input_grad:
                        input_grad = f'grad_{layer_name}_out'

                    # 層の入力を決定
                    if layer_idx == 0:
                        layer_input = 'input'
                    else:
                        layer_input = layers[layer_idx-1]['name']

                    # 出力を作成
                    outputs = [f'grad_{layer_name}_weight', f'grad_{layer_name}_bias']
                    if layer_idx > 0:
                        outputs.append(f'grad_{layer_name}_input')

                    self.graph.add_node('linear_backward',
                                        [input_grad, layer_input, f'{layer_name}_weight'],
                                        outputs, {
                                            'in_features': layer['weight_shape'][1],
                                            'out_features': layer['weight_shape'][0],
                                            'has_bias': layer['has_bias'],
                                            'compute_input_grad': layer_idx > 0
                                        })

                # 辿りを続ける
                if layer_idx is not None and layer_idx > 0 and hasattr(grad_fn, 'next_functions'):
                    for next_fn, _ in grad_fn.next_functions:
                        if next_fn and 'Accumulate' not in type(next_fn).__name__:
                            grad_output_mapping[id(next_fn)] = f'grad_{layer_name}_input'
                            traverse_and_build(next_fn, f'grad_{layer_name}_input')

            elif 'Relu' in grad_fn_type:
                if self.debug:
                    print(f"  ReLU backwardを処理中 (id={grad_fn_id})")
                
                # ReLU backwardのカウントを追跡
                relu_count = sum(1 for node in self.graph.nodes.values() if node.op_type == 'relu_backward')
                if self.debug:
                    print(f"  現在のReLU backward数: {relu_count}")
                    print(f"  現在のgraph node type: {[node.op_type for node in self.graph.nodes.values()]}")
                
                # モデルのLinear層を収集（順序を保持）
                linear_layers = []
                for name, module in self.model_ref.named_modules():
                    if isinstance(module, nn.Linear):
                        linear_layers.append(name)
                
                if self.debug:
                    print(f"  検出されたLinear層: {linear_layers}")
                
                # 最後のLinear層（出力層）を除外
                relu_layers = linear_layers[:-1]
                
                # どのReLU層かを判定（逆順なので、後ろの層から処理される）
                if relu_count > len(relu_layers):
                    raise ValueError(f"ReLU層のインデックスが範囲外です: relu_count={relu_count}, relu_layers={relu_layers}")
                
                # 逆順でインデックスを計算
                layer_idx = len(relu_layers) - 1 - relu_count
                layer_name = relu_layers[layer_idx]
                
                if self.debug:
                    print(f"  layerのReLU backward: {layer_name}")

                # 入力勾配を取得
                input_grad = grad_output_mapping.get(grad_fn_id, grad_output_var)
                if self.debug:
                    print(f"  grad_output_mapping[{grad_fn_id}] = {input_grad}")
                    print(f"  grad_output_var = {grad_output_var}")
                
                # grad_{layer_name}を出力として設定
                output_grad = f'grad_{layer_name}'
                
                self.graph.add_node('relu_backward',
                                    [input_grad, f'{layer_name}_pre'],
                                    [output_grad], {})

                # 辿りを続ける
                if hasattr(grad_fn, 'next_functions'):
                    for next_fn, _ in grad_fn.next_functions:
                        if next_fn:
                            grad_output_mapping[id(next_fn)] = output_grad
                            traverse_and_build(next_fn, output_grad)

            # 未処理の操作に対して常に辿りを続けようとする
            if hasattr(grad_fn, 'next_functions'):
                for next_fn, _ in grad_fn.next_functions:
                    if next_fn and id(next_fn) not in visited_backward:
                        traverse_and_build(next_fn, grad_output_var)

        # lossからトラバーサルを開始
        if self.debug:
            print(f"\nloss grad_fnからトラバーサル開始: {type(loss.grad_fn).__name__}")
        traverse_and_build(loss.grad_fn)

def build_computation_graph_autograd(func, dummy_inputs, model_ref, debug=False) -> Tuple[ComputationGraph, Dict[str, Any]]:
    """自動微分解析を使用して計算グラフを構築"""
    builder = AutogradGraphBuilder(model_ref, debug=debug)
    metadata = builder.build_graph(func, dummy_inputs)
    return builder.graph, metadata


# グローバル一時カウンター
_temp_counter = 0

def clean_unused_nodes_from_onnx(model):
    """ONNXモデルから出力につながっていないノードを削除"""
    graph = model.graph
    
    # 出力名のセットを作成
    output_names = {output.name for output in graph.output}
    
    # 各ノードが生成する出力を記録
    node_outputs = {}
    node_index_map = {}  # ノードをインデックスでマップ
    for i, node in enumerate(graph.node):
        node_index_map[i] = node
        for output in node.output:
            node_outputs[output] = i
    
    # 必要なノードを特定（出力から逆方向にたどる）
    required_node_indices = set()
    required_tensors = set(output_names)
    
    # 入力と初期化子も必要
    for input in graph.input:
        required_tensors.add(input.name)
    for init in graph.initializer:
        required_tensors.add(init.name)
    
    # 出力から逆方向にたどる
    changed = True
    while changed:
        changed = False
        new_required = set()
        
        for tensor_name in required_tensors:
            if tensor_name in node_outputs:
                node_idx = node_outputs[tensor_name]
                if node_idx not in required_node_indices:
                    required_node_indices.add(node_idx)
                    node = node_index_map[node_idx]
                    # このノードの入力も必要
                    for input_name in node.input:
                        if input_name not in required_tensors:
                            new_required.add(input_name)
                            changed = True
        
        required_tensors.update(new_required)
    
    # 不要なノードを特定
    all_node_indices = set(range(len(graph.node)))
    unused_node_indices = all_node_indices - required_node_indices
    
    if unused_node_indices:
        print(f"  ONNXから未使用ノードを削除:")
        for idx in unused_node_indices:
            node = node_index_map[idx]
            print(f"    {node.op_type} - outputs: {list(node.output)}")
        
        # 新しいグラフを作成（必要なノードのみ）
        new_nodes = [node_index_map[idx] for idx in sorted(required_node_indices)]
        
        # グラフのノードを置き換え
        del graph.node[:]
        graph.node.extend(new_nodes)
        
        print(f"  ONNXノード数: {len(all_node_indices)} -> {len(new_nodes)}")
    
    return model

def export_graph_to_onnx(graph: ComputationGraph, metadata: Dict[str, Any], output_path: str, skip_nodes: set, decompose: bool = True):
    """Export computation graph to ONNX format including forward, backward and optimizer
    
    Args:
        graph: 計算グラフ
        metadata: メタデータ
        output_path: 出力パス
        skip_nodes: スキップするノード
        decompose: 複雑なオペレーターを基本的な演算に分解するかどうか
    """
    
    # ONNXノードを作成
    onnx_nodes = []
    value_info = []
    initializers = []

    # テンソルの形状を追跡
    tensor_shapes = {}

    # value infoを作成するためのヘルパー
    def make_tensor_value_info(name, shape, dtype=TensorProto.FLOAT):
        tensor_shapes[name] = shape
        return helper.make_tensor_value_info(name, dtype, shape)

    # ノードに変数名属性を追加するためのヘルパー
    def add_var_name_attr(node, var_name):
        """より良いC++生成のためにONNXノードに変数名属性を追加"""
        if var_name and not var_name.startswith('temp_'):
            node.attribute.append(helper.make_attribute('var_name', var_name))
        return node
    
    # 再帰的にdecomposeを適用する関数
    def apply_decompose_recursively(nodes_list, max_iterations=10):
        """ノードリストに対して再帰的にdecomposeを適用"""
        
        for iteration in range(max_iterations):
            new_nodes = []
            graph_changed = False
            
            for node in nodes_list:
                # オペレータークラスを取得
                op_class = get_operator(node.op_type)
                
                if op_class and decompose:
                    # 仮のオペレーターインスタンスを作成
                    temp_operator = op_class(node, None, {})
                    decomposed_nodes = temp_operator.decompose()
                    
                    if decomposed_nodes:
                        # 分解された場合
                        graph_changed = True
                        for decomposed_node in decomposed_nodes:
                            if decomposed_node.output:
                                add_var_name_attr(decomposed_node, decomposed_node.output[0])
                            new_nodes.append(decomposed_node)
                    else:
                        # 分解されない場合はそのまま追加
                        new_nodes.append(node)
                else:
                    # decompose不可能またはdecomposeが無効な場合
                    new_nodes.append(node)
            
            nodes_list = new_nodes
            
            # グラフが変化しなければ終了
            if not graph_changed:
                break
            
            # 最大反復回数に達した場合
            if iteration == max_iterations - 1:
                raise RuntimeError(f"Decompose did not converge after {max_iterations} iterations")
        
        return nodes_list

    # グラフ内の各ノードを処理
    for node in graph.topological_sort():
        # スキップするノードかチェック
        if node.node_id in skip_nodes:
            if 'DEBUG_ONNX_EXPORT' in os.environ:
                print(f"ノードをスキップ: {node.node_id} ({node.op_type})")
            continue
            
        if 'DEBUG_ONNX_EXPORT' in os.environ:
            print(f"ノードを処理中: {node.op_type}: {node.inputs} -> {node.outputs}")
        try:
            if node.op_type == 'load':
                # 入力/初期化子を作成
                if node.outputs[0] == 'input':
                    shape = node.params['shape']
                    value_info.append(make_tensor_value_info('input', shape))
                elif node.outputs[0] == 'target':
                    shape = (node.params['shape'][0],)
                    value_info.append(make_tensor_value_info('target', shape, TensorProto.INT32))
                else:
                    # パラメータの読み込み - これらは初期化子になる
                    shape = node.params['shape']
                    tensor_shapes[node.outputs[0]] = shape

            elif node.op_type == 'linear':
                # 線形演算: Y = X @ W^T + b
                out_name = node.outputs[0]
                in_name = node.inputs[0]
                weight_name = node.inputs[1]
                
                assert node.params['has_bias']
                bias_name = node.inputs[2]
                # Gemmノード（バイアスなし）: matmul_result = X @ W^T
                matmul_out = f"{out_name}_matmul"
                gemm_node = helper.make_node(
                    'Gemm',
                    [in_name, weight_name],
                    [matmul_out],
                    transB=1,  # Wを転置
                    name=f"{out_name}_gemm"
                )
                add_var_name_attr(gemm_node, matmul_out)
                onnx_nodes.append(gemm_node)
                
                # Addノードでバイアスを追加: Y = matmul_result + bias
                add_node = helper.make_node(
                    'Add',
                    [matmul_out, bias_name],
                    [out_name],
                    name=f"{out_name}_add_bias"
                )
                add_var_name_attr(add_node, out_name)
                onnx_nodes.append(add_node)

            elif node.op_type == 'relu':
                out_name = node.outputs[0]
                relu_node = helper.make_node('Relu', node.inputs, node.outputs,
                                             name=f"{out_name}_relu")
                add_var_name_attr(relu_node, out_name)
                onnx_nodes.append(relu_node)

            elif node.op_type == 'alias':
                out_name = node.outputs[0]
                alias_node = helper.make_node('Identity', node.inputs, node.outputs,
                                              name=f"{out_name}_alias")
                add_var_name_attr(alias_node, out_name)
                onnx_nodes.append(alias_node)
            elif node.op_type == 'softmax':
                out_name = node.outputs[0]
                if 'dim' not in node.params:
                    raise ValueError(f"Softmax node {node.node_id} requires 'dim' parameter to specify the axis for softmax operation")
                
                # ONNXNodeProtoを作成
                softmax_onnx_node = helper.make_node(
                    'Softmax', node.inputs, node.outputs,
                    axis=node.params['dim'],
                    name=f"{out_name}_softmax"
                )
                add_var_name_attr(softmax_onnx_node, out_name)
                onnx_nodes.append(softmax_onnx_node)

            elif node.op_type == 'cross_entropy_forward':
                # Softmax + NLLLoss の組み合わせ
                # cross_entropyは通常最後の次元に対して適用される
                # バッチ次元は0、クラス次元は1と仮定
                if 'dim' in node.params:
                    axis = node.params['dim']
                else:
                    # パラメータから形状情報を推論
                    if 'batch_size' in node.params and 'num_classes' in node.params:
                        # 2次元入力を仮定: (batch_size, num_classes)
                        axis = 1
                    else:
                        # ONNXグラフから入力の形状を取得して推論
                        input_shape = tensor_shapes.get(node.inputs[0], None)
                        if input_shape is None:
                            raise ValueError(f"cross_entropy_forward node {node.node_id} requires either 'dim' parameter or shape information to determine softmax axis")
                        # 2次元の場合は次元1（クラス次元）、それ以外はエラー
                        if len(input_shape) == 2:
                            axis = 1
                        else:
                            raise ValueError(f"cross_entropy_forward node {node.node_id} expects 2D input (batch x classes), got shape {input_shape}")
                softmax_node = helper.make_node('Softmax', [node.inputs[0]], ['probs'], axis=axis)
                add_var_name_attr(softmax_node, 'probs')
                onnx_nodes.append(softmax_node)

                log_node = helper.make_node('Log', ['probs'], ['log_probs'])
                add_var_name_attr(log_node, 'log_probs')
                onnx_nodes.append(log_node)

                # NLLLoss の計算
                nll_node = helper.make_node('NegativeLogLikelihoodLoss', ['log_probs', node.inputs[1]],
                                            ['loss'], reduction='mean')
                add_var_name_attr(nll_node, 'loss')
                onnx_nodes.append(nll_node)

            elif node.op_type == 'cross_entropy_backward':
                # クロスエントロピーの勾配計算
                batch_size = node.params['batch_size']
                num_classes = node.params['num_classes']

                # depthを属性としてone-hotエンコーディングを作成
                onehot_node = helper.make_node('OneHot', ['target', 'depth', 'values'],
                                               ['target_onehot'], axis=1)
                onehot_node.attribute.append(helper.make_attribute('depth', num_classes))
                add_var_name_attr(onehot_node, 'target_onehot')
                onnx_nodes.append(onehot_node)

                # 勾配 = probs - target_onehot
                sub_node = helper.make_node('Sub', ['probs', 'target_onehot'], ['grad_before_scale'])
                add_var_name_attr(sub_node, 'grad_before_scale')
                onnx_nodes.append(sub_node)

                # 1/batch_sizeでスケール
                scale_const = f"scale_{batch_size}"
                initializers.append(
                    helper.make_tensor(scale_const, TensorProto.FLOAT, [], [1.0 / batch_size])
                )
                # Mulノードを作成（スカラーを2番目の入力として）
                mul_node = helper.make_node('Mul', ['grad_before_scale', scale_const], ['grad_output'])
                add_var_name_attr(mul_node, 'grad_output')
                onnx_nodes.append(mul_node)

                # OneHot用のdepthとvaluesを追加
                initializers.append(
                    helper.make_tensor('depth', TensorProto.INT32, [], [num_classes])
                )
                initializers.append(
                    helper.make_tensor('values', TensorProto.FLOAT, [2], [0.0, 1.0])
                )

            elif node.op_type == 'linear_backward':
                grad_out = node.inputs[0]
                layer_input = node.inputs[1]
                weight = node.inputs[2]
                grad_weight = node.outputs[0]
                grad_bias = node.outputs[1]


                # 重みに関する勾配: grad_weight = grad_out^T @ input
                # Gemmノードを使用: transA=1でgrad_outを転置
                gemm_node = helper.make_node(
                    'Gemm',
                    [grad_out, layer_input],
                    [grad_weight],
                    transA=1,  # grad_outを転置
                    name=f"{grad_weight}_gemm"
                )
                add_var_name_attr(gemm_node, grad_weight)
                onnx_nodes.append(gemm_node)

                # バイアスに関する勾配: バッチ次元で合計
                if node.params['has_bias']:
                    reduce_node = helper.make_node('ReduceSum', [grad_out], [grad_bias], axes=[0])
                    add_var_name_attr(reduce_node, grad_bias)
                    onnx_nodes.append(reduce_node)

                # 入力に関する勾配（必要な場合）
                if node.params['compute_input_grad']:
                    grad_input = node.outputs[2]
                    # MatMulの代わりにGemmを使用（transA=0, transB=0, alpha=1.0, beta=0.0）
                    gemm_node = helper.make_node('Gemm', [grad_out, weight], [grad_input],
                                                  transA=0, transB=0, alpha=1.0, beta=0.0)
                    add_var_name_attr(gemm_node, grad_input)
                    onnx_nodes.append(gemm_node)

            elif node.op_type == 'relu_backward':
                # ReLU逆伝播: grad_out * (input > 0)
                input_tensor = node.inputs[1]
                grad_in = node.inputs[0]
                grad_out = node.outputs[0]

                # input > 0のマスクを作成
                zero_const = f"zero_{node.node_id}"
                initializers.append(
                    helper.make_tensor(zero_const, TensorProto.FLOAT, [], [0.0])
                )

                greater_node = helper.make_node('Greater', [input_tensor, zero_const], [f"mask_{node.node_id}"])
                add_var_name_attr(greater_node, f"relu_mask_{input_tensor}")
                onnx_nodes.append(greater_node)

                # boolをfloatにキャスト
                cast_node = helper.make_node('Cast', [f"mask_{node.node_id}"], [f"mask_float_{node.node_id}"],
                                             to=TensorProto.FLOAT)
                add_var_name_attr(cast_node, f"relu_mask_float_{input_tensor}")
                onnx_nodes.append(cast_node)

                # 勾配にマスクを乗算
                mul_node = helper.make_node('Mul', [grad_in, f"mask_float_{node.node_id}"], [grad_out])
                add_var_name_attr(mul_node, grad_out)
                onnx_nodes.append(mul_node)

            elif node.op_type == 'sgd_update':
                param = node.inputs[0]
                grad = node.inputs[1]
                updated = node.outputs[0]
                lr = node.params['learning_rate']

                # 学習率定数を作成
                lr_const = f"lr_{node.node_id}"
                initializers.append(
                    helper.make_tensor(lr_const, TensorProto.FLOAT, [], [lr])
                )

                # 勾配をスケール（スカラーを2番目の入力として）
                mul_node = helper.make_node('Mul', [grad, lr_const], [f"{grad}_scaled"])
                add_var_name_attr(mul_node, f"{grad}_scaled")
                onnx_nodes.append(mul_node)

                # 更新: param - lr * grad
                sub_node = helper.make_node('Sub', [param, f"{grad}_scaled"], [updated])
                add_var_name_attr(sub_node, updated)
                # inplace更新のための属性を追加
                sub_node.attribute.append(helper.make_attribute("inplace_input", param))
                onnx_nodes.append(sub_node)

            elif node.op_type == 'matmul':
                # 行列乗算
                input1, input2 = node.inputs
                output = node.outputs[0]
                
                # 出力形状を推論
                if input1 in tensor_shapes and input2 in tensor_shapes:
                    shape1 = tensor_shapes[input1]
                    shape2 = tensor_shapes[input2]
                    
                    # 必要な場合は転置を処理
                    if node.params.get('transpose_a'):
                        shape1 = shape1[::-1]
                    if node.params.get('transpose_b'):
                        shape2 = shape2[::-1]
                    
                    output_shape = [shape1[0], shape2[1]] if len(shape1) == 2 and len(shape2) == 2 else []
                    tensor_shapes[output] = output_shape
                
                # MatMulの代わりにGemmを使用（transA=0, transB=0, alpha=1.0, beta=0.0）
                gemm_node = helper.make_node('Gemm', node.inputs, node.outputs,
                                             name=f"{output}_gemm",
                                             transA=0, transB=0, alpha=1.0, beta=0.0)
                add_var_name_attr(gemm_node, output)
                onnx_nodes.append(gemm_node)
                
            elif node.op_type == 'transpose':
                # 転置操作
                input_var = node.inputs[0]
                output = node.outputs[0]
                
                # デフォルトは2D転置
                transpose_node = helper.make_node('Transpose', [input_var], [output],
                                                  perm=[1, 0], name=f"{output}_transpose")
                add_var_name_attr(transpose_node, output)
                onnx_nodes.append(transpose_node)
                
                # 形状を更新
                if input_var in tensor_shapes:
                    shape = tensor_shapes[input_var]
                    tensor_shapes[output] = shape[::-1] if len(shape) == 2 else shape
                    
            elif node.op_type == 'reduce_sum':
                # Sum縮約
                input_var = node.inputs[0]
                output = node.outputs[0]
                
                axis = node.params.get('axis')
                keepdims = node.params.get('keepdims', False)
                
                if axis is not None:
                    reduce_node = helper.make_node('ReduceSum', [input_var], [output],
                                                   axes=[axis], keepdims=keepdims,
                                                   name=f"{output}_reduce_sum")
                else:
                    reduce_node = helper.make_node('ReduceSum', [input_var], [output],
                                                   keepdims=keepdims,
                                                   name=f"{output}_reduce_sum")
                add_var_name_attr(reduce_node, output)
                onnx_nodes.append(reduce_node)
                
            elif node.op_type == 'save':
                # 出力を作成
                var_name = node.inputs[0]
                if var_name in tensor_shapes:
                    shape = tensor_shapes[var_name]
                    dtype = TensorProto.FLOAT
                    if var_name == 'loss' and node.params.get('scalar'):
                        shape = []
                    value_info.append(helper.make_tensor_value_info(var_name, dtype, shape))

        except Exception as e:
            print(f"ノード {node.node_id} ({node.op_type}) の処理でエラー: {e}")
            print(f"ノード入力: {node.inputs}")
            print(f"ノード出力: {node.outputs}")
            print(f"ノードパラメータ: {node.params}")
            raise

    # グラフの入出力を作成
    graph_inputs = []
    graph_outputs = []

    # ターゲットが使用されているかチェック（訓練モード）
    has_target = any(node.outputs == ['target'] for node in graph.nodes.values() if node.op_type == 'load')
    
    # カスタム入力を持つモデルフリー関数を処理
    if 'input_shapes' in metadata:
        # モデルフリー関数 - すべての入力を追加
        for input_name, shape in metadata['input_shapes'].items():
            graph_inputs.append(helper.make_tensor_value_info(input_name, TensorProto.FLOAT, shape))
    else:
        # モデルベースの関数
        graph_inputs.append(helper.make_tensor_value_info('input', TensorProto.FLOAT,
                                                           [metadata['batch_size'], metadata['input_dim']]))
        
        # 使用されている場合のみターゲット入力を追加
        if has_target:
            graph_inputs.append(helper.make_tensor_value_info('target', TensorProto.INT32,
                                                               [metadata['batch_size']]))

    # パラメータ入力
    for layer in metadata['layers']:
        name = layer['name']
        weight_shape = layer['weight_shape']
        graph_inputs.append(helper.make_tensor_value_info(f'{name}_weight', TensorProto.FLOAT, weight_shape))
        if layer['has_bias']:
            bias_shape = layer['bias_shape']
            graph_inputs.append(helper.make_tensor_value_info(f'{name}_bias', TensorProto.FLOAT, bias_shape))

    # 勾配ノードを探して推論モードか訓練モードかをチェック
    has_gradients = any(node.op_type.endswith('_backward') for node in graph.nodes.values())
    has_loss = any('loss' in output for node in graph.nodes.values() if node.op_type == 'save' for output in node.inputs)
    
    # 出力を処理
    if 'output_shapes' in metadata:
        # モデルフリー関数 - すべての出力を追加
        for output_name, shape in metadata['output_shapes'].items():
            graph_outputs.append(helper.make_tensor_value_info(output_name, TensorProto.FLOAT, shape))
    else:
        # モデルベースの関数
        graph_outputs.append(helper.make_tensor_value_info('output', TensorProto.FLOAT,
                                                            [metadata['batch_size'], metadata['num_classes']]))
        
        # probs出力が存在する場合は追加（推論用）
        has_probs = any('probs' in output for node in graph.nodes.values() if node.op_type == 'save' for output in node.inputs)
        if has_probs:
            graph_outputs.append(helper.make_tensor_value_info('probs', TensorProto.FLOAT,
                                                                [metadata['batch_size'], metadata['num_classes']]))
    
    # lossが存在する場合のみ追加
    if has_loss:
        graph_outputs.append(helper.make_tensor_value_info('loss', TensorProto.FLOAT, []))

    # 勾配が存在する場合のみ勾配出力
    if has_gradients:
        if 'DEBUG_AUTOGRAD' in os.environ:
            print(f"勾配出力を追加中: has_gradients={has_gradients}, layers={metadata['layers']}")
        for layer in metadata['layers']:
            name = layer['name']
            weight_shape = layer['weight_shape']
            graph_outputs.append(helper.make_tensor_value_info(f'grad_{name}_weight', TensorProto.FLOAT, weight_shape))
            if layer['has_bias']:
                bias_shape = layer['bias_shape']
                graph_outputs.append(helper.make_tensor_value_info(f'grad_{name}_bias', TensorProto.FLOAT, bias_shape))

    # 更新されたパラメータ出力（オプティマイザが検出された場合）
    if metadata['optimizer_detected']:
        for layer in metadata['layers']:
            name = layer['name']
            weight_shape = layer['weight_shape']
            graph_outputs.append(helper.make_tensor_value_info(f'updated_{name}_weight', TensorProto.FLOAT, weight_shape))
            if layer['has_bias']:
                bias_shape = layer['bias_shape']
                graph_outputs.append(helper.make_tensor_value_info(f'updated_{name}_bias', TensorProto.FLOAT, bias_shape))

    # decomposeを再帰的に適用
    if decompose:
        onnx_nodes = apply_decompose_recursively(onnx_nodes)
    
    # 分解されたノードの形状情報を追加
    # (_sub, _exp, _sum, probs など)
    for node in onnx_nodes:
        if node.op_type == 'Sub' and '_sub' in node.output[0]:
            value_info.append(helper.make_tensor_value_info(node.output[0], TensorProto.FLOAT, [metadata['batch_size'], 16]))
        elif node.op_type == 'Exp' and '_exp' in node.output[0]:
            value_info.append(helper.make_tensor_value_info(node.output[0], TensorProto.FLOAT, [metadata['batch_size'], 16]))
        elif node.op_type == 'ReduceSum' and '_sum' in node.output[0]:
            value_info.append(helper.make_tensor_value_info(node.output[0], TensorProto.FLOAT, [metadata['batch_size']]))
        elif node.op_type == 'Div' and 'probs' in node.output[0]:
            value_info.append(helper.make_tensor_value_info(node.output[0], TensorProto.FLOAT, [metadata['batch_size'], 16]))
    
    # ONNXグラフを作成
    onnx_graph = helper.make_graph(
        onnx_nodes,
        'ForwardBackwardOptimizer',
        graph_inputs,
        graph_outputs,
        initializers,
        value_info=value_info  # value_infoを指定
    )

    # モデルを作成
    onnx_model = helper.make_model(onnx_graph)
    onnx_model.opset_import[0].version = 13

    # すべての形状情報を充填するために形状推論を実行
    onnx_model = shape_inference.infer_shapes(onnx_model)
    
    # 未使用ノードをクリーンアップ
    onnx_model = clean_unused_nodes_from_onnx(onnx_model)

    # モデルを保存
    onnx.save(onnx_model, output_path)
    print(f"ONNXモデルを {output_path} に保存しました")
