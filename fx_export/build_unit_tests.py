#!/usr/bin/env python3

import torch
import numpy as np
import os
import shutil
import importlib.util
import onnx
import onnx.helper as helper
import onnx.numpy_helper as numpy_helper
from typing import Dict, List, Any, Tuple, Optional
import glob
import fnmatch
import struct
import sys

# compile モジュールのインポート
from .compile import generate_code_from_onnx

# mntestモード用にvsm_converterからインポート
from .vsm_converter import inspect

# testcase_hint用にオペレーターシステムをインポート
from .operators import get_operator

# 上位互換レイアウトの定義
# キー: 要求されるレイアウト
# 値: 許容される上位互換レイアウトのリスト
LAYOUT_COMPATIBILITY = {
    "((16:1), (8_MAB:2, 8_L1B:1, 2_MAB:1, 4_PE:1, 2_W:1))": [
        "((2_L2B:3), (16:1), (8_MAB:2, 8_L1B:1, 2_MAB:1, 4_PE:1, 2_W:1))"  # DL用
    ],
    "((2:1, 4_PE:1, 2_W:1))": [
        "((8_L2B:1), (2:1, 4_PE:1, 2_W:1))"  # DL用
    ]
}

def is_layout_compatible(required_layout: str, provided_layout: str) -> bool:
    """
    レイアウトが互換性があるかチェック
    
    Args:
        required_layout: 要求されるレイアウト
        provided_layout: 提供されるレイアウト
    
    Returns:
        互換性がある場合True
    """
    # 完全一致
    if required_layout == provided_layout:
        return True
    
    # 上位互換チェック
    if required_layout in LAYOUT_COMPATIBILITY:
        compatible_layouts = LAYOUT_COMPATIBILITY[required_layout]
        if provided_layout in compatible_layouts:
            return True
    
    return False

# ============= 累積テスト用のVSM生成関数 =============

def numpy_to_hex_string(data: np.ndarray) -> Tuple[List[str], int]:
    """
    NumPy配列を16進数文字列のリストに変換する。
    
    Args:
        data: NumPy配列（float32、int32、またはint64）
    
    Returns:
        (16進数文字列のリスト, 要素あたりのバイト数)
    """
    if data.dtype == np.float32:
        # float32を16進数に変換
        flat_data = data.flatten()
        hex_strings = []
        for val in flat_data:
            # float32を4バイトのバイト列に変換し、16進数文字列に
            bytes_val = struct.pack('>f', val)  # big-endian
            hex_str = bytes_val.hex().upper()
            hex_strings.append(hex_str)
        return hex_strings, 4
    # elif data.dtype == np.int64:
    #     # int64を16進数に変換
    #     flat_data = data.flatten()
    #     hex_strings = []
    #     for val in flat_data:
    #         # int64を8バイトのバイト列に変換し、16進数文字列に
    #         bytes_val = struct.pack('>q', val)  # big-endian
    #         hex_str = bytes_val.hex().upper()
    #         hex_strings.append(hex_str)
    #     return hex_strings, 8
    elif data.dtype == np.int32 or data.dtype == np.int64:
        # int32を16進数に変換
        flat_data = data.flatten()
        hex_strings = []
        for val in flat_data:
            # int32を4バイトのバイト列に変換し、16進数文字列に
            bytes_val = struct.pack('>i', val)  # big-endian
            hex_str = bytes_val.hex().upper()
            hex_strings.append(hex_str)
        return hex_strings, 4
    
    # その他のデータ型はサポートしない
    raise ValueError(f"Unsupported dtype: {data.dtype}")

def generate_cumulative_test_vsm(
    graph: onnx.GraphProto,
    intermediate_values: Dict[str, np.ndarray],
    num_nodes: int,
    partial_graph: onnx.GraphProto
) -> str:
    """
    累積テスト用のoffseted.vsmファイルを生成する。
    
    Args:
        graph: 元の完全なONNXグラフ
        intermediate_values: 中間値の辞書
        num_nodes: 含まれるノード数
        partial_graph: 部分グラフ
    
    Returns:
        offseted.vsmの内容（文字列）
    """
    lines: List[str] = []
    
    # VSM_HERE_COMMENTをインポート
    # 現在のファイルから相対的にjudge/judge-pyディレクトリを見つける
    current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    judge_py_dir = os.path.join(current_dir, 'judge', 'judge-py')
    sys.path.append(judge_py_dir)
    from vsm_here import VSM_HERE_COMMENT
    
    # 1. 入力データの書き込み
    # 各入力に対応するDLノードを探す
    input_addresses = {}
    
    # グラフの入力名リストを作成
    graph_input_names = [inp.name for inp in graph.input]
    
    # 最初のnum_nodes個のノードから、入力に関連するDLノードを探す
    for node in graph.node[:num_nodes]:
        if node.op_type == 'DL':
            # DLノードの入力を確認（DLノードの入力がグラフの入力の場合）
            if node.input and node.input[0] in graph_input_names:
                input_name = node.input[0]
                input_idx = graph_input_names.index(input_name)
                
                # addr_in0を取得（DRAMから読み込むアドレス）
                addr_in = None
                for attr in node.attribute:
                    if attr.name == 'addr_in0' and attr.HasField('i'):
                        addr_in = attr.i
                        break
                
                if addr_in is not None:
                    input_addresses[input_idx] = addr_in
                    # print(f"  DLノード発見: Input {input_idx} ({input_name}) -> addr={addr_in}")
    
    # グラフの入力を処理
    for i, input_info in enumerate(graph.input):
        input_name = input_info.name
        if input_name not in intermediate_values:
            continue
        
        # アドレスを取得
        if i not in input_addresses:
            # print(f"⚠ WARNING: Input {i} ({input_name}) のアドレスが見つかりません。スキップします。")
            continue
        addr = input_addresses[i]
        
        data = intermediate_values[input_name]
        if not isinstance(data, np.ndarray):
            data = np.array(data)  # type: ignore[unreachable]
        
        # データを16進数に変換
        hex_strings, bytes_per_elem = numpy_to_hex_string(data)
        total_elements = len(hex_strings)
        
        # 要素数に応じてDRAMに分散
        if total_elements <= 16:
            # DRAM 0のみに書き込み
            write_to_single_dram(lines, 0, addr, hex_strings, data)
        else:
            # 4 DRAMに分散
            if total_elements % 4 != 0:
                raise ValueError(f"Element count {total_elements} is not divisible by 4")
            
            elements_per_dram = total_elements // 4
            for dram_id in range(4):
                start_idx = dram_id * elements_per_dram
                end_idx = start_idx + elements_per_dram
                dram_data = hex_strings[start_idx:end_idx]
                dram_values = data.flatten()[start_idx:end_idx]
                write_to_single_dram(lines, dram_id, addr, dram_data, dram_values)
    
    # 2. 初期化子（パラメータ）も同様に処理
    # 初期化子のアドレスを収集
    initializer_addresses = {}
    initializer_names = [init.name for init in graph.initializer]
    
    for node in graph.node[:num_nodes]:
        if node.op_type == 'DL':
            # DLノードの入力を確認（DLノードの入力が初期化子の場合）
            if node.input and node.input[0] in initializer_names:
                init_name = node.input[0]
                init_idx = initializer_names.index(init_name)
                
                # addr_in0を取得（DRAMから読み込むアドレス）
                addr_in = None
                for attr in node.attribute:
                    if attr.name == 'addr_in0' and attr.HasField('i'):
                        addr_in = attr.i
                        break
                
                if addr_in is not None:
                    initializer_addresses[init_idx] = addr_in
                    # print(f"  DLノード発見: Initializer {init_idx} ({init_name}) -> addr={addr_in}")
    
    for init_idx, init in enumerate(graph.initializer):
        if init.name not in intermediate_values:
            continue
        
        # アドレスを取得
        if init_idx not in initializer_addresses:
            # print(f"⚠ WARNING: Initializer {init_idx} ({init.name}) のアドレスが見つかりません。スキップします。")
            continue
        addr = initializer_addresses[init_idx]
        
        data = intermediate_values[init.name]
        if not isinstance(data, np.ndarray):
            data = np.array(data)  # type: ignore[unreachable]
        
        # データを16進数に変換して書き込み
        hex_strings, bytes_per_elem = numpy_to_hex_string(data)
        total_elements = len(hex_strings)
        
        if total_elements <= 16:
            write_to_single_dram(lines, 0, addr, hex_strings, data)
        else:
            if total_elements % 4 != 0:
                raise ValueError(f"Element count {total_elements} is not divisible by 4")
            
            elements_per_dram = total_elements // 4
            for dram_id in range(4):
                start_idx = dram_id * elements_per_dram
                end_idx = start_idx + elements_per_dram
                dram_data = hex_strings[start_idx:end_idx]
                dram_values = data.flatten()[start_idx:end_idx]
                write_to_single_dram(lines, dram_id, addr, dram_data, dram_values)
    
    # 3. VSM挿入マーカー
    lines.append(VSM_HERE_COMMENT)
    
    # 4. 期待値（出力）の記述
    # 部分グラフの各出力について、対応するULノードまたは最後のノードからアドレスを取得
    output_addresses = {}
    
    # 各出力に対応するノードを探す
    for idx, output in enumerate(partial_graph.output):
        if output.name not in intermediate_values:
            continue
        
        # この出力を生成するノードを探す
        found_addr = False
        
        # 最初のnum_nodes個のノードを逆順で確認
        for node in reversed(graph.node[:num_nodes]):
            # ULノードで、その入力が目的の出力の場合
            if node.op_type == 'UL' and node.input and node.input[0] == output.name:
                # ULノードのaddr_out0（DRAMアドレス）を取得
                for attr in node.attribute:
                    if attr.name == 'addr_out0' and attr.HasField('i'):
                        output_addresses[idx] = attr.i
                        found_addr = True
                        # print(f"  ULノード発見: Output {idx} ({output.name}) -> addr={attr.i}")
                        break
                if found_addr:
                    break
            
            # その他のノードで、その出力が目的の出力の場合
            elif output.name in node.output:
                # ノードのaddr_outXを取得
                out_idx = list(node.output).index(output.name)
                attr_name = f'addr_out{out_idx}'
                for attr in node.attribute:
                    if attr.name == attr_name and attr.HasField('i'):
                        output_addresses[idx] = attr.i
                        found_addr = True
                        # print(f"  ノード発見: Output {idx} ({output.name}) -> addr={attr.i}")
                        break
                if found_addr:
                    break
        
        # if not found_addr:
            # print(f"⚠ WARNING: Output {idx} ({output.name}) のアドレスが見つかりません。スキップします。")
    
    output_idx = 0
    for output in partial_graph.output:
        if output.name not in intermediate_values:
            continue
        
        # アドレスを取得
        if output_idx not in output_addresses:
            output_idx += 1
            continue
        addr = output_addresses[output_idx]
        
        data = intermediate_values[output.name]
        if not isinstance(data, np.ndarray):
            data = np.array(data)  # type: ignore[unreachable]
        
        # 期待値を記述
        write_expectations(lines, addr, data)
        output_idx += 1
    
    return '\n'.join(lines) + '\n'

def write_to_single_dram(lines: List[str], dram_id: int, base_addr: int, hex_strings: List[str], values: np.ndarray):
    """
    単一のDRAMにデータを書き込む行を生成する。
    
    Args:
        lines: 出力行のリスト
        dram_id: DRAM番号（0-3）
        base_addr: ベースアドレス（lw単位）
        hex_strings: 16進数文字列のリスト
        values: 元の値（デバッグ用）
    """
    # valuesをフラットにしておく
    if values.ndim > 1:
        values = values.flatten()
    elif values.ndim == 0:
        values = np.array([values])  # スカラーの場合は1次元配列に変換
    
    # データ型を判定
    if values.dtype == np.int32:
        type_str = "Int"
    elif values.dtype == np.int64:
        type_str = "Int"
    elif values.dtype == np.float32:
        type_str = "Float"
    else:
        type_str = "Unknown"
    
    # 2つのfloat32/int32（4バイト）または1つのint64（8バイト）を1つのlw（8バイト）にまとめる
    addr = base_addr
    i = 0
    while i < len(hex_strings):
        if len(hex_strings[i]) == 8:  # float32またはint32（4バイト）
            if i + 1 < len(hex_strings):
                # 2つの4バイト値を結合
                combined_hex = hex_strings[i] + hex_strings[i+1]
                if values.dtype == np.int32:
                    val_str = f"values=[{int(values[i])}, {int(values[i+1])}] / {type_str} @[0],[1]"
                else:
                    val_str = f"values=[{values[i]:.6g}, {values[i+1]:.6g}] / {type_str} @[0],[1]"
                i += 2
            else:
                # 最後の1つの場合（パディング）
                combined_hex = hex_strings[i] + "00000000"
                if values.dtype == np.int32:
                    val_str = f"values=[{int(values[i])}, 0] / {type_str} @[0],[1]"
                else:
                    val_str = f"values=[{values[i]:.6g}, 0.0] / {type_str} @[0],[1]"
                i += 1
        elif len(hex_strings[i]) == 16:  # int64（8バイト）
            combined_hex = hex_strings[i]
            val_str = f"values=[{int(values[i])}] / {type_str} @[0]"
            i += 1
        else:
            raise ValueError(f"Unexpected hex string length: {len(hex_strings[i])}")
        
        # r <dram_id> <9桁の16進アドレス> 001 <16桁の16進データ>
        line = f"r {dram_id} {addr:09X} 001 {combined_hex} # {val_str}"
        lines.append(line)
        addr += 1

def write_expectations(lines: List[str], base_addr: int, data: np.ndarray):
    """
    期待値の検証行を生成する。
    
    Args:
        lines: 出力行のリスト
        base_addr: ベースアドレス（lw単位）
        data: 期待値データ
    """
    flat_data = data.flatten()
    total_elements = len(flat_data)
    
    if total_elements <= 16:
        # DRAM 0のみから読み出し
        write_expectations_from_dram(lines, 0, base_addr, flat_data)
    else:
        # 4 DRAMから読み出し
        if total_elements % 4 != 0:
            raise ValueError(f"Element count {total_elements} is not divisible by 4")
        
        elements_per_dram = total_elements // 4
        for dram_id in range(4):
            start_idx = dram_id * elements_per_dram
            end_idx = start_idx + elements_per_dram
            dram_data = flat_data[start_idx:end_idx]
            write_expectations_from_dram(lines, dram_id, base_addr, dram_data)

def write_expectations_from_dram(lines: List[str], dram_id: int, base_addr: int, values: np.ndarray):
    """
    単一のDRAMから期待値を読み出す行を生成する。
    
    Args:
        lines: 出力行のリスト
        dram_id: DRAM番号（0-3）
        base_addr: ベースアドレス（10進数）
        values: 期待値
    """
    # bool型の場合はfloat32にキャスト（0.0 または 1.0）
    if values.dtype == np.bool_ or values.dtype == bool:
        values = values.astype(np.float32)

    if values.dtype != np.float32:
        raise ValueError(f"Unsupported dtype for expectations: {values.dtype}")
    
    addr = base_addr
    i = 0
    while i < len(values):
        if i + 1 < len(values):
            expect_str = f"expect=[{values[i]:.6g}, {values[i+1]:.6g}] / Float @[{dram_id},{i}],[{dram_id},{i+1}]"
            i += 2
        else:
            raise NotImplementedError(f"Expectations for odd number of elements not implemented: {len(values)}")
        # d getd $d<10進アドレス>n<dram_id> 1
        line = f"d getd $d{addr}n{dram_id} 1 # {expect_str} atol=1e-04"
        lines.append(line)
        addr += 1

# ============= Build Unit Tests関連の関数 =============

def generate_test_name(node, graph, intermediate_values):
    """ノードから適切なテスト名を生成する"""
    op_type = node.op_type
    
    # 基本名はオペレーションタイプ
    base_name = op_type
    
    # DL/ULノードの場合、特別な処理
    if op_type in ['DL', 'UL']:
        # 形状を取得（入力または出力の最初のもの）
        shape_str = None
        if node.input and node.input[0] in intermediate_values:
            shape = intermediate_values[node.input[0]].shape
            if len(shape) == 2:
                shape_str = f"{shape[0]}x{shape[1]}"
            elif len(shape) == 1:
                shape_str = str(shape[0])
        elif node.output and node.output[0] in intermediate_values:
            shape = intermediate_values[node.output[0]].shape
            if len(shape) == 2:
                shape_str = f"{shape[0]}x{shape[1]}"
            elif len(shape) == 1:
                shape_str = str(shape[0])
        
        # タグを取得（DLならtag_out0、ULならtag_in0）
        tag = "default"  # デフォルト値
        target_attr = "tag_out0" if op_type == "DL" else "tag_in0"
        for attr in node.attribute:
            if attr.name == target_attr:
                if attr.HasField('s'):
                    tag = attr.s.decode('utf-8')
                break
        
        # テスト名を構築: DL_形状_タグ または UL_形状_タグ
        if shape_str:
            base_name = f"{op_type}_{shape_str}_{tag}"
        else:
            base_name = f"{op_type}_{tag}"
    else:
        # DL/UL以外のノードは従来通り
        # 形状情報を追加（主要な入出力の形状）
        shapes = []
        
        # 入力の形状を取得
        for input_name in node.input[:2]:  # 最初の2つの入力のみ
            if input_name and input_name in intermediate_values:
                shape = intermediate_values[input_name].shape
                if len(shape) == 2:
                    shapes.append(f"{shape[0]}x{shape[1]}")
                elif len(shape) == 1:
                    shapes.append(str(shape[0]))
        
        # 出力の形状を取得
        for output_name in node.output[:1]:  # 最初の出力のみ
            if output_name and output_name in intermediate_values:
                shape = intermediate_values[output_name].shape
                if len(shape) == 2:
                    shapes.append(f"{shape[0]}x{shape[1]}")
                elif len(shape) == 1:
                    shapes.append(str(shape[0]))
        
        # パラメータを追加（例：transB for Gemm）
        params = []
        for attr in node.attribute:
            if attr.name in ['transA', 'transB'] and attr.i == 1:
                params.append(attr.name)
        
        # テスト名を構築
        if shapes:
            base_name = f"{base_name}_{'_'.join(shapes)}"
        if params:
            base_name = f"{base_name}_{'_'.join(params)}"
    
    return base_name

def build_unit_tests(export_dir, output_dir, extra=False, onnx_filename="model.onnx", mntest=False):
    """エクスポート済みディレクトリからONNXグラフを分解して個別のテストケースを生成
    
    Args:
        export_dir: エクスポート済みのディレクトリ（model.onnx と *.npy ファイルを含む）
        output_dir: ユニットテストの出力先ディレクトリ（デフォルト: ./unit_tests）
        extra: 累積テスト（z_nodes_*）も生成するかどうか
        onnx_filename: 使用するONNXファイル名（デフォルト: model.onnx）
        mntest: MN-Coreアセンブリ用のテスト作成モード
    """
    
    # output_dirにファイルが存在しないことをチェック
    if os.path.exists(output_dir) and os.listdir(output_dir):
        raise ValueError(f"output_dir '{output_dir}' にファイルが存在します。クリーンアップするか、別のディレクトリを指定してください")
    
    print(f"=== {export_dir} のunit test生成を開始 {'(mntest mode)' if mntest else ''} ===")
    
    # エクスポート済みのONNXとデータを使用
    onnx_path = os.path.join(export_dir, onnx_filename)
    if not os.path.exists(onnx_path):
        raise FileNotFoundError(f"ONNXファイルが見つかりません: {onnx_path}")
    
    # ONNXモデルを読み込み
    model = onnx.load(onnx_path)
    graph = model.graph
    
    # mntestモードの場合、testcasesディレクトリのVSMファイルを事前にinspect
    vsm_testcases = {}
    if mntest:
        testcases_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "testcases")
        if os.path.exists(testcases_dir):
            print(f"VSMテストケースを検索中: {testcases_dir}")
            vsm_files = glob.glob(os.path.join(testcases_dir, "*.vsm"))
            for vsm_file in vsm_files:
                filename = os.path.basename(vsm_file)
                # inspect関数にフルパスを渡す
                info = inspect(vsm_file)
                vsm_testcases[filename] = {
                    'path': vsm_file,
                    'info': info
                }
                # デバッグ出力
                if 'DEBUG_MNTEST' in os.environ:
                    print(f"  {filename}: inputs={info['inputs']}, output={info['output']}")
            print(f"  {len(vsm_testcases)} 個のVSMテストケースを解析しました")
    
    # 入力データを読み込む
    all_inputs_list = []
    
    # input_*.npy ファイルを読み込む
    i = 0
    while True:
        input_file = os.path.join(export_dir, f"input_{i}.npy")
        if not os.path.exists(input_file):
            break
        data = np.load(input_file)
        # ONNXから入力名を取得
        if i < len(model.graph.input):
            name = model.graph.input[i].name
            all_inputs_list.append((name, torch.from_numpy(data)))
        i += 1
    
    # 中間結果を計算するために完全なグラフを実行
    print("中間結果を取得するために完全なグラフを実行...")
    intermediate_values = {}
    
    # PyTorchで関数を実行して中間値を取得
    # ONNXから生成したPythonコードを使用
    generate_code_from_onnx(onnx_path, export_dir)
    python_path = os.path.join(export_dir, "forward_backward.py")
    
    # Pythonコードを読み込んで実行
    spec = importlib.util.spec_from_file_location("generated_func", python_path)
    generated_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(generated_module)
    
    # 関数を取得
    generated_func = generated_module.forward_backward
    
    # 入力を準備（all_inputs_listから取得）
    func_inputs = {}
    
    # all_inputs_listから全ての入力を取得
    for name, tensor in all_inputs_list:
        if isinstance(tensor, torch.Tensor):
            # 入力の形状を調整（例：inputは(64, 1, 28, 28)から(64, 784)へ）
            if name == 'input' and len(tensor.shape) == 4:
                # 形状変換: (batch, 1, 28, 28) -> (batch, 784)
                func_inputs[name] = tensor.detach().flatten(1)
            else:
                func_inputs[name] = tensor.detach()
        elif isinstance(tensor, np.ndarray):
            func_inputs[name] = torch.from_numpy(tensor)
        else:
            func_inputs[name] = tensor
    
    # デバッグ：形状を確認
    print(f"生成された入力キー: {list(func_inputs.keys())}")
    for key, value in func_inputs.items():
        if hasattr(value, 'shape'):
            print(f"  {key}: shape={value.shape}")
    
    # 実行して全ての出力を取得
    with torch.no_grad():
        outputs = generated_func(**func_inputs)
    
    # 入力値を保存
    for key, value in func_inputs.items():
        if isinstance(value, torch.Tensor):
            intermediate_values[key] = value.detach().cpu().numpy()
    
    # 出力値を保存
    for key, value in outputs.items():
        if isinstance(value, torch.Tensor):
            intermediate_values[key] = value.detach().cpu().numpy()
    
    # 各ノードの中間出力も保存が必要
    # ONNXグラフからvalue_infoを読み取って、すべての中間テンソル名を取得
    for value_info in graph.value_info:
        tensor_name = value_info.name
        if tensor_name in outputs:
            intermediate_values[tensor_name] = outputs[tensor_name].detach().cpu().numpy() if isinstance(outputs[tensor_name], torch.Tensor) else outputs[tensor_name]
    
    # 初期化子の値も保存
    for init in graph.initializer:
        tensor = numpy_helper.to_array(init)
        intermediate_values[init.name] = tensor
    
    # 各ノードに対してテストケースを作成
    # export_dirの最後のディレクトリ名を使用（例: /tmp/train_step -> train_step）
    example_name = os.path.basename(export_dir)
    test_base_dir = os.path.join(output_dir, example_name)
    os.makedirs(test_base_dir, exist_ok=True)
    
    # 元となった完全なONNXモデルも保存
    original_model_path = os.path.join(test_base_dir, "original_model.onnx")
    shutil.copy(onnx_path, original_model_path)
    print(f"元のONNXモデルを保存: {original_model_path}")
    
    print(f"テストケースを {test_base_dir} に生成...")

    # 使用済みのテスト名を追跡
    used_test_names = {}

    def build_unique_test_name(base_test_name: str) -> str:
        if base_test_name not in used_test_names:
            used_test_names[base_test_name] = 0
            return f"{base_test_name}_a"
        used_test_names[base_test_name] += 1
        suffix_num = used_test_names[base_test_name]
        suffix = ""
        while suffix_num >= 0:
            suffix = chr(ord('a') + (suffix_num % 26)) + suffix
            suffix_num = suffix_num // 26 - 1
        return f"{base_test_name}_{suffix}"
    
    # mntestモードの処理
    if mntest:
        print("\n=== MN-Core用単体テスト生成 ===")
        print(f"VSMテストケース数: {len(vsm_testcases)}")
        test_count = 0  # 生成されたテストケース数をカウント
        skipped_nodes = []  # スキップしたノードのリスト
        for node in graph.node:
            # Identityノードはスキップ
            if node.op_type == 'Identity':
                continue
            # ノード名とパラメータから基本のテスト名を生成
            base_test_name = generate_test_name(node, graph, intermediate_values)
            
            # 重複を避けるためにサフィックスを追加
            test_name = build_unique_test_name(base_test_name)
            
            # オペレーターからtestcase_hintを取得
            operator_class = get_operator(node.op_type)
            operator = operator_class(node, graph, {})
            hint = operator.testcase_hint()
            
            if not hint:
                skipped_nodes.append((test_name, "no hint"))
                continue
            
            # マッチするVSMテストケースを探す
            matched_vsm = find_matching_vsm(node, graph, hint, vsm_testcases)
            
            if matched_vsm:
                test_dir = os.path.join(test_base_dir, test_name)
                os.makedirs(test_dir, exist_ok=True)
                
                # VSMファイルをシンボリックリンクとして作成
                vsm_src = matched_vsm['path']
                vsm_dst = os.path.join(test_dir, os.path.basename(vsm_src))
                # 既存のリンクがあれば削除
                if os.path.lexists(vsm_dst):
                    os.unlink(vsm_dst)
                # シンボリックリンクを作成
                os.symlink(vsm_src, vsm_dst)
                
                # ノードのアドレス情報を取得してoffsetedバージョンも生成
                node_addresses = get_node_addresses(node)
                if node_addresses:
                    # VSMのinspect情報を使用（すでにmatched_vsmに保存されている）
                    vsm_info = matched_vsm['info']
                    
                    # target_configを構築
                    target_config = {
                        'inputs': [],
                        'output': None
                    }
                    
                    # 入力アドレス設定
                    for i, vsm_input in enumerate(vsm_info.get('inputs', [])):
                        addr_key = f'addr_in{i}'
                        location_key = f'location_in{i}'
                        if addr_key in node_addresses and location_key in node_addresses:
                            mem_type = node_addresses[location_key]
                            offset = node_addresses[addr_key]
                            target_config['inputs'].append([mem_type, offset])
                        else:
                            # デフォルト値を使用
                            target_config['inputs'].append([vsm_input[0], vsm_input[1]])
                    
                    # 出力アドレス設定
                    if vsm_info.get('output'):
                        if 'addr_out0' in node_addresses and 'location_out0' in node_addresses:
                            mem_type = node_addresses['location_out0']
                            offset = node_addresses['addr_out0']
                            target_config['output'] = [mem_type, offset]
                        else:
                            # デフォルト値を使用
                            target_config['output'] = [vsm_info['output'][0], vsm_info['output'][1]]
                    
                    # vsm_converterを使って変換
                    from .vsm_converter import converter
                    converted_vsm = converter(vsm_src, target_config)
                    
                    # offseted.vsmとして保存
                    offseted_path = os.path.join(test_dir, "offseted.vsm")
                    with open(offseted_path, 'w') as f:
                        f.write(converted_vsm)
                    in_addrs = []
                    for inp in target_config.get('inputs', []):
                        if len(inp) >= 2:
                            in_addrs.append(f"{inp[0]}:{inp[1]}")
                    in_str = ','.join(in_addrs) if in_addrs else 'none'
                    
                    # 出力アドレスの文字列を構築
                    out = target_config.get('output')
                    out_str = f"{out[0]}:{out[1]}" if out and len(out) >= 2 else 'none'
                    
                    print(f"  {test_name}: {os.path.basename(vsm_src)} を使用（in={in_str}, out={out_str}）")
                    
                
                # ONNXも保存
                single_node_graph = create_single_node_graph(node, graph, intermediate_values)
                single_model = helper.make_model(single_node_graph)
                onnx.save(single_model, os.path.join(test_dir, "model.onnx"))
                
                # node_addressesがない場合は、target_configなしで出力
                if not node_addresses:
                    print(f"  {test_name}: {os.path.basename(vsm_src)} を使用")
                
                test_count += 1
            else:
                skipped_nodes.append((test_name, "no matching VSM"))
        
        # mntestモードの結果表示
        total_nodes = len([n for n in graph.node if n.op_type != 'Identity'])
        print(f"\n✓ {total_nodes} ノード中、{test_count} 個のテストケースを生成しました: {test_base_dir}")
        
        # スキップしたノードのまとめ表示
        if skipped_nodes:
            print(f"\n⚠ スキップしたノード ({len(skipped_nodes)} 個):")
            for test_name, reason in skipped_nodes:
                print(f"  - {test_name}: {reason}")
    else:
        # 通常モードの処理（既存のコード）
        for node in graph.node:
            # ノード名とパラメータから基本のテスト名を生成
            base_test_name = generate_test_name(node, graph, intermediate_values)

            test_name = build_unique_test_name(base_test_name)
            
            test_dir = os.path.join(test_base_dir, test_name)
            
            # 既存のディレクトリを削除して再作成
            if os.path.exists(test_dir):
                shutil.rmtree(test_dir)
            os.makedirs(test_dir, exist_ok=True)
            
            print(f"  生成中: {test_name}")
            
            # 単一ノードのONNXグラフを作成
            single_node_graph = create_single_node_graph(node, graph, intermediate_values)
            
            # ONNXモデルを保存
            single_model = helper.make_model(single_node_graph)
            onnx.save(single_model, os.path.join(test_dir, "model.onnx"))
            
            # 入力データを保存
            for i, input_name in enumerate(node.input):
                if input_name in intermediate_values:
                    input_path = os.path.join(test_dir, f"input_{i}.npy")
                    np.save(input_path, intermediate_values[input_name])
            
            # 出力データを保存（ダミー実行から取得）
            for i, output_name in enumerate(node.output):
                if output_name in intermediate_values:
                    output_path = os.path.join(test_dir, f"output_{i}.npy")
                    output_data = intermediate_values[output_name]
                    # booleanテンソルはfloatに変換（C++側はfloatを返すため）
                    if hasattr(output_data, 'dtype') and output_data.dtype == np.bool_:
                        output_data = output_data.astype(np.float32)
                    elif isinstance(output_data, torch.Tensor) and output_data.dtype == torch.bool:
                        output_data = output_data.float().numpy()
                    np.save(output_path, output_data)
    
        # 通常モードの結果表示
        print(f"✓ {len(graph.node)} ノード中、{len(graph.node)} 個のテストケースを生成しました: {test_base_dir}")
    
    # --extraオプションが指定された場合のみ累積テストを生成
    if extra:
        print(f"\nノード番号順の累積テストを生成...")
        
        # z_nodes_000に共通入力データを保存
        shared_input_files = {}
        first_cumulative_test_created = False
        first_cumulative_test_dir = None
        
        test_count = 0
        for num_nodes in range(1, len(graph.node) + 1):
            # mntestモードでは、出力がDRAMの場合のみテストを作成
            if mntest:
                # 最後のノードの出力をチェック
                last_node = graph.node[num_nodes - 1]
                is_dram_output = check_if_dram_output(last_node, graph)
                if not is_dram_output:
                    continue  # LM出力の場合はスキップ
            
            # z_プレフィックスを付けて、0からのノード番号で命名（3桁ゼロ埋め）
            test_name = f"z_nodes_{num_nodes-1:03d}"
            test_dir = os.path.join(output_dir, example_name, test_name)
            
            if os.path.exists(test_dir):
                shutil.rmtree(test_dir)
            os.makedirs(test_dir, exist_ok=True)
            
            print(f"  生成中: {test_name}")
            
            # 部分グラフを作成（最初のnum_nodes個のノード）
            partial_graph = create_partial_graph_with_connected_outputs(graph, intermediate_values, num_nodes)
            
            # ONNXモデルを保存
            partial_model = helper.make_model(partial_graph)
            onnx.save(partial_model, os.path.join(test_dir, "model.onnx"))
            
            # mntestモードの場合、offseted.vsmを生成
            if mntest:
                # offseted.vsmを生成
                vsm_content = generate_cumulative_test_vsm(graph, intermediate_values, num_nodes, partial_graph)
                offseted_path = os.path.join(test_dir, "offseted.vsm")
                with open(offseted_path, 'w') as f:
                    f.write(vsm_content)
                # print(f"    offseted.vsm を生成しました")
                test_count += 1
            else:
                # 通常モード：npyファイルを保存
                # 最初の累積テストの場合は入力データを保存
                if not first_cumulative_test_created:
                    # グラフ入力を保存
                    for i, input_info in enumerate(graph.input):
                        input_name = input_info.name
                        if input_name in intermediate_values:
                            input_path = os.path.join(test_dir, f"input_{i}.npy")
                            np.save(input_path, intermediate_values[input_name])
                            shared_input_files[f"input_{i}.npy"] = input_path
                    
                    # 初期化子も保存
                    saved_count = len(graph.input)
                    for init in graph.initializer:
                        if init.name in intermediate_values:
                            input_path = os.path.join(test_dir, f"input_{saved_count}.npy")
                            np.save(input_path, intermediate_values[init.name])
                            shared_input_files[f"input_{saved_count}.npy"] = input_path
                            saved_count += 1
                    
                    first_cumulative_test_created = True
                    first_cumulative_test_dir = test_dir
                else:
                    # 2番目以降の累積テストはハードリンクで参照
                    for filename, shared_path in shared_input_files.items():
                        link_path = os.path.join(test_dir, filename)
                        os.link(shared_path, link_path)
                
                # 出力データを保存（接続されている出力のみ）
                output_idx = 0
                for output in partial_graph.output:
                    if output.name in intermediate_values:
                        output_path = os.path.join(test_dir, f"output_{output_idx}.npy")
                        output_data = intermediate_values[output.name]
                        # booleanテンソルはfloatに変換
                        if hasattr(output_data, 'dtype') and output_data.dtype == np.bool_:
                            output_data = output_data.astype(np.float32)
                        elif isinstance(output_data, torch.Tensor) and output_data.dtype == torch.bool:
                            output_data = output_data.float().numpy()
                        np.save(output_path, output_data)
                        output_idx += 1
                test_count += 1
        print(f"✓ {test_count} 個の累積テストケースを生成しました")

def find_matching_vsm(node, graph, hint, vsm_testcases):
    """ヒントとレイアウトに基づいてマッチするVSMテストケースを探す"""
    # ワイルドカードパターンにマッチするVSMファイルを選択
    matching_files = []
    
    # デバッグ: ヒントとテストケース数を表示
    if 'DEBUG_MNTEST' in os.environ:
        print(f"  検索中: hint={hint}, vsm_testcases数={len(vsm_testcases)}")
    
    for filename, test_info in vsm_testcases.items():
        if fnmatch.fnmatch(filename, hint):
            matching_files.append((filename, test_info))
    
    if not matching_files:
        return None
    
    # ノードの入出力レイアウトを取得
    node_layouts = get_node_layouts(node)
    
    # デバッグ: ノードのレイアウトを表示
    if 'DEBUG_MNTEST' in os.environ:
        print(f"    ノード {node.op_type} のレイアウト:")
        for key, value in node_layouts.items():
            print(f"      {key}: {value}")
    
    # レイアウトがマッチするVSMファイルを探す
    for filename, test_info in matching_files:
        vsm_info = test_info['info']
        
        # 入力レイアウトのチェック
        layout_match = True
        
        # 各入力のレイアウトをチェック
        for i, vsm_input in enumerate(vsm_info.get('inputs', [])):
            node_layout_key = f'layout_in{i}'
            if node_layout_key in node_layouts:
                node_layout = node_layouts[node_layout_key]
                vsm_layout = vsm_input[2]  # [location, offset, layout]のlayout部分
                if node_layout != vsm_layout:
                    if 'DEBUG_MNTEST' in os.environ:
                        print(f"    {filename}: 入力{i}のレイアウト不一致")
                        print(f"      ノード: {node_layout}")
                        print(f"      VSM: {vsm_layout}")
                    layout_match = False
                    break
        
        if not layout_match:
            continue
        
        # 出力レイアウトのチェック（互換性チェックあり）
        if vsm_info.get('output'):
            node_layout_key = 'layout_out0'
            if node_layout_key in node_layouts:
                node_layout = node_layouts[node_layout_key]
                vsm_layout = vsm_info['output'][2]  # [location, offset, layout]のlayout部分
                
                # 出力は互換性チェックを行う
                if not is_layout_compatible(node_layout, vsm_layout):
                    if 'DEBUG_MNTEST' in os.environ:
                        print(f"    {filename}: 出力のレイアウト不一致（互換性なし）")
                        print(f"      ノード要求: {node_layout}")
                        print(f"      VSM提供: {vsm_layout}")
                    continue
        
        # レイアウトがマッチした場合
        if 'DEBUG_MNTEST' in os.environ:
            print(f"    {filename}: レイアウトが一致！")
        return test_info
    
    # マッチするものが見つからない場合
    if 'DEBUG_MNTEST' in os.environ:
        print(f"    レイアウトが一致するVSMファイルが見つかりません")
    return None

def get_node_layouts(node):
    """ONNXノードからレイアウト属性を取得"""
    layouts = {}
    
    for attr in node.attribute:
        # layout_in0, layout_in1, layout_out0 などを探す
        if attr.name.startswith('layout_'):
            if attr.HasField('s'):
                layouts[attr.name] = attr.s.decode('utf-8')
    
    return layouts

def get_node_addresses(node):
    """ONNXノードからアドレス属性を取得"""
    addresses = {}
    
    for attr in node.attribute:
        # addr_in0, addr_out0 などのアドレス情報を探す
        if attr.name.startswith('addr_'):
            if attr.HasField('i'):
                addresses[attr.name] = attr.i
        # location_in0, location_out0 などの位置情報を探す
        elif attr.name.startswith('location_'):
            if attr.HasField('s'):
                addresses[attr.name] = attr.s.decode('utf-8')
    
    return addresses

def check_if_dram_output(node, graph):
    """ノードの出力がDRAMかどうかをチェック"""
    # ノードのattributeからtag_out*を探す
    for attr in node.attribute:
        if attr.name.startswith('tag_out'):
            if attr.HasField('s'):
                tag = attr.s.decode('utf-8')
                if tag == 'DRAM':
                    return True
    
    # ULノードは出力がDRAM
    if node.op_type == 'UL':
        return True
    
    return False

def create_partial_graph_with_connected_outputs(graph, intermediate_values, num_nodes):
    """グラフの最初のnum_nodes個のノードだけを含む部分グラフを作成
    出力は後続のノードに接続されているもののみを含める
    """
    
    # 最初のnum_nodes個のノードを取得
    nodes = graph.node[:num_nodes]
    
    # これらのノードの出力を収集
    outputs_set = set()
    for node in nodes:
        outputs_set.update(node.output)
    
    # 後続のノードの入力として使われている出力のみを残す
    connected_outputs = set()
    
    # 残りのノードの入力をチェック
    if num_nodes < len(graph.node):
        for node in graph.node[num_nodes:]:
            for input_name in node.input:
                if input_name in outputs_set:
                    connected_outputs.add(input_name)
    
    # グラフの最終出力も含める（すでに計算されている場合）
    for output in graph.output:
        if output.name in outputs_set:
            connected_outputs.add(output.name)
    
    # 出力ValueInfoを作成
    outputs = []
    
    # まずgraph.value_infoから探す
    for value_info in graph.value_info:
        if value_info.name in connected_outputs:
            outputs.append(value_info)
    
    # graph.outputからも探す
    for output in graph.output:
        if output.name in connected_outputs:
            outputs.append(output)
    
    # もし出力の型情報が見つからない場合は、intermediate_valuesから作成
    found_outputs = {o.name for o in outputs}
    for output_name in connected_outputs:
        if output_name not in found_outputs and output_name in intermediate_values:
            value = intermediate_values[output_name]
            if isinstance(value, torch.Tensor):
                value = value.detach().cpu().numpy()
            elif not isinstance(value, np.ndarray):
                value = np.array(value)
            
            # 型情報を作成
            elem_type = onnx.helper.np_dtype_to_tensor_dtype(value.dtype)
            shape = list(value.shape)
            output_info = helper.make_tensor_value_info(
                output_name,
                elem_type,
                shape
            )
            outputs.append(output_info)
    
    # 中間テンソルのvalue_infoを作成
    # 部分グラフ内で生成されるすべての中間テンソルの形状情報を追加
    value_infos = []
    for tensor_name in outputs_set:
        # 入力、出力、初期化子でないテンソルを対象とする
        is_input = any(inp.name == tensor_name for inp in graph.input)
        is_output = tensor_name in connected_outputs
        is_initializer = any(init.name == tensor_name for init in graph.initializer)
        
        if not is_input and not is_output and not is_initializer:
            # 元のグラフのvalue_infoから探す
            found = False
            for vi in graph.value_info:
                if vi.name == tensor_name:
                    value_infos.append(vi)
                    found = True
                    break
            
            # 見つからない場合はintermediate_valuesから作成
            if not found and tensor_name in intermediate_values:
                value = intermediate_values[tensor_name]
                if isinstance(value, torch.Tensor):
                    value = value.detach().cpu().numpy()
                elif not isinstance(value, np.ndarray):
                    value = np.array(value)
                
                elem_type = onnx.helper.np_dtype_to_tensor_dtype(value.dtype)
                shape = list(value.shape)
                value_info = helper.make_tensor_value_info(
                    tensor_name,
                    elem_type,
                    shape
                )
                value_infos.append(value_info)
    
    # 部分グラフを作成（value_infoを追加）
    partial_graph = helper.make_graph(
        nodes,
        graph.name + f"_partial_{num_nodes}",
        graph.input,
        outputs,
        graph.initializer,
        value_info=value_infos  # 中間テンソルの形状情報を追加
    )
    
    return partial_graph

def create_partial_graph(graph, num_nodes):
    """グラフの最初のnum_nodes個のノードだけを含む部分グラフを作成"""
    
    # 最初のnum_nodes個のノードを取得
    nodes = graph.node[:num_nodes]
    
    # これらのノードの出力を収集
    outputs_set = set()
    for node in nodes:
        outputs_set.update(node.output)
    
    # 出力ValueInfoを作成
    outputs = []
    for value_info in graph.value_info:
        if value_info.name in outputs_set:
            outputs.append(value_info)
    
    # グラフの元の出力も含める（既に計算されている場合）
    for output in graph.output:
        if output.name in outputs_set:
            outputs.append(output)
    
    # 部分グラフを作成
    partial_graph = helper.make_graph(
        nodes,
        graph.name + "_partial",
        graph.input,
        outputs,
        graph.initializer
    )
    
    return partial_graph

def create_single_node_graph(node, original_graph, intermediate_values):
    """単一ノードのONNXグラフを作成"""
    
    # 入力を作成
    inputs = []
    for input_name in node.input:
        if input_name in intermediate_values:
            shape = intermediate_values[input_name].shape
            dtype = onnx.mapping.NP_TYPE_TO_TENSOR_TYPE[intermediate_values[input_name].dtype]
            inputs.append(helper.make_tensor_value_info(input_name, dtype, shape))
    
    # 出力を作成 - 単一ノードテストでは、ノードの出力をグラフ出力とする
    outputs = []
    for output_name in node.output:
        if output_name in intermediate_values:
            shape = intermediate_values[output_name].shape
            dtype = onnx.mapping.NP_TYPE_TO_TENSOR_TYPE[intermediate_values[output_name].dtype]
            outputs.append(helper.make_tensor_value_info(output_name, dtype, shape))
    
    # 初期化子（定数）を収集
    initializers = []
    for init in original_graph.initializer:
        if init.name in node.input:
            initializers.append(init)
    
    # 中間テンソルのvalue_infoを作成
    # 単一ノードの場合、出力がvalue_infoに含まれる必要がある場合がある
    value_infos = []
    for output_name in node.output:
        if output_name in intermediate_values:
            shape = intermediate_values[output_name].shape
            dtype = onnx.mapping.NP_TYPE_TO_TENSOR_TYPE[intermediate_values[output_name].dtype]
            value_info = helper.make_tensor_value_info(output_name, dtype, shape)
            value_infos.append(value_info)
    
    # グラフを作成（value_infoを追加）
    graph = helper.make_graph(
        [node],
        'single_node_test',
        inputs,
        outputs,
        initializers,
        value_info=value_infos  # 出力の形状情報もvalue_infoに追加
    )
    
    return graph
