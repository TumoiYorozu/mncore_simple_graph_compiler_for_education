#!/usr/bin/env python3

import ctypes
import os
import random
import time
from typing import Any, Dict, List, Tuple, Optional, Union

import numpy as np
import onnx
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from .compile import (
    compile_cpp_library, generate_cpp_from_onnx, generate_vsm_from_onnx,
    parse_cpp_output_info
)
from .export import export_function
from .mncore_transform import transform_to_mncore
from .mncore_utils import (
    assemble_vsm, clear_dram_mncore2, extract_addresses_from_onnx,
    generate_vsm_input_commands, generate_vsm_output_commands,
    parse_emulator_output, prepare_mncore2_workspace,
    read_dram_mncore2, run_emulator, run_mncore2_computation,
    write_dram_mncore2
)
from .train import (
    DEFAULT_BATCH_SIZE, evaluate, get_example_function_and_inputs,
    get_mnist_transform, init_globals
)


def load_cpp_train_step(cpp_path: str, parse_cpp_output_info, compile_cpp_library) -> Tuple[Any, List[Dict[str, Any]], str]:
    """C++のtrain_step関数をロードする"""
    
    # C++コードを読み込む
    with open(cpp_path, 'r') as f:
        cpp_code = f.read()
    
    # 出力情報をパース
    output_info = parse_cpp_output_info(cpp_code)
    
    # C++コードをコンパイル
    lib_path = compile_cpp_library(cpp_code, os.path.join(os.path.dirname(cpp_path), "train_step"))
    
    # ライブラリを読み込み
    lib = ctypes.CDLL(lib_path)
    func = lib.forward_backward
    
    return func, output_info, cpp_code

def format_time(seconds: float) -> str:
    """秒数を読みやすい形式にフォーマット"""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}m"
    else:
        return f"{seconds/3600:.1f}h"

def get_emu_output_info(onnx_path: str, ignore_loss: bool = False) -> List[Dict[str, Any]]:
    """エミュレータ用の出力情報を取得"""
    model = onnx.load(onnx_path)
    graph = model.graph
    
    output_info = []
    for output in graph.output:
        name = output.name
        
        # ignore_lossがTrueの場合、lossをスキップ
        if ignore_loss and name == "loss":
            continue
        
        # 形状を取得
        shape = []
        if output.type.tensor_type.shape.dim:
            for dim in output.type.tensor_type.shape.dim:
                if dim.HasField('dim_value'):
                    shape.append(dim.dim_value)
        
        output_info.append({
            'name': name,
            'shape': shape,
            'is_scalar': len(shape) == 0
        })
    
    return output_info

def train_emu(model: nn.Module, train_loader: DataLoader, asm_base: str, 
              addresses: Dict[str, Dict[str, int]], output_info: List[Dict[str, Any]], 
              expected_batch_size: int, ignore_loss: bool = False, test_loader: Optional[DataLoader] = None) -> float:
    """エミュレータでコンパイルしたtrain_stepを使って訓練"""
    
    assert model is not None, "model must be initialized"
    model.train()
    total_loss = 0.0
    total_iterations = len(train_loader)
    epoch_start_time = time.time()
    iter_times = []
    
    # Epoch表示の後に改行して、進捗表示が上書きしないようにする
    print("")
    
    for iter_idx, (data, target) in enumerate(train_loader):
        iter_start_time = time.time()
        batch_size = data.shape[0]
        
        # バッチサイズが期待値と異なる場合はエラー
        if batch_size != expected_batch_size:
            raise ValueError(f"エラー: バッチサイズ{batch_size}は期待値{expected_batch_size}と異なります。DataLoaderにdrop_last=Trueを設定してください")
        
        # 入力データのVSMコマンドを生成
        vsm_commands = []
        
        # 入力データをセット
        if 'input' in addresses['inputs']:
            addr = addresses['inputs']['input']
            commands = generate_vsm_input_commands(data.view(batch_size, -1).numpy(), addr, show_comments=False)
            vsm_commands.extend(commands)
        
        if 'target' in addresses['inputs']:
            addr = addresses['inputs']['target']
            commands = generate_vsm_input_commands(target.numpy(), addr, show_comments=False)
            vsm_commands.extend(commands)
        
        # モデルパラメータをセット
        for name, param in model.named_parameters():
            param_name = name.replace('.', '_')
            if param_name in addresses['inputs']:
                addr = addresses['inputs'][param_name]
                commands = generate_vsm_input_commands(param.data.cpu().numpy(), addr, show_comments=False)
                vsm_commands.extend(commands)
        
        # ASMコードを追加
        vsm_commands.append(asm_base)
        
        # 出力データの取得コマンドを追加
        for info in output_info:
            output_name = info['name']
            if output_name in addresses['outputs']:
                addr = addresses['outputs'][output_name]
                commands = generate_vsm_output_commands(tuple(info['shape']), addr, show_comments=False)
                vsm_commands.extend(commands)
        
        # エミュレータを実行
        full_commands = '\n'.join(vsm_commands)
        emulator_output = run_emulator(full_commands)
        
        # 出力をパース
        values = parse_emulator_output(emulator_output)
        
        # 出力を辞書に整理
        output_arrays: Dict[str, Union[float, np.ndarray]] = {}
        value_idx = 0
        for info in output_info:
            name = info['name']
            if info['is_scalar']:
                output_arrays[name] = values[value_idx] if value_idx < len(values) else 0.0
                value_idx += 1
            else:
                num_elements = int(np.prod(info['shape']))
                output_arrays[name] = np.array(values[value_idx:value_idx+num_elements]).reshape(info['shape'])
                value_idx += num_elements
        
        # 更新されたパラメータをモデルに反映
        with torch.no_grad():
            for name, param in model.named_parameters():
                updated_name = f"updated_{name.replace('.', '_')}"
                if updated_name in output_arrays:
                    updated_val = output_arrays[updated_name]
                    if isinstance(updated_val, np.ndarray):
                        param.data.copy_(torch.from_numpy(updated_val))
        
        # lossを取得（ignore_lossがFalseの場合のみ）
        if not ignore_loss and 'loss' in output_arrays:
            loss_value = output_arrays['loss']
            if isinstance(loss_value, np.ndarray):
                if loss_value.ndim == 0:
                    total_loss += float(loss_value)
                else:
                    total_loss += float(loss_value[0])
            else:
                total_loss += float(loss_value)
        
        # イテレーション処理時間を記録
        iter_end_time = time.time()
        iter_time = iter_end_time - iter_start_time
        iter_times.append(iter_time)
        
        # 平均時間と残り時間を計算
        avg_iter_time = sum(iter_times) / len(iter_times)
        remaining_iters = total_iterations - (iter_idx + 1)
        eta_seconds = remaining_iters * avg_iter_time
        
        # 進捗表示（カーソルを行頭に戻して上書き）
        progress_str = (f"\r  {iter_idx + 1:3d}/{total_iterations} "
                       f"[{iter_time:.1f}s/iter, avg:{avg_iter_time:.1f}s, ETA:{format_time(eta_seconds)}]")
        print(progress_str, end="", flush=True)
        
        # 10イテレーションごとに評価を実行
        if test_loader is not None and (iter_idx + 1) % 10 == 0:
            # 進捗表示をクリアして評価結果を表示
            print(f"\r  {iter_idx + 1:3d}/{total_iterations} - Evaluating...", end="", flush=True)
            acc, _ = evaluate(model, test_loader)
            print(f"\r  {iter_idx + 1:3d}/{total_iterations} - Accuracy: {acc:.2f}%                    ")
    
    # 進捗表示を完了表示にして改行
    total_time = time.time() - epoch_start_time
    print(f"\r  {total_iterations}/{total_iterations} completed in {format_time(total_time)}", end="")
    
    return total_loss / len(train_loader)

def train_mncore2(model: nn.Module, train_loader: DataLoader, work_dir: str, 
                  addresses: Dict[str, Dict[str, int]], output_info: List[Dict[str, Any]], 
                  expected_batch_size: int, ignore_loss: bool = False,
                  device_id: int = 0, epoch: int = 0) -> float:
    """MN-Core2実機でコンパイルしたtrain_stepを使って訓練
    
    Args:
        model: PyTorchモデル
        train_loader: 訓練データローダー
        work_dir: gpfn3-loader用の作業ディレクトリ（input.*ファイルがある）
        addresses: DL/ULノードのアドレス情報
        output_info: 出力テンソル情報
        expected_batch_size: 期待されるバッチサイズ
        ignore_loss: lossを無視するか
        device_id: MN-Core2デバイスID
        epoch: 現在のエポック番号（0から開始）
    
    Returns:
        平均loss値
    """
    
    assert model is not None, "model must be initialized"
    model.train()
    total_loss = 0.0
    
    # パラメータのアドレスを事前に収集（毎回必要）
    param_addresses = {}
    for name, param in model.named_parameters():
        param_name = name.replace('.', '_')
        if param_name in addresses['inputs']:
            param_addresses[param_name] = addresses['inputs'][param_name]
    
    # エポック0の時のみメモリクリアとパラメータ初期転送
    if epoch == 0:
        # メモリをクリア（プログラム全体で1回のみ実行）
        clear_dram_mncore2(device_id)
        
        # モデルパラメータをセット（エポック0のみ）
        for name, param in model.named_parameters():
            param_name = name.replace('.', '_')
            if param_name in param_addresses:
                addr = param_addresses[param_name]
                param_data = param.data.cpu().numpy()
                write_dram_mncore2(param_data, addr, device_id)
    
    for data, target in train_loader:
        batch_size = data.shape[0]
        
        # バッチサイズが期待値と異なる場合はエラー
        if batch_size != expected_batch_size:
            raise ValueError(f"エラー: バッチサイズ{batch_size}は期待値{expected_batch_size}と異なります。DataLoaderにdrop_last=Trueを設定してください")
        
        # === 1. DRAMに入力データを書き込み ===
        
        # 入力データをセット（毎イテレーション必要）
        if 'input' in addresses['inputs']:
            addr = addresses['inputs']['input']
            input_data = data.view(batch_size, -1).numpy()
            write_dram_mncore2(input_data, addr, device_id)
        
        if 'target' in addresses['inputs']:
            addr = addresses['inputs']['target']
            write_dram_mncore2(target.numpy().astype(np.int32), addr, device_id)
        
        # === 2. 計算を実行 ===
        run_mncore2_computation(work_dir, device_id)
        
        # === 3. lossのみ読み出し（必要な場合） ===
        # パラメータはDRAM上でinplace更新されるため、ループ内での読み出しは不要
        if not ignore_loss and 'loss' in addresses['outputs']:
            for info in output_info:
                if info['name'] == 'loss':
                    total_loss += float(read_dram_mncore2(tuple(info['shape']), addresses['outputs']['loss'], device_id))
                    break
    
    # === 最終的にパラメータをDRAMから読み出してPyTorchモデルと同期 ===
    with torch.no_grad():
        for name, param in model.named_parameters():
            param_name = name.replace('.', '_')
            if param_name in param_addresses:
                addr = param_addresses[param_name]
                # DRAMから最新のパラメータを読み出し
                param_data = read_dram_mncore2(tuple(param.shape), addr, device_id)
                param.data.copy_(torch.from_numpy(param_data))
    
    return total_loss / len(train_loader)

def train_cpp(model: nn.Module, train_loader: DataLoader, cpp_func: Any, output_info: List[Dict[str, Any]], expected_batch_size: int, ignore_loss: bool = False) -> float:
    """C++でコンパイルしたtrain_stepを使って訓練（動的にパラメータを処理）"""
    assert model is not None, "model must be initialized"
    model.train()
    total_loss = 0.0
    
    for data, target in train_loader:
        batch_size = data.shape[0]
        
        # バッチサイズが期待値と異なる場合はエラー（drop_last=Trueを使用すべき）
        if batch_size != expected_batch_size:
            raise ValueError(f"エラー: バッチサイズ{batch_size}は期待値{expected_batch_size}と異なります。DataLoaderにdrop_last=Trueを設定してください")
        
        # 入力データを準備
        all_inputs_list = []
        all_inputs_list.append(('input', data.view(batch_size, -1)))
        all_inputs_list.append(('target', target))
        
        # モデルのパラメータを動的に取得
        for name, param in model.named_parameters():
            all_inputs_list.append((name, param.data))
        
        # 入力と出力に基づいて引数の型を準備
        argtypes = []
        
        # 入力ポインタを追加（リストの順序に従う）
        for name, tensor in all_inputs_list:
            if tensor.dtype in [torch.int32, torch.int64, torch.long]:
                argtypes.append(ctypes.POINTER(ctypes.c_int32))
            else:
                argtypes.append(ctypes.POINTER(ctypes.c_float))  # type: ignore
        
        # 出力ポインタを追加
        for info in output_info:
            argtypes.append(ctypes.POINTER(ctypes.c_float))  # type: ignore
        
        cpp_func.argtypes = argtypes
        cpp_func.restype = None
        
        # 入力配列を準備
        input_arrays = {}
        args = []
        
        for name, tensor in all_inputs_list:
            if tensor.dtype in [torch.int32, torch.int64, torch.long]:
                array = tensor.detach().cpu().numpy().astype(np.int32)
                array = np.ascontiguousarray(array)
                input_arrays[name] = array
                args.append(array.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)))
            else:
                array = tensor.detach().cpu().numpy().astype(np.float32)
                array = np.ascontiguousarray(array)
                input_arrays[name] = array
                args.append(array.ctypes.data_as(ctypes.POINTER(ctypes.c_float)))
        
        # 出力配列を準備（リストの順序に従う）
        output_arrays: Dict[str, np.ndarray] = {}
        
        for info in output_info:
            name = info['name']
            if info['is_scalar']:
                # スカラーの場合
                array = np.zeros(1, dtype=np.float32)
            elif info['shape']:
                # 形状が既知の場合
                array = np.zeros(info['shape'], dtype=np.float32)
            else:
                raise ValueError(f"出力 {name} の形状を決定できません")
            
            output_arrays[name] = array
            args.append(array.ctypes.data_as(ctypes.POINTER(ctypes.c_float)))
        
        # C++関数を呼び出す
        cpp_func(*args)
        
        # 更新されたパラメータをモデルに反映
        with torch.no_grad():
            for name, param in model.named_parameters():
                updated_name = f"updated_{name.replace('.', '_')}"
                if updated_name in output_arrays:
                    updated_val = output_arrays[updated_name]
                    if isinstance(updated_val, np.ndarray):
                        param.data.copy_(torch.from_numpy(updated_val))
        
        # lossを取得（ignore_lossがFalseの場合のみ）
        if not ignore_loss and 'loss' in output_arrays:
            total_loss += float(output_arrays['loss'])
    return total_loss / len(train_loader)

def train_model_cpp(batch_size=256, test_batch_size=1000, ignore_loss=False, backend="cpp"):
    """C++またはエミュレータでコンパイルしたtrain_stepを使ってMNISTモデルを訓練する
    
    Args:
        batch_size: 訓練バッチサイズ
        test_batch_size: テストバッチサイズ
        ignore_loss: loss計算をスキップするか
        backend: バックエンド選択
            - "cpp": C++での実行（安定・高速）
            - "emu": MN-Coreエミュレータでの実行（安定・低速）
            - "mncore2": MN-Core2実機での実行
    """
    
    # 乱数シードを固定
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    # backendパラメータのバリデーション
    valid_backends = ["cpp", "emu", "mncore2"]
    if backend not in valid_backends:
        raise ValueError(f"バックエンド '{backend}' はサポートされていません。{valid_backends} のいずれかを指定してください。")
    
    # データの準備
    transform = get_mnist_transform()
    
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)
    
    # drop_last=Trueで最後の不完全なバッチを削除
    # generatorを使って再現可能なシャッフルを行う
    generator = torch.Generator()
    generator.manual_seed(42)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, generator=generator)
    test_loader = DataLoader(test_dataset, batch_size=test_batch_size)
    
    # drop_lastによってスキップされるサンプル数を警告
    total_samples = len(train_dataset)
    dropped_samples = total_samples % batch_size
    if dropped_samples > 0:
        print(f"注意: バッチサイズ{batch_size}に合わせるため、最後の{dropped_samples}サンプルをスキップします（{total_samples}サンプル中）")
    
    # グローバル変数を初期化
    init_globals()
    
    # 指定されたバッチサイズでコードを生成
    func, dummy_inputs = get_example_function_and_inputs('train_step', batch_size)
    output_dir = f"/tmp/train_step_{backend}_batch{batch_size}"
    
    # エクスポート（ignore_lossがTrueの場合はlossを除外）
    output_dir, _, _ = export_function(func, dummy_inputs, output_dir, 
                                       ignore_outputs=["loss"] if ignore_loss else None)
    
    onnx_path = os.path.join(output_dir, "model.onnx")
    
    if backend == "cpp":
        # C++コードを生成
        cpp_code = generate_cpp_from_onnx(onnx_path)
        
        # C++コードをファイルに保存
        cpp_path = os.path.join(output_dir, "forward_backward.cpp")
        with open(cpp_path, 'w') as f:
            f.write(cpp_code)
        
        # C++関数をロード
        cpp_func, output_info, _ = load_cpp_train_step(cpp_path, parse_cpp_output_info, compile_cpp_library)
        
        # 学習ループ
        from .train import model
        train_model = model
        for epoch in range(100):
            loss = train_cpp(train_model, train_loader, cpp_func, output_info, batch_size, ignore_loss)
            acc, _ = evaluate(train_model, test_loader)
            if ignore_loss:
                print(f'Epoch {epoch+1:2d}: Accuracy: {acc:.2f}% (no loss)', flush=True)
            else:
                print(f'Epoch {epoch+1:2d}: Loss: {loss:.4f}, Accuracy: {acc:.2f}%', flush=True)
    
    elif backend in ["emu", "mncore2"]:
        # MN-Core（エミュレータまたは実機）モード
        is_emu = (backend == "emu")
        mode_name = "エミュレータ" if is_emu else "MN-Core2実機"
        print(f"=== {mode_name}モード初期化 ===")
        
        # MN-Core用に変換
        print("MN-Core用ONNXに変換中...")
        mn_onnx_path = os.path.join(output_dir, "mn_model.onnx")
        transform_to_mncore(onnx_path, mn_onnx_path)
        
        # VSMコードを生成（MN-Core版のONNXから）
        print("VSMコード生成中...")
        vsm_code = generate_vsm_from_onnx(mn_onnx_path)
        
        # VSMファイルに保存
        vsm_path = os.path.join(output_dir, "model.vsm")
        with open(vsm_path, 'w') as f:
            f.write(vsm_code)
        
        # バックエンド固有の準備
        if is_emu:
            # エミュレータ用：アセンブル
            print("VSMをアセンブル中...")
            asm_path = os.path.join(output_dir, "model.asm")
            assemble_vsm(vsm_path, asm_path)
            
            # ASMコードを読み込み
            with open(asm_path, 'r') as f:
                asm_base = f.read()
        else:
            # MN-Core2実機用：作業ディレクトリ準備
            work_dir = os.path.join(output_dir, "mncore2_work")
            print(f"作業ディレクトリ準備中: {work_dir}")
            prepare_mncore2_workspace(vsm_path, work_dir)
            device_id = 0  # デフォルトデバイスID
        
        # アドレス情報を取得（MN-Core版のONNXから）
        print("アドレス情報を抽出中...")
        addresses = extract_addresses_from_onnx(mn_onnx_path)
        
        # 出力情報を取得
        output_info = get_emu_output_info(onnx_path, ignore_loss)
        
        # 学習ループ
        print(f"=== {mode_name}訓練開始 ===")
        from .train import model
        train_model = model
        for epoch in range(100):
            print(f'Epoch {epoch+1:2d}: ', end='', flush=True)
            
            if is_emu:
                loss = train_emu(train_model, train_loader, asm_base, addresses, output_info, 
                               batch_size, ignore_loss, test_loader)
            else:
                loss = train_mncore2(train_model, train_loader, work_dir, addresses, output_info, 
                                   batch_size, ignore_loss, device_id, epoch=epoch)
            
            acc, _ = evaluate(train_model, test_loader)
            if ignore_loss:
                print(f'Accuracy: {acc:.2f}%', flush=True)
            else:
                print(f'Loss: {loss:.4f}, Accuracy: {acc:.2f}%', flush=True)
    
    else:
        raise ValueError(f"Unknown backend: {backend}")
    
    # 学習済みモデルを返す
    from .train import model
    trained_model = model
    return trained_model