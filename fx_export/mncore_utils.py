#!/usr/bin/env python3
"""MN-Core utility functions for emulator and hardware execution"""

import numpy as np
import struct
import onnx
from typing import List, Tuple, Dict, Optional, Any, Union
import re
import subprocess
import os
import platform
import json

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
            bytes_val = struct.pack('>f', val)  # ビッグエンディアン
            hex_str = bytes_val.hex().upper()
            hex_strings.append(hex_str)
        return hex_strings, 4
    elif data.dtype == np.int32 or data.dtype == np.int64:
        # int32を16進数に変換
        flat_data = data.flatten()
        hex_strings = []
        for val in flat_data:
            # int32を4バイトのバイト列に変換し、16進数文字列に
            bytes_val = struct.pack('>i', int(val))  # ビッグエンディアン
            hex_str = bytes_val.hex().upper()
            hex_strings.append(hex_str)
        return hex_strings, 4
    elif data.dtype == np.bool_ or data.dtype == bool:
        # bool型はfloat32に変換
        float_data = data.astype(np.float32)
        return numpy_to_hex_string(float_data)
    else:
        raise ValueError(f"Unsupported dtype: {data.dtype}")

def write_to_single_dram(
    lines: List[str], 
    dram_id: int, 
    base_addr: int, 
    hex_strings: List[str], 
    values: np.ndarray,
    show_comments: bool = True
) -> None:
    """
    単一のDRAMにデータを書き込む行を生成する。
    
    Args:
        lines: 出力行のリスト
        dram_id: DRAM番号（0-3）
        base_addr: ベースアドレス（lw単位）
        hex_strings: 16進数文字列のリスト
        values: 元の値（デバッグ用）
        show_comments: コメントを表示するかどうか
    """
    # valuesをフラットにしておく
    if values.ndim > 1:
        values = values.flatten()
    elif values.ndim == 0:
        values = np.array([values])  # スカラーの場合は1次元配列に変換
    
    # データ型を判定
    if values.dtype == np.int32 or values.dtype == np.int64:
        type_str = "Int"
    elif values.dtype == np.float32:
        type_str = "Float"
    elif values.dtype == np.bool_ or values.dtype == bool:
        type_str = "Float"  # boolはfloat32として扱う
        values = values.astype(np.float32)
    else:
        type_str = "Unknown"
    
    # 2つのfloat32/int32（4バイト）を1つのlw（8バイト）にまとめる
    addr = base_addr
    i = 0
    while i < len(hex_strings):
        if len(hex_strings[i]) == 8:  # float32またはint32（4バイト）
            if i + 1 < len(hex_strings):
                # 2つの4バイト値を結合
                combined_hex = hex_strings[i] + hex_strings[i+1]
                if show_comments:
                    if values.dtype == np.int32 or values.dtype == np.int64:
                        val_str = f"values=[{int(values[i])}, {int(values[i+1])}] / {type_str} @[0],[1]"
                    else:
                        val_str = f"values=[{values[i]:.6g}, {values[i+1]:.6g}] / {type_str} @[0],[1]"
                else:
                    val_str = ""
                i += 2
            else:
                # 最後の1つの場合（パディング）
                combined_hex = hex_strings[i] + "00000000"
                if show_comments:
                    if values.dtype == np.int32 or values.dtype == np.int64:
                        val_str = f"values=[{int(values[i])}, 0] / {type_str} @[0],[1]"
                    else:
                        val_str = f"values=[{values[i]:.6g}, 0.0] / {type_str} @[0],[1]"
                else:
                    val_str = ""
                i += 1
        else:
            raise ValueError(f"Unexpected hex string length: {len(hex_strings[i])}")
        
        # r <dram_id> <9桁の16進アドレス> 001 <16桁の16進データ>
        if show_comments and val_str:
            line = f"r {dram_id} {addr:09X} 001 {combined_hex} # {val_str}"
        else:
            line = f"r {dram_id} {addr:09X} 001 {combined_hex}"
        lines.append(line)
        addr += 1

def generate_vsm_input_commands(
    data: Union[np.ndarray, List, Tuple],
    base_addr: int,
    show_comments: bool = True
) -> List[str]:
    """
    テンソルデータをDRAMにセットするVSMコマンドを生成
    
    Args:
        data: 入力データ
        base_addr: ベースアドレス
        show_comments: コメントを表示するかどうか
    
    Returns:
        VSMコマンドのリスト
    """
    lines: List[str] = []
    
    if not isinstance(data, np.ndarray):
        data = np.array(data)
    
    # bool型の場合はfloat32に変換
    if data.dtype == np.bool_ or data.dtype == bool:
        data = data.astype(np.float32)
    
    # データを16進数に変換
    hex_strings, bytes_per_elem = numpy_to_hex_string(data)
    total_elements = len(hex_strings)
    
    # 要素数に応じてDRAMに分散
    if total_elements <= 16:
        # DRAM 0のみに書き込み
        write_to_single_dram(lines, 0, base_addr, hex_strings, data, show_comments)
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
            write_to_single_dram(lines, dram_id, base_addr, dram_data, dram_values, show_comments)
    
    return lines

def write_expectations_from_dram(
    lines: List[str], 
    dram_id: int, 
    base_addr: int, 
    values: Optional[np.ndarray] = None,
    num_elements: Optional[int] = None,
    show_comments: bool = True
) -> None:
    """
    単一のDRAMから期待値を読み出す行を生成する。
    
    Args:
        lines: 出力行のリスト
        dram_id: DRAM番号（0-3）
        base_addr: ベースアドレス（10進数）
        values: 期待値（show_comments=Trueの場合のみ使用）
        num_elements: 読み出す要素数（valuesがNoneの場合に使用）
        show_comments: expectコメントを表示するかどうか
    """
    if values is not None:
        # bool型の場合はfloat32にキャスト（0.0 または 1.0）
        if values.dtype == np.bool_ or values.dtype == bool:
            values = values.astype(np.float32)
        if values.dtype != np.float32 and values.dtype != np.int32:
            values = values.astype(np.float32)
        num_elements = len(values)
    elif num_elements is None:
        raise ValueError("Either values or num_elements must be provided")
    
    addr = base_addr
    i = 0
    while i < num_elements:
        if show_comments and values is not None:
            if i + 1 < num_elements:
                if values.dtype == np.int32:
                    expect_str = f"expect=[{int(values[i])}, {int(values[i+1])}] / Int @[{dram_id},{i}],[{dram_id},{i+1}]"
                else:
                    expect_str = f"expect=[{values[i]:.6g}, {values[i+1]:.6g}] / Float @[{dram_id},{i}],[{dram_id},{i+1}]"
                i += 2
            else:
                if values.dtype == np.int32:
                    expect_str = f"expect=[{int(values[i])}, 0] / Int @[{dram_id},{i}],[{dram_id},{i+1}]"
                else:
                    expect_str = f"expect=[{values[i]:.6g}, 0.0] / Float @[{dram_id},{i}],[{dram_id},{i+1}]"
                i += 1
            line = f"d getd $d{addr}n{dram_id} 1 # {expect_str} atol=1e-04"
        else:
            line = f"d getd $d{addr}n{dram_id} 1"
            i += 2  # d getdは2要素ずつ読む
        
        lines.append(line)
        addr += 1

def generate_vsm_output_commands(
    shape: Tuple[int, ...],
    base_addr: int,
    show_comments: bool = False
) -> List[str]:
    """
    DRAMから出力データを取得するVSMコマンドを生成
    
    Args:
        shape: 出力テンソルの形状
        base_addr: ベースアドレス
        show_comments: expectコメントを表示するかどうか
    
    Returns:
        VSMコマンドのリスト
    """
    lines: List[str] = []
    
    # 総要素数を計算
    total_elements = int(np.prod(shape)) if shape else 1
    
    if total_elements <= 16:
        # DRAM 0のみから読み出し
        write_expectations_from_dram(lines, 0, base_addr, None, total_elements, show_comments)
    else:
        # 4 DRAMから読み出し
        if total_elements % 4 != 0:
            raise ValueError(f"Element count {total_elements} is not divisible by 4")
        
        elements_per_dram = total_elements // 4
        for dram_id in range(4):
            write_expectations_from_dram(lines, dram_id, base_addr, None, elements_per_dram, show_comments)
    
    return lines

def extract_addresses_from_onnx(onnx_model_path: str) -> Dict[str, Dict[str, int]]:
    """
    ONNXモデルからDL/ULノードのアドレスを抽出
    
    Args:
        onnx_model_path: ONNXモデルのパス
    
    Returns:
        {
            'inputs': {'input': addr, 'fc1_weight': addr, ...},
            'outputs': {'output': addr, 'updated_fc1_weight': addr, ...}
        }
    """
    model = onnx.load(onnx_model_path)
    graph = model.graph
    
    input_addresses = {}
    output_addresses = {}
    
    # グラフの入力名リストを作成
    graph_input_names = [inp.name for inp in graph.input]
    graph_output_names = [out.name for out in graph.output]
    initializer_names = [init.name for init in graph.initializer]
    
    for node in graph.node:
        if node.op_type == 'DL':
            # DLノードから入力アドレスを取得
            if node.input and node.input[0]:
                tensor_name = node.input[0]
                
                # addr_in0を取得
                for attr in node.attribute:
                    if attr.name == 'addr_in0' and attr.HasField('i'):
                        addr = attr.i
                        if tensor_name in graph_input_names or tensor_name in initializer_names:
                            input_addresses[tensor_name] = addr
                        break
    
    # 各出力について、関連するULノードを見つける
    for out in graph.output:
        out_name = out.name
        for node in graph.node:
            if node.op_type == 'UL':
                if node.input and len(node.input) > 0:
                    input_name = node.input[0]
                    # 出力名に関連するULノードを見つける
                    if (out_name in input_name or 
                        input_name.startswith(out_name) or
                        input_name == f"{out_name}_compute_out" or
                        input_name == f"{out_name}_scaled"):
                        for attr in node.attribute:
                            if attr.name == 'addr_out0' and attr.HasField('i'):
                                output_addresses[out_name] = attr.i
                                break
                        break
    
    return {
        'inputs': input_addresses,
        'outputs': output_addresses
    }

def parse_emulator_output(output: str) -> List[float]:
    """
    エミュレータ出力をパースして値のリストを返す
    
    入力形式:
    DEBUG-DRAM(n0,26000000):(0) (0x3F800000)
    
    Args:
        output: エミュレータのstderr出力
    
    Returns:
        浮動小数点数のリスト
    """
    values = []
    
    # パターン: DEBUG-DRAM(n<dram_id>,<addr>):(<val1>) (0x<hex>)
    pattern = r'DEBUG-DRAM\(n\d+,\d+\):\([^)]*\) \(0x([0-9A-Fa-f]+)\)'
    
    for match in re.finditer(pattern, output):
        hex_str = match.group(1)
        
        # 16桁の16進数を2つの32ビット浮動小数点数に変換
        if len(hex_str) == 16:
            # 前半8桁と後半8桁を分離
            hex1 = hex_str[:8]
            hex2 = hex_str[8:]
            
            # それぞれを32ビット浮動小数点数に変換
            for hex_val in [hex1, hex2]:
                int_val = int(hex_val, 16)
                # 32ビット整数を浮動小数点数として解釈
                float_bytes = struct.pack('>I', int_val)
                float_val = struct.unpack('>f', float_bytes)[0]
                values.append(float_val)
        else:
            raise ValueError(f"Unexpected hex string length: {len(hex_str)}")
    
    return values

def assemble_vsm(vsm_path: str, asm_path: str, assembler_path: Optional[str] = None) -> None:
    """
    VSMファイルをアセンブルしてASMファイルを生成
    
    Args:
        vsm_path: 入力VSMファイルのパス
        asm_path: 出力ASMファイルのパス
        assembler_path: アセンブラの実行パス
    """
    with open(vsm_path, 'r') as f:
        vsm_content = f.read()
    
    # OSを判定してアセンブラの実行方法を切り替え
    if platform.system() == "Darwin":  # macOS用
        # macOSではシェルスクリプト経由でDockerを使用
        script_path = os.path.join(
            os.path.dirname(__file__), 
            "tools", "assembler"
        )
        result = subprocess.run(
            ["sh", script_path],
            input=vsm_content,
            capture_output=True,
            text=True
        )
    else:
        # Linux環境では直接実行
        if assembler_path is None:
            # fx_export/ ディレクトリから2つ上の mnist/ ディレクトリ
            mnist_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            assembler_path = os.path.join(mnist_root, "judge/judge-py/mncore2_emuenv/assemble3")
        result = subprocess.run(
            [assembler_path, "--instruction-mode", "flat"],
            input=vsm_content,
            capture_output=True,
            text=True
        )
    
    if result.returncode != 0:
        raise RuntimeError(f"Assembler failed: {result.stderr}")
    
    # ASMファイルに保存
    with open(asm_path, 'w') as f:
        f.write(result.stdout)

def run_emulator(commands: str, emulator_path: Optional[str] = None) -> str:
    """
    エミュレータでコマンドを実行
    
    Args:
        commands: 実行するコマンド（ASM + d getd等）
        emulator_path: エミュレータの実行パス
    
    Returns:
        エミュレータのstderr出力
    """
    # OSを判定してエミュレータの実行方法を切り替え
    if platform.system() == "Darwin":  # macOS用
        # macOSではシェルスクリプト経由でDockerを使用
        script_path = os.path.join(
            os.path.dirname(__file__), 
            "tools", "emulator"
        )
        result = subprocess.run(
            ["sh", script_path],
            input=commands,
            capture_output=True,
            text=True
        )
    else:
        # Linux環境では直接実行
        if emulator_path is None:
            # fx_export/ ディレクトリから2つ上の mnist/ ディレクトリ
            mnist_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            emulator_path = os.path.join(mnist_root, "judge/judge-py/mncore2_emuenv/gpfn3_package_main")
        if not os.path.exists(emulator_path):
            raise FileNotFoundError(f"Emulator not found: {emulator_path}")
        
        result = subprocess.run(
            [emulator_path],
            input=commands,
            capture_output=True,
            text=True
        )
    
    if result.returncode != 0:
        raise RuntimeError(f"Emulator failed: {result.stderr}")
    
    return result.stderr


# MN-Core2実機実行用の関数群

def prepare_mncore2_workspace(vsm_path: str, work_dir: str, assembler_path: Optional[str] = None) -> None:
    """
    VSMをgpfn3-loader用にアセンブルして作業ディレクトリを準備
    
    Args:
        vsm_path: VSMファイルのパス
        work_dir: 作業ディレクトリ
        assembler_path: アセンブラのパス
    """
    os.makedirs(work_dir, exist_ok=True)
    
    # アセンブラのパスを決定
    if assembler_path is None:
        # fx_export/ ディレクトリから2つ上の mnist/ ディレクトリ
        mnist_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        assembler_path = os.path.join(mnist_root, "judge/judge-py/mncore2_emuenv/assemble3")
    if not os.path.exists(assembler_path):
        raise FileNotFoundError(f"Assembler not found: {assembler_path}")
    
    with open(vsm_path, 'r') as f:
        vsm_content = f.read()
    
    result = subprocess.run(
        [assembler_path, "--loader", "--instruction-mode", "flat", "-o", "input"],
        input=vsm_content,
        cwd=work_dir,
        capture_output=True,
        text=True
    )
    
    if result.returncode != 0:
        raise RuntimeError(f"Assembly failed: {result.stderr}")


def write_dram_mncore2(
    data: np.ndarray,
    address: int,
    device_id: int = 0,
    temp_dir: str = "/tmp"
) -> None:
    """
    MN-Core2実機のDRAMにデータを書き込み
    
    Args:
        data: 書き込むデータ（NumPy配列）
        address: DRAMアドレス（LW単位）
        device_id: デバイスID
        temp_dir: 一時ファイル保存ディレクトリ
    """
    
    total_elements = data.size
    
    # MN-Coreはビッグエンディアンで演算するため、バイトスワップが必要
    # データ型に応じて適切に処理
    data_flat = data.flatten()
    data_be: np.ndarray  # data_beの型宣言
    if data_flat.dtype in [np.int32, np.int64]:
        # 整数型の場合はint32として扱う
        data_int32 = data_flat.astype(np.int32)
        data_be = data_int32.byteswap()  # リトルエンディアンからビッグエンディアンへ
    else:
        # float32データをビッグエンディアンに変換
        data_float32 = data_flat.astype(np.float32)
        data_be = data_float32.byteswap()  # リトルエンディアンからビッグエンディアンへ
    
    # バイナリファイル作成（ビッグエンディアンデータをそのままファイルに書き込み）
    temp_file = os.path.join(temp_dir, f"dram_write_{os.getpid()}_{address}.bin")
    # byteswapされたデータをファイルに書き込み
    data_be.tofile(temp_file)
    
    # 設定ファイル作成
    config_file = os.path.join(temp_dir, f"dram_write_{os.getpid()}_{address}.json")
    
    if total_elements <= 16:
        # DRAM0のみ使用
        lw_count = (total_elements + 1) // 2  # float32 2個で1LW、切り上げ
        config = {
            "load": [{
                "region": "dram0",
                "offset": address,
                "length": lw_count
            }]
        }
    else:
        # 4DRAM分散 (Sequential)
        if total_elements % 4 != 0:
            raise ValueError(f"Element count {total_elements} must be divisible by 4 for 4-DRAM distribution")
        elements_per_dram = total_elements // 4
        lw_per_dram = elements_per_dram // 2  # float32 2個で1LW
        config = {
            "load": [{
                "region": "dram0,dram1,dram2,dram3",
                "offset": address,
                "length": lw_per_dram
            }]
        }
    
    with open(config_file, 'w') as f:
        json.dump(config, f)
    
    # gpfn3-smi load実行
    result = subprocess.run(
        ["gpfn3-smi", "load", "-config", config_file, "-file", temp_file, str(device_id)],
        capture_output=True,
        timeout=30
    )
    
    # 一時ファイル削除
    try:
        os.remove(temp_file)
        os.remove(config_file)
    except FileNotFoundError:
        pass
    
    if result.returncode != 0:
        stderr_str = result.stderr.decode() if isinstance(result.stderr, bytes) else str(result.stderr)
        stdout_str = result.stdout.decode() if isinstance(result.stdout, bytes) else str(result.stdout)
        raise RuntimeError(f"gpfn3-smi load failed:\nstderr: {stderr_str}\nstdout: {stdout_str}")


def read_dram_mncore2(
    shape: Tuple[int, ...],
    address: int,
    device_id: int = 0,
    temp_dir: str = "/tmp"
) -> np.ndarray:
    """
    MN-Core2実機のDRAMからデータを読み出し
    
    Args:
        shape: 読み出すデータの形状
        address: DRAMアドレス（LW単位）
        device_id: デバイスID
        temp_dir: 一時ファイル保存ディレクトリ
    
    Returns:
        読み出したデータ（NumPy配列）
    """
    
    total_elements = int(np.prod(shape)) if shape else 1
    
    # 設定ファイル作成
    config_file = os.path.join(temp_dir, f"dram_read_{os.getpid()}_{address}.json")
    output_file = os.path.join(temp_dir, f"dram_read_{os.getpid()}_{address}.bin")
    
    if total_elements <= 16:
        # DRAM0のみから読み出し
        lw_count = (total_elements + 1) // 2  # float32 2個で1LW、切り上げ
        config = {
            "save": [{
                "region": "dram0",
                "offset": address,
                "length": lw_count
            }]
        }
    else:
        # 4DRAMから読み出し (Sequential)
        if total_elements % 4 != 0:
            raise ValueError(f"Element count {total_elements} must be divisible by 4 for 4-DRAM distribution")
        elements_per_dram = total_elements // 4
        lw_per_dram = elements_per_dram // 2  # float32 2個で1LW
        config = {
            "save": [{
                "region": "dram0,dram1,dram2,dram3",
                "offset": address,
                "length": lw_per_dram
            }]
        }
    
    with open(config_file, 'w') as f:
        json.dump(config, f)
    
    # gpfn3-smi save実行
    result = subprocess.run(
        ["gpfn3-smi", "save", "-config", config_file, "-file", output_file, str(device_id)],
        capture_output=True,
        timeout=30
    )
    
    if result.returncode != 0:
        # 一時ファイル削除
        try:
            os.remove(config_file)
        except FileNotFoundError:
            pass
        stderr_str = result.stderr.decode() if isinstance(result.stderr, bytes) else str(result.stderr)
        stdout_str = result.stdout.decode() if isinstance(result.stdout, bytes) else str(result.stdout)
        raise RuntimeError(f"gpfn3-smi save failed:\nstderr: {stderr_str}\nstdout: {stdout_str}")
    
    # バイナリファイル読み込み（MN-Coreからのビッグエンディアンデータ）
    # NumPyのfromfileはネイティブバイトオーダー（little-endian）として読み込む
    data_be = np.fromfile(output_file, dtype=np.float32)
    
    # 一時ファイル削除
    try:
        os.remove(config_file)
        os.remove(output_file)
    except FileNotFoundError:
        pass
    
    # データサイズの確認
    if len(data_be) < total_elements:
        raise RuntimeError(f"Expected {total_elements} elements but got {len(data_be)} from DRAM read")
    
    # MN-Coreはビッグエンディアンなので、リトルエンディアンに変換
    data = data_be[:total_elements].byteswap()  # ビッグエンディアンからリトルエンディアンへ
    
    # 形状を復元
    if shape:
        data = data.reshape(shape)
    
    return data


def clear_dram_mncore2(device_id: int = 0) -> None:
    """
    MN-Core2実機のDRAMメモリをクリア
    
    Args:
        device_id: デバイスID
    """
    result = subprocess.run(
        ["gpfn3-smi", "clear", "--zero", str(device_id)],
        capture_output=True,
        timeout=30
    )
    
    if result.returncode != 0:
        stderr_str = result.stderr.decode() if isinstance(result.stderr, bytes) else str(result.stderr)
        stdout_str = result.stdout.decode() if isinstance(result.stdout, bytes) else str(result.stdout)
        raise RuntimeError(f"gpfn3-smi clear failed:\nstderr: {stderr_str}\nstdout: {stdout_str}")


_GPFN3_LOADER_PATH = None
def _get_gpfn3_loader_path() -> str:
    global _GPFN3_LOADER_PATH
    if _GPFN3_LOADER_PATH is None:
        local_loader = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "gpfn3-loader")
        if os.path.exists(local_loader) and os.access(local_loader, os.X_OK):
            _GPFN3_LOADER_PATH = local_loader
        else:
            raise FileNotFoundError(f"gpfn3-loader が見つからないか実行できません: {local_loader}")
    return _GPFN3_LOADER_PATH

def run_mncore2_computation(
    work_dir: str,
    device_id: int = 0,
    timeout: int = 180  # デフォルトを180秒に設定（MN-Core2実機は非常に時間がかかる）
):
    """
    gpfn3-loaderで計算を実行
    
    Args:
        work_dir: input.*ファイルがある作業ディレクトリ
        device_id: デバイスID（現在は未使用）
        timeout: タイムアウト秒数
    
    Returns:
        gpfn3-loaderの出力
    """
    loader_command = [_get_gpfn3_loader_path(), "-d", str(device_id)]
    
    result = subprocess.run(
        loader_command,
        cwd=work_dir,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        timeout=timeout
    )
    
    if result.returncode != 0:
        raise RuntimeError(f"gpfn3-loader failed with return code {result.returncode}")
