import re
from typing import List, Dict, Any, Optional, Union

import os
testcase_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "testcases")

def inspect(filename: str) -> dict:
    """
    VSMテストケースファイルを解析し、入出力情報とレイアウトを抽出する。
    
    Args:
        filename: VSMファイル名（拡張子なし、例: "add_colvec"）またはフルパス
    
    Returns:
        以下の形式の辞書:
        {
            'inputs': [[in0_type, in0_offset, in0_layout], [in1_type, in1_offset, in1_layout], ...],
            'output': [out_type, out_offset, out_layout]
        }
    
    各要素の意味:
        - type: "LM0", "LM1", "DRAM" のいずれか
        - offset: 整数値
        - layout: "((8_L2B:1, 4:2, 8_L1B:1), (2:1, 4_PE:1, 2_W:1))" のようなレイアウト文字列
    """
    # フルパスかチェック
    if os.path.isfile(filename):
        filepath = filename
    elif os.path.isfile(filename + '.vsm'):
        filepath = filename + '.vsm'
    else:
        filepath = f"{testcase_dir}/{filename}.vsm"
    
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    inputs: List[List[Union[str, int, Any]]] = []
    output: Optional[List[Union[str, int, Any]]] = None
    
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        
        # レイアウトを含むIn/Outセクションヘッダーをチェック
        in_match = re.match(r'# ======= In\((\d+)\):\s*(\(\(.+\)\))@(\w+)', line)
        out_match = re.match(r'# ======= Out\((\d+)\):\s*(\(\(.+\)\))@(\w+)', line)
        
        if in_match:
            in_index = int(in_match.group(1))
            layout = in_match.group(2)  # 完全なレイアウト文字列
            mem_type = in_match.group(3)  # メモリ種別: LM0/LM1/DRAM
            
            # 次の行を処理してオフセットを取得
            i += 1
            if i < len(lines):
                next_line = lines[i].strip()
                offset = extract_input_offset(next_line, mem_type)
                
                # 正しい位置に挿入 (In(0)はindex 0、In(1)はindex 1、など)
                while len(inputs) <= in_index:
                    inputs.append([])  # 空のリストを追加
                inputs[in_index] = [mem_type, offset, layout]
        
        elif out_match:
            out_index = int(out_match.group(1))
            layout = out_match.group(2)  # 完全なレイアウト文字列
            mem_type = out_match.group(3)  # メモリ種別: LM0/LM1/DRAM
            
            # 次の行を処理してオフセットを取得
            i += 1
            if i < len(lines):
                next_line = lines[i].strip()
                offset = extract_output_offset(next_line, mem_type)
                
                # Out(0)をoutputとして保存
                output = [mem_type, offset, layout]
        
        i += 1
    
    # inputsから空のエントリを削除
    inputs = [item for item in inputs if item]
    
    return {
        'inputs': inputs,
        'output': output
    }


def extract_input_offset(line: str, mem_type: str) -> int:
    """
    メモリタイプに基づいて入力行からオフセットを抽出する。
    
    Args:
        line: 解析する入力行
        mem_type: メモリタイプ ("LM0", "LM1", "DRAM")
    
    Returns:
        抽出されたオフセット値（Int）
    """
    if mem_type.startswith("LM"):
        # LM入力の場合: "d set $lm0n0c0b0p0" または "d set $lm16p0"
        # "lm"の後、次の文字の前の数値を抽出
        match = re.search(r'\$lm(\d+)[a-zA-Z]', line)
        if match:
            return int(match.group(1))
    elif mem_type == "DRAM":
        # DRAM入力の場合: "r 0 000000000 001" - 16進数オフセット（3番目のフィールド）を抽出
        parts = line.split()
        if len(parts) >= 3:
            return int(parts[2], 16)
    
    return 0


def extract_output_offset(line: str, mem_type: str) -> int:
    """
    メモリタイプに基づいて出力行からオフセットを抽出する。
    
    Args:
        line: 解析する出力行
        mem_type: メモリタイプ ("LM0", "LM1", "DRAM")
    
    Returns:
        抽出されたオフセット値（Int）
    """
    if mem_type.startswith("LM"):
        # LM出力の場合: "d getd $lm0n0c0b0m0p0" - "lm"の後、次の文字の前の数値を抽出
        match = re.search(r'\$lm(\d+)[a-zA-Z]', line)
        if match:
            return int(match.group(1))
    elif mem_type == "DRAM":
        # DRAM出力の場合: "d getd $d0n0" - "d"の後、"n"の前の数値を抽出
        match = re.search(r'\$d(\d+)n', line)
        if match:
            return int(match.group(1))
    
    return 0


def converter(filename: str, target_config: dict) -> str:
    """
    新しいメモリタイプとオフセットでVSMファイルを変換する。
    
    Args:
        filename: VSMファイル名 (例: "matmul_tra_pkg_lmdram_16_256_16.vsm")
        target_config: 辞書形式: {
                           'inputs': [[in0_type, in0_offset], [in1_type, in1_offset], ...],
                           'output': [out_type, out_offset]
                       }
    
    Returns:
        変更されたVSMコンテンツの文字列
    """
    # target_configから設定を取得してリスト形式に変換
    config_list = []
    if 'output' in target_config:
        config_list.append(target_config['output'])
    if 'inputs' in target_config:
        config_list.extend(target_config['inputs'])
    target_config_list = config_list
    
    # 現在の設定を取得
    info = inspect(filename[:-4])  # .vsm拡張子を削除
    
    # 新しい形式の設定をリストにフラット化
    current_config = []
    if info['output']:
        current_config.append(info['output'][:2])  # [type, offset]のみ
    for inp in info['inputs']:
        current_config.append(inp[:2])  # [type, offset]のみ
    
    # 設定が同じ長さであることを検証
    if len(current_config) != len(target_config_list):
        raise ValueError(f"設定の長さが一致しません: current={len(current_config)}, target={len(target_config_list)}")
    
    # メモリタイプの変更を検証 (LM0/LM1の変更は許可、DRAM/LMの変更は不許可)
    for i, (current, target) in enumerate(zip(current_config, target_config_list)):
        current_type, current_offset = current
        target_type, target_offset = target
        
        # DRAMとLM間の変更を試みているかチェック
        if (current_type == 'DRAM' and target_type.startswith('LM')) or \
           (current_type.startswith('LM') and target_type == 'DRAM'):
            raise ValueError(f"DRAMとLM間の変更はできません (index {i}): {current_type} -> {target_type}")
    
    # ファイルを読み込む（フルパスかチェック）
    if os.path.isfile(filename):
        filepath = filename
    else:
        filepath = f"{testcase_dir}/{filename}"
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    # ファイルを処理
    result_lines = []
    current_section = None
    section_index = None
    
    for line in lines:
        # In/Outセクションヘッダーをチェック
        in_match = re.match(r'(# ======= In\((\d+)\): .+@)(\w+)( .+)', line)
        out_match = re.match(r'(# ======= Out\((\d+)\): .+@)(\w+)( .+)', line)
        
        if in_match:
            section_index = int(in_match.group(2)) + 1  # In(0) は index 1
            current_section = 'input'
            
            # 必要に応じてヘッダーのメモリタイプを更新
            target_type = target_config_list[section_index][0]
            # 元の行末を保持
            line_ending = '\n' if line.endswith('\n') else ''
            new_line = in_match.group(1) + target_type + in_match.group(4) + line_ending
            result_lines.append(new_line)
            
        elif out_match:
            section_index = int(out_match.group(2))  # Out(0) は index 0
            current_section = 'output'
            
            # 必要に応じてヘッダーのメモリタイプを更新
            target_type = target_config_list[section_index][0]
            # 元の行末を保持
            line_ending = '\n' if line.endswith('\n') else ''
            new_line = out_match.group(1) + target_type + out_match.group(4) + line_ending
            result_lines.append(new_line)
            
        elif current_section and section_index is not None:
            # この行がデータ行かセクションの終わりかをチェック
            if line.strip() == '' or (line.startswith('#') and '=======' in line):
                # セクションの終わり
                current_section = None
                section_index = None
                result_lines.append(line)
            else:
                # セクションタイプに基づいてデータ行を処理
                current_offset = current_config[section_index][1]
                target_offset = target_config_list[section_index][1]
                target_type = target_config_list[section_index][0]
                diff = target_offset - current_offset
                
                if current_section == 'input':
                    new_line = process_input_line(line, target_type, diff)
                else:  # 出力
                    new_line = process_output_line(line, target_type, diff)
                
                result_lines.append(new_line)
        else:
            result_lines.append(line)
    
    return ''.join(result_lines)


def process_input_line(line: str, target_type: str, diff: int) -> str:
    """
    アドレス調整を行って入力データ行を処理する。
    
    Args:
        line: 処理対象の入力データ行
        target_type: ターゲットメモリタイプ ("LM0", "LM1", "DRAM")
        diff: オフセットの差分（新しいオフセット - 古いオフセット）
    
    Returns:
        アドレス調整済みの行文字列
    """
    # 元の行末を保持
    line_ending = '\n' if line.endswith('\n') else ''
    line_content = line.rstrip('\n')
    
    if target_type.startswith('LM'):
        # LM入力: "d set $lm0n0c0b0p0 ..." または "d set $ln0n0c0b0p0 ..."
        # $lmと$lnの両方のパターンにマッチ
        match = re.match(r'(d set \$l[mn])(\d+)([a-zA-Z].+)', line_content)
        if match:
            new_offset = int(match.group(2)) + diff
            # ターゲットタイプに基づいて正しいプレフィックスを決定
            prefix = 'd set $lm' if target_type == 'LM0' else 'd set $ln'
            return prefix + str(new_offset) + match.group(3) + line_ending
    elif target_type == 'DRAM':
        # DRAM入力: "r 0 000000000 001 ..."
        match = re.match(r'(r \d+ )([0-9a-fA-F]+)( .+)', line_content)
        if match:
            current_hex = int(match.group(2), 16)
            new_hex = current_hex + diff
            # 元と同じ幅でフォーマット
            hex_width = len(match.group(2))
            return match.group(1) + f"{new_hex:0{hex_width}x}" + match.group(3) + line_ending
    
    return line


def process_output_line(line: str, target_type: str, diff: int) -> str:
    """
    アドレス調整を行って出力データ行を処理する。
    
    Args:
        line: 処理対象の出力データ行
        target_type: ターゲットメモリタイプ ("LM0", "LM1", "DRAM")
        diff: オフセットの差分（新しいオフセット - 古いオフセット）
    
    Returns:
        アドレス調整済みの行文字列
    """
    # 元の行末を保持
    line_ending = '\n' if line.endswith('\n') else ''
    line_content = line.rstrip('\n')
    
    if target_type.startswith('LM'):
        # LM出力: "d getd $lm0n0c0b0m0p0 ..." または "d getd $ln0n0c0b0m0p0 ..."
        # $lmと$lnの両方のパターンにマッチ
        match = re.match(r'(d getd \$l[mn])(\d+)([a-zA-Z].+)', line_content)
        if match:
            new_offset = int(match.group(2)) + diff
            # ターゲットタイプに基づいて正しいプレフィックスを決定
            prefix = 'd getd $lm' if target_type == 'LM0' else 'd getd $ln'
            return prefix + str(new_offset) + match.group(3) + line_ending
    elif target_type == 'DRAM':
        # DRAM出力: "d getd $d0n0 ..."
        match = re.match(r'(d getd \$d)(\d+)(n.+)', line_content)
        if match:
            new_offset = int(match.group(2)) + diff
            return match.group(1) + str(new_offset) + match.group(3) + line_ending
    
    return line


# 関数のテスト
if __name__ == "__main__":
    # inspectをテスト
    print("=== inspect関数のテスト ===")
    print("add_colvec.vsm:")
    result = inspect("add_colvec")
    print(f"  出力: {result['output']}")
    print(f"  入力: {result['inputs']}")
    print()
    
    print("dram_dl_16.vsm:")
    result = inspect("dram_dl_16")
    print(f"  出力: {result['output']}")
    print(f"  入力: {result['inputs']}")
    print("\n" + "="*50 + "\n")

    # converter関数をテスト
    print("=== converter関数のテスト ===")
    # テストケース1: 辞書形式で指定
    converted_result = converter("dram_dl_16.vsm", {
        'output': ['LM0', 10],
        'inputs': [['DRAM', 5]]
    })
    print("テスト1 - dram_dl_16.vsmの変換:")
    print('\n'.join(converted_result.split('\n')[:15]))
    print("\n" + "="*50 + "\n")
    
    # テストケース2: 辞書形式で指定
    converted_result2 = converter("add_colvec.vsm", {
        'output': ['LM1', 10],
        'inputs': [['LM1', 20], ['LM1', 30]]
    })
    print("テスト2 - add_colvec.vsmの変換:")
    print('\n'.join(converted_result2.split('\n')[:15]))
