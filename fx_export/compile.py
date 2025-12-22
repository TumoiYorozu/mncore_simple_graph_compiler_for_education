#!/usr/bin/env python3

import subprocess
import os
import shutil
import re
import onnx
from onnx import shape_inference, numpy_helper
from typing import Dict, List, Any, Tuple, Optional, Union

# メインのtrainモジュールから定数をimport

# operatorシステムをimport
from .operators import get_operator

# 変数名生成用の一時カウンタ
_temp_counter = 0

# ============= Compile関連の関数 =============

def get_cpp_headers() -> List[str]:
    """生成されたコードに必要な共通のC++ヘッダーを返す"""
    return [
        '#include "matrix_operations.hpp"',
        "",
        "// ctypes用のエクスポート関数",
        'extern "C" {',
        ""
    ]

def copy_matrix_operations_header(output_dir: str) -> str:
    """matrix_operations.hppを出力ディレクトリにコピーする"""
    # fx_export/tools/ にある matrix_operations.hpp へのパス
    src_path: str = os.path.join(os.path.dirname(__file__), "tools", "matrix_operations.hpp")
    dst_path: str = os.path.join(output_dir, "matrix_operations.hpp")
    shutil.copy2(src_path, dst_path)
    return dst_path

def parse_cpp_output_info(cpp_code: str) -> List[Dict[str, Any]]:
    """
    C++コードから出力情報を解析する
    
    Returns:
        list: [{'name': str, 'shape': list, 'is_scalar': bool}, ...]
    """
    
    output_info = []
    
    # OUTPUT_INFO_STARTを探す
    if '// OUTPUT_INFO_START' not in cpp_code:
        raise ValueError("OUTPUT_INFO_START not found in C++ code. Cannot parse output information.")
    
    # OUTPUT_INFO_ENDを探す
    if '// OUTPUT_INFO_END' not in cpp_code:
        raise ValueError("OUTPUT_INFO_END not found in C++ code. Cannot parse output information.")
    
    # OUTPUT_INFO_STARTとOUTPUT_INFO_ENDの間の情報を抽出
    pattern = r'// OUTPUT_INFO_START\n(.*?)// OUTPUT_INFO_END'
    match = re.search(pattern, cpp_code, re.DOTALL)
    
    if not match:
        raise ValueError("C++コードに出力情報が見つかりません。OUTPUT_INFO_START/ENDセクションが必要です。")
    
    info_section = match.group(1)
    # 各OUTPUT行を解析
    output_pattern = r'// OUTPUT: (\w+) shape=\[(.*?)\]'
    for line_match in re.finditer(output_pattern, info_section):
        name = line_match.group(1)
        shape_str = line_match.group(2)
        
        # 形状を解析
        if shape_str.strip():
            shape = [int(x.strip()) for x in shape_str.split(',') if x.strip()]
        else:
            shape = []
        
        # shape=[] の場合はスカラー
        is_scalar = len(shape) == 0
        
        output_info.append({
            'name': name,
            'shape': shape,
            'is_scalar': is_scalar
        })
    
    if not output_info:
        raise ValueError("C++コードに有効な出力情報が見つかりません。")
    
    return output_info

def compile_cpp_library(cpp_code: str, output_path: str) -> str:
    """C++コードを共有ライブラリにコンパイルする"""
    cpp_file: str = output_path + ".cpp"
    lib_file: str = output_path + ".so"
    
    # matrix operationsヘッダーを同じディレクトリにコピー
    output_dir = os.path.dirname(output_path)
    copy_matrix_operations_header(output_dir)
    
    # C++コードをファイルに書き込む
    with open(cpp_file, "w") as f:
        f.write(cpp_code)
    
    # 共有ライブラリにコンパイル
    compile_cmd = [
        "g++", 
        "-O3", 
        "-fPIC", 
        "-shared",
        "-std=c++20",
        cpp_file,
        "-o", lib_file
    ]
    
    result = subprocess.run(compile_cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"コンパイルエラー: {result.stderr}")
    
    return lib_file

def generate_code_from_onnx(onnx_path: str, output_prefix: str = "/tmp", mn_onnx_path: Optional[str] = None) -> Tuple[str, str]:
    """
    ONNXモデルからPythonとC++コードを生成する
    
    Args:
        onnx_path: 元のONNXファイルパス
        output_prefix: 出力先ディレクトリ
        mn_onnx_path: MN-Core用ONNXファイルパス（指定時はVSM生成）
    
    Returns:
        tuple: (python_path, cpp_path)
    """
    python_code_path = f"{output_prefix}/forward_backward.py"
    cpp_code_path = f"{output_prefix}/forward_backward.cpp"
    vsm_code_path = f"{output_prefix}/forward_backward.vsm"
    
    # Pythonコードを生成
    print("\n=== コード生成 ===")
    print(f"使用するONNXファイル: {onnx_path}")
    python_code = generate_python_from_onnx_simple(onnx_path)
    
    with open(python_code_path, "w") as f:
        f.write(python_code)
    print(f"Python: {python_code_path}")
    
    # C++コードを生成
    cpp_code = generate_cpp_from_onnx(onnx_path)
    
    with open(cpp_code_path, "w") as f:
        f.write(cpp_code)
    print(f"C++:    {cpp_code_path}")
    
    # VSMコードを生成（MN-Core用ONNXが指定されている場合のみ）
    if mn_onnx_path:
        vsm_code = generate_vsm_from_onnx(mn_onnx_path)
        
        with open(vsm_code_path, "w") as f:
            f.write(vsm_code)
        print(f"VSM:    {vsm_code_path}")
    
    # 行列演算ヘッダーをコピー
    copy_matrix_operations_header(output_prefix)
    
    return python_code_path, cpp_code_path

def get_output_var_name(node: onnx.NodeProto) -> str:
    """処理関数の出力変数名を決定する共通ロジック

    Args:
        node: ONNXノード

    Returns:
        使用する文字列変数名
    """
    global _temp_counter

    # ノードにvar_name属性があるかチェック
    for attr in node.attribute:
        if attr.name == 'var_name':
            var_name_raw = attr.s.decode('utf-8') if isinstance(attr.s, bytes) else str(attr.s)
            if 'DEBUG_ONNX' in os.environ:
                print(f"ONNX属性からvar_name '{var_name_raw}' を使用")
            # C++関数名との衝突を避けるためにサフィックスを追加
            if var_name_raw in ['matmul', 'relu', 'transpose', 'softmax', 'log', 'sub', 'mul']:
                var_name_raw = f"{var_name_raw}_result"
            return var_name_raw

    # デフォルトで一時変数を使用
    temp_name = f"temp_{_temp_counter}"
    _temp_counter += 1
    return temp_name

def get_tensor_shape(tensor_name: str, graph: onnx.GraphProto) -> List[int]:
    """ONNXグラフからテンソルの形状情報を取得

    Args:
        tensor_name: テンソル名
        graph: ONNXグラフオブジェクト

    Returns:
        次元のリスト
    """
    def extract_shape(tensor_type: Any) -> List[int]:
        """tensor_typeから形状を抽出（動的次元に対応）"""
        shape: List[int] = []
        for dim in tensor_type.shape.dim:
            if dim.HasField('dim_value'):
                shape.append(dim.dim_value)
            else:
                raise ValueError(f"Cannot determine concrete shape for tensor {tensor_name}. All shapes must be concrete in ONNX.")
        return shape
    
    # 入力をチェック
    for inp in graph.input:
        if inp.name == tensor_name and inp.type.HasField('tensor_type'):
            return extract_shape(inp.type.tensor_type)

    # 出力をチェック
    for out in graph.output:
        if out.name == tensor_name and out.type.HasField('tensor_type'):
            return extract_shape(out.type.tensor_type)

    # value_info（中間テンソル）をチェック
    for vi in graph.value_info:
        if vi.name == tensor_name and vi.type.HasField('tensor_type'):
            return extract_shape(vi.type.tensor_type)

    raise ValueError(f"tensorのshape情報が見つかりません: {tensor_name}")

def get_output_mapping(graph: onnx.GraphProto, variable_map: Dict[str, str]) -> Dict[str, str]:
    """ONNX出力名を対応する変数にマッピングする"""
    output_mapping: Dict[str, str] = {}

    # グラフからすべての出力名を取得
    output_names = set()
    for out in graph.output:
        output_names.add(out.name)

    # 出力名を変数にマッピング
    for out_name in output_names:
        if out_name in variable_map:
            output_mapping[out_name] = variable_map[out_name]
        else:
            # この出力を生成するノードを見つける
            for node in graph.node:
                if out_name in node.output:
                    # RepeatedScalarContainerをリストに変換してからindex
                    output_list = list(node.output)
                    idx = output_list.index(out_name)
                    if len(output_list) > idx and output_list[idx] in variable_map:
                        output_mapping[out_name] = variable_map[output_list[idx]]
                        break
            else:
                raise ValueError(f"出力 '{out_name}' の変数マッピングが見つかりませんでした")

    return output_mapping

def generate_cpp_from_onnx(onnx_model_path: str) -> str:
    """ONNXモデルからC++コードを生成する"""

    # ONNXモデルを読み込む
    model = onnx.load(onnx_model_path)
    # 中間テンソルの形状を取得するために形状推論を実行
    model = shape_inference.infer_shapes(model)
    graph = model.graph

    lines: List[str] = []
    lines.extend(get_cpp_headers())
    
    # 出力情報をコメントとして追加
    lines.append("// OUTPUT_INFO_START")
    for out in graph.output:
        # 形状情報を取得
        shape = []
        if out.type.HasField('tensor_type'):
            for dim in out.type.tensor_type.shape.dim:
                if dim.HasField('dim_value'):
                    shape.append(dim.dim_value)
        # スカラーの場合は shape=[] として表現
        if len(shape) == 0 or (len(shape) == 1 and shape[0] == 1 and 'loss' in out.name):
            lines.append(f"// OUTPUT: {out.name} shape=[]")
        else:
            lines.append(f"// OUTPUT: {out.name} shape=[{', '.join(map(str, shape))}]")
    lines.append("// OUTPUT_INFO_END")
    lines.append("")

    # 関数シグネチャを構築 - 汎用名を使用
    sig_lines = ["void forward_backward("]

    # ONNXからすべての入力名を取得
    param_names = []
    for inp in graph.input:
        param_names.append(inp.name)
        # 適切なポインタ型のためにデータ型をチェック
        if inp.type.HasField('tensor_type'):
            dtype = inp.type.tensor_type.elem_type
            if dtype == 6 or dtype == 7: # ONNXでのINT32, INT64
                sig_lines.append(f"    const int* {inp.name}_ptr,")
            else:
                sig_lines.append(f"    const float* {inp.name}_ptr,")

    # ONNXグラフ出力に基づく出力ポインタ
    # 出力名の重複を処理するためのカウンタ
    output_name_counts: Dict[str, int] = {}
    for out in graph.output:
        # 出力名の重複を確認
        if out.name in output_name_counts:
            output_name_counts[out.name] += 1
            unique_name = f"{out.name}_{output_name_counts[out.name]}"
        else:
            output_name_counts[out.name] = 0
            unique_name = out.name
        
        # 出力のデータ型をチェック
        if out.type.HasField('tensor_type'):
            dtype = out.type.tensor_type.elem_type
            if dtype == 6 or dtype == 7: # ONNXでのINT32, INT64
                sig_lines.append(f"    int* {unique_name}_ptr,")
            else:
                sig_lines.append(f"    float* {unique_name}_ptr,")
        else:
            # デフォルトはfloat
            sig_lines.append(f"    float* {unique_name}_ptr,")

    sig_lines[-1] = sig_lines[-1].rstrip(',')  # 最後のカンマを削除
    sig_lines.append(") {")
    lines.extend(sig_lines)

    # 変数名を追跡
    variable_map = {}
    # 新しい関数のために一時カウンターをリセット
    global _temp_counter
    _temp_counter = 0

    # すべての入力とパラメータを読み込む
    lines.append("    // 入力とパラメータを読み込む")
    for inp in graph.input:
        if inp.type.HasField('tensor_type'):
            # 形状を取得（動的次元の場合はデフォルト値を使用）
            shape = []
            for dim in inp.type.tensor_type.shape.dim:
                if dim.HasField('dim_value'):
                    shape.append(dim.dim_value)
                else:
                    raise ValueError(f"Cannot determine concrete shape for input {inp.name}. All shapes must be concrete in ONNX.")
            dtype = inp.type.tensor_type.elem_type

            if dtype == 6 or dtype == 7: # ONNXでのINT32, INT64
                # 整数配列
                if len(shape) == 1:
                    # lines.append(f"    array<int, {shape[0]}> {inp.name};")
                    # lines.append(f"    memcpy({inp.name}.data(), {inp.name}_ptr, {shape[0]} * sizeof(int));")
                    lines.append(f"    const array {inp.name} = load<{shape[0]}, int>({inp.name}_ptr);")
                variable_map[inp.name] = inp.name
            else:
                # 通常のfloatテンソル
                if len(shape) == 2:
                    lines.append(f"    const Matrix<{shape[0]}, {shape[1]}> {inp.name} = load<{shape[0]}, {shape[1]}>({inp.name}_ptr);")
                elif len(shape) == 1:
                    lines.append(f"    const Vector<{shape[0]}> {inp.name} = load<{shape[0]}>({inp.name}_ptr);")
                elif len(shape) == 0:
                    # スカラー
                    lines.append(f"    const float {inp.name} = *{inp.name}_ptr;")
                variable_map[inp.name] = inp.name
    lines.append("")

    # ノードを処理
    lines.append("    // 計算")

    # すべてのノードを順番に処理
    for node in graph.node:
        op_type = node.op_type
        if 'DEBUG_ONNX' in os.environ:
            print(f"{op_type}ノードを処理中: 出力={node.output}")

        node_code = None
        
        # 新しいoperatorクラスの仕組みを使用
        try:
            operator_class = get_operator(op_type)
            operator = operator_class(node, graph, variable_map)
            node_code = operator.generate_cpp()
        except ValueError as e:
            # 未知の操作 - エラーを発生させる
            raise ValueError(f"未知のONNX operation type: {op_type}。この操作のC++コードを生成できません。") from e
        
        # ノードのコードがあればlinesに追加
        if node_code:
            lines.extend(node_code)

    # ONNXグラフ出力に基づいてすべての出力を保存
    lines.append("")
    lines.append("    // 出力を保存")

    # 出力マッピングを取得
    try:
        output_mapping = get_output_mapping(graph, variable_map)
    except Exception as e:
        print(f"get_output_mappingでエラー: {e}")
        print(f"variable_mapのkey: {list(variable_map.keys())}")
        print(f"graphの出力: {[out.name for out in graph.output]}")
        raise

    # 出力名の重複を処理するためのカウンタ（関数シグネチャと一致させる）
    output_name_counts = {}
    
    # 各出力を保存
    for out in graph.output:
        out_name = out.name
        
        # 出力名の重複を確認（関数シグネチャの生成と同じロジック）
        if out_name in output_name_counts:
            output_name_counts[out_name] += 1
            unique_ptr_name = f"{out_name}_{output_name_counts[out_name]}_ptr"
        else:
            output_name_counts[out_name] = 0
            unique_ptr_name = f"{out_name}_ptr"
        
        if out_name not in output_mapping:
            raise ValueError(f"出力 '{out_name}' がoutput_mappingに見つかりません")
        else:
            var_name = output_mapping[out_name]

        # すべての出力に統一されたsave関数を使用
        lines.append(f"    save({var_name}, {unique_ptr_name});")

    lines.extend([
        "}",
        "",
        "}  // extern \"C\""
    ])

    cpp_code = "\n".join(lines)
    
    return cpp_code

def generate_python_from_onnx_simple(onnx_model_path: str) -> str:
    """ONNXモデルから簡単なPythonコードを生成する（モデルフリー関数用）"""
    
    # ONNXモデルを読み込む
    model = onnx.load(onnx_model_path)
    model = shape_inference.infer_shapes(model)
    graph = model.graph
    
    lines: List[str] = []
    lines.append("import torch")
    lines.append("import torch.nn.functional as F")
    lines.append("")
    lines.append("def forward_backward(**inputs):")
    lines.append("    # 入力を取得")
    
    # ONNXグラフのすべての入力を引数として扱う
    for inp in graph.input:
        lines.append(f"    {inp.name} = inputs['{inp.name}']")
    
    # 初期化子（定数）をロード
    lines.append("")
    lines.append("    # 初期化子（定数）")
    for init in graph.initializer:
        tensor = numpy_helper.to_array(init)
        if tensor.shape == ():  # スカラー
            value = float(tensor)
            lines.append(f"    {init.name} = {value}")
        elif len(tensor.shape) == 1 and tensor.shape[0] == 1:  # 1要素のテンソル
            value = float(tensor[0])
            lines.append(f"    {init.name} = torch.tensor({value})")
        elif len(tensor.shape) == 1:  # 1次元テンソル
            values_list = [float(v) for v in tensor]
            lines.append(f"    {init.name} = torch.tensor({values_list})")
        else:  # それ以上の次元のテンソル
            raise NotImplementedError(f"初期化子 '{init.name}' の形状 {list(tensor.shape)} は未対応です")
    
    lines.append("")
    lines.append("    # 計算")
    
    # 変数マッピング
    var_map: Dict[str, str] = {}
    for inp in graph.input:
        var_map[inp.name] = inp.name
    # 初期化子も変数マップに追加
    for init in graph.initializer:
        var_map[init.name] = init.name
    
    # ノードを処理
    for node in graph.node:
        op_type = node.op_type
        
        # 新しいoperatorクラスの仕組みを使用
        operator_class = get_operator(op_type)
        operator = operator_class(node, graph, var_map)
        node_code = operator.generate_python()
        lines.extend(node_code)
        
    
    lines.append("")
    lines.append("    # 結果を返す")
    lines.append("    return {")
    
    # ONNXグラフの出力を追加
    for output in graph.output:
        output_name = output.name
        if output_name in var_map:
            python_var = var_map[output_name]
            lines.append(f"        '{output_name}': {python_var},")
    
    # 全ての中間変数も追加（unit test用）
    for var_name, python_var in var_map.items():
        # 入力とすでに追加した出力は除外
        is_input = any(inp.name == var_name for inp in graph.input)
        is_output = any(out.name == var_name for out in graph.output)
        if not is_input and not is_output:
            lines.append(f"        '{var_name}': {python_var},")
    
    # 最後のカンマを削除
    if lines[-1].endswith(','):
        lines[-1] = lines[-1][:-1]
    
    lines.append("    }")
    
    return "\n".join(lines)


def add_mv_wait(vsm_lines: List[str]) -> List[str]:
    """VSMコード内のmv命令にタグを追加し、wait命令を挿入する
    
    Args:
        vsm_lines: VSMコードの行のリスト
    
    Returns:
        処理後のVSMコードの行のリスト
    """
    
    fixed_tag = "i9f"
    # mv命令のパターン: mvp, mvd, mvb/mvb2/mvb4, mvr2.../mvr4.../mvrffadd等
    mv_pattern = re.compile(r'(mvp|mvd|mvb[24]?|mvr[24]?[^/\s]*)/n([^\s;#]*)')
    tag_pattern = re.compile(r'i[0-9a-zA-Z]{2}')
    result = []
    used_tags = set()
    
    for line in vsm_lines:
        # コメントの開始位置を見つける
        comment_idx = line.find('#')
        if comment_idx != -1:
            code_part = line[:comment_idx]
            comment_part = line[comment_idx:]
        else:
            code_part = line
            comment_part = ''
        
        # コード部分に内容がある場合のみ処理
        if code_part.strip():
            # mv命令を検索
            match = mv_pattern.search(code_part)
            if match:
                opcode = match.group(1)
                options = match.group(2)
                # タグが含まれているかチェック
                tag_match = tag_pattern.search(options)
                
                if tag_match:
                    # タグが既に存在する場合
                    existing_tag = tag_match.group(0)
                    used_tags.add(existing_tag)
                    if existing_tag == fixed_tag:
                        raise ValueError(f"エラー: 固定タグ '{fixed_tag}' は既にVSMコード内で使用されています")
                    # タグが既にある場合は元の行をそのまま追加
                    result.append(line)
                else:
                    # タグがない場合は固定タグを追加
                    start_pos = match.start()
                    end_pos = match.end()
                    before_mv = code_part[:start_pos]
                    mv_with_tag = f"{opcode}/n{options}{fixed_tag}"
                    after_mv = code_part[end_pos:]
                    new_line = f"{before_mv}{mv_with_tag}{after_mv}{comment_part}"
                    result.append(new_line)
                    result.append(f"nop; wait {fixed_tag}  # Auto-tagged")
            else:
                # mv命令でない場合は元の行をそのまま追加
                result.append(line)
        else:
            result.append(line)
    return result


def get_tensor_shape_optional(tensor_name: str, graph: onnx.GraphProto) -> Optional[List[Union[int, str]]]:
    """テンソルの形状を取得する
    
    Args:
        tensor_name: テンソル名
        graph: ONNXグラフ
    
    Returns:
        形状のリスト。見つからない場合はNone
    """
    # value_infoから検索
    for value_info in graph.value_info:
        if value_info.name == tensor_name:
            if value_info.type.tensor_type and value_info.type.tensor_type.shape:
                return [dim.dim_value if hasattr(dim, 'dim_value') else '?' 
                       for dim in value_info.type.tensor_type.shape.dim]
    
    # グラフ入力から検索
    for inp in graph.input:
        if inp.name == tensor_name:
            if inp.type.tensor_type and inp.type.tensor_type.shape:
                return [dim.dim_value if hasattr(dim, 'dim_value') else '?' 
                       for dim in inp.type.tensor_type.shape.dim]
    
    # グラフ出力から検索
    for out in graph.output:
        if out.name == tensor_name:
            if out.type.tensor_type and out.type.tensor_type.shape:
                return [dim.dim_value if hasattr(dim, 'dim_value') else '?' 
                       for dim in out.type.tensor_type.shape.dim]
    
    # 初期化子から検索
    for init in graph.initializer:
        if init.name == tensor_name:
            return list(init.dims)
    
    return None


def generate_vsm_from_onnx(onnx_model_path: str) -> str:
    """ONNXモデルからVSM（MN-Core用アセンブリ）コードを生成する"""
    
    # ONNXモデルを読み込む
    model = onnx.load(onnx_model_path)
    model = shape_inference.infer_shapes(model)
    graph = model.graph
    
    lines: List[str] = []
    lines.append("# VSM (MN-Core Assembly) Code")
    lines.append("# Generated from ONNX model")
    lines.append("#")
    lines.append(f"# ONNX Model: {os.path.basename(onnx_model_path)}")
    lines.append(f"# Graph Name: {graph.name}")
    lines.append("")
    lines.append("# ====== Graph Inputs ======")
    
    # グラフの入力情報
    for inp in graph.input:
        shape = get_tensor_shape(inp.name, graph)
        shape_str = f"shape={shape}" if shape else "shape=[]"
        lines.append(f"# Input: {inp.name}, {shape_str}, dtype=float32")
    
    lines.append("")
    lines.append("# ====== Graph Outputs ======")
    
    # グラフの出力情報
    for out in graph.output:
        shape = get_tensor_shape(out.name, graph)
        shape_str = f"shape={shape}" if shape else "shape=[]"
        lines.append(f"# Output: {out.name}, {shape_str}, dtype=float32")
    
    lines.append("")
    lines.append("# ====== Initializers (Constants) ======")
    
    # 初期化子（定数）の情報
    for init in graph.initializer:
        tensor = numpy_helper.to_array(init)
        lines.append(f"# Initializer: {init.name}, shape={list(tensor.shape)}, dtype={tensor.dtype}")
    
    lines.append("")
    lines.append("# ====== Operations ======")
    lines.append("")
    
    # 変数マッピング
    var_map: Dict[str, str] = {}
    for inp in graph.input:
        var_map[inp.name] = inp.name
    for init in graph.initializer:
        var_map[init.name] = init.name
    
    # 各ノードを処理
    for i, node in enumerate(graph.node):
        if i > 0:
            lines.append("\nnop/3")
        op_type = node.op_type
        
        lines.append(f"# --- Operation {i+1}: {op_type} ---")
        lines.append(f"# Node name: {node.name}")
        
        # DRAMアドレス情報を取得（DL/ULノードおよび計算ノードの場合）
        dram_info = []
        for attr in node.attribute:
            if attr.name in ['in_dram_addr_lw', 'in_dram_len_lw', 'out_dram_addr_lw', 'out_dram_len_lw']:
                dram_info.append(f"{attr.name}={attr.i}")
        
        if dram_info:
            lines.append(f"# DRAM: {', '.join(dram_info)}")
        
        # 属性を分類（入力用、出力用、その他）
        input_attrs: Dict[int, Dict[str, Any]] = {}  # 対応: index -> {attr_name: value}
        output_attrs: Dict[int, Dict[str, Any]] = {}  # 対応: index -> {attr_name: value}
        other_attrs: Dict[str, Any] = {}
        
        for attr in node.attribute:
            # 入力用属性のパターン (_in0, _in1, など)
            in_match = re.match(r'(.+)_in(\d+)$', attr.name)
            # 出力用属性のパターン (_out0, _out1, など)
            out_match = re.match(r'(.+)_out(\d+)$', attr.name)
            
            # 属性値を取得
            if attr.HasField('i'):
                value = attr.i
            elif attr.HasField('f'):
                value = attr.f
            elif attr.HasField('s'):
                value = attr.s.decode('utf-8') if isinstance(attr.s, bytes) else str(attr.s)
            elif len(attr.ints) > 0:
                value = list(attr.ints)
            elif len(attr.floats) > 0:
                value = list(attr.floats)
            elif len(attr.strings) > 0:
                value = [s.decode('utf-8') if isinstance(s, bytes) else str(s) for s in attr.strings]
            else:
                value = None
            
            if in_match:
                # 入力用属性
                attr_base_name = in_match.group(1)
                input_idx = int(in_match.group(2))
                if input_idx not in input_attrs:
                    input_attrs[input_idx] = {}
                input_attrs[input_idx][attr_base_name] = value
            elif out_match:
                # 出力用属性
                attr_base_name = out_match.group(1)
                output_idx = int(out_match.group(2))
                if output_idx not in output_attrs:
                    output_attrs[output_idx] = {}
                output_attrs[output_idx][attr_base_name] = value
            else:
                # その他の属性（DRAMアドレス情報は既に処理済みなのでスキップ）
                if attr.name not in ['in_dram_addr_lw', 'in_dram_len_lw', 'out_dram_addr_lw', 'out_dram_len_lw']:
                    other_attrs[attr.name] = value
        
        # その他の属性を表示
        if other_attrs:
            for key in sorted(other_attrs.keys()):
                value = other_attrs[key]
                lines.append(f"# Attr: {key}={value}")
        
        # 入力情報
        for j, input_name in enumerate(node.input):
            if input_name:
                # 初期化子かどうかチェック
                is_initializer = any(init.name == input_name for init in graph.initializer)
                input_type = "initializer" if is_initializer else "tensor"
                # 特殊な入力（depth, values等）は形状取得をスキップ
                try:
                    shape = get_tensor_shape(input_name, graph)
                    shape_str = f"shape={shape}" if shape else "shape=unknown"
                except ValueError:
                    # テンソルとして存在しない入力（例：OneHotのdepth, values）
                    shape_str = "shape=N/A"
                attr_strs = []
                if j in input_attrs:
                    for attr_name in sorted(input_attrs[j].keys()):
                        attr_value = input_attrs[j][attr_name]
                        attr_strs.append(f"{attr_name}={attr_value}")
                
                attr_info = f", {', '.join(attr_strs)}" if attr_strs else ""
                
                lines.append(f"# Input{j}: {input_name}, {shape_str}, type={input_type}{attr_info}")
        
        # 出力情報
        for j, output_name in enumerate(node.output):
            if output_name:
                try:
                    shape = get_tensor_shape(output_name, graph)
                    shape_str = f"shape={shape}" if shape else "shape=unknown"
                except ValueError:
                    shape_str = "shape=unknown"
                attr_strs = []
                if j in output_attrs:
                    for attr_name in sorted(output_attrs[j].keys()):
                        attr_value = output_attrs[j][attr_name]
                        attr_strs.append(f"{attr_name}={attr_value}")
                
                attr_info = f", {', '.join(attr_strs)}" if attr_strs else ""
                
                lines.append(f"# Output{j}: {output_name}, {shape_str}{attr_info}")
        
        lines.append("")
        
        # オペレータークラスを使用してVSMコードを生成
        try:
            operator_class = get_operator(op_type)
        except ValueError as e:
            raise ValueError(f"Unknown operator '{op_type}' in VSM generation") from e
        
        # オペレーターインスタンスを作成してVSMコードを生成
        operator = operator_class(node, graph, var_map)
        
        # VSMコードを生成（エラーはキャッチしない）
        vsm_code = operator.generate_vsm()
        
        # mv命令にタグを追加し、wait命令を挿入
        vsm_code = add_mv_wait(vsm_code)
        
        lines.extend(vsm_code)
        
        # 出力を変数マップに追加
        for output in node.output:
            if output:
                var_map[output] = output
    
    lines.append("")
    lines.append("# ====== End of VSM Code ======")
    lines.append("")
    
    return "\n".join(lines)
