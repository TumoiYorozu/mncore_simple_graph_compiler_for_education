#!/usr/bin/env python3
"""
MN-Core用のONNX変換モジュール
ONNXモデルをMN-Coreアーキテクチャ用にDL/ULノードを追加して変換する
"""

from typing import List, Dict, Set, Union, Tuple, Optional
import onnx
from onnx import helper
import os
import sys
from fx_export.operators import get_operator, OPERATOR_REGISTRY, get_default_layout


def get_type_from_graph(graph: onnx.GraphProto, tensor_name: str) -> int:
    """
    グラフから指定されたテンソルのデータ型を取得
    """
    # グラフ入力から探す
    for inp in graph.input:
        if inp.name == tensor_name and inp.type.HasField('tensor_type'):
            return int(inp.type.tensor_type.elem_type)
    
    # value_infoから探す
    for vi in graph.value_info:
        if vi.name == tensor_name and vi.type.HasField('tensor_type'):
            return int(vi.type.tensor_type.elem_type)
    
    # 出力から探す
    for out in graph.output:
        if out.name == tensor_name and out.type.HasField('tensor_type'):
            return int(out.type.tensor_type.elem_type)
    
    # 見つからない場合はFLOATをデフォルトにする
    return onnx.TensorProto.FLOAT

def get_shape_from_graph(graph: onnx.GraphProto, tensor_name: str, 
                         shape_cache: Dict[str, List[int]]) -> List[int]:
    """
    グラフから指定されたテンソルの形状を取得
    shape_cacheは既に見つかった形状を記録（ULノード出力など）
    """
    # キャッシュから探す
    if tensor_name in shape_cache:
        return shape_cache[tensor_name]
    
    # グラフ入力から探す
    for inp in graph.input:
        if inp.name == tensor_name:
            shape = []
            for dim in inp.type.tensor_type.shape.dim:
                if dim.dim_value:
                    shape.append(dim.dim_value)
                elif dim.dim_param:
                    # symbolic dimensionは-1として扱う
                    shape.append(-1)
            # 空の形状（スカラー）も正常な値
            shape_cache[tensor_name] = shape
            return shape
    
    # value_infoから探す
    for vi in graph.value_info:
        if vi.name == tensor_name:
            shape = []
            for dim in vi.type.tensor_type.shape.dim:
                if dim.dim_value:
                    shape.append(dim.dim_value)
                elif dim.dim_param:
                    shape.append(-1)
            # 空の形状（スカラー）も正常な値
            shape_cache[tensor_name] = shape
            return shape
    
    # 出力から探す
    for out in graph.output:
        if out.name == tensor_name:
            shape = []
            for dim in out.type.tensor_type.shape.dim:
                if dim.dim_value:
                    shape.append(dim.dim_value)
                elif dim.dim_param:
                    shape.append(-1)
            # 空の形状（スカラー）も正常な値
            shape_cache[tensor_name] = shape
            return shape
    
    # 初期化子から探す
    for init in graph.initializer:
        if init.name == tensor_name:
            shape = list(init.dims)
            shape_cache[tensor_name] = shape
            return shape
    
    # 形状が見つからない場合はエラー
    raise ValueError(f"Shape not found for tensor '{tensor_name}' in graph")


def calculate_tensor_len(shape: List[int]) -> int:
    """形状から要素数を計算"""
    if not shape:
        # 空の形状（スカラー）は要素数1
        return 1
    
    total = 1
    for dim in shape:
        if dim > 0:  # -1はシンボリックディメンション
            total *= dim
        elif dim < 0:
            # シンボリックディメンションの場合はエラー
            raise ValueError(f"Cannot calculate tensor length with symbolic dimension: {dim}")
    return total


def verify_tag_layout_consistency(model: onnx.ModelProto) -> None:
    """
    MN-Core ONNXモデルのtag/layout整合性を検証
    DL→Op、Op→UL間の接続でtag/layoutが一致しているか確認
    """
    graph = model.graph
    
    # ノード名から出力テンソル名へのマッピングを構築
    node_outputs: Dict[str, List[str]] = {}
    for node in graph.node:
        node_outputs[node.name] = list(node.output)
    
    # テンソル名から生成元ノードへのマッピングを構築
    tensor_producer: Dict[str, onnx.NodeProto] = {}
    for node in graph.node:
        for output in node.output:
            if output:
                tensor_producer[output] = node
    
    inconsistencies = []
    
    for node in graph.node:
        # 各入力について、接続元ノードとのtag/layout整合性を確認
        for i, input_tensor in enumerate(node.input):
            if not input_tensor:
                continue
                
            # 入力テンソルの生成元ノードを探す
            if input_tensor not in tensor_producer:
                # グラフ入力や初期化子の場合はスキップ
                continue
            
            producer = tensor_producer[input_tensor]
            
            # 生成元ノードの出力インデックスを特定
            producer_output_idx = -1
            for j, output in enumerate(producer.output):
                if output == input_tensor:
                    producer_output_idx = j
                    break
            
            if producer_output_idx == -1:
                continue
            
            # 属性を取得する関数
            def get_attr(node: onnx.NodeProto, attr_name: str) -> Optional[str]:
                for attr in node.attribute:
                    if attr.name == attr_name:
                        if attr.HasField('s'):
                            return str(attr.s.decode('utf-8'))
                        elif attr.HasField('i'):
                            return str(attr.i)
                return None
            
            # tag/layoutの比較
            consumer_tag = get_attr(node, f'tag_in{i}')
            producer_tag = get_attr(producer, f'tag_out{producer_output_idx}')
            consumer_layout = get_attr(node, f'layout_in{i}')
            producer_layout = get_attr(producer, f'layout_out{producer_output_idx}')
            
            # 不整合チェック
            if consumer_tag and producer_tag and consumer_tag != producer_tag:
                msg = f"Tag不一致: {producer.name}[out{producer_output_idx}]='{producer_tag}' → {node.name}[in{i}]='{consumer_tag}'"
                inconsistencies.append(msg)
                print(f"  ❌ {msg}")
            
            if consumer_layout and producer_layout and consumer_layout != producer_layout:
                msg = f"Layout不一致: {producer.name}[out{producer_output_idx}]='{producer_layout[:30]}...' → {node.name}[in{i}]='{consumer_layout[:30]}...'"
                inconsistencies.append(msg)
                print(f"  ❌ {msg}")
    
    if inconsistencies:
        print("\n=== Tag/Layout整合性検証 ===")
        print(f"\n❌ {len(inconsistencies)}個の不整合が見つかりました")
        # 詳細なデバッグ情報を出力
        print("\n詳細デバッグ情報（最初の5個）:")
        for i, msg in enumerate(inconsistencies[:5]):
            print(f"  {i+1}. {msg}")
        print("=" * 40)


def add_dram_addresses(model: onnx.ModelProto) -> None:
    """
    MN-Core用ONNXモデルにDRAMアドレス属性を追加する
    統一フォーマット: tag_in0, layout_in0, location_in0, addr_in0, len_in0
    
    Args:
        model: DRAMアドレスを追加するONNXモデル
    """
    graph = model.graph
    
    # DRAM allocatorの状態
    current_dram_addr_lw = 0  # 現在のDRAMアドレス（lw単位、1lw=8バイト）
    
    # 割り当て済みテンソルとそのアドレスを追跡
    tensor_dram_info: Dict[str, Dict[str, int]] = {}  # 形式: {tensor_name: {"addr": addr_lw, "len": len_lw}}
    
    # DRAMを割り当てるヘルパー関数
    def allocate_dram(tensor_name: str, shape: List[int]) -> Dict[str, int]:
        """
        テンソル用のDRAMを割り当てる
        lw単位で"addr"と"len"を含む辞書を返す
        """
        nonlocal current_dram_addr_lw
        
        # 要素数を計算
        num_elements = calculate_tensor_len(shape)
        
        # バイト数を計算（float32想定）
        num_bytes = num_elements * 4
        
        # 使用するDRAM数を決定
        if num_elements <= 16:
            # 小さなテンソルはDRAM 1枚
            num_drams = 1
        else:
            # 大きなテンソルはDRAM 4枚
            num_drams = 4
        
        # lw単位の長さを計算（1lw=8バイト）
        # DRAM 4枚の場合は4で割る
        bytes_per_dram = num_bytes // num_drams
        len_lw = (bytes_per_dram + 7) // 8  # lw単位で切り上げ
        
        # アドレスを割り当て（64lwアライン）
        addr_lw = current_dram_addr_lw
        
        # 64lwアラインした終端アドレスを計算
        # align=64（64lwアライン）
        align=8000  # デバッグ用
        next_addr = addr_lw + len_lw
        end_lw = ((next_addr + align - 1) // align) * align
        
        # 次の割り当て用に現在のアドレスを更新
        current_dram_addr_lw = end_lw
        
        return {"addr": addr_lw, "len": len_lw}

    def append_attrs(node: onnx.NodeProto, attrs: Dict[str, Union[int, str]]) -> None:
        for attr_name, attr_value in attrs.items():
            node.attribute.append(helper.make_attribute(attr_name, attr_value))

    def get_attr_str(node: onnx.NodeProto, attr_name: str) -> Optional[str]:
        for attr in node.attribute:
            if attr.name == attr_name:
                return attr.s.decode('utf-8') if attr.HasField('s') else None
        return None

    def require_attr_str(node: onnx.NodeProto, attr_name: str) -> str:
        value = get_attr_str(node, attr_name)
        if value is None:
            raise ValueError(f"Node '{node.name}' is missing {attr_name}")
        return value

    def layout_info_for_lm(shape: List[int], tag: str) -> Tuple[int, str]:
        shape_with_str: List[Union[int, str]] = [int(s) for s in shape]
        layout_info = get_default_layout(shape_with_str, None if tag == 'DRAM' else tag)
        if layout_info is None:
            raise ValueError(f"LMレイアウトが見つかりません: shape={shape}, tag={tag}")
        return layout_info

    def layout_info_for_dram(shape: List[int]) -> Tuple[int, str]:
        shape_with_str: List[Union[int, str]] = [int(s) for s in shape]
        layout_info = get_default_layout(shape_with_str, 'DRAM')
        if layout_info is None:
            raise ValueError(f"DRAMレイアウトが見つかりません: shape={shape}")
        return layout_info
    
    # まずグラフ入力用のDRAMを割り当て
    for inp in graph.input:
        shape = get_shape_from_graph(graph, inp.name, {})
        dram_info = allocate_dram(inp.name, shape)
        tensor_dram_info[inp.name] = dram_info
        if 'DEBUG_DRAM_PLANNER' in os.environ:
            print(f"Allocated DRAM for input '{inp.name}': addr={dram_info['addr']} lw, len={dram_info['len']} lw")
    
    # 第1パス: computeとULノードを処理して属性を設定
    # DLノードがcomputeノードのアドレスを参照できるよう、これを最初に行う必要がある
    for node in graph.node:
        if node.op_type == 'UL':
            # ULノード: LM1から入力、DRAMへ出力
            if node.output:
                output_name = node.output[0]
                
                ul_shape = get_shape_from_graph(graph, output_name, {})

                inplace_input = get_attr_str(node, 'inplace_input')

                # inplace更新の場合、既存のDRAMアドレスを使用
                if inplace_input and inplace_input in tensor_dram_info:
                    dram_info = tensor_dram_info[inplace_input]
                    tensor_dram_info[output_name] = dram_info
                elif inplace_input:
                    raise ValueError(f"inplace_input '{inplace_input}' not found in tensor_dram_info for node {output_name}")
                else:
                    # 通常の新規割り当て
                    dram_info = allocate_dram(output_name, ul_shape)
                    tensor_dram_info[output_name] = dram_info

                # DL_planから既存の属性を取得
                tag = require_attr_str(node, 'tag_in0')
                layout_in0 = get_attr_str(node, 'layout_in0')

                # 入力属性を設定（LM1から）
                ul_input_attrs: Dict[str, Union[int, str]] = {
                    'location_in0': 'LM1',
                    'addr_in0': 0,  # 単一のLM入力では常に0
                    'tag_in0': tag
                }

                # 既存のlayout_in0があれば使用
                if layout_in0 is not None:
                    ul_input_attrs['layout_in0'] = layout_in0

                layout_info = layout_info_for_lm(ul_shape, tag)
                ul_input_attrs['len_in0'] = layout_info[0]
                if layout_in0 is None:
                    ul_input_attrs['layout_in0'] = layout_info[1]
                    
                # 出力属性を設定（DRAMへ）
                ul_output_attrs: Dict[str, Union[int, str]] = {
                    'location_out0': 'DRAM',
                    'addr_out0': dram_info['addr'],
                    'len_out0': dram_info['len'],
                    'tag_out0': 'DRAM'
                }

                # 出力用のDRAMレイアウトを取得
                dram_layout_info = layout_info_for_dram(ul_shape)
                ul_output_attrs['layout_out0'] = dram_layout_info[1]

                # すべての属性をノードに追加
                append_attrs(node, {**ul_input_attrs, **ul_output_attrs})

                if 'DEBUG_DRAM_PLANNER' in os.environ:
                    print(f"UL node '{node.name}': location_out0=DRAM, addr_out0={dram_info['addr']}, len_out0={dram_info['len']}")
        
        elif node.op_type not in ['DL', 'Identity']:  # DLとIdentityノードは今はスキップ
            # Computeノード: LM0から入力、LM1へ出力（upload=Falseの場合はDRAMへ）
            # ノードがupload=False属性を持つか確認
            has_upload_false = False
            for attr in node.attribute:
                if attr.name == 'upload' and attr.i == 0:
                    has_upload_false = True
                    break
            
            # ComputeノードはDL_planからtag_in0、layout_in0などを既に持っている
            # location_in*、addr_in*、len_in*を追加する必要がある
            shape_cache: Dict[str, List[int]] = {}
            input_lm_addr = 0  # LM入力の開始アドレス
            
            for i, input_name in enumerate(node.input):
                if input_name:
                    # 初期化子か確認
                    is_initializer = any(init.name == input_name for init in graph.initializer)
                    
                    if not is_initializer:
                        # LM0からの入力
                        node.attribute.append(helper.make_attribute(f'location_in{i}', 'LM0'))
                        node.attribute.append(helper.make_attribute(f'addr_in{i}', input_lm_addr))
                        
                        # 形状を取得してlenを計算
                        shape = get_shape_from_graph(graph, input_name, shape_cache)
                        
                        tag = require_attr_str(node, f'tag_in{i}')
                        layout_info = layout_info_for_lm(shape, tag)
                        node.attribute.append(helper.make_attribute(f'len_in{i}', layout_info[0]))
                        input_lm_addr += layout_info[0]  # LMアドレスを累積
            
            # どの出力がtag="DRAM"属性を持つか確認
            output_tags: List[Optional[str]] = []
            for attr in node.attribute:
                if attr.name.startswith('tag_out'):
                    # tag_out0、tag_out1などからインデックスを抽出
                    suffix = attr.name.replace('tag_out', '')
                    if not suffix.isdigit():
                        raise ValueError(f"Invalid tag_out attribute name '{attr.name}' in node '{node.name}'")
                    idx = int(suffix)
                    while len(output_tags) <= idx:
                        output_tags.append(None)
                    output_tags[idx] = attr.s.decode('utf-8')

            for i, output_name in enumerate(node.output):
                if output_name:
                    if i >= len(output_tags) or output_tags[i] is None:
                        raise ValueError(f"Compute node '{node.name}' is missing tag_out{i}")
            
            # ComputeノードがLM出力をサポートするのは1出力のみと断言
            if not has_upload_false and len(node.output) > 1:
                # すべての出力がDRAMに行くか確認
                all_dram = all(i < len(output_tags) and output_tags[i] == 'DRAM' for i in range(len(node.output)))
                if not all_dram:
                    assert len(node.output) == 1, f"Compute node '{node.name}' with LM output must have only 1 output, got {len(node.output)}"
            
            for i, output_name in enumerate(node.output):
                if output_name:
                    # 形状を取得
                    compute_shape: Optional[List[int]] = None
                    
                    # value_infoから形状を探す
                    for vi in graph.value_info:
                        if vi.name == output_name:
                            compute_shape = []
                            for dim in vi.type.tensor_type.shape.dim:
                                if dim.dim_value:
                                    compute_shape.append(dim.dim_value)
                            break
                    
                    # グラフ出力から形状を探す
                    if compute_shape is None:
                        for out in graph.output:
                            if out.name == output_name:
                                compute_shape = []
                                for dim in out.type.tensor_type.shape.dim:
                                    if dim.dim_value:
                                        compute_shape.append(dim.dim_value)
                                break

                    if compute_shape is None:
                        raise ValueError(f"Shape not found for compute output '{output_name}' in node '{node.name}'")
                    
                    # 出力がDRAMまたはLMに行くか確認
                    goes_to_dram = has_upload_false or (i < len(output_tags) and output_tags[i] == 'DRAM')
                    
                    if goes_to_dram:
                        # DRAMへの出力
                        dram_info = allocate_dram(output_name, compute_shape)
                        tensor_dram_info[output_name] = dram_info
                        
                        node.attribute.append(helper.make_attribute(f'location_out{i}', 'DRAM'))
                        node.attribute.append(helper.make_attribute(f'addr_out{i}', dram_info['addr']))
                        node.attribute.append(helper.make_attribute(f'len_out{i}', dram_info['len']))
                        
                        # DL_planからlayout_outが既に存在するか確認
                        has_layout_out = False
                        for attr in node.attribute:
                            if attr.name == f'layout_out{i}':
                                has_layout_out = True
                                break
                        
                        # layout_outが存在しない場合のみ追加
                        if not has_layout_out:
                            # 出力用のDRAMレイアウトを取得
                            dram_layout_info = layout_info_for_dram(compute_shape)
                            node.attribute.append(helper.make_attribute(f'layout_out{i}', dram_layout_info[1]))
                        
                        if 'DEBUG_DRAM_PLANNER' in os.environ:
                            print(f"Compute node '{node.name}' output {i} ({output_name}) to DRAM: addr={dram_info['addr']}, len={dram_info['len']}")
                    else:
                        # LM1への出力
                        node.attribute.append(helper.make_attribute(f'location_out{i}', 'LM1'))
                        node.attribute.append(helper.make_attribute(f'addr_out{i}', 0))  # LM出力では常に0
                        
                        tag = require_attr_str(node, f'tag_out{i}')
                        layout_info = layout_info_for_lm(compute_shape, tag)
                        node.attribute.append(helper.make_attribute(f'len_out{i}', layout_info[0]))
                        
                        # DL_planからlayout_outが既に存在するか確認
                        has_layout_out = False
                        for attr in node.attribute:
                            if attr.name == f'layout_out{i}':
                                has_layout_out = True
                                break
                        
                        # layout_outが存在しない場合のみ追加
                        if not has_layout_out:
                            node.attribute.append(helper.make_attribute(f'layout_out{i}', layout_info[1]))
    
    # アドレスマッチング用にDL出力からcomputeノード入力へのマップを構築
    dl_output_to_compute_input: Dict[str, Tuple[str, int]] = {}
    dl_outputs = {dl_node.output[0] for dl_node in graph.node
                  if dl_node.op_type == 'DL' and dl_node.output}
    for node in graph.node:
        if node.op_type not in ['DL', 'UL', 'Identity']:
            for i, inp in enumerate(node.input):
                if inp and inp in dl_outputs:
                    if inp in dl_output_to_compute_input:
                        raise ValueError(f"DL output '{inp}' is consumed by multiple compute nodes")
                    dl_output_to_compute_input[inp] = (node.name, i)
    
    # 第2パス: computeノードのaddr_in属性が設定されたので、DLノードを処理
    for node in graph.node:
        if node.op_type == 'DL':
            # DLノード: DRAMから入力、LM0へ出力
            if node.input and node.input[0] in tensor_dram_info:
                dram_info = tensor_dram_info[node.input[0]]
                
                # レイアウト情報のために形状を取得
                shape = get_shape_from_graph(graph, node.input[0], {})
                
                # このDLを入力するcomputeノードを見つけ、その入力タグとアドレスを取得
                dl_output_addr: Optional[int] = None
                dl_tag: Optional[str] = None
                compute_layout: Optional[str] = None  # Computeノードのlayout_in
                
                if node.output and node.output[0] in dl_output_to_compute_input:
                    compute_name, input_idx = dl_output_to_compute_input[node.output[0]]
                    # computeノードを見つけて属性を取得
                    for compute_node in graph.node:
                        if compute_node.name == compute_name:
                            for attr in compute_node.attribute:
                                if attr.name == f'addr_in{input_idx}':
                                    dl_output_addr = attr.i
                                elif attr.name == f'tag_in{input_idx}':
                                    # Computeノードのtag_inと一致させる
                                    dl_tag = attr.s.decode('utf-8') if attr.HasField('s') else None
                                elif attr.name == f'layout_in{input_idx}':
                                    # Computeノードのlayout_inも取得
                                    compute_layout = attr.s.decode('utf-8') if attr.HasField('s') else None
                            break
                else:
                    raise ValueError(f"DL node '{node.name}' output is not consumed by any compute node")

                if dl_output_addr is None:
                    raise ValueError(f"DL node '{node.name}' is missing addr_in for its compute consumer")
                if dl_tag is None:
                    raise ValueError(f"DL node '{node.name}' is missing tag_in for its compute consumer")
                
                # 入力属性を設定（DRAMから）
                # DRAMからの入力は常に'DRAM'タグ
                dl_input_attrs: Dict[str, Union[int, str]] = {
                    'location_in0': 'DRAM',
                    'addr_in0': dram_info['addr'],
                    'len_in0': dram_info['len'],
                    'tag_in0': 'DRAM'
                }
                
                # 入力用のDRAMレイアウトを取得
                dram_layout_info = layout_info_for_dram(shape)
                dl_input_attrs['layout_in0'] = dram_layout_info[1]
                
                # 出力属性を設定（LM0へ）
                dl_output_attrs: Dict[str, Union[int, str]] = {
                    'location_out0': 'LM0',
                    'addr_out0': dl_output_addr,
                    'tag_out0': dl_tag  # Computeノードのtag_inと一致
                }
                
                # LM出力用のレイアウトを取得
                # Computeノードのlayout_inがあれば、それを使用
                if compute_layout:
                    dl_output_attrs['layout_out0'] = compute_layout
                    # lenも計算し直す必要がある
                    layout_info = layout_info_for_lm(shape, dl_tag)
                    dl_output_attrs['len_out0'] = layout_info[0]
                else:
                    layout_info = layout_info_for_lm(shape, dl_tag)
                    dl_output_attrs['len_out0'] = layout_info[0]
                    dl_output_attrs['layout_out0'] = layout_info[1]
                
                # すべての属性をノードに追加
                append_attrs(node, {**dl_input_attrs, **dl_output_attrs})
                
                if 'DEBUG_DRAM_PLANNER' in os.environ:
                    print(f"DL node '{node.name}': location_in0=DRAM, addr_in0={dram_info['addr']}, len_in0={dram_info['len']}, addr_out0={dl_output_addr}")
    
    # グラフ出力を処理（既に割り当て済みのはずなので、既存のアドレスを参照するだけ）
    for output in graph.output:
        if output.name in tensor_dram_info:
            dram_info = tensor_dram_info[output.name]
            if 'DEBUG_DRAM_PLANNER' in os.environ:
                print(f"Graph output '{output.name}': dram_addr_lw={dram_info['addr']}, dram_len_lw={dram_info['len']}")
    
    if 'DEBUG_DRAM_PLANNER' in os.environ:
        print(f"\nTotal DRAM allocated: {current_dram_addr_lw} lw ({current_dram_addr_lw * 8} bytes)")


def transform_to_mncore(input_onnx_path: str, output_onnx_path: str) -> None:
    """
    通常のONNXモデルをMN-Core用に変換
    各計算ノードをDL_planメソッドを使ってDL/計算/ULノードに分解する
    
    Args:
        input_onnx_path: 入力ONNXファイルパス
        output_onnx_path: 出力ONNXファイルパス（mn_model.onnx）
    """
    # ONNXモデルを読み込み
    model = onnx.load(input_onnx_path)
    graph = model.graph
    
    # 新しいノードリスト
    new_nodes: List[onnx.NodeProto] = []
    
    # テンソル名のマッピング（元の名前 -> 現在の名前）
    # ULノードの出力をトラッキングするため
    tensor_mapping: Dict[str, str] = {}
    
    # 形状キャッシュ（ULノード出力の形状を記録）
    shape_cache: Dict[str, List[int]] = {}
    
    # グラフの入力と初期化子は最初からDRAMにある
    graph_inputs: Set[str] = {inp.name for inp in graph.input}
    initializers: Set[str] = {init.name for init in graph.initializer}
    
    # 初期状態：グラフ入力と初期化子はそのまま
    for inp in graph_inputs:
        tensor_mapping[inp] = inp
    for init in initializers:
        tensor_mapping[init] = init
    
    # 各ノードを処理
    for node_idx, node in enumerate(graph.node):
        # ノード名を確保（名前がない場合は生成）
        if not node.name:
            node.name = f"{node.op_type}_{node_idx}"
        
        # 未知のオペレーターの場合
        if node.op_type not in OPERATOR_REGISTRY:
            # DL/ULノードはそのまま追加（これらは変換で生成されたノード）
            if node.op_type in ['DL', 'UL']:
                new_nodes.append(node)
                # ULノードの出力をマッピングに追加
                if node.op_type == 'UL' and node.output:
                    for inp, out in zip(node.input, node.output):
                        tensor_mapping[inp] = out
            else:
                # その他の未知のオペレーターはエラー
                raise ValueError(f"Unknown operator '{node.op_type}' found in ONNX graph. "
                                f"Please implement the operator in fx_export/operators/.")
            continue
        
        # オペレータークラスを取得
        op_class = get_operator(node.op_type)
        
        # 現在のマッピングを反映した入力を作成
        mapped_node = onnx.NodeProto()
        mapped_node.CopyFrom(node)
        new_inputs = []
        for inp in node.input:
            if inp:
                # ULされた出力があればそれを使う
                mapped_inp = tensor_mapping.get(inp, inp)
                new_inputs.append(mapped_inp)
            else:
                new_inputs.append("")
        del mapped_node.input[:]
        mapped_node.input.extend(new_inputs)
        
        # MNCORE_BYPASSフラグを持つノードの特別処理（バイパス）
        if hasattr(op_class, 'MNCORE_BYPASS') and op_class.MNCORE_BYPASS:
            # バイパスするノードはノードを生成しない
            # 入力をそのまま出力にマッピング
            if node.input and node.output:
                # 入力（前のノードの出力またはUL出力）を
                # 出力名にマッピング
                for inp, out in zip(node.input, node.output):
                    # 現在のマッピングから実際の入力を取得
                    actual_input = tensor_mapping.get(inp, inp)
                    # 出力を入力に直接マッピング
                    tensor_mapping[out] = actual_input
                    # 形状も引き継ぐ
                    input_shape = get_shape_from_graph(graph, inp, shape_cache)
                    shape_cache[out] = input_shape
            continue  # 次のノードへ、DL_planを呼ばない
        
        # 変数マップ（ダミー）
        dummy_var_map: Dict[str, str] = {}
        
        # shape_cacheを参照できるように拡張クラスを作成
        class ExtendedOperator(op_class):  # type: ignore[misc,valid-type]
            def get_tensor_shape(self, tensor_name: str) -> List[Union[int, str]]:
                """拡張版：shape_cacheも参照"""
                try:
                    # まず通常の方法で探す
                    result = super().get_tensor_shape(tensor_name)
                    # 型アノテーションを明示的に追加
                    typed_result: List[Union[int, str]] = result
                    return typed_result
                except ValueError:
                    # shape_cacheから探す（UL出力など）
                    if tensor_name in shape_cache:
                        cached_shape: List[Union[int, str]] = [
                            int(dim) if isinstance(dim, int) else str(dim) 
                            for dim in shape_cache[tensor_name]
                        ]
                        return cached_shape
                    # それでも見つからない場合はエラー
                    raise
        
        # オペレーターインスタンスを作成
        operator = ExtendedOperator(mapped_node, graph, dummy_var_map)
        
        # DL_planを実行して、DL/計算/ULノードを取得
        dl_nodes, compute_node, ul_nodes = operator.DL_plan()
        
        # DLノードを追加（形状情報も記録）
        for dl_node in dl_nodes:
            new_nodes.append(dl_node)
            # DLノードの出力形状を記録
            if dl_node.output and dl_node.input:
                output_name = dl_node.output[0]
                input_name = dl_node.input[0]
                # 入力と同じ形状
                input_shape = get_shape_from_graph(graph, input_name, shape_cache)
                shape_cache[output_name] = input_shape
        
        # 計算ノードを追加（Noneでない場合のみ）
        if compute_node is not None:
            # グラフ出力名を確認
            graph_output_names = {out.name for out in graph.output}
            
            # 計算ノードの出力がグラフ出力の場合、名前を変更する必要がある
            needs_rename = False
            for orig_out in compute_node.output:
                if orig_out in graph_output_names:
                    needs_rename = True
                    break
            
            # 出力タグが"DRAM"の場合、または出力がグラフ出力の場合、計算ノードの出力名を変更
            has_dram_output = False
            for attr in compute_node.attribute:
                if attr.name.startswith('tag_out') and attr.s.decode('utf-8') == 'DRAM':
                    has_dram_output = True
                    break
            
            if has_dram_output or needs_rename:
                # 計算ノードの出力名を変更
                modified_outputs = []
                for orig_out in compute_node.output:
                    if orig_out:
                        # var_name属性から名前を取得
                        var_name = orig_out
                        for attr in compute_node.attribute:
                            if attr.name == 'var_name':
                                var_name = attr.s.decode('utf-8') if isinstance(attr.s, bytes) else str(attr.s)
                                break
                        
                        # 新しい出力名を生成
                        # output_locationが"DRAM"の場合は_UL_out、それ以外は_compute_outを使用
                        if operator.output_location == "DRAM":
                            new_out = f"{var_name}_UL_out"
                        else:
                            new_out = f"{var_name}_compute_out"
                        modified_outputs.append(new_out)
                        # マッピングを更新
                        tensor_mapping[orig_out] = new_out
                        # 形状も記録
                        output_shape = get_shape_from_graph(graph, orig_out, shape_cache)
                        shape_cache[new_out] = output_shape
                    else:
                        modified_outputs.append("")
                
                # 計算ノードの出力を変更
                del compute_node.output[:]
                compute_node.output.extend(modified_outputs)
            
            new_nodes.append(compute_node)
            
            # 計算ノードの出力を記録（ULノードがない場合のマッピング用）
            if not ul_nodes:
                for i, orig_out in enumerate(node.output):
                    # compute_nodeの実際の出力名を取得
                    if i < len(compute_node.output):
                        compute_out = compute_node.output[i]
                    else:
                        compute_out = orig_out
                    tensor_mapping[orig_out] = compute_out
        
        # ULノードを追加し、マッピングを更新
        for ul_node in ul_nodes:
            # 計算ノードの出力名が変更されている場合、ULノードの入力も更新
            if compute_node is not None:
                new_inputs = []
                for inp in ul_node.input:
                    # 元のノードの出力と照合
                    mapped_inp = inp
                    for orig_out in node.output:
                        if inp == orig_out and orig_out in tensor_mapping:
                            mapped_inp = tensor_mapping[orig_out]
                            break
                    new_inputs.append(mapped_inp)
                # ULノードの入力を更新
                del ul_node.input[:]
                ul_node.input.extend(new_inputs)
            
            new_nodes.append(ul_node)
            # ULノードの入出力をマッピングに記録
            # 元のテンソル名 -> UL後のテンソル名
            if ul_node.input and ul_node.output:
                for i, (inp, out) in enumerate(zip(ul_node.input, ul_node.output)):
                    # 対応する元の出力名を取得
                    if i < len(node.output):
                        orig_out = node.output[i]
                        # 元の出力名をUL後の名前にマッピング
                        tensor_mapping[orig_out] = out
                        # ULノードの出力形状を記録
                        # 変更された名前（inp）ではなく、元の出力名（orig_out）の形状を使用
                        input_shape = get_shape_from_graph(graph, orig_out, shape_cache)
                        shape_cache[out] = input_shape
                        # ULの入力（変更された名前）の形状も記録
                        shape_cache[inp] = input_shape
    
    # グラフ出力を更新
    # グラフ出力に対して、適切なエイリアスを作成
    new_outputs: List[onnx.ValueInfoProto] = []
    
    # 各グラフ出力について処理
    for output in graph.output:
        # 元の出力名に対応する実際のテンソル名を取得
        actual_tensor = tensor_mapping.get(output.name, output.name)
        
        # 名前が異なる場合のみ、Identityノードを追加
        # (ULノードの出力などを元の出力名にエイリアスする)
        if actual_tensor != output.name:
            # Identityノードを追加してエイリアスを作成
            identity_node = helper.make_node(
                'Identity',
                inputs=[actual_tensor],
                outputs=[output.name],
                name=f"{output.name}_alias"
            )
            # var_name属性を追加
            identity_node.attribute.append(
                helper.make_attribute('var_name', output.name)
            )
            new_nodes.append(identity_node)
            # マッピングを更新
            tensor_mapping[output.name] = output.name
        
        # 出力をそのまま使用
        new_outputs.append(output)
    
    # DL/ULノードの出力の形状情報を作成
    new_value_infos: List[onnx.ValueInfoProto] = list(graph.value_info)  # 元のvalue_infoをコピー
    
    # 追加したDL/UL/計算ノードの出力形状を記録
    for node in new_nodes:
        if node.op_type in ['DL', 'UL']:
            # ノードのshape属性から形状を取得
            if node.output:
                output_name = node.output[0]
                
                # shape_cacheから形状を取得
                if output_name in shape_cache:
                    shape = shape_cache[output_name]
                    
                    # DL/ULは型を変えないので、入力と同じ型を使う
                    # DLノードの場合は入力テンソルの型を取得
                    if node.op_type == 'DL' and node.input:
                        input_name = node.input[0]
                        elem_type = get_type_from_graph(graph, input_name)
                    # ULノードの場合も入力テンソルの型を取得
                    elif node.op_type == 'UL' and node.input:
                        input_name = node.input[0]
                        elem_type = get_type_from_graph(graph, input_name)
                    else:
                        # デフォルトはFLOAT
                        elem_type = onnx.TensorProto.FLOAT
                    
                    # value_infoを作成
                    output_value_info = helper.make_tensor_value_info(
                        output_name,
                        elem_type,
                        shape
                    )
                    new_value_infos.append(output_value_info)
        else:
            # 計算ノードの出力形状も記録（特に名前が変更された出力）
            for output_name in node.output:
                if output_name and output_name in shape_cache:
                    shape = shape_cache[output_name]
                    elem_type = onnx.TensorProto.FLOAT  # デフォルトはFLOAT
                    
                    # value_infoを作成（まだ存在しない場合のみ）
                    already_exists = any(vi.name == output_name for vi in new_value_infos)
                    if not already_exists:
                        output_value_info = helper.make_tensor_value_info(
                            output_name,
                            elem_type,
                            shape
                        )
                        new_value_infos.append(output_value_info)
    
    # 新しいグラフを作成
    # Initializerはそのまま保持（スカラー定数などはDLしない）
    new_graph = helper.make_graph(
        new_nodes,
        graph.name + "_mncore",
        graph.input,  # 元の入力のみ（Initializerは含まない）
        new_outputs,
        initializer=graph.initializer,  # Initializerはそのまま保持
        value_info=new_value_infos  # DL/ULの出力形状を含む
    )
    
    # 新しいモデルを作成
    new_model = helper.make_model(new_graph)
    
    # opset_importをコピー
    for opset in model.opset_import:
        new_opset = new_model.opset_import.add()
        new_opset.CopyFrom(opset)
    
    # DRAMアドレスを追加
    add_dram_addresses(new_model)
    
    # tag/layout整合性検証
    verify_tag_layout_consistency(new_model)
    
    # モデルを保存
    os.makedirs(os.path.dirname(output_onnx_path) or ".", exist_ok=True)
    onnx.save(new_model, output_onnx_path)
    print(f"MN-Coreモデルを保存しました: {output_onnx_path}")
    
    # デバッグ情報を出力
    print(f"元のノード数: {len(graph.node)}")
    print(f"新しいノード数: {len(new_nodes)}")
    dl_count = sum(1 for n in new_nodes if n.op_type == 'DL')
    ul_count = sum(1 for n in new_nodes if n.op_type == 'UL')
    compute_count = len(new_nodes) - dl_count - ul_count
    print(f"  DLノード: {dl_count}")
    print(f"  ULノード: {ul_count}")
    print(f"  計算ノード: {compute_count}")
    
    # layout属性のTODO警告を表示
    todo_layouts = []
    for node in new_nodes:
        for attr in node.attribute:
            # layout_in*, layout_out*パターンの属性を確認
            if ('layout_in' in attr.name or 'layout_out' in attr.name) and \
               attr.HasField('s') and attr.s.decode('utf-8') == 'TODO':
                todo_layouts.append((node.name, attr.name))
                break
    
    if todo_layouts:
        print(f"\n警告: {len(todo_layouts)}個のノードでlayout='TODO'です:")
        # 最初の5個のノード名を表示
        show_item_num = min(5, len(todo_layouts))
        for i, (node_name, attr_name) in enumerate(todo_layouts[:show_item_num]):
            print(f"  - {node_name} ({attr_name})")
        if len(todo_layouts) > show_item_num:
            print(f"  ... 他 {len(todo_layouts) - show_item_num} 個")


if __name__ == "__main__":
    # テスト用
    if len(sys.argv) != 3:
        print("Usage: python mncore_transform.py <input.onnx> <output.onnx>")
        sys.exit(1)
    
    transform_to_mncore(sys.argv[1], sys.argv[2])
