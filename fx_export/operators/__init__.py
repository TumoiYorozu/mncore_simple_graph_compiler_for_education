#!/usr/bin/env python3

from typing import Dict, Type, List, Any, Optional, Tuple, Union
import numpy as np
import onnx
from onnx import helper

# =============== BaseOperatorクラス ===============

class BaseOperator:
    """全オペレーターの基底クラス"""
    
    def __init__(self, node: onnx.NodeProto, graph: onnx.GraphProto, variable_map: Dict[str, str]) -> None:
        self.node: onnx.NodeProto = node
        self.graph: onnx.GraphProto = graph
        self.variable_map: Dict[str, str] = variable_map
        self.inputs: List[str] = list(node.input)
        self.outputs: List[str] = list(node.output)
        self.name: str = node.name
        
    def generate_cpp(self) -> List[str]:
        """C++コード生成（行のリスト）"""
        raise NotImplementedError(f"{self.__class__.__name__} must implement generate_cpp")
        
    def generate_python(self) -> List[str]:
        """Pythonコード生成（行のリスト）"""
        raise NotImplementedError(f"{self.__class__.__name__} must implement generate_python")
        
    def generate_asm(self) -> List[str]:
        """アセンブリコード生成（将来用）"""
        raise NotImplementedError(f"{self.__class__.__name__} must implement generate_asm")
    
    def testcase_hint(self) -> Optional[str]:
        """VSMテストケースファイル名のヒントを返す（ワイルドカード可）
        
        Returns:
            テストケースファイル名パターン（例: "add_colvec.vsm", "matmul_tra_*.vsm"）
            該当するテストケースがない場合はNone
        """
        return None
    
    def generate_vsm(self) -> List[str]:
        raise NotImplementedError(f"{self.__class__.__name__} must implement generate_vsm")
    
    def _get_attribute_int(self, prefix: str, index: int) -> int:
        """ノード属性から整数値を取得（見つからない場合はエラー）"""
        attr_name = f'{prefix}{index}'
        for attr in self.node.attribute:
            if attr.name == attr_name and attr.HasField('i'):
                return int(attr.i)  # 明示的にintにキャスト
        raise ValueError(f"Cannot find {attr_name} in node {self.name}")
    
    def _get_attribute_str(self, prefix: str, index: int) -> str:
        """ノード属性から文字列値を取得（見つからない場合はエラー）"""
        attr_name = f'{prefix}{index}'
        for attr in self.node.attribute:
            if attr.name == attr_name and attr.HasField('s'):
                return str(attr.s.decode('utf-8'))  # 明示的にstrにキャスト
        raise ValueError(f"Cannot find {attr_name} in node {self.name}")
    
    def memory_len_in(self, index: int = 0) -> int:
        """入力のメモリ長（vsm 基準）を取得"""
        return self._get_attribute_int('len_in', index)
    
    def memory_len_out(self, index: int = 0) -> int:
        """出力のメモリ長（vsm 基準）を取得"""
        return self._get_attribute_int('len_out', index)
    
    def addr_in(self, index: int = 0) -> int:
        """入力のアドレスを取得"""
        return self._get_attribute_int('addr_in', index)
    
    def addr_out(self, index: int = 0) -> int:
        """出力のアドレスを取得"""
        return self._get_attribute_int('addr_out', index)
    
    def layout_in(self, index: int = 0) -> str:
        """入力のレイアウトを取得"""
        return self._get_attribute_str('layout_in', index)
    
    def layout_out(self, index: int = 0) -> str:
        """出力のレイアウトを取得"""
        return self._get_attribute_str('layout_out', index)
    
    def tag_in(self, index: int = 0) -> str:
        """入力のタグを取得"""
        return self._get_attribute_str('tag_in', index)
    
    def tag_out(self, index: int = 0) -> str:
        """出力のタグを取得"""
        return self._get_attribute_str('tag_out', index)
    
    
    def memory_len_in_div_ceil(self, align: int, index: int = 0) -> int:
        return (self.memory_len_in(index) + align - 1) // align
    
    def memory_len_out_div_ceil(self, align: int, index: int = 0) -> int:
        return (self.memory_len_out(index) + align - 1) // align
    
    def loc_prefix_in(self, index: int = 0) -> str:
        """入力のロケーションプレフィックスを取得（LM0->m, LM1->n, DRAM->d）"""
        location = self._get_attribute_str('location_in', index)
        if location == 'LM0':
            return 'm'
        elif location == 'LM1':
            return 'n'
        elif location == 'DRAM':
            return 'd'
        else:
            raise ValueError(f"Unknown location {location} for input {index} in node {self.name}")
    
    def loc_prefix_out(self, index: int = 0) -> str:
        """出力のロケーションプレフィックスを取得（LM0->m, LM1->n, DRAM->d）"""
        location = self._get_attribute_str('location_out', index)
        if location == 'LM0':
            return 'm'
        elif location == 'LM1':
            return 'n'
        elif location == 'DRAM':
            return 'd'
        else:
            raise ValueError(f"Unknown location {location} for output {index} in node {self.name}")
    
    def check_layout_in(self, expect: str, index: int = 0) -> None:
        actual = self.layout_in(index)
        assert actual == expect, f"{index=}  expect: '{expect}', but got '{actual}'"

    def check_layout_out(self, expect: str, index: int = 0) -> None:
        actual = self.layout_out(index)
        assert actual == expect, f"{index=}  expect: '{expect}', but got '{actual}'"

    
    def decompose(self) -> Optional[List[onnx.NodeProto]]:
        """
        このオペレーターをより基本的な演算に分解する
        
        Returns:
            分解後のノードのリスト、分解しない場合はNone
            
        Example (Softmax):
            exp(x - max(x)) / sum(exp(x - max(x)))
        """
        return None
    
    def DL_plan(self) -> Tuple[List[onnx.NodeProto], onnx.NodeProto, List[onnx.NodeProto]]:
        """
        MN-Core用にDL/計算/ULノードを生成する
        
        Returns:
            Tuple[DLノードのリスト, 計算ノード（入力を変更済み）, ULノードのリスト]
            - 通常のオペレーター: 各入力に対してDLノード、出力に対してULノード
            - matmul/reduce_sum系: DLノードのみ（ULは計算に含まれる）
            - Initializerへの参照はDLノードを生成しない
        """
        dl_nodes = []
        ul_nodes = []
        
        # メモリレイアウトタグを取得
        layout_tags = self.get_memory_layout_tag()
        if layout_tags is None:
            raise NotImplementedError(f"{self.__class__.__name__}.get_memory_layout_tag() returned None")
        if not isinstance(layout_tags, dict):
            raise NotImplementedError(f"{self.__class__.__name__}.get_memory_layout_tag() must return dict or None, got {type(layout_tags)}")
        
        input_tags = layout_tags.get('inputs', [])
        output_tags = layout_tags.get('outputs', [])
        
        # output_location を設定（最初の出力のタグから決定）
        # outputs[0] が "DRAM" なら出力はDRAMに残る（ULが内部に含まれる）
        # それ以外ならLocal Memoryに残る（ULノードが必要）
        self.output_location = "LM"  # デフォルト
        if output_tags and len(output_tags) > 0 and output_tags[0] == "DRAM":
            self.output_location = "DRAM"
        
        # Initializerのリストを取得
        initializer_names = set()
        if hasattr(self, 'graph') and self.graph:
            initializer_names = {init.name for init in self.graph.initializer}
        
        # 計算ノードのコピーを作成（入力を変更するため）
        compute_node = onnx.NodeProto()
        compute_node.CopyFrom(self.node)
        new_inputs = []
        
        # 各入力に対してDLノードを生成（Initializer以外）
        for i, input_name in enumerate(self.inputs):
            if input_name:  # 空でない入力のみ処理
                # Initializerの場合はDLノードを生成せず、直接使用
                if input_name in initializer_names:
                    new_inputs.append(input_name)
                else:
                    # DLノードを生成（タグ情報を渡す）
                    dl_output = f"{self.name}_DL_{i}_out"
                    tag = input_tags[i] if i < len(input_tags) else "default"
                    
                    # Noneチェック
                    if tag is None:
                        raise ValueError(f"Invalid tag None for input {i} in operator {self.name}. Use 'default' instead.")
                    
                    # tagは常に文字列として使用
                    dl_node = helper.make_node(
                        'DL',
                        inputs=[input_name],
                        outputs=[dl_output],
                        name=f"{self.name}_DL_{i}"
                    )
                    
                
                    dl_nodes.append(dl_node)
                    new_inputs.append(dl_output)
            else:
                new_inputs.append("")
        
        # 計算ノードの入力を更新
        del compute_node.input[:]
        compute_node.input.extend(new_inputs)
        
        # 計算ノードに個別の入力レイアウト・タグ属性を追加
        for i, input_name in enumerate(self.inputs):
            if input_name:  # 空でない入力のみ処理
                # タグ
                tag = input_tags[i] if i < len(input_tags) else "default"
                if tag is None:
                    raise ValueError(f"Invalid tag None for input {i} in operator {self.name}. Use 'default' instead.")
                
                # tag_in0, tag_in1, ... 形式で追加
                attr = compute_node.attribute.add()
                attr.name = f'tag_in{i}'
                attr.type = onnx.AttributeProto.STRING
                attr.s = tag.encode('utf-8')
                
                # レイアウト
                if input_name not in initializer_names:  # 初期化子以外
                    input_shape = self.get_tensor_shape(input_name)
                    # タグに応じたvariantを指定
                    layout_info = get_default_layout(input_shape, tag)
                    if layout_info is not None:
                        _, layout_string = layout_info
                        attr = compute_node.attribute.add()
                        attr.name = f'layout_in{i}'
                        attr.type = onnx.AttributeProto.STRING
                        attr.s = layout_string.encode('utf-8')
    
        # output_location が "DRAM" の場合、upload属性を追加
        if self.output_location == "DRAM":
            attr = compute_node.attribute.add()
            attr.name = 'upload'
            attr.type = onnx.AttributeProto.INT
            attr.i = 0  # False を意味する
        
        # 計算ノードに個別の出力レイアウト・タグ属性を追加
        for j, output_name in enumerate(self.outputs):
            if output_name:
                # タグ
                tag_or_none = output_tags[j] if j < len(output_tags) else None
                if tag_or_none is None:
                    raise ValueError(f"Invalid tag None for output {j} in operator {self.name}. Use 'default' instead.")
                output_tag: str = tag_or_none
                
                attr = compute_node.attribute.add()
                attr.name = f'tag_out{j}'
                attr.type = onnx.AttributeProto.STRING
                attr.s = output_tag.encode('utf-8')
                
                # レイアウト
                output_shape = self.get_tensor_shape(output_name)
                layout_info = get_default_layout(output_shape, output_tag)
                if layout_info is not None:
                    _, layout_string = layout_info
                    attr = compute_node.attribute.add()
                    attr.name = f'layout_out{j}'
                    attr.type = onnx.AttributeProto.STRING
                    attr.s = layout_string.encode('utf-8')
        
        # output_location が "LM" の場合のみULノードを生成
        if self.output_location == "LM":
            for j, output_name in enumerate(self.outputs):
                if output_name:
                    # ULノードの出力名を生成
                    var_name = self.get_output_var_name()
                    ul_output_name = f"{var_name}_UL_out"
                    ul_node_name = f"{var_name}_UL"
                    
                    # タグ情報を取得
                    tag = output_tags[j] if j < len(output_tags) else "default"
                    
                    # Noneチェック
                    if tag is None:
                        raise ValueError(f"Invalid tag None for output {j} in operator {self.name}. Use 'default' instead.")
                    
                    # tagは常に文字列として使用
                    ul_node = helper.make_node(
                        'UL',
                        inputs=[output_name],
                        outputs=[ul_output_name],
                        name=ul_node_name
                    )
                    
                    # 元のノードのinplace_input属性を伝播
                    for orig_attr in self.node.attribute:
                        if orig_attr.name == 'inplace_input':
                            attr = ul_node.attribute.add()
                            attr.name = 'inplace_input'
                            attr.type = onnx.AttributeProto.STRING
                            attr.s = orig_attr.s
                            break
                    
                    # ULノードにtag_in0とlayout_in0属性を追加（Computeノードのtag_out/layout_outと一致）
                    attr = ul_node.attribute.add()
                    attr.name = 'tag_in0'
                    attr.type = onnx.AttributeProto.STRING
                    attr.s = tag.encode('utf-8')
                    
                    # レイアウトも追加
                    output_shape = self.get_tensor_shape(output_name)
                    layout_info = get_default_layout(output_shape, tag)
                    if layout_info is not None:
                        _, layout_string = layout_info
                        attr = ul_node.attribute.add()
                        attr.name = 'layout_in0'
                        attr.type = onnx.AttributeProto.STRING
                        attr.s = layout_string.encode('utf-8')
                    
                    ul_nodes.append(ul_node)
        
        return dl_nodes, compute_node, ul_nodes
    
    def handle_internal_ul_python(self, lines: List[str], out_var: str) -> None:
        """
        ULを内部に含むオペレーターのPythonコード生成を処理する共通メソッド
        
        Args:
            lines: 生成されたコード行のリスト（変更される）
            out_var: 計算結果の変数名（例: "grad_fc3_input"）
        
        このメソッドは以下を行います：
        1. output_locationが"DRAM"の場合
        2. self.outputs[0]が"_UL_out"で終わる場合
        3. UL操作を示す行を追加し、variable_mapを適切に設定
        """
        if hasattr(self, 'output_location') and self.output_location == "DRAM" and self.outputs[0].endswith('_UL_out'):
            # 結果がDRAMに送られることを示すためUL行を追加
            lines.append(f"    {self.outputs[0]} = {out_var}  # UL: Local Memory -> DRAM")
            # DLノード参照用に_UL_out名をそのままマップ
            self.variable_map[self.outputs[0]] = self.outputs[0]
        else:
            # 通常ケース: 出力を計算結果の変数にマップ
            self.variable_map[self.outputs[0]] = out_var
    
    def get_memory_layout_tag(self) -> Optional[Dict[str, List[str]]]:
        """
        このオペレーションの入出力レイアウトタグを返す
        
        サブクラスでオーバーライドして、各入出力のタグを返す
        例:
            return {
                "inputs": ["default", "PE"],  # input0はdefault、input1はPE
                "outputs": ["DRAM"]           # output0はDRAM（"DRAM"ならUL含む）
            }
        
        Returns:
            dictまたはNone。Noneは実装エラーとして扱われる。
            dictのリスト要素は文字列タグ（"default", "PE", "L2B", "DRAM"など）。
        """
        return None  # デフォルトでNoneを返す（実装エラー）

    # =============== 共通ヘルパーメソッド ===============
    
    def get_attr(self, name: str) -> Any:
        """
        ノードの属性を名前で取得する
        
        Args:
            name: 属性名
        
        Returns:
            属性の値（onnx.AttributeProto）
        
        Raises:
            AttributeError: 指定された属性が存在しない場合
        
        Example:
            axes = list(self.get_attr("axes").ints)
            keepdims = self.get_attr("keepdims").i
        """
        for attr in self.node.attribute:
            if attr.name == name:
                return attr
        return None
    
    def get_output_var_name(self) -> str:
        """処理関数の出力変数名を決定する共通ロジック"""
        # var_name属性があればそれを使用
        for attr in self.node.attribute:
            if attr.name == 'var_name':
                var_name: str = attr.s.decode('utf-8') if isinstance(attr.s, bytes) else str(attr.s)
                return var_name
        
        # なければデフォルトでノード名を使用
        return self.node.name
    
    def get_tensor_shape(self, tensor_name: str) -> List[Union[int, str]]:
        """ONNXグラフからテンソルの形状情報を取得"""
        # まず、入力から形状を探す
        for input_tensor in self.graph.input:
            if input_tensor.name == tensor_name:
                shape: List[Union[int, str]] = []
                for dim in input_tensor.type.tensor_type.shape.dim:
                    if dim.dim_value:
                        shape.append(dim.dim_value)
                    elif dim.dim_param:
                        # シンボリックディメンションの場合
                        shape.append(dim.dim_param)
                return shape
        
        # 次に、値情報（中間結果）から形状を探す
        for value_info in self.graph.value_info:
            if value_info.name == tensor_name:
                shape = []
                for dim in value_info.type.tensor_type.shape.dim:
                    if dim.dim_value:
                        shape.append(dim.dim_value)
                    elif dim.dim_param:
                        shape.append(dim.dim_param)
                return shape
        
        # 出力から形状を探す
        for output_tensor in self.graph.output:
            if output_tensor.name == tensor_name:
                shape = []
                for dim in output_tensor.type.tensor_type.shape.dim:
                    if dim.dim_value:
                        shape.append(dim.dim_value)
                    elif dim.dim_param:
                        shape.append(dim.dim_param)
                return shape
        
        # イニシャライザー（定数）から形状を探す
        for initializer in self.graph.initializer:
            if initializer.name == tensor_name:
                return list(initializer.dims)
        
        # 形状が見つからない場合はエラー
        raise ValueError(f"Shape not found for tensor '{tensor_name}' in node '{self.name}'."
                        f" Node type: {self.node.op_type}, inputs: {list(self.node.input)}, outputs: {list(self.node.output)}")
    
    def in_shape(self, index: int = 0) -> List[Union[int, str]]:
        """指定インデックスの入力テンソルの形状を取得
        
        Args:
            index: 入力のインデックス（0-based）
        
        Returns:
            形状のリスト。エラーの場合は空リスト
        
        Example:
            shape = self.in_shape(0)  # 第1入力の形状を取得
            shape = self.in_shape(1)  # 第2入力の形状を取得
        """
        if index < 0 or index >= len(self.inputs):
            raise IndexError(f"Input index {index} out of range for node '{self.name}'")
        return self.get_tensor_shape(self.inputs[index])
    
    def out_shape(self, index: int = 0) -> List[Union[int, str]]:
        """指定インデックスの出力テンソルの形状を取得
        
        Args:
            index: 出力のインデックス（0-based）
        
        Returns:
            形状のリスト。エラーの場合は空リスト
        
        Example:
            shape = self.out_shape(0)  # 第1出力の形状を取得
        """
        if index < 0 or index >= len(self.outputs):
            raise IndexError(f"Output index {index} out of range for node '{self.name}'")
        return self.get_tensor_shape(self.outputs[index])
    
    def get_mapped_var(self, var_name: str) -> str:
        """変数マップから変数名を取得（マップになければそのまま返す）"""
        return self.variable_map.get(var_name, var_name)
    
    def find_shape(self, initializer_name: str) -> Tuple[Optional[int], Optional[int]]:
        """初期化子の形状を取得（Matrix向け）"""
        for init in self.graph.initializer:
            if init.name == initializer_name:
                if len(init.dims) == 2:
                    return int(init.dims[0]), int(init.dims[1])
                elif len(init.dims) == 1:
                    return int(init.dims[0]), None
                else:
                    raise ValueError(f"Unsupported shape dimensions {len(init.dims)} for initializer '{initializer_name}'")
        raise ValueError(f"Initializer '{initializer_name}' not found in graph")

# =============== デフォルトレイアウト関数 ===============

def get_default_layout(shape: Union[Tuple[int, ...], List[Union[int, str]]], tag: Optional[str]) -> Optional[Tuple[int, str]]:
    """
    与えられた形状に対してデフォルトのメモリレイアウトを返す
    
    Args:
        shape: テンソルの形状（タプルまたはリスト）
        variant: レイアウトのバリアント指定（同じ形状で異なるレイアウトが必要な場合）
    
    Returns:
        (len, layout_string) のタプル
    """
    # リストの場合はタプルに変換
    if isinstance(shape, list):
        int_shape = []
        for s in shape:
            if isinstance(s, int):
                int_shape.append(s)
            elif isinstance(s, str) and s.isdigit():
                int_shape.append(int(s))
            else:
                raise ValueError(f"Non-concrete dimension '{s}' found; layout requires static shapes.")
        shape = tuple(int_shape)
    assert tag is not None
    if tag == "default":
        tag = None
    
    if tag == "DRAM":
        if len(shape) == 0:
            dram_len = 1
            return (dram_len, f"(({dram_len}:1, 2_W:1))")
        if len(shape) == 1:
            if shape[0] <= 16:
                dram_len = (shape[0]+1)//2
                return (dram_len, f"(({dram_len}:1, 2_W:1))")
            else:
                dram_len = (shape[0]+7)//8
                return (dram_len, f"((4_L2B:2, {dram_len}:1, 2_W:1))")
        elif len(shape) == 2:
            if shape[1] < 4:
                raise ValueError(f"DRAM layout では shape[1] >= 4 が必要です: {shape}")
            len1 = (shape[1]+1)//2
            len0 = (shape[0]+3)//4
            return (len0 * len1, f"((4_L2B:2, {len0}:{len1}), ({len1}:1, 2_W:1))")
        else:
            raise ValueError(f"Unsupported shape for DRAM layout: {shape}")

    # (shape, variant)をキーとしたレイアウトの辞書
    layout_dict: Dict[Tuple[Tuple[int, ...], Optional[str]], Tuple[int, str]] = {
        ((16,), None):      (4,  "((2:1, 4_PE:1, 2_W:1))"),
        ((16, 16), None):   (64, "((16:2), (2:1, 4_PE:1, 2_W:1))"),
        ((16, 1024), None): (32, "((16:1), (8_MAB:2, 8_L1B:1, 2_MAB:1, 4_PE:1, 2_W:1))"),
        ((1024, 16), None): (64, "((8_L2B:1, 16:2, 8_L1B:1), (2:1, 4_PE:1, 2_W:1))"),
        ((256, 16), None):  (16, "((8_L2B:1, 4:2, 8_L1B:1), (2:1, 4_PE:1, 2_W:1))"),
        ((256, 1024), None): (64, "((8_L2B:1, 32:1), (8_MAB:2, 8_L1B:1, 2_MAB:1, 4_PE:1, 2_W:1))"),
        
        ((256,16),"PE"):    (256, "((4_L2B:2, 64:2), (2:1, 4_PE:1, 2_W:1))"),
        ((256,16),"L2B"):   (128, "((8_L2B:1, 32:2), (2:1, 4_PE:1, 2_W:1))"),
        ((256,), None):     (64, "((4_L2B:2, 32:1, 2_W:1))"),
        
    }
    # 辞書にあればそれを返す、なければエラー
    key = (shape, tag)
    if key in layout_dict:
        return layout_dict[key]
    raise NotImplementedError



# =============== オペレーターレジストリ ===============

OPERATOR_REGISTRY: Dict[str, Type[BaseOperator]] = {}

def register_operator(op_type: str) -> Any:
    """オペレータークラスを登録するデコレーター"""
    def decorator(cls: Type[BaseOperator]) -> Type[BaseOperator]:
        OPERATOR_REGISTRY[op_type] = cls
        return cls
    return decorator

def get_operator(op_type: str) -> Type[BaseOperator]:
    """オペレータータイプから対応するクラスを取得"""
    if op_type not in OPERATOR_REGISTRY:
        raise ValueError(f"Unknown operator type: {op_type}")
    return OPERATOR_REGISTRY[op_type]



# =============== オペレータークラスを自動インポート ===============
# 注: これらのインポートはBaseOperatorとregister_operatorが定義された後に行う必要がある

from .add import AddOperator
from .gemm import GemmOperator
from .relu import ReluOperator
from .softmax import SoftmaxOperator
from .log import LogOperator
from .sub import SubOperator
from .mul import MulOperator
from .reduce_sum import ReduceSumOperator
from .onehot import OneHotOperator
from .greater import GreaterOperator
from .nll_loss import NLLLossOperator
from .identity import IdentityOperator
from .cast import CastOperator
from .reduce_max import ReduceMaxOperator
from .exp import ExpOperator
from .div import DivOperator
from .dl import DLOperator
from .ul import ULOperator
