#!/usr/bin/env python3

from typing import List, Dict, Any, Optional
import onnx
from onnx import helper
from . import BaseOperator
from . import register_operator

@register_operator("Softmax")
class SoftmaxOperator(BaseOperator):
    """Softmax演算オペレーター"""
    
    
    def get_memory_layout_tag(self) -> Dict[str, List[str]]:
        """メモリレイアウトタグを定義"""
        return {
            "inputs": ["default"],
            "outputs": ["default"]
        }
    
    def generate_cpp(self) -> List[str]:
        """C++コード生成"""
        lines = []
        
        assert len(self.inputs) >= 1, f"Softmax node requires at least 1 input, got {len(self.inputs)}"
        
        in_var = self.get_mapped_var(self.inputs[0])
        out_var = self.get_output_var_name()
        
        # 形状を取得
        shape = self.in_shape(0)
        
        assert len(shape) == 2, f"Softmax expects 2D tensor, got {shape}"
        
        lines.append(f"    const Matrix<{shape[0]}, {shape[1]}> {out_var} = softmax<{shape[0]}, {shape[1]}>({in_var});")
        
        self.variable_map[self.outputs[0]] = out_var
        return lines
    
    def generate_python(self) -> List[str]:
        """Pythonコード生成"""
        lines = []
        
        inp = self.inputs[0]
        out = self.outputs[0]
        out_var = self.get_output_var_name()

        shape = self.in_shape(0)
        if len(shape) != 2:
            raise NotImplementedError(f"Softmax expects 2D tensor, got {shape}")

        axis = None
        for attr in self.node.attribute:
            if attr.name == 'axis':
                axis = attr.i
                break
        if axis is not None and axis not in (-1, 1):
            raise NotImplementedError(f"Softmax: axis {axis}")

        # 最後の次元に対してsoftmax
        lines.append(f"    {out_var} = F.softmax({self.get_mapped_var(inp)}, dim=-1)")
        self.variable_map[out] = out_var
        
        return lines
    
    def decompose(self) -> Optional[List[onnx.NodeProto]]:
        """
        Softmaxをより基本的な演算に分解する
        
        softmax(x) = exp(x - max(x)) / sum(exp(x - max(x)))
        
        Returns:
            分解後のノードのリスト
        """
        if not self.inputs or not self.outputs:
            return None
            
        input_name = self.inputs[0]
        output_name = self.outputs[0]
        
        # 中間変数名を生成
        max_out = f"{self.name}_max"
        sub_out = f"{self.name}_sub"  
        exp_out = f"{self.name}_exp"
        sum_out = f"{self.name}_sum"
        
        nodes = []
        
        # 1. max(x) : axis=-1 の ReduceMax (keepdims=False)
        nodes.append(helper.make_node(
            'ReduceMax',
            inputs=[input_name],
            outputs=[max_out],
            name=f"{self.name}_reducemax",
            axes=[-1],
            keepdims=0
        ))
        
        # 2. x - max(x) : Sub（減算）
        nodes.append(helper.make_node(
            'Sub',
            inputs=[input_name, max_out],
            outputs=[sub_out],
            name=f"{self.name}_sub"
        ))
        
        # 3. exp(x - max(x)) : Exp（指数）
        nodes.append(helper.make_node(
            'Exp',
            inputs=[sub_out],
            outputs=[exp_out],
            name=f"{self.name}_exp"
        ))
        
        # 4. sum(exp(x - max(x))) : axis=-1 の ReduceSum (keepdims=False)
        nodes.append(helper.make_node(
            'ReduceSum',
            inputs=[exp_out],
            outputs=[sum_out],
            name=f"{self.name}_reducesum",
            axes=[-1],
            keepdims=0
        ))
        
        # 5. exp(x - max(x)) / sum(exp(x - max(x))) : Div（除算）
        nodes.append(helper.make_node(
            'Div',
            inputs=[exp_out, sum_out],
            outputs=[output_name],
            name=f"{self.name}_div"
        ))
        
        return nodes
