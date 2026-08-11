#!/usr/bin/env python3

from typing import List, Dict, Any, Optional
from . import BaseOperator
from . import register_operator

@register_operator("Relu")
class ReluOperator(BaseOperator):
    """ReLU活性化関数オペレーター"""
    
    def generate_vsm(self) -> List[str]:
        # グラフから形状を取得
        in_shape = self.in_shape()
        assert in_shape == self.out_shape()
        
        in_prefix = self.loc_prefix_in()    # $lm0v の "m" の部分
        out_prefix = self.loc_prefix_out()  # $ln0v の "n" の部分
        
        in_offset = self.addr_in()          # $lm0v の "0" の部分
        out_offset = self.addr_out()        # $ln0v の "0" の部分
        
        # test unit_tests/train_step/Relu_*
        lines = []
        for i in range(self.memory_len_in_div_ceil(8, 0)):  # PE あたりの長さを、8単語で割って切り上げ
            in_addr  = in_offset  + i * 8
            out_addr = out_offset + i * 8
            lines.append(f"frelu $l{in_prefix}{in_addr}v $l{in_prefix}{in_addr}v $l{out_prefix}{out_addr}v")
        return lines

    def testcase_hint(self) -> Optional[str]:
        return "relu.vsm"

    def get_memory_layout_tag(self) -> Dict[str, List[str]]:
        return {
            "inputs": ["default"],
            "outputs": ["default"]
        }


    def generate_cpp(self) -> List[str]:
        """C++コード生成"""
        lines = []
        
        assert len(self.inputs) >= 1, f"Relu node requires at least 1 input, got {len(self.inputs)}"
        
        in_var = self.get_mapped_var(self.inputs[0])
        out_var = self.get_output_var_name()
        
        shape = self.in_shape(0)
        
        if len(shape) == 2:
            lines.append(f"    const Matrix<{shape[0]}, {shape[1]}> {out_var} = relu<{shape[0]}, {shape[1]}>({in_var});")
        else:
            raise NotImplementedError(f"Relu: shape {shape}")
        
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
            raise NotImplementedError(f"Relu: shape {shape}")

        lines.append(f"    {out_var} = F.relu({self.get_mapped_var(inp)})")
        self.variable_map[out] = out_var
        
        return lines
