#!/usr/bin/env python3

from typing import List, Dict, Any, Optional
import numpy as np
from onnx import numpy_helper
from . import BaseOperator, register_operator

@register_operator("Mul")
class MulOperator(BaseOperator):
    """乗算演算オペレーター"""

    def generate_vsm(self) -> List[str]:
        
        shape0 = self.in_shape(0)
        shape1 = self.in_shape(1)
        
        in0_prefix = self.loc_prefix_in(0)  # $lm0v  の "m" の部分
        out_prefix = self.loc_prefix_out()  # $ln0v  の "n" の部分
        
        in0_offset = self.addr_in(0)        # $lm0v  の "0" の部分
        out_offset = self.addr_out()        # $ln0v  の "0" の部分
        
        lines = []
        if shape1 == []:
            assert shape0 == self.out_shape()
            
            # 定数倍の場合
            # 乗算する値を取得
            for init in self.graph.initializer:
                if init.name == self.inputs[1]:
                    tensor = numpy_helper.to_array(init)
                    scalar_val = float(tensor.flat[0])
                    scalar_value = f"{scalar_val}"
                    break
            else:
                raise NotImplementedError(f"Initializer for {self.inputs[1]} not found in graph initializers")
            
            # 乗算メインパート
            for i in range(self.memory_len_in_div_ceil(8, 0)):
                lines.append(f'imm f"{scalar_value}" $t')
                lines.append(f'nop')
                lines.append(f"fvmul $l{in0_prefix}{in0_offset + i * 8}v $lt $l{out_prefix}{out_offset + i * 8}v")
            return lines

        in1_prefix = self.loc_prefix_in(1)
        in1_offset = self.addr_in(1)
        if shape0 == shape1:
            # 同じ形状の場合
            for i in range(self.memory_len_in_div_ceil(8, 0)):
                lines.append(f"ipassa $l{in1_prefix}{in1_offset + i * 8}v $nowrite")
                lines.append(f"fvmul  $l{in0_prefix}{in0_offset + i * 8}v $aluf $l{out_prefix}{out_offset + i * 8}v")
            return lines

        raise NotImplementedError
    

    
    def testcase_hint(self) -> Optional[str]:
        # return "mul_const.vsm"  # これは *7 固定のテストケースなので、使用できない
        return None

    def get_memory_layout_tag(self) -> Dict[str, List[str]]:
        shape0 = self.in_shape(0)
        shape1 = self.in_shape(1)
        if shape0 == shape1:
            # 同じシェイプで、要素ごとの乗算
            return {
                "inputs": ["default", "default"],
                "outputs": ["default"]
            }
        elif shape1 == []:
            # スカラー乗算
            return {
                "inputs": ["default"],
                "outputs": ["default"]
            }
        raise NotImplementedError(f"Mul: shapes {shape0} * {shape1}")

    def generate_cpp(self) -> List[str]:
        """C++コード生成"""
        lines = []
        
        assert len(self.inputs) >= 2, f"Mul node requires at least 2 inputs, got {len(self.inputs)}"
        
        in1_var = self.get_mapped_var(self.inputs[0])
        in2_var = self.get_mapped_var(self.inputs[1])
        out_var = self.get_output_var_name()
        
        shape1 = self.in_shape(0)
        shape2 = self.in_shape(1)
        
        # 2番目の入力が初期化子（定数）かチェック
        is_initializer = self.inputs[1] in [init.name for init in self.graph.initializer]
        
        # 2番目の入力がスカラーかチェック（形状が[]または[1]、または初期化子の場合）
        is_scalar_multiply = (len(shape2) == 0)
        
        if is_scalar_multiply:
            # 定数倍の場合
            if is_initializer:
                # 初期化子の場合は、その値を直接使用
                # 初期化子の値を取得
                for init in self.graph.initializer:
                    if init.name == self.inputs[1]:
                        tensor = numpy_helper.to_array(init)
                        scalar_val = float(tensor.flat[0])  # 最初の要素を取得
                        scalar_value = f"{scalar_val}f"
                        break
                else:
                    raise NotImplementedError(f"Initializer for {self.inputs[1]} not found in graph initializers")
            else:
                # 2番目の入力がスカラー変数の場合
                scalar_value = in2_var
            
            if len(shape1) == 2:
                lines.append(f"    const Matrix<{shape1[0]}, {shape1[1]}> {out_var} = mul_constant<{shape1[0]}, {shape1[1]}>({in1_var}, {scalar_value});")
            elif len(shape1) == 1:
                lines.append(f"    const Vector<{shape1[0]}> {out_var} = mul_constant<{shape1[0]}>({in1_var}, {scalar_value});")
            else:
                raise NotImplementedError(f"Mul: shape {shape1}")
        else:
            # 要素ごとの乗算
            assert shape1 == shape2, f"Mul expects same shapes for element-wise multiplication, got {shape1} and {shape2}"
            
            if len(shape1) == 2:
                lines.append(f"    const Matrix<{shape1[0]}, {shape1[1]}> {out_var} = mul_elem<{shape1[0]}, {shape1[1]}>({in1_var}, {in2_var});")
            else:
                raise NotImplementedError(f"Mul: shape {shape1}")
        
        self.variable_map[self.outputs[0]] = out_var
        return lines
    
    def generate_python(self) -> List[str]:
        """Pythonコード生成"""
        lines = []
        
        assert len(self.inputs) >= 2, f"Mul node requires 2 inputs, got {len(self.inputs)}"
        
        a = self.inputs[0]
        b = self.inputs[1]
        out = self.outputs[0]
        out_var = self.get_output_var_name()
        
        shape_a = self.in_shape(0)
        shape_b = self.in_shape(1)
        is_scalar_multiply = (len(shape_b) == 0)

        if is_scalar_multiply:
            if len(shape_a) not in (1, 2):
                raise NotImplementedError(f"Mul: shape {shape_a}")
            lines.append(f"    {out_var} = {self.get_mapped_var(a)} * {self.get_mapped_var(b)}")
        elif shape_a == shape_b and len(shape_a) == 2:
            lines.append(f"    {out_var} = {self.get_mapped_var(a)} * {self.get_mapped_var(b)}")
        else:
            raise NotImplementedError(f"Mul: shapes {shape_a} * {shape_b}")
        
        self.variable_map[out] = out_var
        
        return lines
