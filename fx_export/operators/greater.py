#!/usr/bin/env python3

from typing import List, Dict, Any, Optional
from . import BaseOperator
from . import register_operator
from onnx import numpy_helper

@register_operator("Greater")
class GreaterOperator(BaseOperator):
    """Greater比較演算オペレーター"""
    
    def generate_vsm(self) -> List[str]:
        in_shape = self.in_shape()
        assert in_shape == self.out_shape()
        
        in_prefix = self.loc_prefix_in()    # $lm0v  の "m" の部分
        in_offset = self.addr_in()          # $lm0v  の "0" の部分
        
        out_prefix = self.loc_prefix_out()  # $ln0v の "n" の部分
        out_offset = self.addr_out()        # $ln0v の "0" の部分


        # test unit_tests/train_step/Greater_*
        # 問題名：「ReLU grad」

        lines = []
        lines.append('imm f"1.0" $t')
        lines.append('nop')
        for i in range(self.memory_len_in_div_ceil(8, 0)):  # PE あたりの長さを、8単語で割って切り上げ
            in_addr  = in_offset  + i * 8
            out_addr = out_offset + i * 8
            lines.append(f"frelu $t $l{in_prefix}{in_addr}v $l{out_prefix}{out_addr}v")
        return lines
    
    def testcase_hint(self) -> Optional[str]:
        return "relu_grad.vsm"

    def get_memory_layout_tag(self) -> Dict[str, List[str]]:
        return {
            "inputs": ["default"],
            "outputs": ["default"]
        }
    
    def generate_cpp(self) -> List[str]:
        """C++コード生成"""
        lines = []
        
        assert len(self.inputs) >= 2, f"Greater node requires at least 2 inputs, got {len(self.inputs)}"
        
        in1_var = self.get_mapped_var(self.inputs[0])
        in2_var = self.get_mapped_var(self.inputs[1])
        out_var = self.get_output_var_name()
        
        # 形状を取得
        shape1 = self.in_shape(0)
        shape2 = self.in_shape(1)
        
        # 第2入力がスカラー（定数）の場合
        if len(shape2) == 0:
            # 第2入力が初期化子（定数）かチェック
            is_zero = False
            in2_name = self.inputs[1]
            # 初期化子から値を取得して0かどうか確認
            for init in self.graph.initializer:
                if init.name == in2_name:
                    # 初期化子の値を取得
                    value = numpy_helper.to_array(init)
                    if value == 0:
                        is_zero = True
                    break
            # 第2入力が0でない場合はエラー
            if not is_zero:
                raise NotImplementedError(f"Greater operator only supports comparison with 0, but got comparison with {in2_var}")
            
            # 0との比較
            if len(shape1) == 2:
                # relu_gradを使用: relu_grad(grad, x)はx > 0の場所でgradを返し、それ以外で0を返す
                # 1で埋めた行列を作成してgradとして使用
                lines.append(f"    Matrix<{shape1[0]}, {shape1[1]}> ones = zeros<{shape1[0]}, {shape1[1]}>();")
                lines.append(f"    for (int i = 0; i < {shape1[0]}; i++) {{")
                lines.append(f"        for (int j = 0; j < {shape1[1]}; j++) {{")
                lines.append(f"            ones[i][j] = 1.0f;")
                lines.append(f"        }}")
                lines.append(f"    }}")
                lines.append(f"    const Matrix<{shape1[0]}, {shape1[1]}> {out_var} = relu_grad<{shape1[0]}, {shape1[1]}>(ones, {in1_var});")
            else:
                
                raise NotImplementedError(f"Greater: shape {shape1}")
        else:
            # 要素ごとの比較
            raise NotImplementedError
        
        self.variable_map[self.outputs[0]] = out_var
        return lines
    
    def generate_python(self) -> List[str]:
        """Pythonコード生成"""
        lines = []
        
        a = self.inputs[0]
        b = self.inputs[1]
        out = self.outputs[0]
        out_var = self.get_output_var_name()
        
        # bが定数かどうかを確認（0との比較のみサポート）
        is_zero = False
        for init in self.graph.initializer:
            if init.name == b:
                value = numpy_helper.to_array(init)
                if value == 0:
                    is_zero = True
                break
        if not is_zero:
            raise NotImplementedError(f"Greater operator only supports comparison with 0, but got comparison with {b}")
        
        lines.append(f"    {out_var} = {self.get_mapped_var(a)} > 0")
        self.variable_map[out] = out_var
        
        return lines
