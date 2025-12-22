#!/usr/bin/env python3

from typing import List, Dict, Any, Optional
from . import BaseOperator, register_operator

@register_operator("Exp")
class ExpOperator(BaseOperator):
    """Exp（指数関数）演算オペレーター"""
    def generate_vsm(self) -> List[str]:
        
        assert self.loc_prefix_in()  == "m" # LM0 に入力を仮定し、実装を軽減
        assert self.addr_in()        ==  0  # Addr 0 に入力を仮定し、実装を軽減
                                            # ↑、つまり、入力が "$lm0v" だと仮定している
        
        assert self.loc_prefix_out() == "n" # LM1 に出力を仮定し、実装を軽減
        assert self.addr_out()       ==  0  # Addr 0 に出力を仮定し、実装を軽減
                                            # ↑、つまり、出力が "$ln0v" だと仮定している
        
        assert self.memory_len_in(0) == 16  # 実装の簡略化のため、PE あたり 16 単語を仮定
        
        # test unit_tests/train_step/Exp_*
        lines = []

        lines.append(f'imm f"1.4426950408889634" $lr0v')
        raise NotImplementedError("Please implement the VSM code!!")
        return lines

    def testcase_hint(self) -> Optional[str]:
        return "exp.vsm"

    def get_memory_layout_tag(self) -> Dict[str, List[str]]:
        return {
            "inputs": ["default"],
            "outputs": ["default"]
        }
    
    def generate_cpp(self) -> List[str]:
        """C++コード生成"""
        lines = []
        
        assert len(self.inputs) >= 1, f"Exp node requires at least 1 input, got {len(self.inputs)}"
        
        in_var = self.get_mapped_var(self.inputs[0])
        out_var = self.get_output_var_name()
        
        # 出力形状を取得（Expの出力形状は入力形状と同じ）
        shape = self.in_shape(0)
        
        if not shape:
            # 形状情報が全くない場合はエラー
            raise ValueError(f"Exp: No shape information available for output '{self.outputs[0]}'. "
                           f"This is likely a bug in the ONNX graph generation.")
        
        if len(shape) == 2:
            lines.append(f"    const Matrix<{shape[0]}, {shape[1]}> {out_var} = exp<{shape[0]}, {shape[1]}>({in_var});")
        else:
            raise NotImplementedError(f"Exp: Unsupported shape {shape}. Only 2D matrices are supported.")
        
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
            raise NotImplementedError(f"Exp: shape {shape}")

        lines.append(f"    {out_var} = torch.exp({self.get_mapped_var(inp)})")
        
        self.variable_map[out] = out_var
        
        return lines
