#!/usr/bin/env python3

from typing import List
from . import BaseOperator
from . import register_operator

@register_operator("Identity")
class IdentityOperator(BaseOperator):
    """Identity演算オペレーター（恒等変換）"""
    
    # MN-Core変換時にバイパスすべきオペレーター
    MNCORE_BYPASS = True

    
    def generate_cpp(self) -> List[str]:
        """C++コード生成"""
        lines: List[str] = []
        
        # 入力をエイリアスするだけ
        if len(self.inputs) >= 1:
            self.variable_map[self.outputs[0]] = self.get_mapped_var(self.inputs[0])
        
        # 何もコードを生成しない
        return lines
    
    def generate_python(self) -> List[str]:
        """Pythonコード生成"""
        lines: List[str] = []
        
        if len(self.inputs) >= 1:
            out_var = self.get_output_var_name()
            lines.append(f"    {out_var} = {self.get_mapped_var(self.inputs[0])}")
            self.variable_map[self.outputs[0]] = out_var
        
        return lines
    
    def generate_vsm(self) -> List[str]:
        """VSMコード生成（Identityは何も生成しない）"""
        # Identityは単なるエイリアスなので、VSMコードは生成しない
        return []