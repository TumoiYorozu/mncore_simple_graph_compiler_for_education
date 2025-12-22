#!/usr/bin/env python3

from typing import List
from . import BaseOperator
from . import register_operator

@register_operator("Cast")
class CastOperator(BaseOperator):
    """Cast演算オペレーター（型変換）"""
    
    # MN-Core変換時にバイパスすべきオペレーター
    MNCORE_BYPASS = True
    
    def generate_cpp(self) -> List[str]:
        """C++コード生成"""
        lines: List[str] = []
        
        assert len(self.inputs) >= 1, f"Cast node requires at least 1 input, got {len(self.inputs)}"
        
        in_var = self.get_mapped_var(self.inputs[0])
        out_var = self.get_output_var_name()
        
        # 型を取得
        to_type = None
        for attr in self.node.attribute:
            if attr.name == 'to':
                to_type = attr.i
                break
        
        # 現在は浮動小数点への変換のみサポート（to=1がfloat）
        if to_type == 1:
            # すでにfloatと仮定し、単にエイリアスを作成
            self.variable_map[self.outputs[0]] = in_var
        else:
            raise NotImplementedError(f"Cast: type {to_type}")
        
        return lines
    
    def generate_python(self) -> List[str]:
        """Pythonコード生成"""
        lines: List[str] = []
        
        inp = self.inputs[0]
        out = self.outputs[0]
        out_var = self.get_output_var_name()
        
        # 型を取得
        to_type = None
        for attr in self.node.attribute:
            if attr.name == 'to':
                to_type = attr.i
                break
        
        if to_type == 1:  # float型
            lines.append(f"    {out_var} = {self.get_mapped_var(inp)}.float()")
        elif to_type == 7:  # int64型
            lines.append(f"    {out_var} = {self.get_mapped_var(inp)}.long()")
        else:
            raise NotImplementedError(f"Cast: type {to_type}")
        
        self.variable_map[out] = out_var
        
        return lines
