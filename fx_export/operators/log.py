#!/usr/bin/env python3

from typing import List, Dict, Any, Optional
from . import BaseOperator
from . import register_operator

@register_operator("Log")
class LogOperator(BaseOperator):
    """対数演算オペレーター"""
    
    
    def get_memory_layout_tag(self) -> Dict[str, List[str]]:
        """メモリレイアウトタグを定義"""
        return {
            "inputs": ["default"],
            "outputs": ["default"]
        }
    
    def generate_cpp(self) -> List[str]:
        """C++コード生成"""
        lines = []
        
        assert len(self.inputs) >= 1, f"Log node requires at least 1 input, got {len(self.inputs)}"
        
        in_var = self.get_mapped_var(self.inputs[0])
        out_var = self.get_output_var_name()
        
        # 形状を取得
        shape = self.in_shape(0)
        
        if len(shape) == 2:
            lines.append(f"    const Matrix<{shape[0]}, {shape[1]}> {out_var} = log<{shape[0]}, {shape[1]}>({in_var});")
        else:
            raise NotImplementedError(f"Log: shape {shape}")
        
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
            raise NotImplementedError(f"Log: shape {shape}")

        lines.append(f"    {out_var} = torch.log({self.get_mapped_var(inp)})")
        self.variable_map[out] = out_var
        
        return lines
