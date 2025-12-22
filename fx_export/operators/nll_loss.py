#!/usr/bin/env python3

from typing import List, Dict, Any, Optional
from . import BaseOperator
from . import register_operator

@register_operator("NegativeLogLikelihoodLoss")
class NLLLossOperator(BaseOperator):
    """NegativeLogLikelihoodLoss演算オペレーター"""
    
    def get_memory_layout_tag(self) -> Dict[str, List[str]]:
        """メモリレイアウトタグを定義"""
        return {
            "inputs": ["default", "default"],  # 2つの入力
            "outputs": ["DRAM"]  # スカラーなのでDRAMに直接出力
        }
    
    def generate_cpp(self) -> List[str]:
        """C++コード生成"""
        lines = []
        
        assert len(self.inputs) >= 2, f"NLLLoss node requires at least 2 inputs, got {len(self.inputs)}"
        
        log_probs_var = self.get_mapped_var(self.inputs[0])
        target_var = self.get_mapped_var(self.inputs[1])
        out_var = self.get_output_var_name()
        
        # 形状を取得
        log_probs_shape = self.in_shape(0)
        target_shape = self.in_shape(1)
        
        assert len(log_probs_shape) == 2, f"NLLLoss expects 2D log_probs, got {log_probs_shape}"
        assert len(target_shape) == 1, f"NLLLoss expects 1D target, got {target_shape}"
        
        batch_size = log_probs_shape[0]
        num_classes = log_probs_shape[1]
        
        # リダクションを取得（デフォルトはmean）
        reduction = 'mean'
        for attr in self.node.attribute:
            if attr.name == 'reduction':
                reduction = attr.s.decode('utf-8')
                break
        
        if reduction == 'mean':
            # gather_sum関数を使用
            lines.append(f"    const float {out_var} = gather_sum<{batch_size}, {num_classes}>(mul_constant<{batch_size}, {num_classes}>({log_probs_var}, -1.0f / {batch_size}), {target_var});")
        else:
            raise NotImplementedError(f"NLLLoss: reduction {reduction}")
        
        self.variable_map[self.outputs[0]] = out_var
        return lines
    
    def generate_python(self) -> List[str]:
        """Pythonコード生成"""
        lines = []
        
        log_probs = self.inputs[0]
        target = self.inputs[1]
        out_var = self.get_output_var_name()
        
        reduction = 'mean'
        for attr in self.node.attribute:
            if attr.name == 'reduction':
                reduction = attr.s.decode('utf-8')
                break
        if reduction != 'mean':
            raise NotImplementedError(f"NLLLoss: reduction {reduction}")
        
        lines.append(f"    {out_var} = F.nll_loss({self.get_mapped_var(log_probs)}, {self.get_mapped_var(target)}, reduction='mean')")
        
        # 内部ULの処理は共通メソッドを使用
        self.handle_internal_ul_python(lines, out_var)
        
        return lines
