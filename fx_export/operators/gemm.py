#!/usr/bin/env python3

from typing import List, Dict, Any, Optional
from . import BaseOperator
from . import register_operator
from .gemm_impls import gemm_trb_256_1024_16, gemm_trb_256_16_16, gemm_tra_16_256_16, gemm_tra_16_256_1024, gemm_notr_256_16_16


@register_operator("Gemm")
class GemmOperator(BaseOperator):
    """Gemm演算オペレーター: Y = alpha * A @ B^T（バイアスなし、転置のみサポート）"""
    
    def generate_vsm(self) -> List[str]:
        transA = self.get_attr('transA')
        transB = self.get_attr('transB')
        transA = 0 if transA is None else transA.i
        transB = 0 if transB is None else transB.i
        
        shape0 = self.in_shape(0)
        shape1 = self.in_shape(1)
        if transA == 0 and transB == 1:
            if shape0 == [256, 16] and shape1 == [16, 16]:
                return gemm_trb_256_16_16.generate_vsm(self)
            if shape0 == [256, 1024] and shape1 == [16, 1024]:
                return gemm_trb_256_1024_16.generate_vsm(self)

        elif transA == 0 and transB == 0:
            if shape0 == [256, 16] and shape1 == [16, 16]:
                return gemm_notr_256_16_16.generate_vsm(self)
        elif transA == 1 and transB == 0:
            if shape0 == [256, 16] and shape1 == [256, 16]:
                return gemm_tra_16_256_16.generate_vsm(self)
            if shape0 == [256, 16] and shape1 == [256, 1024]:
                return gemm_tra_16_256_1024.generate_vsm(self)
        raise NotImplementedError(f"{transA=}, {transB=}, {shape0=}, {shape1=}")


    def testcase_hint(self) -> Optional[str]:
        transA = self.get_attr('transA')
        transB = self.get_attr('transB')
        transA = 0 if transA is None else transA.i
        transB = 0 if transB is None else transB.i
        if transA == 0 and transB == 1:
            return "matmul_trb_*.vsm"
        elif transA == 0 and transB == 0:
            return "matmul_notr_*.vsm"
        elif transA == 1 and transB == 0:
            return "matmul_tra_*.vsm"
        return None

    def get_memory_layout_tag(self) -> Dict[str, List[str]]:
        transA = self.get_attr('transA')
        transB = self.get_attr('transB')
        transA = 0 if transA is None else transA.i
        transB = 0 if transB is None else transB.i
        
        shape0 = self.in_shape(0)
        shape1 = self.in_shape(1)
        if transA == 0 and transB == 1:
            if shape0 == [256, 16] and shape1 == [16, 16]:
                return {
                    "inputs": ["default", "default"],
                    "outputs": ["default"]
                }
            if shape0 == [256, 1024] and shape1 == [16, 1024]:
                return {
                    "inputs": ["default", "default"],
                    "outputs": ["DRAM"]
                }
        elif transA == 0 and transB == 0:
            if shape0 == [256, 16] and shape1 == [16, 16]:
                return {
                    "inputs": ["default", "default"],
                    "outputs": ["default"]
                }
        elif transA == 1 and transB == 0:
            if shape0 == [256, 16] and shape1 == [256, 16]:
                return {
                    "inputs": ["default", "default"],
                    "outputs": ["DRAM"]
                }
            if shape0 == [256, 16] and shape1 == [256, 1024]:
                return {
                    "inputs": ["L2B", "default"],
                    "outputs": ["DRAM"]
                }
        raise NotImplementedError(f"{transA=}, {transB=}, {shape0=}, {shape1=}")



    def generate_cpp(self) -> List[str]:
        """C++コード生成"""
        lines = []
        
        assert len(self.inputs) == 2, f"Gemm node requires exactly 2 inputs (no bias support), got {len(self.inputs)}"
        
        in_a = self.get_mapped_var(self.inputs[0])
        in_b = self.get_mapped_var(self.inputs[1])
        out_var = self.get_output_var_name()
        
        # 属性を取得してサポート範囲をチェック
        alpha = 1.0
        beta = 0.0
        transA = 0
        transB = 0
        
        for attr in self.node.attribute:
            if attr.name == 'alpha':
                alpha = attr.f
                if alpha != 1.0:
                    raise ValueError(f"Gemm node: alpha={alpha} is not supported. Only alpha=1.0 is supported.")
            elif attr.name == 'beta':
                beta = attr.f
                if beta != 0.0:
                    raise ValueError(f"Gemm node: beta={beta} is not supported. Only beta=0.0 (no bias) is supported.")
            elif attr.name == 'transA':
                transA = attr.i
            elif attr.name == 'transB':
                transB = attr.i
        
        # 形状を取得
        shape_a = self.in_shape(0)
        shape_b = self.in_shape(1)
        
        assert len(shape_a) == 2 and len(shape_b) == 2, f"Gemm requires 2D inputs, got {shape_a} and {shape_b}"
        
        # 転置を考慮した実際の形状
        m = shape_a[0] if not transA else shape_a[1]
        k_a = shape_a[1] if not transA else shape_a[0]
        k_b = shape_b[0] if not transB else shape_b[1]
        n = shape_b[1] if not transB else shape_b[0]
        
        assert k_a == k_b, f"Inner dimensions must match for Gemm: {k_a} != {k_b}"
        
        # C++コードを生成（インライン転置版）
        
        # 転置をインラインで適用
        if transA:
            in_a = f"trans<{shape_a[0]}, {shape_a[1]}>({in_a})"
        
        if transB:
            in_b = f"trans<{shape_b[0]}, {shape_b[1]}>({in_b})"
        
        # MatMul実行（バイアスなし、転置はインライン）
        lines.append(f"    const Matrix<{m}, {n}> {out_var} = matmul<{m}, {k_a}, {n}>({in_a}, {in_b});")
        
        self.variable_map[self.outputs[0]] = out_var
        return lines
    
    def generate_python(self) -> List[str]:
        """Pythonコード生成"""
        lines = []
        
        assert len(self.inputs) == 2, f"Gemm node requires exactly 2 inputs (no bias support), got {len(self.inputs)}"
        
        a = self.inputs[0]
        b = self.inputs[1]
        out_var = self.get_output_var_name()
        
        # 属性を取得
        alpha = 1.0
        beta = 0.0
        transA = 0
        transB = 0
        
        for attr in self.node.attribute:
            if attr.name == 'alpha':
                alpha = attr.f
                if alpha != 1.0:
                    raise ValueError(f"Gemm node in Python generation: alpha={alpha} is not supported")
            elif attr.name == 'beta':
                beta = attr.f
                if beta != 0.0:
                    raise ValueError(f"Gemm node in Python generation: beta={beta} is not supported (no bias)")
            elif attr.name == 'transA':
                transA = attr.i
            elif attr.name == 'transB':
                transB = attr.i
        
        # 行列乗算部分
        a_var = self.get_mapped_var(a)
        b_var = self.get_mapped_var(b)
        
        if transA:
            a_var = f"{a_var}.T"
        if transB:
            b_var = f"{b_var}.T"
        
        lines.append(f"    {out_var} = torch.matmul({a_var}, {b_var})")
        
        # 内部ULの処理は共通メソッドを使用
        self.handle_internal_ul_python(lines, out_var)
        
        return lines
