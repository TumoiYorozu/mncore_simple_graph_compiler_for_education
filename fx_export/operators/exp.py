#!/usr/bin/env python3

from typing import List, Dict, Any, Optional
from . import BaseOperator, register_operator


@register_operator("Exp")
class ExpOperator(BaseOperator):
    """Exp（指数関数）演算オペレーター"""

    def generate_vsm(self) -> List[str]:

        assert self.loc_prefix_in() == "m"  # LM0 に入力を仮定し、実装を軽減
        assert self.addr_in() == 0  # Addr 0 に入力を仮定し、実装を軽減
        # ↑、つまり、入力が "$lm0v" だと仮定している

        assert self.loc_prefix_out() == "n"  # LM1 に出力を仮定し、実装を軽減
        assert self.addr_out() == 0  # Addr 0 に出力を仮定し、実装を軽減
        # ↑、つまり、出力が "$ln0v" だと仮定している

        assert self.memory_len_in(0) == 16  # 実装の簡略化のため、PE あたり 16 単語を仮定

        # test unit_tests/train_step/Exp_*
        lines = []

        lines.append("""
            # lm -> ls(backup)
            ipassa $llm[0,4,8,12] $lls[128,132,136,140]
            
            # k = round(x / ln(2))
            imm f"1.44269504" $t
            imm f"0.5" $s[0,1,2,3]
            fvfma $lm[0,2,4,6] $t $aluf $lr[16,18,20,22]
            fvfma $lm[8,10,12,14] $t $ls[0,0,0,0] $lr[24,26,28,30]
            ffloor $lr[16,18,20,22] $lr[16,18,20,22]
            ffloor $lr[24,26,28,30] $lr[24,26,28,30]
            
            # r = k * -ln(2) + x
            imm f"0.69314718056" $t
            fvfma $lr[16,18,20,22] -$aluf $lm[0,2,4,6] $ls[16,18,20,22]
            fvfma $lr[24,26,28,30] -$t $lm[8,10,12,14] $ls[24,26,28,30]
            nop
            
            # r -> lm
            ipassa $lls[16,20,24,28] $llm[0,4,8,12]
            
            # exp
            imm f"0.00019841269841270" $t
            nop
            fvmul $lm[0,2,4,6] $aluf $lr[0,2,4,6]
            fvmul $lm[8,10,12,14] $t $lr[8,10,12,14]
            
            imm f"0.00138888888888889" $t
            fvadd $lr[0,2,4,6] $aluf $lr[0,2,4,6]
            fvadd $lr[8,10,12,14] $t $lr[8,10,12,14]
            fvmul $lm[0,2,4,6] $lr[0,2,4,6] $lr[0,2,4,6]
            fvmul $lm[8,10,12,14] $lr[8,10,12,14] $lr[8,10,12,14]
            
            imm f"0.00833333333333333" $t
            fvadd $lr[0,2,4,6] $aluf $lr[0,2,4,6]
            fvadd $lr[8,10,12,14] $t $lr[8,10,12,14]
            fvmul $lm[0,2,4,6] $lr[0,2,4,6] $lr[0,2,4,6]
            fvmul $lm[8,10,12,14] $lr[8,10,12,14] $lr[8,10,12,14]
            
            imm f"0.04166666666666666" $t
            fvadd $lr[0,2,4,6] $aluf $lr[0,2,4,6]
            fvadd $lr[8,10,12,14] $t $lr[8,10,12,14]
            fvmul $lm[0,2,4,6] $lr[0,2,4,6] $lr[0,2,4,6]
            fvmul $lm[8,10,12,14] $lr[8,10,12,14] $lr[8,10,12,14]
            
            imm f"0.16666666666666666" $t
            fvadd $lr[0,2,4,6] $aluf $lr[0,2,4,6]
            fvadd $lr[8,10,12,14] $t $lr[8,10,12,14]
            fvmul $lm[0,2,4,6] $lr[0,2,4,6] $lr[0,2,4,6]
            fvmul $lm[8,10,12,14] $lr[8,10,12,14] $lr[8,10,12,14]
            
            imm f"0.50000000000000000" $t
            fvadd $lr[0,2,4,6] $aluf $lr[0,2,4,6]
            fvadd $lr[8,10,12,14] $t $lr[8,10,12,14]
            fvmul $lm[0,2,4,6] $lr[0,2,4,6] $lr[0,2,4,6]
            fvmul $lm[8,10,12,14] $lr[8,10,12,14] $lr[8,10,12,14]
            
            imm f"1.00000000000000000" $t
            fvadd $lr[0,2,4,6] $aluf $lr[0,2,4,6]
            fvadd $lr[8,10,12,14] $t $lr[8,10,12,14]
            fvmul $lm[0,2,4,6] $lr[0,2,4,6] $lr[0,2,4,6]
            fvmul $lm[8,10,12,14] $lr[8,10,12,14] $lr[8,10,12,14]
            
            fvadd $t $lr[0,2,4,6] $ln[0,2,4,6]
            fvadd $t $lr[8,10,12,14] $ln[8,10,12,14] ;imm i"23" $t
            fftoi $lr[16,18,20,22] $ls[32,34,36,38]
            fftoi $lr[24,26,28,30] $ls[40,42,44,46]
            ilsl $ls[32,34,36,38] $t $lr[32,34,36,38]
            ilsl $ls[40,42,44,46] $t $lr[40,42,44,46]
            iadd $ln[0,2,4,6] $lr[32,34,36,38] $ls[48,50,52,54]
            iadd $ln[8,10,12,14] $lr[40,42,44,46] $ls[56,58,60,62]
            nop
            ipassa $lls[48,52,56,60] $lln[0,4,8,12]
            
            
            # x - 60.0
            imm f"60.0" $t
            nop
            fvadd $ls[128,130,132,134] $t $lr[128,130,132,134]
            fvadd $ls[136,138,140,142] $t $lr[136,138,140,142] ;imm f"1.0" $t
            frelu $lr[128,130,132,134] $aluf $lr[128,130,132,134]
            frelu $lr[136,138,140,142] $t $lr[136,138,140,142]
            nop
            fvmul $ls[48,50,52,54] $lr[128,130,132,134] $ln[0,2,4,6]
            fvmul $ls[56,58,60,62] $lr[136,138,140,142] $ln[8,10,12,14]
        """)
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

        assert len(self.inputs) >= 1, f"""
        Exp node requires at least 1 input, got {len(self.inputs)}
        """

        in_var = self.get_mapped_var(self.inputs[0])
        out_var = self.get_output_var_name()

        # 出力形状を取得（Expの出力形状は入力形状と同じ）
        shape = self.in_shape(0)

        if not shape:
            # 形状情報が全くない場合はエラー
            raise ValueError(f"Exp: No shape information available for output '{self.outputs[0]}'. "
                             f"This is likely a bug in the ONNX graph generation.")

        if len(shape) == 2:
            lines.append(f"const Matrix<{shape[0]}, {shape[1]}> {out_var} = exp<{shape[0]}, {shape[1]}>({in_var});")
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
