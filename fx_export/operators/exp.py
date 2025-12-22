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

        lines.append(f'# --- Range Reduction: r = x - k * ln(2) ---')
        lines.append(f'imm f"1.4426950408889634" $lr30v # 1/ln(2)')

        lines.append(f'imm f"0.6931471805599453" $lr40v # ln(2)')
        lines.append(f'imm f"0.5"                  $lt')
        lines.append(f'imm i"23"                  $ls50v')
        lines.append(f'imm f"-15"                 $lr88v')
        
        # 1つ目の入力のkを計算
        lines.append(f'fvfma $lm0v $lr30v $lt $nowrite # x/ln(2) + 0.5')
        lines.append(f'ffloor $mauf $lr100v             # k = floor(x/ln(2) + 0.5)')
        lines.append(f'fvfma $aluf -$lr40v $lm0v $lr0v # r = x - k*ln(2)')
        lines.append(f'fftoi $lr100v $nowrite           # convert k to integer')
        lines.append(f'ilsl $aluf $ls50v $lr100v      # k << 23 (for 2^k multiplication)')
        # 2つ目の入力のkを計算
        lines.append(f'fvfma $lm8v $lr30v $lt $nowrite')
        lines.append(f'ffloor $mauf $lr200v')
        lines.append(f'fvfma $aluf -$lr40v $lm8v $lr8v')
        lines.append(f'fftoi $lr200v $nowrite')
        lines.append(f'ilsl $aluf $ls50v $lr200v')
        # --- exp(r) のテイラー展開を初期化 ---
        # 初期化: sum = 1.0 + r, power = r
        lines.append(f'imm f"1.0" $lr20v')
        lines.append(f'fvadd $lr0v $aluf  $ls0v      # ls0v = r + 1.0')
        lines.append(f'imm f"1.0" $lr20v')
        lines.append(f'fvadd $lr8v $aluf $ls8v       # ls8v = r + 1.0')
        lines.append(f'fvpassa $lr0v $ls100v         # ls100v = r (r^1)')
        lines.append(f'fvpassa $lr8v $ls200v         # ls200v = r (r^1)')

        # --- テイラー展開のループ ---
        coeff = 1.0
        for k in range(2, 8):  # r^2～r^7
            coeff /= k
            lines.append(f'imm f"{coeff:.17e}" $lr20v')
            lines.append(f'fvmul $lr0v $ls100v $ls100v   # r * r^(k-1) -> r^k')
            lines.append(f'fvfma $mauf $lr20v $ls0v $ls0v # sum += (r^k) * (1/k!)')
            lines.append(f'fvmul $lr8v $ls200v $ls200v')
            lines.append(f'fvfma $mauf $lr20v $ls8v $ls8v')

        # --- 最終結果の合成: exp(x) = exp(r) * 2^k ---
        lines.append(f'fmax $lm0v $lr88v $omr1')
        lines.append(f'fmax $lm8v $lr88v $omr2')
        lines.append(f'iadd/$imr1 $ls0v $lr100v $ln0v')
        lines.append(f'iadd/$imr2 $ls8v $lr200v $ln8v')
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
