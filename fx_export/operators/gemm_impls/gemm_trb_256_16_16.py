from typing import List
from .. import BaseOperator



def generate_vsm(operator: BaseOperator) -> List[str]:
    operator.check_layout_in("((8_L2B:1, 4:2, 8_L1B:1), (2:1, 4_PE:1, 2_W:1))", 0)
    operator.check_layout_in("((16:2), (2:1, 4_PE:1, 2_W:1))", 1)
    operator.check_layout_out("((8_L2B:1, 4:2, 8_L1B:1), (2:1, 4_PE:1, 2_W:1))")


    assert operator.loc_prefix_in(0) == "m" # LM0 に入力を仮定し、実装を軽減
    assert operator.loc_prefix_in(1) == "m" # LM0 に入力を仮定し、実装を軽減
    
    assert operator.addr_in(0) == 0   # 入力1 を addr0 に仮定し、実装を軽減
                                            # ↑、つまり、入力 0 が "$lm0v" だと仮定している
    assert operator.addr_in(1) == operator.memory_len_in(0)
                                            # 入力2 を、入力 0 の直後と仮定し、実装を軽減

    assert operator.loc_prefix_out() == "n" # LM1 に出力を仮定し、実装を軽減
    assert operator.addr_out() == 0         # 出力 を addr0 に仮定し、実装を軽減
                                            # ↑、つまり、出力が "$ln0v" だと仮定している
    
    # test unit_tests/train_step/Gemm_256x16_16x16_256x16_transB_a/
    # 問題名：「Mmul TB 256_16_16」
    
    lines = []

    lines.append(f"gbfn    $lm16v4 $nowrite;")
    lines.append(f"gmwrite $aluf $ly0;")
    lines.append(f"gbfn    $lm32v4 $nowrite;")
    lines.append(f"gmwrite $aluf $ly4;")
    lines.append(f"")
    lines.append(f"gbfn $lm0v4 $nowrite;")
    lines.append(f"gmmul $ly $aluf $ls0v4")
    lines.append(f"")
    lines.append(f"##############################")
    lines.append(f"")
    lines.append(f"gbfn    $lm18v4 $nowrite;")
    lines.append(f"gmwrite $aluf $ly0;")
    lines.append(f"gbfn    $lm34v4 $nowrite;")
    lines.append(f"gmwrite $aluf $ly4;")
    lines.append(f"")
    lines.append(f"gbfn $lm2v4 $nowrite;")
    lines.append(f"gmfma $ly $aluf $ls0v4 $ln0v4")
    lines.append(f"")
    lines.append(f"##############################")
    lines.append(f"##############################")
    lines.append(f"")
    lines.append(f"gbfn    $lm48v4 $nowrite;")
    lines.append(f"gmwrite $aluf $ly0;")
    lines.append(f"gbfn    $lm64v4 $nowrite;")
    lines.append(f"gmwrite $aluf $ly4;")
    lines.append(f"")
    lines.append(f"gbfn $lm0v4 $nowrite;")
    lines.append(f"gmmul $ly $aluf $ls0v4")
    lines.append(f"")
    lines.append(f"##############################")
    lines.append(f"")
    lines.append(f"gbfn    $lm50v4 $nowrite;")
    lines.append(f"gmwrite $aluf $ly0;")
    lines.append(f"gbfn    $lm66v4 $nowrite;")
    lines.append(f"gmwrite $aluf $ly4;")
    lines.append(f"")
    lines.append(f"gbfn $lm2v4 $nowrite;")
    lines.append(f"gmfma $ly $aluf $ls0v4 $ln2v4")

    return lines