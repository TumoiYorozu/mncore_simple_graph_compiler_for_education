from typing import List
from .. import BaseOperator



def generate_vsm(operator: BaseOperator) -> List[str]:
    operator.check_layout_in("((8_L2B:1, 4:2, 8_L1B:1), (2:1, 4_PE:1, 2_W:1))", 0)
    operator.check_layout_in("((16:2), (2:1, 4_PE:1, 2_W:1))", 1)
    operator.check_layout_out("((8_L2B:1, 4:2, 8_L1B:1), (2:1, 4_PE:1, 2_W:1))")


    assert operator.loc_prefix_in(0) == "m" # LM0 に入力を仮定し、実装を軽減
    assert operator.loc_prefix_in(1) == "m" # LM0 に入力を仮定し、実装を軽減
    
    assert operator.addr_in(0) == 0   # 入力0 を addr0 に仮定し、実装を軽減
                                            # ↑、つまり、入力0が "$lm0v" だと仮定している
    assert operator.addr_in(1) == operator.memory_len_in(0)
                                            # 入力1 を、入力 0 の直後と仮定し、実装を軽減

    assert operator.loc_prefix_out() == "n" # LM1 に出力を仮定し、実装を軽減
    assert operator.addr_out() == 0         # 出力 を addr0 に仮定し、実装を軽減
                                            # ↑、つまり、出力が "$ln0v" だと仮定している
    
    # test unit_tests/train_step/Gemm_256x16_16x16_256x16_a/
    # 問題名：「Mmul 256_16_16」
    
    lines = []

    lines.append("gmwrite $lm16v4 $ly0;")
    lines.append("gmwrite $lm32v4 $ly4;")
    lines.append("")
    lines.append("gmread $ly0  $nowrite")
    lines.append("gbfn $mreadf $nowrite")
    lines.append("gmwrite $aluf $lx0")
    lines.append("")
    lines.append("gmread $ly4  $nowrite")
    lines.append("gbfn $mreadf $nowrite")
    lines.append("gmwrite $aluf $lx4")
    lines.append("")
    lines.append("")
    lines.append("# d getf $ly0n0c0b0m0p0 8")
    lines.append("# d getbf $lx0n0c0b0m0p0 8")
    lines.append("")
    lines.append("gbfn $lm0v4 $ls0v")
    lines.append("gmmul $lx $aluf $lr0v;")
    lines.append("")
    lines.append("# d getbf $ls0n0c0b0m0p- 1")
    lines.append("# d getf $lr0n0c0b0m0p- 1")
    lines.append("")
    lines.append("##################")
    lines.append("")
    lines.append("gmwrite $lm48v4 $ly0;")
    lines.append("gmwrite $lm64v4 $ly4;")
    lines.append("")
    lines.append("gmread $ly0  $nowrite")
    lines.append("gbfn $mreadf $nowrite")
    lines.append("gmwrite $aluf $lx0")
    lines.append("")
    lines.append("gmread $ly4  $nowrite")
    lines.append("gbfn $mreadf $nowrite")
    lines.append("gmwrite $aluf $lx4")
    lines.append("")
    lines.append("")
    lines.append("gbfn $lm2v4 $ls0v")
    lines.append("gmfma $lx $aluf $lr0v $ln0v4;")
    lines.append("")
    lines.append("")
    lines.append("")
    lines.append("")
    lines.append("")
    lines.append("gmwrite $lm18v4 $ly0;")
    lines.append("gmwrite $lm34v4 $ly4;")
    lines.append("")
    lines.append("gmread $ly0  $nowrite")
    lines.append("gbfn $mreadf $nowrite")
    lines.append("gmwrite $aluf $lx0")
    lines.append("")
    lines.append("gmread $ly4  $nowrite")
    lines.append("gbfn $mreadf $nowrite")
    lines.append("gmwrite $aluf $lx4")
    lines.append("")
    lines.append("")
    lines.append("gbfn $lm0v4 $ls0v")
    lines.append("gmmul $lx $aluf $lr0v;")
    lines.append("")
    lines.append("##################")
    lines.append("")
    lines.append("gmwrite $lm50v4 $ly0;")
    lines.append("gmwrite $lm66v4 $ly4;")
    lines.append("")
    lines.append("gmread $ly0  $nowrite")
    lines.append("gbfn $mreadf $nowrite")
    lines.append("gmwrite $aluf $lx0")
    lines.append("")
    lines.append("gmread $ly4  $nowrite")
    lines.append("gbfn $mreadf $nowrite")
    lines.append("gmwrite $aluf $lx4")
    lines.append("")
    lines.append("")
    lines.append("gbfn $lm2v4 $ls0v")
    lines.append("gmfma $lx $aluf $lr0v $ln2v4;")


    return lines