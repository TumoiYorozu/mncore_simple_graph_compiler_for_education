from typing import List
from .. import BaseOperator


def generate_vsm(operator: BaseOperator) -> List[str]:
    operator.check_layout_in("((8_L2B:1, 4:2, 8_L1B:1), (2:1, 4_PE:1, 2_W:1))", 0)
    operator.check_layout_in("((8_L2B:1, 4:2, 8_L1B:1), (2:1, 4_PE:1, 2_W:1))", 1)
    operator.check_layout_out("((4_L2B:2, 4:8), (8:1, 2_W:1))")
    assert operator.loc_prefix_out() == "d" # DRAM 出力


    assert operator.loc_prefix_in(0) == "m" # LM0 に入力を仮定し、実装を軽減
    assert operator.loc_prefix_in(1) == "m" # LM0 に入力を仮定し、実装を軽減
    
    assert operator.addr_in(0) == 0   # 入力1 を addr0 に仮定し、実装を軽減
                                            # ↑、つまり、入力0が "$lm0v" だと仮定している
    assert operator.addr_in(1) == operator.memory_len_in(0)
                                            # 入力2 を、入力 0 の直後と仮定し、実装を軽減


    c = operator.addr_out()      # "$d0" の "0" の部分

    # test unit_tests/train_step/Gemm_256x16_256x16_16x16_transA_a/
    # 問題名：「Mmul TA D 16_256_16」
    
    lines = []

    lines.append("zero $ls96v")
    lines.append("gmwrite $lm0v4  $lx0")
    lines.append("gmwrite $ls96 $lx4")
    lines.append("gmread $lx0 $nowrite")
    lines.append("gbfn $mreadf  $lr0v4")
    lines.append("gmread $lx4 $nowrite")
    lines.append("gbfn $mreadf $lr16v4")
    lines.append("")
    lines.append("")
    lines.append("gmwrite $lm2v4  $lx0")
    lines.append("# gmwrite $ls96 $lx4")
    lines.append("gmread $lx0 $nowrite")
    lines.append("gbfn $mreadf  $lr2v4")
    lines.append("gmread $lx4 $nowrite")
    lines.append("gbfn $mreadf $lr18v4")
    lines.append("")
    lines.append("######")
    lines.append("")
    lines.append("gmwrite $lm16v4 $lx0")
    lines.append("# gmwrite $ls96 $lx4")
    lines.append("gmread $lx0 $nowrite")
    lines.append("gbfn $mreadf $lr32v4")
    lines.append("gmread $lx4 $nowrite")
    lines.append("gbfn $mreadf $lr48v4")
    lines.append("")
    lines.append("")
    lines.append("gmwrite $lm18v4 $lx0")
    lines.append("# gmwrite $ls96 $lx4")
    lines.append("gmread $lx0 $nowrite")
    lines.append("gbfn $mreadf $lr34v4")
    lines.append("gmread $lx4 $nowrite")
    lines.append("gbfn $mreadf $lr50v4")
    lines.append("")
    lines.append("######")
    lines.append("######")
    lines.append("")
    lines.append("gmwrite $lr32v4 $lx0")
    lines.append("gmwrite $lr48v4 $lx4")
    lines.append("gmmul $lx $lr0v4 $ln0v4")
    lines.append("gmmul $lx $lr16v4 $ln16v4")
    lines.append("gmmul $lx $lr2v4 $ln32v4")
    lines.append("gmmul $lx $lr18v4 $ln48v4")
    lines.append("")
    lines.append("")
    lines.append("gmwrite $lr34v4 $lx0")
    lines.append("gmwrite $lr50v4 $lx4")
    lines.append("gmmul $lx $lr0v4 $ln2v4")
    lines.append("gmmul $lx $lr16v4 $ln18v4")
    lines.append("gmmul $lx $lr2v4 $ln34v4")
    lines.append("gmmul $lx $lr18v4 $ln50v4")
    lines.append("")
    lines.append("################################################")
    lines.append("")
    lines.append("# $ln[0:64]")
    lines.append("nop/2")
    lines.append("l1bmm@0 $lln0v  $llb0")
    lines.append("l1bmm@0 $lln16v $llb32")
    lines.append("l1bmm@0 $lln32v $llb64")
    lines.append("l1bmm@0 $lln48v $llb96")
    lines.append("")
    lines.append("nop")
    lines.append("# パディング")
    lines.append("l2bmrffadd $lb0  $lc0")
    lines.append("l2bmrffadd $lb32 $lc64")
    lines.append("l2bmrffadd $lb64 $lc128")
    lines.append("l2bmrffadd $lb96 $lc192")
    lines.append("")
    lines.append("nop")
    lines.append("mvrffadd/n64 $lc0   $p0@0")
    lines.append("mvrffadd/n64 $lc64  $p0@1")
    lines.append("mvrffadd/n64 $lc128 $p0@2")
    lines.append("mvrffadd/n64 $lc192 $p0@3")
    lines.append("")
    lines.append("")
    lines.append(f"mvp/n64 $p0@0 $d{c}@0")
    lines.append(f"mvp/n64 $p0@1 $d{c}@1")
    lines.append(f"mvp/n64 $p0@2 $d{c}@2")
    lines.append(f"mvp/n64 $p0@3 $d{c}@3")

    return lines