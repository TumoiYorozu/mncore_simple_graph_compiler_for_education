from typing import List
from .. import BaseOperator

def generate_vsm(operator: BaseOperator) -> List[str]:
    assert operator.layout_in() == "((4_L2B:2, 4:8), (8:1, 2_W:1))", operator.layout_in()
    assert operator.layout_out() == "((16:2), (2:1, 4_PE:1, 2_W:1))", operator.layout_out()
    
    assert operator.loc_prefix_out() == "m" # LM0 に出力を仮定し、実装を軽減
    
    x = operator.addr_in()      # "$d0" の "0" の部分
    y = operator.addr_out()     # "$lm0v" の "0" の部分

    # test unit_tests/train_step/DL_16x16*
    # 問題名：「DL 16_16」
    
    lines = []

    lines.append(f"mvb/n128 $d{x+0} $lc0")
    lines.append(f"l2bmb $lc0  $lb0")
    lines.append(f"l2bmb $lc64 $lb64")
    lines.append(f"nop/2")
    lines.append(f"l1bmm $lb0  $lm{0+y}v")
    lines.append(f"l1bmm $lb16 $lm{16+y}v")
    lines.append(f"l1bmm $lb32 $lm{32+y}v")
    lines.append(f"l1bmm $lb48 $lm{48+y}v")
    lines.append(f"l1bmm $lb64 $lm{8+y}v")
    lines.append(f"l1bmm $lb80 $lm{24+y}v")
    lines.append(f"l1bmm $lb96 $lm{40+y}v")
    lines.append(f"l1bmm $lb112 $lm{56+y}v")

    return lines