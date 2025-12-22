from typing import List
from .. import BaseOperator

def generate_vsm(operator: BaseOperator) -> List[str]:
    assert operator.layout_in() == "((4_L2B:2, 32:1, 2_W:1))", operator.layout_in()
    assert operator.layout_out() == "((4_L2B:2, 32:1, 2_W:1))", operator.layout_out()
    
    assert operator.loc_prefix_out() == "m" # LM0 に出力を仮定し、実装を軽減
    
    x = operator.addr_in()      # "$d0" の "0" の部分
    y = operator.addr_out()     # "$lm0v" の "0" の部分

    # test unit_tests/train_step/DL_256_*
    # 問題名：「DL 256」
    
    lines = []

    lines.append(f"mvb2/n64 $d{x+0} $lc0")
    lines.append(f"l2bmb $lc0 $lb0")
    lines.append(f"nop")
    lines.append(f"l1bmp $lb0 $lm{y+0}v")
    lines.append(f"l1bmp $lb4 $lm{y+8}v")
    lines.append(f"l1bmp $lb8 $lm{y+16}v")
    lines.append(f"l1bmp $lb12 $lm{y+24}v")
    lines.append(f"l1bmp $lb16 $lm{y+32}v")
    lines.append(f"l1bmp $lb20 $lm{y+40}v")
    lines.append(f"l1bmp $lb24 $lm{y+48}v")
    lines.append(f"l1bmp $lb28 $lm{y+56}v")

    return lines