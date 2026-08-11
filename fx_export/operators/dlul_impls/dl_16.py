from typing import List
from .. import BaseOperator

def generate_vsm(operator: BaseOperator) -> List[str]:
    assert operator.layout_in() == "((8:1, 2_W:1))", operator.layout_in()
    assert operator.layout_out() == "((2:1, 4_PE:1, 2_W:1))", operator.layout_out()
    
    assert operator.loc_prefix_out() == "m" # LM0 に出力を仮定し、実装を軽減
    
    x = operator.addr_in()      # "$d0" の "0" の部分
    y = operator.addr_out()     # "$lm0v" の "0" の部分

    # test unit_tests/train_step/DL_16_*
    # 問題名：「DL 16」

    lines = []

    lines.append(f"mvp/n64 $d{x+0}@0 $p0@0")
    lines.append(f"mvb/n64 $p0@0 $lc0")
    # [16] は全グループで同じ値が必要なので分配(l2bmd)ではなく放送(l2bmb)する
    lines.append(f"l2bmb $lc0 $lb0")
    lines.append(f"nop")
    lines.append(f"l1bmm $lb0 $lm{y}v")

    return lines
