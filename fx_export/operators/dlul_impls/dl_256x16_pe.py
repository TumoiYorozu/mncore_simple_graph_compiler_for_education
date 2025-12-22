from typing import List
from .. import BaseOperator

def generate_vsm(operator: BaseOperator) -> List[str]:
    assert operator.layout_in() == "((4_L2B:2, 64:8), (8:1, 2_W:1))", operator.layout_in()
    assert operator.layout_out() == "((4_L2B:2, 64:2), (2:1, 4_PE:1, 2_W:1))", operator.layout_out()
    
    assert operator.loc_prefix_out() == "m" # LM0 に出力を仮定し、実装を軽減
    
    x = operator.addr_in()      # "$d0" の "0" の部分
    y = operator.addr_out()     # "$lm0v" の "0" の部分

    # test unit_tests/train_step/DL_256x16_PE_*
    # 問題名：「DL PE 256_16」
    
    lines = []
    lines.append(f"mvb2/n512 $d{x+0} $lc0")
    raise NotImplementedError("Please implement the VSM code!!")

    return lines
    