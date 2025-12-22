from typing import List
from .. import BaseOperator

def generate_vsm(operator: BaseOperator) -> List[str]:
    assert operator.layout_in() == "((4_L2B:2, 4:512), (512:1, 2_W:1))", operator.layout_in()
    assert operator.layout_out() == "((16:1), (8_MAB:2, 8_L1B:1, 2_MAB:1, 4_PE:1, 2_W:1))", operator.layout_out()
    
    assert operator.loc_prefix_out() == "m" # LM0 に出力を仮定し、実装を軽減
    
    x = operator.addr_in()      # "$d0" の "0" の部分
    y = operator.addr_out()     # "$lm0v" の "0" の部分

    # test unit_tests/train_step/DL_16x1024*
    # 問題名：「DL 16_1024」
    
    lines = []
    lines.append(f"mvp/n2048 $d{x+0}@0 $p0@0")
    raise NotImplementedError("Please implement the VSM code!!")

    return lines