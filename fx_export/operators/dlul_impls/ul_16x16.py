from typing import List
from .. import BaseOperator

def generate_vsm(operator: BaseOperator) -> List[str]:
    assert operator.layout_in() == "((16:2), (2:1, 4_PE:1, 2_W:1))", operator.layout_in()
    assert operator.layout_out() == "((4_L2B:2, 4:8), (8:1, 2_W:1))", operator.layout_out()
    
    assert operator.loc_prefix_in() == "n" # LM1 に入力を仮定し、実装を軽減
    assert operator.addr_in() == 0         # addr = 0 に入力を仮定し、実装を軽減
                                           # ↑、つまり、入力が "$ln0v" だと仮定している

    y = operator.addr_out()      # "$d0" の "0" の部分
    
    # test unit_tests/train_step/UL_16x16_*
    # 問題名：「UL 16_16」
    
    lines = []
    lines.append(f"l1bmm@0 $ln0v $lb0")
    raise NotImplementedError("Please implement the VSM code!!")

    return lines