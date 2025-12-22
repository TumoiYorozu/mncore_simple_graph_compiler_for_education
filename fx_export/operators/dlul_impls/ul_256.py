from typing import List
from .. import BaseOperator

def generate_vsm(operator: BaseOperator) -> List[str]:
    assert operator.layout_in() == "((4_L2B:2, 32:1, 2_W:1))", operator.layout_in()
    assert operator.layout_out() == "((4_L2B:2, 32:1, 2_W:1))", operator.layout_out()
    
    assert operator.loc_prefix_in() == "n" # LM1 に入力を仮定し、実装を軽減
    assert operator.addr_in() == 0         # addr = 0 に入力を仮定し、実装を軽減
                                           # ↑、つまり、入力が "$ln0v" だと仮定している

    y = operator.addr_out()      # "$d0" の "0" の部分
    
    # test unit_tests/train_step/UL_256_*
    # 問題名：「UL 256」
    
    lines = []
    lines.append(f"dmwrite $ln0v $lx0")
    raise NotImplementedError("Please implement the VSM code!!")

    return lines