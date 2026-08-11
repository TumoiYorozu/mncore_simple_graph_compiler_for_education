from typing import List
from .. import BaseOperator

def generate_vsm(operator: BaseOperator) -> List[str]:
    assert operator.layout_in() == "((16:1), (8_MAB:2, 8_L1B:1, 2_MAB:1, 4_PE:1, 2_W:1))", operator.layout_in()
    assert operator.layout_out() == "((4_L2B:2, 4:512), (512:1, 2_W:1))", operator.layout_out()
    
    assert operator.loc_prefix_in() == "n" # LM1 に入力を仮定し、実装を軽減
    assert operator.addr_in() == 0         # addr = 0 に入力を仮定し、実装を軽減
                                           # ↑、つまり、入力が "$ln0v" だと仮定している

    y = operator.addr_out()      # "$d0" の "0" の部分
    
    # test unit_tests/train_step/UL_16x1024_*
    # 問題名：「UL 16_1024」

    lines = []
    for i in range(4):
        lines.append(f"l1bmd $ln{i*8}v $lb{i*256}")
    lines.append('nop')
    lines.append('nop')
    for i in range(32):
        lines.append(f"l2bmd $lb{i*32}v $lc{i*256}")
    lines.append('nop')
    for i in range(4):
        lines.append(f"mvp/n2048 $lc{i*2048}v $d{y}@{i}")

    return lines
