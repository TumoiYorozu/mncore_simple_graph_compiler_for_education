from typing import List
from .. import BaseOperator

def generate_vsm(operator: BaseOperator) -> List[str]:
    # assert operator.layout_in() == "((4_L2B:2, 64:8), (8:1, 2_W:1))", operator.layout_in()
    # assert operator.layout_out() == "((8_L2B:1, 4:2, 8_L1B:1), (2:1, 4_PE:1, 2_W:1))", operator.layout_out()
    
    assert operator.loc_prefix_out() == "m" # LM0 に出力を仮定し、実装を軽減
    
    
    x = operator.addr_in()      # "$d0" の "0" の部分
    y = operator.addr_out()     # "$lm0v" の "0" の部分

    # test unit_tests/train_step/DL_256x1024_*
    # 問題名：「DL 256_1024」
    
    lines = []
    lines.append(f"mvp/n16384 $d{x+0} $lc0@.0")
    lines.append(f"mvp/n16384 $d{x+16384} $lc0@.1")
    for i in range(64):
        lines.append(f"l2bmd $lc{i*256} $lb{i*32}")
    for i in range(8):
        lines.append(f"l1bmd $lb{i*256} $lm{y+i*8}v")

    return lines
