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

    for i in range(8):
        lines.append(f'dmwrite $ln{i*8}v $lx0')
        lines.append('dmread $lx0 $lr0v')
        lines.append('nop')
        lines.append(f'l1bmm@0 $lr0v $lb{i*4}')

    lines.append('nop')
    lines.append('nop')
    lines.append('nop')
    lines.append('l2bm@0 $lb0 $lc0')
    lines.append('nop')
    lines.append(f'mvp/n64 $lc0@.0 $d{y}')

    return lines
