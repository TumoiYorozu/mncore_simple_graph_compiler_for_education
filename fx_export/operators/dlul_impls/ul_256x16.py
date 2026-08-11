from typing import List
from .. import BaseOperator

def generate_vsm(operator: BaseOperator) -> List[str]:
    assert operator.layout_in() == "((8_L2B:1, 4:2, 8_L1B:1), (2:1, 4_PE:1, 2_W:1))", operator.layout_in()
    assert operator.layout_out() == "((4_L2B:2, 64:8), (8:1, 2_W:1))", operator.layout_out()
    
    assert operator.loc_prefix_in() == "n" # LM1 に入力を仮定し、実装を軽減
    assert operator.addr_in() == 0         # addr = 0 に入力を仮定し、実装を軽減
                                           # ↑、つまり、入力が "$ln0v" だと仮定している

    y = operator.addr_out()      # "$d0" の "0" の部分
    
    # test unit_tests/train_step/UL_256x16_default_*
    # 問題名：「UL 256_16」
    
    lines = []
    lines.append('l1bmm@0 $lln0v $llb0')
    lines.append('nop')
    lines.append('nop')
    lines.append('l2bmd $lb0 $lc0')
    lines.append(f'mvp/n256 $lc0@.0 $d{y+0}')
    lines.append(f'mvp/n256 $lc0@.1 $d{y+256}')

    return lines
