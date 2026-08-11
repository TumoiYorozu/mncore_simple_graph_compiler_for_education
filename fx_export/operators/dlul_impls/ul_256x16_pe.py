from typing import List
from .. import BaseOperator

def generate_vsm(operator: BaseOperator) -> List[str]:
    assert operator.layout_in() == "((4_L2B:2, 64:2), (2:1, 4_PE:1, 2_W:1))", operator.layout_in()
    assert operator.layout_out() == "((4_L2B:2, 64:8), (8:1, 2_W:1))", operator.layout_out()
    
    assert operator.loc_prefix_in() == "n" # LM1 に入力を仮定し、実装を軽減
    assert operator.addr_in() == 0         # addr = 0 に入力を仮定し、実装を軽減
                                           # ↑、つまり、入力が "$ln0v" だと仮定している

    y = operator.addr_out()      # "$d0" の "0" の部分
    
    # test unit_tests/train_step/UL_256x16_PE_*
    # 問題名：「UL PE 256_16」
    
    lines = []
    for i in range(16):
        lines.append(f'l1bmm@0 $lln{i*16}v $llb{i*32}')
    lines.append('nop')
    for i in range(8):
        lines.append(f'l2bm@0 $lb{i*64} $lc{i*64}')
    lines.append('nop')
    lines.append(f'mvp/n512 $lc0@.0 $d{y}')

    return lines
