from typing import List
from .. import BaseOperator

def generate_vsm(operator: BaseOperator) -> List[str]:
    assert operator.layout_in() == "((4_L2B:2, 64:8), (8:1, 2_W:1))", operator.layout_in()
    assert operator.layout_out() == "((8_L2B:1, 32:2), (2:1, 4_PE:1, 2_W:1))", operator.layout_out()
    
    assert operator.loc_prefix_out() == "m" # LM0 に出力を仮定し、実装を軽減
    
    x = operator.addr_in()      # "$d0" の "0" の部分
    y = operator.addr_out()     # "$lm0v" の "0" の部分

    # test unit_tests/train_step/DL_256x16_L2B_*
    # 問題名：「DL L2B 256_16」

    lines = []
    lines.append(f"mvp/n256 $d{x+0}   $lc0@.0")
    lines.append(f"mvp/n256 $d{x+256}   $lc0@.1")
    for i in range(8):
        lines.append(f"l2bmb $lc{i*64} $lb{i*64}")
    lines.append("nop")
    for i in range(16):
        lines.append(f"l1bmm $llb{i*32} $llm{y+i*16}v")

    return lines
    
