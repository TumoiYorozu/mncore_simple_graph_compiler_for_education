from typing import List
from .. import BaseOperator

def generate_vsm(operator: BaseOperator) -> List[str]:
    assert operator.layout_in() == "((4_L2B:2, 4:8), (8:1, 2_W:1))", operator.layout_in()
    assert operator.layout_out() == "((16:2), (2:1, 4_PE:1, 2_W:1))", operator.layout_out()
    
    assert operator.loc_prefix_out() == "m" # LM0 に出力を仮定し、実装を軽減
    
    x = operator.addr_in()      # "$d0" の "0" の部分
    y = operator.addr_out()     # "$lm0v" の "0" の部分

    # test unit_tests/train_step/DL_16x16*
    # 問題名：「DL 16_16」
    
    lines = []

    lines.append(f"mvp/n64 $d{x}@0 $p0@0")
    lines.append(f"mvp/n64 $d{x}@1 $p64@0")
    lines.append(f"mvp/n64 $d{x}@2 $p128@0")
    lines.append(f"mvp/n64 $d{x}@3 $p192@0")
    lines.append(f"mvb/n256 $p0@0 $lc0")
    for i in range(4):
        lines.append(f"l2bmb $lc{i*64}  $lb{i*32}")
    for i in range(4):
        lines.append(f"l1bmm $llb{i*32} $llm{y+i*16}v")
    return lines
