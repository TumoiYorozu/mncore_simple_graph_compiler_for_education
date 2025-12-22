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
    lines.append(f"mvp/n256 $d{x+256} $lc0@.1")
    lines.append(f"")
    lines.append(f"l2bmb $lc0  $lb0  ")
    lines.append(f"l2bmb $lc64  $lb64 ")
    lines.append(f"l2bmb $lc128  $lb128")
    lines.append(f"l2bmb $lc192  $lb192")
    lines.append(f"")
    lines.append(f"l1bmm $llb0   $llm{y+0}v")
    lines.append(f"l1bmm $llb32  $llm{y+16}v")
    lines.append(f"l1bmm $llb64  $llm{y+32}v")
    lines.append(f"l1bmm $llb96  $llm{y+48}v")
    lines.append(f"l1bmm $llb128 $llm{y+64}v")
    lines.append(f"l1bmm $llb160 $llm{y+80}v")
    lines.append(f"l1bmm $llb192 $llm{y+96}v")
    lines.append(f"l1bmm $llb224 $llm{y+112}v")

    return lines
    