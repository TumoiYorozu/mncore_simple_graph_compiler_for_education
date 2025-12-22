from typing import List
from .. import BaseOperator

def generate_vsm(operator: BaseOperator) -> List[str]:
    assert operator.layout_in() == "((4_L2B:2, 64:8), (8:1, 2_W:1))", operator.layout_in()
    assert operator.layout_out() == "((4_L2B:2, 64:2), (2:1, 4_PE:1, 2_W:1))", operator.layout_out()
    
    assert operator.loc_prefix_out() == "m" # LM0 に出力を仮定し、実装を軽減
    
    x = operator.addr_in()      # "$d0" の "0" の部分
    y = operator.addr_out()     # "$lm0v" の "0" の部分

    # test unit_tests/train_step/DL_256x16_PE_*
    # 問題名：「DL PE 256_16」
    
    lines = []
    lines.append(f"mvb2/n512 $d{x+0} $lc0")
    lines.append(f"l2bmb $lc0 $lb0")
    lines.append(f"l2bmb $lc64 $lb64")
    lines.append(f"l2bmb $lc128 $lb128")
    lines.append(f"l2bmb $lc192 $lb192")
    lines.append(f"l2bmb $lc256 $lb256")
    lines.append(f"l2bmb $lc320 $lb320")
    lines.append(f"l2bmb $lc384 $lb384")
    lines.append(f"l2bmb $lc448 $lb448")
    lines.append(f"l1bmm $llb0 $llm{y+0}v")
    lines.append(f"l1bmm $llb32 $llm{y+16}v")
    lines.append(f"l1bmm $llb64 $llm{y+32}v")
    lines.append(f"l1bmm $llb96 $llm{y+48}v")
    lines.append(f"l1bmm $llb128 $llm{y+64}v")
    lines.append(f"l1bmm $llb160 $llm{y+80}v")
    lines.append(f"l1bmm $llb192 $llm{y+96}v")
    lines.append(f"l1bmm $llb224 $llm{y+112}v")
    lines.append(f"l1bmm $llb256 $llm{y+128}v")
    lines.append(f"l1bmm $llb288 $llm{y+144}v")
    lines.append(f"l1bmm $llb320 $llm{y+160}v")
    lines.append(f"l1bmm $llb352 $llm{y+176}v")
    lines.append(f"l1bmm $llb384 $llm{y+192}v")
    lines.append(f"l1bmm $llb416 $llm{y+208}v")
    lines.append(f"l1bmm $llb448 $llm{y+224}v")
    lines.append(f"l1bmm $llb480 $llm{y+240}v")

    return lines
    