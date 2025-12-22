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
    lines.append(f"l1bmm@0 $lln0v $llb0")
    lines.append(f"l1bmm@0 $lln16v $llb32")
    lines.append(f"l1bmm@0 $lln32v $llb64")
    lines.append(f"l1bmm@0 $lln48v $llb96")
    lines.append(f"l1bmm@0 $lln64v $llb128")
    lines.append(f"l1bmm@0 $lln80v $llb160")
    lines.append(f"l1bmm@0 $lln96v $llb192")
    lines.append(f"l1bmm@0 $lln112v $llb224")
    lines.append(f"l1bmm@0 $lln128v $llb256")
    lines.append(f"l1bmm@0 $lln144v $llb288")
    lines.append(f"l1bmm@0 $lln160v $llb320")
    lines.append(f"l1bmm@0 $lln176v $llb352")
    lines.append(f"l1bmm@0 $lln192v $llb384")
    lines.append(f"l1bmm@0 $lln208v $llb416")
    lines.append(f"l1bmm@0 $lln224v $llb448")
    lines.append(f"l1bmm@0 $lln240v $llb480")

    lines.append(f"l2bm@0 $lb0 $lc0")
    lines.append(f"l2bm@0 $lb64 $lc64")
    lines.append(f"l2bm@0 $lb128 $lc128")
    lines.append(f"l2bm@0 $lb192 $lc192")
    lines.append(f"l2bm@0 $lb256 $lc256")
    lines.append(f"l2bm@0 $lb320 $lc320")
    lines.append(f"l2bm@0 $lb384 $lc384")
    lines.append(f"l2bm@0 $lb448 $lc448")

    lines.append(f"nop")

    lines.append(f"mvp/n512 $lc0@.0 $d{y+0}")



    return lines