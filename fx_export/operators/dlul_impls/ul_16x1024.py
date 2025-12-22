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
    lines.append(f"l1bmd $ln0v  $lb0")
    lines.append(f"l1bmd $ln8v  $lb256")
    lines.append(f"l1bmd $ln16v $lb512")
    lines.append(f"l1bmd $ln24v $lb768")
    lines.append(f"")
    lines.append(f"nop/2")
    lines.append(f"")
    lines.append(f"l2bmd $lb0   $lc0")
    lines.append(f"l2bmd $lb32  $lc256")
    lines.append(f"l2bmd $lb64  $lc512")
    lines.append(f"l2bmd $lb96  $lc768")
    lines.append(f"l2bmd $lb128 $lc1024")
    lines.append(f"l2bmd $lb160 $lc1280")
    lines.append(f"l2bmd $lb192 $lc1536")
    lines.append(f"l2bmd $lb224 $lc1792")
    lines.append(f"l2bmd $lb256 $lc2048")
    lines.append(f"l2bmd $lb288 $lc2304")
    lines.append(f"l2bmd $lb320 $lc2560")
    lines.append(f"l2bmd $lb352 $lc2816")
    lines.append(f"l2bmd $lb384 $lc3072")
    lines.append(f"l2bmd $lb416 $lc3328")
    lines.append(f"l2bmd $lb448 $lc3584")
    lines.append(f"l2bmd $lb480 $lc3840")
    lines.append(f"l2bmd $lb512 $lc4096")
    lines.append(f"l2bmd $lb544 $lc4352")
    lines.append(f"l2bmd $lb576 $lc4608")
    lines.append(f"l2bmd $lb608 $lc4864")
    lines.append(f"l2bmd $lb640 $lc5120")
    lines.append(f"l2bmd $lb672 $lc5376")
    lines.append(f"l2bmd $lb704 $lc5632")
    lines.append(f"l2bmd $lb736 $lc5888")
    lines.append(f"l2bmd $lb768 $lc6144")
    lines.append(f"l2bmd $lb800 $lc6400")
    lines.append(f"l2bmd $lb832 $lc6656")
    lines.append(f"l2bmd $lb864 $lc6912")
    lines.append(f"l2bmd $lb896 $lc7168")
    lines.append(f"l2bmd $lb928 $lc7424")
    lines.append(f"l2bmd $lb960 $lc7680")
    lines.append(f"l2bmd $lb992 $lc7936")
    lines.append(f"")
    lines.append(f"nop")
    lines.append(f"mvp/n2048 $lc0@0.0    $d{y+0}@0")
    lines.append(f"mvp/n2048 $lc2048@0.0 $d{y+0}@1")
    lines.append(f"mvp/n2048 $lc4096@0.0 $d{y+0}@2")
    lines.append(f"mvp/n2048 $lc6144@0.0 $d{y+0}@3")

    return lines