from typing import List
from .. import BaseOperator

def generate_vsm(operator: BaseOperator) -> List[str]:
    assert operator.layout_in() == "((4_L2B:2, 4:512), (512:1, 2_W:1))", operator.layout_in()
    assert operator.layout_out() == "((16:1), (8_MAB:2, 8_L1B:1, 2_MAB:1, 4_PE:1, 2_W:1))", operator.layout_out()
    
    assert operator.loc_prefix_out() == "m" # LM0 に出力を仮定し、実装を軽減
    
    x = operator.addr_in()      # "$d0" の "0" の部分
    y = operator.addr_out()     # "$lm0v" の "0" の部分

    # test unit_tests/train_step/DL_16x1024*
    # 問題名：「DL 16_1024」
    
    lines = []
    lines.append(f"mvp/n2048 $d{x+0}@0 $p0@0")
    lines.append(f"mvp/n2048 $d{x+0}@1 $p2048@0")
    lines.append(f"mvp/n2048 $d{x+0}@2 $p4096@0")
    lines.append(f"mvp/n2048 $d{x+0}@3 $p6144@0")
    lines.append(f"mvb/n8192 $p0@0 $lc0")
    lines.append(f"l2bmd $lc0 $lb0")
    lines.append(f"l2bmd $lc256 $lb32")
    lines.append(f"l2bmd $lc512 $lb64")
    lines.append(f"l2bmd $lc768 $lb96")
    lines.append(f"l2bmd $lc1024 $lb128")
    lines.append(f"l2bmd $lc1280 $lb160")
    lines.append(f"l2bmd $lc1536 $lb192")
    lines.append(f"l2bmd $lc1792 $lb224")
    lines.append(f"l2bmd $lc2048 $lb256")
    lines.append(f"l2bmd $lc2304 $lb288")
    lines.append(f"l2bmd $lc2560 $lb320")
    lines.append(f"l2bmd $lc2816 $lb352")
    lines.append(f"l2bmd $lc3072 $lb384")
    lines.append(f"l2bmd $lc3328 $lb416")
    lines.append(f"l2bmd $lc3584 $lb448")
    lines.append(f"l2bmd $lc3840 $lb480")
    lines.append(f"l2bmd $lc4096 $lb512")
    lines.append(f"l2bmd $lc4352 $lb544")
    lines.append(f"l2bmd $lc4608 $lb576")
    lines.append(f"l2bmd $lc4864 $lb608")
    lines.append(f"l2bmd $lc5120 $lb640")
    lines.append(f"l2bmd $lc5376 $lb672")
    lines.append(f"l2bmd $lc5632 $lb704")
    lines.append(f"l2bmd $lc5888 $lb736")
    lines.append(f"l2bmd $lc6144 $lb768")
    lines.append(f"l2bmd $lc6400 $lb800")
    lines.append(f"l2bmd $lc6656 $lb832")
    lines.append(f"l2bmd $lc6912 $lb864")
    lines.append(f"l2bmd $lc7168 $lb896")
    lines.append(f"l2bmd $lc7424 $lb928")
    lines.append(f"l2bmd $lc7680 $lb960")
    lines.append(f"l2bmd $lc7936 $lb992")
    lines.append(f"l1bmd $lb0 $lm{0+y}v")
    lines.append(f"l1bmd $lb256 $lm{8+y}v")
    lines.append(f"l1bmd $lb512 $lm{16+y}v")
    lines.append(f"l1bmd $lb768 $lm{24+y}v")

    return lines