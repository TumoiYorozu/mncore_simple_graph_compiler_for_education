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
    lines.append(f"dmread $lx0 $lr0v")
    lines.append(f"dmwrite $ln8v $lx0")
    lines.append(f"dmread $lx0 $lr8v")
    lines.append(f"dmwrite $ln16v $lx0")
    lines.append(f"dmread $lx0 $lr16v")
    lines.append(f"dmwrite $ln24v $lx0")
    lines.append(f"dmread $lx0 $lr24v")
    lines.append(f"dmwrite $ln32v $lx0")
    lines.append(f"dmread $lx0 $lr32v")
    lines.append(f"dmwrite $ln40v $lx0")
    lines.append(f"dmread $lx0 $lr40v")
    lines.append(f"dmwrite $ln48v $lx0")
    lines.append(f"dmread $lx0 $lr48v")
    lines.append(f"dmwrite $ln56v $lx0")
    lines.append(f"dmread $lx0 $lr56v")
    lines.append(f"l1bmm@0 $lr0v $lb0")
    lines.append(f"l1bmm@0 $lr8v $lb4")
    lines.append(f"l1bmm@0 $lr16v $lb8")
    lines.append(f"l1bmm@0 $lr24v $lb12")
    lines.append(f"l1bmm@0 $lr32v $lb16")
    lines.append(f"l1bmm@0 $lr40v $lb20")
    lines.append(f"l1bmm@0 $lr48v $lb24")
    lines.append(f"l1bmm@0 $lr56v $lb28")
    lines.append(f"")
    lines.append(f"nop/2")
    lines.append(f"l2bm@0 $lb0 $lc0")
    lines.append(f"nop")
    lines.append(f"mvp/n64 $lc0@.0 $d{y+0}")
    return lines