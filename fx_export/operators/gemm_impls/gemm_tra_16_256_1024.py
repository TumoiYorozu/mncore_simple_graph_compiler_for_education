from typing import List
from .. import BaseOperator



def generate_vsm(operator: BaseOperator) -> List[str]:
    operator.check_layout_in("((8_L2B:1, 32:2), (2:1, 4_PE:1, 2_W:1))", 0)
    operator.check_layout_in("((8_L2B:1, 32:1), (8_MAB:2, 8_L1B:1, 2_MAB:1, 4_PE:1, 2_W:1))", 1)
    operator.check_layout_out("((4_L2B:2, 4:512), (512:1, 2_W:1))")
    assert operator.loc_prefix_out() == "d" # DRAM 出力


    assert operator.loc_prefix_in(0) == "m" # LM0 に入力を仮定し、実装を軽減
    assert operator.loc_prefix_in(1) == "m" # LM0 に入力を仮定し、実装を軽減
    
    assert operator.addr_in(0) == 0   # 入力1 を addr0 に仮定し、実装を軽減
                                           # ↑、つまり、入力 0 が "$lm0v" だと仮定している
    assert operator.addr_in(1) == operator.memory_len_in(0)
                                            # 入力2 を、入力 0 の直後と仮定し、実装を軽減


    c = operator.addr_out()      # "$d0" の "0" の部分

    # test unit_tests/train_step/Gemm_256x16_256x16_16x16_transA_a/
    # 問題名： 「Mmul TA D 16_256_1024」
    
    lines = []
    lines.append(f"gmwrite $lm0v4 $ly0")
    raise NotImplementedError("Please implement the VSM code!!")

    return lines