#!/usr/bin/env python3

from typing import List, Dict, Any, Optional
from . import BaseOperator, register_operator

@register_operator("Div")
class DivOperator(BaseOperator):
    """除算演算オペレーター"""
    
    def generate_vsm(self) -> List[str]:
        lines = []
        shape0 = self.in_shape(0)
        shape1 = self.in_shape(1)
        
        if len(shape0) == 2 and len(shape1) == 1 and shape0[0] == shape1[0]:
            # ブロードキャスト除算（ベクトルで各行を除算） DivRow
            self.check_layout_in("((4_L2B:2, 64:2), (2:1, 4_PE:1, 2_W:1))", 0)
            self.check_layout_in("((4_L2B:2, 32:1, 2_W:1))", 1)
            self.check_layout_out("((4_L2B:2, 64:2), (2:1, 4_PE:1, 2_W:1))")
            
            assert self.loc_prefix_in(0)  == "m" # LM0 に入力0を仮定し、実装を軽減
            assert self.loc_prefix_in(1)  == "m" # LM0 に入力1を仮定し、実装を軽減
            assert self.addr_in(0)        ==  0  # Addr 0 に入力を仮定し、実装を軽減
                                                       # ↑、つまり、入力0が "$lm0v" だと仮定している
            
            assert self.loc_prefix_out(0) == "n" # LM1 に出力を仮定し、実装を軽減
            assert self.addr_in(1)        ==  self.memory_len_in(0)  # IN0 の次に入力を仮定し、実装を軽減
            assert self.addr_out(0)       ==  0  # Addr 0 に出力を仮定し、実装を軽減
                                                       # ↑、つまり、出力が "$ln0v" だと仮定している

            # test unit_tests/train_step/Div_*
            # 問題名：「DivRow vec」
            
            lines.append(f'imm f"2.0" $lt')
            lines.append(f"frsqrt $lm256v $lr0v")
            lines.append(f"frsqrt $lm264v $lr8v")
            lines.append(f"frsqrt $lm272v $lr16v")
            lines.append(f"frsqrt $lm280v $lr24v")
            lines.append(f"frsqrt $lm288v $lr32v")
            lines.append(f"frsqrt $lm296v $lr40v")
            lines.append(f"frsqrt $lm304v $lr48v")
            lines.append(f"frsqrt $lm312v $lr56v")
            lines.append(f"fvmul $lr0v $lr0v $lr0v")
            lines.append(f"fvmul $lr8v $lr8v $lr8v")
            lines.append(f"fvmul $lr16v $lr16v $lr16v")
            lines.append(f"fvmul $lr24v $lr24v $lr24v")
            lines.append(f"fvmul $lr32v $lr32v $lr32v")
            lines.append(f"fvmul $lr40v $lr40v $lr40v")
            lines.append(f"fvmul $lr48v $lr48v $lr48v")
            lines.append(f"fvmul $lr56v $lr56v $lr56v")
            lines.append(f"fvfma $lm256v -$lr0v $t $nowrite")
            lines.append(f"fvmul $mauf $lr0v $lr0v")
            lines.append(f"fvfma $lm264v -$lr8v $t $nowrite")
            lines.append(f"fvmul $mauf $lr8v $lr8v")
            lines.append(f"fvfma $lm272v -$lr16v $t $nowrite")
            lines.append(f"fvmul $mauf $lr16v $lr16v")
            lines.append(f"fvfma $lm280v -$lr24v $t $nowrite")
            lines.append(f"fvmul $mauf $lr24v $lr24v")
            lines.append(f"fvfma $lm288v -$lr32v $t $nowrite")
            lines.append(f"fvmul $mauf $lr32v $lr32v")
            lines.append(f"fvfma $lm296v -$lr40v $t $nowrite")
            lines.append(f"fvmul $mauf $lr40v $lr40v")
            lines.append(f"fvfma $lm304v -$lr48v $t $nowrite")
            lines.append(f"fvmul $mauf $lr48v $lr48v")
            lines.append(f"fvfma $lm312v -$lr56v $t $nowrite")
            lines.append(f"fvmul $mauf $lr56v $lr56v")
            lines.append(f"fvfma $lm256v -$lr0v $t $nowrite")
            lines.append(f"fvmul $mauf $lr0v $lr0v")
            lines.append(f"fvfma $lm264v -$lr8v $t $nowrite")
            lines.append(f"fvmul $mauf $lr8v $lr8v")
            lines.append(f"fvfma $lm272v -$lr16v $t $nowrite")
            lines.append(f"fvmul $mauf $lr16v $lr16v")
            lines.append(f"fvfma $lm280v -$lr24v $t $nowrite")
            lines.append(f"fvmul $mauf $lr24v $lr24v")
            lines.append(f"fvfma $lm288v -$lr32v $t $nowrite")
            lines.append(f"fvmul $mauf $lr32v $lr32v")
            lines.append(f"fvfma $lm296v -$lr40v $t $nowrite")
            lines.append(f"fvmul $mauf $lr40v $lr40v")
            lines.append(f"fvfma $lm304v -$lr48v $t $nowrite")
            lines.append(f"fvmul $mauf $lr48v $lr48v")
            lines.append(f"fvfma $lm312v -$lr56v $t $nowrite")
            lines.append(f"fvmul $mauf $lr56v $lr56v")
            lines.append(f"fvmul $m0v $r0 $n0v")
            lines.append(f"fvmul $m4v $r1 $n4v")
            lines.append(f"fvmul $m8v $r2 $n8v")
            lines.append(f"fvmul $m12v $r3 $n12v")
            lines.append(f"fvmul $m16v $r4 $n16v")
            lines.append(f"fvmul $m20v $r5 $n20v")
            lines.append(f"fvmul $m24v $r6 $n24v")
            lines.append(f"fvmul $m28v $r7 $n28v")
            lines.append(f"fvmul $m32v $r8 $n32v")
            lines.append(f"fvmul $m36v $r9 $n36v")
            lines.append(f"fvmul $m40v $r10 $n40v")
            lines.append(f"fvmul $m44v $r11 $n44v")
            lines.append(f"fvmul $m48v $r12 $n48v")
            lines.append(f"fvmul $m52v $r13 $n52v")
            lines.append(f"fvmul $m56v $r14 $n56v")
            lines.append(f"fvmul $m60v $r15 $n60v")
            lines.append(f"fvmul $m64v $r16 $n64v")
            lines.append(f"fvmul $m68v $r17 $n68v")
            lines.append(f"fvmul $m72v $r18 $n72v")
            lines.append(f"fvmul $m76v $r19 $n76v")
            lines.append(f"fvmul $m80v $r20 $n80v")
            lines.append(f"fvmul $m84v $r21 $n84v")
            lines.append(f"fvmul $m88v $r22 $n88v")
            lines.append(f"fvmul $m92v $r23 $n92v")
            lines.append(f"fvmul $m96v $r24 $n96v")
            lines.append(f"fvmul $m100v $r25 $n100v")
            lines.append(f"fvmul $m104v $r26 $n104v")
            lines.append(f"fvmul $m108v $r27 $n108v")
            lines.append(f"fvmul $m112v $r28 $n112v")
            lines.append(f"fvmul $m116v $r29 $n116v")
            lines.append(f"fvmul $m120v $r30 $n120v")
            lines.append(f"fvmul $m124v $r31 $n124v")
            lines.append(f"fvmul $m128v $r32 $n128v")
            lines.append(f"fvmul $m132v $r33 $n132v")
            lines.append(f"fvmul $m136v $r34 $n136v")
            lines.append(f"fvmul $m140v $r35 $n140v")
            lines.append(f"fvmul $m144v $r36 $n144v")
            lines.append(f"fvmul $m148v $r37 $n148v")
            lines.append(f"fvmul $m152v $r38 $n152v")
            lines.append(f"fvmul $m156v $r39 $n156v")
            lines.append(f"fvmul $m160v $r40 $n160v")
            lines.append(f"fvmul $m164v $r41 $n164v")
            lines.append(f"fvmul $m168v $r42 $n168v")
            lines.append(f"fvmul $m172v $r43 $n172v")
            lines.append(f"fvmul $m176v $r44 $n176v")
            lines.append(f"fvmul $m180v $r45 $n180v")
            lines.append(f"fvmul $m184v $r46 $n184v")
            lines.append(f"fvmul $m188v $r47 $n188v")
            lines.append(f"fvmul $m192v $r48 $n192v")
            lines.append(f"fvmul $m196v $r49 $n196v")
            lines.append(f"fvmul $m200v $r50 $n200v")
            lines.append(f"fvmul $m204v $r51 $n204v")
            lines.append(f"fvmul $m208v $r52 $n208v")
            lines.append(f"fvmul $m212v $r53 $n212v")
            lines.append(f"fvmul $m216v $r54 $n216v")
            lines.append(f"fvmul $m220v $r55 $n220v")
            lines.append(f"fvmul $m224v $r56 $n224v")
            lines.append(f"fvmul $m228v $r57 $n228v")
            lines.append(f"fvmul $m232v $r58 $n232v")
            lines.append(f"fvmul $m236v $r59 $n236v")
            lines.append(f"fvmul $m240v $r60 $n240v")
            lines.append(f"fvmul $m244v $r61 $n244v")
            lines.append(f"fvmul $m248v $r62 $n248v")
            lines.append(f"fvmul $m252v $r63 $n252v")
            return lines
        raise NotImplementedError(f"Div: shapes {shape0} / {shape1}")
        
    def testcase_hint(self) -> Optional[str]:
        return "div_row.vsm"

    def get_memory_layout_tag(self) -> Dict[str, List[str]]:
        shape0 = self.in_shape(0)
        shape1 = self.in_shape(1)
        if shape0 == [256, 16] and shape1 == [256]:
            return {
                "inputs": ["PE", "default"],
                "outputs": ["PE"]
            }
        raise NotImplementedError(f"Div: shapes {shape0} / {shape1}")
    
        
    def generate_cpp(self) -> List[str]:
        """C++コード生成"""
        lines = []
        
        assert len(self.inputs) >= 2, f"Div node requires 2 inputs, got {len(self.inputs)}"
        
        a = self.inputs[0]
        b = self.inputs[1]
        out = self.outputs[0]
        out_var = self.get_output_var_name()
        
        a_var = self.get_mapped_var(a)
        b_var = self.get_mapped_var(b)
        
        # 形状を取得
        shape0 = self.in_shape(0)
        shape1 = self.in_shape(1)
        
        if len(shape0) == 2 and len(shape1) == 1 and shape0[0] == shape1[0]:
            # ブロードキャスト除算（ベクトルで各行を除算）
            lines.append(f"    const Matrix<{shape0[0]}, {shape0[1]}> {out_var} = div_rowvec<{shape0[0]}, {shape0[1]}>({a_var}, {b_var});")
        else:
            raise NotImplementedError(f"Div: shapes {shape0} / {shape1}")
        
        self.variable_map[out] = out_var
        
        return lines
    
    def generate_python(self) -> List[str]:
        """Pythonコード生成"""
        lines = []
        
        a = self.inputs[0]
        b = self.inputs[1]
        out = self.outputs[0]
        out_var = self.get_output_var_name()
        
        shape0 = self.in_shape(0)
        shape1 = self.in_shape(1)
        if len(shape0) == 2 and len(shape1) == 1 and shape0[0] == shape1[0]:
            lines.append(f"    {out_var} = {self.get_mapped_var(a)} / {self.get_mapped_var(b)}.unsqueeze(-1)")
        else:
            raise NotImplementedError(f"Div: shapes {shape0} / {shape1}")
        
        self.variable_map[out] = out_var
        
        return lines
