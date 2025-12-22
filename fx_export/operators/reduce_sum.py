#!/usr/bin/env python3

from typing import List, Dict, Any, Optional
from . import BaseOperator
from . import register_operator

@register_operator("ReduceSum")
class ReduceSumOperator(BaseOperator):
    """ReduceSum演算オペレーター"""

    def generate_vsm(self) -> List[str]:
        # グラフから形状を取得
        shape = self.in_shape()
        lines = []
        
        axes = list(self.get_attr('axes').ints)
        
        # 形状によって処理を分岐
        if len(shape) == 2 and (axes == [0]): # sum_col（列方向の和）
            if shape == [256, 16]:
                self.check_layout_in("((8_L2B:1, 4:2, 8_L1B:1), (2:1, 4_PE:1, 2_W:1))")
                self.check_layout_out("((8:1, 2_W:1))")
                assert self.loc_prefix_out() == "d"
                
                assert self.loc_prefix_in()  == "m" # LM0 に入力を仮定し、実装を軽減
                assert self.addr_in()        ==  0  # Addr 0 に入力を仮定し、実装を軽減
                                                    # ↑、つまり、入力が "$lm0v" だと仮定している

                #  test unit_tests/train_step/ReduceSum_256x16_16_*
                # 問題名：「SumCol」
                
                x = self.addr_out()        # "$d0" の "0" の部分
                lines.append(f"fvpassa $m0v $nowrite")
                lines.append(f"fvadd $m4v $mauf $nowrite")
                lines.append(f"fvadd $m8v $mauf $nowrite")
                lines.append(f"fvadd $m12v $mauf $n0v")
                lines.append(f"nop")
                lines.append(f"nop")
                lines.append(f"l1bmm@0 $ln0v $lb0")
                lines.append(f"nop")
                lines.append(f"nop")
                lines.append(f"nop")
                lines.append(f"l2bmrffadd $lb0 $lc0")
                lines.append(f"nop")
                lines.append(f"mvrffadd/n64 $lc0 $d{x}")
                return lines
            
        if len(shape) == 2 and (axes == [-1]): # sum_row（行方向の和）
            if shape == [256, 16]:
                self.check_layout_in("((4_L2B:2, 64:2), (2:1, 4_PE:1, 2_W:1))")
                self.check_layout_out("((4_L2B:2, 32:1, 2_W:1))")
                
                assert self.loc_prefix_in()  == "m" # LM0 に入力を仮定し、実装を軽減
                assert self.addr_in()        ==  0  # Addr 0 に入力を仮定し、実装を軽減
                                                    # ↑、つまり、入力が "$lm0v" だと仮定している
                
                assert self.loc_prefix_out() == "n" # LM1 に出力を仮定し、実装を軽減
                assert self.addr_out()       ==  0  # Addr 0 に出力を仮定し、実装を軽減
                                                    # ↑、つまり、出力が "$ln0v" だと仮定している


                #  test unit_tests/train_step/ReduceSum_256x16_256_*
                # 問題名：「SumRow」
                

                lines.append(f'''
                    fvpassa $m0v4 $nowrite
                    fvadd $m1v4 $mauf $nowrite
                    fvadd $m2v4 $mauf $nowrite
                    fvadd $m3v4 $mauf $r0v
                    msr $mauf $s0v
                    fvadd $aluf $r0v $r0v
                    msr $s0v $s0v
                    fvadd $aluf $r0v $r0v
                    msr $s0v $s0v
                    fvadd $aluf $r0v $n0v
                    fvpassa $m16v4 $nowrite
                    fvadd $m17v4 $mauf $nowrite
                    fvadd $m18v4 $mauf $nowrite
                    fvadd $m19v4 $mauf $r0v
                    msr $mauf $s0v
                    fvadd $aluf $r0v $r0v
                    msr $s0v $s0v
                    fvadd $aluf $r0v $r0v
                    msr $s0v $s0v
                    fvadd $aluf $r0v $n4v
                    fvpassa $m32v4 $nowrite
                    fvadd $m33v4 $mauf $nowrite
                    fvadd $m34v4 $mauf $nowrite
                    fvadd $m35v4 $mauf $r0v
                    msr $mauf $s0v
                    fvadd $aluf $r0v $r0v
                    msr $s0v $s0v
                    fvadd $aluf $r0v $r0v
                    msr $s0v $s0v
                    fvadd $aluf $r0v $n8v
                    fvpassa $m48v4 $nowrite
                    fvadd $m49v4 $mauf $nowrite
                    fvadd $m50v4 $mauf $nowrite
                    fvadd $m51v4 $mauf $r0v
                    msr $mauf $s0v
                    fvadd $aluf $r0v $r0v
                    msr $s0v $s0v
                    fvadd $aluf $r0v $r0v
                    msr $s0v $s0v
                    fvadd $aluf $r0v $n12v
                    fvpassa $m64v4 $nowrite
                    fvadd $m65v4 $mauf $nowrite
                    fvadd $m66v4 $mauf $nowrite
                    fvadd $m67v4 $mauf $r0v
                    msr $mauf $s0v
                    fvadd $aluf $r0v $r0v
                    msr $s0v $s0v
                    fvadd $aluf $r0v $r0v
                    msr $s0v $s0v
                    fvadd $aluf $r0v $n16v
                    fvpassa $m80v4 $nowrite
                    fvadd $m81v4 $mauf $nowrite
                    fvadd $m82v4 $mauf $nowrite
                    fvadd $m83v4 $mauf $r0v
                    msr $mauf $s0v
                    fvadd $aluf $r0v $r0v
                    msr $s0v $s0v
                    fvadd $aluf $r0v $r0v
                    msr $s0v $s0v
                    fvadd $aluf $r0v $n20v
                    fvpassa $m96v4 $nowrite
                    fvadd $m97v4 $mauf $nowrite
                    fvadd $m98v4 $mauf $nowrite
                    fvadd $m99v4 $mauf $r0v
                    msr $mauf $s0v
                    fvadd $aluf $r0v $r0v
                    msr $s0v $s0v
                    fvadd $aluf $r0v $r0v
                    msr $s0v $s0v
                    fvadd $aluf $r0v $n24v
                    fvpassa $m112v4 $nowrite
                    fvadd $m113v4 $mauf $nowrite
                    fvadd $m114v4 $mauf $nowrite
                    fvadd $m115v4 $mauf $r0v
                    msr $mauf $s0v
                    fvadd $aluf $r0v $r0v
                    msr $s0v $s0v
                    fvadd $aluf $r0v $r0v
                    msr $s0v $s0v
                    fvadd $aluf $r0v $n28v
                    fvpassa $m128v4 $nowrite
                    fvadd $m129v4 $mauf $nowrite
                    fvadd $m130v4 $mauf $nowrite
                    fvadd $m131v4 $mauf $r0v
                    msr $mauf $s0v
                    fvadd $aluf $r0v $r0v
                    msr $s0v $s0v
                    fvadd $aluf $r0v $r0v
                    msr $s0v $s0v
                    fvadd $aluf $r0v $n32v
                    fvpassa $m144v4 $nowrite
                    fvadd $m145v4 $mauf $nowrite
                    fvadd $m146v4 $mauf $nowrite
                    fvadd $m147v4 $mauf $r0v
                    msr $mauf $s0v
                    fvadd $aluf $r0v $r0v
                    msr $s0v $s0v
                    fvadd $aluf $r0v $r0v
                    msr $s0v $s0v
                    fvadd $aluf $r0v $n36v
                    fvpassa $m160v4 $nowrite
                    fvadd $m161v4 $mauf $nowrite
                    fvadd $m162v4 $mauf $nowrite
                    fvadd $m163v4 $mauf $r0v
                    msr $mauf $s0v
                    fvadd $aluf $r0v $r0v
                    msr $s0v $s0v
                    fvadd $aluf $r0v $r0v
                    msr $s0v $s0v
                    fvadd $aluf $r0v $n40v
                    fvpassa $m176v4 $nowrite
                    fvadd $m177v4 $mauf $nowrite
                    fvadd $m178v4 $mauf $nowrite
                    fvadd $m179v4 $mauf $r0v
                    msr $mauf $s0v
                    fvadd $aluf $r0v $r0v
                    msr $s0v $s0v
                    fvadd $aluf $r0v $r0v
                    msr $s0v $s0v
                    fvadd $aluf $r0v $n44v
                    fvpassa $m192v4 $nowrite
                    fvadd $m193v4 $mauf $nowrite
                    fvadd $m194v4 $mauf $nowrite
                    fvadd $m195v4 $mauf $r0v
                    msr $mauf $s0v
                    fvadd $aluf $r0v $r0v
                    msr $s0v $s0v
                    fvadd $aluf $r0v $r0v
                    msr $s0v $s0v
                    fvadd $aluf $r0v $n48v
                    fvpassa $m208v4 $nowrite
                    fvadd $m209v4 $mauf $nowrite
                    fvadd $m210v4 $mauf $nowrite
                    fvadd $m211v4 $mauf $r0v
                    msr $mauf $s0v
                    fvadd $aluf $r0v $r0v
                    msr $s0v $s0v
                    fvadd $aluf $r0v $r0v
                    msr $s0v $s0v
                    fvadd $aluf $r0v $n52v
                    fvpassa $m224v4 $nowrite
                    fvadd $m225v4 $mauf $nowrite
                    fvadd $m226v4 $mauf $nowrite
                    fvadd $m227v4 $mauf $r0v
                    msr $mauf $s0v
                    fvadd $aluf $r0v $r0v
                    msr $s0v $s0v
                    fvadd $aluf $r0v $r0v
                    msr $s0v $s0v
                    fvadd $aluf $r0v $n56v
                    fvpassa $m240v4 $nowrite
                    fvadd $m241v4 $mauf $nowrite
                    fvadd $m242v4 $mauf $nowrite
                    fvadd $m243v4 $mauf $r0v
                    msr $mauf $s0v
                    fvadd $aluf $r0v $r0v
                    msr $s0v $s0v
                    fvadd $aluf $r0v $r0v
                    msr $s0v $s0v
                    fvadd $aluf $r0v $n60v
                ''')

                return lines
        
        raise NotImplementedError(f"ReduceSum: shape={shape}, axes={axes}")

    def testcase_hint(self) -> Optional[str]:
        return "sum_*.vsm"

    
    def get_memory_layout_tag(self) -> Dict[str, List[str]]:
        shape = self.in_shape(0)
        axes = list(self.get_attr('axes').ints)
        if axes == [0]:  # col_sum（列方向の和）
            if shape == [256, 16]:
                return {
                    "inputs": ["default"],
                    "outputs": ["DRAM"]
                }
        elif axes == [1] or axes == [-1]:  # row_sum（行方向の和）
            if shape == [256, 16]:
                return {
                    "inputs": ["PE"],
                    "outputs": ["default"]
                }
        raise NotImplementedError(f"ReduceSum: shape={shape}, axes={axes}")


    def generate_cpp(self) -> List[str]:
        """C++コード生成"""
        lines = []
        
        assert len(self.inputs) >= 1, f"ReduceSum node requires at least 1 input, got {len(self.inputs)}"
        
        in_var = self.get_mapped_var(self.inputs[0])
        out_var = self.get_output_var_name()
        
        # 軸を取得
        axes = list(self.get_attr('axes').ints)
        
        # 形状を取得
        shape = self.in_shape(0)
        
        if len(shape) != 2:
            raise NotImplementedError
        if axes == [0]:
            # 行方向の合計（列ベクトルを返す）
            assert len(shape) == 2, f"ReduceSum with axis=0 expects 2D tensor, got {shape}"
            lines.append(f"    const Vector<{shape[1]}> {out_var} = col_sum<{shape[0]}, {shape[1]}>({in_var});")
        elif axes == [1] or axes == [-1]:
            # 列方向の合計（行ベクトルを返す）
            assert len(shape) == 2, f"ReduceSum with axis=1 expects 2D tensor, got {shape}"
            lines.append(f"    const Vector<{shape[0]}> {out_var} = row_sum<{shape[0]}, {shape[1]}>({in_var});")
        else:
            NotImplementedError
        self.variable_map[self.outputs[0]] = out_var
        return lines

    
    def generate_python(self) -> List[str]:
        """Pythonコード生成"""
        lines = []
        
        inp = self.inputs[0]
        out = self.outputs[0]
        out_var = self.get_output_var_name()
        
        # 軸を取得
        axes = None
        for attr in self.node.attribute:
            if attr.name == 'axes':
                axes = list(attr.ints)
                break
        
        if axes:
            lines.append(f"    {out_var} = torch.sum({self.get_mapped_var(inp)}, dim={axes[0]})")
        else:
            lines.append(f"    {out_var} = torch.sum({self.get_mapped_var(inp)})")
        
        self.variable_map[out] = out_var
        
        return lines
