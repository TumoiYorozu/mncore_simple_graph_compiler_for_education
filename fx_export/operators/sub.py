#!/usr/bin/env python3

from typing import List, Dict, Any, Optional
from . import BaseOperator
from . import register_operator

@register_operator("Sub")
class SubOperator(BaseOperator):
    """減算演算オペレーター"""
    
    def generate_vsm(self) -> List[str]:
        lines = []
        shape0 = self.in_shape(0)
        shape1 = self.in_shape(1)
        
        in0_prefix = self.loc_prefix_in(0)  # $lm0v  の "m" の部分
        in1_prefix = self.loc_prefix_in(1)  # $lm16v の "m" の部分
        out_prefix = self.loc_prefix_out()        # $ln0v  の "n" の部分
        
        in0_offset = self.addr_in(0)        # $lm0v  の "0" の部分
        in1_offset = self.addr_in(1)        # $lm16v の "16" の部分
        out_offset = self.addr_out()              # $ln0v の "0" の部分
        
        if shape0 == shape1:
            # 同じ形状の場合
            # test unit_tests/train_step/Sub_256x16_256x16_256x16_*
            # 問題名：「A - B」
            
            for i in range(self.memory_len_in_div_ceil(8, 0)):  # PE あたりの長さを、8単語で割って切り上げ
                lines.append(f"ipassa $l{in1_prefix}{in1_offset + i * 8}v $nowrite")
                raise NotImplementedError("Please implement the VSM code!!")

        elif len(shape0) == 2 and len(shape1) == 1 and shape0[0] == shape1[0]:
            # ブロードキャスト（ベクトルを各行から減算） SubRow
            # test unit_tests/train_step/Sub_256x16_256_256x16_*
            # 問題名：「SubRow vec」
            
            
            self.check_layout_in("((4_L2B:2, 64:2), (2:1, 4_PE:1, 2_W:1))", 0)
            self.check_layout_in("((4_L2B:2, 32:1, 2_W:1))", 1)
            self.check_layout_out("((4_L2B:2, 64:2), (2:1, 4_PE:1, 2_W:1))")
            
            assert self.loc_prefix_in(0)  == "m" # LM0 に入力 0 を仮定し、実装を軽減
            assert self.addr_in(0)        ==  0  # Addr 0 に入力 0 を仮定し、実装を軽減
                                                       # ↑、つまり、入力 0 が "$lm0v" だと仮定している
            
            assert self.loc_prefix_in(1)  == "m" # LM0 に入力 1 を仮定し、実装を軽減
            assert self.addr_in(1)        ==  self.memory_len_in(0)  # IN0 の次に入力を仮定し、実装を軽減
            
            assert self.loc_prefix_out(0) == "n" # LM1 に出力を仮定し、実装を軽減
            assert self.addr_out(0)       ==  0  # Addr 0 に出力を仮定し、実装を軽減
                                                       # ↑、つまり、出力 0 が "$ln0v" だと仮定している
        
            lines.append("fvpassa $m[256,257,258,259] $r[0,2,4,6]")
            lines.append("fvpassa $m[256,257,258,259] $r[1,3,5,7]")
            lines.append("fvpassa $m[260,261,262,263] $s[0,2,4,6]")
            lines.append("fvpassa $m[260,261,262,263] $s[1,3,5,7]")
            lines.append("fvadd $llm0v -$lr[0,2,4,6] $lln[0,4,8,12]")
            lines.append("fvadd $lm[2,6,10,14] -$lr[0,2,4,6] $ln[2,6,10,14]")
            lines.append("fvpassa $m[264,265,266,267] $r[0,2,4,6]")
            lines.append("fvpassa $m[264,265,266,267] $r[1,3,5,7]")
            lines.append("fvadd $llm16v -$ls[0,2,4,6] $lln[16,20,24,28]")
            lines.append("fvadd $lm[18,22,26,30] -$ls[0,2,4,6] $ln[18,22,26,30]")
            lines.append("fvpassa $m[268,269,270,271] $s[0,2,4,6]")
            lines.append("fvpassa $m[268,269,270,271] $s[1,3,5,7]")
            lines.append("fvadd $llm32v -$lr[0,2,4,6] $lln[32,36,40,44]")
            lines.append("fvadd $lm[34,38,42,46] -$lr[0,2,4,6] $ln[34,38,42,46]")
            lines.append("fvpassa $m[272,273,274,275] $r[0,2,4,6]")
            lines.append("fvpassa $m[272,273,274,275] $r[1,3,5,7]")
            lines.append("fvadd $llm48v -$ls[0,2,4,6] $lln[48,52,56,60]")
            lines.append("fvadd $lm[50,54,58,62] -$ls[0,2,4,6] $ln[50,54,58,62]")
            lines.append("fvpassa $m[276,277,278,279] $s[0,2,4,6]")
            lines.append("fvpassa $m[276,277,278,279] $s[1,3,5,7]")
            lines.append("fvadd $llm64v -$lr[0,2,4,6] $lln[64,68,72,76]")
            lines.append("fvadd $lm[66,70,74,78] -$lr[0,2,4,6] $ln[66,70,74,78]")
            lines.append("fvpassa $m[280,281,282,283] $r[0,2,4,6]")
            lines.append("fvpassa $m[280,281,282,283] $r[1,3,5,7]")
            lines.append("fvadd $llm80v -$ls[0,2,4,6] $lln[80,84,88,92]")
            lines.append("fvadd $lm[82,86,90,94] -$ls[0,2,4,6] $ln[82,86,90,94]")
            lines.append("fvpassa $m[284,285,286,287] $s[0,2,4,6]")
            lines.append("fvpassa $m[284,285,286,287] $s[1,3,5,7]")
            lines.append("fvadd $llm96v -$lr[0,2,4,6] $lln[96,100,104,108]")
            lines.append("fvadd $lm[98,102,106,110] -$lr[0,2,4,6] $ln[98,102,106,110]")
            lines.append("fvpassa $m[288,289,290,291] $r[0,2,4,6]")
            lines.append("fvpassa $m[288,289,290,291] $r[1,3,5,7]")
            lines.append("fvadd $llm112v -$ls[0,2,4,6] $lln[112,116,120,124]")
            lines.append("fvadd $lm[114,118,122,126] -$ls[0,2,4,6] $ln[114,118,122,126]")
            lines.append("fvpassa $m[292,293,294,295] $s[0,2,4,6]")
            lines.append("fvpassa $m[292,293,294,295] $s[1,3,5,7]")
            lines.append("fvadd $llm128v -$lr[0,2,4,6] $lln[128,132,136,140]")
            lines.append("fvadd $lm[130,134,138,142] -$lr[0,2,4,6] $ln[130,134,138,142]")
            lines.append("fvpassa $m[296,297,298,299] $r[0,2,4,6]")
            lines.append("fvpassa $m[296,297,298,299] $r[1,3,5,7]")
            lines.append("fvadd $llm144v -$ls[0,2,4,6] $lln[144,148,152,156]")
            lines.append("fvadd $lm[146,150,154,158] -$ls[0,2,4,6] $ln[146,150,154,158]")
            lines.append("fvpassa $m[300,301,302,303] $s[0,2,4,6]")
            lines.append("fvpassa $m[300,301,302,303] $s[1,3,5,7]")
            lines.append("fvadd $llm160v -$lr[0,2,4,6] $lln[160,164,168,172]")
            lines.append("fvadd $lm[162,166,170,174] -$lr[0,2,4,6] $ln[162,166,170,174]")
            lines.append("fvpassa $m[304,305,306,307] $r[0,2,4,6]")
            lines.append("fvpassa $m[304,305,306,307] $r[1,3,5,7]")
            lines.append("fvadd $llm176v -$ls[0,2,4,6] $lln[176,180,184,188]")
            lines.append("fvadd $lm[178,182,186,190] -$ls[0,2,4,6] $ln[178,182,186,190]")
            lines.append("fvpassa $m[308,309,310,311] $s[0,2,4,6]")
            lines.append("fvpassa $m[308,309,310,311] $s[1,3,5,7]")
            lines.append("fvadd $llm192v -$lr[0,2,4,6] $lln[192,196,200,204]")
            lines.append("fvadd $lm[194,198,202,206] -$lr[0,2,4,6] $ln[194,198,202,206]")
            lines.append("fvpassa $m[312,313,314,315] $r[0,2,4,6]")
            lines.append("fvpassa $m[312,313,314,315] $r[1,3,5,7]")
            lines.append("fvadd $llm208v -$ls[0,2,4,6] $lln[208,212,216,220]")
            lines.append("fvadd $lm[210,214,218,222] -$ls[0,2,4,6] $ln[210,214,218,222]")
            lines.append("fvpassa $m[316,317,318,319] $s[0,2,4,6]")
            lines.append("fvpassa $m[316,317,318,319] $s[1,3,5,7]")
            lines.append("fvadd $llm224v -$lr[0,2,4,6] $lln[224,228,232,236]")
            lines.append("fvadd $lm[226,230,234,238] -$lr[0,2,4,6] $ln[226,230,234,238]")
            lines.append("fvadd $llm240v -$ls[0,2,4,6] $lln[240,244,248,252]")
            lines.append("fvadd $lm[242,246,250,254] -$ls[0,2,4,6] $ln[242,246,250,254]")
            return lines
        else:
            raise NotImplementedError
        
        return lines
    
    def testcase_hint(self) -> Optional[str]:
        shape0 = self.in_shape(0)
        shape1 = self.in_shape(1)
        if shape0 == shape1:
            return "mat_sub.vsm"
        else:
            return "sub_rowvec.vsm"
    
    def get_memory_layout_tag(self) -> Dict[str, List[str]]:
        shape0 = self.in_shape(0)
        shape1 = self.in_shape(1)
        if shape0 == shape1:
            # 同じシェイプで、要素ごとの減算
            return {
                "inputs": ["default", "default"],
                "outputs": ["default"]
            }
        elif shape0 == [256, 16] and shape1 == [256]:
            return {
                "inputs": ["PE", "default"],
                "outputs": ["PE"]
            }
        raise NotImplementedError(f"Sub: shapes {shape0} - {shape1}")
    

    def generate_cpp(self) -> List[str]:
        """C++コード生成"""
        lines = []
        
        assert len(self.inputs) >= 2, f"Sub node requires at least 2 inputs, got {len(self.inputs)}"
        
        in1_var = self.get_mapped_var(self.inputs[0])
        in2_var = self.get_mapped_var(self.inputs[1])
        out_var = self.get_output_var_name()
        
        # 形状を取得
        shape0 = self.in_shape(0)
        shape1 = self.in_shape(1)
        
        if shape0 == shape1:
            # 同じ形状の場合
            if len(shape0) == 2:
                lines.append(f"    const Matrix<{shape0[0]}, {shape0[1]}> {out_var} = sub<{shape0[0]}, {shape0[1]}>({in1_var}, {in2_var});")
            elif len(shape0) == 1:
                lines.append(f"    const Vector<{shape0[0]}> {out_var} = sub<{shape0[0]}>({in1_var}, {in2_var});")
            else:
                raise NotImplementedError(f"Sub: shape {shape0}")
        elif len(shape0) == 2 and len(shape1) == 1 and shape0[0] == shape1[0]:
            # ブロードキャスト（ベクトルを各行から減算）
            lines.append(f"    const Matrix<{shape0[0]}, {shape0[1]}> {out_var} = sub_rowvec<{shape0[0]}, {shape0[1]}>({in1_var}, {in2_var});")
        else:
            raise NotImplementedError(f"Sub: shapes {shape0} - {shape1}")
        
        self.variable_map[self.outputs[0]] = out_var
        return lines
    
    def generate_python(self) -> List[str]:
        """Pythonコード生成"""
        lines = []
        
        a = self.inputs[0]
        b = self.inputs[1]
        out = self.outputs[0]
        out_var = self.get_output_var_name()
        
        shape_a = self.in_shape(0)
        shape_b = self.in_shape(1)
        if shape_a == shape_b:
            lines.append(f"    {out_var} = {self.get_mapped_var(a)} - {self.get_mapped_var(b)}")
        elif len(shape_a) == 2 and len(shape_b) == 1 and shape_a[0] == shape_b[0]:
            lines.append(f"    {out_var} = {self.get_mapped_var(a)} - {self.get_mapped_var(b)}.unsqueeze(-1)")
        else:
            raise NotImplementedError(f"Sub: shapes {shape_a} - {shape_b}")
        self.variable_map[out] = out_var
        
        return lines
