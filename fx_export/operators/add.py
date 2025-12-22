#!/usr/bin/env python3

from typing import List, Dict, Any, Optional
from . import BaseOperator, register_operator

@register_operator("Add")
class AddOperator(BaseOperator):
    """加算演算オペレーター"""

    def generate_vsm(self) -> List[str]:
        # グラフから形状を取得
        shape0 = self.in_shape(0)
        shape1 = self.in_shape(1)
        lines = []
        
        # 形状によって処理を分岐
        if shape0 == [256, 16] and shape1 == [16]: # add_colvec（列ベクトル加算）
            self.check_layout_in("((8_L2B:1, 4:2, 8_L1B:1), (2:1, 4_PE:1, 2_W:1))", 0)
            self.check_layout_in("((2:1, 4_PE:1, 2_W:1))", 1)
            self.check_layout_out("((8_L2B:1, 4:2, 8_L1B:1), (2:1, 4_PE:1, 2_W:1))")
            
            
            assert self.loc_prefix_in(0) == "m" # LM0 に入力を仮定し、実装を軽減
            assert self.loc_prefix_in(1) == "m" # LM0 に入力を仮定し、実装を軽減
            assert self.addr_in(0) == 0   # 入力0 を addr0 に仮定し、実装を軽減
                                                # ↑、つまり、入力0が "$lm0v" だと仮定している
            
            assert self.addr_in(1) == self.memory_len_in(0)
                                                # 入力1 を、入力 0 の直後と仮定し、実装を軽減

            assert self.loc_prefix_out(0) == "n" # LM1 に出力を仮定し、実装を軽減
            assert self.addr_out(0) == 0        # 出力 を addr0 に仮定し、実装を軽減
                                                      # ↑、つまり、出力が "$ln0v" だと仮定している

            
            # test unit_tests/train_step/Add_*
            lines.append(f"fvpassa $m16v $s0v")
            lines.append(f"nop")
            for i in range(4):
                lines.append(f"fvadd $m{4 * i}v $s0v $n{4 * i}v")
            
            return lines
        raise NotImplementedError(f"Add: shapes {shape0} + {shape1}")


    def testcase_hint(self) -> Optional[str]:
        return "add_colvec.vsm"
    
    def get_memory_layout_tag(self) -> Dict[str, List[str]]:
        """メモリレイアウトタグを定義"""
        shape0 = self.in_shape(0)
        shape1 = self.in_shape(1)
        if shape0 == [256, 16] and shape1 == [16]:
            return {
                "inputs": ["default", "default"],
                "outputs": ["default"]
            }
        raise NotImplementedError(f"Add: shapes {shape0} + {shape1}")

    def generate_cpp(self) -> List[str]:
        """C++コード生成"""
        lines = []
        
        assert len(self.inputs) >= 2, f"Add node requires at least 2 inputs, got {len(self.inputs)}"
        
        in0 = self.get_mapped_var(self.inputs[0])
        in1 = self.get_mapped_var(self.inputs[1])
        out_var = self.get_output_var_name()
        
        # グラフから形状を取得
        shape0 = self.in_shape(0)
        shape1 = self.in_shape(1)
        
        # 形状によって処理を分岐
        if len(shape0) == 2 and len(shape1) == 1:
            # バイアス加算のケース (2D + 1D)
            assert shape0[1] == shape1[0], f"Dimension mismatch: shape0[1]={shape0[1]} != shape1[0]={shape1[0]}"
            lines.append(f"    const Matrix<{shape0[0]}, {shape0[1]}> {out_var} = add_colvec<{shape0[0]}, {shape0[1]}>({in0}, {in1});")
        else:
            raise NotImplementedError(f"Add: shapes {shape0} + {shape1}")
        
        # 変数マップを更新
        self.variable_map[self.outputs[0]] = out_var
        
        return lines
    
    def generate_python(self) -> List[str]:
        """Pythonコード生成"""
        lines = []
        
        assert len(self.inputs) >= 2, f"Add node requires at least 2 inputs, got {len(self.inputs)}"
        
        in0 = self.get_mapped_var(self.inputs[0])
        in1 = self.get_mapped_var(self.inputs[1])
        out_var = self.get_output_var_name()
        
        # グラフから形状を取得
        shape0 = self.in_shape(0)
        shape1 = self.in_shape(1)
        
        if len(shape0) == 2 and len(shape1) == 1:
            assert shape0[1] == shape1[0], f"Dimension mismatch: shape0[1]={shape0[1]} != shape1[0]={shape1[0]}"
            lines.append(f"    {out_var} = {in0} + {in1}")
        else:
            raise NotImplementedError(f"Add: shapes {shape0} + {shape1}")
        
        # 変数マップを更新
        self.variable_map[self.outputs[0]] = out_var
        
        return lines
