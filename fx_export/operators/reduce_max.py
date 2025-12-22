#!/usr/bin/env python3

from typing import List, Dict, Any, Optional
from . import BaseOperator
from . import register_operator


@register_operator("ReduceMax")
class ReduceMaxOperator(BaseOperator):
    """ReduceMax演算オペレーター"""

    def generate_vsm(self) -> List[str]:
        # グラフから形状を取得
        shape = self.in_shape()
        lines = []

        axes = list(self.get_attr("axes").ints)

        # 形状によって処理を分岐
        if len(shape) == 2 and (axes == [-1]):  # max_row（行方向の最大）
            if shape == [256, 16]:
                self.check_layout_in("((4_L2B:2, 64:2), (2:1, 4_PE:1, 2_W:1))")
                self.check_layout_out("((4_L2B:2, 32:1, 2_W:1))")
                assert self.loc_prefix_in() == "m"  # LM0 に入力を仮定し、実装を軽減
                assert self.addr_in() == 0  # Addr 0 に入力を仮定し、実装を軽減
                # ↑、つまり、入力が "$lm0v" だと仮定している

                assert self.loc_prefix_out() == "n"  # LM1 に出力を仮定し、実装を軽減
                assert self.addr_out() == 0  # Addr 0 に出力を仮定し、実装を軽減
                # ↑、つまり、出力が "$ln0v" だと仮定している

                # test unit_tests/train_step/ReduceMax_*
                # 問題名：「MaxRow」
                
                # 各PEで16回ループ
                for block in range(16):
                    # 現在のブロックの開始メモリアドレスを計算
                    # block * 16で該当する行のベースアドレスを取得
                    base_addr = block * 16
                    
                    # 最初の4要素（要素0-3）をALUに読み込み（初期値として設定）
                    lines.append(f"fpassa $m{base_addr}v4 $nowrite")

                    # 次の4要素（要素4-7）と前の結果を比較、最大値を$alufに保存
                    lines.append(f"fmax $m{base_addr + 1}v4 $aluf $nowrite")
                    
                    # さらに次の4要素（要素8-11）と比較、最大値を$alufに保存
                    lines.append(f"fmax $m{base_addr + 2}v4 $aluf $nowrite")
                    
                    # 最後の4要素（要素12-15）と比較、結果を$r0vに保存
                    lines.append(f"fmax $m{base_addr + 3}v4 $aluf $r0v")
                    
                    # ベクトルレジスタの内容をスカラーレジスタに移動
                    lines.append("msr $aluf $s0v")
                    
                    # ベクトル内の4要素同士で最大値を計算（水平方向の比較）
                    lines.append("fmax $aluf $r0v $r0v")
                    
                    # 計算結果をスカラーレジスタ間で移動
                    lines.append("msr $s0v $s0v")
                    
                    lines.append("fmax $aluf $r0v $r0v")
                    lines.append("msr $s0v $s0v")

                    # 出力は256要素のベクトル（各要素が対応する行の最大値）
                    output_addr = block * 4
                    lines.append(f"fmax $aluf $r0v $n{output_addr}v")

                return lines
        raise NotImplementedError

    def testcase_hint(self) -> Optional[str]:
        return "max_row.vsm"

    def get_memory_layout_tag(self) -> Dict[str, List[str]]:
        shape0 = self.in_shape(0)
        oshape = self.out_shape(0)
        axes = list(self.get_attr("axes").ints)
        if shape0 == [256, 16] and oshape == [256] and axes == [-1]:
            return {"inputs": ["PE"], "outputs": ["default"]}
        raise NotImplementedError(f"ReduceMax: shape={shape0}, axes={axes}")

    def generate_cpp(self) -> List[str]:
        """C++コード生成"""
        lines = []

        assert (
            len(self.inputs) >= 1
        ), f"ReduceMax node requires at least 1 input, got {len(self.inputs)}"

        in_var = self.get_mapped_var(self.inputs[0])
        out_var = self.get_output_var_name()

        # 属性を取得
        axes = list(self.get_attr("axes").ints)

        # 形状を取得
        shape = self.in_shape(0)

        if len(shape) == 2 and (axes == [-1]):
            # 行方向の最大値（最後の次元）
            lines.append(
                f"    const Vector<{shape[0]}> {out_var} = row_max<{shape[0]}, {shape[1]}>({in_var});"
            )
        else:
            raise NotImplementedError(f"ReduceMax: axes={axes}, shape={shape}")

        self.variable_map[self.outputs[0]] = out_var
        return lines

    def generate_python(self) -> List[str]:
        """Pythonコード生成"""
        lines = []

        inp = self.inputs[0]
        out = self.outputs[0]
        out_var = self.get_output_var_name()

        # 属性を取得
        axes = list(self.get_attr("axes").ints)
        if len(axes) != 1:
            raise NotImplementedError(f"ReduceMax: axes={axes}")

        axis = axes[0]
        lines.append(
            f"    {out_var}, _ = torch.max({self.get_mapped_var(inp)}, dim={axis}, keepdim=False)"
        )

        self.variable_map[out] = out_var

        return lines
