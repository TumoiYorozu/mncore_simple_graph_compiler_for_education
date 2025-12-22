"""UL (Upload) operator for MN-Core memory transfer simulation."""

import onnx
from typing import List, Dict, Tuple, Optional
from . import BaseOperator, register_operator

from .dlul_impls import ul_256x16, ul_256x16_pe, ul_256, ul_16, ul_16x16, ul_16x1024


@register_operator('UL')
class ULOperator(BaseOperator):
    """LM → DRAM へのデータ転送オペレーター"""
    
    
    def generate_vsm(self) -> List[str]:
        in_shape = self.in_shape()
        assert in_shape == self.out_shape()
        assert self.loc_prefix_out() == "d"

        if self.in_shape() == [256, 16]:
            if self.tag_in() == "default":
                return ul_256x16.generate_vsm(self)
            if self.tag_in() == "PE":
                return ul_256x16_pe.generate_vsm(self)

        if self.in_shape() == [256]:
            if self.tag_in() == "default":
                return ul_256.generate_vsm(self)

        if self.in_shape() == [16]:
            if self.tag_in() == "default":
                return ul_16.generate_vsm(self)

        if self.in_shape() == [16, 16]:
            if self.tag_in() == "default":
                return ul_16x16.generate_vsm(self)
            
        if self.in_shape() == [16, 1024]:
            if self.tag_in() == "default":
                return ul_16x1024.generate_vsm(self)

        raise NotImplementedError(f"UL operator not implemented for shape {in_shape} with tag {self.tag_in()}")
    
    def testcase_hint(self) -> Optional[str]:
        return "dram_ul_*.vsm"


    def generate_cpp(self) -> List[str]:
        # UL操作のC++コード生成（単純なコピー）
        input_var = self.inputs[0]
        output_var = self.outputs[0]
        
        cpp_input_var = self.get_mapped_var(input_var)
        
        # ULでは実際の型を保つため常にautoを使う
        # 入力型は前段で決まっているため
        code = [f"    const auto {output_var} = {cpp_input_var};  // UL: Local Memory -> DRAM"]
        
        # ULオペレータも他のオペレータと同様にvariable_mapを更新
        self.variable_map[output_var] = output_var
        return code
    
    def generate_python(self) -> List[str]:
        # UL操作のPythonコード生成（単純なコピー）
        input_var = self.inputs[0]
        output_var = self.outputs[0]
        
        python_input_var = self.get_mapped_var(input_var)
        
        # ULオペレータも他のオペレータと同様にvariable_mapを更新
        self.variable_map[output_var] = output_var
            
        return [f"    {output_var} = {python_input_var}  # UL: Local Memory -> DRAM"]
    
    def DL_plan(self) -> Tuple[List[onnx.NodeProto], onnx.NodeProto, List[onnx.NodeProto]]:
        # UL自体は分解しない
        return [], self.node, []
    
    def get_memory_layout_tag(self) -> Dict[str, List[str]]:
        """Get memory layout tags for UL operator"""
        return {"inputs": [], "outputs": []}
