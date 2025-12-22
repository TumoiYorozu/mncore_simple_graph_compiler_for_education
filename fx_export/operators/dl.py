"""DL (Download) operator for MN-Core memory transfer simulation."""

import onnx
from typing import List, Dict, Optional, Tuple
from . import BaseOperator, register_operator
from .dlul_impls import dl_256x16, dl_256x16_pe, dl_256x16_l2b, dl_256x1024, dl_16x1024, dl_16, dl_16x16, dl_256

@register_operator('DL')
class DLOperator(BaseOperator):
    """DRAM → LM へのデータ転送オペレーター"""
    
    def generate_vsm(self) -> List[str]:
        in_shape = self.in_shape()
        assert in_shape == self.out_shape()
        assert self.loc_prefix_in() == "d"
        
        if self.in_shape() == [16]:
            if self.tag_out() == "default":
                return dl_16.generate_vsm(self)
            
        if self.in_shape() == [256]:
            if self.tag_out() == "default":
                return dl_256.generate_vsm(self)
            
        if self.in_shape() == [16,16]:
            if self.tag_out() == "default":
                return dl_16x16.generate_vsm(self)

        if self.in_shape() == [256, 16]:
            if self.tag_out() == "default":
                return dl_256x16.generate_vsm(self)
            if self.tag_out() == "PE":
                return dl_256x16_pe.generate_vsm(self)
            if self.tag_out() == "L2B":
                return dl_256x16_l2b.generate_vsm(self)
            
        if self.in_shape() == [256, 1024]:
            if self.tag_out() == "default":
                return dl_256x1024.generate_vsm(self)
            
        if self.in_shape() == [16, 1024]:
            if self.tag_out() == "default":
                return dl_16x1024.generate_vsm(self)
        
        raise NotImplementedError(f"DL operator not implemented for shape {in_shape} with tag {self.tag_out()}")

    def testcase_hint(self) -> Optional[str]:
        return "dram_dl_*.vsm"
    

    def generate_cpp(self) -> List[str]:
        # DL操作のC++コード生成（単純なコピー）
        input_var = self.inputs[0]
        output_var = self.outputs[0]
        
        cpp_input_var = self.get_mapped_var(input_var)
        
        # DLでは型変換に対応するためautoを使う
        # 既知の特殊ケースは別処理
        if cpp_input_var == 'target':
            # targetはint配列なのでそのまま使う
            code = [f"    const auto {output_var} = {cpp_input_var};  // DL: DRAM -> Local Memory (int array)"]
        else:
            # それ以外は型保持のためautoを使う
            code = [f"    const auto {output_var} = {cpp_input_var};  // DL: DRAM -> Local Memory"]
        
        # DLオペレータも他のオペレータと同様にvariable_mapを更新
        self.variable_map[output_var] = output_var
        return code
    
    def generate_python(self) -> List[str]:
        # DL操作のPythonコード生成（単純なコピー）
        input_var = self.inputs[0]
        output_var = self.outputs[0]
        
        python_input_var = self.get_mapped_var(input_var)
        
        # DLオペレータも他のオペレータと同様にvariable_mapを更新
        self.variable_map[output_var] = output_var
            
        return [f"    {output_var} = {python_input_var}  # DL: DRAM -> Local Memory"]
    
    def DL_plan(self) -> Tuple[List[onnx.NodeProto], onnx.NodeProto, List[onnx.NodeProto]]:
        # DL自体は分解しない
        return [], self.node, []
    
    def get_memory_layout_tag(self) -> Dict[str, List[str]]:
        """Get memory layout tags for DL operator"""
        return {"inputs": [], "outputs": []}
