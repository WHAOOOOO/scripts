# about merge_bn and trans_gn
## merge_bn
> 将bn层的参数merge到conv层
## trans_gn
> onnx不支持gn，将gn转化为In+reshape，或者将gn转化为In+bn