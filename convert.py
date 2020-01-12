from modules import onnx_convert_api

HD5_MODEL = './model/pnuemonia.h5'
ONNX_TARGET = './model/pnuemonia.onnx'

onnx_convert_api(HD5_MODEL, ONNX_TARGET, './samples')
