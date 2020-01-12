from flask import Flask, request

from modules import KerasContext, KerasBatchedInferenceProvider
from modules import ONNXContext, ONNXBatchedInferenceProvider

import os, sys 

ONNX_MODEL_PATH = './model/pnuemonia.onnx'
HD5_MODEL_PATH = './model/pnuemonia.h5'

if not os.path.exists(ONNX_MODEL_PATH):
    print('ONNX Model not found. Try running convert.py')

runtime_provider_fn = None
runtime = "onnx"

try:
    context = ONNXContext(ONNX_MODEL_PATH)
    batched_provider = ONNXBatchedInferenceProvider(context, wait = False)
    runtime_provider_fn = batched_provider.add_to_batch
    print('Running with ONNX Runtime')

except :
    context = KerasContext(HD5_MODEL_PATH)
    batched_provider = KerasBatchedInferenceProvider(context, wait = False)
    runtime_provider_fn = batched_provider.add_to_batch
    print('Running with Keras runtime, fallback to normal execution provider.')
    runtime = "keras-tensorflow"


app = Flask(__name__)

@app.route("/api/status")
def status():

    return {
        "status" : "active",
        "runtime" : runtime,
        "success" : True
    }

@app.route("/api/infer", methods = ['POST'])
def infer():

    image = request.files['image']
    image = image.stream.read()

    isinfer, result = runtime_provider_fn(image)

    if isinfer :
        return {
            "success" : True,
            "isinfer" : True,
            "result" : result,
            "remaining" : 0,
            "runtime" : runtime
        }
    
    else :
        return {
            "success" : True,
            "isinfer" : False,
            "result" : "Model has wait enabled in batch model",
            "remaining" : result,
            "runtme" : runtime
        }


app.run('0.0.0.0', port = 5000)