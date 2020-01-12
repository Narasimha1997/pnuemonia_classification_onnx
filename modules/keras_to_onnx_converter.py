import onnxmltools
from tensorflow.keras.models import Model, load_model
import onnx
import cv2
import numpy as np
import os

def convert(model, parameters, samples = './samples'):

    if not isinstance(model, Model) :
        model = load_model(model)

    #REFERENCE PARAMETERS
    opset_version = parameters['opset']
    targeted_onnx = parameters['target_version']
    name = parameters['name']
    default_batch_size = parameters['batch_size']

    onnx_model = onnxmltools.convert_keras(
        model,
        name = name,
        target_opset = 7,
        default_batch_size = default_batch_size,
        targeted_onnx = targeted_onnx
    )

    #onnxmltools.save_model(onnx_model, '../model/' + name )

    return onnx_model.SerializeToString()

#test the model's accuracy after conversion :
import onnxruntime

#load the saved model
#ONNX session when created from Python API uses the most suitable provider

def create_onnx(model_path, output_path, samples):

    onnx_model = convert( model_path, {
        "name" : "pnuemonia.onnx",
        "batch_size" : 1,
        "opset" : 7,
        "target_version" : onnx.__version__
    })

    options = onnxruntime.SessionOptions()
    options.enable_cpu_mem_arena = True
    options.enable_mem_pattern = True
    options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
    options.optimized_model_filepath = output_path

    session = onnxruntime.InferenceSession(onnx_model, options)
    print('Inputs : ',  [ip.shape for ip in session.get_inputs()])
    print('Outputs : ', [op.shape for op in session.get_outputs()])
    #print('Providers : ', [provider for provider in session.get_providers()])

    #print('Device : ', [session.get_session_options()])

    #Run test use case with batch:
    image_1 = cv2.imread(os.path.join(samples, 'no.jpeg'))
    image_2 = cv2.imread(os.path.join(samples, 'yes.jpeg'))
    image_1 = cv2.resize(image_1, (224, 224)) / 255.
    image_2 = cv2.resize(image_2, (224, 224)) / 255.

    import time
    batch = np.array([image_1, image_2], dtype='float32')

    batch = [batch]
    feed = dict([(input.name , batch[i]) for i, input in enumerate(session.get_inputs())])
    outputs = [output.name for output in session.get_outputs()]

    st = time.time()
    predictions = session.run(outputs, feed)
    et = time.time()

    print('Predictions : ', predictions)
    print('Inference time : ', (et - st))

    #assertions :
    assert [np.argmax(predictions[0][0]), np.argmax(predictions[0][1])] == [0, 1], "Model conversion error, output predictions are not correct"
    print('Assertions passed, Model correctly converted.. You can use the saved model for execution.')
