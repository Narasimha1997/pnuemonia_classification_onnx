# pnuemonia_classification_onnx
Pnuemonia classification at scale using ONNX runtime and an inference server with batching support.

The project pnuemonia_classification is aimed at understanding the complete AI Research-Development-Production pipeline. The best approach for learning 
is identified in `train.ipynb`, the model is built and trained. Later it is made ready for optimized inference by converting it to ONNX and executing on ONNX Runtime.
This repo also provides a comphrensive understanding of how to export and use ONNX Models, if you are looking for a source to understand ONNX practically, you are
at the right place.

### Quick Setup:
To set-up the project locally, run `setup_native.sh`, this downloads latest version of Tensorflow and other python dependencies, and also
converts the h5 model to ONNX.

To launch inference server natively :
```
python3 server.py
```

### Creating a docker image :
To setup a docker image, run `setup_docker.sh` and wait for the image to build. Later you can spin up the container by using following command :
```
docker run --net=host  pnuemonia_onnx:latest
```

### APIs :
The inference server which hosts ONNX or Keras runtime provides following APIs:
<br/>
Query status : `/api/status` GET
<br/>
Example :
```
curl http://localhost:5000/api/status
```
<br/>

Inference : `/api/inference`  POST  multipart/form-data
<br/>
Example : 
```
curl -X POST -F "image=@/home/narasimha/pnuemonia_detection/samples/no.jpeg" -H "Content-Type: multipart/form-data"  http://localhost:5000/api/infer
```


### Training your own model 
If you don't wat to use the pretrained model provided in the repo you can train your own model or optimize the existing one. ( You can port
the model to more optimized backends like MobileNet or ResNet50 ). The model is trained using VGG16 transfer learning approach and was benchmarked 
for `93.8%` validation accuracy over 20 epochs. The model was trained on Google Colab on T4 GPU support.
<br/>
Notebook : `train.ipynb`
<br/>
Dataset : [Kaggle Pnuemonia Chest X-Ray Images](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia)


### ONNX Runtime
The project adds support to run ONNX models directly from the inference server. As a Fallback if ONNX Runtime fails, Keras Runtime takes over 
and starts providing runtime support. ONNX Runtime provides an optimized model runtime, these optimizaitions are hardware specific and are 
created during keras to ONNX conversion, see `modules/keras_to_onnx_converter.py` for more. 

#### Converting to ONNX :
To convert model to ONNX format, you can quickly run `convert.py`. To configure/ set-up optimizations for your target hardware edit
 `modules/keras_to_onnx_converter.py`. The current optimizations and options are as follows :
 
 ```pyhthon
    options = onnxruntime.SessionOptions()
    options.enable_cpu_mem_arena = True
    options.enable_mem_pattern = True
    options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
    options.optimized_model_filepath = output_path
 
 ```
 Target Opset and Version information:
 ```python
 onnx_model = convert( model_path, {
        "name" : "pnuemonia.onnx",
        "batch_size" : 1,
        "opset" : 7,
        "target_version" : onnx.__version__
  })
```
