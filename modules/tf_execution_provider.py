import tensorflow
import tensorflow.keras as keras 
from tensorflow.keras.models import load_model
from tensorflow.python.keras.backend import set_session
import numpy as np
import time

from queue import Queue
import os, sys, cv2

MAX_BATCH_SIZE = 32

class BatchedInferenceProvider :

    #Implements a enqueue function that holds all the image data in
    #a queue and passes those images to an execution context when the 
    #queue size is greater than batch size. This method works only if 
    #the model supports batching, and wait can be set to false, if you don't 
    #want batched execution.
    def __init__(self, context, batch_size = 8, wait = True):
        if not isinstance(context, KerasContext):
            print('context must be an instance of KerasContext')
            sys.exit(0)
        self.context = context
        self.batch_size = batch_size
        self.wait = wait
        #Queue is pre-allocated for MAX_BATCH_SIZE
        #so you can change batch_size dynamically
        self.queue = Queue(MAX_BATCH_SIZE)
    

    def add_to_batch(self, image_data):

        image_data = preprocess(image_data)

        if not self.wait : return True, [int(np.argmax(self.context.infer(image_data)[0]))]

        self.queue.put(image_data)
        if self.queue.qsize() < self.batch_size :
            return (False, None)
        
        batch = list()
        print()
        for i in range(self.batch_size):
            image = self.queue.get()
            batch.append(image)
        
        batch = np.array(batch)
        return True, [int(np.argmax(tensor)) for tensor in self.context.infer(batch)]


class KerasContext :

    def __init__(self, model_file):
        if not os.path.exists(model_file):
            print('Model {} not found'.format(model_file))
            sys.exit(0)
        self.graph = tensorflow.compat.v1.get_default_graph()
        self.session = tensorflow.Session()
        set_session(self.session)
        self.model = load_model(model_file)
    
    def infer(self, image_data):
        if not type(image_data) == np.ndarray or image_data.ndim < 3:
            print('Provide a numpy array as NWHC or WHC format')
            sys.exit(0)
        if image_data.ndim == 3 :
            image_data = np.expand_dims(image_data, axis = 0)
        
        image_data = image_data.astype('float32')
        
        with self.graph.as_default() :
            st = time.time()
            set_session(self.session)
            result = self.model.predict(image_data)
            et = time.time()
            print('Inference time : {}s'.format(et - st))
            return result

def preprocess(image_data):
    #image_data : bytes data from an image
    if not type(image_data) == bytes :
        print('image_data should be a byte-string')
        sys.exit(0)

    image_data = np.fromstring(image_data, dtype = 'uint8')
    image_np = cv2.imdecode(image_data, cv2.IMREAD_COLOR)
    
    if image_np.shape[-1] != 3 :
        print('Provide an RGB image')
        sys.exit(0)
    
    return ( cv2.resize(image_np, (225, 225)) / 255. )

        
#context = KerasContext('../model/VGG16_extended_v1.h5')
#provider = BatchedInferenceProvider(context, batch_size = 8, wait = False)

#image = open('no.jpeg', 'rb').read()
#while True :
#    print(provider.add_to_batch(image))

