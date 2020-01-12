import onnxruntime
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
        if not isinstance(context, ONNXContext):
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
            return (False, self.batch_size - self.queue.qsize())
        
        batch = list()
        print()
        for i in range(self.batch_size):
            image = self.queue.get()
            batch.append(image)
        
        batch = np.array(batch, dtype = 'float32')
        return True,[int(np.argmax(tensor)) for tensor in self.context.infer(batch)]


class ONNXContext :

    def __init__(self, model_file):
        if not os.path.exists(model_file):
            print('Model {} not found'.format(model_file))
            sys.exit(0)
        self.session = onnxruntime.InferenceSession(model_file)
    
    def infer(self, image_data):
        if not type(image_data) == np.ndarray or image_data.ndim < 3:
            print('Provide a numpy array as NWHC or WHC format')
            sys.exit(0)
        if image_data.ndim == 3 :
            image_data = np.expand_dims(image_data, axis = 0)
        
        image_data = [image_data.astype('float32')]
        feed = dict([ (input.name, image_data[i]) for i, input in  enumerate(self.session.get_inputs()) ])
        outputs = [ output.name for output in self.session.get_outputs() ]

        return self.session.run(outputs, feed)

def preprocess(image_data):
    #image_data : bytes data from an image
    if not type(image_data) == bytes :
        print('image_data should be a byte-string')
        sys.exit(0)

    image_data = np.fromstring(image_data, dtype = 'uint8')
    image_np = cv2.imdecode(image_data, cv2.IMREAD_COLOR)

    print(image_np.shape)
    
    if image_np.shape[-1] != 3 :
        print('Provide an RGB image')
        sys.exit(0)
    
    return ( cv2.resize(image_np, (225, 225)) / 255. )

        

