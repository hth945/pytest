import tensorflow as tf
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.framework import tensor_util
import numpy as np
import argparse
import cv2

# If load from pb, you may have to use get_tensor_by_name heavily.


class CNN(object):
    def __init__(self, model_filepath):

        # The file path of model
        self.model_filepath = model_filepath
        # Initialize the model
        self.load_graph(model_filepath=self.model_filepath)

    def load_graph(self, model_filepath):
        '''
        Lode trained model.
        '''
        print('Loading model...')
        self.graph = tf.Graph()

        # with tf.gfile.GFile(model_filepath, 'rb') as f:
        #     graph_def = tf.GraphDef()
        #     graph_def.ParseFromString(f.read())
        with tf.io.gfile.GFile (model_filepath, 'rb') as f:
            graph_def = tf.compat.v1.GraphDef()
            data = f.read()
            graph_def.ParseFromString(data)

        print('Check out the input placeholders:')
        nodes = [
            n.name + ' => ' + n.op for n in graph_def.node
            if n.op in ('Placeholder')
        ]
        for node in nodes:
            print(node)

        with self.graph.as_default():
            # Define input tensor
            self.input = tf.placeholder(np.float32,
                                        shape=[None, 224, 224, 3],
                                        name='input')
            tf.import_graph_def(graph_def, {
                'input': self.input,
            })

        self.graph.finalize()

        print('Model loading complete!')

        # Get layer names
        layers = [op.name for op in self.graph.get_operations()]
        for layer in layers:
            print(layer)
        """
        # Check out the weights of the nodes
        weight_nodes = [n for n in graph_def.node if n.op == 'Const']
        for n in weight_nodes:
            print("Name of the node - %s" % n.name)
            # print("Value - " )
            # print(tensor_util.MakeNdarray(n.attr['value'].tensor))
        """

        # In this version, tf.InteractiveSession and tf.Session could be used interchangeably.
        # self.sess = tf.InteractiveSession(graph = self.graph)
        self.sess = tf.Session(graph=self.graph)

    def test(self, data):

        # Know your output node name
        output_tensor = self.graph.get_tensor_by_name("import/cnn/output:0")
        output = self.sess.run(output_tensor,
                               feed_dict={
                                   self.input: data,
                               })

        return output

test_images = cv2.imread('1.png') /255.0
test_images = tf.constant(test_images[np.newaxis,],dtype=tf.float32)
model_filepath = './frozen_models/frozen_graph.pb'
model = CNN(model_filepath=model_filepath)

test_prediction_onehot = model.test(data=test_images)
# Please use tf.io.gfile.GFile instead.
print(test_prediction_onehot)