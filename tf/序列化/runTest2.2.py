import tensorflow as tf
import cv2
import numpy as np

def wrap_frozen_graph(graph_def, inputs, outputs, print_graph=False):
    def _imports_graph_def():
        tf.compat.v1.import_graph_def(graph_def, name="")

    wrapped_import = tf.compat.v1.wrap_function(_imports_graph_def, [])
    import_graph = wrapped_import.graph

    if print_graph == True:
        print("-" * 50)
        print("Frozen model layers: ")
        layers = [op.name for op in import_graph.get_operations()]
        for layer in layers:
            print(layer)
        print("-" * 50)

    return wrapped_import.prune(
        tf.nest.map_structure(import_graph.as_graph_element, inputs),
        tf.nest.map_structure(import_graph.as_graph_element, outputs))


# Load frozen graph using TensorFlow 1.x functions
with tf.io.gfile.GFile("./frozen_models/frozen_graph.pb", "rb") as f:
    graph_def = tf.compat.v1.GraphDef()
    loaded = graph_def.ParseFromString(f.read())

# Wrap frozen graph to ConcreteFunctions
frozen_func = wrap_frozen_graph(graph_def=graph_def,
                                inputs=["x:0"],
                                outputs=["Identity:0"],
                                print_graph=True)

test_images = cv2.imread('1.png') /255.0
test_images = tf.constant(test_images[np.newaxis,],dtype=tf.float32)
# cv2.imshow('1', test_images)
# cv2.waitKey(0)

print("-" * 50)
print("Frozen model inputs: ")
print(frozen_func.inputs)
print("Frozen model outputs: ")
print(frozen_func.outputs)

# Get predictions for test images
frozen_graph_predictions = frozen_func(x=tf.constant(test_images))[0]

# Print the prediction for the first image
print("-" * 50)
print("Example TensorFlow frozen graph prediction reference:")
print(frozen_graph_predictions[0].numpy())


