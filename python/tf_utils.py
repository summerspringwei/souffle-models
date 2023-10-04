
import tensorflow as tf
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2


def tf_freeze_keras_model(keras_model, frozen_out_path, frozen_graph_filename, input_tensors_spec=None):
    """Save tensorflow keras model as frozen pb file

    keras_model: model created using keras api
    frozen_out_path: path of the directory where you want to save your model
    frozen_graph_filename: name of the .pb file
    """
    
    full_model = tf.function(lambda x: keras_model(x))
    if input_tensors_spec == None:
        spec_list = []
        for input in keras_model.inputs:
            spec_list.append(tf.TensorSpec(input.shape, input.dtype))
        full_model = full_model.get_concrete_function(tuple(spec_list))
        # full_model = full_model.get_concrete_function(
        #     tf.TensorSpec(keras_model.inputs[0].shape, keras_model.inputs[0].dtype))
    else:
        full_model = full_model.get_concrete_function(input_tensors_spec)
    # Convert to tflite model
    # tflite_converter = tf.lite.TFLiteConverter.from_concrete_functions(full_model)
    # tflite_converter.convert()
    
    # Get frozen ConcreteFunction
    frozen_func = convert_variables_to_constants_v2(full_model)
    frozen_func.graph.as_graph_def()
    layers = [op.name for op in frozen_func.graph.get_operations()]
    print("-" * 60)
    print("Frozen model layers: ")
    for layer in layers:
        print(layer)
    print("-" * 60)
    print("Frozen model inputs: ")
    print(frozen_func.inputs)
    print("Frozen model outputs: ")
    print(frozen_func.outputs)
    # Save frozen graph to disk
    tf.io.write_graph(graph_or_graph_def=frozen_func.graph,
                    logdir=frozen_out_path,
                    name=f"{frozen_graph_filename}.pb",
                    as_text=False)
    # Save its text representation
    tf.io.write_graph(graph_or_graph_def=frozen_func.graph,
                    logdir=frozen_out_path,
                    name=f"{frozen_graph_filename}.pbtxt",
                    as_text=True)



def tf_load_frozen_model(model_filepath, fetches, feed_dict):
    with tf.compat.v1.gfile.GFile(model_filepath, 'rb') as f:
            graph_def = tf.compat.v1.GraphDef()
            graph_def.ParseFromString(f.read())
    
    tf.compat.v1.import_graph_def(graph_def)
    with tf.compat.v1.Session() as sess:
        graph = tf.compat.v1.get_default_graph()
        print(type(graph))
        init=tf.compat.v1.global_variables_initializer()
        sess.run(init)
        print(sess.run(fetches, feed_dict=feed_dict))
        # for node in graph.get_operations():
        #     print(node)
    # tf.compat.v1.import_graph_def(graph_def)
    # tf.compat.v1.disable_eager_execution()
    # tf.compat.v1.global_variables_initializer()
    # g = tf.compat.v1.Graph()
    # with g.as_default():
    #     sess = tf.compat.v1.Session(graph=graph_def)
    #     print(sess.run(fetches, feed_dict=feed_dict))
    
