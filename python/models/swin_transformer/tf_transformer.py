import os
import sys
import math

import numpy as np
import tensorflow as tf


def tf_query_key_mul_softmax(query_layer, key_layer, size_per_head):
    """Taken from bert github
    `query_layer` = [B, N, F, H]
    `key_layer` = [B, N, T, H]
  """
    # Take the dot product between "query" and "key" to get the raw
    # attention scores.
    # `attention_scores` = [B, N, F, T]
    attention_scores = tf.matmul(query_layer, key_layer, transpose_b=True)
    attention_scores = tf.multiply(attention_scores,
                                   1.0 / math.sqrt(float(size_per_head)))
    # Normalize the attention scores to probabilities.
    # `attention_probs` = [B, N, F, T]
    attention_scores = tf.nn.softmax(attention_scores)

    return attention_scores


def tf_freeze_query_key_mul_softmax(batch_size, num_heads, seq_length,
                                    size_per_head, model_name):
    sys.path.insert(0, '/home/xiachunwei/Software/tensor-compiler/src/ocv')
    from tf_utils import tf_freeze_keras_model
    query = tf.keras.Input(shape=(num_heads, seq_length, size_per_head),
                           batch_size=batch_size,
                           dtype=tf.float16)
    key = tf.keras.Input(shape=(num_heads, seq_length, size_per_head),
                         batch_size=batch_size,
                         dtype=tf.float16)
    output = tf_query_key_mul_softmax(query, key, size_per_head)
    model = tf.keras.Model(inputs=[query, key], outputs=output)
    model_folder = os.path.join("/home/xiachunwei", "models", model_name)
    tf_freeze_keras_model(model, model_folder, model_name)
    sh_cmd = "python3 -m tf2onnx.convert --input {} --output {} --inputs {} --outputs {}".format(\
      os.path.join(model_folder, model_name+".pb"), os.path.join(model_folder, "{}.onnx".format(model_name)),\
        "x:0", "Identity:0")
    os.system(sh_cmd)
    sh_cmd = "trtexec --onnx={} --workspace=256 --fp16 --noDataTransfers --saveEngine={}".format(\
      os.path.join(model_folder, "{}.onnx".format(model_name)), os.path.join(model_folder, "{}_engine.trt".format(model_name)))
    os.system(sh_cmd)


def bench_trt_query_key_mul_softmax(model_name):
    model_folder = os.path.join("/home/xiachunwei", "models", model_name)
    sh_cmd = "trtexec --loadEngine={} --fp16 --noDataTransfers > tmp.txt".format(\
      os.path.join(model_folder, "{}_engine.trt".format(model_name)))
    os.system(sh_cmd)
    os.system("cat tmp.txt | grep 'Compute Time'")


def test_tf_query_key_mul_softmax():
    batch_size, num_heads, seq_length, size_per_head = 64, 4, 64, 64
    query = np.ones(
        (batch_size, num_heads, seq_length, size_per_head), np.float16) / 10
    key = np.ones(
        (batch_size, num_heads, seq_length, size_per_head), np.float16) / 10
    output = tf_query_key_mul_softmax(query, key, size_per_head)
    print(output)


def run_query_key_mul_softmax():
    # batch_size, num_heads, seq_length, size_per_head = 1, 12, 384, 64
    batch_size, num_heads, seq_length, size_per_head = 64, 4, 64, 32
    sub_graph_name = "query_key_mul_softmax"
    model_name = "tf_{}_{}_{}_{}_{}".format(sub_graph_name, batch_size,
                                            num_heads, seq_length,
                                            size_per_head)
    tf_freeze_query_key_mul_softmax(batch_size, num_heads, seq_length,
                                    size_per_head, model_name)
    bench_trt_query_key_mul_softmax(model_name)


if __name__ == "__main__":
    run_query_key_mul_softmax()
    # test_tf_query_key_mul_softmax()
