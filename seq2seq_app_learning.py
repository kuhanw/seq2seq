# coding: utf-8

import json
import argparse
import pickle
import data_formatting
import time
import tensorflow as tf

from tensorflow.contrib.seq2seq.python.ops import beam_search_ops
from flask import Flask, request, jsonify

app = Flask(__name__)

'''
A small webapp to serve seq2seq results
Kuhan Wang 
17-12-01
'''

def load_graph(filename):

    graph_file = tf.gfile.GFile(filename, "rb")
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(graph_file.read())

    graph = tf.Graph().as_default()
    tf.import_graph_def(graph_def, name="prefix")
 
    return graph

@app.route('/about')
def about():
    return 'The about page'

@app.route('/api/predict', methods=['POST'])
def predict():
    start = time.time()
   
    input_json = request.get_json(force=True) 
    x_in = input_json['string']
    train_data_encoder = data_formatting.encodeSent(x_in, vocab_dict)
    inf_out = sess.run(y, feed_dict={x:[train_data_encoder], x_len:[len(train_data_encoder)]})

    inf_sent = inf_out[0]

    json_data = json.dumps({'inf_sent': decodeSent(inf_sent)})
    print("Time spent handling the request: %f" % (time.time() - start))
    return json_data											       

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, help='Filename of the frozen model.', required=True)
    parser.add_argument("--vocab_name", type=str, help='Filename of the vocabulary corresponding to the model.', required=True)

    args = parser.parse_args()
    model_filename = args.model_name
    vocab_filename = args.vocab_name
    
    print('Load model')
    model_graph = load_graph(model_filename)

    tf.reset_default_graph()
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.3)
    
    print ('Load Vocabulary')

    vocab_dict = pickle.load(open(vocab_filename, 'rb'))
    inv_map = data_formatting.createInvMap(vocab_dict)

    #Access input and output operations. 
    x = graph.get_tensor_by_name('prefix/training_model/encoder_inputs:0')
    x_len = graph.get_tensor_by_name('prefix/training_model/encoder_inputs_length:0')

    y = graph.get_tensor_by_name('prefix/training_model/decoder_pred_decode:0')
    
    session = tf.Session(graph=model_graph)
    print('Start API')
    app.run(host='0.0.0.0')
    
    session.close()

