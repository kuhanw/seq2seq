# coding: utf-8

import json
import argparse
import pickle
import helpers
import time

import tensorflow as tf
from flask import Flask, request, jsonify

app = Flask(__name__)

'''
A small webapp to serve seq2seq results
Kuhan Wang 
17-12-01
'''


def make_inference_inputs(input_seq):
    inputs_, inputs_length_ = helpers.batch(input_seq)

    return {
	       'encoder_inputs:0': inputs_,
	       'encoder_inputs_length:0': inputs_length_,
	       }

def encodeSent(sent):
    if type(sent) == str: sent = sent.split(' ')
	
    return [vocab_dict[word] if word in vocab_dict else 2 for word in sent]

def decodeSent(sent):
    return ' '.join([inv_map[i] for i in sent])

@app.route('/about')
def about():
    return 'The about page'

@app.route('/api/predict', methods=['POST'])
def predict():
    start = time.time()
   
    input_json = request.get_json(force=True) 
    x_in = input_json['string']

    inf_input = make_inference_inputs([encodeSent(x_in.split(' '))])
    y_out = session.run([op_inf, op_inf_prob], inf_input)

    inf_sent = list(zip(*y_out[0]))[0]
    inf_prob = list(zip(*y_out[1]))

    json_data = json.dumps({'inf_sent': decodeSent(inf_sent)})
    print("Time spent handling the request: %f" % (time.time() - start))
    return json_data											       

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="seq2seq_enron_encode_128_decode_256_vocab_7239_embedding_256_seq_3_49_batch_32_layers_2_enkeep_10_dekeep_10-28679", type=str, help="Metagraph filename")
    
    args = parser.parse_args()
    filename = args.model_name
    
    print('Load model')

    tf.reset_default_graph()
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.3)
    
    session = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True))
    #Import the meta graph in the current default Graph
    saver = tf.train.import_meta_graph(filename + '.meta')

    #Restore the weights
    saver.restore(session, filename)

    print ('Graph loaded, session initiated')

    graph = tf.get_default_graph()
    
	#Access operations. 
    op_inf = graph.get_tensor_by_name('Decoder/decoder_prediction_inference:0')
    op_inf_prob = graph.get_tensor_by_name('Decoder/decoder_prediction_prob_inference:0')

    print ('Load Vocabulary')
    vocab_dict = pickle.load(open('dicts\word_dict_v01_enron_py35_seq_length_3_49_sample_4256_limited_vocab.pkl', 'rb'))

    inv_map = {v: k for k, v in vocab_dict.items()}

    print('Start API')
    app.run()
    
    session.close()

