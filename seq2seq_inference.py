#!/usr/bin/env python

import tensorflow as tf
import data_formatting
import create_model
import pickle
import argparse

parser = argparse.ArgumentParser(description='Perform inference from a seq2seq model.')

parser.add_argument('-r', '--restore', type = str, help = 'path to chkpt file to restore.', required = False)
parser.add_argument('-cells', '--cells', type = int, help = 'Number of Hidden units.', required = True)
parser.add_argument('-n_layers', '--n_layers', type = int, help = 'Number of LSTM layers.', required = True)
parser.add_argument('-n_embedding', '--n_embedding', type = int, help = 'Dimensionality of word embedding.', required = True)
parser.add_argument('-vocab', '--vocab', type = str, help = 'Path to vocabulary pickle file.', required = True)
parser.add_argument('-lm', '--lm', type = str, help = 'Path to language model pickle file, leave blank to disable.', required = False)
parser.add_argument('-input', '--input', type = str, help = 'Encoder input to decode.', required = False)
parser.add_argument('-freeze', '--freeze', type = str, help = 'Path to save frozen model.', required=False)
parser.add_argument('-beam_length', '--beam_length', type = int, help = 'Length of beam at inference time, set to 1 to remove.', required = False)
parser.add_argument('-limit_decode_steps', '--limit_decode_steps', type = int, help = 'Limit the number of decoding steps.', required = False)

args = parser.parse_args()
print (args)

restore_path = args.restore 
vocab_path = args.vocab

vocab_dict = pickle.load(open(vocab_path, 'rb'))
inv_map = data_formatting.createInvMap(vocab_dict)

inf_model_params = {'n_cells':args.cells, 'num_layers':args.n_layers, 
                    'embedding_size':args.n_embedding, 
                    'vocab_size':len(vocab_dict) + 1,
                    'anti_lm_max_step':3, 'anti_lm_weight':0.5 
                   }

if args.limit_decode_steps is not None:
    inf_model_params['limit_decode_steps'] = args.limit_decode_steps
else:
    inf_model_params['limit_decode_steps'] = None

if args.beam_length is not None:
    inf_model_params['beam_length'] = args.beam_length
else:
    inf_model_params['beam_length'] = None
    
tf.reset_default_graph()

if parser.lm is not None:
    language_model = pickle.load(open(parser.lm, 'rb'))
else:
    language_model = None
    
with tf.variable_scope('training_model'):
    
    inf_model = create_model.Model(inf_model_params, 'infer', None, language_model)

with tf.Session() as session:
    
    session.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    saver.restore(session, args.restore)

    if args.freeze is not None:
        
        #Grab the encoder input and decoder output nodes
        relevant_nodes = [n.name for n in tf.get_default_graph().as_graph_def().node 
                         if 'decoder_pred' in n.name or 'encoder_input' in n.name]
    
        print('Input and output nodes: %s' % ', '.join(relevant_nodes))
        
        output_graph_def = tf.graph_util.convert_variables_to_constants(
        session, tf.get_default_graph().as_graph_def(), relevant_nodes)
        frozen_model_name = args.freeze + '_' + '_'.join(sorted(['%s_%s' % (key, str(inf_model_params[key])) for key in inf_model_params.keys()])) + '.pb'        
        
        with tf.gfile.GFile(frozen_model_name, 'wb') as f:
            f.write(output_graph_def.SerializeToString())
    
#        print ('model frozen under %s' % frozen_model_name)
    
    if args.input is not None:
       
        encoder_input = data_formatting.encodeSent(args.input, vocab_dict)
    
        inf_out = session.run([inf_model.encoder_inputs, inf_model.decoder_pred_decode],
                feed_dict = {'training_model/encoder_inputs:0':[encoder_input], 
                             'training_model/encoder_inputs_length:0':[len(encoder_input)]}
                             )

        encoder_input = inf_out[0]
        decoder_inference = inf_out[1]

        for idx, e_in in enumerate(encoder_input):

            print ('###%d###' % idx)

            print ('e_in', data_formatting.decodeSent(e_in, inv_map))
            beam_inf = list(zip(*decoder_inference[idx]))
            for idx_inf, dt_inf in enumerate(beam_inf):

                print ('dt_inf', data_formatting.decodeSent(dt_inf, inv_map))
                if idx_inf>1: break                

