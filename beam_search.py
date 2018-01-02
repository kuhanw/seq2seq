import copy
import data_formatting
import tensorflow as tf
import numpy as np
from collections import OrderedDict

def executeStep(session, model, encoder_input, encoder_input_length, decoder_target, decoder_target_length):

    test = session.run([model.decoder_pred_train_prob],
           feed_dict = { 'training_model/encoder_inputs:0':encoder_input, 
                         'training_model/encoder_inputs_length:0':encoder_input_length,
                         'training_model/decoder_targets:0':decoder_target, 
                         'training_model/decoder_targets_length:0':decoder_target_length})

    return test[0]

def argMaxTopN(array, n):
    
    ind = np.argpartition(array, -n)[-n:]    
    
    return ind[np.argsort(array[ind])], array[ ind[np.argsort(array[ind])]]

def beamDecode(beam_width, model, input_sent, checkpoint, max_decode_steps, question_mark_token, debug=False):

    beams = [{'eos':0, 'encoder_seq':input_sent, 'encoder_seq_length': len(input_sent), 
           'decoder_seq':[], 'decoder_seq_length':0, 'total_log_prob':0} for i in range(beam_width)]

    config = tf.ConfigProto(device_count = {'GPU': 0})

    session = tf.Session(config=config)

    saver = tf.train.Saver()

    saver.restore(session, checkpoint)

    for pos in range(max_decode_steps):
        
    #while beams_remaining!=0:    
        print ('CURRENT STEP', pos)

        encoder_input = [beams[i]['encoder_seq'] for i in range(len(beams)) if beams[i]['eos']!=1]
        encoder_input_length = [beams[i]['encoder_seq_length'] for i in range(len(beams)) if beams[i]['eos']!=1]
        decoder_targets = [beams[i]['decoder_seq'] for i in range(len(beams)) if beams[i]['eos']!=1]
        decoder_targets_length = [beams[i]['decoder_seq_length'] for i in range(len(beams)) if beams[i]['eos']!=1]

        beams_remaining = len(encoder_input)

        if beams_remaining == 0 :
            print ('Finished Beam search')
            break

        if debug==True: print ('seq_lengths', encoder_input_length, decoder_targets_length)
        
        logits = executeStep(session, model, encoder_input, encoder_input_length, decoder_targets, decoder_targets_length)

        beam_results = np.concatenate([argMaxTopN(logits[i][-1], beam_width) for i in range(beams_remaining)])
        if debug==True: print (beam_results)

        beam_results_vocab = list(np.concatenate([i for idx, i in enumerate(beam_results) if idx%2==0]))
        beam_results_probs = list(np.concatenate([i for idx, i in enumerate(beam_results) if idx%2==1]))

        beam_results_vocab = list(OrderedDict.fromkeys(beam_results_vocab))

        beam_results_probs = [beam_results_probs[beam_results_vocab.index(i)] for i in beam_results_vocab]

        #beam_results_probs = list(OrderedDict.fromkeys(np.concatenate([i for idx, i in enumerate(beam_results) if idx%2==1])))
        if debug==True: print ('beam_results_vocab ', beam_results_vocab)

        if debug==True: print ('beam_results_probs ', beam_results_probs)
        top_n_index = np.argsort(beam_results_probs)[::-1][:beam_width]
        if debug==True: print ('top_n_index ', top_n_index)

        temp_beam_list = []

        for idx_beam, beam in enumerate(beams):
            temp_beam = copy.deepcopy(beam)

            for idx, arg in enumerate(top_n_index[:beam_width]):

                if debug==True: print ('IDX, ARG', idx, arg, idx_beam)
                if debug==True: print ('CURRENT BEAM', temp_beam)

                if temp_beam['eos']==1:
                    continue

                temp_beam['total_log_prob']+=beam_results_probs[arg]

                temp_beam['decoder_seq'].append(beam_results_vocab[arg])

                if beam_results_vocab[arg] == 1 or beam_results_vocab[arg]==question_mark_token: #Hacking for question mark
                    temp_beam['eos'] = 1

                temp_beam['decoder_seq_length']+=1

                if True not in [temp_beam['decoder_seq']==current_beam['decoder_seq'] for current_beam in temp_beam_list]:

                    temp_beam_list.append(temp_beam)

                temp_beam = copy.deepcopy(beam)

                if debug==True: print ('TEMP BEAM', temp_beam)
        beams_all = sorted(temp_beam_list, key=lambda x: x['total_log_prob'], reverse=True)
        if debug==True: print (beams_all)
        beams = beams_all[:beam_width]
        if debug==True: print (beams)

    session.close()
    
    if debug==True: return beams
    else: return [{key:beam[key] for key in ['decoder_seq', 'total_log_prob']} for beam in beams]