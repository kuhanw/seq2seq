# seq2seq-learning

### Introduction

This my implementation of a seq2seq Neural translation model using the Tensorflow API. There are many tutorials and examples on the internet demonstrating how to implement a seq2seq translation model in Tensorflow, including from Google (https://github.com/tensorflow/nmt), I studied many of them and recreated my own as a way to teach myself the core functionality of Tensorflow and to internalize my understanding of many of the underlying ideas behind seq2seq and deep learning in general.

### Requirements
Python 3.5, Tensorflow 1.3, Pandas 0.18

### Execution

The training code can be launched from seq2seq_training.py, for example: 

python seq2seq_training.py -name test_set_name -s /path_to_chkpt_folder/ -cells 128 -n_layers 2 -n_embedding 256 -layer_dropout 1 -recurrent_dropout 0.5 -n_epochs 100 -minibatch_size 32 -vocab /path_to_vocab/vocabulary.pkl -data /path_to_data/data_file.pkl

where data_file.pkl andd vocabulary.pkl are Python 3 compatible pickles consisting of:
  
  - data_file.pkl, rows of encoder, decoder inputs in numerical representation as a pandas dataframe i.e., 
    - Encoder_input[4, 5, 3, 1], decoder_target:[123, 44, 5 ,1093, 1]
  - vocabulary.pkl, a dictionary of key:value mappings between string tokens and integers,
  
representative examples can be found in this github repository under: https://github.com/kuhanw/processed_data.

Once you have a saved a model, at any time you can perform inference (i.e. return a response with only the encoder input) by
executing seq2seq_training.py on the checkpoint file:

python seq2seq_inference.py -r checkpoint_file -cells 256 -n_layers 3 -n_embedding 256 -beam_length 10 -vocab vocabulary.pkl	-freeze frozen_model_path -input encoder_input 

where

  - checkpoint_file is the checkpoint to restore from
  - encoder_input is the string input to the model
  - frozen_model_path is a optional param if you wish to create a frozen instance of the model for serving as a API.
  
A simple API serving the model will be added at a later date.

## Decoding and Language Models

### Introduction

Once I had familiarized myself with the seq2seq API and its basic functionality, as an exercise I wanted to see if I can go a step further and implement some ideas from research papers that do not exist as a ready baked APIs. 

I was inspired by this, [arXiv:1510.03055 [cs.CL]](https://arxiv.org/abs/1510.03055), on suppressing generic responses in seq2seq decoder responses. 

I wanted to build my implementation directly into Tensorflow as opposed to performing the decoding externally in python as in [here](https://github.com/Marsan-Ma/tf_chatbot_seq2seq_antilm). In order to do this, I set out on a long journey, beginning with understanding how exactly RNN decoders work...

### Decoding

The goal of decoding in a seq2seq model is to find the the target sequence (*T*) given the source sequence (*S*), i.e. *max(P(T|S))* for all possible *T*'s. Where *T* is a sequence of tokens of unknown length.

In a seq2seq model inference is performed by passing the final state of the encoder input to the decoder network and iteratively generating the output sequence.

Practically this means we begin by initializing the decoder,

  `finished, first_inputs, initial_state = inference_decoder.initialize()`

which provides the initial input and state from the last state of the encoder, at this time we say we are at step 0.

The `first_input` is of dimension [batch_size, beam_width, embedding_size], representing the embedded representation of the first "token" (in this case, from the encoder).

The `initial_state` represents the decoder network in terms of the hidden and cell states of the LSTM and the attention states, at step 0 these are initialized to zero.

The `cell_state` represents the vocabulary at a given time step. We can pass the `cell_state` (which is of size [batch_size, beam_width, n_cells] through a fully connected dense layer with size equal to the vocabulary size to obtain a representation of [batch_size, beam_width, vocab_size], if we apply a softmax layer to this output, the elements of the output can then be interpreted as the probability of emission for each vocabulary term.

In order to proceed to the next time step, the current "best" token is selected via its probability and an embedded representation, `next_inputs = inference_decoder._embedding_fn(token_id)`, of it is passed along with the current `cell_state` back into the decoder network, 

  `cell_outputs, next_cell_state = inference_decoder.decoder._cell(next_inputs, current_cell_state)`

generating a new network cell state and a output. If we encounter the special end token, *\<EOS\>*, we terminate the decoding. Otherwise we continuously repeat the process.

## Selecting the "Best" Token

The rank of best to worst tokens at each time step can follow a number of heuristics. 

As the vocabulary size is typically large, it is computationally too expensive to perform a full search and enumerate all sequence combinations to find the one that maximizes *P(T|S)*. During greedy decoding at each state we simply select the "best" token according to the softmax of each vocabulary term.

An alternatie is beam search, at each time step we keep the top *N* best sequences according to a heuristic (i.e. sum of softmax of tokens), creating a truncated breadth first search. A beam search decoder with a beam size of 1 is a greedy decoder, if the beam width, *N* is of the vocabulary size it is equivalent to searching the whole space.

Regardless, at each step we have to order the vocabulary by a ranking method. For a large corpus, there will typically be a overabundance of common replies and phrases and tokens. Using just the decoder output a typical seq2seq model can be biased towards emitting common sequences ('Thank you, You are welcome, I don't know', I am not sure...). 

## Anti-LM

We will take our queue from [arXiv:1510.03055 [cs.CL]](https://arxiv.org/abs/1510.03055) and introduce an anti-Language model.  Anti-LM being a fancy phrase to mean we will somehow quantify the frequency of target sequences *not* predicated on the source sequence in the (but not necessarily limited to!) corpus and use this information to reward or penalize the decoder so as to encourage diversity and punish generic responses.

Practically, this means modifying the Tensorflow beam search decoder, `tf.contrib.seq2seq.BeamSearchDecoder`, to rank the decoder outputs by a new heuristic,

*Score = log(P(T|S) - L \* log(P(T))*,

where *P(T|S)* is the original decoder *Score*, to which we now subtract the probability of the sequence, *T*. *L* is a strength parameter to tune how strong we want this Anti-LM effect to be. 

For practical and technical reasons, see equation (12) of [arXiv:1510.03055 [cs.CL]](https://arxiv.org/abs/1510.03055), we will restrict the correction only up to nth step in the decoding. We will control this via a new parameter *y*. Sequence steps smaller than *y* will be corrected, steps beyond will remain untouched.

## Generating P(T) and correcting the Score

I was not sure how the authors of the original paper generated the values of P(T) during decoding. As an ansatz, I simply tabulated them from the training corpus. In practice this means building n-gram models out of the corpus where "n" represents the sequence length and inserting these at run-time during decoding. This is what my code does.

At decoding time we load these precomputed tables into Tensorflow and at each step in the beam search we do a look up of each current beam and modify the score (log P(T|S)) for all possible next time steps by their sequence probabilities P(T). The overall effect is then to guide each beam down a "sub-optimal" path that it would otherwise not consider but produce more interesting responses.

In code, we introduce a while loop inside the step function of `tf.contrib.seq2seq.BeamSearchDecoder`, this while loop cycles through each current beam and maps them to counts of all possible next sequence values from the training corpus, `n_grams_tf` ,

  `matched_seqs = tf.to_int32(tf.equal(n_grams_tf, beam_pad[current_beam]))`

  `y_equal  = tf.equal(current_beam_step, tf.reduce_sum(matched_seqs[:,:current_beam_step], axis=1))`
  
   `...`
  
  `y_diff_gather = tf.gather(n_grams_tf, tf.where(y_equal))`

  `last_token_ids = y_diff_gather[:,0][:,-2]`

  `indices = tf.reshape(last_token_ids, [tf.shape(last_token_ids)[0], 1])`

  `last_token_counts = y_diff_gather[:,0][:,-1]`

  `total_count = tf.reduce_sum(last_token_counts)`

  `test_add_result = tf.scatter_nd(indices=indices, 
              updates=last_token_counts, shape=scatter_base)/total_count`

  `test_add_result = tf.log(test_add_result + 10e-10)`,

where sequences that do not appear in the corpus are assigned probability zero (i.e. they are untouched, more on this at the end). The result of the while loop is a tensor of shape [beam_width, vocab_size] at each time step which we add onto the cell output logits to obtain the anti-LM corrected scores, from which then the `tf.nn.top_k` scoring `word_indices` and `next_beam_scores` are selected.

## Results

Here are some results from the revised decoder, showing the top three ranked beam search results,

Source Sequence| Target Sequence | Rank| Decoder 
---|--- | --- |---
'what are you doing tonight?'|'i m go to be home tomorrow'|1|No Anti-LM
'what are you doing tonight?'|'i m go to be home for thanksgiv' | 2 | No Anti-LM
'what are you doing tonight?'|'i m go to be home for a while'| 3| No Anti-LM
'what are you doing tonight?'|'chillin in nyc'|1| *L*=0.4, *y*=1
'what are you doing tonight?'|'chillin on a spiritu level', |2| *L*=0.4, *y*=1
'what are you doing tonight?'|'chillin in the citi'| 3| *L*=0.4, *y*=1
'what are you doing tonight?'|'rosemari <unk> hbu'|1|*L*=0.4, *y*=2 
  'what are you doing tonight?'|'rosemari <unk> hbu ? ?' |2|*L*=0.4, *y*=2
  'what are you doing tonight?'|'rosemari <unk> hbu ? ? ?'|3|*L*=0.4, *y*=2
'what are you doing tonight?'|'chillin here in nyc'|1|*L*=0.4, *y*=3
 'what are you doing tonight?'|'nm but i m just chillin'|2|*L*=0.4, *y*=3
'what are you doing tonight?'|'nm but i m go to get some sleep'|3 | *L*=0.4, *y*=3

You can see even with our simple model, the idea works and can return sensible results. Which brings us to the cavaet that there is a need to tune *L* and *y* as hyperparameters. One can see that in this instance, the case *y*=2 produced a strange result. The inference results will also be sensitive to the construction of the n-gram model (see next section).

## Future Steps

There are a couple of flaws to this method of constructing *P(T)* and implementation. From a technical, coding perspective I went through a lot of contortions to load a precomputed table at run time for decoding in Tensorflow. Much of this comes down to the functional programming nature of parts ofTensorflow. As a Tensorflow novice, it was an immensely useful learning experience, but may not be technically elegant.

Conceptually, as the "n" of the n-gram model grows the sequence permutations rapidly grow so it becomes computationally unfeasible to store such massive tables in memory during run time. In addition, in any real world situation the decoder will encounter input sequences that were never presented in the training corpus, our naive n-gram model would then state *P(T)=0*. 

The first problem can be overcome by applying smarter data structures for storage and during computation. I have not had time to do any implementation but tries can be a natural solution for storing massive n-gram models, [http://pages.di.unipi.it/pibiri/papers/SIGIR17.pdf]. We can additionally redefine *P(T)* to be predicated only on *m* prior tokens instead of the full sequence, 

P(T) = P(t<sub>n</sub>) P(t<sub>n-1</sub>) P(t<sub>n-2</sub>)... P(t<sub>1</sub>) ~ P(t<sub>n</sub>) P(t<sub>n-1</sub>) P(t<sub>n-2</sub>)... P(t<sub>n-m</sub>), 

for *n=[1..Sequence Length]*, This will have the effect of dramatically controlling and bounding our n-gram tables and is commonly used in n-gram models to put an upper bound on their size.

For the latter case we can apply smoothing to introduce non-zero probabilities for new sequences. The easiest and most naive approach is simply to add 1 to unknown sequences. More advance and sophisticated methods have readily implemented avaliablity in [NLTK](http://www.nltk.org/_modules/nltk/probability.html).

## Retrospectives

What started of as a simple idea turned out to be a fairly large and substantial undertaking. A considerable amount of effort was expended in compressing the simple idea of anti-LMs into the framework and syntax of Tensorflow particularly in terms of flow control and shape and reshaping of tensors. Each step of the implementation revealed ignorances on my part regarding not just the Tensorflow framework but my core understanding of the basic ideas behind seq2seq such as RNN, decoders that would not have been exposed had I stuck to the pre-baked API functionalities. Many critical nuances are not apparent until one tries to implement a model at a fundamental level.

Thankfully, the results are reasonably sound and do not just contain gibberish. As well, now that we have this framework there are natural pathways to improvement. I have highlighted some of them in the preceding section. Technically the challenge is to implement these ideas in the Tensorflow framework in a elegant and computational efficient way!


### To do list

  - Update argparser
  - Implement a bidirectional encoder
  - Implement anti-lm for batch decoding
  - <s>Clean up API code to use frozen models instead of restoring from checkpoint</s>
  - <s>Clean up notebooks</s>
  - <s>Clean up text preprocessing code</s>

### References and Readings

These are some of the tutorials and implementations of seq2seq that I found most useful.

- https://github.com/ematvey/tensorflow-seq2seq-tutorials, these jupyter notebooks are what got me started when I was starting from zero and did not know much about Tensorflow or seq2seq. Note the final advance notebook with attention is using the old Tensorflow 1.0 API.
- A more fleshed out and advance implementation in Tensorflow 1.3 which I based my implementation on. Reverse engineering this code really helped me understand the core mechanics and syntax of Tensorflow, https://github.com/JayParks/tf-seq2seq.
- A still more advance implementation with more bells and whistles, https://github.com/Marsan-Ma/tf_chatbot_seq2seq_antilm.
