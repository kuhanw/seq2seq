# seq2seq-learning

### Introduction

This my implementation of a seq2seq Neural translation model using the Tensorflow API. There are many tutorials and examples on the internet demonstrating how to implement a seq2seq translation model in Tensorflow, including from Google (https://github.com/tensorflow/nmt), I studied many of them and recreated my own as a way to teach myself the core functionality of Tensorflow and to internalize my understanding of many of the underlying ideas behind seq2seq and deep learning in general.

### Requirements
Python 3.5, Tensorflow 1.3, Pandas 0.18

### Execution

The training code can be launched from seq2seq_training.py, for example: 

python seq2seq_training.py -s ../../efs/chkpt -cells 256 -n_layers 3 -n_embedding 256 -dropout 0.8 -beam_length 10 -minibatch_size 128 -data data_file.pkl -vocab vocabulary.pkl

where data_file.pkl andd vocabulary.pkl are Python 3 compatible pickles consisting of:
  
  - data_file.pkl, rows of encoder, decoder inputs in numerical representation as a pandas dataframe,
  - vocabulary.pkl, a dictionary of key:value mappings between string tokens and integers,
  
representative examples can be found in this github repository under: https://github.com/kuhanw/processed_data.

Once you have a checkpointed model, at any time you can perform inference (i.e. return a response with only the encoder input) by
executing seq2seq_training.py:

python seq2seq_inference.py -r checkpoint_file -cells 256 -n_layers 3 -n_embedding 256 -beam_length 10 -vocab vocabulary.pkl	-freeze frozen_model_path -input encoder_input 

where

  - checkpoint_file is the checkpoint to restore from
  - encoder_input is the string input to the model
  - frozen_model_path is a optional param if you wish to create a frozen instance of the model for serving as a API.
  
A simple API serving the model will be added at a later date.

## Decoding and Language Models

### Introduction

As an exercise I wanted to see if it was possible to incorporate a language model based on some a priori information about the corpus into my seq2seq model. I was inspired by this [arXiv:1510.03055 [cs.CL]](https://arxiv.org/abs/1510.03055). I wanted to build my implementation directly into Tensorflow as opposed to performing the decoding externally in python as in [here](https://github.com/Marsan-Ma/tf_chatbot_seq2seq_antilm). In order to do this, I set out on a long journey, beginning with understanding how decoders work...

### Decoding

The goal of decoding in a seq2seq model is to find the the target sequence (T) given the source sequence (S), i.e. max(P(T|S)) for all possible T's. T is a sequence of tokens, in abstract, of unknown length.

In a sequence to sequence model inference is performed by passing the final state of the encoder input to the decoder network and iteratively generated the output sequence.

Practically this means we begin by initializing the decoder,

  `self.finished, self.first_inputs, self.initial_state = self.inference_decoder.initialize()`

which provides the initial input and state from the last state of the encoder, at this time we say we are at step 0.

The `first_input` is of dimension [batch_size, beam_width, embedding_size], representing the embedded representation of the first "token" (in this case, from the encoder).

The `initial_state` represents the decoder network in terms of the hidden and cell states of the LSTM and the attention states, at step 0 these are initialized to zero.

The cell states represents the vocabulary at a given time step. We can pass the cell state (which is of size [batch_size, beam_width, n_cells] through a fully connected dense layer with size equal to the vocabulary size to obtain a representation of [batch_size, beam_width, vocab_size], if we apply a softmax layer to this output, the elements of the output can then be interpreted as the probability of emission for each vocabulary term.

In order to proceed to the next time step, the current "best" token is selected via its probability and an embedded representtation of it is passed along with the cell state (i.e. hidden state,cell state and attention states, hereafter called cell state) back into the decoder network, generating a new network cell state and a output. 

The output is "densified" and from it the optimal next token is selected, if it is the special end token, <EOS>, we terminate the decoding and finish. Otherwise we do an embedding lookup of the token and pass it back through the decoder network along with the current state and repeat the process.

## Selecting the "Best" Token

The rank of best to worst tokens at each time step can follow a number of heuristics. 

As the vocabulary size is typically large, it is computationally too expensive to perform a full search and enumerate all sequence combinations to find the one that maximizes P(T|S). During greedy decoding at each state we simply select the "best" token according to the softmax of each vocabulary term.

An alternatie is beam search, at each time step we keep the top N best sequence "beams" according to a heuristic (i.e. sum of softmax of tokens), thus creating a truncated breadth first search. A beam search decoder with a beam size of 1 is a greedy decoder, if the beam size = vocabulary size is equivalent to searching the whole space.

Regardless, at each step we have to order the vocabulary by a ranking method. For a large corpus, there will typically be a overabundance of common replies and phrases and tokens. Using just the decoder output a typical seq2seq model will be biased to emitting these sequences.

## Anti-LM

Instead, we will take our queue from [arXiv:1510.03055 [cs.CL]](https://arxiv.org/abs/1510.03055) and introduce an anti-Language model.  Anti-LM being a fancy phrase to mean we will somehow tally up the most common sequences in the (but not necessarily limited to!) corpus and use this information to reward or penalize the decoder so as to encourage diversity and punish generic replies.

Practically, this means modifying the Tensorflow beamsearch decoder to rank the decoder outputs by a new heuristic,

log(P(T|S) - lambda P(T),

where P(T|S) is the original decoder output, to which we now subtract the probability of the sequence, T. Lambda is a strength parameter to tune how strong we want this Anti-LM effect to be. 

Technically, we dive into the Tensorflow API code and modify the scoring function of the beamsearch to accept an addition parameter so that at each step the decoder determine the beams according to our new equation. For practical and technical purposes, we will restrict the correction only up to nth step in the decoding. 

## Generating P(T)

The original paper was unclear as to how they generated the P(T) during decoding. As an ansatz I simply tabulated them from the training corpus. In practice this means building n-gram models out of the corpus where "n" represents the sequence length and inserting these at run-time during decoding. This is what my code does. 

Here are some results,

Target Sequence | Decoder 
--- | --- 
'thank', 'you', 'for', 'your', 'support'| No Anti-LM
'is', 'a', 'great', 'idea' | Lambda=0.8, gamma=1
'thank', 'bet', 'i', 'love', 'you', '<eos>' | Lambda=0.8, gamma=4
'thank', 'for', 'share', 'i', 'appreci', 'it' | Lambda=0.1, gamma=4

You can see even with our simple model, the idea is sound and can return interesting results. It is interesting to note, when we correct only the first token (gamma=1), the entire meaning of the entire output is changed, as would be expected, while adding corrections to the first four steps return more diverse variations of the original response.

There are a couple of flaws to this method of constructing P(T). 

As "n" grows the permutations rapidly grow so it becomes computationally unfeasible to store such massive tables in memory during run time. Secondly, in any real world situation the decoder will encounter input sequences that were never presented in the training corpus thus our naive n-gram model states P(T)=0. 

The first problem can be overcome by using better data structures (I am keeping the models on disk as lists) for storage. Tries can be a good solution []. In addition, we can redefine P(T) to be predicated only m prior tokens instead of the full sequence, 

P(T) = Prod( equation).

For the latter case we can apply [smoothing](https://en.wikipedia.org/wiki/Good%E2%80%93Turing_frequency_estimation) to introduce non-zero probabilities for new sequences.

### To do list

  - Implement a bidirectional encoder
  - Implement anti-language model to increase the diversity of the decoder inference responses (Here is an implementation: https://github.com/Marsan-Ma/tf_chatbot_seq2seq_antilm)
  - <s>Clean up API code to use frozen models instead of restoring from checkpoint</s>
  - <s>Clean up notebooks</s>
  - Clean up text preprocessing code

### References and Readings

- https://github.com/ematvey/tensorflow-seq2seq-tutorials, these jupyter notebooks are what got me started when I was starting from zero and did not know much about Tensorflow or seq2seq. Note the final advance notebook with attention is using the old Tensorflow 1.0 API.
- A more fleshed out and advance implementation in Tensorflow 1.3 which I based my implementation on. Reverse engineering this code really helped me understand the core mechanics and syntax of Tensorflow, https://github.com/JayParks/tf-seq2seq.
- A still more advance implementation with more bells and whistles, https://github.com/Marsan-Ma/tf_chatbot_seq2seq_antilm.
