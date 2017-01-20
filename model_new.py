# Working with TF commit 24466c2e6d32621cd85f0a78d47df6eed2c5c5a6

import math

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow.contrib.seq2seq as seq2seq
from tensorflow.contrib.layers import embedding_lookup_unique
from tensorflow.contrib.rnn import EmbeddingWrapper, LSTMCell, LSTMStateTuple

import helpers


class Seq2SeqModel():
    """ Everything is time-major """

    PAD = 0
    EOS = 1

    def __init__(self, 
                 vocab_size=10,
                 input_embedding_size=20,
                 encoder_hidden_units=3,
                 bidirectional=True,
                 attention=False,
                 debug=False):
        self.debug = debug
        self.bidirectional = bidirectional
        self.attention = attention

        self.vocab_size = vocab_size
        self.input_embedding_size = input_embedding_size
        self.encoder_hidden_units = encoder_hidden_units
        self.decoder_hidden_units = encoder_hidden_units
        if self.bidirectional:
            self.decoder_hidden_units *= 2

        if self.debug:
            self._init_debug_inputs()
        else:
            self._init_placeholders()
        
        self._init_decoder_train_connectors()
        self._init_embeddings()

        self.encoder_cell = LSTMCell(self.encoder_hidden_units)
        if self.bidirectional:
            self._init_bidirectional_encoder()
        else:
            raise NotImplementedError
        
        self.decoder_cell = LSTMCell(self.decoder_hidden_units)
        if self.attention:
            self._init_attention_decoder()
        else:
            self._init_simple_decoder()

        self._init_optimizer()

    def _init_debug_inputs(self):
        x = [[5, 6, 7],
             [7, 6, 0],
             [0, 7, 0]]
        xl = [2, 3, 1]
        self.encoder_inputs = tf.constant(x, dtype=tf.int32, name='encoder_inputs')
        self.encoder_inputs_length = tf.constant(xl, dtype=tf.int32, name='encoder_inputs_length')

        self.decoder_targets = tf.constant(x, dtype=tf.int32, name='decoder_targets')
        self.decoder_targets_length = tf.constant(xl, dtype=tf.int32, name='decoder_targets_length')
    
    def _init_placeholders(self):
        self.encoder_inputs = tf.placeholder(
            shape=(None, None),
            dtype=tf.int32,
            name='encoder_inputs',
        )
        self.encoder_inputs_length = tf.placeholder(
            shape=(None,),
            dtype=tf.int32,
            name='encoder_inputs_length',
        )

        self.decoder_targets = tf.placeholder(
            shape=(None, None),
            dtype=tf.int32,
            name='decoder_targets'
        )
        self.decoder_targets_length = tf.placeholder(
            shape=(None,),
            dtype=tf.int32,
            name='decoder_targets_length',
        )
    
    def _init_decoder_train_connectors(self):
        """
        During training, `decoder_targets` would serve as basis for both `decoder_inputs` 
        and decoder logits. This means that their shapes should be compatible.

        Here we do a bit of plumbing to set this up.
        """
        with tf.name_scope('DecoderTrainFeeds'):
            sequence_size, batch_size = tf.unstack(tf.shape(self.decoder_targets))

            EOS_SLICE = tf.ones([1, batch_size], dtype=tf.int32) * self.EOS
            PAD_SLICE = tf.ones([1, batch_size], dtype=tf.int32) * self.PAD

            self.decoder_train_inputs = tf.concat_v2([EOS_SLICE, self.decoder_targets], axis=0)
            self.decoder_train_length = self.decoder_targets_length + 1

            decoder_train_targets = tf.concat_v2([self.decoder_targets, PAD_SLICE], axis=0)
            decoder_train_targets_seqlen, _ = tf.unstack(tf.shape(decoder_train_targets))
            decoder_train_targets_eos_mask = tf.transpose(
                tf.one_hot(self.decoder_train_length - 1, decoder_train_targets_seqlen, dtype=tf.int32),
                [1, 0])
            decoder_train_targets = tf.add(
                decoder_train_targets,
                decoder_train_targets_eos_mask,
            )  # hacky way using one_hot to put EOS symbol at the end of target sequence

            self.decoder_train_targets = decoder_train_targets

            self.loss_weights = tf.ones([
                batch_size,
                tf.reduce_max(self.decoder_train_length)
            ], dtype=tf.float32, name="loss_weights")

    def _init_embeddings(self):
        with tf.variable_scope("embedding") as scope:

            # Uniform(-sqrt(3), sqrt(3)) has variance=1.
            sqrt3 = math.sqrt(3)
            initializer = tf.random_uniform_initializer(-sqrt3, sqrt3)

            self.embedding_matrix = tf.get_variable(
                name="embedding_matrix",
                shape=[self.vocab_size, self.input_embedding_size],
                initializer=initializer,
                dtype=tf.float32)
            
            self.encoder_inputs_embedded = embedding_lookup_unique(
                self.embedding_matrix, self.encoder_inputs)
            
            self.decoder_train_inputs_embedded = embedding_lookup_unique(
                self.embedding_matrix, self.decoder_train_inputs)

    def _init_bidirectional_encoder(self):

        with tf.variable_scope("BidirectionalEncoder") as scope:
            
            ((encoder_fw_outputs,
              encoder_bw_outputs),
             (encoder_fw_state,
              encoder_bw_state)) = (
                tf.nn.bidirectional_dynamic_rnn(cell_fw=self.encoder_cell,
                                                cell_bw=self.encoder_cell,
                                                inputs=self.encoder_inputs_embedded,
                                                sequence_length=self.encoder_inputs_length,
                                                time_major=True,
                                                dtype=tf.float32, 
                                                scope=scope)
                )

            self.encoder_outputs = tf.concat_v2((encoder_fw_outputs, encoder_fw_outputs), 2)

            encoder_state_c = tf.concat_v2(
                (encoder_fw_state.c, encoder_bw_state.c), 1)

            encoder_state_h = tf.concat_v2(
                (encoder_fw_state.h, encoder_bw_state.h), 1)

            self.encoder_state = LSTMStateTuple(c=encoder_state_c, h=encoder_state_h)

    def _init_simple_decoder(self):

        with tf.variable_scope("SimpleDecoder") as scope:
            decoder_fn_train = seq2seq.simple_decoder_fn_train(encoder_state=self.encoder_state)
            
            (
                self.decoder_outputs_train,
                self.decoder_state_train,
                self.decoder_context_state_train
            ) = (
                seq2seq.dynamic_rnn_decoder(
                    cell=self.decoder_cell,
                    decoder_fn=decoder_fn_train,
                    inputs=self.decoder_train_inputs_embedded,
                    sequence_length=self.decoder_train_length,
                    time_major=True,
                    scope=scope,
                )
            )
            
            def output_fn(outputs):
                return tf.contrib.layers.linear(outputs, self.vocab_size, scope=scope)
            
            self.decoder_logits_train = output_fn(self.decoder_outputs_train)
            
            self.decoder_prediction_train = tf.argmax(self.decoder_logits_train, axis=-1, name='decoder_prediction')
            
            scope.reuse_variables()
            
            decoder_fn_inference = seq2seq.simple_decoder_fn_inference(
                output_fn=output_fn,
                encoder_state=self.encoder_state,
                embeddings=self.embedding_matrix,
                start_of_sequence_id=1,
                end_of_sequence_id=1,
                maximum_length=tf.reduce_max(self.encoder_inputs_length) + 3,
                num_decoder_symbols=self.vocab_size,
            )
            
            (decoder_outputs_inference,
             decoder_state_inference,
             decoder_context_state_inference) = (
                seq2seq.dynamic_rnn_decoder(
                    cell=self.decoder_cell,
                    decoder_fn=decoder_fn_inference,
                    time_major=True,
                    scope=scope,
                )
            )
    
    def _init_attention_decoder(self):
        # Somehow attention slows down convergence significantly and makes it volatile.

        self.decoder_cell = LSTMCell(self.decoder_hidden_units)

        with tf.variable_scope("AttentionDecoder") as scope:
            (attention_keys,
             attention_values,
             attention_score_fn,
             attention_construct_fn) = seq2seq.prepare_attention(
                attention_states=self.encoder_state.h, 
                attention_option="bahdanau", 
                num_units=self.decoder_hidden_units,
            )
            
            decoder_fn_train = seq2seq.attention_decoder_fn_train(
                encoder_state=self.encoder_state,
                attention_keys=attention_keys,
                attention_values=attention_values,
                attention_score_fn=attention_score_fn,
                attention_construct_fn=attention_construct_fn,
                name='attention_decoder'
            )
            
            (
                self.decoder_outputs_train,
                self.decoder_state_train,
                self.decoder_context_state_train
            ) = (
                seq2seq.dynamic_rnn_decoder(
                    cell=self.decoder_cell,
                    decoder_fn=decoder_fn_train,
                    inputs=self.decoder_train_inputs_embedded,
                    sequence_length=self.decoder_train_length,
                    time_major=True,
                    scope=scope,
                )
            )
            
            def output_fn(outputs):
                return tf.contrib.layers.linear(outputs, self.vocab_size, scope=scope)
            
            self.decoder_logits_train = output_fn(self.decoder_outputs_train)
            
            self.decoder_prediction_train = tf.argmax(self.decoder_logits_train, axis=-1, name='decoder_prediction_train')
            
            scope.reuse_variables()
            
            decoder_fn_inference = seq2seq.attention_decoder_fn_inference(
                output_fn=output_fn,
                encoder_state=self.encoder_state,
                attention_keys=attention_keys,
                attention_values=attention_values,
                attention_score_fn=attention_score_fn,
                attention_construct_fn=attention_construct_fn,
                embeddings=self.embedding_matrix,
                start_of_sequence_id=self.EOS,
                end_of_sequence_id=self.EOS,
                maximum_length=tf.reduce_max(self.encoder_inputs_length) + 3,
                num_decoder_symbols=self.vocab_size,
            )
            
            (
                self.decoder_outputs_inference,
                self.decoder_state_inference,
                self.decoder_context_state_inference
            ) = (
                seq2seq.dynamic_rnn_decoder(
                    cell=self.decoder_cell,
                    decoder_fn=decoder_fn_inference,
                    time_major=True,
                    scope=scope,
                )
            )

    def _init_optimizer(self):
        logits = tf.transpose(self.decoder_logits_train, [1, 0, 2])
        targets = tf.transpose(self.decoder_train_targets, [1, 0])
        self.loss = seq2seq.sequence_loss(logits=logits, targets=targets,
                                          weights=self.loss_weights)
        self.train_op = tf.train.AdamOptimizer().minimize(self.loss)
    

import traceback
import warnings
import sys

def warn_with_traceback(message, category, filename, lineno, file=None, line=None):
    traceback.print_stack()
    log = file if hasattr(file,'write') else sys.stderr
    log.write(warnings.formatwarning(message, category, filename, lineno, line))

warnings.showwarning = warn_with_traceback


if 'debug' in sys.argv:
    tf.reset_default_graph()
    sess = tf.InteractiveSession()
    model = Seq2SeqModel(debug=True, attention=True)
    sess.run(tf.global_variables_initializer())
    print(model.loss.eval())

def train():
    tf.reset_default_graph()
    sess = tf.InteractiveSession()
    model = Seq2SeqModel()
    sess.run(tf.global_variables_initializer())
    batch_size = 100
    max_batches = 100000
    batches_in_epoch = 1000
    batches = helpers.random_sequences(length_from=3, length_to=8,
                                       vocab_lower=2, vocab_upper=10,
                                       batch_size=batch_size)
    def next_feed():
        batch = next(batches)
        inputs_, inputs_length_ = helpers.batch(batch)
        return {
            model.encoder_inputs: inputs_,
            model.encoder_inputs_length: inputs_length_,
            model.decoder_targets: inputs_,
            model.decoder_targets_length: inputs_length_,
        }
    loss_track = []
    nested_transpose = lambda l: [x.T for x in l]
    try:
        for batch in range(max_batches):
            fd = next_feed()
            _, l = sess.run([model.train_op, model.loss], fd)
            loss_track.append(l)

            if batch == 0 or batch % batches_in_epoch == 0:
                print('batch {}'.format(batch))
                print('  minibatch loss: {}'.format(sess.run(model.loss, fd)))
                for i, (e_in, d_tg, dt_in, dt_tg, dt_pred) in enumerate(zip(
                        fd[model.encoder_inputs].T, 
                        fd[model.decoder_targets].T,
                        *nested_transpose(sess.run([
                            model.decoder_train_inputs,
                            model.decoder_train_targets,
                            model.decoder_prediction_train,
                        ], fd))
                    )):
                    print('  sample {}:'.format(i + 1))
                    print('    enc input           > {}'.format(e_in))
                    #print('    dec target          > {}'.format(d_tg))
                    #print('    dec train input     > {}'.format(dt_in))
                    #print('    dec train target    > {}'.format(dt_tg))
                    print('    dec train predicted > {}'.format(dt_pred))
                    if i >= 2:
                        break
                print()
    except KeyboardInterrupt:
        print('training interrupted')

    import matplotlib.pyplot as plt
    plt.plot(loss_track)
    print('loss {:.4f} after {} examples (batch_size={})'      .format(loss_track[-1], len(loss_track)*batch_size,
                batch_size))
