# tensorflow-RNN
Implementing RNNs in TensorFlow for learning purposes

**1 - [simple sequence-to-sequence model with dynamic unrolling](1-seq2seq.ipynb)** (work in progress)
> Deliberately slow-moving, explicit tutorial. I tried to thoroughly explain everything that I found in any way confusing.

> Implements simple seq2seq model described in [Sutskever at al., 2014](https://arxiv.org/abs/1409.3215) and tests it against toy memorization task.

![1-seq2seq](pictures/1-seq2seq.png)

**2 - [advanced dynamic seq2seq](2-seq2seq-advanced.ipynb)** (work in progress)
> Encoder is bidirectional now. Decoder is implemented using awesome `tf.nn.raw_rnn`. It feeds previously generated tokens during training as inputs, instead of target sequence.

![2-seq2seq-feed-previous](pictures/2-seq2seq-feed-previous.png)

More to follow.
