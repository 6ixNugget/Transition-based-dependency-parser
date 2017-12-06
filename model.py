'''Statistical modelling/parsing classes'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time

from itertools import islice
from sys import stdout
from tempfile import NamedTemporaryFile
import sys

import tensorflow as tf

from utils.model import Model
from data import load_and_preprocess_data
from data import score_arcs
from initialization import xavier_weight_init
from parser import minibatch_parse
from utils.generic_utils import Progbar

class Config(object):
    """Holds model hyperparams and data information.

    The config class is used to store various hyperparameters and dataset
    information parameters. Model objects are passed a Config() object at
    instantiation.
    """
    n_word_ids = None # inferred
    n_tag_ids = None # inferred
    n_deprel_ids = None # inferred
    n_word_features = None # inferred
    n_tag_features = None # inferred
    n_deprel_features = None # inferred
    n_classes = None # inferred
    dropout = 0.5
    embed_size = None # inferred
    hidden_size = 200 
    batch_size = 2048
    n_epochs = 10
    lr = 0.001
    
    # Part3 settings
    """
    hidden2_size = 300
    addHidden2 = True
    """

class ParserModel(Model):
    """
    Implements a feedforward neural network with an embedding layer and single hidden layer.
    This network will predict which transition should be applied to a given partial parse
    configuration.
    """

    def add_placeholders(self):
        """Generates placeholder variables to represent the input tensors

        These placeholders are used as inputs by the rest of the model
        building and will be fed data during training.  Note that when
        "None" is in a placeholder's shape, it's flexible (so we can use
        different batch sizes without rebuilding the model).

        Adds following nodes to the computational graph

        word_id_placeholder:
            Word feature placeholder of shape (None, n_word_features),
            type tf.int32
        tag_id_placeholder:
            POS tag feature placeholder of shape (None, n_tag_features),
            type tf.int32
        deprel_id_placeholder:
            Dependency relation feature placeholder of shape
            (None, n_deprel_features), type tf.int32
        class_placeholder:
            Labels placeholder tensor of shape (None, n_classes),
            type tf.float32
        dropout_placeholder: Dropout value placeholder (scalar), type
            tf.float32

        Add these placeholders to self as attributes
            self.word_id_placeholder
            self.tag_id_placeholder
            self.deprel_id_placeholder
            self.class_placeholder
            self.dropout_placeholder

        (Don't change the variable names)
        """
        ### BEGIN YOUR CODE
        self.word_id_placeholder = tf.placeholder(tf.int32, shape=(None, self.config.n_word_features))
        self.tag_id_placeholder = tf.placeholder(tf.int32, shape=(None, self.config.n_tag_features))
        self.deprel_id_placeholder = tf.placeholder(tf.int32, shape=(None, self.config.n_deprel_features))
        self.class_placeholder = tf.placeholder(tf.float32, shape=(None, self.config.n_classes))
        self.dropout_placeholder = tf.placeholder(tf.float32)
        ### END YOUR CODE

    def create_feed_dict(
            self, word_id_batch, tag_id_batch, deprel_id_batch,
            class_batch=None, dropout=1):
        """Creates the feed_dict for the dependency parser.

        A feed_dict takes the form of:

        feed_dict = {
                <placeholder>: <tensor of values to be passed for placeholder>,
                ....
        }

        The keys for the feed_dict should be a subset of the placeholder
        tensors created in add_placeholders. When an argument is None,
        don't add it to the feed_dict.

        Args:
            word_id_batch: A batch of word id features
            tag_id_batch: A batch of POS tag id features
            deprel_id_batch: A batch of dependency relation id features
            class_batch: A batch of class label data
            dropout: The dropout rate
        Returns:
            feed_dict: The feed dictionary mapping from placeholders to values.
        """
        ### BEGIN YOUR CODE
        feed_dict = {
            self.word_id_placeholder: word_id_batch,
            self.tag_id_placeholder: tag_id_batch,
            self.deprel_id_placeholder: deprel_id_batch,
            self.class_placeholder: class_batch,
            self.dropout_placeholder: dropout
        }
        feed_dict = dict((k, v) for k, v in feed_dict.items() if v is not None)
        ### END YOUR CODE
        return feed_dict



    def add_embeddings(self):
        """Creates embeddings that map word, tag, deprels to vectors

        Embedding layers convert (sparse) ID representations to dense,
        lower-dimensional representations. Inputs are integers, outputs
        are floats.

         - Create 3 embedding matrices, one for each of the input types.
           Input values index the rows of the matrices to extract. The
           max bound (exclusive) on the values in the input can be found
           in {n_word_ids, n_tag_ids, n_deprel_ids}
           After lookup, the resulting tensors should each be of shape
           (None, n, embed_size), where n is one of
           {n_word_features, n_tag_features, n_deprel_features}.
         - Initialize the word_id embedding matrix with
           self.word_embeddings. Initialize the other two matrices
           with the Xavier initialization you implemented
         - Reshape the embedding tensors into shapes
           (None, n * embed_size)

        ** Embedding matrices should be variables, not constants! **

        Use tf.nn.embedding_lookup. Also take a look at tf.reshape

        Returns:
            word_embeddings : tf.Tensor of type tf.float32 and shape
                (None, n_word_features * embed_size)
            tag_embeddings : tf.float32 (None, n_tag_features * embed_size)
            deprel_embeddings : tf.float32
                (None, n_deprel_features * embed_size)
        """
        ### BEGIN YOUR CODE
        xavier_initializer = xavier_weight_init()

        word_embedding_matrix = tf.get_variable("word_embedding_matrix", 
            [self.config.n_word_ids, self.config.embed_size], 
            initializer=tf.constant_initializer(self.word_embeddings))

        tag_embedding_matrix = tf.get_variable("tag_embedding_matrix", 
            [self.config.n_tag_ids, self.config.embed_size], 
            initializer=xavier_initializer)

        deprel_embedding_matrix = tf.get_variable("deprel_embedding_matrix", 
            [self.config.n_deprel_ids, self.config.embed_size], 
            initializer=xavier_initializer)


        word_lookup = tf.nn.embedding_lookup(word_embedding_matrix, self.word_id_placeholder)
        tag_lookup = tf.nn.embedding_lookup(tag_embedding_matrix, self.tag_id_placeholder)
        deprel_lookup = tf.nn.embedding_lookup(deprel_embedding_matrix, self.deprel_id_placeholder)

        word_embeddings = tf.reshape(word_lookup, [-1, self.config.n_word_features * self.config.embed_size])
        tag_embeddings = tf.reshape(tag_lookup, [-1, self.config.n_tag_features * self.config.embed_size])
        deprel_embeddings = tf.reshape(deprel_lookup, [-1, self.config.n_deprel_features * self.config.embed_size])
        ### END YOUR CODE
        return word_embeddings, tag_embeddings, deprel_embeddings

    def add_prediction_op(self):
        """Adds the single layer neural network

        The l
            h = Relu(x_w W_w + x+t W_t + x_d W_d + b1)
            h_drop = Dropout(h, dropout_rate)
            pred = h_drop U + b2

        Note that we are not applying a softmax to pred. The softmax
        will instead be done in the add_loss_op function, which improves
        efficiency because we can use
            tf.nn.softmax_cross_entropy_with_logits
        Excluding the softmax in predictions won't change the expected
        transition.

        Use the Xavier initializer from initialization.py for W_ and
        U. Initialize b1 and b2 with zeros.

        The dimensions of the various variables you will need to create
        are:
            W_w : (n_word_features * embed_size, hidden_size)
            W_t : (n_tag_features * embed_size, hidden_size)
            W_d : (n_deprel_features * embed_size, hidden_size)
            b1: (hidden_size,)
            U:  (hidden_size, n_classes)
            b2: (n_classes)

        Use the value self.dropout_placeholder in tf.nn.dropout directly

        Returns:
            pred: tf.Tensor of shape (batch_size, n_classes)
        """
        ### BEGIN YOUR CODE
        x_w, x_t, x_d = self.add_embeddings()
        
        # Code for Part3
        """
        if (self.config.addHidden2):
            w_w = tf.get_variable("W_w", [self.config.n_word_features * self.config.embed_size, self.config.hidden_size])
            w_t = tf.get_variable("W_t", [self.config.n_tag_features * self.config.embed_size, self.config.hidden_size])
            w_d = tf.get_variable("W_d", [self.config.n_deprel_features * self.config.embed_size, self.config.hidden_size])
            wh = tf.get_variable("wh", [self.config.hidden_size, self.config.hidden2_size])
            b1 = tf.get_variable("b1", [self.config.hidden_size])
            b2 = tf.get_variable("b2", [self.config.hidden2_size])
            b3 = tf.get_variable("b3", [self.config.n_classes])
            u1 = tf.get_variable("U1", [self.config.hidden_size, self.config.hidden2_size])
            u2 = tf.get_variable("U2", [self.config.hidden2_size, self.config.n_classes])

            h1 = tf.nn.relu(tf.matmul(x_w, w_w) + tf.matmul(x_t, w_t) + tf.matmul(x_d, w_d) + b1, name="relu1")

            h2 = tf.nn.relu(tf.matmul(h1, wh) + b2, name="relu2")

            h2_drop = tf.nn.dropout(h2, self.dropout_placeholder, name="dropout2")
            h2_output = tf.matmul(h2_drop, u2) + b3
            pred = h2_output
        else:
        """
        w_w = tf.get_variable("W_w", [self.config.n_word_features * self.config.embed_size, self.config.hidden_size])
        w_t = tf.get_variable("W_t", [self.config.n_tag_features * self.config.embed_size, self.config.hidden_size])
        w_d = tf.get_variable("W_d", [self.config.n_deprel_features * self.config.embed_size, self.config.hidden_size])
        b1 = tf.get_variable("b1", [self.config.hidden_size])
        b2 = tf.get_variable("b2", [self.config.n_classes])
        u = tf.get_variable("U", [self.config.hidden_size, self.config.n_classes])

        h = tf.nn.relu(tf.matmul(x_w, w_w) + tf.matmul(x_t, w_t) + tf.matmul(x_d, w_d) + b1, name="relu")
        h_drop = tf.nn.dropout(h, self.dropout_placeholder, name="dropout")
        pred = tf.matmul(h_drop, u) + b2
        ### END YOUR CODE
        return pred

    def add_loss_op(self, pred):
        """Adds Ops for the loss function to the computational graph.

        In this case we are using cross entropy loss. The loss should be
        averaged over all examples in the current minibatch.

        Use tf.nn.softmax_cross_entropy_with_logits to simplify your
        implementation. You might find tf.reduce_mean useful.

        Args:
            pred:
                A tensor of shape (batch_size, n_classes) containing
                the output of the neural network before the softmax layer.
        Returns:
            loss: A 0-d tensor (scalar)
        """
        ### BEGIN YOUR CODE
        cross_entropy_loss = tf.nn.softmax_cross_entropy_with_logits(labels=self.class_placeholder, logits=pred)
        loss = tf.reduce_mean(cross_entropy_loss)
        ### END YOUR CODE
        return loss

    def add_training_op(self, loss):
        """Sets up the training Ops.

        Creates an optimizer and applies the gradients to all trainable
        variables. The Op returned by this function is what must be
        passed to the `sess.run()` call to cause the model to train.

        Use tf.train.AdamOptimizer for this model.
        Calling optimizer.minimize() will return a train_op object.

        Args:
            loss: Loss tensor, from cross_entropy_loss.
        Returns:
            train_op: The Op for training.
        """
        ### BEGIN YOUR CODE
        train_op = tf.train.AdamOptimizer(self.config.lr).minimize(loss)
        ### END YOUR CODE
        return train_op

    def fit_batch(
            self,
            word_id_batch, tag_id_batch, deprel_id_batch, class_batch):
        feed = self.create_feed_dict(
            word_id_batch, tag_id_batch, deprel_id_batch,
            class_batch=class_batch, dropout=self.config.dropout
        )
        _, loss = self.sess.run([self.train_op, self.loss], feed_dict=feed)
        return loss

    def fit_epoch(self, train_data, batch_size=None, incl_progbar=True):
        '''Fit on training data for an epoch'''
        if incl_progbar:
            progbar = Progbar(target=len(train_data)*batch_size if batch_size else len(train_data))
        for (word_id_batch, tag_id_batch, deprel_id_batch), class_batch in \
                train_data:
            loss = self.fit_batch(
                word_id_batch, tag_id_batch, deprel_id_batch, class_batch)
            if incl_progbar:
                progbar.add(word_id_batch.shape[0], [("Cross-entropy", loss)])

    def predict_on_batch(self, inputs_batch):
        feed = self.create_feed_dict(*inputs_batch)
        predictions = self.sess.run(self.pred, feed_dict=feed)
        return predictions

    def predict(self, partial_parses):
        '''Use this model to predict the next transitions/deprels of pps'''
        feats = self.transducer.pps2feats(partial_parses)
        td_vecs = self.predict_on_batch(feats)
        preds = [
            self.transducer.td_vec2trans_deprel(td_vec) for td_vec in td_vecs]
        return preds

    def eval(self, sentences, ex_arcs):
        '''LAS on either training or test sets'''
        act_arcs = minibatch_parse(sentences, self, self.config.batch_size)
        ex_arcs = tuple([(a[0], a[1], self.transducer.id2deprel[a[2]]) for a in pp] for pp in ex_arcs)
        return score_arcs(act_arcs, ex_arcs)

    def __init__(self, transducer, sess, config, word_embeddings):
        self.transducer = transducer
        # we have to store the session here in order to avoid passing
        # the session to minibatch_parse. Don't try this at home!
        self.sess = sess
        self.word_embeddings = word_embeddings
        self.config = config
        self.build()

def main(debug):
    '''Main function

    Args:
    debug :
        whether to use a fraction of the data. Make sure to set to False
        when you're ready to train your model for real!
    '''
    print(80 * "=")
    print("INITIALIZING")
    print(80 * "=")
    config = Config()
    data = load_and_preprocess_data(
        max_batch_size=config.batch_size)
    transducer, word_embeddings, train_data = data[:3]
    dev_sents, dev_arcs = data[3:5]
    test_sents, test_arcs = data[5:]
    config.n_word_ids = len(transducer.id2word) + 1 # plus null
    config.n_tag_ids = len(transducer.id2tag) + 1
    config.n_deprel_ids = len(transducer.id2deprel) + 1
    config.embed_size = word_embeddings.shape[1]
    for (word_batch, tag_batch, deprel_batch), td_batch in \
            train_data.get_iterator(shuffled=False):
        config.n_word_features = word_batch.shape[-1]
        config.n_tag_features = tag_batch.shape[-1]
        config.n_deprel_features = deprel_batch.shape[-1]
        config.n_classes = td_batch.shape[-1]
        break
    print(
        'Word feat size: {}, tag feat size: {}, deprel feat size: {}, '
        'classes size: {}'.format(
            config.n_word_features, config.n_tag_features,
            config.n_deprel_features, config.n_classes))
    if debug:
        dev_sents = dev_sents[:500]
        dev_arcs = dev_arcs[:500]
        test_sents = test_sents[:500]
        test_arcs = test_arcs[:500]
    if not debug:
        weight_file = NamedTemporaryFile(suffix='.weights')
    with tf.Graph().as_default(), tf.Session() as session:
        print("Building model...", end=' ')
        start = time.time()
        model = ParserModel(transducer, session, config, word_embeddings)
        print("took {:.2f} seconds\n".format(time.time() - start))
        init = tf.global_variables_initializer()
        session.run(init)
        
        saver = None if debug else tf.train.Saver()
        print(80 * "=")
        print("TRAINING")
        print(80 * "=")
        best_las = 0.
        for epoch in range(config.n_epochs):
            print('Epoch {}'.format(epoch))
            if debug:
                model.fit_epoch(list(islice(train_data,3)), config.batch_size)
            else:
                model.fit_epoch(train_data)
            stdout.flush()
            dev_las, dev_uas = model.eval(dev_sents, dev_arcs)
            best = dev_las > best_las
            if best:
                best_las = dev_las
                if not debug:
                    saver.save(session, weight_file.name)
            print('Validation LAS: ', end='')
            print('{:.2f}{}'.format(dev_las, ' (BEST!), ' if best else ', '))
            print('Validation UAS: ', end='')
            print('{:.2f}'.format(dev_uas))
        if not debug:
            print()
            print(80 * "=")
            print("TESTING")
            print(80 * "=")
            print("Restoring the best model weights found on the dev set")
            saver.restore(session, weight_file.name)
            stdout.flush()
            las,uas = model.eval(test_sents, test_arcs)
            if las:
                print("Test LAS: ", end='')
                print('{:.2f}'.format(las), end=', ')
            print("Test UAS: ", end='')
            print('{:.2f}'.format(uas))
            print("Done!")
    return 0

if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == '-r':
        main(False)
    else:
        main(True)
