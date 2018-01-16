import tensorflow as tf
import numpy as np



def define_weights_gru(n_in,n_hidden):
    theta = {
        # Input weights for update gate
        'U_Z': tf.Variable(tf.random_normal(shape=[n_in, n_hidden],
                                            seed=2,
                                            mean=0.0,
                                            stddev=1 / np.sqrt(n_hidden))),

        # State weights for update gate
        'W_Z': tf.Variable(tf.random_normal(shape=[n_hidden, n_hidden],
                                            seed=2,
                                            mean=0.0,
                                            stddev=1 / np.sqrt(n_hidden))),

        # Input weights for reset gate
        'U_R': tf.Variable(tf.random_normal(shape=[n_in, n_hidden],
                                            seed=2,
                                            mean=0.0,
                                            stddev=1 / np.sqrt(n_hidden))),

        # State weights for reset gate
        'W_R': tf.Variable(tf.random_normal(shape=[n_hidden, n_hidden],
                                            seed=2,
                                            mean=0.0,
                                            stddev=1 / np.sqrt(n_hidden))),

        # Input weights for H
        'U_H': tf.Variable(tf.random_normal(shape=[n_in, n_hidden],
                                            seed=2,
                                            mean=0.0,
                                            stddev=1 / np.sqrt(n_hidden))),

        # State weights for H
        'W_H': tf.Variable(tf.random_normal(shape=[n_hidden, n_hidden],
                                            seed=2,
                                            mean=0.0,
                                            stddev=1 / np.sqrt(n_hidden))),

        # Update gate biases
        'B_Z': tf.Variable(tf.random_normal(shape=[n_hidden],
                                            seed=3,
                                            mean=0.0,
                                            stddev=0.5)),

        # Reset gate biases
        'B_R': tf.Variable(tf.random_normal(shape=[n_hidden],
                                            seed=3,
                                            mean=0.0,
                                            stddev=0.5)),

        # H biases
        'B_H': tf.Variable(tf.random_normal(shape=[n_hidden],
                                            seed=3,
                                            mean=0.0,
                                            stddev=0.5)),
    }
    return theta


def define_weights_rnn(n_in,n_hidden):
    theta = {
        # Input weights
        'w_input': tf.Variable(tf.random_normal(shape=[n_in, n_hidden],
                                              seed= 1,
                                                 mean  	= 0.0,
                                                 stddev = 1 / np.sqrt( n_in ) ) ),
        # Hidden layer weights
        'w_hidden': tf.Variable( tf.random_normal( shape  = [ n_hidden, n_hidden ],
                                                 seed  	= 1,
                                                 mean  	= 0.0,
                                                 stddev = 1.0 / np.sqrt( n_hidden ) ) ),
        # Hidden layer biases
        'b_hidden': tf.Variable(tf.random_normal(shape=[n_hidden],
                                                 seed=3,
                                                 mean=0.0,
                                                 stddev=0.005))
    }
    return theta

def define_weights_out(n_in,n_out):
    theta = {

        # Output layer weights
        'w_out': tf.Variable(tf.random_normal(shape=[n_in, n_out],
                                              seed=2,
                                              mean=0.0,
                                              stddev=0.001 / np.sqrt(n_in))),
        # Output layer biases
        'b_out': tf.Variable( tf.random_normal( shape  = [ n_out ],
                                              seed   = 4,
                                              mean   = 0.0,
                                              stddev = 0.005 ) )
    }
    return theta

def basic_rnn_cell(rnn_input, state, theta, nph, layer_prefix='' ):
    Wih = theta[ layer_prefix+"w_input" ]
    Whh = theta[ layer_prefix+"w_hidden" ]
    b = theta[ layer_prefix+"b_hidden" ]
    return tf.nn.dropout(
        tf.tanh( tf.matmul( state, Whh ) + tf.nn.dropout(tf.matmul(rnn_input, Wih),nph.p_keep['ih'])+ b ),
        nph.p_keep['hh'])


def gru_cell(rnn_input, state, theta, nph, layer_prefix='' ):
    l=layer_prefix
    rnn_input_d=tf.nn.dropout(rnn_input,nph.p_keep['ih'])
    z= tf.sigmoid(tf.matmul(rnn_input_d, theta[l+"U_Z"]) + tf.matmul(state, theta[l+"W_Z"]) + theta[l+"B_Z"])
    r = tf.sigmoid(tf.matmul(rnn_input_d, theta[l+"U_R"]) + tf.matmul(state, theta[l+"W_R"]) + theta[l+"B_R"])
    h = tf.tanh(tf.matmul(rnn_input_d, theta[l+"U_H"]) + tf.matmul(state * r, theta[l+"W_H"]) + theta[l+"B_H"])
    s = tf.multiply((1 - z), h) + tf.multiply(z, state)
    return tf.nn.dropout(s,nph.p_keep['hh'])

def prep_flatFromBatch(fun_name_list,x_in,y_in):
    fun_list=list(map(eval,fun_name_list))
    tt1, X_rec, Y_rec =runTestFun(x_in,y_in,  num_steps, fun_list, rand_ini=False)
    fun_flat={}
    tt2=list(map(list, zip(*tt1)))
    for ii,(this_fun,this_fun_name) in enumerate(zip(fun_list,fun_name_list)):
            fun_out=np.array(tt2[ii])
            fun_flat[this_fun_name]=fun_out.transpose(
                [2,0,1]+([3] if len(fun_out.shape)==4 else [])).reshape(-1,
                (fun_out.shape[-1] if len(fun_out.shape)==4 else 1))
    return fun_flat


def runTestFun(x_test, y_test, num_steps, inFun, rand_ini=False, rand_sigma=1, man_ini=[]):
    firstStepFlag = True
    outRec = []
    X_rec = []
    Y_rec = []
    inFun.append(final_state)
    for val_epoch in (gen_epochs(1, num_steps, x_test, y_test, for_train=False)):
        for val_step, (X_val, Y_val) in enumerate(val_epoch):
            feed_dict_prep = {x: X_val, y: Y_val}

            if firstStepFlag:
                if rand_ini:
                    ini_dict = {init_state: rand_sigma * np.random.normal(size=[BATCH_SIZE, N_HIDDEN])}
                elif len(man_ini):
                    ini_dict = {init_state: man_ini}
                else:
                    ini_dict = {}
            else:
                ini_dict = {init_state: outLists[-1]}
            feed_dict_prep.update(ini_dict)
            outLists = sess.run(inFun,
                                feed_dict=feed_dict_prep)
            # epo_val_loss+=val_loss_
            outRec.append(outLists)
            X_rec.append(X_val)
            Y_rec.append(Y_val)
            firstStepFlag = False
    return outRec, X_rec, Y_rec

def run_epoch(sess, nph, hp, epoch_data , in_fun_with_fs, rand_ini=False, rand_sigma=1, man_ini=[],for_train=False):
    firstStepFlag = True
    X_rec = []
    Y_rec = []
    dropout_feed={}
    if for_train:
        for key in hp.dropout.keys():
            dropout_feed.update({nph.p_keep[key]:hp.dropout[key]})
    outRec = dict(zip(in_fun_with_fs, [[] for qq in range(len(in_fun_with_fs))]))
    for val_step, (X_in, Y_in) in enumerate(epoch_data):
        feed_dict_prep = {nph.x: X_in, nph.y: Y_in}
        feed_dict_prep.update(dropout_feed)

        if firstStepFlag:
            if rand_ini:
                ini_dict = {nph.init_state: rand_sigma * np.random.normal(size=[hp.BATCH_SIZE, hp.N_HIDDEN])}
            elif len(man_ini):
                ini_dict = {nph.init_state: man_ini}
            else:
                ini_dict = {}
        else:
            ini_dict = {nph.init_state: outLists[-1]}
        feed_dict_prep.update(ini_dict)

        outLists = sess.run(in_fun_with_fs,feed_dict=feed_dict_prep)

        for ii, (this_fun,this_fun_out) in enumerate(zip(in_fun_with_fs,outLists)):
            outRec[this_fun].append(this_fun_out)
        X_rec.append(X_in)
        Y_rec.append(Y_in)

        firstStepFlag = False
    return outRec, X_rec, Y_rec

class NetPlaceholders:
    def __init__(self,hp):
        self.x = tf.placeholder("float", [hp.BATCH_SIZE, hp.num_steps, hp.VOCAB_SIZE])
        self.y = tf.placeholder(tf.int32, [hp.BATCH_SIZE, hp.num_steps])
        self.p_keep = {}
        self.p_keep['hh'] = tf.placeholder_with_default(1.0,())
        self.p_keep['ih'] = tf.placeholder_with_default(1.0,())
        # self.p_keep['ih'] = tf.ones((),dtype="float")
        self.init_state = tf.zeros([hp.BATCH_SIZE, hp.N_HIDDEN])

class NetFunctions:
    def __init__(self):
        pass

    def create_recurrent_net_for_lm(self,hp,theta,nph):
        state = nph.init_state
        self.rnn_inputs = tf.unstack(nph.x, axis=1)
        self.rnn_outputs = []
        for rnn_input in self.rnn_inputs:
            state = hp.network_cell(rnn_input, state, theta, nph)
            self.rnn_outputs.append(state)
        self.final_state = self.rnn_outputs[-1]
        self.logits = [tf.matmul(rnn_output, theta["w_out"]) + theta["b_out"] for rnn_output in self.rnn_outputs]
        self.predictions = tf.identity(  # warped into identity function to make the list hashable
            [tf.nn.softmax(logit) for logit in self.logits])
        # Turn our y placeholder into a list of labels
        self.y_as_list = [tf.squeeze(i, squeeze_dims=[1]) for i in
                     tf.split(axis=1, num_or_size_splits=hp.num_steps, value=nph.y)]
        self.rnn_outputs_h = tf.identity(self.rnn_outputs)
        # losses and optimizer
        self.losses = [tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logit, labels=label) for logit, label in
                  zip(self.logits, self.y_as_list)]
        self.cost = tf.reduce_mean(self.losses)
        self.optimizer = tf.train.RMSPropOptimizer(hp.LEARNING_RATE).minimize(self.cost)
        self.init = tf.global_variables_initializer()


