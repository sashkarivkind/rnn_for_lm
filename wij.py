import tensorflow as tf
import numpy as np
import copy



def define_weights_gru(n_in,n_hidden):
    theta = {
        # Input weights for update gate
        'U_Z': tf.Variable(tf.random_normal(shape=[n_in, n_hidden],
                                            # seed=2,
                                            mean=0.0,
                                            stddev=1 / np.sqrt(n_hidden))),

        # State weights for update gate
        'W_Z': tf.Variable(tf.random_normal(shape=[n_hidden, n_hidden],
                                            # seed=2,
                                            mean=0.0,
                                            stddev=1 / np.sqrt(n_hidden))),

        # Input weights for reset gate
        'U_R': tf.Variable(tf.random_normal(shape=[n_in, n_hidden],
                                            #seed=2,
                                            mean=0.0,
                                            stddev=1 / np.sqrt(n_hidden))),

        # State weights for reset gate
        'W_R': tf.Variable(tf.random_normal(shape=[n_hidden, n_hidden],
                                            #seed=2,
                                            mean=0.0,
                                            stddev=1 / np.sqrt(n_hidden))),

        # Input weights for H
        'U_H': tf.Variable(tf.random_normal(shape=[n_in, n_hidden],
                                            #seed=2,
                                            mean=0.0,
                                            stddev=1 / np.sqrt(n_hidden))),

        # State weights for H
        'W_H': tf.Variable(tf.random_normal(shape=[n_hidden, n_hidden],
                                            #seed=2,
                                            mean=0.0,
                                            stddev=1 / np.sqrt(n_hidden))),

        # Update gate biases
        'B_Z': tf.Variable(tf.random_normal(shape=[n_hidden],
                                            #seed=3,
                                            mean=0.0,
                                            stddev=0.5)),

        # Reset gate biases
        'B_R': tf.Variable(tf.random_normal(shape=[n_hidden],
                                            #seed=3,
                                            mean=0.0,
                                            stddev=0.5)),

        # H biases
        'B_H': tf.Variable(tf.random_normal(shape=[n_hidden],
                                            #seed=3,
                                            mean=0.0,
                                            stddev=0.5)),
    }
    return theta


def define_weights_rnn(n_in,n_hidden):
    theta = {
        # Input weights
        'w_input': tf.Variable(tf.random_normal(shape=[n_in, n_hidden],
                                              #seed= 1,
                                                 mean  	= 0.0,
                                                 stddev = 1 / np.sqrt( n_in ) ) ),
        # Hidden layer weights
        'w_hidden': tf.Variable( tf.random_normal( shape  = [ n_hidden, n_hidden ],
                                                 #seed  	= 1,
                                                 mean  	= 0.0,
                                                 stddev = 1.0 / np.sqrt( n_hidden ) ) ),
        # Hidden layer biases
        'b_hidden': tf.Variable(tf.random_normal(shape=[n_hidden],
                                                 #seed=3,
                                                 mean=0.0,
                                                 stddev=0.005))
    }
    return theta

def define_weights_out(n_in,n_out):
    theta = {

        # Output layer weights
        'w_out': tf.Variable(tf.random_normal(shape=[n_in, n_out],
                                              #seed=2,
                                              mean=0.0,
                                              stddev=0.001 / np.sqrt(n_in))),
        # Output layer biases
        'b_out': tf.Variable( tf.random_normal( shape  = [ n_out ],
                                              #seed   = 4,
                                              mean   = 0.0,
                                              stddev = 0.005 ) )
    }
    return theta

def define_weights_hopf(n_hidden, p_hopf):
    theta = {

        # hopfield part vectors
        'hopf_v': tf.Variable(tf.random_normal(shape=[n_hidden, p_hopf],
                                              #seed=2,
                                              mean=0.0,
                                              stddev=1 / np.sqrt(n_hidden))),

        # hopfield vector names
        'hopf_mu': tf.Variable(tf.ones(shape=[p_hopf,1]))
    }
    return theta

def basic_rnn_cell(rnn_input, state, theta, nph, layer_prefix='' ):
    Wih = theta[ layer_prefix+"w_input" ]
    Whh = theta[ layer_prefix+"w_hidden" ]
    b = theta[ layer_prefix+"b_hidden" ]
    return tf.nn.dropout(
        tf.tanh( tf.matmul( state, Whh )
                 + tf.nn.dropout(tf.matmul(rnn_input, Wih),nph.p_keep['ih'])+ b ),nph.p_keep['hh'])


def gru_cell(rnn_input, state, theta, nph, layer_prefix='' ):
    l=layer_prefix
    rnn_input_d=tf.nn.dropout(rnn_input,nph.p_keep['ih'])
    z= tf.sigmoid(tf.matmul(rnn_input_d, theta[l+"U_Z"]) + tf.matmul(state, theta[l+"W_Z"]) + theta[l+"B_Z"])
    r = tf.sigmoid(tf.matmul(rnn_input_d, theta[l+"U_R"]) + tf.matmul(state, theta[l+"W_R"]) + theta[l+"B_R"])
    h = tf.tanh(tf.matmul(rnn_input_d, theta[l+"U_H"]) + tf.matmul(state * r, theta[l+"W_H"]) + theta[l+"B_H"])
    s = tf.multiply((1 - z), h) + tf.multiply(z, state)
    return tf.nn.dropout(s,nph.p_keep['hh'])

def debug_plus1_cell(rnn_input, state, theta, nph, layer_prefix='' ):
    return state+1

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
                ini_dict = {nph.init_state: rand_sigma * np.random.normal(size=[hp.BATCH_SIZE, hp.N_HIDDEN])} #TODO
            elif len(man_ini):
                ini_dict = {nph.init_state: man_ini} #TODO
            else:
                ini_dict = {}
        else:
            ini_dict = {a:b for a,b in zip(nph.init_state[1:],out_list[-(hp.hidden_layers):])} #todo clean the input layer interpretation as part of state
        #print('debug ini_dict',ini_dict)
        feed_dict_prep.update(ini_dict)
        #print('debug feed_dict',feed_dict_prep)

        out_list = sess.run(in_fun_with_fs,feed_dict=feed_dict_prep)
        # print('debug, session output len:', len(out_list))
        for ii, (this_fun,this_fun_out) in enumerate(zip(in_fun_with_fs,out_list)):
            outRec[this_fun].append(this_fun_out)
        X_rec.append(X_in)
        Y_rec.append(Y_in)

        firstStepFlag = False
    return outRec, X_rec, Y_rec


def prep_list_to_opt(theta,hp):
    param_to_optimize=[]
    if 'theta_to_exclude' not in hp.__dict__.keys():
        hp.theta_to_exclude=[]
    for this_key in theta.keys():
        if this_key not in hp.theta_to_exclude:
            param_to_optimize.append(theta[this_key])
    return param_to_optimize

class NetPlaceholders:
    def __init__(self,hp):
        self.x = tf.placeholder("float", [hp.BATCH_SIZE, hp.num_steps, hp.VOCAB_SIZE])
        self.y = tf.placeholder(tf.int32, [hp.BATCH_SIZE, hp.num_steps])
        self.p_keep = {}
        self.p_keep['hh'] = tf.placeholder_with_default(1.0,())
        self.p_keep['ih'] = tf.placeholder_with_default(1.0,())
        # self.p_keep['ih'] = tf.ones((),dtype="float")
        self.init_state=[0]
        for layer in range(1,hp.hidden_layers+1):
            self.init_state.append( define_state_placeholder_for_layer(
                hp.BATCH_SIZE, hp.layer_size[layer],hp.layer_type[layer]))

class NetFunctions:
    def __init__(self):
        pass

    def create_recurrent_net_for_lm(self,hp,theta,nph,theta_raw=None):
        if theta_raw is None:
            theta_raw=theta
        state = copy.copy(nph.init_state)
        self.rnn_inputs = tf.unstack(nph.x, axis=1)
        self.rnn_outputs = []
        for rnn_input in self.rnn_inputs:
            state[0] = copy.copy(rnn_input)
            for layer in range(1, hp.hidden_layers + 1):    #layer zero rezerved for input
                state[layer] = hp.network_cell[layer](state[layer-1], state[layer], theta['hidden'][layer], nph)
            # print('debug state',state)
            cpstate=copy.copy(state)
            self.rnn_outputs.append(cpstate)
            # print('debug rnn_outputs',self.rnn_outputs)

        self.final_state = cpstate # todo self.rnn_outputs[-1]
        # todo self.final_state_flat = deep_dict_into_list(self.rnn_outputs[-1])
        self.logits = [tf.matmul
                       (rnn_output[-1],theta["w_out"])
                       + theta["b_out"] for rnn_output in self.rnn_outputs]
        self.predictions = tf.identity(  # warped into identity function to make the list hashable
            [tf.nn.softmax(logit) for logit in self.logits])
        # Turn our y placeholder into a list of labels
        self.y_as_list = [tf.squeeze(i, squeeze_dims=[1]) for i in
                     tf.split(axis=1, num_or_size_splits=hp.num_steps, value=nph.y)]
        #todo self.rnn_outputs_h = tf.identity(self.rnn_outputs)
        # losses and optimizer
        self.losses = [tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logit, labels=label) for logit, label in
                  zip(self.logits, self.y_as_list)]
        self.cost = tf.reduce_mean(self.losses)
        self.optimizer =\
            tf.train.RMSPropOptimizer(hp.LEARNING_RATE).minimize(self.cost)
        # TODO tf.train.RMSPropOptimizer(hp.LEARNING_RATE).minimize(self.cost, var_list=prep_list_to_opt(theta_raw, hp))
        self.init = tf.global_variables_initializer()


def clip_rho(W,rho_max):
    [ee,vv]=np.linalg.eig(W)
    ee_clip=ee/np.abs(ee)*np.minimum(np.abs(ee),rho_max)
    return np.matmul(vv,
                     np.matmul(np.diag(ee_clip),np.linalg.inv(vv))).real

def define_weights_for_layer(input_size, hidden_size, layer_type):
    if layer_type=='gru':
        return define_weights_gru(input_size,hidden_size)
    elif layer_type=='rnn':
        return define_weights_rnn(input_size,hidden_size)
    elif layer_type == 'debug_plus1':
        return define_weights_rnn(input_size,hidden_size) #return same parameters as basic cell. no use of parameters inside the cell
    else:
        raise Exception('unknown cell type!')

def define_state_placeholder_rnn(batch_size, hidden_size,layer_name=''):
    return tf.ones([batch_size, hidden_size],name='state_placeholder'+layer_name)

def define_state_placeholder_gru(batch_size,hidden_size):
    return tf.zeros([batch_size, hidden_size])

def define_state_placeholder_for_layer(batch_size, hidden_size, layer_type):
    if layer_type=='gru':
        return define_state_placeholder_gru(batch_size, hidden_size)
    elif layer_type=='rnn':
        return define_state_placeholder_rnn(batch_size, hidden_size)
    elif layer_type == 'debug_plus1':
        return define_state_placeholder_rnn(batch_size, hidden_size) #needs same placeholder for the internal state as a basic cell.
    else:
        raise Exception('unknown cell type!')

def define_weights(hp):
    theta={}
    theta['hidden']={}
    for layer in range(1,hp.hidden_layers+1):
        print('debug layer#:',layer)
        print('debug layer details:',hp.layer_size[layer-1],hp.layer_size[layer],hp.layer_type[layer])
        theta['hidden'][layer] = define_weights_for_layer(input_size=hp.layer_size[layer-1],hidden_size=hp.layer_size[layer],layer_type=hp.layer_type[layer])
    theta.update(define_weights_out(hp.layer_size[-1],hp.VOCAB_SIZE))
    return theta

def deep_dict_into_list(din):
    lout=[]
    for k in sorted(din.keys()):
        if isinstance(din[k],dict):
            lout+=deep_dict_into_list(din[k])
        else:
            lout.append(din[k])
    return lout

def prep_network_cells(hp):
    cells=[0]
    for layer in range(1, hp.hidden_layers + 1):
        if hp.layer_type[layer] == 'gru':
            cells.append(gru_cell)
        elif hp.layer_type[layer] == 'rnn':
            cells.append(basic_rnn_cell)
        elif hp.layer_type[layer] == 'debug_plus1':
            cells.append(debug_plus1_cell)
        else:
            raise Exception('unknown cell type!')
    return cells

