''' added a positive symmetric part to whh,
 to encourage possibility of memory formation
 search for 'modification' to find the update vs. the vanilla version'''
import numpy as np
import tensorflow as tf
import time
import re
import random
import pickle
import sys
import os
from wij import *
from minRnn_utils import *
from hyper_params import hyper_params

hp=hyper_params()
hp.parse_from_command_line(sys.argv)

dataAll = open(hp.CORPUS_PATH+hp.CORPUS_NAME, 'r').read()  # should be simple plain text file
chars = sorted(list(set(dataAll)))
data_size, vocab_size = len(dataAll), len(chars)
hp.VOCAB_SIZE = vocab_size
hp.N_CLASSES = hp.VOCAB_SIZE
hp.P_HOPF = 2
hp.MAG_HOPF=1
hp.cont_prev_run = None
hp.cont_iter = None
hp.theta_to_exclude=['hopf_v', 'hopf_mu']
hp.parse_from_command_line(sys.argv)

char_to_ix = {ch: i for i, ch in enumerate(chars)}
ix_to_char = {i: ch for i, ch in enumerate(chars)}
hp.char_to_ix = char_to_ix
hp.ix_to_char = ix_to_char

hp.THIS_RUN_NAME_WITH_PATH=hp.SAVE_PATH+hp.THIS_RUN_NAME+hp.THIS_RUN_SUFFIX

sys.stdout = Logger(hp.THIS_RUN_NAME_WITH_PATH+'.log')
with open(hp.THIS_RUN_NAME_WITH_PATH+'.hp.pkl','wb') as f:
    pickle.dump(hp,f)
hp.network_cell = gru_cell if hp.using_gru else basic_rnn_cell

print('Command line:', sys.argv)
print('\n\n data has %d characters, %d unique.' % (data_size, vocab_size ))
print('\n\n ---------Hyper parameters------------')
hp.print_params()
print()

# Arrange data into training and test sets
data_train = dataAll[round(hp.TRAIN_LIM[0] * data_size):round(hp.TRAIN_LIM[1] * data_size)]
data_val = dataAll[round(hp.VALIDATION_LIM[0] * data_size):round(hp.VALIDATION_LIM[1] * data_size)]

config = tf.ConfigProto()
# config = session_conf = tf.ConfigProto(
#       device_count={'CPU': 1, 'GPU': 0},
#       allow_soft_placement=True,
#       log_device_placement=False
#       )
config.gpu_options.allocator_type = 'BFC'


# Initialize input placeholders
# Define weights
if hp.using_gru:
    theta = define_weights_gru(hp.VOCAB_SIZE,hp.N_HIDDEN)
else:
    theta = define_weights_rnn(hp.VOCAB_SIZE,hp.N_HIDDEN)
theta.update(define_weights_out(hp.N_HIDDEN,hp.VOCAB_SIZE))

'''' modification'''''
theta.update(define_weights_hopf(hp.N_HIDDEN, hp.P_HOPF))

theta_prime=theta.copy()
theta_prime['w_hidden']=theta['w_hidden']+\
                        tf.matmul(theta['hopf_v'],((hp.MAG_HOPF)*tf.transpose(theta['hopf_v'])))

# Define the network
nph=NetPlaceholders(hp)
nfn=NetFunctions()
nfn.create_recurrent_net_for_lm(hp,theta_prime,nph,theta_raw=theta)

state = nph.init_state
saver = tf.train.Saver(max_to_keep=900)
sess = tf.Session(config=config)
sess.run(nfn.init)

if hp.cont_prev_run is not None:
    if hp.cont_iter is not None:
        load_theta(theta=theta, sess=sess, filename=hp.cont_prev_run+'_theta_'+str(hp.cont_iter)+'.pkl')
    else:
        saver.restore(sess,hp.cont_prev_run+'.final.ckpt')


training_losses = []
nBits = hp.VOCAB_SIZE
x_train, y_train = LM_xy_prep2(data_train, char_to_ix, nBits)
x_val, y_val = LM_xy_prep2(data_val, char_to_ix, nBits)

loss_rec = []
epo_val_loss_best = 9999.0

train_fun_list=[ nfn.cost, nfn.optimizer, nfn.predictions, nfn.final_state]
validation_fun_list=[nfn.cost, nfn.rnn_outputs_h, nfn.final_state]

#Main training list
for idx, epoch_data in enumerate(gen_epochs( x_train, y_train,hp)):
    tic = time.clock()
    if hp.verbose:
        print("\nEPOCH", idx)
    out_fun_training, X_rec, Y_rec = run_epoch(sess,nph,hp, epoch_data,  train_fun_list, for_train=True)
    mean_acc = np.mean(np.transpose(Y_rec,[0,2,1])==np.argmax(out_fun_training[nfn.predictions],axis=3))
    epo_training_loss = np.mean(out_fun_training[nfn.cost])
    toc = time.clock()
    if hp.verbose:
        validation_batch = gen_batch((x_val, y_val), hp)
        out_fun_validation, X_rec, Y_rec = run_epoch(sess,nph,hp, validation_batch , validation_fun_list)
        epo_validation_loss = np.mean(out_fun_validation[nfn.cost])
        print("Training loss                     : ", epo_training_loss,
                "\tMean accuracy: ", mean_acc,""
                +"\nValidation loss                   : ", epo_validation_loss, ""
              )
        print("hopfield mu^2",(hp.MAG_HOPF))
        print("hopfield var", np.var(theta['hopf_v'].eval(sess),axis=0))
        print("training time consumed:", toc - tic, " seconds\n")
        save_theta(theta=theta, sess=sess, filename=hp.THIS_RUN_NAME_WITH_PATH+'_theta_'+str(idx)+'.pkl')
    loss_rec.append([idx, epo_training_loss, epo_validation_loss])
    if epo_validation_loss<epo_val_loss_best:
        save_path = saver.save(sess, hp.THIS_RUN_NAME_WITH_PATH + ".best.ckpt")
        epo_val_loss_best = epo_validation_loss

print("Optimization Finished!")

save_path = saver.save(sess, hp.THIS_RUN_NAME_WITH_PATH + ".final.ckpt")
sess.close()
print('Results are at:', hp.THIS_RUN_NAME_WITH_PATH)