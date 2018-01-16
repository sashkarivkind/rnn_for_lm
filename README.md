# rnn_for_lm
RNN for Language modeling. Based on Tensorflow. Some functions to explore the RNN from the dynamic systems point of view are available.
### Usage
python3 rnn_for_lm_vanilla.py <hparam1=value1 ... >

Hyperparameters are pretty much self explaining. Here are the defaults:

SAVE_PATH="../savedStates/"  
CORPUS_PATH = "../corpus//"  
CORPUS_NAME = "warpeace_input_Karpathy.txt"  # [I took it from here](https://cs.stanford.edu/people/karpathy/char-rnn/)  
THIS_RUN_NAME = sys.argv[0]+'_noname_'+str(int(time.time()))  
THIS_RUN_SUFFIX=''  
TRAIN_LIM = [0, 0.8] #starting and ending point in corpus. default - first 80% of the corpus  
VALIDATION_LIM = [0.8, 0.9]  
N_HIDDEN = 512  
LEARNING_RATE = 2e-3  
X_sc = 1 #not in use for now  
BATCH_SIZE = 100  
EPSILON = 1e-7  
num_steps = 100 # number of rnn steps  
using_gru = 0 # set to 1 to use a gru cell, zero for plain RNN  
num_epochs = 100  
verbose = 1  
dropout = {'hh':0.95 , 'ih':1.0}  
    
### Further documentation
Will be added soon.
### Reference
The starting point for this code was a [flat RNN by Andrej Karpathy](https://gist.github.com/karpathy/d4dee566867f8291f086.js).
It aslo uses some building blocks from an unpublished code by Maayan Shviro and Itay Zalic.
### License
MIT
