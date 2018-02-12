import numpy as np
import tensorflow as tf
import time
import re
import random
import sys
import scipy.cluster.hierarchy as schi
import pickle
from termcolor import colored
import matplotlib.pyplot as plt

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

def space_dist_prep(x_string):
    space_dist={}
    this_key=0
    for ii, (this_char) in enumerate(list(x_string)):
        if this_char not in ' \n':
            this_key+=1
        else:
            this_key=0
        if this_key in space_dist.keys():
            space_dist[this_key].append(ii)
        else:
            space_dist[this_key]=[ii]
        return space_dist

def unOneHot(x):
    return np.matmul(x,np.array(range(np.shape(x)[1])))


def plot_dim_chart(data_in, metric='euclidean', chunk_len=0, verbose=False, normalize=True):
    d_vec2, nn = prep_dim_chart(data_in, metric=metric, chunk_len=chunk_len, normalize=normalize)
    plot_handler = plt.plot(np.log10(d_vec2), np.log10(nn))
    plt.grid()
    return plot_handler


def prep_dim_chart(data_in, metric='euclidean', chunk_len=0, verbose=False, normalize=True):
    if not (chunk_len):
        chunk_len = np.shape(data_in)[0]
    nn = np.linspace(1, chunk_len * (chunk_len - 1) // 2,
                     chunk_len * (chunk_len - 1) // 2)
    d_vec2 = nn  # initialization
    n_chunks = np.shape(data_in)[0] // chunk_len
    if verbose:
        print('total chunks =', n_chunks)
    for ii in range(np.shape(data_in)[0] // chunk_len):
        d_vec = scpd.pdist(data_in[ii * chunk_len:(ii + 1) * chunk_len], metric=metric)
        d_vec2 = d_vec2 * (1 - 1.0 / (ii + 1)) + (1.0 / (ii + 1)) * np.sort(d_vec)
        if verbose:
            print('.'),
    if normalize:
        nn = np.float16(nn) / np.float(chunk_len)
    return d_vec2, nn

def shuffle_string(s):
    s_list=list(s)
    random.shuffle(s_list)
    return ''.join(s_list)

def shuffle_data(data, shuffle_words=False, shuffle_sentences=False):
    if shuffle_sentences:
        sentence_list = re.split(r'( *[\.\?!][\'"\)\]]* *)', data)
        sentence_list_stop = []
        for ii in range(len(sentence_list) // 2):
            sentence_list_stop.append(sentence_list[2 * ii] + sentence_list[2 * ii + 1])
        # print(''.join(sentence_list_stop))
        sentence_list_stop_shuffle = sentence_list_stop
        random.shuffle(sentence_list_stop_shuffle)
        return ''.join(sentence_list_stop_shuffle)
    if shuffle_words:
        word_list = re.split(r'( *[\.\s\?!][\'"\)\]]* *)', data)
        word_list_stop = []
        for ii in range(len(word_list) // 2):
            word_list_stop.append(word_list[2 * ii] + word_list[2 * ii + 1])
        word_list_shuffle = word_list_stop
        random.shuffle(word_list_shuffle)
        return ''.join(word_list_shuffle)
    return data


def LM_xy_prep(data, char_to_ix, nBits):
    x = myOneHot([char_to_ix[ch] for ch in data[:-2]], nBits)
    y = [char_to_ix[ch] for ch in data[1:]]
    return x, y

def gen_batch( raw_data, hp ):
    raw_x, raw_y = raw_data
    data_length = len( raw_x )

    unrolled_x = raw_x

    batch_partition_length = data_length // hp.BATCH_SIZE  # divide all the data into number of batches
    data_x = []
    data_y = np.zeros( [ hp.BATCH_SIZE, batch_partition_length ], dtype=np.int32 )
    for i in range( hp.BATCH_SIZE ):
        x_slice = np.array ( unrolled_x[ batch_partition_length * i : batch_partition_length * ( i + 1 ) ] )
        data_x.append( x_slice )
        data_y[ i ] = raw_y[ batch_partition_length * i : batch_partition_length * ( i + 1 ) ]
    epoch_size = batch_partition_length // hp.num_steps
    for i in range( epoch_size ):
        x = np.array(
            [myOneHot(data_x[ this_slice ][ i * hp.num_steps: (i + 1) * hp.num_steps],hp.VOCAB_SIZE) for this_slice in range(hp.BATCH_SIZE)]
        )
        y = data_y[ : , i * hp.num_steps : ( i + 1 ) * hp.num_steps ]
        yield ( x, y )

def gen_epochs( x, y, hp ):
    for i in range( hp.num_epochs ):
        yield gen_batch( ( x, y ),hp )

def myOneHot(x, nBits):
    xlen = np.shape(x)[0]
    xOneHot = np.zeros([xlen, nBits])
    for i in range(xlen):
        xOneHot[i, x[i]] = 1
    return xOneHot


def LM_xy_prep2(data, char_to_ix, nBits):
    x = [char_to_ix[ch] for ch in data[:-2]]
    y = [char_to_ix[ch] for ch in data[1:]]
    return x, y


def seek_long_bifurcations_original_version():
    l1 = 50
    l2 = num_steps - l1
    n_strings = 10000
    half_batch_size = BATCH_SIZE // 2
    dd_rec = np.zeros([n_strings, num_steps])
    data_prep = np.zeros([BATCH_SIZE, num_steps, VOCAB_SIZE])
    even_indexes = [2 * uu for uu in range(half_batch_size)]
    odd_indexes = [2 * uu + 1 for uu in range(half_batch_size)]
    d12 = zip(np.random.randint(0, len(data_val) - l1 - l2, [n_strings]),
              np.random.randint(0, len(data_val) - l1 - l2, [n_strings]))
    for i1, (d1, d2) in enumerate(d12):
        str1 = data_val[d1:d1 + l1]
        str2 = data_val[d2:d2 + l1]
        str1a = data_val[d1 + l1:d1 + l1 + l2]
        data_in_list = [str1 + str1a,
                        str2 + str1a]
        state_rec_list = []
        logit_rec_list = []
        for i2, data_in in enumerate(data_in_list):
            data_prep[((i1 * 2) % BATCH_SIZE) + i2, :, :] = myOneHot([char_to_ix[qq] for qq in data_in], VOCAB_SIZE)
        if (i1 + 1) % (half_batch_size) == 0:
            sates_from_batch = np.array(sess.run([rnn_outputs], feed_dict={x: data_prep, p_keephh: 1.0})[0]).transpose(
                [1, 0, 2])
            dd_h = np.sum((sates_from_batch[even_indexes, :, :] - sates_from_batch[odd_indexes, :, :]) ** 2,
                          axis=2) ** 0.5
            dd_rec[i1 + 1 - half_batch_size:(i1 + 1), :] = dd_h

    print('log distance >0.3:', end='')
    for offset in range(0, 100, 10):
        n_high = (np.mean(np.log10(dd_rec)[:, -1 - offset] > 0.3))
        print(n_high, end=', ')
        print('\n')

class Logger:
    def __init__(self,log_name):
        self.terminal = sys.stdout
        self.log = open(log_name, "w")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        #this flush method is needed for python 3 compatibility.
        #this handles the flush command by doing nothing.
        #you might want to specify some extra behavior here.
        pass

def run_states_fixed_string(sess, nph, hp,ic_states,fixed_string,final_state_fun):
    bb_max = ic_states.shape[0] // hp.BATCH_SIZE
    final_states = np.zeros([ic_states.shape[0], hp.N_HIDDEN])
    fixed_string = fixed_string[:hp.num_steps]
    x_prep=np.zeros([hp.BATCH_SIZE,hp.num_steps,hp.VOCAB_SIZE])
    x_prep[:,:] = myOneHot([hp.char_to_ix[c] for c in fixed_string],hp.VOCAB_SIZE)
    for bb in range(bb_max):
        ic = ic_states[bb * hp.BATCH_SIZE:(bb + 1) * hp.BATCH_SIZE, :]
        final_states[bb * hp.BATCH_SIZE:(bb + 1) * hp.BATCH_SIZE, :], = sess.run([final_state_fun],
                                                                           feed_dict={nph.x: x_prep,
                                                                                      nph.init_state: ic})
    return final_states

def cluster_states_fixed_string(sess, nph, hp,ic_states,fixed_string,final_state_fun,d_tol = 0.001):
    final_states=run_states_fixed_string(sess, nph, hp, ic_states, fixed_string, final_state_fun)
    tt1 = schi.linkage(final_states)
    return sum(tt1[:, 2] > d_tol) + 1


def cluster_states_fixed_string_gap(sess, nph, hp, ic_states, fixed_string,
                                    final_state_fun, d_tol=0.001, final_states=None,
                                    return_output_linkage=False,
                                    return_final_states=False,
                                    return_flat_clustering=False):
    if final_states is None:
        final_states = run_states_fixed_string(sess, nph, hp, ic_states, fixed_string, final_state_fun)

    tt1 = schi.linkage(final_states)
    links = tt1[:, 2]
    log_links = np.log10(np.sort(links))
    link_diff_log = np.diff(log_links)
    diff_indexes = np.argsort(link_diff_log)
    found_gap = False
    for ii in reversed(diff_indexes):
        if log_links[ii + 1] > np.log10(d_tol):
            found_gap = True
            break
    if found_gap:
        gap_value = link_diff_log[ii]
        gap_dist = 10**log_links[ii + 1]
        num_clusters = len(link_diff_log) - ii + 1
    else:
        gap_value = 0
        gap_dist = 0
        num_clusters = 1

    out_dict = {'gap': [gap_value,gap_dist], 'clusters': num_clusters}

    if return_output_linkage:
        out_dict.update({'linkage': tt1})
    if return_final_states:
        out_dict.update({'final_states': final_states})
    if return_flat_clustering:
        out_dict.update({'flat_clustering': schi.fcluster(tt1, num_clusters, criterion='maxclust')})

    return out_dict


def save_theta(theta=None,sess=None,filename=None):
    theta_numeric = {this_key:this_item.eval(sess) for this_key,this_item in zip(theta.keys(),theta.values())}
    with open(filename,'wb') as f:
        pickle.dump(theta_numeric,f)

def load_theta(theta=None,sess=None,filename=None):
    with open(filename,'rb') as f:
        theta_numeric=pickle.load(f)
    for this_key, this_item in zip(theta_numeric.keys(), theta_numeric.values()):
        theta[this_key].assign(this_item).op.run(session=sess)

def batch_to_flat(batch_result):
    return batch_result.transpose(
            [2,0,1]+
        ([3]
         if len(batch_result.shape)==4
         else [])).reshape(-1,
            (batch_result.shape[-1]
             if len(batch_result.shape)==4
             else 1))

def entropy(s):
    vals = sorted(set(s))
    m=len(vals)
    p=np.zeros(m)
    for ii in range(m):
        p[ii]=np.sum((np.array(s)==vals[ii]))/len(s)
    return np.sum(-p*np.log(p))


def conditional_entropy(s1,s2):
    vals = sorted(set(s2))
    m=len(vals)
    p=np.zeros(m)
    ent=np.zeros(m)
    for ii in range(m):
        this_subset=(np.array(s2)==vals[ii])
        p[ii]=np.sum(this_subset)/len(s2)
        ent[ii] = entropy(np.array(s1)[this_subset])
    return np.sum(p*ent)

def assign_to_flat_clusters_by_means(
                                        data, #data to be clustered entries as rows
                                        centroids): #centroids provided as a dictionary {cluster:centroid}
    c_names, c_cent = zip(*centroids.items())
    c_cent = np.array(c_cent)
    dd = np.argmin(
        np.linalg.norm(data[:,None]-c_cent,axis=2),
        axis=1)
    return dd

def compute_centroids(data,clustering):
    centroids = {}
    clusters =  set(clustering)
    for cc in clusters:
        centroids[cc] = np.mean(data[clustering==cc,:],axis=0)
    return centroids

def color_with_binary_labels(data,labels):
    colored_data = ''
    for uu,this_d in enumerate(labels.astype(list)):
            cc= 'green' if this_d!=labels[0] else 'blue'
            colored_data+=colored(data[uu],cc)
    return colored_data

def prep_no_2loops(N):
    A=np.random.randn(N,N)
    Q=A>0
    Qb=A<0
    return np.triu(Q,1)+np.triu(Qb,1).transpose()

def plot_spectrum(A,sym='x'):
    ee=np.linalg.eigvals(A)
    plt.plot(ee.real,ee.imag,sym)



















































