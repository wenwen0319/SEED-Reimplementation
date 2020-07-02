# ==========================
# This is the reimplementation of the ICLR'20 paper:
# Inductive and Unsupervised Representation Learning on Graph Structured Objects

# Note:
# The original graph is pre-arranged based on the WEAVE strategy to improve the efficiency
# The pre-arranged file is in ./MUTAG_dataset/MUTAG_sample_200_length_10.mat
# This code is specifically for walk_length=10 and walks_per_graph=200

# This reimplementation is only for graph representation learning
# The quality evaluation (e.g., classification and t-SNE visualization) could be done by other toolbox
# ==========================

import tensorflow as tf
import scipy.io
import random
import os
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = "0" # assign GPU number

# Load pre-arranged samples
mat = scipy.io.loadmat('./MUTAG_dataset/MUTAG_sample_200_length_10.mat')
walk_list = mat['whole_id_list']
label_list = mat['whole_label_list'] # Ground-truth label
walk_num_list = mat['whole_num_list'] # Raw WEAVE encoding
walk_nnNum_list = mat['whole_nn_list']
walk_feature_list = mat['whole_feature_list']

# Set parameter
walks_per_graph = 200
walk_length = 10 # walk length of the random walk
h_WEAVE_dim = 30
rep_WEAVE_dim = 8 # WEAVE representation dimension
mb_size = 2000
total_samples = 188 # MUTAG dataset contains 188 graphs

# Arrange label list
label_list_real = np.empty([total_samples, 1])
for i in range(total_samples):
    label_list_real[i,0] = label_list[i * walks_per_graph, 0]


# Initialize network weights
def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
    return tf.random_normal(shape=size, stddev=xavier_stddev)

# Define leaky ReLU
def leak_relu(x, alpha):
    return tf.nn.relu(x) - alpha * tf.nn.relu(-x)

# Mean operation to Obtain final representation
def get_matrix_mean(rep_matrix):
    out_m = np.mean(rep_matrix, axis=0)
    return out_m

# =======================
# direct generate the revisit feature
# generate a size=(1, walk_length*walk_length) matrix
def gen_onehot_walk_list(walk_num_list):
    m = walk_num_list.shape
    m1 = m[0]
    out_m = np.zeros([1, 0]) # size of the one-hot label
    for i in range(walk_length):
        target_row = np.zeros([1, m1])
        walk_step_num = walk_num_list[i]
        walk_step_num = walk_step_num - 1
        target_row[0, walk_step_num] = 1
        out_m = np.concatenate((out_m, target_row), axis=1)
    return out_m

# generate the feature vector of the samples
def gen_concat_feature_list(whole_feature_list, idx_num):
    concat_onehot_feature = np.zeros([1, 7 * walk_length])
    target_feature_list = whole_feature_list[idx_num, :]

    for i in range(walk_length):
        feature_value = target_feature_list[i] # each node is a one-value feature
        each_onehot = np.zeros([1, 7])
        each_onehot[0, feature_value - 1] = 1 # ori feature value in [1,7] for 7 types
        concat_onehot_feature[0, i*7:(i+1)*7] = each_onehot # concat all onehot label

    return concat_onehot_feature

# ====================
# Arrange training feature and the edge
# output of the arranged data --> batch_size * [onehot_walk, nnNum_list, concat_feature]
def gen_train_walk_nn_fea(whole_walk_list, walk_num_list, walk_nnNum_list, walk_feature_list, batch_size):
    out_revisit_m = np.empty([batch_size, walk_length * walk_length])
    out_nn_m = np.empty([batch_size, walk_length])
    out_feature_m = np.empty([batch_size, walk_length * 7]) # 7 different atom type in MUTAG dataset
    whole_walk_list_copy = np.tile(whole_walk_list, [1])
    whole_walk_list_copy = whole_walk_list_copy - 1
    m_walk, n_walk = whole_walk_list.shape
    walk_num_list_copy = np.tile(walk_num_list, [1])
    random_idx = np.random.choice(m_walk, batch_size)

    for i in range(batch_size):
        onehot_walk_step = gen_onehot_walk_list(walk_num_list_copy[random_idx[i], :])
        walk_nnNum_step = walk_nnNum_list[random_idx[i], :]
        onehot_feature_step = gen_concat_feature_list(walk_feature_list, random_idx[i])        
        out_revisit_m[i, :] = onehot_walk_step
        out_nn_m[i, :] = walk_nnNum_step
        out_feature_m[i, :] = onehot_feature_step
    
    return out_revisit_m, out_nn_m, out_feature_m

# ====================
# Arrange referring feature and the edge
# output of the arranged data --> batch_size * [onehot_walk, nnNum_list, concat_feature]
def gen_refer_walk_nn_fea(whole_walk_list, walk_num_list, walk_nnNum_list, walk_feature_list):
    m_walk, n_walk = whole_walk_list.shape
    out_revisit_m = np.empty([m_walk, walk_length * walk_length])
    out_nn_m = np.empty([m_walk, walk_length])
    out_feature_m = np.empty([m_walk, walk_length * 7]) # 7 different atom type in MUTAG dataset
    whole_walk_list_copy = np.tile(whole_walk_list, [1])
    whole_walk_list_copy = whole_walk_list_copy - 1
    walk_num_list_copy = np.tile(walk_num_list, [1])
    
    for i in range(m_walk):
        onehot_walk_step = gen_onehot_walk_list(walk_num_list_copy[i, :])
        walk_nnNum_step = walk_nnNum_list[i, :]
        onehot_feature_step = gen_concat_feature_list(walk_feature_list, i)        
        out_revisit_m[i, :] = onehot_walk_step
        out_nn_m[i, :] = walk_nnNum_step
        out_feature_m[i, :] = onehot_feature_step
    
    return out_revisit_m, out_nn_m, out_feature_m

# Load training batches
def gen_train_samples(whole_walk_list, walk_num_list, walk_nnNum_list, walk_feature_list, batch_size):
    batch_WEAVE, batch_nn, batch_feature = gen_train_walk_nn_fea(whole_walk_list, walk_num_list, walk_nnNum_list, walk_feature_list, batch_size)
    return batch_WEAVE

# Load all samples for refer
def gen_refer_samples(whole_walk_list, walk_num_list, walk_nnNum_list, walk_feature_list):
    refer_WEAVE, refer_nn, refer_feature = gen_refer_walk_nn_fea(whole_walk_list, walk_num_list, walk_nnNum_list, walk_feature_list)
    return refer_WEAVE

WEAVE_dim = walk_length
WEAVE_X = tf.placeholder(tf.float32, shape=[None, walk_length * walk_length]) # Input WEAVE encoding

# ===============
# Encoder Network
# ===============
Encoder_W1 = tf.Variable(xavier_init([walk_length*walk_length, h_WEAVE_dim]))
Encoder_b1 = tf.Variable(tf.zeros(shape=[h_WEAVE_dim]))
Encoder_W2 = tf.Variable(xavier_init([h_WEAVE_dim, rep_WEAVE_dim]))
Encoder_b2 = tf.Variable(tf.zeros(shape=[rep_WEAVE_dim]))
theta_Encoder = [Encoder_W1, Encoder_W2, Encoder_b1, Encoder_b2]

def WEAVE_encoder(X):
    inputs = X    
    E_h1 = leak_relu(tf.matmul(inputs, Encoder_W1) + Encoder_b1, 0.3)
    C_log_prob = tf.matmul(E_h1, Encoder_W2) + Encoder_b2
    return C_log_prob

# ===============
# Decoder Network
# ===============
Decoder_W1 = tf.Variable(xavier_init([rep_WEAVE_dim, h_WEAVE_dim]))
Decoder_b1 = tf.Variable(tf.zeros(shape=[h_WEAVE_dim]))
Decoder_W2 = tf.Variable(xavier_init([h_WEAVE_dim, walk_length*walk_length]))
Decoder_b2 = tf.Variable(tf.zeros(shape=[walk_length*walk_length]))
theta_Decoder = [Decoder_W1, Decoder_W2, Decoder_b1, Decoder_b2]

def WEAVE_decoder(X):
    inputs = X
    D_h1 = leak_relu(tf.matmul(inputs, Decoder_W1) + Decoder_b1, 0.3)
    C_log_prob = tf.matmul(D_h1, Decoder_W2) + Decoder_b2
    return C_log_prob

encode_WEAVE_X = WEAVE_encoder(WEAVE_X)
decode_WEAVE_X = WEAVE_decoder(encode_WEAVE_X)
cost_WEAVE_ED = tf.reduce_mean(tf.square(decode_WEAVE_X - WEAVE_X)) # encoder-decoder loss

# Optimizer
train_WEAVE_ED = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost_WEAVE_ED, var_list=[theta_Encoder, theta_Decoder])

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
sess.run(tf.global_variables_initializer())

# ===============
# Start training
# ===============
for epoch_i in range(4000):

	# Get WEAVE batch
    walk_train_batch = gen_train_samples(walk_list, walk_num_list, walk_nnNum_list, walk_feature_list, mb_size)

    # Training encoder and decoder
    for repeat_i in range(1):
        _ = sess.run([train_WEAVE_ED], {WEAVE_X: walk_train_batch})

    # Show progress
    if (epoch_i % 50)==0:
        loss_WEAVE = sess.run([cost_WEAVE_ED], {WEAVE_X: walk_train_batch})
        print('Iteration = ', epoch_i, ' Loss = ', loss_WEAVE)

    # Calculate & save representations
    if (epoch_i % 200)==0:
        
        out_WEAVE_rep_each = np.empty([total_samples * walks_per_graph, rep_WEAVE_dim]) # representation of all walks
        refer_batch = gen_refer_samples(walk_list, walk_num_list, walk_nnNum_list, walk_feature_list) # get complete target
        rep_WEAVE_each = sess.run(encode_WEAVE_X, {WEAVE_X: refer_batch}) # get the represent of each walk
        out_WEAVE_rep_each = rep_WEAVE_each
        out_WEAVE_rep = np.empty([total_samples, rep_WEAVE_dim]) # final representations of all graphs

        # Calculate (mean operation) the representation of each graph
        for i in range(total_samples):
        	# representations of all walk representatoins of a graph
            target_matrix = out_WEAVE_rep_each[i*walks_per_graph:(i+1)*walks_per_graph, :]
            mean_result = get_matrix_mean(target_matrix) # mean operation
            out_WEAVE_rep[i,:] = mean_result

        # save results
        save_folder_name = './representation_MUTAG_sample_200_length_10'        
        mat_save = {}
        mat_save['rep_WEAVE_representation'] = out_WEAVE_rep
        mat_save['ground_truth_label'] = label_list_real
        print('Inferring and saving graph representations to -> ', save_folder_name + '/rep_loop_{}.mat'.format(str(epoch_i)))
        scipy.io.savemat((save_folder_name + '/rep_loop_{}.mat'.format(str(epoch_i))), mat_save)

