# coding:utf-8
import numpy as np
from chainer import cuda, Function, gradient_check, report, training, utils, Variable
from chainer import datasets, iterators, optimizers, serializers
import chainer.links as L
from LSTM import LSTM

import sys
import argparse
import _pickle as pickle
import time



# arguments
parser = argparse.ArgumentParser()
parser.add_argument('--unit_size',        type=int,   default=100)
parser.add_argument('--batch_size',     type=int,   default=5)
parser.add_argument('--epochs',         type=int,   default=1)
parser.add_argument('--seed',           type=int,   default=1)
parser.add_argument('--gpu',            type=int,   default=-1)


args = parser.parse_args()


xp = cuda.cupy if args.gpu >= 0 else np
xp.random.seed(args.seed)


vocab = pickle.load(open('data/vocab.bin','rb'))
train_data = pickle.load(open('data/train_data.bin', 'rb'))
train_data_len = len(train_data)



rnn =  LSTM(len(vocab),args.unit_size)
model = L.Classifier(rnn)
if args.gpu >= 0:
    print('use GPU!')
    cuda.get_device(args.gpu).use()
    model.to_gpu()


optimizer = optimizers.SGD()
optimizer.setup(model)



def align_length(seq_list):
    global xp
    # 長さを揃えるため max_length に合わせて -1 で埋める
    max_length = 0
    for seq in seq_list:
        length = len(seq)
        if length > max_length:
            max_length = length
    seq_batch = [ np.full((max_length), -1, dtype=np.int32) for i in range(len(seq_list)) ]
    for i, data in enumerate(seq_list):
        seq_batch[i][:len(data)] = seq_list[i]
    return xp.array(seq_batch,dtype=xp.int32)

def compute_loss(seq_batch):
    loss = 0
    counter=0;
    for cur_word, next_word in zip(seq_batch.T, seq_batch.T[1:]):
        counter+=1
        loss += model(cur_word, next_word)
    print("loss:"+str(loss.data/counter))
    return loss



for epoch in range(1, args.epochs+1):
    print('epoch %d/%d' % (epoch, args.epochs))
    start = time.time()
    #shuffle
    np.random.shuffle(train_data)

    for i in range(0, train_data_len, args.batch_size):
        sys.stdout.write( '%d/%d\r' % (i, train_data_len) )
        sys.stdout.flush()
        model.predictor.reset_state()
        seq_list = [ train_data[(i+j)%train_data_len] for j in range(args.batch_size) ]
        seq_batch = align_length(seq_list)
        optimizer.update(compute_loss, seq_batch)


    print('epoch %d end.' % (epoch))
    elapsed_time = time.time() - start
    print(("epoch_time:{0}".format(elapsed_time)) + "[sec]")


serializers.save_npz('data/latest.model', model)
print('trained!')
