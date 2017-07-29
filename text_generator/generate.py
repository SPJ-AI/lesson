# coding:utf-8
import numpy as np
from chainer import cuda, Function, gradient_check, report, training, utils, Variable
from chainer import datasets, iterators, optimizers, serializers
import chainer.functions as F
import chainer.links as L

import sys
import argparse
import _pickle as pickle
import MeCab
from LSTM import LSTM


BOS_INDEX = 0
EOS_INDEX = 1

# arguments
parser = argparse.ArgumentParser()
parser.add_argument('--unit_size',        type=int,   default=100)
parser.add_argument('--seed',           type=int,   default=1)
parser.add_argument('--gpu',            type=int,   default=-1)


args = parser.parse_args()


xp = cuda.cupy if args.gpu >= 0  else np
xp.random.seed(args.seed)
mecab = MeCab.Tagger ("-Ochasen")



vocab = pickle.load(open('data/vocab.bin','rb'))
train_data = pickle.load(open('data/train_data.bin', 'rb'))


rnn =  LSTM(len(vocab),args.unit_size)
model = L.Classifier(rnn)
if args.gpu >= 0:
    print('use GPU!')
    cuda.get_device(args.gpu).use()
    model.to_gpu()

serializers.load_npz('data/latest.model',model)

# vocabのキーと値を入れ替えたもの
ivocab = {}
for c, i in vocab.items():
    ivocab[i] = c


def get_index_a(_model):
    _model.predictor.reset_state()
    _sentence_index_a = []
    index = BOS_INDEX
    while index != EOS_INDEX:
        y = _model.predictor(xp.array([index], dtype=xp.int32))
        probability = F.softmax(y)
        probability.data[0] /= sum(probability.data[0])
        try:
            #確率によって、ランダムに１つ単語を選択
            #index = np.argmax(probability.data[0])
            index = xp.random.choice(range(len(probability.data[0])), p=probability.data[0])
            if index!=EOS_INDEX:
                #終了<EOS>でなかった場合
                _sentence_index_a.append(index)
        except Exception as e:
            print('probability error')
            break

    return _sentence_index_a


print('\n-=-=-=-=-=-=-=-')
for i in range(10):
    sentence_index_a = get_index_a(model)

    for index in sentence_index_a:
        sys.stdout.write( ivocab[index].split("::")[0] )
    print('\n-=-=-=-=-=-=-=-')

print('generated!')