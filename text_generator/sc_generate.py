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
m = MeCab.Tagger ("-O wakati")



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

def get_next_word_prob(_model, word, next_word, needModelStateReset=False):
    if needModelStateReset:
        _model.predictor.reset_state()
    _sentence_index_a = []
    index = vocab[word]
    while index != EOS_INDEX:
        y = _model.predictor(xp.array([index], dtype=xp.int32))
        probability = F.softmax(y)
        next_probs = probability.data[0]
        m = np.argsort(probability.data[0])
        break
    
    # In this case, the input could be an unknow word.
    if next_word not in vocab:
        return (0.0, 0.0)
    
    next_index = vocab[next_word]
    k, = np.where(m == next_index)
    order_prob = k[0] / len(m)
    next_prob = next_probs[k[0]]

    return (order_prob, next_prob, k[0])

def suggest_corrections(order_prob, next_prob, index, num=5):
    suggestions = []
    
    # Step1: If it's lower than 25% of the over-all probs
    if order_prob < 0.25:
        count = 0
        for ind in order_prob[::-1]:
            w = ivocab[ind]
            suggestions.append(w)
            count += 1
            if count >= num:
                break
    return suggestions

def text_correction(_model, text):
    tokens = m.parse(text).split()
    for i in range(len(tokens)):
        if i == len(tokens) - 1:
            break
        
        word = tokens[i]
        next_word = tokens[i + 1]
        needModelStateReset = True if i == 0 else False
        
        (order_prob, next_prob, index) = get_next_word_prob(_model, word, next_word, needModelStateReset)
        suggestions = suggest_corrections(order_prob, next_prob, index)
        if len(suggestions) > 0:
            print("low prob detected", order_prob, next_word, suggestions)

print('\n-=-=-=-=-=-=-=-')
#for i in range(1):
    #sentence_index_a = get_index_a(model)
order_prob, next_prob, index = get_next_word_prob(model, "最大", "の", needModelStateReset=True)
print(order_prob, next_prob, index)
order_prob, next_prob, index = get_next_word_prob(model, "の", "害悪")
print(order_prob, next_prob, index)

'''
    for index in sentence_index_a:
        sys.stdout.write( ivocab[index].split("::")[0] )
    print('\n-=-=-=-=-=-=-=-')
'''

print('generated!')