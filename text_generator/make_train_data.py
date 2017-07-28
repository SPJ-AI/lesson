# coding:utf-8
import numpy as np
import MeCab

import _pickle as pickle
import codecs
import re




# テキストファイルを行ごとのリストとして読み込み
input_txt = 'text/texts.txt'
f = codecs.open(input_txt, 'rb', 'utf-8')
lines = f.readlines()
lines = re.findall(r"<BOS>(.*?)<EOS>","".join(lines))

# 行ごとに形態素に分解
m = MeCab.Tagger ("-O wakati")
words_by_lines = []
for line in lines:
    words = m.parse(line).split()
    words = ['<BOS>'] + words + ['<EOS>']
    words_by_lines.append(words)

# vocabを作る
vocab = {}
vocab["<BOS>"] = 0
vocab["<EOS>"] = 1
for i,words_by_line in enumerate(words_by_lines):
    for word in words_by_line:
        if word not in vocab:
            vocab[word] = len(vocab)

# 行ごとにvocabで表現
dataset = []
max_length = 0
for i, words_by_line in enumerate(words_by_lines):
    length = len(words_by_line)
    if length > max_length:
        max_length = length
    datasetline = np.ndarray((length), dtype=np.int32)
    for j, word in enumerate(words_by_line):
        datasetline[j] = vocab[word]
    dataset.append(datasetline)

import os
if not os.path.exists("data"):
    os.mkdir("data")

print('line num:', len(dataset))
print('line_max_length:', max_length)
print('vocab size:', len(vocab))
pickle.dump(vocab, open('data/vocab.bin', 'wb'))
pickle.dump(dataset, open('data/train_data.bin', 'wb'))
