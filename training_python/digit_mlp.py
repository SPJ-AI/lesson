# -*- coding: utf-8 -*-
#! /usr/bin/python
from sklearn import datasets #テスト用のデータセット読み込み
digits = datasets.load_digits() #digitデータのロード

from sklearn.neural_network import MLPClassifier #アルゴリズムとしてMLPを使用
clf = MLPClassifier(max_iter=50,hidden_layer_sizes=(100,))

clf.fit(digits.data[:-1], digits.target[:-1]) #トレーニングデータを全て学習

array = clf.predict(digits.data[-1]) #トレーニングデータの最後のエントリの値を予測
print(array) #分類予測結果は8なので、あっている