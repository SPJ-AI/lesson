# -*- coding: utf-8 -*-
#! /usr/bin/python
from sklearn import datasets #テスト用のデータセット読み込み
digits = datasets.load_digits() #digitデータのロード

from sklearn import svm #アルゴリズムとしてsvmを使用
clf = svm.SVC(gamma=0.001, C=100.) # おまじない

clf.fit(digits.data[:-1], digits.target[:-1]) #トレーニングデータを全て学習

array = clf.predict(digits.data[-1]) #トレーニングデータの最後のエントリの値を予測
print(array) #分類予測結果は8なので、あっている