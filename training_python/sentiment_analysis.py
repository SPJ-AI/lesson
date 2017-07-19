# -*- coding: utf-8 -*-
#! /usr/bin/python

f = open('pn.csv.m3.120408.trim') # 評価極性辞書ファイルの読み込み
lines = f.readlines()

dict = {} # 評価極性辞書の内容を保持する

for line in lines:
    array = line.split("\t")
    if len(array) > 1:
        score = 0;
        if array[1].strip() == 'p': #pならば+1
            score = 1
        elif array[1].strip() == 'n': #nならば-1
            score = -1
        
        dict[array[0].strip()] = score

f.close()

import MeCab
mecab = MeCab.Tagger("-Ochasen")
str = "好調で、アイデアが沸いてくる"

senti_score = 0 #感情スコア

tokens = mecab.parse(str) #Tokenize
token = tokens.split("\n") #以下、トークン結果確認
for ele in token:
    element = ele.split("\t")
    surface = element[0] #トークンの表層取得
    if surface == "EOS":
        break
        
    if surface in dict:
        senti_score += dict[surface] #辞書に単語が存在する場合、スコア加算

print(senti_score) # 1.0より大きい場合positive 2.0の場合neutral 3.0より小さい場合negative
