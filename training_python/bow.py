# -*- coding: utf-8 -*-
#! /usr/bin/python
import sys
import MeCab

mecab = MeCab.Tagger("-Ochasen")
str = '''自然言語の理解をコンピュータにさせることは、自然言語理解とされている。
自然言語理解と、自然言語処理の差は、意味を扱うか、扱わないかという説もあったが、
最近は数理的な言語解析手法（統計や確率など）が広められた為、
パーサ（統語解析器）などが一段と精度や速度が上がり、
その意味合いは違ってきている。
もともと自然言語の意味論的側面を全く無視して達成できることは非常に限られている。
'''
split = str.split("\n")
dict = {}

for line in split:
    tokens = mecab.parse(line)
    token = tokens.split("\n")
    index = 0
    for ele in token:
        index += 1
        element = ele.split("\t")
        surface = element[0]
        if surface == "EOS":
            break
        
        count = 1
        if surface in dict:
            count += dict[surface] #キーが存在する場合はカウントアップ
        dict[surface] = count

dict = sorted(dict.items(), key=lambda x:x[1], reverse=True) #降順ソート
print(dict)