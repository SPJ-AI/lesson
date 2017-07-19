# -*- coding: utf-8 -*-
#! /usr/bin/python
import MeCab # TokenizerとしてMeCabを使用
mecab = MeCab.Tagger("-Ochasen") # MeCabのインスタンス化
f = open('text.tsv') # トレーニングファイルの読み込み
lines = f.readlines()

words = [] # 単語トークン表層一覧を保持するリスト
count = 0
dict = {} # テキスト:カテゴリのペアを保持する辞書

for line in lines:
    count += 1
    if count == 1:
        continue # ヘッダをSkip
    split = line.split("\t")
    if len(split) < 2:
        continue
    dict[split[0].strip()] = split[1].strip() # テキスト:カテゴリのペアを格納
    tokens = mecab.parse(split[0].strip()) # テキストの形態素解析
    token = tokens.split("\n")
    for ele in token:
        element = ele.split("\t")
        surface = element[0] #トークンの表層
        if surface == "EOS":
            break
        if surface not in words:
            words.append(surface) #表層の属するカテゴリを格納
f.close()
#print(words) # リストの中身の確認

data_array = [] # ベクトル化されたトレーニングデータをストアする配列
target_array = [] # ベクトル化された正解データをストアする配列
category_array = [] # 分類対象カテゴリ一覧をダブりなくストアする配列

for category in dict.values():
    if category not in category_array:
        category_array.append(category)

for text in dict.keys():
    print(text)
    entry_array = [0] * len(words) # 初期値0の配列を、wordsの長さ分生成（空ベクトル）
    target_array.append(category_array.index(dict[text])) # カテゴリ配列のインデックス番号をストア
    
    tokens = mecab.parse(text) # テキストの形態素解析
    token = tokens.split("\n")
    for ele in token:
        element = ele.split("\t")
        surface = element[0] #トークンの表層
        if surface == "EOS":
            break
        try:
            index = words.index(surface)
            entry_array[index] += 1
        except Exception as e:
            print(str(e))
            continue
    data_array.append(entry_array)

print(data_array)
print(category_array)
print(target_array)

from sklearn import svm #アルゴリズムとしてsvmを使用
clf = svm.SVC(gamma=0.001, C=100.)

clf.fit(data_array, target_array) #トレーニングデータを全て学習

query = "人工知能は人間を近々凌駕する"
query_array = [0] * len(words) # ベクトル化したクエリを格納する配列

tokens = mecab.parse(query) # クエリの形態素解析
token = tokens.split("\n")
for ele in token:
    element = ele.split("\t")
    surface = element[0] #トークンの表層
    if surface == "EOS":
        break
    try:
        index = words.index(surface)
        query_array[index] += 1
    except Exception as e:
        print(str(e))
        continue

print(query_array)
res = clf.predict(query_array) #トレーニングデータの最後のエントリの値を予測
print(res)
print(category_array[res[0]]) 