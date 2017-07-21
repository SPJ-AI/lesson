# -*- coding: utf-8 -*-
#! /usr/bin/python
import sys
import MeCab

mecab = MeCab.Tagger("-Ochasen")

# 入力テキストの表層情報を取得し、セットに格納
def get_token_bow(text):
    token_set = set()
    tokens = mecab.parse(text)
    token = tokens.split("\n")
    for ele in token:
        element = ele.split("\t")
        surface = element[0]
        if surface == "EOS":
            break
        token_set.add(surface)
    return token_set

text1 = "燃料サーチャージが改定になりましたが、差額調整は行われますか？"
text2 = "サーチャージ変わった場合の差額調整はどうなる？"

set1 = get_token_bow(text1)
set2 = get_token_bow(text2)

# text1とtext2の間のヒットトークン数をカウント
hit_count = 0
for word in set1:
    if word in set2:
        hit_count += 1

# ヒットトークン数を、文章A、Bのトークン数で正規化
score = hit_count / ((len(set1) + len(set2)) / 2)
print(score)



