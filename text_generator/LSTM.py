# coding:utf-8
from chainer import Link, Chain, ChainList
import chainer.links as L


class LSTM(Chain):
    def __init__(self, n_vocab, n_units):
        #n_units = 単語ベクトルのサイズ
        super(LSTM, self).__init__(
            embed=L.EmbedID(n_vocab, n_units, ignore_label=-1),
            l1=L.LSTM(n_units, n_units),
            l2=L.Linear(n_units, n_vocab)
        )

    def reset_state(self):
        self.l1.reset_state()

    def __call__(self, x):
        h0 = self.embed(x)
        h1 = self.l1(h0)
        y = self.l2(h1)
        return y


