from chainer import Chain
from chainer import reporter
import chainer.links as L
import chainer.functions as F

class BiLSTMBase(Chain):
  def __init__(self, input_size, n_labels, dropout=0.5):
    super(BiLSTMBase, self).__init__()
    with self.init_scope():
      # self.f_lstm = L.LSTM(None, feature_size, dropout)
      # self.b_lstm = L.LSTM(None, feature_size, dropout)
      self.nstep_bilstm = L.NStepBiLSTM(n_layers=1, in_size=input_size, out_size=input_size, dropout=dropout)
      self.l1 = L.Linear(input_size*2, n_labels)
      self.dropout = dropout

  def __call__(self, xs, ys):
    pred_ys = self.traverse(xs)
    loss = .0

    for pred_y, y in zip(pred_ys[:ys.size], ys):
      _loss = F.softmax_cross_entropy(pred_y, y)
      loss += _loss

    reporter.report({'loss': loss.data}, self)
    return loss

  def traverse(self, xs):
    hx, cx = None, None
    hx, cx, ys = self.nstep_bilstm(xs=xs, hx=hx, cx=cx)
    return [self.l1(y) for y in ys]

  def predict(self, xs):
    pred_ys = self.traverse(xs)
    pred_ys = [F.softmax(pred_y) for pred_y in pred_ys]
    pred_ys = [pred_y.data.argmax(axis=1) for pred_y in pred_ys]
    return pred_ys

  # def reset_state(self):
  #   self.f_lstm.reset_state()
  #   self.b_lstm.reset_state()

# class RNN(Chain):
#   def __init__(self, units):
#     """
#     units (tuple): e.g. (4, 5, 3)
#         - 1層目のLSTM: 4つのneuron
#         - 2層目のLSTM: 5つのneuron
#         - 3層目のLSTM: 3つのneuron
#     """
#     super(RNN, self).__init__()
#     with self.init_scope():
#       self.embed = L.EmbedID(1000, 100) # word embedding
#       self.mid = L.LSTM(None, 50) # the first LSTM layer  <-- mid層がLSTM
#       self.out = L.Linear(None, 1000) # the feed-forward output layer

#   def reset_state(self):
#     self.mid.reset_state()

#   def __call__(self, x):
#     # Given the current word ID, predict the next word.
#     h = self.embed(x) # <-- 入力cur_wordをembed+feed forward NNしてxへ
#     h = self.mid(h) # <-- xをLSTM層へ、更にその出力をhへ
#     h = self.out(h) # <-- hをfeed forward NNしてyへ
#     return h