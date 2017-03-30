"""
Minimal character-level Vanilla RNN model. Written by Andrej Karpathy (@karpathy)
BSD License
"""
import numpy as np

# data I/O


a = np.load(open("char-rnn-snapshot.npz"))
Wxh = a["Wxh"]
Whh = a["Whh"]
Why = a["Why"]
bh = a["bh"]
by = a["by"]
mWxh, mWhh, mWhy = a["mWxh"], a["mWhh"], a["mWhy"]
mbh, mby = a["mbh"], a["mby"]
chars, data_size, vocab_size, char_to_ix, ix_to_char = a["chars"].tolist(), a["data_size"].tolist(), a["vocab_size"].tolist(), a["char_to_ix"].tolist(), a["ix_to_char"].tolist()




def sample(h, seed_ix, n, alpha):
  """
  sample a sequence of integers from the model
  h is memory state, seed_ix is seed letter for first time step
  """
  x = np.zeros((vocab_size, 1))
  x[seed_ix] = 1
  ixes = []
  for t in xrange(n):
    h = np.tanh(np.dot(Wxh, x) + np.dot(Whh, h) + bh)
    y = (np.dot(Why, h)  + by) * alpha
    p = np.exp(y) / np.sum(np.exp(y))
    ix = np.random.choice(range(vocab_size), p=p.ravel())
    x = np.zeros((vocab_size, 1))
    x[ix] = 1
    ixes.append(ix)
  return ixes





def part1(alpha):
    inds = sample(np.zeros([250,1]), 1, 200, alpha)
    m = ""
    for i in inds:
        m += ix_to_char[i]
    print m
# tries = [0.001, 0.1, 0.3, 0.6, 2, 3, 4, 5]
# for i in tries:    # best alpha = 2
#     print("alpha=" + str(i) + ":")
#     part1(i)



def part2(inputs):
    xs, hs = {}, {}
    hs[-1] = np.copy(np.zeros((250,1)))
    print inputs
    for t in xrange(len(inputs)):
        xs[t] = np.zeros((vocab_size,1)) # encode in 1-of-k representation
        xs[t][char_to_ix[inputs[t]]] = 1
        hs[t] = np.tanh(np.dot(Wxh, xs[t]) + np.dot(Whh, hs[t-1]) + bh) # hidden state
    inds = sample(hs[len(inputs)-1], char_to_ix[inputs[-1]], 50, 2)
    m = ""
    for i in inds:
        m += ix_to_char[i]
    print m

# part2("Beauty")


def part3():
    ind = char_to_ix[":"]
    x = np.zeros((vocab_size, 1))
    x[ind] = 1
    h = np.tanh(np.dot(Wxh, x) + bh)
    y = (np.dot(Why, h)  + by) * 2
    p = np.exp(y) / np.sum(np.exp(y))
    highest_xh = 0
    for i in range(250):
        if Why[0, i] * Wxh[i, 9] > highest_xh:
            highest_xh = i
    print highest_xh


# part3()
print Why[2, 73]
