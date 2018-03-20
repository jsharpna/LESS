import numpy as np
import Signals, graphs
from submodDetect import *
from wavelet_settings import FIGDIR
import pickle, time

def ROC(alg, S,
                  n=400,
                  d=2,
                  k=3,
                  rho = 1.,
                  sigma=1.0,
                  iters=1000,
                  mu=5):
    positives = []
    negatives = []
    alg.preprocess(S.graph())
    positives.extend([alg.detect(x, k**2, rho) for x in [S.signal(k, mu, sigma) for i in range(iters)]])
    negatives.extend([alg.detect(x, k**2, rho) for x in [graphs.fixed_signal(S.graph().numVertices(), [], 0, sigma) for i in range(iters)]])
    positives.sort()
    negatives.sort()
    return positives, negatives

SM = submodDetect()

pt = []
n1 = 15
S = Signals.Torus(n1**2,2)
k = int(n1**.5)
n = n1**2.
mu = 4.*k**.5 / 1.7
pt = ROC(SM, S, n=n1**2, d=2, k=k, mu=mu, iters=10, rho = 4.*k)

pickle.dump(pt,open('torus_submod_beta=.5_n1=15_mu=4_' + str(int(time.time())) + '.pt','w'))


