import numpy as np
import Signals, graphs
from submodDetect import *
import matplotlib.pyplot as plt
from wavelet_settings import FIGDIR
from matplotlib import rc, mlab
from matplotlib import font_manager
from matplotlib.collections import LineCollection

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
    for i in range(10):
        print "iters range: ", i
        positives.extend([alg.detect(x, k**2, rho) for x in [S.signal(k, mu, sigma) for i in range(iters/10)]])
        negatives.extend([alg.detect(x, k**2, rho) for x in [graphs.fixed_signal(S.graph().numVertices(), [], 0, sigma) for i in range(iters/10)]])
    positives.sort()
    negatives.sort()
    tmppoints = []
    for i in range(len(negatives)):
        tmppoints.append((float(iters-i)/iters, float(len([x for x in positives if x > negatives[i]]))/iters))
    return tmppoints


def plot_alg_scores(points,names,col,filename):
    fig = plt.figure(figsize=(6, 4))
    ax1 = fig.add_subplot(111)
    lines = [ax1.plot([x[0] for x in points[i]], [x[1] for x in points[i]], col[i], linewidth=3) for i in range(len(points))]
    ax1.plot([i for i in np.arange(0, 1.0, 0.01)], [i for i in np.arange(0, 1.0, 0.01)], "k-", markersize=1)
    ax1.legend(names, loc=4, prop={'size': 18})
    ax1.set_xlabel("False Positive Rate", size=18)
    ax1.set_ylabel("True Positive Rate", size=18)
    plt.gcf().subplots_adjust(bottom=0.20)
    fig.savefig(filename,dpi=100)

SD = SSSDetect()
SM = submodDetect()

pt = []
n1 = 10
S = Signals.Torus(n1**2,2)
k = int(n1**.5)
n = n1**2.
mu = 4.
for alg, rho in [(SM,4.*k),(SD,4./k)]:
    pt.append(ROC(alg, S, n=n1**2, d=2, k=k, mu=mu, iters=200, rho = rho))

plot_alg_scores(pt,['SM','SSS'],['k-','k-.'],'torus_submod_beta=.5_n1=10_mu=4_SMSSS.png')

import pickle
pickle.dump(pt,open('torus_submod_beta=.5_n1=10_mu=4.pt','w'))


n1 = 15
S = Signals.Torus(n1**2,2)
k = int(n1**.5)
n = n1**2.
mu = 4.
x = S.signal(k, mu, 1.)
SM.preprocess(S.graph())
rho = 4.*k
print SM.detect(x, k**2, rho)

