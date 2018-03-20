import numpy as np
import Signals, graphs
from Detectors import SSSDetect, TreeDetect, AggregateDetect, MaxDetect
from wavelet_settings import FIGDIR
import pickle, time
from matplotlib import pyplot as plt

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
    for i in range(iters):
        print i
        positives.append(alg.detect(S.signal(k, mu, sigma), k=k, rho=rho))
        negatives.append(alg.detect(graphs.fixed_signal(S.graph().numVertices(), [], 0, sigma), k=k, rho=rho))
    positives.sort()
    negatives.sort()
    return positives, negatives

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


pt = []
SS = SSSDetect()
n1 = 15
S = Signals.Torus(n1**2,2)
k = int(n1**.5)
n = n1**2.
mu = 4.*k**.5 / 1.7
iters = 500
pt.append(ROC(SS, S, n=n1**2, d=2, k=k, mu=mu, iters=iters, rho = 4./k))

WD = TreeDetect()
n1 = 15
S = Signals.Torus(n1**2,2)
k = int(n1**.5)
n = n1**2.
mu = 4.*k**.5 / 1.7
iters = 500
pt.append(ROC(WD, S, n=n1**2, d=2, k=k, mu=mu, iters=iters, rho = 4./k))

MD = MaxDetect()
n1 = 15
S = Signals.Torus(n1**2,2)
k = int(n1**.5)
n = n1**2.
mu = 4.*k**.5 / 1.7
iters = 500
pt.append(ROC(MD, S, n=n1**2, d=2, k=k, mu=mu, iters=iters, rho = 4./k))


pt.append(pickle.load(open('data/submod_tor_n=225_mu=4.pt','r')))
def pt_to_points(pt,iters):
    Rpoints = []
    for i in range(iters):
        Rpoints.append((float(iters-i)/iters, float(len([x for x in pt[0] if x > pt[1][i]]))/iters))
    return Rpoints

Rpoints = [pt_to_points(p,len(p[0])) for p in pt]
filename = 'tor_n=225_mu=4.png'
col = ['k-','k-.','k:','k--']
names = ['SSS','Wavelet','Max','LESS']
plot_alg_scores(Rpoints,names,col,filename)



filename = 'tor_n=225_mu=4.png'
points = Rpoints[0:4]
points.append([Rpoints[4][i*10] for i in range(50)])
col = ['k-','k-.','k--','k:','k^']
names = ['SSS','Wavelet','Ave','Max','LESS']
fig = plt.figure(figsize=(6, 4))
ax1 = fig.add_subplot(111)
lines = [ax1.plot([x[0] for x in points[i]], [x[1] for x in points[i]], col[i], linewidth=3) for i in range(len(points))]
ax1.plot([x[0] for x in Rpoints[4]], [x[1] for x in Rpoints[4]], 'k-', linewidth=2)
ax1.plot([i for i in np.arange(0, 1.0, 0.01)], [i for i in np.arange(0, 1.0, 0.01)], "k-", markersize=1)
ax1.legend(names, loc=4, prop={'size': 18})
ax1.set_xlabel("False Positive Rate", size=18)
ax1.set_ylabel("True Positive Rate", size=18)
plt.gcf().subplots_adjust(bottom=0.20)
fig.savefig(filename,dpi=100)

