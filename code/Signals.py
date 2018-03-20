import numpy as np
import random
from Graph import Graph
import graphs
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

"""
Signal objects define graphs and activation patterns and provide methods for
construction noisy activations and accessing the graph structure.
"""
class Torus2d(object):
    """
    A 2-dimensional torus on n vertices with square signals.
    """
    def __init__(self, n, d):
        self.n1 = int(np.sqrt(n))
        self.adjMat = graphs.torus2d(self.n1)
        self.G = Graph(self.adjMat)
        self.name = "torus2d"

    def graph(self):
        return self.G

    def signal(self, rho, mu, sigma):
        k = int(np.floor(rho/4.0))
        return graphs.square_signal(self.n1, k, mu, sigma, normalized=False)

    def rhoscaling(self):
        return self.n1


class Torus(object):
    """
    A d-dimensional torus on n vertices with square signals.
    """
    def __init__(self, n, d):
        self.d = d
        self.n1 = int(n**(1.0/self.d))
        Cn1 = np.matrix(np.zeros((self.n1, self.n1)))
        for i in range(self.n1):
            Cn1[i, (i+1) % self.n1] = 1
            Cn1[i, (i-1) % self.n1] = 1
            Cn1[(i+1) % self.n1, i] = 1
            Cn1[(i-1) % self.n1, i] = 1
        Ln1 = np.diagflat(np.sum(Cn1, axis=1)) - Cn1
        self.L = graphs.make_kron(Ln1, self.d)
        self.adjMat = np.diagflat(np.diag(self.L)) - self.L
        self.n = n
        self.G = Graph(self.adjMat)
        self.name = "torus"

    def graph(self):
        return self.G

    def signal(self, k, mu, sigma):
#         k = int(np.floor((rho/(2*self.d))**(1.0/(self.d-1))))
        return graphs.hypercube_signal(self.n1, k, self.d, mu, sigma)

    def signal_subsamp(self, k, mu, sigma, p=.5):
        init_point = random.choice(range(self.n))
        a = [1]*k + [0]*(self.n1 - k)
        b = [a[(init_point + j) % self.n1] for j in range(self.n1)]
        beta = np.kron(b,b)
        coords = np.where((np.random.uniform(0,1,self.n) < p)*(beta > 0))[0]
        return graphs.fixed_signal(self.n, coords, mu / p**.5, sigma)


class PlantedClique(object):
    def __init__(self, n, d, poff = 0.01):
        self.n1 = n
        M = np.matrix(np.zeros((self.n1, self.n1)))
        self.csize = int(np.floor(float(n)/d))
        self.d = d
        M[0:self.csize, 0:self.csize] = np.matrix(np.ones((self.csize, self.csize)) - np.eye(self.csize))
        M[self.csize:self.n1, self.csize:self.n1] = np.random.binomial(1, 0.5, [self.n1-self.csize, self.n1-self.csize])
        M[self.csize:self.n1, self.csize:self.n1] = np.array(M[self.csize:self.n1, self.csize:self.n1])*np.array(M[self.csize:self.n1, self.csize:self.n1].T)
        M[0:self.csize, self.csize:self.n1] = np.random.binomial(1, poff, [self.csize, self.n1-self.csize])
        M[self.csize:self.n1, 0:self.csize] = M[0:self.csize, self.csize:self.n1].T
        self.adjMat = M
        self.G = Graph(self.adjMat)
        self.name = "pclique"

    def graph(self):
        return self.G

    def signal(self, k, mu, sigma):
        return graphs.fixed_signal(self.n1, range(self.csize), mu, sigma)

class BalancedBinaryTree(object):
    """
    A balanced binary tree graph on n vertices with subtrees as signals
    """
    def __init__(self, n):
        self.n = n
        self.n1 = int(n/2) + 1
        self.adjMat = graphs.balanced_rooted_binary(self.n1)
        self.G = Graph(self.adjMat)
        self.name = "tree"

    def graph(self):
        return self.G

    def signal(self, rho, mu, sigma):
        """
        We're going to active rho subtrees in total. The depth of the subtrees
        depends on the rho parameter, it is np.ceil(np.log2(rho)). rho better be
        smaller than n/2.
        """
        level = int(np.ceil(np.log2(rho)))
        subtrees = random.sample(range(2**level), rho)
        coords = []
        for s in subtrees:
            coords.extend(self.subtree_coords(level, s))
        return graphs.fixed_signal(self.n, coords, mu, sigma)

    def subtree_coords(self, level, index):
        """
        Return the indices of all of the nodes in the index-th subtree at
        depth=level. This is a helper function to the signal procedure.
        """
        maxind = self.n
        numsubtrees = 2**level
        curr = -1
        while numsubtrees > 1:
            if index >= numsubtrees/2:
                curr = 2*curr+3
                index -= numsubtrees/2
            else:
                curr = 2*curr+2
            numsubtrees /= 2
        coords = []
        queue = [curr]
        while len(queue)>0:
            next = queue.pop()
            coords.append(next)
            if 2*next+2 < maxind:
                queue.append(2*next+2)
                queue.append(2*next+3)
        return coords


class Complete(object):

    def __init__(self, n):
        self.G = Graph(np.matrix(np.ones((n,n)) - np.eye(n)))
        self.n = n
        self.name = "complete"

    def graph(self):
        return self.G

    def signal(self, rho, mu, sigma):
        """
        Cut size of a k-cluster signal is k*(n-k). If we want rho = k*(n-k) then
        we pick k to be the solution to that quadratic equation.
        """
        if self.n**2 - 4*rho > 0:
            k = int(np.floor(float(self.n)/2 - np.sqrt(self.n**2 - 4*rho)/2))
        else:
            k = self.n/2
        coords = random.sample(range(self.n), k)
        return graphs.fixed_signal(self.n, coords, mu, sigma)

    def rhoscaling(self):
        return self.n

class KNN(object):

    def __init__(self, n,d, k=0):
        self.k = k
        if k == 0:
            self.k = int(n**(2.0/3))
        self.n = n
        self.X = np.random.uniform(0,1, [n,2])
        G = np.matrix(np.zeros((n,n)))
        self.neighs = []
        for i in range(n):
            dists = [np.sqrt((self.X[i,0] - self.X[j,0])**2 + (self.X[i,1] - self.X[j,1])**2) for j in range(n)]
            idxes = np.argsort(dists)
            self.neighs.append(idxes)
            idxes = idxes[1:self.k]
            G[i, idxes] = 1
            G[idxes, i] = 1
        self.G = Graph(G)
        self.name = "knn"


    def graph(self):
        return self.G

    def signal(self, k, mu, sigma):
        start = random.choice(range(self.n))
        coords = self.neighs[start][0:k]
        return graphs.fixed_signal(self.n, list(coords), mu, sigma)

    def signal_old(self, rho, mu, sigma):
        """
        Grow a ball until it can't be grown anymore without exceeding the cut-size criteria.
        """
        assert rho > self.k, "Rho < self.k = n^{2/3} so no candidate patterns"
        start = random.choice(range(self.n))
        coords = set([start])
        cutsize = np.sum([len(set(self.G.neighbors(x)) - coords) for x in coords])
        while cutsize < rho:
            neighbors = reduce(lambda x,y: x | y, [set(self.G.neighbors(x)) for x in coords])
            neighbors -= coords
            toadd = random.choice(list(neighbors))
            newset = coords | set([toadd])
            newcutsize = np.sum([len(set(self.G.neighbors(x)) - newset) for x in newset])
            if newcutsize > rho:
                break
            if newcutsize < cutsize:
                break
            cutsize = newcutsize
            coords = newset
        return graphs.fixed_signal(self.n, list(coords), mu, sigma)

    def rhoscaling(self):
        return self.k

    def kNNSignal(self, rho, mu, sigma=0):
        """
        This is going to be kind of for fun, but we're going to generate a
        scatter plot of the graph in 2-d and color the nodes based on their
        values according to a signal.
        """
        s = self.signal(rho,mu,sigma)
#         s = s-np.min(s)
#         s = s/np.max(s)
        fig = plt.figure(figsize=(10, 6.5))
        ax1 = fig.add_subplot(111)
        ax1.scatter(self.X[:,0], self.X[:,1], c = s, s=50)
        lines = []
        colors = []
        for i in range(self.n):
            for j in range(i, self.n):
                if self.G.getEdge(i,j) > 0:
                    lines.append(([self.X[i,0], self.X[i,1]], [self.X[j,0], self.X[j,1]]))
                    if np.abs(s[i]-s[j]) > 0:
                        colors.append((1.0, 0.0, 0.0, 0.5))
                    elif s[i] > 0:
                        colors.append((0.0, 1.0, 0.0, 1.0))
                    else:
                        colors.append((0.0, 0.0, 1.0, 0.1))
        segs = LineCollection(lines, colors=colors)
        ax1.add_collection(segs)

class EpsilonGraph(object):
    def __init__(self, n, d, eps_beta):
        self.epsilon = n**(-1.0/eps_beta)
        self.n = n
        self.X = np.random.uniform(0,1, [n,2])
        G = np.matrix(np.zeros((n,n)))
        self.neighs = []
        for i in range(n):
            dists = [np.sqrt((self.X[i,0] - self.X[j,0])**2 + (self.X[i,1] - self.X[j,1])**2) for j in range(n)]
            idxes = [j for j in range(n) if j != i and dists[j] < self.epsilon]
            self.neighs.append(np.argsort(dists))
            G[i,idxes] = 1
            G[idxes, i] = 1
        self.G = Graph(G)
        self.name = "epsilon"

    def graph(self):
        return self.G

    def signal(self, k, mu, sigma):
        start = random.choice(range(self.n))
        coords = self.neighs[start][0:k]
        return graphs.fixed_signal(self.n, list(coords), mu, sigma)

    def signal_old(self, rho, mu, sigma):
        """
        Grow a ball until it can't be grown anymore without exceeding the cut-size criteria.
        """
#         assert rho > self.k, "Rho < self.k = n^{2/3} so no candidate patterns"
        start = random.choice(range(self.n))
        coords = set([start])
        cutsize = np.sum([len(set(self.G.neighbors(x)) - coords) for x in coords])
        while cutsize < rho:
            neighbors = reduce(lambda x,y: x | y, [set(self.G.neighbors(x)) for x in coords])
            neighbors -= coords
            toadd = random.choice(list(neighbors))
            newset = coords | set([toadd])
            newcutsize = np.sum([len(set(self.G.neighbors(x)) - newset) for x in newset])
            if newcutsize > rho:
                break
            if newcutsize < cutsize:
                break
            cutsize = newcutsize
            coords = newset
        return graphs.fixed_signal(self.n, list(coords), mu, sigma)


    def rhoscaling(self):
        """
        Mean of the degree distribution is like n\epsilon^d. Since \epsilon is
        like n^{-1/5}, rho scales like n^{4/5}.
        """
        return np.mean(self.G.degree_dist())

    def EpsilonSignal(self, rho, mu, sigma=0):
        """
        This is going to be kind of for fun, but we're going to generate a
        scatter plot of the graph in 2-d and color the nodes based on their
        values according to a signal.
        """
        s = self.signal(rho,mu,sigma)
#         s = s-np.min(s)
#         s = s/np.max(s)
        fig = plt.figure(figsize=(10, 6.5))
        ax1 = fig.add_subplot(111)
        ax1.scatter(self.X[:,0], self.X[:,1], c = s, s=50)
        lines = []
        colors = []
        for i in range(self.n):
            for j in range(i, self.n):
                if self.G.getEdge(i,j) > 0:
                    lines.append(([self.X[i,0], self.X[i,1]], [self.X[j,0], self.X[j,1]]))
                    if np.abs(s[i]-s[j]) > 0:
                        colors.append((1.0, 0.0, 0.0, 0.5))
                    elif s[i] > 0:
                        colors.append((0.0, 1.0, 0.0, 1.0))
                    else:
                        colors.append((0.0, 0.0, 1.0, 0.1))
        segs = LineCollection(lines, colors=colors)
        ax1.add_collection(segs)


def createW(edgeOr,n):
    W = np.zeros((n,n))
    m = len(edgeOr)
    for eInd in range(m):
        eh = edgeOr[eInd][0]
        et = edgeOr[eInd][1]
        W[et,eh] += 1.
        W[eh,et] += 1.
    return W

def create_lap(edgeOr,n,gamma = 0.):
    graph_W = createW(edgeOr,n)
    lap_W = - graph_W
    diagb = graph_W.sum(0)
    for i_lap in range(n):
        lap_W[i_lap,i_lap] = diagb[i_lap] - gamma
    return lap_W

def make_grid(n1):
    n = n1**2
    edgeOr = []
    for i in range(n1-1):
        for j in range(n1 -1):
            edgeOr += [[(i+1)*n1 + j,i*n1+j]]
            edgeOr += [[i*n1 + j + 1,i*n1+j]]
    for i in range(n1-1):
        edgeOr += [[(n1-1)*n1 + i, (n1-1)*n1 + i + 1]]
        edgeOr += [[(i+1)*n1 -1, (i+2)*n1 -1]]
    beta =np.zeros((n))
    k1 = n1/2
    k2 = n1
    for i in range(k1,k2):
        for j in range(k1,k2):
            beta[i*n1 + j] = 1.
    return (edgeOr,beta)

def make_BBT(l,beta_seed = 6):
    n = 2**(l + 1) - 1
    beta = np.zeros(n)
    beta[beta_seed] = 1.
    edgeOr = []
    for i in np.arange(l) + 1:
        for j in range(2**(i -1)):
            edgeOr += [[2**(i-1) - 1 + j, 2**i - 1 + 2*j]]
            edgeOr += [[2**(i-1) - 1 + j, 2**i - 1 + 2*j + 1]]
            if beta[2**(i-1) - 1 + j] > 0:
                beta[2**i - 1 + 2*j] = 1.
                beta[2**i - 1 + 2*j + 1] = 1.
    return(edgeOr, beta)

def make_kron(lap_base, l):
    lap = lap_base.copy()
    n = 1.*lap_base.shape[0]
    for i in range(l-1):
        lap = np.kron(lap_base/n**(i+1),np.eye(n**(i+1))) + np.kron(np.eye(n),lap)
    return lap

"""
lap_base = np.array([[2,-1,-1,0,0,0],[-1,2,-1,0,0,0],[-1,-1,3,-1,0,0],[0,0,-1,3,-1,-1],[0,0,0,-1,2,-1],[0,0,0,-1,-1,2]])
        lap_W = make_kron(lap_base,n1)
        edgeOr = lap_to_el(lap_W)
        n = 6**n1
        beta = np.zeros(n)
        beta[0:(n/2)] = 1.
"""

class BBT(object):

    def __init__(self, n):
        l = int(np.log2(n + 1)) - 1
        EO, beta = make_BBT(l)
        self.G = Graph(np.matrix(createW(EO,n)))
        self.n = n
        self.l = l
        self.name = "BBT"

    def graph(self):
        return self.G

    def signal(self, depth, mu, sigma):
        beta_seed = 2**(depth + 1)
        EO, beta = make_BBT(self.l,beta_seed)
        coords = np.where(beta > 0)[0]
        return graphs.fixed_signal(self.n, coords, mu, sigma)

    def signal_subsamp(self, depth, mu, sigma, p = .5):
        beta_seed = 2**(depth + 1)
        EO, beta = make_BBT(self.l,beta_seed)
        coords = np.where((beta > 0)*(np.random.uniform(0,1,self.n) < p))[0]
        return graphs.fixed_signal(self.n, coords, mu/p**.5, sigma)

    def rhoscaling(self):
        return self.n



class Kron(object):

    def __init__(self, l):
        lap_base = np.array([[2,-1,-1,0,0,0],[-1,2,-1,0,0,0],[-1,-1,3,-1,0,0],[0,0,-1,3,-1,-1],[0,0,0,-1,2,-1],[0,0,0,-1,-1,2]])
        lap_W = make_kron(lap_base,l)
        W = np.diag(np.diag(lap_W)) - lap_W
        n = 6**l
        self.G = Graph(np.matrix(W))
        self.n = n
        self.l = l
        self.name = "BBT"

    def graph(self):
        return self.G

    def signal(self, height, mu, sigma):
        beta = [1]
        de = self.l - height
        for i in range(self.l):
            if i > de:
                beta = np.kron([1,0,0,0,0,0],beta)
            if i == de:
                beta = np.kron([1,1,1,0,0,0],beta)
            if i < de:
                beta = np.kron([1]*6, beta)
        coords = np.where(np.array(beta) > 0)[0]
        return graphs.fixed_signal(self.n, coords, mu, sigma)

    def signal_subsamp(self, height, mu, sigma, p=.5):
        beta = [1]
        de = self.l - height
        for i in range(self.l):
            if i > de:
                beta = np.kron([1,0,0,0,0,0],beta)
            if i == de:
                beta = np.kron([1,1,1,0,0,0],beta)
            if i < de:
                beta = np.kron([1]*6, beta)
        coords = np.where((np.random.uniform(0,1,self.n) < p)*(np.array(beta) > 0))[0]
        return graphs.fixed_signal(self.n, coords, mu/p**.5, sigma)


    def rhoscaling(self):
        return self.n




class KNN_var(object):

    def __init__(self, n,d,k = 0):
        self.k = int(n**(2.0/3))
        self.n = n
        self.X = np.random.uniform(0,1, [n,2])
        G = np.matrix(np.zeros((n,n)))
        for i in range(n):
            dists = [np.sqrt((self.X[i,0] - self.X[j,0])**2 + (self.X[i,1] - self.X[j,1])**2) for j in range(n)]
            idxes = np.argsort(dists)
            idxes = idxes[1:self.k]
            G[i, idxes] = 1
            G[idxes, i] = 1
        self.G = Graph(G)
        self.name = "knn"

    def graph(self):
        return self.G

    def signal(self, rho, mu, sigma):
        """
        Grow a ball until it can't be grown anymore without exceeding the cut-size criteria.
        """
        assert rho > self.k, "Rho < self.k = n^{2/3} so no candidate patterns"
        start = random.choice(range(self.n))
        coords = set([start])
        cutsize = np.sum([len(set(self.G.neighbors(x)) - coords) for x in coords])
        while cutsize < rho:
            neighbors = reduce(lambda x,y: x | y, [set(self.G.neighbors(x)) for x in coords])
            neighbors -= coords
            toadd = random.choice(list(neighbors))
            newset = coords | set([toadd])
            newcutsize = np.sum([len(set(self.G.neighbors(x)) - newset) for x in newset])
            if newcutsize > rho:
                break
            if newcutsize < cutsize:
                break
            cutsize = newcutsize
            coords = newset
        return graphs.fixed_signal(self.n, list(coords), mu, sigma)

    def rhoscaling(self):
        return self.k

    def kNNSignal(self, rho, mu, sigma=0):
        """
        This is going to be kind of for fun, but we're going to generate a
        scatter plot of the graph in 2-d and color the nodes based on their
        values according to a signal.
        """
        s = self.signal(rho,mu,sigma)
#         s = s-np.min(s)
#         s = s/np.max(s)
        fig = plt.figure(figsize=(10, 6.5))
        ax1 = fig.add_subplot(111)
        ax1.scatter(self.X[:,0], self.X[:,1], c = s, s=50)
        lines = []
        colors = []
        for i in range(self.n):
            for j in range(i, self.n):
                if self.G.getEdge(i,j) > 0:
                    lines.append(([self.X[i,0], self.X[i,1]], [self.X[j,0], self.X[j,1]]))
                    if np.abs(s[i]-s[j]) > 0:
                        colors.append((1.0, 0.0, 0.0, 0.5))
                    elif s[i] > 0:
                        colors.append((0.0, 1.0, 0.0, 1.0))
                    else:
                        colors.append((0.0, 0.0, 1.0, 0.1))
        segs = LineCollection(lines, colors=colors)
        ax1.add_collection(segs)
