import numpy as np
import scipy
import scipy.sparse.linalg as splinalg
import scipy.linalg as sclinalg
import SpanningTree
from Tree import Tree
from Graph import Graph

class AggregateDetect(object):
    """
    Global aggregate test statistic. Given a signal, simply compute the mean of
    the signal components and return that.
    """
    def __init__(self):
        self.name="aggregate"

    def preprocess(self, G):
        pass

    def detect(self, x, **kwargs):
        return np.mean(x)

    def MSE(self, S, k, mu, sigma, **kwargs):
        x = np.array(S.signal(k,mu,0.0))
        p = len(x)
        y = x + np.random.normal(0.,sigma,p)
        return np.sum((x - np.mean(y))**2)

    def threshold(self, G, sigma, delta):
        threshold = sigma*np.sqrt(2 * np.log(1.0/delta)/G.numVertices())
        return threshold

class EnergyDetect(object):
    """
    Given a signal, compute the ell_2 norm of the signal and return that.
    """
    def __init__(self):
        self.name="energy"

    def preprocess(self, G):
        pass

    def detect(self, x, **kwargs):
        return np.linalg.norm(x)**2

    def threshold(self, G, sigma, delta):
        """
        Uses P(||X|| > c\sqrt{n}) \le \exp \{-(c-1)^2 n/2\}
        """
        return sigma *(G.numVertices() + 2 * np.log(1.0/delta) + 2*np.sqrt(2*G.numVertices()*np.log(1.0/delta)))

    def compute_thresholds(self, total, G, sigma):
        return [sigma*G.numVertices()+i for i in range(-2*total, 2*total, 5)]

class MaxDetect(object):
    """
    Global aggregate test statistic. Given a signal, simply compute the max of
    the signal components and return that.
    """
    def __init__(self):
        self.name="max"

    def preprocess(self, G):
        pass

    def detect(self, x, **kwargs):
        return np.max(np.abs(x))

    def MSE(self, S, k, mu, sigma, **kwargs):
        x = S.signal(k,mu,0.0)
        p = len(x)
        x.shape = (p)
        y = x + np.random.normal(0.,sigma,p)
        return np.sum((x - np.mean(y))**2)

    def threshold(self, G, sigma, delta):
        return sigma*np.sqrt(2*np.log(float(G.numVertices())/delta))

    def compute_thresholds(self, total, G, sigma):
        return np.arange(0, 10, 10.0/total)


class SSSDetect(object):
    """
    This is the framework for signal detection on graphs using the spectral
    scan statistic. This takes as input a graph given by the dense weight
    matrix W.
    """
    def __init__(self):
        """
        Given a spanning tree algorithm and a graph, compute the spanning tree
        on the graph and also compute the corresponding tree wavelet basis.
        """
        self.name="SSS"
        self.L = None

    def preprocess(self,G):
        W = np.array(G.adjMat)
        self.L = np.diag(W.sum(axis=1)) - W
        self.eigval, self.eigvec = sclinalg.eigh(self.L)
        return 1

    def detect(self, y, **kwargs):
        tol = 1e-5
        n = y.shape[0]
        yt = y - np.mean(y)
        rho = kwargs['rho']
        Uy = np.dot(self.eigvec.transpose(),yt)
        Uy.shape = (n)
        eMult = (self.eigval > tol) * ((rho / (self.eigval + 1e-6)) * (self.eigval > rho) + 1. * (self.eigval <= rho))
        sss   = np.inner(eMult,Uy**2)
        return sss

    def MSE(self, S, k, mu, sigma, **kwargs):
        x = np.array(S.signal(k,mu,0.0))
        p = len(x)
        x.shape = (p)
        y = x + np.random.normal(0.,sigma,p)
        tol = 1e-5
        n = y.shape[0]
        yt = y - np.mean(y)
        rho = kwargs['rho']
        Uy = np.dot(self.eigvec.transpose(),yt)
        Uy.shape = (n)
        eMult = (self.eigval > tol) * ((rho / self.eigval) * (self.eigval > rho) + 1. * (self.eigval <= rho))
        sss = np.dot(self.eigvec,eMult**.5 * Uy) + np.mean(y)
        return np.sum((x - sss)**2)

class SSSAdaptDetect(object):
    """
    This is the framework for signal detection on graphs using the spectral
    scan statistic. This takes as input a graph given by the dense weight
    matrix W.
    """
    def __init__(self):
        """
        Given a spanning tree algorithm and a graph, compute the spanning tree
        on the graph and also compute the corresponding tree wavelet basis.
        """
        self.name="SSS"
        self.L = None

    def preprocess(self,G):
        W = np.array(G.adjMat)
        self.L = np.diag(W.sum(axis=1)) - W
        self.eigval, self.eigvec = sclinalg.eigh(self.L)
        return 1

    def detect(self, y, **kwargs):
        tol = 1e-5
        n = y.shape[0]
        yt = y - np.mean(y)
        Uy = np.dot(self.eigvec.transpose(),yt)
        Uy.shape = (n)
        sss_adapt = 0.
        for rho in self.eigval:
            if rho > tol:
                eMult = (self.eigval > tol) * ((rho / self.eigval) * (self.eigval > rho) + 1. * (self.eigval <= rho))
                sss   = np.inner(eMult,Uy**2.) - np.sum(eMult)
                bv = np.sum(eMult**2.)
                pval  = .5 * bv**.5 * ((1. + 2.*sss/bv)**.5 - 1.)
                if pval > sss_adapt:
                    sss_adapt = pval
        return sss_adapt

    def MSE(self, S, k, mu, sigma, **kwargs):
        x = np.array(S.signal(k,mu,0.0))
        p = len(x)
        x.shape = (p)
        y = x + np.random.normal(0.,sigma,p)
        tol = 1e-5
        n = y.shape[0]
        yt = y - np.mean(y)
        Uy = np.dot(self.eigvec.transpose(),yt)
        Uy.shape = (n)
        sss_adapt = 0.
        sss_x = np.mean(y)
        for rho in self.eigval:
            if rho > tol:
                eMult = (self.eigval > tol) * ((rho / self.eigval) * (self.eigval > rho) + 1. * (self.eigval <= rho))
                sss   = np.inner(eMult,Uy**2.) - np.sum(eMult)
                bv = np.sum(eMult**2.)
                pval  = .5 * bv**.5 * ((1. + 2.*sss/bv)**.5 - 1.)
                if pval > sss_adapt:
                    sss_adapt = pval
                    sss_x = np.dot(self.eigvec,eMult**.5 * Uy)
                    sss_x = sss_x - np.mean(sss_x) + np.mean(y)
        return np.sum((x - sss_x)**2)

class ErmDetect(object):
    """
    This is the framework for signal detection on graphs using the spectral
    scan statistic. This takes as input a graph given by the dense weight
    matrix W.
    """
    def __init__(self):
        """
        Given a spanning tree algorithm and a graph, compute the spanning tree
        on the graph and also compute the corresponding tree wavelet basis.
        """
        self.name="SSS"
        self.L = None

    def preprocess(self,G):
        W = np.array(G.adjMat)
        self.L = np.diag(W.sum(axis=1)) - W
        self.eigval, self.eigvec = sclinalg.eigh(self.L)
        return 1

    def detect(self, y, **kwargs):
        tol = 1e-5
        n = y.shape[0]
        mu = kwargs['mu']
        yt = y - np.mean(y)
        rho = kwargs['rho']
        Uy = np.dot(self.eigvec.transpose(),yt)
        Uy.shape = (n)
        eMult = (self.eigval > tol) * ((rho / self.eigval) * (self.eigval > rho) + 1. * (self.eigval <= rho))
        eMult = eMult[1:][::-1]
        Uy = Uy[1:][::-1]
        cum_sum = []
        cum_squ = []
        for k in range(n):
            cum_sum.append(np.sum(eMult[0:k]))
            cum_squ.append(np.sum(eMult[0:k]**2))
        cum_sum = np.array(cum_sum)
        cum_squ = np.array(cum_squ)
        K = np.arange(n)
        b1 = (2. - mu**2 * cum_sum / K)/(cum_sum**2 - cum_squ)
        b0 = (mu**2 + b1 * cum_sum)/K
        ktrue = 0
        for k in range(n):
            if not (np.isnan(b0[k]) or np.isinf(b0[k]) or np.isnan(b1[k]) or np.isinf(b1[k])):
                s1 = b0[k] - b1[k] * eMult
                s1 = s1 * (s1 > 0)
                if abs(np.sum(s1) - mu**2) < tol and abs(np.sum(eMult * s1) - 1) < tol:
                    ktrue = k
                    s1t = s1
        if ktrue == 0:
            s1t = 1.*(eMult == 1)
        erm = np.sum(s1t * Uy**2)
        if ktrue > 0:
            print "HAHA!"
        return erm


class TreeDetect(object):
    """
    This is the framework for signal detection on graphs using the tree wavelet
    basis. This object is given as input a spanning tree algorithm and a graph
    G, and it computes the tree wavelet basis for the resulting spanning tree
    returned by the algorithm. Then given a signal, it hits the signal with the
    basis and returns the maximum entry of the resulting vector.
    """
    def __init__(self, STAlg=SpanningTree.abTree,num_trees = 2):
        """
        Given a spanning tree algorithm and a graph, compute the spanning tree
        on the graph and also compute the corresponding tree wavelet basis.
        """
        self.st = STAlg
        self.name="wavelet"
        self.T = None
        self.clust_tree = None
        self.num_trees = num_trees

    def preprocess(self, G):
        self.clust_trees = []
        for clust in G.connectedComponents():
            T = self.st(G, vert_rest = list(clust))
            T.buildNeighborDict()
            self.clust_trees.append(self.compute_clust(T))

    def compute_clust(self, T):
        """
        Use the FindBalance algorithm to find a vertex partitioning and form the
        basis out of the chain structure of the partition.
        """
        n = T.numVertices()
        clust_tree = {1: [range(n)]}
        queue = [(T, range(n), 1)]
        while len(queue) > 0:
            T, mapping, depth = queue.pop()
            clusters = T.getRoot2()
            for c in clusters:
                try:
                    clust_tree[depth + 1].append([mapping[x] for x in c])
                except KeyError:
                    clust_tree[depth + 1] = [[mapping[x] for x in c]]
                if len(c) > 1:
                    Tnew = Tree(adjMat=np.matrix(T.adjMat[np.ix_(list(c), list(c))]))
                    queue.append((Tnew, [mapping[x] for x in c], depth + 1))
        return clust_tree

    def detect_stat(self, y,clust_tree):
        max_stat = 0
        for lev, clusts in clust_tree.items():
            lendep = len(clusts)
            if lendep == 1:
                c = clusts[0]
                max_stat = np.sum(y[c])/len(c)**0.5
            if lendep > 1:
                mst = np.max([np.sum(y[c])/len(c)**0.5 for c in clusts])
                bn = 1. # (np.log(np.log(lendep)) + 4*np.pi) / (2 * (2 * np.log(lendep))**0.5)
                an = (2 * np.log(lendep))**0.5
                if (mst - an) / bn > max_stat:
                    max_stat = (mst - an) / bn
        return max_stat


    def detect(self, y, **kwargs):
        return np.max([self.detect_stat(y,c) for c in self.clust_trees])



class WaveletDetect(object):
    """
    This is the framework for signal detection on graphs using the tree wavelet
    basis. This object is given as input a spanning tree algorithm and a graph
    G, and it computes the tree wavelet basis for the resulting spanning tree
    returned by the algorithm. Then given a signal, it hits the signal with the
    basis and returns the maximum entry of the resulting vector.
    """
    def __init__(self, STAlg=SpanningTree.abTree):
        """
        Given a spanning tree algorithm and a graph, compute the spanning tree
        on the graph and also compute the corresponding tree wavelet basis.
        """
        self.st = STAlg
        self.name="wavelet"
        self.T = None
        self.B = None

    def preprocess(self, G):
        self.T = self.st(G)
        self.B = self.compute_basis(self.T)

    def compute_basis(self, T):
        """
        Use the FindBalance algorithm to find a vertex partitioning and form the
        basis out of the chain structure of the partition.
        """
        n = T.numVertices()
        B = [np.array([1.0/np.sqrt(n) for i in range(n)])]
        queue = [(T, range(n))]
        while len(queue) > 0:
            T, mapping = queue.pop()
            if len(mapping) <= 1:
                continue
            clusters = T.getRoot2()
            subqueue = [clusters]
            while len(subqueue) > 0:
                c= subqueue.pop()
                if len(c) <= 1:
                    continue
                split = len(c)/2
                left = reduce(lambda x,y: x.union(y), [c[i] for i in range(split)])
                right = reduce(lambda x,y: x.union(y), [c[i] for i in range(split, len(c))])
                b = np.zeros(n)
                for x in left:
                    b[mapping[x]] = 1.0/(len(left))
                for x in right:
                    b[mapping[x]] = -1.0/(len(right))
                b /= np.sqrt(float(len(left) + len(right))/(len(left)*len(right)))
                B.append(b)
                subqueue.append([c[i] for i in range(split)])
                subqueue.append([c[i] for i in range(split, len(c))])
            for c in clusters:
                Tnew = Tree(adjMat=np.matrix(T.adjMat[np.ix_(list(c), list(c))]))
                queue.append((Tnew, [mapping[x] for x in c]))
        return np.matrix(B)

    def detect(self, x, G, **kwargs):
        """
        Return ||B^Tx||_{\infty}, which is the test statistic we are interested
        in.
        """
        if self.B == None:
            self.preprocess(G)
        return np.max(np.abs(self.B*x))

    def basis_sparsity(self, x, G):
        """
        Given a signal and graph (and optionally a basis), return the ||Bx||_0,
        computing the tree-wavelet basis as needed.
        """
        if self.B == None:
            self.preprocess(G)
        t = self.B*x
        return len([i for i in t if np.abs(i) >= 1e-10])

    def MSE(self, S, k, mu, sigma, **kwargs):
        x = S.signal(k,mu,0.0)
        B = np.array(self.B)
        p = len(x)
        x.shape = (p)
        y = x + np.random.normal(0.,sigma,p)
        t = np.dot(B,y)
        tau = (2*sigma**2. * np.log(p))**.5
        xhat = np.dot(B.T,t * (t > tau))
        xhat = xhat - np.mean(xhat) + np.mean(y)
        return np.sum((x - xhat)**2)

    def threshold(self, G, sigma, delta):
        return sigma*np.sqrt(2*np.log(float(G.numVertices())/delta))

    def compute_thresholds(self, total, G, sigma):
        return np.arange(0, 10, 10.0/total)
