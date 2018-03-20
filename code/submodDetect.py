import numpy as np
import scipy
import scipy.sparse.linalg as splinalg
import scipy.linalg as sclinalg
import Signals, graphs
from Graph import Graph

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

    def spect_scan(self,y,rho,L):
        yt = y - np.mean(y)
        n = yt.shape[0]
        yt.shape = (n)
        nu_0 = .2
        nu_1 = .2
        tol = 1e-2
        obj_old = 100
        obj = 0
        max_iter = 1000
        Ls = scipy.sparse.csc_matrix(L)
        Is = scipy.sparse.csc_matrix(np.identity(n))
        for t in range(5):
            gamma = 10**-(t+1)
            k = 1
            while (abs(obj_old - obj) > tol and k < max_iter):
                k = k + 1
                step_size = .1
                obj_old = obj
                A = nu_0 * Ls + nu_1 * Is
                a = splinalg.spsolve(A,yt)
                La = np.dot(L,a)
                La.shape = (n)
                obj = nu_0 * rho + nu_1 + .25 * np.inner(yt,a) - gamma * (np.log(nu_0) + np.log(nu_1))
                nu_0 = max(nu_0 + step_size*(.25 * np.inner(a,La) - rho + gamma / nu_0),1e-5)
                nu_1 = max(nu_1 + step_size * (.25*np.sum(a**2) - 1 + gamma / nu_1),1e-5)
                # print nu_0, nu_1, np.inner(a,La)
        return nu_0 * rho + nu_1 + .25 * np.dot(yt,a)

    def preprocess(self,G):
        self.G = G
        return 1

    def detect(self, y, k = 1, rho = 0):
        W = np.array(self.G.adjMat)
        L = np.diag(W.sum(axis=1)) - W
        n = y.shape[0]
        if rho == 0:
            rho = 4. / (n)**0.5
        return self.spect_scan(y,rho,L)


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

    def detect(self, y, G, rho = 0, tol = 1e-5):
        n = y.shape[0]
        yt = y - np.mean(y)
        Uy = np.dot(self.eigvec.transpose(),yt)
        Uy.shape = (n)
        sss_adapt = 0.
        for rho in self.eigval:
            if rho > tol:
                eMult = (self.eigval > tol) * ((rho / self.eigval) * (self.eigval > rho) + 1. * (self.eigval <= rho))
                sss   = np.inner(eMult,Uy**2.)
                tau   = np.sum(eMult)
                if sss / tau > sss_adapt:
                    sss_adapt = sss / tau
        return sss_adapt


def edmonds_karp(C, source, sink, F = 0):
    n = len(C) # C is the capacity matrix
    if F == 0:
        F = [[0] * n for i in xrange(n)]
    # residual capacity from u to v is C[u][v] - F[u][v]
    while True:
        ret, path = EK_bfs(C, F, source, sink)
        if not ret:
            break
        # traverse path to find smallest capacity
        flow = min(C[u][v] - F[u][v] for u,v in path)
        # traverse path to update flow
        for u,v in path:
            F[u][v] += flow
            F[v][u] -= flow
    return sum(F[source][i] for i in xrange(n)), F, path

def EK_bfs(C, F, source, sink, tol=1e-5):
    xs = set()
    queue = [source]
    paths = {source: []}
    while queue:
        u = queue.pop(0)
        for v in xrange(len(C)):
            if C[u][v] - F[u][v] > tol and v not in paths:
                paths[v] = paths[u] + [(u,v)]
                if v == sink:
                    return 1, paths[v]
                queue.append(v)
                xs.add(v - 2)
    return 0, xs

def min_cut(y,W,eta):
    n = len(y)
    z = eta[0] - y
    C = [[0.]*(n+2),[0.]*(n+2)]
    for i in range(n):
        C.append([0,0] + [w*eta[1] for w in W[i]])
        if z[i] < 0:
            C[0][i + 2] = -z[i]
        else:
            C[i + 2][1] = z[i]
    Eval, F, xb = edmonds_karp(C,0,1)
    return Eval - sum(C[0]), xb

def edge_diff(y,W):
    n = len(y)
    ed = 0.
    for i in range(n):
        for j in range(n):
            if W[i][j] > 0:
                ed += abs(y[i] - y[j]) * W[i][j]
    return ed

def MRF_grad_barrier(y,W,eta,r,k,rho):
    n = len(y)
    val, x_set = min_cut(y,W,eta)
    x = np.zeros(n)
    x[list(x_set)] = 1.
    grad = np.array([-k + 1./(r * eta[0]) + np.sum(x), - rho + 1./(r * eta[1]) + edge_diff(x,W)])
    return grad, x

def MRF_grad(y,W,eta,k,rho,x_seed = [],nabla = []):
    n = len(y)
    a = (eta[0] - y) / eta[1]
    a.shape = (n)
    if len(x_seed) > 0:
        if MRF_test_sol(a,nabla,x_seed):
            val = np.dot(a,x_seed) + np.sum(np.abs(np.dot(nabla,x_seed))) - eta[0] * k - eta[1] * rho
            grad = np.array([np.sum(x_seed) - k, edge_diff(x_seed,W) - rho])
            return val, grad, x_seed
    val, x_set = min_cut(y,W,eta)
    x = np.zeros(n)
    x[list(x_set)] = 1.
    grad = np.array([np.sum(x) - k, edge_diff(x,W) - rho])
    return val - k*eta[0] - rho*eta[1], grad, x

def MRF_obj(y,W,eta,k,rho,x_seed = [], nabla = []):
    n = len(y)
    a = (eta[0] - y) / eta[1]
    a.shape = (n)
    if len(x_seed) > 0:
        if MRF_test_sol(a,nabla,x_seed):
            val = np.dot(a,x_seed) + np.sum(np.abs(np.dot(nabla,x_seed))) - eta[0] * k - eta[1] * rho
            return val, x_seed
    val, x_set = min_cut(y,W,eta)
    x = np.zeros(n)
    x[list(x_set)] = 1.
    return val - k*eta[0] - rho*eta[1], x

def bisect_rat(y,W,eta,k,rho,nabla = [],maxiters=20):
    alpha = 1.
    t = 1
    val, x = MRF_obj(y,W,eta,k,rho)
    val_best = -np.inf
    while abs(val - val_best) > 1e-1 and t < maxiters:
        if val > val_best:
            val_best = val
        alpha_l = alpha - 1 / (2.)**t
        alpha_r = alpha + 1 / (2.)**t
        Ml = MRF_obj(y,W,eta*alpha_l,k,rho, x, nabla)
        Mr = MRF_obj(y,W,eta*alpha_r,k,rho, x, nabla)
        if Ml[0] > Mr[0]:
            val, x = Ml
            alpha = alpha_l
        else:
            val, x = Mr
            alpha = alpha_r
        t += 1
    return - val / k**0.5


def W_to_nabla(W):
    n = W.shape[0]
    nabla = []
    for i in range(n):
        for j in range(i):
            if W[i,j] > 0:
                rt = [0.]*n
                rt[i] = W[i,j]
                rt[j] = -W[i,j]
                nabla.append(rt)
    return np.array(nabla)

def MRF_test_sol(a,nabla,x):
    z = np.dot(nabla,x)
    s = (z > 0) - (z < 0)
    B = np.dot(np.abs(nabla.T),(z == 0))
    beta = a + np.dot(nabla.T,s)
    return np.sum((beta - B > 0) * (x == 0) + (beta + B < 0) * (x == 1)) == 0.

def MRF_back_step(y,W,eta,k,rho,r=100., alpha = 1., tau=.5, c=.01, x_seed = [], nabla = []):
    val, grad, x = MRF_grad(y,W,eta,k,rho,r)
    eta_s = eta + alpha * grad
    val_s, grad_s, x_s = MRF_grad(y,W,eta_s,k,rho,r,x_seed = x_seed, nabla = nabla)
    while val_s < val - c * alpha * np.sum(grad_s*grad) or np.any(eta_s < 0.):
        alpha = alpha * tau
        eta_s = eta + alpha * grad
        val_s, grad_s, x_s = MRF_grad(y,W,eta_s,k,rho,r,x_seed = x_seed, nabla = nabla)
    return eta_s, val_s, grad_s, x_s, alpha

def MRF_bisect_old(y,W,eta,k,rho,alpha = 1.):
    val, grad, x = MRF_grad(y,W,eta,k,rho)
    nabla = W_to_nabla(np.array(W))
    eta_s = eta + alpha * grad
    val_s, grad_s, x_s = MRF_grad(y,W,eta_s,k,rho)
    val_last = np.inf
    k = 2.
    while val_s < val_last:
        if val_s > 0:
            print 'POSitive Objective error', eta_s, val_s, alpha
            break
        val_last = val_s
        alpha_l = alpha *.75
        alpha_r = alpha / .75
        MRF_left = MRF_grad(y,W,eta + alpha_l * grad,k,rho, x_seed = x_s, nabla = nabla)
        MRF_right = MRF_grad(y,W,eta + alpha_r * grad,k,rho, x_seed = x_s, nabla = nabla)
        print MRF_left[0], MRF_right[0]
        if MRF_left[0] > MRF_right[0]:
            alpha = alpha_l
            MRF_center = MRF_left
            val_s, grad_s, x_s = MRF_left
        else:
            alpha = alpha_r
            MRF_center = MRF_right
            val_s, grad_s, x_s = MRF_right
        eta_s = eta + alpha * grad
        k = 2.*k
    return eta_s, val_s, grad_s, x_s, alpha


def MRF_bisect_grad(y,W,k,rho,grad,alpha = 1.):
    nabla = W_to_nabla(np.array(W))
    eta_s = alpha * grad
    val_s, grad_s, x_s = MRF_grad(y,W,eta_s,k,rho)
    val_last = np.inf
    k = 2.
    while abs(val_s - val_last) > 1e-2:
        if val_s > 0:
            print 'POSitive Objective error', eta_s, val_s, alpha
            break
        val_last = val_s
        alpha_l = alpha *.75
        alpha_r = alpha / .75
        MRF_left = MRF_grad(y,W,alpha_l * grad,k,rho, x_seed = x_s, nabla = nabla)
        MRF_right = MRF_grad(y,W,alpha_r * grad,k,rho, x_seed = x_s, nabla = nabla)
        if MRF_left[0] > MRF_right[0]:
            alpha = alpha_l
            MRF_center = MRF_left
            val_s, grad_s, x_s = MRF_left
        else:
            alpha = alpha_r
            MRF_center = MRF_right
            val_s, grad_s, x_s = MRF_right
        eta_s = alpha * grad
        k = 2.*k
        print alpha, val_s
    return eta_s, val_s, grad_s, x_s, alpha

def MRF_bisect(y,W,eta,k,rho,alpha = 1.):
    val, grad, x = MRF_grad(y,W,eta,k,rho)
    nabla = W_to_nabla(np.array(W))
    eta_s = eta + alpha * grad
    alpha_l = alpha
    MRF_l = MRF_grad(y,W,eta_s,k,rho)
    val_last = np.inf
    while MRF_l[0] < val_last:
        if MRF_l[0] > 0:
            print 'POSitive Objective error', eta_s, val_s, alpha
            break
        val_last = MRF_l[0]
        MRF_c = MRF_l
        eta_s = eta + alpha * grad
        alpha = alpha_l
        alpha_l = alpha *.75
        MRF_l = MRF_grad(y,W,eta + alpha_l * grad,k,rho, x_seed = MRF_l[2], nabla = nabla)
    return eta_s, MRF_l[0], MRF_l[1], MRF_l[2], alpha

class submodDetect(object):
    def __init__(self):
        self.name="submod"

    def preprocess(self,G):
        self.W = G.adjMat
        return 1

    def detect_grid(self, y, G, rho = 0, eta_step = 20., grid_n = 50):
        n = y.shape[0]
        W = self.W.tolist()
        M = np.zeros((grid_n, grid_n))
        for eI0 in range(grid_n):
            for eI1 in range(grid_n):
                F = 0
                if F != 0:
                    F = [[val * eta[1] / (eta[1] + eta_step) for val in row] for row in F]
                eta = [eta_step * (eI0 + 1), eta_step * (eI1 + 1)]
                z = eta[0] - y
                C = [[0.]*(n+2),[0.]*(n+2)]
                for i in range(n):
                    C.append([0,0] + [w*eta[1] for w in W[i]])
                    if z[i] < 0:
                        C[0][i + 2] = -z[i]
                    else:
                        C[i + 2][1] = z[i]
                Eval, F = edmonds_karp(C,0,1,F = F)
                M[eI0, eI1] = sum(C[0]) - Eval
            print eI0
        print M
        val_min = 0.
        for k in np.arange(n) + 1:
            R = np.zeros((grid_n,grid_n))
            for eI0 in range(grid_n):
                for eI1 in range(grid_n):
                    R[eI0,eI1] = eta_step * ((eI0 + 1) * k + (eI1 + 1) * rho)
            val = np.min(M + R) / (k**0.5)
            if val > val_min:
                val_min = val
                k_min = k
        print k_min
        return val_min

    def detect_grad(self, y, k, rho, tol = 1e-5):
        n = y.shape[0]
        W = self.W.tolist()
        nabla = W_to_nabla(np.array(W))
        eta = np.array([1.,1.])
        alpha = 1.
        val_best = -np.inf
        val_last = 0
        val = 0.
        i = 0
        x = []
        maxiters = 100
        while abs(val - val_last) + abs(val - val_best) > 1e-1 and i < maxiters:
            i += 1
            val_last = val
            eta, val, grad, x, alpha = MRF_back_step(y,W,eta,k,rho,i,tau = .5, c=.001,x_seed = x, nabla = nabla)
            print val, grad, eta
            eta = (eta > tol) * (eta - tol) + tol
            if val > val_best:
                parm_best = (eta, val, grad, x)
                val_best = val
        print 'WOOOOOOOOOOOOOOOOOOOOOOO!!!!!!!!!!!!!!!!!!!!11111111111111'
        return - val_best / k**0.5

    def detect(self, y, k, rho, tol = 1e-5):
        n = y.shape[0]
        W = self.W.tolist()
        nabla = W_to_nabla(np.array(W))
        d_ave = np.mean(np.array(W).sum(1))
        eta = (np.array([rho / (k*d_ave),(k*d_ave / rho)]) * np.log(n))**0.5
        val = bisect_rat(y,W,eta,k,rho,nabla)
        return val

"""
n1 = 10
S = Signals.Torus(n1**2,2)
G = S.graph()
sigma = 1.
mu = 0.
k = n1/4
rho = 4*k
beta = S.signal(k, mu, 0.)
beta.shape = (n1**2)
W = G.adjMat.tolist()

stat_0 = []
tol = 1e-5
for trail in range(5):
    y = beta + np.random.normal(0,sigma,n1**2)
    # k = np.sum(beta > 0)
    # rho = edge_diff(beta > 0,W)
    eta = np.array([0.,0.])
    alpha = 1.
    val_best = -np.inf
    val_last = 0
    val = 0.
    i = 0
    maxiters = 100
    while abs(val - val_last) + abs(val - val_best) > 1e-1 and i < maxiters:
        i += 1
        val_last = val
        if val > tol:
            print 'pos obj'
            break
        eta, val, grad, x, alpha = MRF_bisect(y,W,eta,k,rho,alpha = alpha)
        eta = (eta > tol) * (eta - tol) + tol
        if val > val_best:
            parm_best = (eta, val, grad, x, alpha)
            val_best = val
        print eta, val
    stat_0.append(val)


# Accelerated Gradient

eta = np.array([.12,.13])
lam_last = 0.
ns_last = 0.
beta = 10.



for i in range(100):
    lam = (1. + (1. + 4.*lam_last**2.)**.5)/2.
    gam = (1 - lam_last)/lam
    grad, x = MRF_grad(y,W,eta,4,16)
    ns = eta + grad / beta
    eta = (1 - gam)*ns + gam*ns_last
    eta = (eta > 0) * eta
    ns_last = ns[:]
    gam_last = gam
    lam_last = lam
    print eta, sum(x)

eta += grad * .1 * step**-.5
"""


