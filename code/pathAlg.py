import numpy as np
import Signals, graphs
from wavelet_settings import FIGDIR
import pickle, time
from matplotlib import pyplot

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
    return sum(F[source][i] for i in xrange(n)), F, path

def EK_bfs(C, F, source, sink, tol=0.):
    xs = set()
    queue = [source]
    paths = {source: []}
    while queue:
        u = queue.pop(0)
        for v in xrange(len(C)):
            if C[u][v] - F[u][v] > tol and C[u][v] > tol and v not in paths:
                paths[v] = paths[u] + [(u,v)]
                if v == sink:
                    return 1, paths[v]
                queue.append(v)
                xs.add(v - 2)
    return 0, xs

def edge_diff(y,W):
    n = len(y)
    ed = 0.
    for i in range(n):
        for j in range(n):
            if W[i][j] > 0:
                ed += (y[i] - y[j])*(y[i] > y[j]) * W[i][j]
    return ed


n1 = 15
S = Signals.Torus(n1**2,2)
sigma = 1.
k = 4
n = n1**2
mu = 4.*k**.5 / 1.7
x = np.array(S.signal(k, mu, 0.))
x.shape = (n)
y = x + np.random.normal(0,sigma,n)
W = S.G.adjMat.tolist()
rho = 4./k
C_star = np.where(x > 0)[0]
x_star = 1.*(x > 0)
print edge_diff(x_star,W)
torus_image(y)
torus_image(x_star)
def make_C(y,W,rho,nu):
    z = rho - nu * y
    n = len(z)
    C = [[0.]*(n+2),[0.]*(n+2)]
    for i in range(n):
        C.append([0,0] + W[i])
        if z[i] < 0:
            C[0][i + 2] = -z[i]
        else:
            C[i + 2][1] = z[i]
    return C

nu = max(rho/ np.max(y),0)
m = n + 2
F = [[0] * m for i in xrange(m)]
tol=1e-5
reg_path = [(nu,F,F)]

rad = []
X_quiv = []
Y_quiv = []
U_quiv = []
V_quiv = []
W = np.zeros((n,n))
for u1 in range(n1):
    for v1 in range(n1):
        for u2 in range(n1):
            for v2 in range(n1):
                if np.abs(u1 - v1) <= 1 and np.abs(u2 - v2) <= 1 and (u1 != v1 or u2 != v2):
                    W[u2*n1 + u1, v2*n1 + v1] = np.exp((v1 - u1 + v2 - u2) / 3.)
                    if (v1 - u1 + v2 - u2) > 0.:
                        rad.append((v1 - u1 + v2 - u2))
                        X_quiv.append(u1)
                        Y_quiv.append(u2)
                        U_quiv.append((v1 - u1)*.5)
                        V_quiv.append((u2 - v2)*.5)

immat = np.zeros((n1,n1))
for i in range(n1):
    immat[i,:] = x_star[i*n1:(i+1)*n1]
fig = pyplot.figure(frameon=False)
ax_size = [0,0,1,1]
fig.add_axes(ax_size)
pyplot.imshow(immat,cmap='Greys',interpolation='nearest')
pyplot.axis('off')
for u1 in range(n1):
    for u2 in range(n1):
        pyplot.arrow(u1,u2,.5,.5,width=.05,fc='w')
pyplot.show()

trials = 100
#while nu < np.inf:
for knots in range(trials):
    C = make_C(y,W,rho,nu)
    I_pos_resid = []
    I_pos_cap = []
    for i in range(n):
        if nu*y[i] >= rho:
            if F[0][i+2] > C[0][i+2] - tol:
                I_pos_cap.append(i)
            else:
                I_pos_resid.append(i)
    I_neg_resid = []
    I_neg_cap = []
    for i in range(n):
        if rho - nu*y[i] > 0:
            if F[i+2][1] >= C[i+2][1] - tol:
                I_neg_cap.append(i)
            else:
                I_neg_resid.append(i)
    if I_neg_cap:
        partial_C_neg = [[0] * m for i in xrange(m)]
        for i in I_neg_cap:
            partial_C_neg[i+2][1] = y[i]
        for u in range(n):
            for v in range(n):
                if F[u+2][v+2] > tol:
                    partial_C_neg[u+2][v+2] = np.inf
        for v in range(n):
            if F[0][v+2] > tol:
                partial_C_neg[0][v+2] = np.inf
        val_neg, partial_F_neg, path = edmonds_karp(partial_C_neg, 0, 1)
    else:
        partial_C_neg = [[0.]*m for it in xrange(m)]
        val_neg, partial_F_neg = 0., [[0.]*m for it in xrange(m)]
    if 1:
        partial_C_pos = [[0] * m for i in xrange(m)]
        for u in range(n):
            for v in range(n):
                if F[u+2][v+2] < C[u+2][v+2] - tol:
                    partial_C_pos[u+2][v+2] = np.inf
                else:
                    partial_C_pos[u+2][v+2] = partial_F_neg[u+2][v+2]
        for u in range(n):
            if 0 < rho - nu*y[u] > F[u+2][1] + tol:
                partial_C_pos[u+2][1] = np.inf
        for v in range(n):
            if 0 <= nu*y[v] - rho > F[0][v + 2] + tol:
                partial_C_pos[0][v+2] = np.inf
            if 0 <= nu*y[v] - rho < F[0][v + 2] + tol:
                partial_C_pos[0][v+2] = y[v] + partial_F_neg[0][v+2]
        val_pos, partial_F_pos, path = edmonds_karp(partial_C_pos, 0, 1)
    else:
        val_pos, partial_F_pos = 0, [[0.]*m for it in xrange(m)]
    partial_F = [[partial_F_pos[u][v] - partial_F_neg[u][v] for v in xrange(m)] for u in xrange(m)]
    nu_cap_int = np.inf
    for u in np.arange(n)+2:
        for v in np.arange(n)+2:
            if partial_F[u][v] > tol and C[u][v] > F[u][v] + tol:
                nu_cap_int = min((C[u][v] - F[u][v])/partial_F[u][v],nu_cap_int)
    nu_zero_int = np.inf
    for u in np.arange(n)+2:
        for v in np.arange(n)+2:
            if partial_F[u][v] < -tol:
                nu_zero_int = min(-F[u][v]/partial_F[u][v],nu_zero_int)
    nu_zero_source = np.inf
    for v in np.arange(n)+2:
        if partial_F[0][v] < -tol and F[0][v] > tol:
            nu_zero_source = min(-F[0][v]/partial_F[0][v],nu_zero_source)
    nu_zero_sink = np.inf
    for u in np.arange(n)+2:
        if partial_F[u][1] < -tol:
            nu_zero_sink = min(-F[u][1]/partial_F[u][1],nu_zero_sink)
    nu_cap_source = np.inf
    for v in I_pos_resid:
        if y[v] < partial_F[0][v+2] - tol and C[0][v+2] - F[0][v+2] > tol:
            nu_cap_source = min((C[0][v+2] - F[0][v+2]) / (partial_F[0][v+2] - y[v]), nu_cap_source)
    nu_cap_sink = np.inf
    for u in I_neg_resid:
        if y[u] + partial_F[u+2][1] > tol and C[u+2][1] > F[u+2][1] + tol:
            #print C[u+2][1] - F[u+2][1], (partial_F[u+2][1] + y[u]), rho, (C[u+2][1] - F[u+2][1]) / (partial_F[u+2][1] + y[u]), y[u], rho / y[u]
            nu_cap_sink = min((C[u+2][1] - F[u+2][1]) / (partial_F[u+2][1] + y[u]), nu_cap_sink)
    y_next_newso = np.max(y[nu*y < rho - tol])
    #print y_next_newso
    nu_next_newso = rho / y_next_newso
    if nu_next_newso < 0:
        nu_next_newso = np.inf
    nu_next = min(nu_next_newso, nu + min( nu_cap_sink,nu_cap_source,nu_cap_int, nu_zero_sink,nu_zero_source,nu_zero_int ))
    if 1:
        print nu, I_pos_cap, I_neg_cap, I_pos_resid
        print nu_next_newso - nu, nu_cap_sink,nu_cap_source,nu_cap_int, nu_zero_sink,nu_zero_source,nu_zero_int
        print np.sum(np.array(partial_F)), np.sum(np.array(partial_F_pos)), np.sum(np.array(partial_F_neg))
    F = [[F[u][v] + (nu_next - nu) * partial_F[u][v] for v in xrange(m)] for u in xrange(m)]
    if nu_next - nu < 1e-2:
        nu = nu + 1e-2
        nu_next = nu
        C = make_C(y,W,rho,nu)
        val, F, path = edmonds_karp(C,0,1)
    nu = nu_next
    C = make_C(y,W,rho,nu)
    val, F, path = edmonds_karp(C,0,1,F=F)
    reg_path.append((nu,F,partial_F))

reg_flows = [sum(pa[1][0]) for pa in reg_path]
reg_nus = [pa[0] for pa in reg_path]
pyplot.plot(reg_nus,reg_flows)
pyplot.show()

print [(partial_F[v+2][1],y[v]) for v in range(n)]
print [(F[v+2][1],C[v+2][1],partial_C_pos[v+2][1]) for v in I_neg_resid]
x_path = []
nu_path = [0.]
MSE_path = [np.sum(x_star)]
C_path = []
for nu,rp,df in reg_path:
    x = EK_bfs(make_C(y,W,rho,nu),rp,0,1)[1]
    C_path.append(list(x))
    if x not in x_path:
        nu_path.append(nu)
        x_path.append(x)
        x_hat = np.zeros(n)
        x_hat[list(x)] = 1.
        MSE_path.append(np.sum((x_hat - x_star)**2.))

x_hat = np.zeros(n)
x_hat[list(x_path[3])] = 1.
pyplot.plot(nu_path,MSE_path)
pyplot.xlabel('nu')
pyplot.ylabel('MSE')
pyplot.show()

def torus_image(x,filename):
    n = int(len(x)**0.5)
    immat = np.zeros((n,n))
    for i in range(n):
        immat[i,:] = x[i*n:(i+1)*n]
    fig = pyplot.figure(frameon=False)
    ax_size = [0,0,1,1]
    fig.add_axes(ax_size)
    pyplot.imshow(immat,cmap='Greys',interpolation='nearest')
    pyplot.axis('off')
    pyplot.savefig(filename,dpi=100)

torus_image(y, 'rect_n=15_noisy.png')
torus_image(x_star, 'rect_n=15_true.png')
torus_image(x_hat, 'rect_n=15_hat.png')

ipa = 100
nu, f, df = reg_path[ipa]
C = make_C(y,W,rho,nu)

for u in range(m):
    for v in range(m):
        if f[u][v] < 0. or f[u][v] > C[u][v]:
            print C[u][v], f[u][v]

val, f, path = edmonds_karp(C,0,1)
A = list(EK_bfs(C,f,0,1)[1])
x = np.zeros(n)
x[A] = 1.
print sum(f[0])
print sum((rho - nu*y)*(rho < nu*y))
print sum((rho - nu*y)[A])
print edge_diff(x,W)
print sum(f[0]) + sum((rho - nu*y)*(rho < nu*y))
print sum((rho - nu*y)[A]) + edge_diff(x,W)

for u in xrange(n):
    for v in xrange(n):
        if f[u][v] != 0.:
            print u,v, f[u][v]

np.where(nu*y >= rho)[0]
I_neg = np.where(nu*y >= rho)[0]
for u in xrange(m):
    for v in xrange(m):
        if u == 0 and v > 1 and y[v - 2] :
            partial_C[u][v] =


i = I[0]
ret, path = EK_bfs(C, F, i+2, 1)
sink_e = path.pop()
flow_int = min(C[u][v] - F[u][v] for u,v in path)
flow_sink = C[sink_e[0]][sink_e[1]]
y_sink = y[sink_e[0]]
y_source = y[i]
nu_next_int = nu + flow_int / y_source
if y_source > -y_sink:
    nu_next_sink = 2*rho / (y_source + y_sink)
else:
    nu_next_sink = np.inf
y_next_newso = np.max(y[nu*y < rho])
nu_next_newso = rho / y_next_newso
if nu_next_newso < min(nu_next_int,nu_next_sink):
    flow_grad = (nu_next_newso, y_source, path)
for u,v in path:
    F[u][v] += (nu_next - nu)* y_source
    F[v][u] -= (nu_next - nu)* y_source



if not ret:
            break
        # traverse path to find smallest capacity

        # traverse path to update flow
        for u,v in path:
            F[u][v] += flow
            F[v][u] -= flow





