import numpy as np
import random

def planted_cluster(n, k, p, q):
    """
    Make a graph on n vertices with a planted cluster of size k. Within cluster
    edge probability is p and ambient edge connectivity is q (both between
    cluster and for anything not in the cluster. Return the adjacency matrix and
    the cut size. 
    """
    M = np.matrix(np.random.binomial(1, q, (n,n)))
    M[np.ix_(range(k), range(k))] = np.matrix(np.random.binomial(1, p, (k,k)))
    M = np.triu(M, k=1) + np.triu(M, k=1).T

    cutsize = np.sum([1 if M[i,j] == 1 else 0 for i in range(k) for j in range(k, n)])
    return (M, cutsize)

def balanced_binary(n):
    """
    Generates a balanced binary tree with n leaves. Returns an unweighted,
    undirected adjacency matrix. Note that n should be 3*2^k for some integer k.
    """
    assert np.log2(float(n)/3) == np.ceil(np.log2(float(n)/3)), "n is not of the form 3*2^k for some k"
    tree = np.zeros([2*n-2, 2*n-2])
    tree[0, 1:4] = [1, 1, 1]
    tree[1:4, 0] = [1, 1, 1]
    for i in xrange(1, n-2):
        tree[i, 2*(i+1)] = 1
        tree[i, 2*(i+1)+1] = 1
        tree[2*(i+1), i] = 1
        tree[2*(i+1)+1, i] = 1
    return tree

def subtree_signal(n,d,mu,sigma):
    """
    Pick a subtree (at depth d < log n) and activate all of the nodes in that
    subtree. This is for the balanced_binary tree (i.e. the unrooted one).
    """
    assert d <= np.log2(n/3+1), "d is too big (choosing a subtree that is deeper than the tree)"
    maxind = 2*n-2
    curr = 0
    while d > 0:
        curr = random.sample([2*(curr+1), 2*(curr+1)+1], 1)[0]
        d -= 1
    coords = []
    queue = [curr]
    while len(queue)>0:
        next = queue.pop()
        coords.append(next)
        if 2*(next+1) < maxind:
            queue.append(2*(next+1))
            queue.append(2*(next+1)+1)
    return (fixed_signal(2*n-2, coords, mu, sigma), 1)

def balanced_rooted_binary(n):
    """
    Generates a rooted balanced binary tree with n leaves. Returns an unweighted
    undirected adjacency matrix. n should be 2^k for some integet k.
    """
    assert np.log2(float(n)) == np.ceil(np.log2(float(n))), "n is not of the form 2^k"
    tree = np.zeros([2*n-2, 2*n-2])
    tree[0,1] = 1
    tree[1,0] = 1
    for i in xrange(0, n-2):
        tree[i, 2*i+2] = 1
        tree[i, 2*i+3] = 1
        tree[2*i+2, i] = 1
        tree[2*i+3, i] = 1
    return tree

def rooted_subtree_signal(n,d,mu,sigma):
    """
    For the rooted binary tree, this generates a signal by activating a subtree
    at depth d (i.e. d=0 activates one half of the tree). d must be less than
    log2(n+1). Returns the signal and the cut size (Which is always 1 here).
    """
    assert d <= np.log2(n+1), "d is too big"
    maxind = 2*n-2
    curr = random.sample([0,1],1)[0]
    while d > 0:
        curr = random.sample([2*curr+2, 2*curr+3], 1)[0]
        d -= 1
    coords = []
    queue = [curr]
    while len(queue)>0:
        next = queue.pop()
        coords.append(next)
        if 2*next+2 < maxind:
            queue.append(2*next+2)
            queue.append(2*next+3)
    return (fixed_signal(2*n-2, coords, mu, sigma), 1)

def erdos_renyi(n,p):
    """
    Erdos-Renyi random graph.
    """
    M = np.matrix(np.random.binomial(1, p, (n,n)))
    M = np.triu(M, k=1) + np.triu(M, k=1).T
    return M

def random_signal(G, k, mu, sigma):
    """
    Generate a random signal by activating k nodes in the graph. Compute the cut
    size of the activation and return the tuple of (signal, cut size)
    """
    n = G.shape[0]
    S = random.sample(range(n), k)
    if sigma != 0:
        v = np.random.normal(0, sigma, (n, 1))
    else:
        v = np.zeros((n,1))
    v[S,0] += mu
    cut = np.sum(G[np.ix_(S,[x for x in range(n) if x not in S])])
    return (v, cut)

def fixed_signal(n, coords, mu, sigma, normalized=True):
    """
    Construct a signal by activating coords (so the total energy is mu) and
    corrupting everything with gaussian noise with variance sigma. Coords should
    be a list of indices.
    """
    if sigma != 0:
        v = np.random.normal(0, sigma, (n, 1))
    else:
        v = np.zeros((n,1))
    if len(coords) > 0:
        if normalized:
            v[coords,0] += float(mu)/np.sqrt(len(coords))
        else:
            v[coords,0] += float(mu)
    return v
    
def square_signal(n1, k, mu, sigma, normalized=True):
    """
    For the torus graph, pick a k x k square and activate that square. Returns
    the pattern and the cut size (which is always 4k).
    """
    ## pick a random top left corner and then extend in each direction by k
    i = random.sample(range(n1), 1)[0]
    j = random.sample(range(n1), 1)[0]
    xinds = [np.mod(i+x, n1) for x in range(k)]
    yinds = [np.mod(j+x, n1) for x in range(k)]
    coords = []
    for x in xinds:
        for y in yinds:
            coords.append(x*n1+y)
    return fixed_signal(n1**2, coords, mu, sigma, normalized=normalized)
    
def hypercube_signal(n1, k, d, mu, sigma):
    """
    For the d-dimensional torus graph, pick a k^d hypercube and activate that
    square. Returns the pattern.
    """
    def vec_to_index(vec, n1, d):
        curr = 0
        multiplier = 1
        for i in range(d):
            curr += vec[i]*multiplier
            multiplier *= n1
        return curr

    def recursive_enumerate(vec, j, k, n1):
        if j == len(vec)-1:
            oldval = vec[j]
            for i in range(k):
                vec[j] = (vec[j]+1) % n1
                inds.append(vec_to_index(vec, n1, len(vec)))
            vec[j] = oldval
            return
        oldval = vec[j]
        for i in range(k):
            vec[j] = (vec[j]+1) % n1
            recursive_enumerate(vec, j+1, k, n1)
        vec[j] = oldval
    topleft = [random.randint(0, n1-1) for i in range(d)]
    inds = []
    recursive_enumerate(topleft, 0, k, n1)
    return fixed_signal(n1**d, inds, mu, sigma)

def createW(edgeOr,n):
    """
    converts an oriented edge list into a adjacency matrix
    """
    W = np.zeros((n,n))
    m = len(edgeOr)
    for eInd in range(m):
        eh = edgeOr[eInd][0]
        et = edgeOr[eInd][1]
        W[et,eh] += 1.
        W[eh,et] += 1.
    return W
        
def grid2D(n1):
    """
    2d
    """
    n = n1**2
    edgeOr = []
    for i in range(n1-1):
        for j in range(n1 -1):
            edgeOr += [[(i+1)*n1 + j,i*n1+j]]
            edgeOr += [[i*n1 + j + 1,i*n1+j]]
    for i in range(n1-1):
        edgeOr += [[(n1-1)*n1 + i, (n1-1)*n1 + i + 1]]
        edgeOr += [[(i+1)*n1 -1, (i+2)*n1 -1]]
    return createW(edgeOr,n)

def torus2d(n1):
    n = n1**2
    M = np.matrix(np.zeros((n,n)))
    for i in range(n1):
        for j in range(n1):
            M[n1*i+j, n1*i+(np.mod(j+1,n1))] = 1
            M[n1*i+(np.mod(j+1,n1)), n1*i+j] = 1
            M[n1*i+j, n1*i+(np.mod(j-1, n1))] = 1
            M[n1*i+(np.mod(j-1, n1)), n1*i+j] = 1

            M[n1*i+j, n1*(np.mod(i+1, n1))+j] = 1
            M[n1*(np.mod(i+1, n1))+j, n1*i+j] = 1
            M[n1*i+j, n1*(np.mod(i-1, n1))+j] = 1
            M[n1*(np.mod(i-1, n1))+j, n1*i+j] = 1
    return M

def linegraph(n):
    """
    Line graph on n vertices. For nice results make n=2**k for some k
    """
    M = np.matrix(np.zeros((n,n)))
    for i in range(n-1):
        M[i,i+1] = 1
        M[i+1,i] = 1
    return M

def line_connected_signal(n, k, mu, sigma):
    i = random.sample(range(0, n-k+1), 1)[0]
    coords = range(i, i+k)
    cut = 2
    if i == 0 or i == n-k:
        cut = 1
    return (fixed_signal(n, coords, mu, sigma), cut)

def makeComplete(n):
    edgeOr = []
    for i in range(n-1):
        for j in range(i+1,n):
            edgeOr.append([i,j])
    return createW(edgeOr,n)

def make_kron(lap_base, l):
    lap = lap_base.copy()
    n = 1.*lap_base.shape[0]
    for i in range(l-1):
        lap = np.kron(lap_base,np.eye(n**(i+1))) + np.kron(np.eye(n),lap)
    return lap

def printGrid2D(Y,n1):
    for i in range(n1):
        print ' '.join(['%.2f'%y for y in Y[i*n1:(i+1)*n1]])


def incidence_matrix(M):
    n = M.shape[0]
    edges = []
    for i in range(0,n):
        for j in range(i+1, n):
            if M[i,j] != 0:
                edges.append((i,j))
    E = np.matrix(np.zeros((len(edges), n)))
    for i in range(len(edges)):
        E[i, edges[i][0]] = 1
        E[i, edges[i][1]] = -1
    return E


def intel_lab_graph():
    f = open("intel_lab.txt").readlines()
    n = len(f)
    M = np.matrix(np.zeros((n,n)))
    coords = []
    for i in f:
        coords.append((float(i.split(" ")[1]), float(i.split(" ")[2])))
    for i in range(n):
        for j in range(i,n):
            if np.sqrt((coords[i][0] - coords[j][0])**2 + (coords[i][1]-coords[j][1])**2) < 7:
                M[i,j] = 1
                M[j,i] = 1
    return M


def reparse_intel_data():
    for epoch in range(100):
        f = open("shrunk.txt")
        measurements = ["0" for i in range(54)]
        for line in f:
            if int(line.split(" ")[2]) == epoch and line.split(" ")[3] != '' and line.split(" ")[4] != '' and int(line.split(" ")[3]) <= 54:
                print "%d %s" % (int(line.split(" ")[3]), line.split(" ")[4])
                measurements[int(line.split(" ")[3])-1] = line.split(" ")[4]
        x = open("intel_data/%d.txt" % (epoch), "w")
        x.write(" ".join(measurements))
        x.close()

def shrink_intel_data():
    f = open("data.txt")
    f2 = open("shrunk.txt", "w")
    for line in f:
        if int(line.split(" ")[2]) <= 100:
            f2.write(line)
    f2.close()
