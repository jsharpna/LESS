import numpy as np
from Tree import Tree

def prims(G):
    """
    Prim's algorithm for computing minimum weight spanning trees.

    NOTE: doesn't give you trees with low degree and probably fails if the graph
    is not connected.
    """
    G = G.adjMat
    n = G.shape[0]
    T = np.zeros((n,n))
    v = set([0])
    ## these are the sets of edges on the boundary
    cands = [(i, 0, G[0,i]) for i in range(1,n) if G[0,i] != 0]
    while len(v) != n or len(cands) != 0:
        (next, prev, dist) = cands[np.argmin([x[2] for x in cands])]
        T[next,prev] = 1
        T[prev,next] = 1
        cands = filter(lambda x: x[0] != next, cands)
        v.add(next)
        cands.extend([(j, next, G[next,j]) for j in range(n) if G[next,j] != 0 and j not in v])
    return Tree(adjMat=T)

def dfs(G):
    """
    DFS will typically give us a much lower degree spanning tree than Prim's
    algorithm. I don't know if there are any guarantees on this though. It is
    certainly not obvious that there are.
    """
    G = G.adjMat
    n = G.shape[0]
    T = np.zeros((n,n))
    visited = set([0])
    def dfs_rec(curr):
        neighbors = [i for i in range(1,n) if G[curr,i] != 0]
        if len(neighbors) == 0:
            return
        for next in neighbors:
            if next not in visited:
                T[curr,next] = G[curr,next]
                T[next,curr] = G[next,curr]
                visited.add(next)
                dfs_rec(next)
    dfs_rec(0)
    return Tree(adjMat=T)

def abTree(g, vert_rest = []):
    """
    the AB (Aldous-Broder) tree requires a graph for which buildNeighTransDict()
    should have already been run should assert that g is connected
    """
    if not g.hasNeighDict:
        g.buildNeighborDict()
        g.buildNeighborTransDict()
    if vert_rest == []:
        n = g.numVertices()
        vert_rest = range(n)
    else:
        n = len(vert_rest)
    comp = g.connectedComponents()
    tree_edges = []
    not_visited = [True]*n
    v = np.where(np.random.multinomial(1,[1./n]*n) == 1)[0][0]
    not_visited[v] = False
    while any(not_visited):
        nv = vert_rest.index(g.neighDict[vert_rest[v]][np.where(np.random.multinomial(1,np.array(g.neighTrans[vert_rest[v]])[0,:]) == 1)[0][0]])
        if not_visited[nv]:
            not_visited[nv] = False
            tree_edges.append([v,nv])
        v = nv
    tree_adj = np.zeros((n,n))
    for ed in tree_edges:
        tree_adj[ed[0],ed[1]] = 1
        tree_adj[ed[1],ed[0]] = 1
    ABTree = Tree(tree_adj)
    return ABTree

