import numpy as np

def load_graph_from_file(f):
    lines = open(f).readlines()
    data = np.array([[float(i) for i in x.split(" ")] for x in lines])
    return Graph(adjMat=data, computePaths = False)


class Graph(object):
    """
    A Graph object. Graphs are represented as a adjacency matrices for now. If
    this gets too unweildy we can change to a sparse matrix representation but
    computations are easier here.
    """
    def __init__(self, adjMat = np.zeros([1, 1])):
        """
        Graphs only keep track of their adjacency matrix.
        """
        self.adjMat = np.matrix(adjMat)
        self.hasNeighDict = False

    def addEdge(self, i, j, val=1):
        """
        Add an edge (i,j) with weight val. Necessarily an undirected graph.
        """
        ## HACK to handle edges with 0 weight, rather just make them really
        ## small, shouldn't mess with results too much
        if val == 0:
            val = 1*10**-20

        self.adjMat[i,j] = val
        self.adjMat[j,i] = val

    def addEdges(self, edges):
        """
        Add a set of edges in (i,j,val) tuples
        """
        edges1 = [(e[0], e[1], 1) for e in edges if len(e) == 2]
        edges2 = [e for e in edges if len(e) == 3]
        edges2.extend(edges1)
        for e in edges2:
            self.addEdge(e[0], e[1], e[2])

    def getEdge(self, i, j):
        """
        Get the weight of an edge (i,j) or 0 if there is no edge present.
        """
        return self.adjMat[i,j]

    def neighbors(self, i):
        """
        Return the list of neighbors of any vertex i in the graph. 
        """
        if self.hasNeighDict:
            return self.neighDict[i]
        else:
            return [x for x in range(self.adjMat.shape[0]) if self.adjMat[i,x] != 0]

    def buildNeighborDict(self):
        """
        Construct a neighborhood dictionary for use in neighbors
        """
        neighDict = {}
        for i in xrange(self.adjMat.shape[0]):
            neighDict[i] = [x for x in xrange(self.adjMat.shape[0]) if self.adjMat[i,x] != 0]
        self.neighDict = neighDict
        self.hasNeighDict = True

    def buildNeighborTransDict(self):
        """
        Construct a dictionary akin to the the neighdict but with only the transistion probabilities
        """
        neighTrans = {}
        for vert, neighs in self.neighDict.iteritems():
            weight_temp = self.adjMat[vert,neighs]
            if np.sum(weight_temp) > 0:
                neighTrans[vert] = weight_temp / np.sum(weight_temp)
            else:
                neighTrans[vert] = None
        self.neighTrans = neighTrans
            
    def numVertices(self):
        """
        number of vertices in the graph.
        """
        return self.adjMat.shape[0]


    def numEdges(self):
        """
        number of edges in the graph.
        """
        return self.adjMat.nnz()/2

    def delEdge(self, i, j):
        """
        Remove an edge
        """
        self.adjMat[i,j] = 0
        self.adjMat[j,i] = 0

    def maxDegree(self):
        """
        Compute the maximum degree in the graph.
        """
        return max([len(self.neighbors(x)) for x in range(self.adjMat.shape[0])])

    def findleaves(self):
        """
        Find all leaves in the graph. A leaf is any object with degree 1.
        """
        n = self.adjMat.shape[0]
        leaves = [x for x in range(n) if len(self.neighbors(x)) == 1]
        return leaves

    def connectedComponents(self):
        """
        Find the connected components and return a list of sets (each set
        corresponds to a connected component).
        """
        n = self.adjMat.shape[0]
        visited = set([])
        comps = []
        queue = set([0])

        while len(visited) < n:
            x = [y for y in range(n) if y not in visited][0]
            queue = [x]
            currcomp = set([])
            while len(queue) != 0:
                x = queue.pop()
                currcomp.add(x)
                visited.add(x)
                neighbors = [y for y in self.neighbors(x) if y not in visited]
                queue.extend(neighbors)
            comps.append(currcomp)
        return comps

    def degree_dist(self):
        """
        Return the empirical degree distribution of this graph.
        """
        return [len(self.neighbors(i)) for i in range(self.numVertices())]

    def save_as_dot(self, file, leaves=None):
        """
        For visualization purposes we can save this graph as a dot file which
        can be opened in GraphViz.
        """
        name = file.split("/")
        name = name[len(name)-1]
        f = open(file + ".dot", "w")
        f.write("graph %s{\n" % name)
        f.write('size="7,6"\n')
        f.write('node[shape=circle,height=0.2,width=0.2,style=filled]\n')
        if leaves == None:
            leaves = self.findleaves()
        for l in leaves:
            f.write("%d [color=red]\n" % l)
        M = self.adjMat.todok()
        for k in M.iterkeys():
            if k[0] < k[1]:
                f.write('"%d" -- "%d"\n' % (k[0], k[1]))
        f.write("}")
        f.close()

    def Dijkstra(self,start = 0):
        n = self.adjMat.shape[0]
        if not self.hasNeighDict:
            self.createNeighborDict()
        neighs = self.neighDict
        add_order = []
        final_dist = np.array([np.inf]*n)
        final_dist[start] = 0.
        rem_ind = range(n)
        for curr_size in xrange(n):
            curr_rem_pos = np.argmin(final_dist[rem_ind])
            curr_pos = rem_ind[curr_rem_pos]
            curr_dist = final_dist[curr_pos]
            add_order.append(curr_pos)
            for eni in neighs[curr_pos]:
                if eni in rem_ind:
                    weighTemp = self.adjMat[curr_pos,eni]
                    final_dist[eni] = min(1./weighTemp + curr_dist,final_dist[eni])
            del rem_ind[curr_rem_pos]
        return (final_dist, add_order)
                                                          
