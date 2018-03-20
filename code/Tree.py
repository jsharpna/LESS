import Graph
import numpy as np
import copy

class Tree(Graph.Graph):
    """
    A Tree object. Maintains the invariant that it is a tree. I.e. you cannot
    add a vertex without adding an appropriate edge to connect it to the
    tree. This is a connected graph with out any cycles. Allows for more
    optimized computations of all pairs shortest paths.
    """
    def __init__(self, adjMat = np.zeros([0, 0])):
        """
        Constructor, checks if the tree is connected and ensures that there are
        no cycles.
        """
        Graph.Graph.__init__(self, adjMat)
        self.nodesDict = {}
        assert self.connected() == True, "tree is not connected"
        assert self.cycleCheck() == False, "tree initialized with cycles"


    def addEdge(self, i, j, val=1):
        """
        You cannot add edges to a tree without also adding vertices.
        """
        raise Exception('addEdge is an invalid operation on Trees')

    def addEdges(self, edges):
        """
        You cannot add edges to a tree without also adding vertices.
        """
        raise Exception('addEdges is an invalid operation on Trees')

    def addVertex(self, connections = {}):
        """
        Add a vertex to the tree, you can use this to add a leaf node, so
        len(connections) must be 1.
        """
        assert len(connections) == 1, "Attempt to add node with either too many or too few connections"
        node = Graph.Graph.addVertex(self, connections)
        return node
        
    def addVertexBetween(self, n1, n2, d1):
        """
        Add a vertex on an existing edge between n1 and n2. d1 is the distance
        between n1 and the new node.
        """
        assert self.adjMat.tocsr()[n1, n2] != 0, 'Attempt to insert node where no edge exists'
        d2 = self.getEdge(n1, n2) - d1
        Graph.Graph.delEdge(self, n1,n2)
        n3 = Graph.Graph.addVertex(self)
        Graph.Graph.addEdge(self, n1, n3, d1)
        Graph.Graph.addEdge(self, n2, n3, d2)

        return n3

    def findLeavesBelow(self, root, parent):
        """
        Find the set of leaves below root in the tree rooted with edges away
        from parent. Returns a list.
        """
        visited = set()
        if parent != None:
            visited.add(parent)
        if len(self.neighbors(root)) == 0:
            ## this is a degenerate case
            return [root]
        queue = [root]
        leaves = []
        while len(queue) > 0:
            i = queue.pop()
            visited.add(i)
            neighbors = self.neighbors(i)
            neighbors = np.setdiff1d(neighbors, visited)
            if len(neighbors) == 0:
                leaves.append(i)
            else:
                neighbors = np.setdiff1d(neighbors, visited)
                queue.extend(neighbors)
        return leaves

    def findNodesBelow(self, root, parent):
        """
        Find the set of nodes below root in the tree rooted with edges away from
        parent. returns a set.
        """
        visited = set()
        toret = set([root])
        if parent != None:
            visited.add(parent)
        if len(self.neighbors(root)) == 0:
            return toret
        queue = [root]
        while len(queue) > 0:
            i = queue.pop()
            visited.add(i)
            neighbors = self.neighbors(i)
            neighbors = np.setdiff1d(neighbors, visited)
            [toret.add(x) for x in neighbors]
            queue.extend(neighbors)
        return toret

    def cycleCheck(self):
        """
        Check if this tree has any cycles. 
        """
        visited = []
        self.found = False
        def dfs(start, prev):
            if self.found == True:
                return
            visited.append(start)
            cands = [x for x in self.neighbors(start) if x != prev]
            cycles = np.intersect1d(cands, visited)
            if len(cycles) > 0:
                self.found = True
            cands = [c for c in cands if c not in visited]
            for c in cands:
                dfs(c, start)
        dfs(0, None)
        return self.found
    
    def connected(self):
        """
        Ensure that this tree is connected. i.e. it has only one connected
        component.
        """
        return (len(Graph.Graph.connectedComponents(self)) == 1)

    def getRoot(self):
        """
        Get a Edge Pearl Point in the tree. 
        """
        v0 = self.findleaves()[0]
        nodes = range(self.numVertices())
        s = float(len(nodes))
        k = self.maxDegree()
        if k == 1:
            v1 = self.findleaves()[1]
            return (v0, v1)
        queue = [(v0, None, self.numVertices())]
        while len(queue) > 0:
            (pointer,parent,d) = queue.pop()
            if (s/(k)-1.0/(k-1) <= d and d <= s*(k-1)/(k)):
                return (pointer, parent)
            children = [x for x in self.neighbors(pointer) if x != parent]
            des_children = [len(self.findNodesBelow(x, pointer)) for x in children]
            idx = np.argmax(des_children)
            queue.append((children[idx], pointer, des_children[idx]))

    def getRoot2(self):
        """
        Get a vertex-pearl point in the tree using the FindBalance algorithm.
        """
        v0 = self.findleaves()[0]
        nodes = range(self.numVertices())
        s = float(len(nodes))
        if self.maxDegree() == 1:
            v1 = self.findleaves()[1]
            return [set([v0]), set([v1])]
        queue = [(v0, None, s)]
        while len(queue) > 0:
            (pointer, parent, obj) = queue.pop()
            children = [x for x in self.neighbors(pointer)]
            des_children = [len(self.getNodesBelow(x, pointer)) for x in children]
            if np.max(des_children) >= obj:
                children = [x for x in self.neighbors(parent)]
                des_children = [self.getNodesBelow(x, parent) for x in children]
                lengths = [len(x) for x in des_children]
                idx = np.argmin(lengths)
                des_children[idx].add(parent)
                return des_children
            idx = np.argmax(des_children)
            queue.append((children[idx], pointer, des_children[idx]))
        
    def getNodesBelow(self, curr, parent):
        """
        This is a subroutine used by the root-selection procedure. We have to
        record the elements in each subtree, and this is a memoized way of doing
        that. Calling this returns the set of nodes in the subtree rooted at
        curr (including curr) if all the edges are oriented away from parent.
        """
        if curr not in self.nodesDict.keys():
            self.nodesDict[curr] = {}
        if parent in self.nodesDict[curr].keys():
            return self.nodesDict[curr][parent]
        neighbors = [x for x in self.neighbors(curr) if x != parent]
        sol = set([])
        for x in neighbors:
            sol |= self.getNodesBelow(x, curr)
        sol |= set([curr])
        self.nodesDict[curr][parent] = sol
        return self.nodesDict[curr][parent]

            
                
