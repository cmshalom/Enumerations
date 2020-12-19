from bisect import *
from combin import *
from util import *

import networkx as nx
import matplotlib.pyplot as plt

class Tree:
    class Color:
        GRAY = 0
        BLUE = 1
        RED = 2
        YELLOW = 3
        def __init__(self, name, minVertexWeight, maxVertexWeight, minTreeWeight, childrenColor):
            self.name = name
            self.minVertexWeight = minVertexWeight
            self.maxVertexWeight = maxVertexWeight
            self.minTreeWeight = minTreeWeight
            self.childrenColor = childrenColor

    colors = [Color("gray", 1, 1, 1, Color.GRAY),
              Color("blue", 1, sys.maxsize, 1, Color.BLUE),
              Color("red", 0, sys.maxsize, 1, Color.YELLOW),
              Color("yellow", 1, 1, 2, Color.RED)
              ]

    def __init__(self, children=None, weight=1, color=Color.GRAY):
        assert weight >= 0, "weight = %d should be non-negative" % weight
        self.weight = weight

        if type(color) == int:
            assert color <= len(Tree.colors), "Color %d exceeds %d" % (color, len(Tree.colors))
            color = Tree.colors[color]
        self.color = color

        if children is None:
            self.children = []
            self.leaves = 1
        else:
            self.children = children
            self.leaves = sum([child.leaves for child in children])
        self.vertices = 1 + sum([child.vertices for child in self.children])

    def addTree(self, tree):
        '''
        Makes the root of tree a child of the root of the current tree.
        The addition will preserve the order of the subtrees which is from the largest tree to the smallest.
        '''
        # TODO Can use binary search to improve
        self.vertices += tree.vertices
        for i in range(len(self.children)):
            if tree.vertices >= self.children[i].vertices:
                self.children.insert(i,tree)
                return
        self.children.append(tree)

    def nodesDFS(self):
        yield self
        for child in self.children:
            for node in nodesDFS(child):
                yield node

    def __str__(self):
        '''
        :return: a string with the number of vertices of the tree and every subtree
        (not recursive)
        '''
        result = str(self.vertices + "w=" + self.weight)
        if len(self.children) > 0:
            result += "->"
            for child in self.children:
                result += " " + str(child.vertices)
                if child.weight != 1:
                    result += "w=" + str(child.weight)
        return result

    def _setOffset(self, offset):
        '''
        Recursively sets all the offsets (i.e. index of leftmost leaf in subtree) for every vertex of the tree
        :param offset: the index of the leftmost leaf of this tree
        '''
        self.offset = offset
        for child in self.children:
            child._setOffset(offset)
            offset+=child.leaves

    @staticmethod
    def _levelString(trees):
        '''
        :param trees: is a non-empty list of tree objects
        :return: a pair (str, nextLevel) where str is the string representation of the roots of trees and
                 nextLevel is the list of roots of the next level
        '''
        str = ""
        nextLevel = []
        lastOffset = 0
        for tree in trees:
            str += " " * (tree.offset - lastOffset)
            str += "O"
            lastOffset = tree.offset
            nextLevel += tree.children
        return (str,nextLevel)

    def __repr__(self):
        self._setOffset(0)  # Recursively sets all the offsets of all subtrees
        level = [self]
        lines = []
        while (len(level) != 0):
            (str,level) = Tree._levelString(level)
            lines.append(str)
        str = ""
        for line in lines:
            str += line + "\n"
        return str

    def plot(self, isDirected=False):
        g = nx.DiGraph() if isDirected else nx.Graph()
        vertexColors = []
        vertexLabels = {}
        self._plot(g, vertexColors, vertexLabels)
        nx.draw(g, with_labels=True, labels=vertexLabels, font_weight='bold', node_color=vertexColors)

    def _plot(self, g, colors, labels, parent=None):
        '''
        :param g: A new vertex and an edge to parent (if any) is added to the graph g
        :param colors: The color of the vertex is added to to this list
        :param labels:  A label encoding the number and weight of the vertex is added to this dictionnary.
        '''
        root = g.number_of_nodes()
        g.add_node(root)
        colors.append(self.color.name)
        labels[root] = "%d" % (self.weight)
        if parent is not None:
            g.add_edge(parent,root)
        for child in self.children:
            child._plot(g, colors, labels, root)

    def show(self, isDirected=False):
        '''
        Exhibits self depending on the global boolean variables textOutput, graphicsToScreen and GraphicsToFile
        '''
        if textOutput:
            print("Tree", end="")
            if ("rank" in self.__dict__):
                print (" #", self.rank, end="")
            print (" (%d vertices) :" % self.vertices)
            print(self.__repr__())
        if graphicsToScreen or graphicsToFile:
            plt.clf()
            self.plot(isDirected=isDirected)
            if graphicsToFile:
                assert "rank" in self.__dict__
                plt.savefig("Tree-%d-%d.png" % (self.vertices, self.rank))
            if graphicsToScreen:
                plt.show()

###########################################################################################
#         GENERATORS
###########################################################################################

def rootedTrees(w, color, maxSubtreeWeight=None):
    if type(color) == int:
        color = Tree.colors[color]
    if maxSubtreeWeight is None or maxSubtreeWeight > w:
        maxSubtreeWeight = w
    if w < color.minTreeWeight:
        return

    maxRootWeight = min(color.maxVertexWeight, w)
    for rootWeight in range(color.minVertexWeight, maxRootWeight+1):
        for f in forestsnm_leq(w - rootWeight, maxSubtreeWeight, color.childrenColor):
            yield Tree(f, color=color, weight=rootWeight)

def forests(w, color):
    return forestsnm_leq(w, w,color)

def forestsnm_leq(w,m,color):
    if w==0:
        yield []
        return

    if type(color) == int:
        color = Tree.colors[color]
    if m < color.minTreeWeight:
        return

    for mm in range (color.minTreeWeight,m+1):
        for mu in range (1, w // mm + 1):
            for forest in forestsnmmu(w,mm,mu,color):
                yield forest
    return

def forestsnmmu(w,m,mu,color):
    assert type(color) == Tree.Color

    def f():
        return rootedTrees(m,color)

    for set in multisets(f, mu):
        for forest in forestsnm_leq(w-m*mu, min(w-m*mu,m-1), color):
            yield set + forest


def _freeTrees(n, color):
    if type(color) == int:
        color = Tree.colors[color]

    # Generate monocentroidal trees
    for t in rootedTrees(n,color, maxSubtreeWeight=(n-1)//2):
            yield t

    # Generate bicentroidal trees
    if n % 2 == 0:
        for pair in multisets(lambda : rootedTrees(n//2, color),2):
            yield Tree([pair[0]]+pair[1].children, color=color, weight=pair[1].weight)

def freeTrees(n):
    return _freeTrees(n, Tree.Color.GRAY)

def weightedFreeTrees(n):
    return _freeTrees(n, Tree.Color.BLUE)

def demoEnumeration(L,H, f, formatString=None, isDirected=False):
    for n in range(L,H+1):
        if formatString is not None:
            print (formatString % n)
        rank = 0
        for t in f(n):
            t.rank = rank
            rank = rank+1
            t.show(isDirected=isDirected)

def blockTrees(n):
    # Generate monocentroidal trees
    for t in rootedTrees(n, Tree.Color.RED, maxSubtreeWeight=(n-1)//2):
            yield t
    for t in rootedTrees(n, Tree.Color.YELLOW, maxSubtreeWeight=(n-1)//2):
            yield t

    if n % 2 == 0:
        # Generate bicentroidal trees
        for redTree in rootedTrees(n//2, Tree.Color.RED, maxSubtreeWeight=n // 2 - 1):
            for yellowTree in rootedTrees(n // 2, Tree.Color.YELLOW):
                yield Tree([redTree]+yellowTree.children, color=Tree.Color.YELLOW, weight=yellowTree.weight)
        # Generate trees with tree centroids trees
        for pair in multisets(lambda : rootedTrees(n//2, Tree.Color.YELLOW),2):
            yield Tree(pair, color=Tree.Color.RED, weight=0)

def demoRootedTreesEnumeration(L,H):
    demoEnumeration(L,H, lambda n : rootedTrees(n, Tree.Color.GRAY), formatString="ROOTED TREES ON %d VERTICES", isDirected=True)

def demoWeightedRootedTreesEnumeration(L,H):
    demoEnumeration(L,H, lambda n : rootedTrees(n, Tree.Color.BLUE), formatString="ROOTED TREES OF WEIGHT %d", isDirected=True)

def demoFreeTreesEnumeration(L,H):
    demoEnumeration(L,H, freeTrees, formatString="FREE TREES ON %d VERTICES", isDirected=False)

def demoWeightedFreeTreesEnumeration(L,H):
    demoEnumeration(L,H, weightedFreeTrees, formatString="FREE TREES OF WEIGHT %d", isDirected=False)

def demoBlockTreesEnumeration(L,H):
    demoEnumeration(L,H, blockTrees, formatString="BLOCK TREES OF GRAPHS ON %d VERTICES", isDirected=False)

###########################################################################################
#         RECURRENCES
###########################################################################################

def _computeNumberOfRootedTreesWilf(N):
    # Time complexity = \sum i log i = O(n^2 log n)
    if "wtn" not in globals():
        global wtn
        wtn = [None, 1]   #t(1)=1, t(0) undefined
    for n in range(len(wtn), N + 1):
        t = 0
        # Time complexity n + n/2 + n/3 + n/4..  = O(n log n)
        for j in range(1, n):
            # The running time of this iteration is O(n/j)
            t += sum ([d * wtn[d] * wtn[n - d * j] for d in range(1, (n - 1) // j + 1)])
        wtn.append (t // (n - 1))

def numberOfRootedTreesWilf(n):
    '''
    :return: the number of rooted trees of n vertices according the Wilfian recurrence relation
    Upon exit the class variable tn.contains the number of trees on i vertices for every i <= n
    Existing values in tn upn entry, if any, are considered as correct
    '''
    if "wtn" not in globals() or len(wtn) <= n:
        _computeNumberOfRootedTreesWilf(n)
    return wtn[n]

def _computeNumberOfForests(N):
    # To be consistent with the paper, we work with one based lists.
    # So, we prepend a dummy zero'th entry to every list
    if "fn" not in globals():
        global fn, fnm, fnm_leq, fnmmu, fnmmu_leq
        fn = [1]             #1 dimensional (n)   f(0)=1,
        fnm = [[1]]          #2 dimensional (n,m) f(0,0) = 1
        fnm_leq = [[1]]      #2 dimensional (n,m) f^<=(0,0) = 1
        fnmmu = [[[1]]]      #3 dimensional (n,m,mu) f(0,0,0)=1
        fnmmu_leq = [[[1]]]  #3 dimensional (n,m,mu) f^<=(0,0,0)=1
    for n in range(len(fn),N+1):
        fnm_row = [0]             #1 dimensional
        fnmmu_plane = [[0]]       #2 dimensional
        fnmmu_leq_plane = [[0]]   #2 dimensional
        for m in range(1,n+1):
           fnmmu_row = [0] + [cc(fn[m-1], mu) * fnm_leq[n - m*mu][min(n - m * mu,m - 1)] for mu in range(1, n // m + 1)]
           fnmmu_plane.append(fnmmu_row)
           fnmmu_leq_plane.append(partialSums(fnmmu_row))
           fnm_row.append(sum(fnmmu_row))
        # Append all results for n to the main lists
        fnmmu.append(fnmmu_plane)
        fnmmu_leq.append(fnmmu_leq_plane)
        fnm.append(fnm_row)
        fnm_leq.append(partialSums(fnm_row))
        fn.append(fnm_leq[n][n])

def numberOfForests(n):
    assert n >= 0, "n=%d should be non-negative" % n
    if "fn" not in globals() or len(fn) <= n:
        _computeNumberOfForests(n)
    return fn[n]

def numberOfRootedTrees(n):
    assert n > 0, "n = %d should be positive" % n
    return numberOfForests(n-1)

def demoRecurrences(N):
    print ()
    print ("Numbers of unlabeled rooted trees computed using Wilf's formula")
    numberOfRootedTreesWilf(N)
    print("wtn=", wtn)

    print ()
    numberOfRootedTrees(N)
    print ("Numbers of unlabeled rooted forests computed from the breakdown")
    print("fn=", fn)

###########################################################################################
#         UNRANKING
###########################################################################################

def rootedTree(n,i):
    assert n > 0, "n=%d should be positive" % n
    assert i >= 0, "i=%d should be non-negative" % i
    assert i < numberOfRootedTrees(n), "i=%d should be at most the number of trees %d on %d vertices" % (i,numberOfRootedTrees(n),n)
    return Tree() if n == 1 else Tree(forest(n-1,i))

def forest(n,i):
    assert n >= 0, "n=%d should be non-negative" % n
    assert i >= 0, "i=%d should be non-negative" % i
    if i >= numberOfForests(n):
        print("i=", i,  "should be less than the number of forests", numberOfForests(n),  "on", n, "vertices")

    if n==0:
        return []
    m = bisect_left(fnm_leq[n], i+1)   #find the smallest index m such that f<=(n,m) > i
    i -= fnm_leq[n][m - 1]             #this is the index of the required tree in F(n,m)

    mu = bisect_left(fnmmu_leq[n][m], i+1) #find the smallest index mu such that f<=(n,m,mu) > i
    i -= fnmmu_leq[n][m][mu - 1]       #this is the index of the required tree in F(n,m, mu)
    return forestnmmu(n, m, mu, i)

def forestnmmu(n,m,mu,i):
    '''
    :return: The i-th forest F on n vertices with m(F)=m and mu(F)=mu
    '''
    numberOfSmallForests = fnm_leq[n - m * mu] [min(n - m * mu, m - 1)]
    (i1,i2) = divmod (i, numberOfSmallForests)
    bigChildren = [rootedTree(m, treeIndex) for treeIndex in multiset(fn[m-1], mu, i1)]
    smallChildren = forest (n - m  * mu, i2) # the choice of i2 and the order of the trees guarantees that m(F) <= m-1 for the returned forest F
    return bigChildren + smallChildren

def numberOfFreeTrees(n):
    assert n > 0, "n = %d should be positive" % n
    return _numberOfMonocentroidalTrees(n) + _numberOfBicentroidalTrees(n)

def _numberOfBicentroidalTrees(n):
    return cc(fn[n // 2 - 1], 2) if n % 2 == 0 else 0

def _numberOfMonocentroidalTrees(n):
    return fnm_leq[n-1] [(n-1) // 2]

def freeTree(n,i):
    if n % 2 == 0 and i >= _numberOfMonocentroidalTrees(n):
        i -= _numberOfMonocentroidalTrees(n)
        pair = [ rootedTree(n//2, treeIndex) for treeIndex in multiset(fn[n // 2 - 1],2,i) ]
        pair[0].addTree(pair[1])
        return pair[0]
    else:
        return rootedTree(n,i)  # The order of the trees and the value of i guarantee that there will be no big subtrees

def demoUnranking(L,H, numberFunction, unrankingFuntion, formatString=None, isDirected=False):
    for n in range(L,H+1):
        print (formatString % n)
        for i in range(numberFunction(n)):
            print ("Tree ", i, ":")
            t = unrankingFuntion(n,i)
            t.rank = i
            t.show(isDirected=isDirected)

def demoRootedTreesUnranking(L,H):
    demoUnranking(L,H, numberOfRootedTrees, rootedTree, formatString="ROOTED TREES ON %d VERTICES", isDirected=True)

def demoFreeTreesUnranking(L, H):
    demoUnranking(L,H, numberOfFreeTrees, freeTree, formatString="FREE TREES ON %d VERTICES", isDirected=False)

#######################################################################################
#                   MAIN
#######################################################################################
def inputBoolean(str):
    return input(str + "(Y/N)").upper().startswith("Y")

if __name__ == "__main__":
    Low = int(input("Smallest Tree Size ?"))
    High = int(input("Largest Tree Size ?"))
    textOutput = inputBoolean("Text Output ?")
    graphicsToScreen = inputBoolean("Graphics to Screen")
    graphicsToFile = inputBoolean("Graphics to File ?")
#    demoRecurrences(High)
#    demoRootedTreesUnranking(Low,High)
#    demoFreeTreesUnranking(Low,High)
#    demoRootedTreesEnumeration(Low,High)
#    demoWeightedRootedTreesEnumeration(Low,High)
#    demoFreeTreesEnumeration(Low,High)
#    demoWeightedFreeTreesEnumeration(Low,High)
    demoBlockTreesEnumeration(Low,High)