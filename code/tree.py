from bisect import *
from combin import *
from util import *

import networkx as nx
import matplotlib.pyplot as plt

class Tree:
    class Color:
        GRAY = 0
        BLUE = 1
        YELLOW = 2
        RED = 3
        def __init__(self, name, minVertexWeight, maxVertexWeight, minTreeWeight, childrenColor):
            self.name = name
            self.minVertexWeight = minVertexWeight
            self.maxVertexWeight = maxVertexWeight
            self.minTreeWeight = minTreeWeight
            self.childrenColor = childrenColor

    colors = [Color("gray", 1, 1, 1, Color.GRAY),
              Color("blue", 1, sys.maxsize, 1, Color.BLUE),
              Color("yellow", 1, 1, 2, Color.RED),
              Color("red", 0, sys.maxsize, 1, Color.YELLOW)
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
        self.totalWeight = self.weight + sum([child.totalWeight for child in self.children])

    def addTree(self, tree):
        '''
        Makes the root of tree a child of the root of the current tree.
        The addition will preserve the order of the subtrees which is from the largest tree to the smallest.
        '''
        # TODO Can use binary search to improve
        self.vertices += tree.vertices
        self.totalWeight += tree.totalWeight
        for i in range(len(self.children)):
            if tree.totalWeight >= self.children[i].totalWeight:
                self.children.insert(i,tree)
                return
        self.children.append(tree)

    def nodesDFS(self):
        yield self
        for child in self.children:
            for node in nodesDFS(child):
                yield node

    def getName(self):
        name = "Weight:" + str(self.totalWeight)
        if "rank" in self.__dict__:
            name += "-" + str(self.rank)
        return name

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
        nx.draw_planar(g, with_labels=True, labels=vertexLabels, font_weight='bold', font_size=20,
                       node_color=vertexColors,node_size=700)

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

    def show(self, isDirected=False, textOutput=True, graphicsToScreen=False, graphicsToFile=False):
        '''
        Exhibits self depending on the global boolean variables textOutput, graphicsToScreen and GraphicsToFile
        '''
        name = self.getName()
        if textOutput:
            print(name)
            print(self.__repr__())
        if graphicsToScreen or graphicsToFile:
            plt.clf()
            if graphicsToScreen:
                plt.title(name)
            self.plot(isDirected=isDirected)
            if graphicsToFile:
                plt.savefig(name)
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

def freeMonochromaticTrees(n, color):
    if type(color) == int:
        color = Tree.colors[color]

    # Generate monocentroidal trees
    for t in rootedTrees(n,color, maxSubtreeWeight=(n-1)//2):
            yield t

    # Generate bicentroidal trees
    if n % 2 == 0:
        for pair in multisets(lambda : rootedTrees(n//2, color),2):
            yield Tree([pair[0]]+pair[1].children, color=color, weight=pair[1].weight)

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
        # Generate tricentroidal trees
        for pair in multisets(lambda : rootedTrees(n//2, Tree.Color.YELLOW),2):
            yield Tree(pair, color=Tree.Color.RED, weight=0)

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

def _computeNumbersForWC(w,c):
    color = Tree.colors[c]
    rtwcm_leq1 = [0]     # 1 dimensional
    fwcm1 = [0]          # 1 dimensional
    fwcmmu1 = [[0]]      # 2 dimensional
    fwcmmu_leq1 = [[0]]  # 2 dimensional
    for m in range(1, w + 1):  # The results of this loop is used in the net loop (see "dirty trick" below)
        rtwcm_leq1.append(0 if w < color.minTreeWeight else sum(
            [fwcm_leq[w - r][color.childrenColor][min(m, w - r)] for r in
             range(color.minVertexWeight, min(color.maxVertexWeight, w) + 1)]))
    for m in range(1, w + 1):
        if m < color.minTreeWeight:
            temp = [0] * (w // m + 1)
        else:
            rtmc = rtwcm_leq1[-1] if m == w else rtwc[m][c]  # this dirty trick is needed, since the main arrays are not yet updated
            temp = [0] + [cc(rtmc, mu) * fwcm_leq[w - m * mu][c][min(w - m * mu, m - 1)] for mu in
                          range(1, w // m + 1)]
        fwcmmu1.append(temp)
        temp = partialSums(temp);
        fwcmmu_leq1.append(temp)
        fwcm1.append(temp[-1])
    return (rtwcm_leq1[-1], rtwcm_leq1, sum(fwcm1), fwcm1, partialSums(fwcm1), fwcmmu1, fwcmmu_leq1)

def _computeNumberOfForests(W):
    # To be consistent with the paper, we work with one based lists.
    # So, we prepend a dummy zero'th entry to every list
    numberOfColors = len(Tree.colors)
    if "rtwc" not in globals():
        global rtwc, rtwcm_leq, fwc, fwcm, fwcm_leq, fwcmmu, fwcmmu_leq
        rtwc = [[0]*numberOfColors]             #2 dimensional (w,c)   rt(0,c)=0
        rtwcm_leq = [[[0]]*numberOfColors]      #3 dimensional (w,c,m) rt<=(0,c,0)=0
        fwc = [[1]*numberOfColors]              #2 dimensional (w,c)   f(0,c)=1,
        fwcm = [[[1]]*numberOfColors]           #3 dimensional (w,c,m) f(0,c,0) = 1
        fwcm_leq = [[[1]]*numberOfColors]       #3 dimensional (w,c,m) f^<=(0,c,0) = 1
        fwcmmu = [[[[1]]]*numberOfColors]       #4 dimensional (w,c,m,mu) f(0,c,0,0)=1
        fwcmmu_leq = [[[[1]]]*numberOfColors]   #4 dimensional (w,c,m,mu) f^<=(0,c,0,0)=1
    for w in range(len(fwc),W+1):
        rtwc.append([])
        rtwcm_leq.append([])
        fwc.append([])
        fwcm.append([])
        fwcm_leq.append([])
        fwcmmu.append([])
        fwcmmu_leq.append([])
        for c in range(0, numberOfColors):
            results = _computeNumbersForWC(w,c)
            rtwc[-1].append(results[0])
            rtwcm_leq[-1].append(results[1])
            fwc[-1].append(results[2])
            fwcm[-1].append(results[3])
            fwcm_leq[-1].append(results[4])
            fwcmmu[-1].append(results[5])
            fwcmmu_leq[-1].append(results[6])

def numberOfForests(w,c):
    assert w >= 0, "n=%d should be non-negative" % w
    assert c >= 0 and c < len(Tree.colors), "color %d out of bounds" % c
    if "rtwc" not in globals() or len(rtwc) <= w:
        _computeNumberOfForests(w)
    return fwc[w][c]

def numberOfRootedTrees(w,c,m=None):
    assert w > 0, "w = %d should be positive" % w
    assert c >= 0 and c < len(Tree.colors), "color %d out of bounds" % c
    if "rtwc" not in globals() or len(rtwc) <= w:
        _computeNumberOfForests(w)
    return rtwc[w][c] if m is None else rtwcm_leq[w][c][m]

###########################################################################################
#         UNRANKING
###########################################################################################

def rootedTree(w,c,i,m=None):
    assert c >= 0 and c < len(Tree.colors), "color %d out of bounds" % c
    color = Tree.colors[c]
    assert w >= color.minTreeWeight, "w=%d should be at least %d" % (w,color.minTreeWeight)
    assert i >= 0, "i=%d should be non-negative" % i

    if m is None or m > w:
        m = w
    r = color.minVertexWeight;
    childColor = color.childrenColor
    i1=i
    while i1 >= fwcm_leq[w-r][childColor][min(m,w-r)]:
        i1 -= fwcm_leq[w-r][childColor][min(m,w-r)]
        r += 1
    assert r <= color.maxVertexWeight, "%d exceeded the max vertex weight %d" % (r, color.maxVertexWeight)
    ret = Tree(None if r == w else forest(w-r, childColor, i1),weight=r,color=color)
    ret.rank = i
    return ret

def forest(w, c, i):
    assert w >= 0, "n=%d should be non-negative" % w
    assert c >= 0 and c < len(Tree.colors), "color %d out of bounds" % c
    assert i >= 0, "i=%d should be non-negative" % i
    if i >= numberOfForests(w,c):
        print("i=", i,  "should be less than the number of forests", numberOfForests(w,c),  "of weight", w)

    if w==0:
        return []
    m = bisect_left(fwcm_leq[w][c], i+1)   #find the smallest index m such that f<=(w,c,m) > i
    i -= fwcm_leq[w][c][m - 1]             #this is the index of the required tree in F(w,c,m)

    mu = bisect_left(fwcmmu_leq[w][c][m], i+1) #find the smallest index mu such that f<=(w,m,mu) > i
    i -= fwcmmu_leq[w][c][m][mu - 1]           #this is the index of the required tree in F(n,m, mu)
    return forestwcmmu(w, c, m, mu, i)

def forestwcmmu(w,c, m, mu,i):
    '''
    :return: The i-th forest of weight w, root colored c and m(F)=m and mu(F)=mu
    '''
    numberOfSmallForests = fwcm_leq[w - m * mu][c][min(w - m * mu, m - 1)]
    (i1,i2) = divmod (i, numberOfSmallForests)
    bigChildren = [rootedTree(m, c, treeIndex) for treeIndex in multiset(rtwc[m][c], mu, i1)]
    smallChildren = forest (w - m  * mu, c, i2) # the choice of i2 and the order of the trees guarantees that m(F) <= m-1 for the returned forest F
    return bigChildren + smallChildren

def numberOfMonochromaticFreeTrees(w,c):
    assert w > 0, "n = %d should be positive" % w
    assert c >= 0 and c < len(Tree.colors), "color %d out of bounds" % c
    assert Tree.colors[c].childrenColor == c, Tree.colors[c].name + " is not monochromatic"
    return _numberOfMonocentroidalMonochromaticTrees(w,c) + _numberOfBicentroidalMonochromaticTrees(w,c)

def _numberOfBicentroidalMonochromaticTrees(w,c):
    return cc(numberOfRootedTrees(w // 2, c, w//2-1), 2) if w % 2 == 0 else 0

def _numberOfMonocentroidalMonochromaticTrees(w,c):
    numberOfForests(w,c)  # Trigger computation of the f-values  // TODO: Make this cleaner & safer
    return numberOfRootedTrees(w, c, (w - 1) // 2)

def freeMonochromaticTree(w,c,i):
    i1 = i;
    if w % 2 == 0 and i1 >= _numberOfMonocentroidalMonochromaticTrees(w,c):
        # return bicentroidal tree
        i1 -= _numberOfMonocentroidalMonochromaticTrees(w,c)
        pair = [ rootedTree(w//2, c, treeIndex) for treeIndex in multiset(numberOfRootedTrees(w // 2, c, w//2-1),2,i1) ]
        t = pair[0]
        t.addTree(pair[1])
    else:
        #return monocentroidal tree
        t = rootedTree(w, c, i, (w - 1) // 2)
    t.rank = i
    return t

def numberOfBlockTrees(w):
    assert w > 0, "n = %d should be positive" % w
    return _numberOfMonocentroidalBlockTrees(w) + _numberOfBicentroidalBlockTrees(w) + _numberOfTricentroidalBlockTrees(w)

def _numberOfMonocentroidalBlockTrees(w):
    numberOfForests(w,Tree.Color.RED)     # Trigger computation of the f-values  // TODO: Make this cleaner & safer
    return rtwcm_leq[w][Tree.Color.RED][(w-1)//2] + rtwcm_leq[w][Tree.Color.YELLOW][(w-1)//2]

def _numberOfBicentroidalBlockTrees(w):
    return numberOfRootedTrees(w // 2, Tree.Color.RED, w//2-1) * numberOfRootedTrees(w//2, Tree.Color.YELLOW, w//2-1) if w % 2 == 0 else 0

def _numberOfTricentroidalBlockTrees(w):
    return cc(numberOfRootedTrees(w // 2, Tree.Color.YELLOW),2) if w % 2 == 0 else 0

def blockTree(w,i):
    assert w > 0, "n = %d should be positive" % w
    i1 = i
    for c in [Tree.Color.RED, Tree.Color.YELLOW]:
        if i1 >= rtwcm_leq[w][c][(w-1)//2]:
            i1 -= rtwcm_leq[w][c][(w - 1) // 2]
        else:
            t = rootedTree(w,c,i1,(w-1)//2)
            t.rank = i
            return t

    assert w % 2 == 0, "w should be even at this point"

    if i1 >= _numberOfBicentroidalBlockTrees(w):
        i1 -= _numberOfBicentroidalBlockTrees(w)
    else:
        # Generate a tricentroidal tree
        (i1,i2) = divmod (i1, numberOfRootedTrees(w // 2,Tree.Color.RED))
        redTree = rootedTree(w//2, Tree.Color.RED, i2, m=w//2-1)
        yellowTree = rootedTree(w//2, Tree.Color.YELLOW,i1, m=w//2-1)
        t = Tree([redTree]+yellowTree.children, color=Tree.Color.YELLOW, weight=yellowTree.weight)
        t.rank = i
        return t

    # Generate a tricentroidal tree
    pair = [rootedTree(w // 2, Tree.Color.YELLOW, treeIndex) for treeIndex in
            multiset(rtwc[w // 2 - 1][Tree.Color.YELLOW], 2, i1)]
    t = Tree(pair, color=Tree.Color.RED, weight=0)
    t.rank = i
    return t