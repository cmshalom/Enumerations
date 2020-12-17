from bisect import *
from combin import *
from util import *

import networkx as nx
import matplotlib.pyplot as plt

class Tree:
    def __init__(self, children=None):
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

    def __str__(self):
        '''
        :return: a string with the number of vertices of the tree and every subtree
        (not recursive)
        '''
        result = str(self.vertices)
        if len(self.children) > 0:
            result += "->"
            for child in self.children:
                result += " " + str(child.vertices)
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

    def plot(self):
        g = nx.Graph()
        self._plot(g)
        nx.draw(g, with_labels=True, font_weight='bold')

    def _plot(self, g, parent=None):
        root = g.number_of_nodes();
        g.add_node(root)
        if parent is not None:
            g.add_edge(root,parent)
        for child in self.children:
            child._plot(g, root)

    def show(self):
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
            self.plot()
            if graphicsToFile:
                assert "rank" in self.__dict__
                plt.savefig("Tree-%d-%d.png" % (self.vertices, self.rank))
            if graphicsToScreen:
                plt.show()

###########################################################################################
#         GENERATORS
###########################################################################################

def rootedTrees(n):
    if n == 1:
        yield Tree()
        return
    for f in forests(n-1):
        yield Tree(f)

def forests(n):
    return forestsnm_leq(n,n)

def forestsnm_leq(n,m):
    if n==0:
        yield []
    elif m==0:
        return
    else:
        for forest in forestsnmmu(n, 1, n):
            yield forest
        for mm in range (2,m+1):
            for mu in range (1, n // mm + 1):
                for forest in forestsnmmu(n,mm,mu):
                    yield forest
    return

def forestsnmmu(n,m,mu):
    def f():
        return rootedTrees(m)

    for set in multisets(f, mu):
        for forest in forestsnm_leq(n-m*mu, min(n-m*mu,m-1)):
            yield set + forest


def freeTrees(n):
    def f():
        return rootedTrees(n // 2)
    # Generate monocentroidal trees
    for forest in forestsnm_leq(n-1, (n-1) // 2):
        yield Tree(forest)

    # Generate bicentroidal trees
    if n % 2 == 0:
        for pair in multisets(f,2):
            yield Tree([pair[0]]+pair[1].children)

def demoRootedTreesEnumeration(L,H):
    print ()
    print ("ROOTED TREES ENUMERATION")
    for n in range(L,H+1):
        print ("Rooted Trees on %d vertices" % n)
        rank = 0
        for t in rootedTrees(n):
            t.rank = rank
            rank = rank+1
            t.show()

def demoFreeTreesEnumeration(L,H):
    print ()
    print ("FREE TREES ENUMERATION")
    for n in range(L,H+1):
        print ("Free Trees on %d vertices" % n)
        rank = 0
        for t in freeTrees(n):
            t.rank = rank
            rank = rank+1
            t.show()

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

def demoRootedTreesUnranking(L,H):
    print ()
    print ("ROOTED TREES UNRANKING")
    numberOfRootedTrees(H)
    for n in range(L,H+1):
        print ("Rooted Trees on %d vertices" % n)
        for i in range(numberOfRootedTrees(n)):
            print ("Tree ", i, ":")
            t = rootedTree(n,i)
            t.rank = i
            t.show()

def demoFreeTreesUnranking(L, H):
    print ()
    print ("FREE TREES")
    numberOfRootedTrees(H)
    for n in range(L,H+1):
        print ("Free Trees on %d vertices" % n)
        for i in range(numberOfFreeTrees(n)):
            t = freeTree(n,i)
            t.rank = i
            t.show()

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
#    demoReccurrences(20)
#    demoRootedTreesUnranking(5,5)
#    demoFreeTreesUnranking(1,5)
#    demoRootedTreesEnumeration(5,5)
    demoFreeTreesEnumeration(Low,High)