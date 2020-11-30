from bisect import *
from combin import *
from util import *

class Tree:
    def __init__(self, children=None):
        self.children = [] if children is None else children
        self.vertices = 1 + sum([child.vertices for child in self.children])

    def __str__(self):
        result = str(self.vertices)
        if len(self.children) > 0:
            result += "->"
            for child in self.children:
                result += " " + str(child.vertices)
        return result

    @staticmethod
    def _levelString(trees):
        '''
        :param trees: is a non-empty list of tree objects
        :return: a pair (str, nextLevel) where str is the string representation of the roots of trees and
                 nextLevel is the list of roots of the next level
        '''
        str = ""
        nextLevel = []
        for tree in trees:
            nextLevel += tree.children
            str += "O"
            str += " " * (tree.vertices - 1)
        return (str,nextLevel)

    def __repr__(self):
        level = [self]
        lines = []
        while (len(level) != 0):
            (str,level) = Tree._levelString(level)
            lines.append(str)
        str = ""
        for line in lines:
            str += line + "\n"
        return str

def _computeNumberOfTreesWilf(N):
    # Time complexity = \sum i log i = O(n^2 log n)
    if "tn" not in globals():
        global tn
        tn = [None, 1]   #t(1)=1, t(0) undefined
    for n in range(len(tn), N + 1):
        t = 0
        # Time complexity n + n/2 + n/3 + n/4..  = O(n log n)
        for j in range(1, n):
            # The running time of this iteration is O(n/j)
            t += sum ([d * tn[d] * tn[n - d * j] for d in range(1, (n - 1) // j + 1)])
        tn.append (t // (n - 1))

def numberOTreesWilf(n):
    '''
    :return: the number of rooted trees of n vertices according the Wilfian recurrence relation
    Upon exit the class variable tn.contains the number of trees on i vertices for every i <= n
    Existing values in tn upn entry, if any, are considered as correct
    '''
    if "tn" not in globals() or len(tn) <= n:
        _computeNumberOfTreesWilf(n)
    return tn[n]

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

def numberOfTrees(n):
    assert n > 0, "n = %d should be positive" % n
    return numberOfForests(n-1)

def tree(n,i):
    assert n > 0, "n=%d should be positive" % n
    assert i >= 0, "i=%d should be non-negative" % i
    assert i < numberOfTrees(n), "i=%d should be at most the number of trees %d on %d vertices" % (i,numberOfTrees(n),n)
    return Tree(None) if n == 1 else Tree(forest(n-1,i))

def forest(n,i):
    assert n >= 0, "n=%d should be non-negative" % n
    assert i >= 0, "i=%d should be non-negative" % i
    if i >= numberOfForests(n):
        print("i=", i,  "should be less than the number of forests", numberOfForests(n),  "on", n, "vertices")

    if n==0:
        return []
    m = bisect_left(fnm_leq[n], i+1)   #find the smallest index m such that f<=(m) > i
    i -= fnm_leq[n][m - 1]             #this is the index of the required tree in F(n,m)
    return forestnm(n,m,i)

def forestnm(n, m, i):
    '''
    :return: The i-th forest F on n vertices with m(F)=m
    '''
    mu = bisect_left(fnmmu_leq[n][m], i+1)
    i -= fnmmu_leq[n][m][mu - 1]       #this is the index of the required tree in F(n,m, mu)
    return forestnmmu(n, m, mu, i)

def forestnmmu(n,m,mu,i):
    '''
    :return: The i-th forest F on n vertices with m(F)=m and mu(F)=mu
    '''
    numberOfSmallForests = fnm_leq[n - m * mu] [min(n - m * mu, m - 1)]
    (i1,i2) = divmod (i, numberOfSmallForests)
    bigChildren = [tree(m, treeIndex) for treeIndex in multiset(tn[m], mu, i1)]
    smallChildren = forest (n - m  * mu, i2) # the choice of i2 and the oder of the trees guarantees that m(F) <= m-1 for the returned forest F
    return bigChildren + smallChildren

if __name__ == "__main__":
    print ("Numbers of unlabeled rooted trees computed using Wilf's formula")
    print(numberOTreesWilf(20))

    print ()
    print ("Numbers of unlabeled rooted forest computed from the breakdown")
    print(numberOfTrees(20))
    print("tn=", tn)
    print("fn=", fn)

    print ("COMPACT (and ambiguous) REPRESENTATIONS of TREES")
    for n in range(1,7):
        print ("Trees on %d vertices" % n)
        for i in range(numberOfTrees(n)):
            print ("Tree ", i, "=", str(tree(n,i)))

    print ("REPRESENTATIONS of TREES")
    for n in range(1,5):
        print ("Trees on %d vertices" % n)
        for i in range(numberOfTrees(n)):
            print ("Tree ", i, ":")
            print (repr(tree(n,i)))
