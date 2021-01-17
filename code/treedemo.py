from tree import *

def demoRecurrences(N):
    print ()
    print ("Numbers of unlabeled rooted trees computed using Wilf's formula")
    numberOfRootedTreesWilf(N)
    print(wtn)

    print ()
    numberOfRootedTrees(N,Tree.Color.GRAY)
    print ("Numbers of unlabeled rooted trees computed from the breakdown")
    print([None] + [numberOfRootedTrees(i,Tree.Color.GRAY) for i in range(1,N+1)])


def demoEnumeration(L,H,f, color=None, formatString=None, isDirected=False):
    '''
    :param L: Smallest weight of a tree
    :param H: Largest weight of a tree
    :param c: Color of root
    :param f: A generator getting an integer argument (the common weights of the trees)
    :param formatString: A text to print before every weight (has one % sign)
    :param isDirected: Whether or not to draw the tree as rooted.
    '''
    for w in range(L,H+1):
        if formatString is not None:
            print (formatString % w)
        rank = 0
        for t in (f(w) if color is None else f(w,color)):
            t.rank = rank
            rank = rank+1
            t.show(isDirected=isDirected,textOutput=textOutput,graphicsToScreen=graphicsToScreen,graphicsToFile=graphicsToFile)

def demoRootedTreesEnumeration(L,H, c):
    demoEnumeration(L,H, lambda w, c : rootedTrees(w, c), color=c, formatString="ROOTED TREES ON %d VERTICES", isDirected=True)

def demoFreeTreesEnumeration(L,H,c=Tree.Color.GRAY):
    demoEnumeration(L,H, freeMonochromaticTrees, color=c, formatString="FREE TREES OF WEIGHT %d", isDirected=False)

def demoBlockTreesEnumeration(L,H):
    demoEnumeration(L,H, blockTrees, formatString="BLOCK TREES OF GRAPHS ON %d VERTICES", isDirected=False)

def demoUnranking(L,H, c, numberFunction, unrankingFunction, formatString=None, isDirected=False):
    for w in range(L,H+1):
        print (formatString % w)
        for i in range(numberFunction(w) if c is None else numberFunction(w, c)):
            t = unrankingFunction(w, i) if c is None else unrankingFunction(w,c,i)
            t.show(isDirected=isDirected,textOutput=textOutput,graphicsToScreen=graphicsToScreen,graphicsToFile=graphicsToFile)

def demoRootedTreesUnranking(L,H, c):
    demoUnranking(L,H,c, numberOfRootedTrees, rootedTree, formatString="ROOTED TREES OF WEIGHT %d", isDirected=True)

def demoFreeTreesUnranking(L, H, c):
    demoUnranking(L,H,c, numberOfMonochromaticFreeTrees, freeMonochromaticTree, formatString="FREE TREES OF WEIGHT %d", isDirected=False)

def demoBlockTreesUnranking(L, H):
    demoUnranking(L,H,None, numberOfBlockTrees, blockTree, formatString="FREE TREES OF WEIGHT %d", isDirected=False)


def inputBoolean(str):
    return input(str + "(Y/N)").upper().startswith("Y")

if __name__ == "__main__":
    low = int(input("Smallest Tree Size ?"))
    high = int(input("Largest Tree Size ?"))
    color = int(input("Color number ?"))
    textOutput = inputBoolean("Text Output ?")
    graphicsToScreen = inputBoolean("Graphics to Screen")
    graphicsToFile = inputBoolean("Graphics to File ?")
##    demoRecurrences(high)
##    demoRootedTreesUnranking(low,high,color)
##    demoFreeTreesUnranking(low,high, color)
    demoBlockTreesUnranking(low,high)
##    demoRootedTreesEnumeration(low,high,color)
##    demoFreeTreesEnumeration(low,high,color)
    demoBlockTreesEnumeration(low,high)