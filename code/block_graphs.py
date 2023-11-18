from typing import Iterable

import tree
from tree import Tree, Color
import combin

def blockTrees(n:int) -> Iterable[Tree]:
    # Generate monocentroidal trees
    yield from tree.rootedTrees(n, Color.RED, maxSubtreeWeight=(n-1)//2)
    yield from tree.rootedTrees(n, Color.YELLOW, maxSubtreeWeight=(n-1)//2)

    if n % 2 == 0:
        # Generate bicentroidal trees
        for redTree in tree.rootedTrees(n//2, Color.RED, maxSubtreeWeight=n // 2 - 1):
            for yellowTree in tree.rootedTrees(n // 2, Color.YELLOW):
                yield Tree([redTree]+yellowTree.children, color=Color.YELLOW, weight=yellowTree.weight)
        # Generate tricentroidal trees
        for pair in combin.Multisets(lambda : tree.rootedTrees(n//2, Color.YELLOW),2):
            yield Tree(pair, color=Color.RED, weight=0)

def _numberOfMonocentroidalBlockTrees(w:int) -> int:
    tree.numberOfForests(w, Color.RED)     # Trigger computation of the f-values  // TODO: Make this cleaner & safer
    return tree.rtwcm_leq[w][Color.RED][(w-1)//2] + tree.rtwcm_leq[w][Color.YELLOW][(w-1)//2]

def _numberOfBicentroidalBlockTrees(w:int) -> int:
    return tree.numberOfRootedTrees(w // 2, Color.RED, w//2-1) * tree.numberOfRootedTrees(w//2, Color.YELLOW, w//2-1) if w % 2 == 0 else 0

def _numberOfTricentroidalBlockTrees(w:int) -> int:
    return combin.CC(tree.numberOfRootedTrees(w // 2, Color.YELLOW), 2) if w % 2 == 0 else 0

def numberOfBlockTrees(w:int) -> int:
    assert w > 0, f"w = {w}. Should be positive"
    return _numberOfMonocentroidalBlockTrees(w) + _numberOfBicentroidalBlockTrees(w) + _numberOfTricentroidalBlockTrees(w)

def blockTree(w:int, i:int) -> Tree:
    assert w > 0, f"w = {w}. Should be positive"
    i1 = i
    for c in [Color.RED, Color.YELLOW]:
        if i1 >= tree.rtwcm_leq[w][c][(w-1)//2]:
            i1 -= tree.rtwcm_leq[w][c][(w - 1) // 2]
        else:
            t = tree.rootedTree(w,c,i1,(w-1)//2)
            t.rank = i
            return t

    assert w % 2 == 0, "w should be even at this point"

    if i1 >= _numberOfBicentroidalBlockTrees(w):
        i1 -= _numberOfBicentroidalBlockTrees(w)
    else:
        # Generate a tricentroidal tree
        (i1,i2) = divmod (i1, tree.numberOfRootedTrees(w // 2,Color.RED))
        redTree = tree.rootedTree(w//2, Color.RED, i2, m=w//2-1)
        yellowTree = tree.rootedTree(w//2, Color.YELLOW,i1, m=w//2-1)
        t = Tree([redTree]+yellowTree.children, color=Color.YELLOW, weight=yellowTree.weight)
        t.rank = i
        return t

    # Generate a tricentroidal tree
    pair = [tree.rootedTree(w // 2, Color.YELLOW, treeIndex) for treeIndex in
            combin.Multiset(tree.rtwc[w // 2 - 1][Color.YELLOW], 2, i1)]
    t = Tree(pair, color=Color.RED, weight=0)
    t.rank = i
    return t