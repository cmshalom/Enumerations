import copy
import sys
import util

def Choose(n,k):
    ''' Computes \choose {n} {k}'''
    if n < 0 or k < 0:
        return 0
    k = min(k,n-k)
    result = 1
    for i in range(k):
        result *= (n-i)
        result //= i+1
    return result

def ChooseMinGE(k,i):
    '''
    :param k:
    :param i:
    :return: The smallest value n such that choose(n,k) >= i
    Time complexity O(result)
    '''
    value = 1
    n = k
    denom = 0;
    while value < i:
        n += 1
        denom +=1
        value *= n
        value //= denom
    return n

def CC(n,k):
    return Choose(n-1+k, k)

def CCMinGE(k,i):
    '''
    :param k:
    :param i:
    :return: The smallest value n such that cc(n,k) >= i
    Time complexity O(result+k)
    '''
    return ChooseMinGE(k, i) - k + 1

def Multisets(n,k):
    '''
    :param n: is either an integer of an a generator function
    :return: yields all the multisets of k elements from [0,n-1] or from the  generator n()
    '''
    if k == 0:
        yield []
        return
    if isinstance(n, int):
        if n == 1:
            yield [0]*k
            return
        for firstElement in range(n):
            for multiset in Multisets(firstElement+1,k-1):
                yield [firstElement] + multiset
    else:
        yield from _multisets(n, k, sys.maxsize)

def _multisets(f, k, bound):
    if k == 0:
        yield []
        return
    i = 0
    for item in util.BoundedGenerator(f(),bound):
        i += 1
        for multiset in _multisets(f, k-1, i):
            yield [item] + multiset

def Multiset(n,k,i):
    '''
    :return: The i'th multiset among the multisets consisting k elements from the integers 0 to n-1
    Time complexity <= O(k(n+k))
    '''
    assert i < CC(n,k), "i=%d should be less than the number %d of multisets of 0 - %d with %d elements" % (i, cc(n,k), n-1, k)
    if n == 1 or k == 0:
        return [0]*k
    firstElement = CCMinGE(k,i+1)          # Time complexity O(firstElement+k) \in O(n+k)
    smallerMultisets = CC(firstElement-1,k) # Time complexity O(k)
    return [firstElement-1] + Multiset(firstElement,k-1,i - smallerMultisets)
