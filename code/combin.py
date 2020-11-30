def choose(n,k):
    ''' Computes \choose {n} {k}'''
    if n < 0 or k < 0:
        return 0
    k = min(k,n-k)
    result = 1
    for i in range(k):
        result *= (n-i)
        result //= i+1
    return result

def printPascal(N):
    for n in range(N+1):
        for k in range(n+1):
            print (choose(n,k), end=" ")
        print ()

def choose_minGE(k,i):
    '''
    :param k:
    :param i:
    :return: The smallest value n such that choose(n,k) >= i
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

def cc(n,k):
    return choose(n-1+k, k)

def cc_minGE(k,i):
    '''
    :param k:
    :param i:
    :return: The smallest value n such that cc(n,k) >= i
    '''
    return choose_minGE(k, i) - k + 1

def multiset(n,k,i):
    '''
    :return: The i'th multiset among the multisets consisting k elements from the integers 0 to n-1
    '''
    assert i < cc(n,k), "i=%d should be less than the number %d of multisets of 0 - %d with %d elements" % (i, cc(n,k), n-1, k)
    if n == 1:
        return [0]*k
    if k == 0:
        return []
    firstElement = cc_minGE(k,i+1)
    smallerMultisets = cc(firstElement-1,k)
    return [firstElement-1] + multiset(firstElement,k-1,i - smallerMultisets)
