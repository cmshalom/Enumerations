from partition import *

def choose(n,k):
    k = min(k,n-k)
    result = 1
    for i in range(k):
        result *= (n-i)
        result //= i+1
    return result

def numberOfRootedTreesUpToN(n):
    # Time complexity = \sum i log i = O(n^2 log n)
    results = [0] * (n + 1)
    for i in range(1, n + 1):
        numberOfRootedTrees(i, results)
    return results


def numberOfRootedTrees(n, results):
    '''
    Computes the number of rooted trees of n vertices according the Wilfian reccurence relation
    :param results: Upon entry, results[i] contains the number of unlabeled rooted trees of i vertices
           for every i < n. Upon exit results[n] too, contains the correct value.
    '''
    if n == 1:
        results[1] = 1
        return
    results[n] = 0
    # Time complexity n + n/2 + n/3 + n/4..  = O(n log n)
    for j in range(1, n):
        # The running time of this iteration is O(n/j)
        m = n
        for d in range(1, (n - 1) // j + 1):
            m -= j
            results[n] += d * results[d] * results[m]
    results[n] //= (n - 1)


def numberOfRootedTreesOfPartition(partition, numberOfTrees):
    maxTreeSize = len(numberOfTrees) - 1
    if (partition.len() == 0):
        return 1
    if partition.max() > maxTreeSize:
        raise AssertionError("Number of trees with", partition.max(), "vertices unknown")
    # For every element p with multiplicty k we choose a multiset of k elements from the set of T(p)
    # rootedTrees of p vertices.
    # This number is choose (T(p)-1+k, k)
    result = 1
    for b in partition.bases:
        result *= choose(numberOfTrees[b[0]] - 1 + b[1] ,b[1])
    return result


def numberOfRootedTreesUpToNViaPartitions(n):
    numbersOfTrees = numberOfRootedTreesUpToN(n)
    return [sum(numberOfRootedTreesOfPartition(partition, numbersOfTrees) for partition in partitions(i - 1))
            for i in range(n + 1)
            ]


if __name__ == "__main__":
    print()
    print ("Numbers of unlabeled rooted trees computed using Wilf's formula")
    print(numberOfRootedTreesUpToN(20))

    print()
    print ("Numbers of unlabeled rooted trees computed parition by partition using Wilf's formula for each partition")
    print(numberOfRootedTreesUpToNViaPartitions(20))
