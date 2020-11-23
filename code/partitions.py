def _partitions(n, max):
    ### Generates all the partitions of n into numbers less than or equal max
    if n == 0:
        yield []
    if max > n:
        max = n
    for i in range(max, 0, -1):
        for set in _partitions(n - i, i):
            yield [i] + set


def partitions(n):
    return _partitions(n, n)


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
    if (len(partition) == 0):
        return 1
    if partition[0] > maxTreeSize:
        raise AssertionError("Number of trees with", partition[0], "vertices unknown")
    # We assume that a partition is given as a non-increasing list of numbers
    # For every maximal subsequence of k identical elements i we choose a multiset of k elements from the set of T(i)
    # rootedTrees of i vertices.
    # This number is \choose {T(i)-1+k} {k}
    # We compute it iteratively, by multiplying by T(i)/1, (T(i)+1)/2, and so on until (T(i)+k-1)/k
    if len(partition) == 0:
        return 1
    result = 1
    previousI = partition[0] + 1;
    for i in partition:
        rank = 1 if i < previousI else rank + 1
        result *= (numberOfTrees[i] + rank - 1)
        result //= rank
        previousI = i
    return result


def numberOfRootedTreesUpToNViaPartitions(n):
    numbersOfTrees = numberOfRootedTreesUpToN(n)
    return [sum(numberOfRootedTreesOfPartition(partition, numbersOfTrees) for partition in partitions(i - 1))
            for i in range(n + 1)
            ]


if __name__ == "__main__":
    print ('Partitions of', 10)
    for set in partitions(10):
        print (set)
    print()
    print ("Numbers of unlabeled rooted trees computed using Wilf's formula")
    print(numberOfRootedTreesUpToN(20))

    print()
    print ("Numbers of unlabeled rooted trees computed parition by partition using Wilf's formula for each partition")
    print(numberOfRootedTreesUpToNViaPartitions(20))
