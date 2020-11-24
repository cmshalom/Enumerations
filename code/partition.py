import copy

class Partition:

    def __init__(self,number, partition=None):
        if partition is None:
            if number == 0:
                self.numbers = []
                self.bases = []
            else:
                self.numbers = [number]
                self.bases = [[number,1]]
        elif number < partition.max():
            raise ValueError(number + "<" + partition.max() + "when creating partition")
        else:
            self.numbers = [number] + partition.numbers
            if number == partition.max():
                self.bases = copy.deepcopy(partition.bases)
                self.bases[0][1] += 1
            else:
                self.bases = [[number,1]] + partition.bases

    def max(self):
        return 0 if len(self.numbers) == 0 else self.numbers[0]

    def len(self):
        return len(self.numbers)

    def __str__(self):
        return str(self.numbers)

    def __repr__(self):
        result = ""
        for b in self.bases:
            if result != "":
                result += " + "
            result += str(b[0]) + "^" + str(b[1])
        return result

def _partitions(n, max):
    ### Generates all the partitions of n into numbers less than or equal max
    if n == 0:
        yield Partition(0)
    if max > n:
        max = n
    for i in range(max, 0, -1):
        for partition in _partitions(n - i, i):
            yield Partition(i, partition)

def partitions(n):
    return _partitions(n, n)


if __name__ == "__main__":
    print ('Partitions of', 10)
    for partition in partitions(10):
        print (partition, "=", repr(partition))
