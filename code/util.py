
def PartialSums(a: list[int]) -> list[int]:
    result=[a[0]]
    for elt in a[1:]:
        result.append(result[-1]+elt)
    return result

def BoundedGenerator(g, n: int):
    '''
    Acts as a generator that bounds the generator g
    yields at most n items
    :return:
    '''
    try:
        for i in range(n):
            yield next(g)
    except StopIteration:
        return 
    

def ConcatenateGenerators(generators):
    for g in generators:
      yield from g
