def partialSums(a):
    result=[a[0]]
    for elt in a[1:]:
        result.append(result[-1]+elt)
    return result

def boundedGenerator(g, n):
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
    

def concatenateGenerators(generators):
    for g in generators:
        for elt in g:
            yield elt