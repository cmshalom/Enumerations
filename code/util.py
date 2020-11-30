def partialSums(a):
    result=[a[0]]
    for elt in a[1:]:
        result.append(result[-1]+elt)
    return result