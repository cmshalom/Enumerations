import networkx as nx
import matplotlib.pyplot as plt

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

def DrawColoredGraph(g: nx.Graph, weights:bool = False, planar:bool=True) -> None:
    colors= [v[1].get('color', 'gray') for v in g.nodes(data=True)]
    if weights:
      labels= {v[0]: str(v[1].get('weight', '')) for v in g.nodes(data=True)}
      if planar:
        nx.draw_planar(g, with_labels=True, labels=labels, font_weight='bold', font_size=20,
                      node_color=colors, node_size=700)
      else:
        nx.draw_networkx(g, with_labels=True, labels=labels, font_weight='bold', font_size=20,
                         node_color=colors, node_size=700)
    else:
      if planar:
        nx.draw_planar(g, node_color=colors, node_size=700)
      else:
        nx.draw_networkx(g, node_color=colors, node_size=700)

def Show(obj, isDirected=False, textOutput=True, graphicsToScreen=False, 
              graphicsToFile=False, toGraph6=False):
    '''
    Exhibits obj on one or more destinations
    Args:
      The argument toGraph6 may have one of the following values
      None: No output
      '': Output to the default file whose name is deduced from the name of the object.
      str: (except '') Output to file having this name
      file: An open file that will receive the output. 
    '''
    if textOutput:
        print(obj.name)
        print(obj.__repr__())
    if graphicsToScreen or graphicsToFile or toGraph6 is not None:
      g = obj if isinstance(obj, nx.Graph) else obj.graph(isDirected=isDirected)
      if graphicsToScreen or graphicsToFile:
        plt.clf()
        if graphicsToScreen:
            plt.title(obj.name)
        DrawColoredGraph(g, weights=True, planar=obj.is_planar)
        if graphicsToFile:
            plt.savefig(obj.name)
        if graphicsToScreen:
            plt.show()
      if toGraph6 is None:
        pass
      elif toGraph6=='':
        nx.write_graph6(g, f"{obj.name}.g6", header=False)
      elif isinstance(toGraph6, str):
        nx.write_graph6(g, toGraph6, header=False)
      else:
        toGraph6.write(nx.to_graph6_bytes(g, header=False))


