import networkx as nx

class RepoSearcher:
    def __init__(self, graph):
        self.graph = graph

    def one_hop_neighbors(self, query):
        # get one-hop neighbors from networkx graph
        return list(self.graph.neighbors(query))

    def two_hop_neighbors(self, query):
        # get two-hop neighbors from networkx graph
        one_hop = self.one_hop_neighbors(query)
        two_hop = []
        for node in one_hop:
            two_hop.extend(self.one_hop_neighbors(node))
        return list(set(two_hop))

    def dfs(self, query, depth):
        # perform depth-first search on networkx graph
        visited = []
        stack = [(query, 0)]
        while stack:
            node, level = stack.pop()
            if node not in visited:
                visited.append(node)
                if level < depth:
                    stack.extend(
                        [(n, level + 1) for n in self.one_hop_neighbors(node)]
                    )
        return visited
    
    def bfs(self, query, depth):
        # perform breadth-first search on networkx graph
        visited = []
        queue = [(query, 0)]
        while queue:
            node, level = queue.pop(0)
            if node not in visited:
                visited.append(node)
                if level < depth:
                    queue.extend(
                        [(n, level + 1) for n in self.one_hop_neighbors(node)]
                    )
        return visited