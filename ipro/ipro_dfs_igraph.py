import numpy as np
import igraph as ig

from ipro.linear_solvers import LinearSolver


class DijkstraSolver(LinearSolver):
    def __init__(self, graph, source, target, dimensions, upper_bounds=None, lower_bounds=None, shortest_paths=None):
        super().__init__(problem=graph, direction="minimize")
        self.graph = graph  # igraph.Graph instance
        self.source = source  # vertex name or id
        self.target = target  # vertex name or id
        self.dimensions = dimensions
        self.upper_bounds = upper_bounds
        self.lower_bounds = lower_bounds
        self.shortest_paths = shortest_paths

    def weighter(self, weight_vec, edge_vec):
        """Compute scalar weight from edge attribute vector and input weight vector."""
        return float(np.dot(edge_vec, weight_vec))

    def solve(self, weight_vec, precomputed=True):
        if np.sum(weight_vec) < 0:
            bounds = np.array(list(self.upper_bounds))
            return bounds, []
        elif precomputed and np.any(weight_vec == 1):
            bounds = np.array(list(next(lb for lb, w in zip(self.lower_bounds, weight_vec) if w == 1)))
            shortest_path = next(sp for sp, w in zip(self.shortest_paths, weight_vec) if w == 1)
            return bounds, shortest_path
        else:
            # Compute scalar weights for each edge
            weights = []
            for e in self.graph.es:
                edge_vec = e["vec"]
                weights.append(self.weighter(weight_vec, edge_vec))

            # Get vertex indices
            src = self.graph.vs.find(name=self.source).index if isinstance(self.source, str) else self.source
            tgt = self.graph.vs.find(name=self.target).index if isinstance(self.target, str) else self.target

            # Run Dijkstra
            path_vertices = self.graph.get_shortest_paths(src, to=tgt, weights=weights, output="vpath")[0]

            # Convert path vertices to vertex names (if available)
            path = [self.graph.vs[v]["name"] if "name" in self.graph.vs.attributes() else v for v in path_vertices]

            # Sum up edge vectors along the path
            vec_return = np.zeros(self.dimensions)
            for i in range(len(path_vertices) - 1):
                eid = self.graph.get_eid(path_vertices[i], path_vertices[i + 1])
                vec_return += np.array(self.graph.es[eid]["vec"])

            print("Linear solver vec_return:", vec_return)
            return vec_return, path
