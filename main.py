#!/usr/bin/env python3
"""
The algorithm reads an undirected graph (one edge per line: "src dst"), runs multiple
iterations of community detection combining modularity optimization and regularization,
and writes the best partition (one community per line) into "partition.graph".
"""

import sys
import random
import time
from collections import defaultdict
from typing import List, Tuple, Dict

# Constants
REPS = 15
MAX_STEPS = 20000


def read_graph_data(filename: str) -> Tuple[int, int, List[Tuple[int, int]]]:
    """
    Each line must contain two integers representing an edge: "src dst"
    Returns:
        V: number of vertices (assumes vertices are 0-indexed)
        E: number of edges (before doubling for undirected graph)
        edges: list of (src, dst) tuples.
    """
    edges = []
    V = 0
    with open(filename, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            i, j = int(parts[0]), int(parts[1])
            edges.append((i, j))
            V = max(V, i, j)
    V += 1
    E = len(edges)
    return V, E, edges


class InitGraph:
    """
    Builds the initial graph representation from a list of edges.
    The graph is undirected so each edge is added in both directions.
    The adjacency lists are sorted for consistency.
    """
    def __init__(self, V: int, E: int, edges: List[Tuple[int, int]]):
        self.V = V
        self.E = 2 * E  # undirected graph: count both directions
        self.adj: List[List[int]] = [[] for _ in range(V)]
        for u, v in edges:
            self.adj[u].append(v)
            self.adj[v].append(u)
        for i in range(V):
            self.adj[i].sort()

    def debug_print(self):
        print(f"Initial graph G(V: {self.V}, E: {self.E}) with adjacency list:")
        for i in range(self.V):
            print(f"({i})", " -> ".join(map(str, self.adj[i])))
        print()


class Graph:
    """
    Represents the weighted graph.
    The internal representation flattens the per-vertex adjacency lists and
    computes cumulative degree indices.
    """
    def __init__(self, init_graph: InitGraph):
        self.V = init_graph.V
        self.E = init_graph.E  # total number of directed edges
        self.W = self.E        # total weight (initially, all weights are 1)
        self.degrees: List[int] = []
        cum = 0
        for i in range(self.V):
            cum += len(init_graph.adj[i])
            self.degrees.append(cum)
        self.edges: List[int] = []
        for i in range(self.V):
            for neighbor in init_graph.adj[i]:
                self.edges.append(neighbor)
        self.weights: List[int] = [1] * len(self.edges)

    def degree(self, v: int) -> int:
        """Returns the (local) degree of vertex v."""
        return self.degrees[v] if v == 0 else self.degrees[v] - self.degrees[v - 1]

    def adjacent(self, v: int) -> Tuple[List[int], List[int]]:
        """
        Returns the list of neighbors and corresponding weights for vertex v.
        """
        start = 0 if v == 0 else self.degrees[v - 1]
        end = self.degrees[v]
        return self.edges[start:end], self.weights[start:end]

    def self_loops(self, v: int) -> int:
        """
        Returns the weight of the self-loop at vertex v (if any).
        """
        nbrs, wts = self.adjacent(v)
        for i, nbr in enumerate(nbrs):
            if nbr == v:
                return wts[i]
        return 0

    def weighted_degree(self, v: int) -> int:
        """Returns the sum of weights of edges adjacent to vertex v."""
        nbrs, wts = self.adjacent(v)
        return sum(wts)

    def debug_print(self):
        print(f"Graph G(V: {self.V}, E: {self.E}, W: {self.W})")
        print("degrees:", self.degrees)
        print("edges:", self.edges)
        print("weights:", self.weights)
        print()


class Community:
    """
    Implements the community detection operations:
      - Phase 1: Greedy modularity optimization.
      - Phase 2: Graph aggregation.
      - Other helper functions.
    """
    def __init__(self, graph: Graph, originalV: int, communitySizes: List[int]):
        self.originalV = originalV         # Original number of vertices
        self.graph = graph
        self.networkSize = graph.V         # Current number of nodes (could be communities)
        self.adjWeights: List[int] = [-1] * self.networkSize
        self.adjPositions: List[int] = [0] * self.networkSize
        self.adjLast = 0

        self.vertexToCommunityMap: List[int] = list(range(self.networkSize))
        self.ein: List[float] = [0.0] * self.networkSize  # internal weights
        self.eout: List[float] = [0.0] * self.networkSize  # total degree weights
        self.c: List[int] = communitySizes[:]  # community sizes

        for i in range(self.networkSize):
            self.ein[i] = self.graph.self_loops(i)
            self.eout[i] = self.graph.weighted_degree(i)

    def remove(self, vertex: int, community: int, edgesToCommunity: int):
        """Removes a vertex from a community."""
        self.eout[community] -= self.graph.weighted_degree(vertex)
        self.ein[community] -= 2 * edgesToCommunity + self.graph.self_loops(vertex)
        self.vertexToCommunityMap[vertex] = -1

    def insert(self, vertex: int, community: int, edgesToCommunity: int):
        """Inserts a vertex into a community."""
        self.eout[community] += self.graph.weighted_degree(vertex)
        self.ein[community] += 2 * edgesToCommunity + self.graph.self_loops(vertex)
        self.vertexToCommunityMap[vertex] = community

    def modularity(self) -> float:
        """Calculates the modularity of the current partition."""
        mod = 0.0
        for i in range(self.networkSize):
            if self.eout[i] > 0:
                mod += self.ein[i] / self.graph.W - (self.eout[i] / self.graph.W) ** 2
        return mod

    def regularization(self) -> float:
        """
        Computes the regularization term.
        For singleton communities, adds 1; otherwise, computes density.
        """
        densitySum = 0.0
        for i in range(self.networkSize):
            if self.c[i] == 1:
                densitySum += 1
            else:
                den = 0.5 * self.c[i] * (self.c[i] - 1)
                densitySum += self.ein[i] / den
        n = float(self.networkSize)
        V = float(self.originalV)
        return 0.5 * (densitySum / n - (n / V))

    def modularity_gain(self, v: int, newCommunity: int, numEdges: int, degree: int) -> float:
        """
        Computes the change in modularity by moving vertex v to a new community.
        """
        eouts = self.eout[newCommunity]
        m = self.graph.W
        return numEdges - eouts * degree / m

    def adj_communities(self, vertex: int):
        """
        Collects the weights from the neighbors of a vertex for each community.
        Uses adjWeights and adjPositions as temporary arrays.
        """
        # Reset previously stored community weights
        for i in range(self.adjLast):
            self.adjWeights[self.adjPositions[i]] = -1
        self.adjLast = 0

        nbrs, wts = self.graph.adjacent(vertex)
        deg = self.graph.degree(vertex)

        # Always add the vertex's own community first
        current_comm = self.vertexToCommunityMap[vertex]
        self.adjPositions[0] = current_comm
        self.adjWeights[current_comm] = 0
        self.adjLast = 1

        for i in range(deg):
            neighbour = nbrs[i]
            neighbour_comm = self.vertexToCommunityMap[neighbour]
            neighbour_weight = wts[i] if len(wts) != 0 else 1
            if neighbour != vertex:
                if self.adjWeights[neighbour_comm] == -1:
                    self.adjWeights[neighbour_comm] = 0
                    self.adjPositions[self.adjLast] = neighbour_comm
                    self.adjLast += 1
                self.adjWeights[neighbour_comm] += neighbour_weight

    def phase1(self, dontLimit: bool) -> bool:
        """
        Phase 1: Greedy modularity optimization.
        The function reassigns vertices to communities as long as modularity increases.
        """
        hasChanged = False
        counter = 0
        newMod = self.modularity()
        currentMod = newMod

        while True:
            moves = 0
            randomOrder = list(range(self.networkSize))
            random.shuffle(randomOrder)

            for vertex in randomOrder:
                vertex_comm = self.vertexToCommunityMap[vertex]
                vertex_degree = self.graph.weighted_degree(vertex)
                self.adj_communities(vertex)
                self.remove(vertex, vertex_comm, self.adjWeights[vertex_comm])
                best_comm = vertex_comm
                best_num_edges = 0
                best_increase = 0.0

                for j in range(self.adjLast):
                    comm = self.adjPositions[j]
                    increase = self.modularity_gain(vertex, comm, self.adjWeights[comm], vertex_degree)
                    if increase > best_increase:
                        best_comm = comm
                        best_num_edges = self.adjWeights[comm]
                        best_increase = increase

                self.insert(vertex, best_comm, best_num_edges)
                if best_comm != vertex_comm:
                    moves += 1

                counter += 1
                if not dontLimit and counter >= MAX_STEPS:
                    break

            newMod = self.modularity()
            if moves > 0:
                hasChanged = True

            # Continue only if under step limit and there is an improvement.
            if not ((dontLimit or counter < MAX_STEPS) and moves > 0 and newMod > currentMod):
                break
            currentMod = newMod

        return hasChanged

    def phase2(self) -> Tuple[Graph, List[int]]:
        """
        Phase 2: Aggregates the current communities into a new graph.
        Returns a tuple of (new_graph, new_communitySizes)
        """
        # Build a mapping from old community id to a new sequential id.
        existing = sorted(set(comm for comm in self.vertexToCommunityMap if comm != -1))
        mapping = {old: new for new, old in enumerate(existing)}

        # Build list of vertices per new community.
        community_vertices: Dict[int, List[int]] = {i: [] for i in range(len(mapping))}
        for v in range(self.networkSize):
            comm = mapping[self.vertexToCommunityMap[v]]
            community_vertices[comm].append(v)
        num_communities = len(mapping)

        # Create a new graph instance (without calling __init__).
        new_graph = Graph.__new__(Graph)
        new_graph.V = num_communities
        new_graph.degrees = [0] * num_communities
        new_graph.edges = []
        new_graph.weights = []
        new_graph.E = 0
        new_graph.W = 0

        # New community sizes are the sum of sizes from the aggregated communities.
        communitySizes_new = [0] * num_communities
        for i in range(num_communities):
            communitySizes_new[i] = sum(self.c[v] for v in community_vertices[i])

        # Build the new graph.
        for c in range(num_communities):
            m: Dict[int, float] = {}
            for v in community_vertices[c]:
                nbrs, wts = self.graph.adjacent(v)
                deg = self.graph.degree(v)
                for i in range(deg):
                    neighbour = nbrs[i]
                    neighbour_comm = mapping[self.vertexToCommunityMap[neighbour]]
                    neighbour_weight = wts[i] if len(wts) != 0 else 1
                    m[neighbour_comm] = m.get(neighbour_comm, 0) + neighbour_weight
            new_graph.degrees[c] = len(m) if c == 0 else new_graph.degrees[c - 1] + len(m)
            new_graph.E += len(m)
            for comm, weight in m.items():
                new_graph.W += weight
                new_graph.edges.append(comm)
                new_graph.weights.append(weight)
        return new_graph, communitySizes_new

    def partition(self) -> List[int]:
        """
        Returns the current partition as a list mapping each original vertex to its community.
        """
        existing = sorted(set(comm for comm in self.vertexToCommunityMap if comm != -1))
        mapping = {old: new for new, old in enumerate(existing)}
        return [mapping[comm] for comm in self.vertexToCommunityMap]

    def debug_print(self):
        print("Community:")
        print("Network size =", self.networkSize)
        print("vertexToCommunityMap:", self.vertexToCommunityMap)
        print("ein:", self.ein)
        print("eout:", self.eout)
        print("c:", self.c)
        print("Modularity:", self.modularity())
        print("Regularization:", self.regularization())
        print()


def main(argv):
    if len(argv) < 2:
        print("Usage: python main.py <graph_file>")
        sys.exit(1)

    # Read the initial graph from file
    filename = argv[1]
    V, E, edges = read_graph_data(filename)
    init_graph = InitGraph(V, E, edges)

    best_Q = 0.0
    best_partition: Dict[int, List[int]] = {}

    # Run multiple repetitions to overcome local optima
    for rep in range(REPS):
        # Reinitialize the graph and communities
        graph = Graph(init_graph)
        cs = [1] * V
        community = Community(graph, V, cs)

        Qs: List[float] = []
        levels: List[List[int]] = []

        # Optionally limit the number of phase 1 steps
        dontLimit = (random.randint(0, 2) == 0)  # about 1/3 chance to be True

        # Iteratively apply Phase 1 and Phase 2 until no further changes occur
        while True:
            hasChanged = community.phase1(dontLimit)
            newMod = community.modularity()
            # Save current level partition
            levels.append(community.partition())
            # Aggregate communities to form a new graph
            graph, communitySizes = community.phase2()
            community = Community(graph, V, communitySizes)
            newReg = community.regularization()
            Qs.append(newMod + newReg)
            if not hasChanged:
                break

        # Pick the level (iteration) with maximum Q = modularity + regularization
        maxQ_index = 0
        maxQ_value = 0.0
        for idx, val in enumerate(Qs):
            if val > maxQ_value:
                maxQ_index = idx
                maxQ_value = val

        # If this repetition produced a better Q, compute the final partition.
        if maxQ_value > best_Q:
            vertexToCommunity = list(range(V))
            # Propagate the partition across levels up to maxQ_index
            for level in range(maxQ_index):
                vertexToCommunity = [levels[level][vertexToCommunity[v]] for v in range(V)]
            communities: Dict[int, List[int]] = defaultdict(list)
            for v in range(V):
                communities[vertexToCommunity[v]].append(v)
            best_Q = maxQ_value
            best_partition = communities

    # Output the best Q value and write the partition to file.
    print(best_Q)
    with open("partition.graph", "w") as f:
        for comm in best_partition.values():
            f.write(" ".join(map(str, comm)) + "\n")


if __name__ == '__main__':
    main(sys.argv)
