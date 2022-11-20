#include "brute_force.h"

using namespace std;

// implementation of traveling Salesman Problem
int travllingSalesmanProblem(vector<vector<int>> graph, int s)
{   
    int V = graph.size();
	// store all vertex apart from source vertex
	vector<int> vertex;
	for (int i = 0; i < V; i++)
		if (i != s)
			vertex.push_back(i);

	// store minimum weight Hamiltonian Cycle.
	int min_path = INT_MAX;
	do {

		// store current Path weight(cost)
		int current_pathweight = 0;

		// compute current path weight
		int k = s;
        bool deadend = false;
		for (int i = 0; i < vertex.size(); i++) {
            if (graph[k][vertex[i]] == INT_MAX) {
                deadend = true;
                break;
            }
			current_pathweight += graph[k][vertex[i]];
			k = vertex[i];
		}
        if (graph[k][s] == INT_MAX || deadend) {
            continue;
        }
		current_pathweight += graph[k][s];

		// update minimum
		min_path = min(min_path, current_pathweight);

	} while (
		next_permutation(vertex.begin(), vertex.end()));

	return min_path;
}
