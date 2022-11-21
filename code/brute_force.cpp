#include "brute_force.h"
#include <map>

using namespace std;

// implementation of traveling Salesman Problem
int travllingSalesmanProblem(vector<vector<int>> graph, int s, string& ans)
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

		if (current_pathweight < min_path) {
			ans = "";
			for (int i = 0; i < vertex.size(); ++i) {
				ans += to_string(vertex[i]);
				ans += "->";
			}
		}

		// update minimum
		min_path = min(min_path, current_pathweight);

	} while (
		next_permutation(vertex.begin(), vertex.end()));

	return min_path;
}
// Function to find the minimum
// cost path for all the paths
int tsp_greedy(vector<vector<int> > graph)
{	
	for(int i = 0; i < graph.size(); ++i) {
		for (int j = 0; j < graph[0].size(); ++j) {
			if (graph[i][j] == 0 || graph[i][j] == INT_MAX) {
				graph[i][j] = INT_MAX;
			}
		}
	}
	int sum = 0;
	int counter = 0;
	int j = 0, i = 0;
	int min = INT_MAX;
	map<int, int> visitedRouteList;

	// Starting from the 0th indexed
	// city i.e., the first city
	visitedRouteList[0] = 1;
	int route[graph.size()];

	// Traverse the adjacency
	// matrix graph[][]
	while (i < graph.size() && j < graph[i].size())
	{

		// Corner of the Matrix
		if (counter >= graph[i].size() - 1)
		{
			break;
		}

		// If this path is unvisited then
		// and if the cost is less then
		// update the cost
		if (j != i && (visitedRouteList[j] == 0))
		{
			if (graph[i][j] < min)
			{
				min = graph[i][j];
				route[counter] = j + 1;
			}
		}
		j++;

		// Check all paths from the
		// ith indexed city
		if (j == graph[i].size())
		{
			sum += min;
			min = INT_MAX;
			visitedRouteList[route[counter] - 1] = 1;
			j = 0;
			i = route[counter] - 1;
			counter++;
		}
	}

	// Update the ending city in array
	// from city which was last visited
	i = route[counter - 1] - 1;

	for (j = 0; j < graph.size(); j++)
	{

		if ((i != j) && graph[i][j] < min)
		{
			min = graph[i][j];
			route[counter] = j + 1;
		}
	}
	sum += min;

	// Started from the node where
	// we finished as well.
	return sum;
}
