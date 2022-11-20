// Sequential Version Genetic Algorithm for Travelling Salesman Problem
// Author: Haoxuan Zhu (haoxuanz@andrew.cmu.edu)
// Date: Nov 15, 2022
#include <limits.h>
#include "chromosome.h"
#include <vector>
#include <cmath>

#include "brute_force.h"

using namespace std;

#define GEN_THRES 5000
#define POPLUATION_SIZE 10
// #define DEBUG

vector<vector<int>> load_tsp_from_file(string filename) {
    vector<vector<int>> map;
    ifstream fp(filename);
    string line;
    getline(fp, line);
    size_t dim = (int)atoi(line.c_str());
    while (getline(fp, line)) {
        vector<int> curr_line;
        stringstream sstream(line);
        string str;
        for(int i = 0; i < dim; ++i) {
            getline(sstream, str, ' ');
            int val = atoi(str.c_str());
            val = val == -1 ? INT_MAX : val;
            curr_line.push_back(val);
        }
        map.push_back(curr_line);
    }
    fp.close();
    return map;
}

int cooldown(int temp)
{
    return (999 * temp) / 1000;
}

int cal_fitness(vector<vector<int>> map, string gnome)
{   
    // cout << "cal_fitness() is called" << endl;
    // cout << gnome << endl;
    int f = 0;
    for (int i = 0; i < gnome.size() - 1; i++) {
        if (map[gnome[i] - 48][gnome[i + 1] - 48] == INT_MAX) {
            // cout << "INT_MAX" << endl;
            return INT_MAX;
        }
        f += map[gnome[i] - 48][gnome[i + 1] - 48];
        // cout << "f: " << f << endl;
    }
    return f;
}

void solver(vector<vector<int>> map, int gen_thres, int pop_size) {
    int gen = 1;
    int num_node = map.size();
    vector<Chromosome> population;
    for (int i = 0; i < pop_size; ++i) {
        Chromosome chro = Chromosome(num_node);
        chro.create();
        chro.fitness = cal_fitness(map, chro.get_gnome());
        population.push_back(chro);
    }

    #ifdef DEBUG
    cout << "\nInitial population: " << endl << "GNOME     FITNESS VALUE\n";
    for (int i = 0; i < pop_size; i++)
        cout << population[i].get_gnome() << " "<< population[i].fitness << endl;
    cout << "\n";
    #endif

    bool found = false;
    int temperature = 10000;

    while (temperature > 1000 && gen <= gen_thres) {
        sort(population.begin(), population.end(), worsethan);
        // cout << "\nCurrent temp: " << temperature << "\n";
        vector<Chromosome> new_population;

        for (int i = 0; i < pop_size; i++) {
            Chromosome offspring = population[i];
            while (true) {
                offspring.mutate();
                offspring.fitness = cal_fitness(map, offspring.get_gnome());
                if (offspring.fitness <= population[i].fitness) {
                    new_population.push_back(offspring);
                    break;
                }
                else {
                    float prob = pow(2.7, -1 * ((float)(offspring.fitness
                                                - population[i].fitness)
                                        / temperature));
                    if (prob > 0.5) {
                        new_population.push_back(population[i]);
                        break;
                    }
                }
            } 
        }

        temperature = cooldown(temperature);
        population = new_population;

        #ifdef DEBUG
        cout << "Generation " << gen << " \n";
        cout << "GNOME\tFITNESS VALUE\n";
        sort(population.begin(), population.end(), worsethan);
        for (int i = 0; i < pop_size; i++)
            cout << population[i].get_gnome() << "\t"
                << population[i].fitness << endl;
        #endif
        gen++;
    }
    cout << "Generation " << gen << " \n";
    cout << "GNOME\tFITNESS VALUE\n";
    sort(population.begin(), population.end(), worsethan);
    for (int i = 0; i < pop_size; i++)
        cout << population[i].get_gnome() << "\t"
            << population[i].fitness << endl;
}

int main() {
    vector<vector<int>> map = load_tsp_from_file("testcase/5.txt");
    for(int i = 0; i < map.size(); ++i) {
        for (int j = 0; j < map[i].size(); ++j) {
            cout << map[i][j] << " ";
        }
        cout << endl;
    }
    solver(map, GEN_THRES, POPLUATION_SIZE);

    int s = 0;
	cout << "brute_force result: " << travllingSalesmanProblem(map, s) << endl;
    return 0;
}


