// Sequential Version Genetic Algorithm for Travelling Salesman Problem
// Author: Haoxuan Zhu (haoxuanz@andrew.cmu.edu)
// Date: Nov 15, 2022
#include <limits.h>
#include "chromosome.h"
#include <vector>
#include <cmath>
#include <random>
#include <chrono>

#include "brute_force.h"

using namespace std;
using namespace std::chrono;

#define GEN_THRES 2000
#define POPULATION_SIZE 50000
#define TOURNAMENT_SIZE 8
#define MATING_SIZE  1000 //(POPULATION_SIZE / 10)
#define TOLERANCE 2000
#define TESTFILE "../testcase/dj38.txt"
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

int cal_fitness(vector<vector<int>>& map, vector<int> gnome)
{   
    int f = 0;
    for (int i = 0; i < gnome.size() - 1; i++) {
        if (map[gnome[i]][gnome[i + 1]] == INT_MAX) {
            return INT_MAX;
        }
        f += map[gnome[i]][gnome[i + 1]];
    }
    return f;
}

int uniform_rand(int start, int end) {
    // std::default_random_engine generator;
    // std::uniform_int_distribution<int> distribution(start,end-1);
    // return distribution(generator);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(start, end-1);
    int res = dis(gen);
    return res;
}

void cross_over(vector<Chromosome>& mating_pop, Chromosome& chro1, vector<vector<int>>& map) {
    int idx1 = uniform_rand(0, mating_pop.size());
    int idx2 = uniform_rand(0, mating_pop.size());
    int pos = uniform_rand(1, chro1.size);
    chro1.cross_over(mating_pop[idx1], mating_pop[idx2], pos, map);
}

void mutation(Chromosome& chro) {
    chro.mutate();
}

vector<Chromosome> selection_tournament(vector<Chromosome>& pop, int mating_size) {
    vector<Chromosome> mating_pop;
    int tournament_size = TOURNAMENT_SIZE;
    for (int i = 0; i < mating_size; ++i) {
        Chromosome best_chro = pop[0];
        int best_fitness = INT_MAX;

        for (int j = 0; j < tournament_size; ++j) {
            // int victim = uniform_rand(0, pop.size());
            int victim = 0 + rand() % (pop.size());
            if (pop[victim].fitness < best_fitness) {
                best_fitness = pop[victim].fitness;
                best_chro = pop[victim];
            }
        }
        mating_pop.push_back(best_chro);
    }
    
    return mating_pop;
}

vector<Chromosome> selection(vector<Chromosome>& pop, int mating_size) {
    vector<Chromosome> mating_pop;
    int fitness_min = INT_MAX;
    int fitness_max = -INT_MAX;
    int fitness_sum = 0;
    int fitness_avg = 0;
    for (int i = 0; i < pop.size(); ++i) {
        fitness_min = fitness_min < pop[i].fitness ? fitness_min : pop[i].fitness;
        fitness_max = fitness_max > pop[i].fitness ? fitness_max : pop[i].fitness;
        fitness_sum += pop[i].fitness;
    }
    fitness_avg = fitness_sum / pop.size();
    int fitness_thres = 0.25*fitness_avg+1;

    for (int j = 0; j < pop.size(); j++) {
        if (fitness_avg > pop[j].fitness) {
            mating_pop.push_back(pop[j]);
        }
        if (mating_pop.size() == mating_size) {
            break;
        }
    }

    int cj = pop.size()-1;
    while (mating_pop.size() < mating_size) {
        mating_pop.push_back(pop[cj--]);
    }
    
    return mating_pop;
}

vector<Chromosome> selection_by_sort(vector<Chromosome>& population) { // Abandoned
    vector<Chromosome> mating_pop;
    sort(population.begin(), population.end(), worsethan);
    for (int i = 0; i < MATING_SIZE; ++i) {
        mating_pop.push_back(population[i]);
    }
    return mating_pop;
}

void solver(vector<vector<int>> map, int gen_thres, int pop_size) {
    int gen = 1;
    int num_node = map.size();
    vector<Chromosome> population;
    auto start_time = high_resolution_clock::now();
    for (int i = 0; i < pop_size; ++i) {
        Chromosome chro = Chromosome(num_node);
        chro.create();
        chro.fitness = cal_fitness(map, chro.get_gnome_vec());
        population.push_back(chro);
    }
    auto end_time = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(end_time - start_time);
    cout << "Time taken by Initialization: "
         << duration.count() << " microseconds" << endl;

    Chromosome best_seen_chro = population[0];
    int best_seen_fitness = INT_MAX;
    int prev_best_fitness = INT_MAX;
    int tol = 0;

    #ifdef DEBUG
    cout << "\nInitial population: " << endl << "GNOME\tFITNESS VALUE\n";
    for (int i = 0; i < pop_size; i++)
        cout << population[i].get_gnome() << "\t" << population[i].fitness << endl;
    cout << "\n";
    #endif

    cout << "Generation " << gen << " \n";
    while (gen <= gen_thres) {
        // selection
        auto start_time_0 = high_resolution_clock::now();
        vector<Chromosome> mating_pop;
        // mating_pop = selection(population, MATING_SIZE);
        mating_pop = selection_tournament(population, MATING_SIZE);
        end_time = high_resolution_clock::now();
        duration = duration_cast<microseconds>(end_time - start_time_0);
        cout << "Time taken by SELECTION: "
            << duration.count() << " microseconds" << endl;
        // cout << "Mating Pop:" <<endl;
        // vector<int> cnter;
        // cnter.resize(num_node, 0);
        // for (auto c: mating_pop) {
        //     cnter[c.get_gnome_vec()[1]]++;
        // }
        // for (auto x: cnter) cout << x << " ";
        // cout <<endl;

        // cout << "Mating Pop:" <<endl;
        // for(auto x:mating_pop) {
        //     cout << x.get_gnome() << endl;
        // }
        // cout << "SELECTION done " << endl;

        // evoluation
        start_time = high_resolution_clock::now();
        vector<Chromosome> new_population;
        for (int i = 0; i < POPULATION_SIZE; i+=1) {
            // cross-over
            Chromosome chro1 = Chromosome(num_node);
            chro1.create();

            cross_over(mating_pop, chro1, map);
            #ifdef DEBUG
            cout << chro1.get_gnome() <<endl;
            cout << "CROSSOVER done " << endl;
            #endif

            // // mutation
            // if (uniform_rand(0, 10) < 4) {
            //     // cout << "MUTATION happened" <<endl;
            //     chro1.mutate_by_invert();
            // }
            new_population.push_back(chro1);
        }
        end_time = high_resolution_clock::now();
        duration = duration_cast<microseconds>(end_time - start_time);
        cout << "Time taken by CROSSOVER: "
            << duration.count() << " microseconds" << endl;

        start_time = high_resolution_clock::now();
        for (int i = 0; i < POPULATION_SIZE; i+=1) {
            if (uniform_rand(0, 10) < 4) {
                new_population[i].mutate_by_invert();
            }
        }
        end_time = high_resolution_clock::now();
        duration = duration_cast<microseconds>(end_time - start_time);
        cout << "Time taken by MUTATION: "
            << duration.count() << " microseconds" << endl;

        // evaluation
        start_time = high_resolution_clock::now();
        for (int i = 0; i < new_population.size(); i++) {
            new_population[i].fitness = cal_fitness(map, new_population[i].get_gnome_vec());
            if (new_population[i].fitness < best_seen_chro.fitness) {
                best_seen_chro = new_population[i];
                best_seen_fitness = new_population[i].fitness;
            }
        }
        end_time = high_resolution_clock::now();
        duration = duration_cast<microseconds>(end_time - start_time);
        cout << "Time taken by EVAL: "
            << duration.count() << " microseconds" << endl;

        duration = duration_cast<microseconds>(end_time - start_time_0);
        cout << "Total time taken for an iteration: "
            << duration.count() << " microseconds" << endl;
        
        // Chromosome mutated_best_seen_chro = best_seen_chro;
        // mutated_best_seen_chro.mutate_by_invert();
        // new_population.push_back(mutated_best_seen_chro);
        population = new_population;
        

        #ifdef DEBUG
        cout << "Generation " << gen << " \n";
        cout << "GNOME\tFITNESS VALUE\n";
        sort(population.begin(), population.end(), worsethan);
        for (int i = 0; i < pop_size; i++)
            cout << population[i].get_gnome() << "\t"
                << population[i].fitness << endl;
        #endif
        // sort(new_population.begin(), new_population.end(), worsethan);
        cout << "GEN:" << gen << ":best -> " << best_seen_chro.get_gnome() << "\t" << best_seen_chro.fitness << endl;
        
        gen++;
        if (tol >= TOLERANCE) {
            break;
        } else {
            if (prev_best_fitness == best_seen_fitness) {
                tol++;
            } else {
                tol = 0;
                prev_best_fitness = best_seen_fitness;
            }
        }
    }
    cout << "Final Generation " << gen << " \n";
    #ifdef DEBUG
    cout << "GNOME\tFITNESS VALUE\n";
    sort(population.begin(), population.end(), worsethan);
    for (int i = 0; i < pop_size; i++)
        cout << population[i].get_gnome() << "\t"
            << population[i].fitness << endl;
    #endif
    cout << "Result" << best_seen_chro.get_gnome() << "\t" << best_seen_chro.fitness << endl;
}

int main() {
    vector<vector<int>> map = load_tsp_from_file(TESTFILE);
    #ifdef DEBUG
    for(int i = 0; i < map.size(); ++i) {
        for (int j = 0; j < map[i].size(); ++j) {
            cout << map[i][j] << " ";
        }
        cout << endl;
    }
    #endif
    solver(map, GEN_THRES, POPULATION_SIZE);

    // // control group
    // string ans_str = "";
	// cout << "brute_force result: " << travllingSalesmanProblem(map, 0, ans_str) << " " << ans_str << endl;
    // cout << "greedy result: " << tsp_greedy(map) << endl; // not correct
    return 0;
}


