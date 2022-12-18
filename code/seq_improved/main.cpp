// Sequential Version Genetic Algorithm for Travelling Salesman Problem
// Author: Haoxuan Zhu (haoxuanz@andrew.cmu.edu)
// Date: Nov 15, 2022
#include <limits.h>
#include <vector>
#include <cmath>
#include <random>
#include <cmath>
#include <bits/stdc++.h>

#include <omp.h>
#include <chrono>
#include "brute_force.h"

using namespace std;
using namespace std::chrono;

#define GEN_THRES 2000
#define POPULATION_SIZE 50000
#define MATING_SIZE 1000 //(POPULATION_SIZE / 100)
#define TOURNAMENT_SIZE 8
#define TESTFILE "../testcase/11.txt"
#define NUM_CITY 11
#define TOLERANCE 30

// #define DEBUG

typedef struct {
    int* gnome;
    float fitness;
} Chro;

vector<vector<int>> dist_matrix;
Chro best_chro;
int best_fitness = INT_MAX;

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
    dist_matrix = map;
    return map;
}

int uniform_rand(int start, int end) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(start, end-1);
    int res = dis(gen);
    return res;
}

int rand_num_from_range(int start, int end) {
    return start + rand() % (end - start);
}

void print_chro(Chro* chro) {
    for (int i = 0; i < NUM_CITY; ++i) {
        cout << chro->gnome[i] << "->";
    }
    cout << "\t";
    cout << "fitness: " << chro->fitness << endl;
    return;
}

void print_all_chro(Chro* pop, int pop_size) {
    cout << "---\n";
    for (int i = 0; i < pop_size; ++i) {
        print_chro(&pop[i]);
    }
    cout << "---" << endl;
    return;
}

void calculate_fitness(Chro* chro, vector<vector<int>>& dist_matrix) {
    float f = 0;

    for (int i = 0; i < NUM_CITY-1; i++) {
        f += dist_matrix[chro->gnome[i]][chro->gnome[i+1]];
    }
    f += dist_matrix[chro->gnome[NUM_CITY-1]][chro->gnome[0]];
    chro->fitness = f;
}

void calculate_pop_fitness(Chro* pop, vector<vector<int>>& dist_matrix) {
    // #pragma omp parallel for schedule(dynamic) shared(dist_matrix, pop)
    for (int p = 0; p < POPULATION_SIZE; ++p) {
        calculate_fitness(&pop[p], dist_matrix);
        if (pop[p].fitness < best_fitness) {
            best_chro = pop[p];
            best_fitness = pop[p].fitness;
        }
    }
}

void create_random_chro(Chro* chro, int num_city, vector<vector<int>>& dist_matrix) {
    int pool[num_city];
    chro->gnome = (int*)malloc(num_city * sizeof(int));
    for (int i = 0; i < num_city; ++i) {
        pool[i] = num_city-1-i;
    }
    chro->gnome[0] = 0;
    // #pragma omp parallel for schedule(dynamic)
    for (int i = 1; i < num_city; ++i) {
        int vic_idx = uniform_rand(0, num_city-i);
        int victim = pool[vic_idx];
        pool[vic_idx] = pool[num_city-1-i];
        chro->gnome[i] = victim;
    }
    calculate_fitness(chro, dist_matrix);
}


void selection_tournament(Chro* pop, Chro* mating_pop, int mating_size) {
    int tournament_size = TOURNAMENT_SIZE;
    // #pragma omp parallel for schedule(auto) shared(mating_pop, pop)
    for (int i = 0; i < mating_size; ++i) {
        Chro best_chro = pop[0];
        int best_fitness = INT_MAX;

        for (int j = 0; j < tournament_size; ++j) {
            int victim = 0 + rand() % (POPULATION_SIZE);
            if (pop[victim].fitness < best_fitness) {
                best_fitness = pop[victim].fitness;
                best_chro = pop[victim];
            }
        }
        mating_pop[i] = best_chro;

    }
    return;
}

int get_index(int* v, int val) {
    for (int i = 0; i < NUM_CITY; ++i) {
        if (v[i] == val) {
            return i;
        }
    }
    return -1;
}

void mutate(Chro* chro) {
    int victim_gene1_idx = uniform_rand(1, NUM_CITY);
    int victim_gene2_idx = uniform_rand(1, NUM_CITY);
    while (victim_gene2_idx == victim_gene1_idx) {
        victim_gene2_idx = uniform_rand(1, NUM_CITY);
    }

    if (victim_gene1_idx > victim_gene2_idx) { // swap
        int tmp = victim_gene1_idx;
        victim_gene1_idx = victim_gene2_idx;
        victim_gene2_idx = tmp;
    }
    while (victim_gene1_idx < victim_gene2_idx) {
        int tmp = chro->gnome[victim_gene1_idx];
        chro->gnome[victim_gene1_idx] = chro->gnome[victim_gene2_idx];
        chro->gnome[victim_gene2_idx] = tmp;
        victim_gene1_idx++;
        victim_gene2_idx--;
    }
}

void cross_over(Chro* pop, Chro* mating_pop, vector<vector<int>>& map) {
    // #pragma omp parallel for schedule(dynamic)
    for (int p = 0 ; p < POPULATION_SIZE; p++) {
        int idx1 = uniform_rand(0, MATING_SIZE);
        int idx2 = uniform_rand(0, MATING_SIZE);
        int* parent1_gnome = mating_pop[idx1].gnome;
        int* parent2_gnome = mating_pop[idx2].gnome;
        int new_chro_gnome[NUM_CITY] = {0};
        for (int i = 1; i < NUM_CITY; ++i) {
            int gnome_from_p1, gnome_from_p2;
            int p1_cand_idx = get_index(parent1_gnome, new_chro_gnome[i-1])+1;
            int p1_cand = (p1_cand_idx >= NUM_CITY-1) ? 0 : parent1_gnome[p1_cand_idx];
            if (p1_cand == 0 || get_index(new_chro_gnome, p1_cand) != -1) {
                for (int j = 1; j < NUM_CITY; ++j) {
                    if (get_index(new_chro_gnome, j) == -1) {
                        p1_cand = j;
                        break;
                    }
                }
            }
            // p2
            int p2_cand_idx = get_index(parent2_gnome, new_chro_gnome[i-1])+1;
            int p2_cand = (p2_cand_idx >= NUM_CITY-1) ? 0 : parent2_gnome[p2_cand_idx];
            if (p2_cand == 0 || get_index(new_chro_gnome, p2_cand) != -1) {
                for (int j = 1; j < NUM_CITY; ++j) {
                    if (get_index(new_chro_gnome, j) == -1) {
                        p2_cand = j;
                        break;
                    }
                }
            }
            // compete
            int p1_fitness = map[new_chro_gnome[i-1]][p1_cand];
            int p2_fitness = map[new_chro_gnome[i-1]][p2_cand];
            if (p1_cand == 0 || p2_cand == 0) {
                cout << "wandanle" << p1_cand_idx << " " << parent1_gnome[p1_cand_idx] << " " << p2_cand_idx << " " << parent2_gnome[p2_cand_idx] << endl;
                exit(-1);
            }
            new_chro_gnome[i] = p1_fitness <= p2_fitness ? p1_cand : p2_cand;
            pop[p].gnome[i] = new_chro_gnome[i];
        }

        if (uniform_rand(0, 10) < 4) {
            mutate(&pop[p]);
        }
    }
}


void print_eval_avg_fitness(Chro* pop, int pop_size) {
    float avg = 0;
    for (int i = 0; i < pop_size; ++i) {
        avg += pop[i].fitness;
    }
    cout << "avg fitness of pop = " << avg/pop_size << endl;
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

    Chro* pop = (Chro*)malloc(POPULATION_SIZE*sizeof(Chro));
    Chro* mating_pop = (Chro*) malloc(MATING_SIZE*sizeof(Chro));

    auto start_time = high_resolution_clock::now();
    // #pragma omp parallel for schedule(guided)
    for (int i = 0; i < POPULATION_SIZE; ++i) {
        // cout << "Generating " << i << " chro" << endl;
        create_random_chro(&pop[i], NUM_CITY, map);
    }
    auto end_time = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(end_time - start_time);
    cout << "Time taken by Initialization: "
         << duration.count() << " microseconds" << endl;

    int gen = 1;
    for (; gen < GEN_THRES; gen++) {
        cout <<"GEN " << gen << endl;
        // cout << "after initial generate" << endl;
        // print_all_chro(pop, POPULATION_SIZE);

        auto start_time_0 = high_resolution_clock::now();
        selection_tournament(pop, mating_pop, MATING_SIZE);
        // cout << "after selection generate" << endl;
        // print_all_chro(mating_pop, MATING_SIZE);
        end_time = high_resolution_clock::now();
        duration = duration_cast<microseconds>(end_time - start_time_0);
        cout << "Time taken by SELECTION: "
            << duration.count() << " microseconds" << endl;

        start_time = high_resolution_clock::now();
        cross_over(pop, mating_pop, dist_matrix);
        // cout << "after crossover generate" << endl;
        // print_all_chro(pop, POPULATION_SIZE);
        end_time = high_resolution_clock::now();
        duration = duration_cast<microseconds>(end_time - start_time);
        cout << "Time taken by CROSSOVER: "
            << duration.count() << " microseconds" << endl;

        start_time = high_resolution_clock::now();
        calculate_pop_fitness(pop, dist_matrix);
        // cout << "after updating fitness" << endl;
        // print_all_chro(pop, POPULATION_SIZE);
        end_time = high_resolution_clock::now();
        duration = duration_cast<microseconds>(end_time - start_time);
        cout << "Time taken by EVAL: "
            << duration.count() << " microseconds" << endl;

        duration = duration_cast<microseconds>(end_time - start_time_0);
        cout << "Total time taken for an iteration: "
            << duration.count() << " microseconds" << endl;
        
        print_eval_avg_fitness(pop, POPULATION_SIZE);
        print_chro(&best_chro);
    }

    return 0;
}


