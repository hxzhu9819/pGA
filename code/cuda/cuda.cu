#include <string>
#include <algorithm>
#define _USE_MATH_DEFINES
#include <math.h>
#include <stdio.h>
#include <vector>
#include <curand_kernel.h>
#include <bits/stdc++.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>
// #include "chromosome.h"

#define BLOCK_DIM 32 // current GPU supports at most 1024 threads in one block. Hence 32*32=1024
#define BLOCK_SIZE ( BLOCK_DIM * BLOCK_DIM )

#define GEN_THRES 100
#define POPULATION_SIZE 50000
#define MATING_SIZE  (POPULATION_SIZE / 100)
// #define SELECTION_SIZE 50
#define TOLERANCE 5
#define MUTATE_PROB 0.3
inline int NUM_CITIES;
__constant__ int cudaMap[NUM_CITIES][NUM_CITIES];

struct Chromosome{
    int fitness;
    int* gnome_vec;
}

__global__ void setup_curand(curandState *state)
{
    int thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (thread_idx >= POPULATION_SIZE)
        return;
    /* Each thread gets same seed, a different sequence
       number, no offset */
    curand_init(1234, thread_idx, 0, &state[thread_idx]);
}

__device__ int single_fitness(int* gnome){
    int f = 0;
    for (size_t i = 0; i < NUM_CITIES; i++) {
        if (cudaMap[gnome[i]][gnome[i + 1]] == INT_MAX) {
            return INT_MAX;
        }
        f += cudaMap[gnome[i]][gnome[i + 1]];
    }
    return f;

}

__global__ void fitness(Chromosome* cudaPop){
    int thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (thread_idx >= POPULATION_SIZE)
        return;

    Chromosome chro = cudaPop[thread_idx];
    int gnome[NUM_CITIES];
    gnome = chro.gnome_vec;
    chro.fitness = single_fitness(gnome);
}

__device__ Chromosome single_select(Chromosome* cudaPop, curandState localState){
    int rand_idx, cur_fitness;
    int min_fitness = INT_MAX;
    Chromosome cur_chro, best_chro;
    for (int i = 0; i < MATING_SIZE; i++){
        rand_idx = curand_uniform(localState) * (POPULATION_SIZE - 1);
        cur_chro = cudaPop[rand_idx];
        // cur_fitness = single_fitness(cur_chro.get_gnome_vec());
        cur_fitness = cur_chro.fitness;
        if (cur_fitness < min_fitness) {
            best_chro = cur_chro;
            min_fitness = cur_fitness;
        }
    }
    return best_chro;
}

__global__ void selection(Chromosome* cudaPop, Chromosome* cudaParents, curandState *state){
    int thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (thread_idx >= POPULATION_SIZE)
        return;
    curandState localState = state[thread_idx];
    cudaParents[thread_idx * 2] = single_select(cudaPop, localState);
    cudaParents[thread_idx * 2 + 1] = single_select(cudaPop, localState);
}

__device__ int get_index(int* v, int val, int length) {
    for (int i = 0; i < length; i++){
        if (v[i] == val){
            return i;
        }
    }
    return -1;
}

__global__ void crossover(Chromosome* cudaPop, Chromosome* cudaParents){
    int thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (thread_idx >= POPULATION_SIZE)
        return;
    // int* parent1_gnome_vec, parent2_gnome_vec, child_gnome_vec;
    // parent1_gnome_vec = cudaParents[thread_idx * 2].gnome_vec;
    // parent2_gnome_vec = cudaParents[thread_idx * 2 + 1].gnome_vec;
    // child_gnome_vec = cudaPop[thread_idx].gnome_vec;
    // int size = child_gnome_vec.size();

    Chromosome parent1, parent2, child;
    parent1 = cudaParents[thread_idx * 2];
    parent2 = cudaParents[thread_idx * 2 + 1];
    child = cudaPop[thread_idx];

    for (int i = 1; i < NUM_CITIES; ++i) {
        int gnome_from_p1, gnome_from_p2;
        int p1_cand_idx = get_index(parent1.gnome_vec, child.gnome_vec[i-1], NUM_CITIES)+1;
        int p1_cand = parent1.gnome_vec[p1_cand_idx];
        if (p1_cand == 0 || get_index(child.gnome_vec, p1_cand, i) != -1) {
            for (int j = 1; j < NUM_CITIES; ++j) {
                if (get_index(child.gnome_vec, j, i) == -1) {
                    p1_cand = j;
                    break;
                }
            }
        }
        // p2
        int p2_cand_idx = get_index(parent2.gnome_vec, child.gnome_vec[i-1], NUM_CITIES)+1;
        int p2_cand = parent2.gnome_vec[p2_cand_idx];
        if (p2_cand == 0 || get_index(child.gnome_vec, p2_cand, i) != -1) {
            for (int j = 1; j < NUM_CITIES; ++j) {
                if (get_index(child.gnome_vec, j, i) == -1) {
                    p2_cand = j;
                    break;
                }
            }
        }
        // compete
        int p1_fitness = cudaMap[child.gnome_vec[i-1]][p1_cand];
        int p2_fitness = cudaMap[child.gnome_vec[i-1]][p2_cand];
        child.gnome_vec[i] = p1_fitness <= p2_fitness ? p1_cand : p2_cand;
    }
    cudaPop[thread_idx] = child;
    // cudaPop[thread_idx].fitness = single_fitness(child_gnome_vec);
}

__global__ void mutation(vector<Chromosome> cudaPop, curandState *state){
    int thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (thread_idx >= POPULATION_SIZE)
        return;
    curandState localState = state[thread_idx];
    float prob = curand_uniform(localState);
    if (prob > MUTATE_PROB){
        while (true) {
        int idx1 = curand_uniform(localState) * (NUM_CITIES - 1);
        int idx2 = curand_uniform(localState) * (NUM_CITIES - 1);
        if (idx1 != idx2) {
            int tmp = cudaPop[thread_idx].gnome_vec[idx1];
            cudaPop[thread_idx].gnome_vec[idx1] = cudaPop[thread_idx].gnome_vec[idx2];
            cudaPop[thread_idx].gnome_vec[idx2] = tmp;
            break;
        }
    }
    }
}

int** load_tsp_from_file(string filename) {
    ifstream fp(filename);
    string line;
    getline(fp, line);
    size_t dim = (int)atoi(line.c_str());
    int** map = new int*[dim];
    int line_num = 0;
    while (getline(fp, line)) {
        vector<int> curr_line;
        map[line_num] = new int[dim];
        stringstream sstream(line);
        string str;
        for(int i = 0; i < dim; ++i) {
            getline(sstream, str, ' ');
            int val = atoi(str.c_str());
            val = val == -1 ? INT_MAX : val;
            map[line_num][i] = val
            // curr_line.push_back(val);
        }
        // map.push_back(curr_line);
        line_num += 1;
    }
    fp.close();
    return map;
}

int main(){
    // host var
    int** map = load_tsp_from_file("../testcase/5.txt");
    NUM_CITIES = sizeof(map) / sizeof(map[0]);
    
    Chromosome population[POPULATION_SIZE];
    for (int i = 0; i < POPULATION_SIZE; ++i) {
        int* route = new int[NUM_CITIES+1];
        route[0] = 0;
        route[NUM_CITIES] = 0;
        for (int j = 1; j < NUM_CITIES; j++){
            route[j] = j;
        }
        shuffle(&route[1],&route[NUM_CITIES-1], default_random_engine(i));
        population[i].gnome_vec = route;
    }
    // Chromosome population[POPULATION_SIZE];
    // for (int i = 0; i < POPULATION_SIZE; ++i) {
    //     Chromosome chro = Chromosome(NUM_CITIES);
    //     chro.create();
    //     population[i] = chro;
    // }
   

    //cuda var
    Chromosome* cudaPop;
    Chromosome* cudaParents;
    curandState *cudaStates;

    cudaMalloc(&cudaPop, sizeof(Chromosome) * NUM_CITIES * POPULATION_SIZE);
    cudaMalloc(&cudaParents, sizeof(Chromosome) * NUM_CITIES * POPULATION_SIZE * 2);
    cudaMalloc(&cudaStates, POP_SIZE * sizeof(curandState));

    cudaMemcpy(cudaPop, population, sizeof(Chromosome) * NUM_CITIES * POPULATION_SIZE, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(cudaMap, map, sizeof(int) * NUM_CITIES * NUM_CITIES);

    Chromosome best_seen_chro = cudaPop[0];
    int best_seen_fitness = INT_MAX;
    int prev_best_fitness = INT_MAX;
    int tol = 0;

    int num_blocks = (POPULATION_SIZE + BLOCK_SIZE - 1) / BLOCK_SIZE;
    setup_curand<<num_blocks, BLOCK_SIZE>>(cudaStates);
    fitness<<num_blocks, BLOCK_SIZE>>(cudaPop, best_seen_fitness);

    for (int i = 0; i < GEN_THRES; i++){
        selection<<num_blocks, BLOCK_SIZE>>(cudaPop, cudaParents, cudaStates);
        crossover<<num_blocks, BLOCK_SIZE>>(cudaPop, cudaParents);
        mutation<<num_blocks, BLOCK_SIZE>>(cudaPop, cudaStates);
        fitness<<num_blocks, BLOCK_SIZE>>(cudaPop, best_seen_fitness);

        for (int j = 0; j < POPULATION_SIZE; j++){
            if (cudaPop[j].fitness < best_seen_fitness){
                best_seen_chro = cudaPop[j];
                best_seen_fitness = cudaPop[j].fitness;
            }
        }
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

}