#include <string>
#include <algorithm>
#define _USE_MATH_DEFINES
#include <math.h>
#include <stdio.h>
#include <vector>
#include <bitset>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string.h>
#include <stdlib.h>
#include <random>
using namespace std;


#include <bits/stdc++.h>
#include <thrust/scan.h>
#include <thrust/device_ptr.h>
#include <thrust/device_malloc.h>
#include <thrust/device_free.h>

#include <cuda.h>
#include <curand.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <driver_functions.h>
// #include "chromosome.h"

#define BLOCK_DIM 32 // current GPU supports at most 1024 threads in one block. Hence 32*32=1024
#define BLOCK_SIZE ( BLOCK_DIM * BLOCK_DIM )

#define GEN_THRES 5000
#define POPULATION_SIZE 50000
#define MATING_SIZE  100
#define MUTATE_PROB 0.5
// #define SELECTION_SIZE 50
#define TOLERANCE 5

#define NUM_CITIES 38
string filename = "../testcase/dj38.txt";
__constant__ int cudaMap[NUM_CITIES][NUM_CITIES];

// #define PAR_SIZE_2

struct Chromosome{
    int fitness;
    int gnome_vec[NUM_CITIES+1];
};

__global__ void setup_curand(curandState *state){
    int thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (thread_idx >= POPULATION_SIZE)
        return;
    /* Each thread gets same seed, a different sequence
       number, no offset */
    curand_init(1233, thread_idx, 0, &state[thread_idx]);
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
    int fitness = single_fitness(cudaPop[thread_idx].gnome_vec);
    cudaPop[thread_idx].fitness = fitness;
}

__device__ Chromosome single_select(Chromosome* cudaPop, curandState localState){
    int rand_idx, cur_fitness;
    int min_fitness = INT_MAX;
    Chromosome cur_chro, best_chro;
    for (int i = 0; i < MATING_SIZE; i++){
        rand_idx = curand_uniform(&localState) * (POPULATION_SIZE - 1);
        cur_chro = cudaPop[rand_idx];
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
    cudaParents[thread_idx] = single_select(cudaPop, localState);
    #ifdef PAR_SIZE_2
    cudaParents[thread_idx * 2 + 1] = single_select(cudaPop, localState);
    #endif
}

__device__ int get_index(int* v, int val, int length) {
    for (int i = 0; i < length; i++){
        if (v[i] == val){
            return i;
        }
    }
    return -1;
}

__global__ void crossover_single(Chromosome* cudaPop, Chromosome* cudaParents, int i){
    int thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (thread_idx >= POPULATION_SIZE)
        return;
    Chromosome parent1, parent2, child;
    parent1 = cudaParents[thread_idx * 2];
    parent2 = cudaParents[thread_idx * 2 + 1];
    child = cudaPop[thread_idx];

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
    cudaPop[thread_idx].gnome_vec[i] = p1_fitness <= p2_fitness ? p1_cand : p2_cand;
}

__global__ void crossover(Chromosome* cudaPop, Chromosome* cudaParents, curandState *state){
    int thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (thread_idx >= POPULATION_SIZE)
        return;

    // version 3 - keep cudaParents = POPULATION_SIZE, add sync after selection, same result as seq below 20
    curandState localState = state[thread_idx];
    Chromosome parent1, parent2, child;
    // int pidx1 = curand_uniform(&localState) * (POPULATION_SIZE-1);
    // int pidx2 = curand_uniform(&localState) * (POPULATION_SIZE-1);
    // parent1 = cudaParents[pidx1];
    // parent2 = cudaParents[pidx2];

    // version 2
    // int thread_begin = blockIdx.x * blockDim.x;
    // int tmp = blockIdx.x * blockDim.x + BLOCK_SIZE;
    // int thread_end = tmp < POPULATION_SIZE-1 ? tmp: POPULATION_SIZE-1;
    // curandState localState = state[thread_idx];
    // int pidx = curand_uniform(&localState) * (thread_end - thread_begin);

    // version 1
    // Chromosome parent1, parent2, child;
    // parent1 = cudaParents[thread_idx * 2];
    // parent2 = cudaParents[thread_idx * 2 + 1];
    // child = cudaPop[thread_idx];

    for (int i = 1; i < NUM_CITIES; ++i) {
        // version 4
        int pidx1 = curand_uniform(&localState) * (POPULATION_SIZE-1);
        int pidx2 = curand_uniform(&localState) * (POPULATION_SIZE-1);
        parent1 = cudaParents[pidx1];
        parent2 = cudaParents[pidx2];

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
}

__global__ void mutation(Chromosome* cudaPop, curandState* state){
    int thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (thread_idx >= POPULATION_SIZE)
        return;
    curandState localState = state[thread_idx];
    float prob = curand_uniform(&localState);
    if (prob > MUTATE_PROB){
        while (true) {
        int idx1 = curand_uniform(&localState) * (NUM_CITIES - 2)+1;
        int idx2 = curand_uniform(&localState) * (NUM_CITIES - 2)+1;
        if (idx1 != idx2) {
            int tmp = cudaPop[thread_idx].gnome_vec[idx1];
            cudaPop[thread_idx].gnome_vec[idx1] = cudaPop[thread_idx].gnome_vec[idx2];
            cudaPop[thread_idx].gnome_vec[idx2] = tmp;
            break;
        }
    }
    }
}

__global__ void mutation_invert(Chromosome* cudaPop, curandState* state){
    int thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (thread_idx >= POPULATION_SIZE)
        return;
    
    curandState localState = state[thread_idx];
    float prob = curand_uniform(&localState);
    if (prob > MUTATE_PROB){
        for (int i = 0; i < 3; i++){
            int idx1 = curand_uniform(&localState) * (NUM_CITIES - 2)+1;
            int idx2 = curand_uniform(&localState) * (NUM_CITIES - 2)+1;
            while (idx1 == idx2){
                idx2 = curand_uniform(&localState) * (NUM_CITIES - 2)+1;
            }
            if (idx1 > idx2) { // swap
                int tmp = idx1;
                idx1 = idx2;
                idx2 = tmp;
            }
            while (idx1 < idx2) {
                int tmp = cudaPop[thread_idx].gnome_vec[idx1];
                cudaPop[thread_idx].gnome_vec[idx1] = cudaPop[thread_idx].gnome_vec[idx2];
                cudaPop[thread_idx].gnome_vec[idx2] = tmp;
                idx1++;
                idx2--;
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
            map[line_num][i] = val;
        }
        line_num += 1;
    }
    fp.close();
    return map;
}

int main(){
    // host var
    // read map
    ifstream fp(filename);
    string line;
    getline(fp, line);
    size_t dim = (int)atoi(line.c_str());
    int map[dim][dim];
    int line_num = 0;
    while (getline(fp, line)) {
        stringstream sstream(line);
        string str;
        for(int i = 0; i < dim; ++i) {
            getline(sstream, str, ' ');
            int val = atoi(str.c_str());
            val = val == -1 ? INT_MAX : val;
            map[line_num][i] = val;
        }
        line_num += 1;
    }
    fp.close();
    
    // init population
    Chromosome population[POPULATION_SIZE];
    for (int i = 0; i < POPULATION_SIZE; ++i) {
        population[i].gnome_vec[0] = 0;
        population[i].gnome_vec[NUM_CITIES] = 0;
        for (int j = 1; j < NUM_CITIES; j++){
            population[i].gnome_vec[j] = j;
        }
        shuffle(&population[i].gnome_vec[1],&population[i].gnome_vec[NUM_CITIES-1], default_random_engine(i));
    }

    //cuda var
    Chromosome* cudaPop;
    Chromosome* cudaParents;
    curandState* cudaStates;

    cudaMalloc(&cudaPop, sizeof(Chromosome) * POPULATION_SIZE);
    cudaMalloc(&cudaStates, POPULATION_SIZE * sizeof(curandState));
    #ifdef PAR_SIZE_2
    cudaMalloc(&cudaParents, sizeof(Chromosome) * POPULATION_SIZE*2);
    #else
    cudaMalloc(&cudaParents, sizeof(Chromosome) * POPULATION_SIZE);
    #endif
    

    cudaMemcpy(cudaPop, population, sizeof(Chromosome) * POPULATION_SIZE, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(cudaMap, map, sizeof(int) * NUM_CITIES * NUM_CITIES, 0, cudaMemcpyHostToDevice);
    
    int* best_seen_chro = population[0].gnome_vec;
    int best_seen_fitness = INT_MAX;
    // int prev_best_fitness = INT_MAX;
    // int tol = 0;

    int num_blocks = (POPULATION_SIZE + BLOCK_SIZE - 1) / BLOCK_SIZE;
    cout<<"all init done"<<endl;
    
    setup_curand<<<num_blocks, BLOCK_SIZE>>>(cudaStates);
    fitness<<<num_blocks, BLOCK_SIZE>>>(cudaPop);
    for (int i = 0; i < GEN_THRES; i++){
        selection<<<num_blocks, BLOCK_SIZE>>>(cudaPop, cudaParents, cudaStates);
        cudaDeviceSynchronize();
        crossover<<<num_blocks, BLOCK_SIZE>>>(cudaPop, cudaParents, cudaStates);
        // for (int c = 0; c < NUM_CITIES; c++){
        //     crossover_single<<<num_blocks, BLOCK_SIZE>>>(cudaPop, cudaParents, c);
        // }
        mutation<<<num_blocks, BLOCK_SIZE>>>(cudaPop, cudaStates);
        fitness<<<num_blocks, BLOCK_SIZE>>>(cudaPop);

        if (i%50 == 0){
            cout << "iter " << i <<endl;
            cudaDeviceSynchronize();
            cudaMemcpy(&population, cudaPop, sizeof(Chromosome) * POPULATION_SIZE, cudaMemcpyDeviceToHost);
            for (int j = 0; j < POPULATION_SIZE; j++){
                if (population[j].fitness < best_seen_fitness){
                    best_seen_chro = population[j].gnome_vec;
                    best_seen_fitness = population[j].fitness;
                }
            }
            for (int c = 0; c < NUM_CITIES+1; c++){
                cout<< best_seen_chro[c] << "->";
            }
            cout << "\t" << best_seen_fitness <<endl;
        }
        
    }
    printf("FINAL----------------------------------------\n");
    cudaMemcpy(&population, cudaPop, sizeof(Chromosome) * POPULATION_SIZE, cudaMemcpyDeviceToHost);
    for (int j = 0; j < POPULATION_SIZE; j++){
        if (population[j].fitness < best_seen_fitness){
            best_seen_chro = population[j].gnome_vec;
            best_seen_fitness = population[j].fitness;
        }
    }
    for (int c = 0; c < NUM_CITIES+1; c++){
        cout<< best_seen_chro[c] << "->";
    }
    cout << "\t" << best_seen_fitness <<endl;

}