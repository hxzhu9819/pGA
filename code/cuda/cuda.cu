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
#include <chrono>
using namespace std;
using namespace std::chrono;


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
#include "CycleTimer.h"

// #define PAR_SIZE_2
// #define DEBUG
// #define PSEUDO_RAND
// #define XOVER_SINGLE

#define BLOCK_DIM 16 // current GPU supports at most 1024 threads in one block. 32*32=1024
#define BLOCK_SIZE ( BLOCK_DIM * BLOCK_DIM )

#define GEN_THRES 5000
#define POPULATION_SIZE 50000
#define MATING_SIZE  1000
#define MUTATE_PROB 0.6
#define MUTATE_TIMES 1
#define SELECTION_SIZE 8
#define TOLERANCE 500
#define CHECK_INTERVAL 5

#define NUM_CITIES 734
string filename = "../testcase/734.txt";
__device__ int cudaMap[NUM_CITIES * NUM_CITIES];

#define CUDA_CALL(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}


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
        if (cudaMap[gnome[i] * NUM_CITIES + gnome[i + 1]] == INT_MAX) {
            return INT_MAX;
        }
        f += cudaMap[gnome[i] * NUM_CITIES + gnome[i + 1]];
    }
    return f;

}

__global__ void fitness(Chromosome* cudaPop){
    int thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (thread_idx >= POPULATION_SIZE)
        return;
    int fitness = single_fitness(cudaPop[thread_idx].gnome_vec);
    cudaPop[thread_idx].fitness = fitness;
    #ifdef DEBUG
    if (thread_idx == 100){
        printf("fitness %d\n", cudaPop[thread_idx].fitness);
    }
    #endif
}

__device__ Chromosome single_select(Chromosome* cudaPop, curandState localState, int thread_idx){
    int rand_idx, cur_fitness;
    int min_fitness = INT_MAX;
    Chromosome cur_chro, best_chro;
    for (int i = 0; i < SELECTION_SIZE; i++){
        rand_idx = curand_uniform(&localState) * (POPULATION_SIZE - 1);
        #ifdef DEBUG
        if (thread_idx == 100){
            printf("TOURNAMENT - rand_idx %d\n", rand_idx);
        }
        #endif
        cur_chro = cudaPop[rand_idx];
        cur_fitness = cur_chro.fitness;
        if (cur_fitness < min_fitness) {
            best_chro = cur_chro;
            min_fitness = cur_fitness;
        }
    }
    return best_chro;
}

__global__ void selection(Chromosome* cudaPop, Chromosome* cudaParents, curandState *state, int seed){
    int thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (thread_idx >= MATING_SIZE)
        return;
    #ifdef PSEUDO_RAND
    curandState localState = state[thread_idx];
    #else
    curandState localState = state[(seed+thread_idx) % POPULATION_SIZE];
    #endif
    // printf("thread id %d, rand %d\n", thread_idx, (seed+thread_idx) % POPULATION_SIZE);
    cudaParents[thread_idx] = single_select(cudaPop, localState, thread_idx);
    // if (thread_idx == 100){
    //     printf("parent fitness %d\n", cudaParents[thread_idx].fitness);
    // }
    #ifdef PAR_SIZE_2
    cudaParents[thread_idx * 2 + 1] = single_select(cudaPop, localState, thread_idx);
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

__global__ void crossover_single(Chromosome* cudaPop, Chromosome* cudaParents, int i, curandState *state, int seed){
    int thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (thread_idx >= POPULATION_SIZE)
        return;
    Chromosome parent1, parent2, child;
    
    #ifdef PAR_SIZE_2
    parent1 = cudaParents[thread_idx * 2];
    parent2 = cudaParents[thread_idx * 2 + 1];
    #else
    #ifdef PSEUDO_RAND
    curandState localState = state[thread_idx];
    #else
    curandState localState = state[thread_idx];
    #endif
    int pidx1 = curand_uniform(&localState) * (MATING_SIZE-1);
    int pidx2 = curand_uniform(&localState) * (MATING_SIZE-1);
    parent1 = cudaParents[pidx1];
    parent2 = cudaParents[pidx2];
    // if (thread_idx == 100){
    //     printf("CROSSOVER_SINGLE -  %d, pidx1 %d fitness %d, pidx2 %d fitness %d\n", i, pidx1, cudaParents[pidx1].fitness, pidx2, cudaParents[pidx2].fitness);
    // }
    #endif
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
    int p1_fitness = cudaMap[child.gnome_vec[i-1] * NUM_CITIES + p1_cand];
    int p2_fitness = cudaMap[child.gnome_vec[i-1] * NUM_CITIES + p2_cand];
    cudaPop[thread_idx].gnome_vec[i] = p1_fitness <= p2_fitness ? p1_cand : p2_cand;
}

__global__ void crossover(Chromosome* cudaPop, Chromosome* cudaParents, curandState *state, int seed){
    int thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (thread_idx >= POPULATION_SIZE)
        return;

    // version 3 - keep cudaParents = POPULATION_SIZE, add sync after selection, same result as seq below 20
    #ifdef PSEUDO_RAND
    curandState localState = state[thread_idx];
    #else
    curandState localState = state[thread_idx];
    #endif    
    
    Chromosome parent1, parent2, child;
    #ifdef PAR_SIZE_2
    parent1 = cudaParents[thread_idx * 2];
    parent2 = cudaParents[thread_idx * 2 + 1];
    #else
    int pidx1 = curand_uniform(&localState) * (MATING_SIZE-1);
    int pidx2 = curand_uniform(&localState) * (MATING_SIZE-1);
    parent1 = cudaParents[pidx1];
    parent2 = cudaParents[pidx2];
    #endif
    // if (thread_idx == 100){
    //     printf("crossover, pidx1 %d fitness %d, pidx2 %d fitness %d\n", pidx1, cudaParents[pidx1].fitness, pidx2, cudaParents[pidx2].fitness);
    // }

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
    #ifdef DEBUG
    if (thread_idx == 100){
        printf("CROSSOVER - before crossover\n");
        for (int i = 0; i < NUM_CITIES; i++){
            printf("%d->", cudaPop[thread_idx].gnome_vec[i]);
        }
        printf("\n");
    }
    #endif
    child = cudaPop[thread_idx];
    int p1_cand_idx, p1_cand, p2_cand_idx, p2_cand, p1_fitness, p2_fitness;
    for (int i = 1; i < NUM_CITIES; ++i) {
        // version 4
        // pidx1 = curand_uniform(&localState) * (MATING_SIZE-1);
        // pidx2 = curand_uniform(&localState) * (MATING_SIZE-1);
        // if (thread_idx == 100){
        //     printf("crossover %d, pidx1 %d, pidx2 %d\n", i, pidx1, pidx2);
        // }
        // parent1 = cudaParents[pidx1];
        // parent2 = cudaParents[pidx2];

        p1_cand_idx = get_index(parent1.gnome_vec, child.gnome_vec[i-1], NUM_CITIES)+1;
        p1_cand = parent1.gnome_vec[p1_cand_idx];
        if (p1_cand == 0 || get_index(child.gnome_vec, p1_cand, i) != -1) {
            for (int j = 1; j < NUM_CITIES; ++j) {
                if (get_index(child.gnome_vec, j, i) == -1) {
                    p1_cand = j;
                    break;
                }
            }
        }
        // p2
        p2_cand_idx = get_index(parent2.gnome_vec, child.gnome_vec[i-1], NUM_CITIES)+1;
        p2_cand = parent2.gnome_vec[p2_cand_idx];
        if (p2_cand == 0 || get_index(child.gnome_vec, p2_cand, i) != -1) {
            for (int j = 1; j < NUM_CITIES; ++j) {
                if (get_index(child.gnome_vec, j, i) == -1) {
                    p2_cand = j;
                    break;
                }
            }
        }
        // compete
        p1_fitness = cudaMap[child.gnome_vec[i-1] * NUM_CITIES + p1_cand];
        p2_fitness = cudaMap[child.gnome_vec[i-1] * NUM_CITIES + p2_cand];
        child.gnome_vec[i] = p1_fitness <= p2_fitness ? p1_cand : p2_cand;
    }
    // for (int i = 0; i < NUM_CITIES; i++){
    //     cudaPop[thread_idx].gnome_vec[i] = child.gnome_vec[i];
    // }
    cudaPop[thread_idx] = child;
    #ifdef DEBUG
    if (thread_idx == 100){
        printf("CROSSOVER - done crossover\n");
        for (int i = 0; i < NUM_CITIES; i++){
            printf("%d->", cudaPop[thread_idx].gnome_vec[i]);
        }
        printf("\n");
    }
    #endif
}

__global__ void mutation(Chromosome* cudaPop, curandState* state, int seed){
    int thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (thread_idx >= POPULATION_SIZE)
        return;
    #ifdef PSEUDO_RAND
    curandState localState = state[thread_idx];
    #else
    curandState localState = state[(seed+thread_idx) % POPULATION_SIZE];
    #endif
    float prob = curand_uniform(&localState);
    #ifdef DEBUG
    if (thread_idx == 100){
        printf("MUTATION - fitness %d, prob %f\n", cudaPop[thread_idx].fitness, prob);
    }
    #endif
    if (prob > MUTATE_PROB){
        #ifdef DEBUG
        if (thread_idx == 100){
            printf("MUTATION - before mutation\n");
            for (int i = 0; i < NUM_CITIES; i++){
                printf("%d->", cudaPop[thread_idx].gnome_vec[i]);
            }
            printf("\n");
        }
        #endif
        for (int i = 0; i < MUTATE_TIMES; i++){
            
            while (true) {
                int idx1 = curand_uniform(&localState) * (NUM_CITIES - 2)+1;
                int idx2 = curand_uniform(&localState) * (NUM_CITIES - 2)+1;
                if (idx1 != idx2) {
                    int tmp = cudaPop[thread_idx].gnome_vec[idx1];
                    cudaPop[thread_idx].gnome_vec[idx1] = cudaPop[thread_idx].gnome_vec[idx2];
                    cudaPop[thread_idx].gnome_vec[idx2] = tmp;
                    // if (thread_idx == 100){
                    //     printf("idx1, idx2: %d, %d\n", idx1, idx2);
                    // }
                    break;
                }
                
            }
        }
        #ifdef DEBUG
        if (thread_idx == 100){
            printf("MUTATION - after mutation\n");
            for (int i = 0; i < NUM_CITIES; i++){
                printf("%d->", cudaPop[thread_idx].gnome_vec[i]);
            }
            printf("\n");
        }
        #endif
    }
}

__global__ void mutation_invert(Chromosome* cudaPop, curandState* state){
    int thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (thread_idx >= POPULATION_SIZE)
        return;
    
    curandState localState = state[cudaPop[thread_idx].fitness % thread_idx];
    float prob = curand_uniform(&localState);
    if (prob > MUTATE_PROB){
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

int uniform_rand(int start, int end) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(start, end-1);
    int res = dis(gen);
    return res;
}

void create_random_chro(Chromosome* chro) {
    int pool[NUM_CITIES];
    
    for (int i = 0; i < NUM_CITIES; ++i) {
        pool[i] = NUM_CITIES-1-i;
    }
    chro->gnome_vec[0] = 0;
    for (int i = 1; i < NUM_CITIES; ++i) {
        int vic_idx = uniform_rand(0, NUM_CITIES-i);
        int victim = pool[vic_idx];
        pool[vic_idx] = pool[NUM_CITIES-1-i];
        chro->gnome_vec[i] = victim;
    }
    chro->gnome_vec[NUM_CITIES] = 0;
}


int main(){
    // host var
    // read map
    ifstream fp(filename);
    string line;
    getline(fp, line);
    size_t dim = (int)atoi(line.c_str());
    int map[NUM_CITIES][NUM_CITIES];
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
    Chromosome* population = new Chromosome[POPULATION_SIZE];
    for (int i = 0; i < POPULATION_SIZE; ++i) {
        population[i].gnome_vec[0] = 0;
        population[i].gnome_vec[NUM_CITIES] = 0;
        for (int j = 1; j < NUM_CITIES; j++){
            population[i].gnome_vec[j] = j;
        }
        shuffle(&population[i].gnome_vec[1],&population[i].gnome_vec[NUM_CITIES-1], default_random_engine(std::time(0)));
    }

    // for (int c = 0; c < NUM_CITIES+1; c++){
    //     cout<< population[0].gnome_vec[c] << "->";
    // }
    // cout <<endl;

    // for (int c = 0; c < NUM_CITIES; c++){
    //     cout<< map[c][0] << "->";
    // }
    // cout <<endl;

    //cuda var
    Chromosome* cudaPop;
    Chromosome* cudaParents;
    curandState* cudaStates;

    CUDA_CALL(cudaMalloc(&cudaPop, sizeof(Chromosome) * POPULATION_SIZE));
    CUDA_CALL(cudaMalloc(&cudaStates, POPULATION_SIZE * sizeof(curandState)));
    #ifdef PAR_SIZE_2
    cudaMalloc(&cudaParents, sizeof(Chromosome) * POPULATION_SIZE*2);
    #else
    CUDA_CALL(cudaMalloc(&cudaParents, sizeof(Chromosome) * MATING_SIZE));
    #endif
    

    CUDA_CALL(cudaMemcpy(cudaPop, population, sizeof(Chromosome) * POPULATION_SIZE, cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpyToSymbol(cudaMap, map, sizeof(int) * NUM_CITIES * NUM_CITIES, 0, cudaMemcpyHostToDevice));
    
    int* best_seen_chro = population[0].gnome_vec;
    int best_seen_fitness = INT_MAX;
    int prev_best_fitness = INT_MAX;
    int tol = 0;

    int num_blocks = (POPULATION_SIZE + BLOCK_SIZE - 1) / BLOCK_SIZE;
    int num_blocks_selection = (MATING_SIZE + BLOCK_SIZE - 1) / BLOCK_SIZE;
    cout<<"all init done"<<endl;
    
    setup_curand<<<num_blocks, BLOCK_SIZE>>>(cudaStates);
    fitness<<<num_blocks, BLOCK_SIZE>>>(cudaPop);
    CUDA_CALL(cudaGetLastError());

    double sel_duration = 0;
    double xover_duration = 0;
    double mut_duration = 0;
    double eval_duration = 0;
    double cpu_duration = 0;
    int stop_iter = 0;
    for (int i = 0; i < GEN_THRES; i++){

        auto start_time = high_resolution_clock::now();
        #ifdef PAR_SIZE_2
        selection<<<num_blocks, BLOCK_SIZE>>>(cudaPop, cudaParents, cudaStates, i);
        CUDA_CALL(cudaGetLastError());
        #else
        selection<<<num_blocks_selection, BLOCK_SIZE>>>(cudaPop, cudaParents, cudaStates, i);
        CUDA_CALL(cudaGetLastError());
        #endif
        CUDA_CALL(cudaDeviceSynchronize());
        auto end_time = high_resolution_clock::now();
        auto duration = duration_cast<microseconds>(end_time - start_time);
        sel_duration += duration.count();

        start_time = high_resolution_clock::now();
        #ifdef XOVER_SINGLE
        for (int c = 1; c < NUM_CITIES; c++){
            crossover_single<<<num_blocks, BLOCK_SIZE>>>(cudaPop, cudaParents, c, cudaStates, i);
            CUDA_CALL(cudaGetLastError());
        }
        #else
        crossover<<<num_blocks, BLOCK_SIZE>>>(cudaPop, cudaParents, cudaStates, i);
        CUDA_CALL(cudaGetLastError());
        #endif
        CUDA_CALL(cudaDeviceSynchronize());
        end_time = high_resolution_clock::now();
        duration = duration_cast<microseconds>(end_time - start_time);
        xover_duration += duration.count();
        
        start_time = high_resolution_clock::now();
        mutation<<<num_blocks, BLOCK_SIZE>>>(cudaPop, cudaStates, i);
        CUDA_CALL(cudaGetLastError());
        CUDA_CALL(cudaDeviceSynchronize());
        end_time = high_resolution_clock::now();
        duration = duration_cast<microseconds>(end_time - start_time);
        mut_duration += duration.count();

        start_time = high_resolution_clock::now();
        fitness<<<num_blocks, BLOCK_SIZE>>>(cudaPop);
        CUDA_CALL(cudaGetLastError());
        CUDA_CALL(cudaDeviceSynchronize());
        end_time = high_resolution_clock::now();
        duration = duration_cast<microseconds>(end_time - start_time);
        eval_duration += duration.count();

        if (i%CHECK_INTERVAL == 0){
            start_time = high_resolution_clock::now();
            cout << "iter " << i <<endl;
            CUDA_CALL(cudaDeviceSynchronize());
            CUDA_CALL(cudaMemcpy(population, cudaPop, sizeof(Chromosome) * POPULATION_SIZE, cudaMemcpyDeviceToHost));
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
            if (prev_best_fitness == best_seen_fitness){
                tol += 1;
                if (tol >= TOLERANCE){
                    stop_iter = i+1;
                    break;
                }
            } else {
                prev_best_fitness = best_seen_fitness;
                tol = 0;
            }
            cout << "Avg Time taken by SELECTION: "
            << sel_duration/(i+1) << " microseconds" << endl;
            cout << "Avg Time taken by CROSSOVER: "
            << xover_duration/(i+1) << " microseconds" << endl;
            cout << "Avg Time taken by MUTATION: "
            << mut_duration/(i+1) << " microseconds" << endl;
            cout << "Avg Time taken by EVALUATION: "
            << eval_duration/(i+1) << " microseconds" << endl;
            end_time = high_resolution_clock::now();
            duration = duration_cast<microseconds>(end_time - start_time);
            cpu_duration += duration.count();
        }
        
    }
    if (stop_iter == 0){
        stop_iter = GEN_THRES;
    }
    printf("FINAL----------------------------------------\n");
    cudaMemcpy(population, cudaPop, sizeof(Chromosome) * POPULATION_SIZE, cudaMemcpyDeviceToHost);
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
    int total_time = sel_duration + xover_duration + mut_duration + eval_duration;
    cout << "Total GPU time: "
    << total_time << " microseconds" << endl;
    cout << "Total CPU time: "
    << cpu_duration << " microseconds" << endl;
    cout << "Converged after "
    << stop_iter << " iterations" << endl;
    cout << "Avg Time taken by SELECTION: "
    << sel_duration/stop_iter << " microseconds" << endl;
    cout << "Avg Time taken by CROSSOVER: "
    << xover_duration/stop_iter << " microseconds" << endl;
    cout << "Avg Time taken by MUTATION: "
    << mut_duration/stop_iter << " microseconds" << endl;
    cout << "Avg Time taken by EVALUATION: "
    << eval_duration/stop_iter << " microseconds" << endl;

}