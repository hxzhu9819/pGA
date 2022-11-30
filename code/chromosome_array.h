#include <string>
#include <bitset>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string.h>
#include <stdlib.h>
#include <vector>
#include <random>
using namespace std;

int rand_num_from_range(int start, int end) {
    return start + rand() % (end - start);
    // std::random_device rd;
    // std::mt19937 gen(rd());
    // std::uniform_int_distribution<> dis(start, end-1);
    // return dis(gen);
}

// bool repeat(string s, char c) {
//     for (auto sc : s) {
//         if (sc == c) return true;
//     }
//     return false;
// }

bool repeat(int s[], size_t size, int c) {
    for (int i = 0; i < size; ++i) {
        if (s[i] == c) return true;
    }
    return false;
}

class Chromosome {
    public:
        int* gnome_vec;
        int fitness;
        int size; // number of city
        Chromosome(int size);
        Chromosome(const Chromosome &t);
        void create();
        void reset();
        // void cross_over(Chromosome& x, Chromosome& y, int p);
        void cross_over(Chromosome& x, Chromosome& y, int p, int** map);
        int* mutate();
        string get_gnome();
        int* get_gnome_vec();
        void mutate_by_invert();
};

bool worsethan(Chromosome c1, Chromosome c2) {
    return c1.fitness < c2.fitness;
}

Chromosome::Chromosome(int size) {
    this->size = size;
    this->gnome_vec = new int[size+1]{0};
    // cout << "chromosome::constructor is called" << endl;
}

Chromosome::Chromosome(const Chromosome &t) {
    size = t.size;
    gnome_vec = new int[size+1];
    for (int i = 0; i < size; ++i) {
        gnome_vec[i] = t.gnome_vec[i];
    }
    fitness = t.fitness;
}

string Chromosome::get_gnome() {
    string gnome_str = "";
    for (int i = 0; i < size; ++i) {
        gnome_str += to_string(gnome_vec[i]);
        gnome_str += "->";
        // cout << "gnome_str: " << gnome_str << endl;
    }
    return gnome_str;
}

vector<int> Chromosome::get_gnome_vec() {
    return gnome_vec;
}

void Chromosome::create() {
    gnome_vec[0] = 0;
    for(int i = 1;i < size; ++i) {
        int tmp = rand_num_from_range(1, size);
        while (repeat(gnome_vec, size, tmp)) {
            tmp = rand_num_from_range(1, size);
        }
        gnome_vec[i] = tmp;
    }
    gnome_vec[size] = 0;
}

int get_index(int* v, int length, int val) {
    for (int i = 0; i < length; i++){
        if (v[i] == val){
            return i;
        }
    }
    return -1;
}

void Chromosome::cross_over(Chromosome& x, Chromosome& y, int p, int** map) {
    int* parent1_gnome_vec = x.get_gnome_vec();
    int* parent2_gnome_vec = y.get_gnome_vec();
    gnome_vec[0] = 0;
    gnome_vec[size] = 0;
    for (int i = 1; i < size; ++i) {
        int gnome_from_p1, gnome_from_p2;
        int p1_cand_idx = get_index(parent1_gnome_vec, size, gnome_vec[i-1])+1;
        int p1_cand = parent1_gnome_vec[p1_cand_idx];
        if (p1_cand == 0 || get_index(gnome_vec, i, p1_cand) != -1) {
            for (int j = 1; j < size; ++j) {
                if (get_index(gnome_vec, i, j) == -1) {
                    p1_cand = j;
                    break;
                }
            }
        }
        // p2
        int p2_cand_idx = get_index(parent2_gnome_vec, gnome_vec[i-1])+1;
        int p2_cand = parent2_gnome_vec[p2_cand_idx];
        if (p2_cand == 0 || get_index(gnome_vec, i, p2_cand) != -1) {
            for (int j = 1; j < size; ++j) {
                if (get_index(new_chro, i, j) == -1) {
                    p2_cand = j;
                    break;
                }
            }
        }
        // compete
        int p1_fitness = map[gnome_vec[i-1]][p1_cand];
        int p2_fitness = map[gnome_vec[i-1]][p2_cand];
        gnome_vec[i] = p1_fitness <= p2_fitness ? p1_cand : p2_cand;
    }
}

// void Chromosome::cross_over(Chromosome& x, Chromosome& y, int p) {
//     vector<int> parent1_gnome_vec = x.get_gnome_vec();
//     vector<int> parent2_gnome_vec = y.get_gnome_vec();
//     vector<int> new_chro;
//     new_chro.resize(x.size+1, 0);
    
//     // cout <<"p1: ";
//     // for(auto x: parent1_gnome_vec) {
//     //     cout << x << " ";
//     // }
//     // cout << endl;

//     // cout <<"p2: ";
//     // for(auto x: parent2_gnome_vec) {
//     //     cout << x << " ";
//     // }
//     // cout << endl;

//     int g_idx = 1;
//     for (int i = 1; i < p; i++) {
//         new_chro[g_idx] = parent1_gnome_vec[i];
//         g_idx++;
//     }

//     // cout << "[CV] pivot " << p << endl;
//     // for(auto x : new_chro) {
//     //     cout << x << " ";
//     // }
//     // cout << endl;

//     int i = 1;
//     while (g_idx < x.size) {
//         if (find(new_chro.begin(), new_chro.end(), parent2_gnome_vec[i]) == new_chro.end()) {
//             // cout << parent2_gnome_vec[i] << endl;
//             new_chro[g_idx++] = parent2_gnome_vec[i++];
//         } else {
//             i++;
//         }
//     }
//     new_chro[g_idx] = 0;
//     gnome_vec = new_chro;
//     // cout << "CROSSOVER ONE DONE" <<endl;
// }

vector<int> Chromosome::mutate() {
    while (true) {
        int victim_gene1 = rand_num_from_range(1, size);
        int victim_gene2 = rand_num_from_range(1, size);
        if (victim_gene1 != victim_gene2) {
            int tmp = gnome_vec[victim_gene1];
            gnome_vec[victim_gene1] = gnome_vec[victim_gene2];
            gnome_vec[victim_gene2] = tmp;
            break;
        }
    }
    return gnome_vec;
} 

void Chromosome::mutate_by_invert() {
    int victim_gene1_idx = rand_num_from_range(1, size);
    int victim_gene2_idx = rand_num_from_range(1, size);
    while (victim_gene2_idx == victim_gene1_idx) {
        victim_gene2_idx = rand_num_from_range(1, size);
    }
    // cout << victim_gene1_idx << " " << victim_gene2_idx << endl;

    if (victim_gene1_idx > victim_gene2_idx) { // swap
        int tmp = victim_gene1_idx;
        victim_gene1_idx = victim_gene2_idx;
        victim_gene2_idx = tmp;
    }
    while (victim_gene1_idx < victim_gene2_idx) {
        int tmp = gnome_vec[victim_gene1_idx];
        gnome_vec[victim_gene1_idx] = gnome_vec[victim_gene2_idx];
        gnome_vec[victim_gene2_idx] = tmp;
        victim_gene1_idx++;
        victim_gene2_idx--;
    }
    return;
}