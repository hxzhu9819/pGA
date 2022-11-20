#include <string>
#include <bitset>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string.h>
#include <stdlib.h>
using namespace std;

int rand_num_from_range(int start, int end) {
    return start + rand() % (end - start);
}

bool repeat(string s, char c) {
    for (auto sc : s) {
        if (sc == c) return true;
    }
    return false;
}

class Chromosome {
    private:
        string gnome;
        int size;
    public:
        int fitness;
        Chromosome(int size);
        Chromosome(const Chromosome &t);
        void create();
        void reset();
        string mutate();
        string get_gnome();
};

bool worsethan(Chromosome c1, Chromosome c2) {
    return c1.fitness < c2.fitness;
}

Chromosome::Chromosome(int size) {
    this->size = size;
    cout << "chromosome::constructor is called" << endl;
}

Chromosome::Chromosome(const Chromosome &t) {
    gnome = t.gnome;
    size = t.size;
    fitness = t.fitness;
}

string Chromosome::get_gnome() {
    return gnome;
}

void Chromosome::create() {
    gnome = "0";
    while (true) {
        if (gnome.size() == size) {
            gnome += gnome[0];
            break;
        }
        int tmp = rand_num_from_range(1, size);
        if (!repeat(gnome, (char)(tmp+48))) {
            gnome += (char)(tmp + 48);
        }
    } 
}

string Chromosome::mutate() {
    while (true) {
        int victim_gene1 = rand_num_from_range(1, size);
        int victim_gene2 = rand_num_from_range(1, size);
        if (victim_gene1 != victim_gene2) {
            char tmp = gnome[victim_gene1];
            gnome[victim_gene1] = gnome[victim_gene2];
            gnome[victim_gene2] = tmp;
            break;
        }
    }
    return gnome;
} 