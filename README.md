# pGA
This project explored different parallelizing strategies to accelerate the genetic algorithm for solving the traveling salesman problem in C++ and used real-life examples to evaluate speedup. 

This code base provides several implementation of the genetic alorightm to solve the traveling salesman problem and a test generator that can randomly generate distance matrices for future tests.

## Usage
### To run genetic algorithm to solve TSP
Each approach is stored in its own folder (cuda, openmp, seq_improved, seq), to run an implementation, go to the corresponding folder
1. cd path_to_implmentation_folder
2. Terminal> make
3. Terminal> ./genetic-alg

### To generate random distance matrix
1. cd testcase
2. python3 generator.py outputfilename dimension

### To load and convert a tsp file to distance matrix
1. cd testcase
2. python3 tsplib_loader.py inputfile.tsp outputfilename dimension

## File organization
.  
├── brute_force.cpp  
├── brute_force.h  
├── cuda  
│   ├── CycleTimer.h  
│   ├── Makefile  
│   ├── cuda.cu  
│   └── objs  
│       └── cuda.o  
├── openmp  
│   ├── Makefile  
│   ├── brute_force.cpp  
│   ├── brute_force.h  
│   ├── chromosome.h  
│   ├── genetic-alg  
│   ├── main.cpp  
│   └── openmp-alg  
├── openmp-fail1  
│   ├── Makefile  
│   ├── chromosome.h  
│   ├── main.cpp  
│   └── openmp-alg  
├── seq_improved  
│   ├── Makefile  
│   ├── brute_force.cpp  
│   ├── brute_force.h  
│   ├── chromosome.h  
│   ├── genetic-alg  
│   └── main.cpp  
├── seq_vec  
│   ├── Makefile  
│   ├── brute_force.cpp  
│   ├── brute_force.h  
│   ├── chromosome.h  
│   ├── genetic-alg  
│   └── main.cpp  
└── testcase  
    ├── 100.txt  
    ├── 11.txt  
    ├── 12.txt  
    ├── 13.txt  
    ├── 20.txt  
    ├── 30.txt  
    ├── 4.txt  
    ├── 5.txt  
    ├── 6.txt  
    ├── 7.txt  
    ├── 8.txt  
    ├── 9.txt  
    ├── dj38.tsp  
    ├── dj38.txt  
    ├── generator.py  
    ├── qa194.tsp  
    ├── qa194.txt  
    ├── tsplib_loader.py  
    ├── uy734.tsp  
    └── uy734.txt  