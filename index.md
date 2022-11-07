## Summary
We are going to implement and benchmark different parallelizing strategies, including openMP and CUDA, to accelerate the genetic algorithm for solving the traveling salesman problem.

## Background
Genetic Algorithm (GA) is a search-based algorithm that is widely used to solve NP-complete problems. Inspired by genetics principles, the algorithm converges to solutions by reflecting the process of natural selection where the fittest chromosomes are chosen for reproduction to produce better offspring of the next generation until an optimized solution is generated.

Genetic Algorithm is an iterative process. After randomly initializing a population, the algorithm executes the following 4 steps recursively, scoring and selecting elite parents from the population for mating, applying crossover to generate next generation chromosomes, conducting mutation to introduce data augmentation, and updating existing population by removing low-score chromosomes, until a predefined criteria is met. 

Different parallelization strategies can be applied to different phases based on their computation and data dependency nature and may ideally accelerate the process. Data parallelism can be performed during chromosome evaluation by exploiting independencies of chromosomes, and ad-hoc parallelism strategy for chromosome evolution (crossover and mutation phases) can be designed to accelerate the process. The huge population and computation requirements allows parallel programming to scale well.
