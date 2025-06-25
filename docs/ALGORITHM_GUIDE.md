# BMR and BWR Algorithm Guide

This document provides an in-depth explanation of the Best-Mean-Random (BMR) and Best-Worst-Random (BWR) parameter-free optimization algorithms implemented in the BMR-model-merge framework.

## Table of Contents

- [Introduction](#introduction)
- [Algorithm Fundamentals](#algorithm-fundamentals)
- [BMR Algorithm](#bmr-algorithm)
  - [Mathematical Foundation](#bmr-mathematical-foundation)
  - [Implementation Details](#bmr-implementation-details)
  - [Hyperparameter Sensitivity](#bmr-hyperparameter-sensitivity)
  - [Convergence Properties](#bmr-convergence-properties)
- [BWR Algorithm](#bwr-algorithm)
  - [Mathematical Foundation](#bwr-mathematical-foundation)
  - [Implementation Details](#bwr-implementation-details)
  - [Hyperparameter Sensitivity](#bwr-hyperparameter-sensitivity)
  - [Convergence Properties](#bwr-convergence-properties)
- [Comparison with Genetic Algorithm](#comparison-with-genetic-algorithm)
- [Applications in Model Merging](#applications-in-model-merging)
- [Practical Guidelines](#practical-guidelines)
- [References](#references)

## Introduction

Parameter-free optimization algorithms represent a significant advancement in evolutionary computation by eliminating the need to tune hyperparameters, which is often time-consuming and requires domain expertise. The BMR and BWR algorithms implemented in this framework are designed specifically for optimizing the merging of machine learning models for Japanese OCR tasks, offering robust performance across varied problem landscapes without extensive parameter tuning.

## Algorithm Fundamentals

Both BMR and BWR follow the general structure of population-based optimization algorithms:

1. **Initialization**: Create an initial population of potential solutions
2. **Evaluation**: Assess the fitness of each solution
3. **Generation**: Create new solutions based on existing ones
4. **Selection**: Select solutions to proceed to the next generation
5. **Termination**: Stop when a termination criterion is met

The key innovation in BMR and BWR lies in the generation step, where new solutions are created using a combination of exploitation (using knowledge of good solutions) and exploration (searching new areas of the solution space) without requiring manual parameter tuning.

## BMR Algorithm

### BMR Mathematical Foundation

The Best-Mean-Random (BMR) algorithm generates new candidate solutions using the following formula:

For each individual V in the population:

```
If random() ≥ 0.5 (exploitation):
    V' = V + r₁·(Best - T·Mean) + r₂·(Best - Random)
Else (exploration):
    V' = Upper - (Upper-Lower)·r₃
```

Where:
- `V'` is the new candidate solution
- `V` is the current solution
- `Best` is the best solution found so far
- `Mean` is the mean of the current population
- `Random` is a randomly selected solution from the population
- `r₁`, `r₂`, and `r₃` are random numbers in [0,1]
- `T` is a parameter that controls the influence of the mean (typically 1.0)
- `Upper` and `Lower` are the bounds of the search space

### BMR Implementation Details

The BMR algorithm is implemented in `evomerge/evolution/bmr.py` with the following key components:

```python
class BMROptimizer:
    def __init__(self, population_size, dimension, lower_bound, upper_bound, fitness_function):
        # Initialize population randomly within bounds
        self.population = np.random.uniform(lower_bound, upper_bound, 
                                           (population_size, dimension))
        self.best_solution = None
        self.best_fitness = float('-inf')
        # Other initialization...
        
    def evolve(self):
        # Evaluate fitness for all individuals
        fitness_values = np.array([self.fitness_function(ind) for ind in self.population])
        
        # Find best individual
        best_idx = np.argmax(fitness_values)
        best = self.population[best_idx].copy()
        
        # Calculate mean of population
        mean = np.mean(self.population, axis=0)
        
        # Create new population
        new_population = np.zeros_like(self.population)
        
        for i in range(self.population_size):
            # Pick random individual
            random_idx = np.random.randint(0, self.population_size)
            random_individual = self.population[random_idx]
            
            if np.random.random() >= 0.5:  # Exploitation
                # BMR formula
                r1 = np.random.random()
                r2 = np.random.random()
                new_population[i] = self.population[i] + \
                                   r1 * (best - self.T * mean) + \
                                   r2 * (best - random_individual)
            else:  # Exploration
                # Random exploration within bounds
                r3 = np.random.random()
                new_population[i] = self.upper_bound - \
                                   (self.upper_bound - self.lower_bound) * r3
            
            # Ensure bounds
            new_population[i] = np.clip(new_population[i], 
                                       self.lower_bound, 
                                       self.upper_bound)
        
        # Elitism: preserve best individual
        new_population[0] = best
        
        self.population = new_population
        # Update best solution if improved
        # ...
```

### BMR Hyperparameter Sensitivity

While BMR is largely parameter-free, it has a few internal mechanisms that affect its behavior:

1. **Exploitation-Exploration Balance**: Fixed at 0.5, controlling how often the algorithm explores vs. exploits
2. **T Parameter**: Controlling the influence of the mean in the exploitation formula (typically set to 1.0)
3. **Population Size**: While not part of the algorithm itself, affects diversity and convergence speed

In our implementation, we found the algorithm to be robust across different settings, making it truly "parameter-free" in practice.

### BMR Convergence Properties

BMR typically exhibits faster initial convergence compared to genetic algorithms, especially in:
- Unimodal landscapes where the best solution guides the search effectively
- Problems where the mean provides useful directional information
- When random exploration helps escape local optima

## BWR Algorithm

### BWR Mathematical Foundation

The Best-Worst-Random (BWR) algorithm modifies the BMR approach by incorporating information about the worst solution:

For each individual V in the population:

```
If random() ≥ 0.5 (exploitation):
    V' = V + r₁·(Best - T·Random) - r₂·(Worst - Random)
Else (exploration):
    V' = Upper - (Upper-Lower)·r₃
```

Where:
- All variables are the same as in BMR
- `Worst` is the worst solution in the current population

### BWR Implementation Details

The BWR algorithm is implemented in `evomerge/evolution/bwr.py` with key differences from BMR:

```python
class BWROptimizer:
    # Initialization similar to BMR...
    
    def evolve(self):
        # Evaluate fitness for all individuals
        fitness_values = np.array([self.fitness_function(ind) for ind in self.population])
        
        # Find best individual
        best_idx = np.argmax(fitness_values)
        best = self.population[best_idx].copy()
        
        # Find worst individual
        worst_idx = np.argmin(fitness_values)
        worst = self.population[worst_idx].copy()
        
        # Create new population
        new_population = np.zeros_like(self.population)
        
        for i in range(self.population_size):
            # Pick random individual
            random_idx = np.random.randint(0, self.population_size)
            random_individual = self.population[random_idx]
            
            if np.random.random() >= 0.5:  # Exploitation
                # BWR formula
                r1 = np.random.random()
                r2 = np.random.random()
                new_population[i] = self.population[i] + \
                                   r1 * (best - self.T * random_individual) - \
                                   r2 * (worst - random_individual)
            else:  # Exploration
                # Random exploration within bounds
                r3 = np.random.random()
                new_population[i] = self.upper_bound - \
                                   (self.upper_bound - self.lower_bound) * r3
            
            # Ensure bounds
            new_population[i] = np.clip(new_population[i], 
                                       self.lower_bound, 
                                       self.upper_bound)
        
        # Elitism: preserve best individual
        new_population[0] = best
        
        self.population = new_population
        # Update best solution if improved
        # ...
```

### BWR Hyperparameter Sensitivity

BWR shares the same robustness to parameter settings as BMR, but with some unique considerations:

1. The use of the worst individual can cause more aggressive movement away from poor solutions
2. It may be more effective in deceptive landscapes where avoiding poor areas is as important as finding good ones

### BWR Convergence Properties

BWR often exhibits:
- Better performance in multi-modal landscapes compared to BMR
- More aggressive exploration due to repulsion from worst solutions
- Stronger resistance to premature convergence in complex landscapes

## Comparison with Genetic Algorithm

The traditional genetic algorithm approach implemented in `evomerge/evolution/genetic.py` differs in several key ways:

| Feature | Genetic Algorithm | BMR/BWR |
|---------|------------------|---------|
| Parameters | Many (crossover rate, mutation rate, selection pressure, etc.) | Minimal (population size, generations) |
| Exploration | Through mutation | Through random reset mechanism |
| Exploitation | Through selection and crossover | Through mathematical formulas |
| Performance on simple landscapes | Good | Similar or better |
| Performance on complex landscapes | Variable, depends on parameter tuning | More consistent |
| Computational complexity | Higher due to selection and crossover operations | Lower, more efficient formula |
| Implementation complexity | Higher | Lower |

## Applications in Model Merging

For model merging optimization specifically, BMR and BWR offer several advantages:

1. **Efficiency in High-Dimensional Spaces**: Model weight spaces are typically very high-dimensional, and parameter-free algorithms handle this well without extensive tuning.

2. **Handling Multiple Objectives**: Japanese OCR requires balancing character recognition accuracy with field extraction precision - the BMR/BWR combined fitness approach handles this naturally.

3. **Adaptation to Different Document Types**: The algorithms can adapt to different document templates and structures without requiring specialized parameter tuning for each.

### Model Merging Implementation

For weight-space optimization:

```python
# Simplified example from implementation
def merge_models_with_bmr(model_paths, config):
    # Extract weights from models
    weights_list = [extract_model_weights(path) for path in model_paths]
    dimension = len(weights_list[0])
    
    # Determine bounds based on weight distribution
    lower_bounds = np.min([weights for weights in weights_list], axis=0) - 0.2
    upper_bounds = np.max([weights for weights in weights_list], axis=0) + 0.2
    
    # Setup fitness function (OCR accuracy)
    def fitness_function(weights):
        merged_model = create_model_from_weights(weights)
        return evaluate_model_on_validation_set(merged_model)
    
    # Initialize optimizer
    optimizer = BMROptimizer(
        population_size=config.population_size,
        dimension=dimension,
        lower_bound=lower_bounds,
        upper_bound=upper_bounds,
        fitness_function=fitness_function
    )
    
    # Run optimization
    for generation in range(config.generations):
        optimizer.evolve()
        
    # Return best weights
    return optimizer.get_best_individual()
```

## Practical Guidelines

Based on our extensive experimentation, we recommend the following guidelines:

1. **When to use BMR**:
   - For simpler document layouts with clear structure
   - When you need faster initial convergence
   - When averaging models provides a good starting point

2. **When to use BWR**:
   - For complex document types with varied layouts
   - When avoiding bad solutions is as important as finding good ones
   - When you suspect local optima issues

3. **Population Size Guidelines**:
   - For standard invoices: 20-30 individuals
   - For complex varied documents: 30-50 individuals

4. **Generation Count Guidelines**:
   - Quick prototyping: 10-20 generations
   - Production-quality optimization: 50-100 generations

5. **Computational Resources**:
   - Both algorithms are efficient, but fitness evaluation (model testing) is typically the bottleneck
   - Consider parallelizing fitness evaluations if possible

## References

1. Das, S., Suganthan, P.N. "Differential Evolution: A Survey of the State-of-the-Art." IEEE Trans. Evol. Comput. (2011)
2. Hansen, N. "The CMA Evolution Strategy: A Tutorial." (2016)
3. Yu, X., Gen, M. "Introduction to Evolutionary Algorithms." (2010)
4. Zhang, J., Sanderson, A.C. "JADE: Adaptive Differential Evolution With Optional External Archive." IEEE Trans. Evol. Comput. (2009)
5. Li, X., Yin, M. "Modified differential evolution with self-adaptive parameters method." Journal of Combinatorial Optimization. (2016)