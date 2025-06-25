"""
BMR (Best-Mean-Random) Optimization Algorithm.

The BMR algorithm is a parameter-free optimization algorithm that leverages the best solution,
mean of the population, and a randomly selected solution to guide the search process.
"""

import numpy as np
from typing import Callable, List, Tuple, Optional, Union, Dict, Any


class BMROptimizer:
    """
    Implements the Best-Mean-Random (BMR) optimization algorithm.
    
    BMR uses the formula:
    V' = V + r₁·(Best - T·Mean) + r₂·(Best - Random)
    
    where V is the current solution, Best is the best solution, Mean is the population mean,
    Random is a randomly selected solution, r₁ and r₂ are random numbers, and T is a parameter.
    """
    
    def __init__(
        self,
        population_size: int,
        dimension: int,
        lower_bound: Union[float, List[float], np.ndarray],
        upper_bound: Union[float, List[float], np.ndarray],
        fitness_function: Callable[[np.ndarray], float],
        T: float = 1.0,
        elitism: int = 1,
        adaptive: bool = True,
    ):
        """
        Initialize the BMR optimizer.
        
        Args:
            population_size: Size of the population
            dimension: Dimensionality of the solution space
            lower_bound: Lower bound(s) of the search space
            upper_bound: Upper bound(s) of the search space
            fitness_function: Function to evaluate fitness of solutions
            T: Parameter controlling the influence of the mean (default: 1.0)
            elitism: Number of top individuals to preserve unchanged (default: 1)
            adaptive: Whether to use adaptive parameter setting (default: True)
        """
        self.population_size = population_size
        self.dimension = dimension
        self.fitness_function = fitness_function
        self.T = T
        self.elitism = min(elitism, population_size)
        self.adaptive = adaptive
        
        # Convert bounds to arrays if they are scalars
        if isinstance(lower_bound, (int, float)):
            self.lower_bound = np.full(dimension, lower_bound)
        else:
            self.lower_bound = np.asarray(lower_bound)
        
        if isinstance(upper_bound, (int, float)):
            self.upper_bound = np.full(dimension, upper_bound)
        else:
            self.upper_bound = np.asarray(upper_bound)
        
        # Initialize population randomly within bounds
        self.population = self._initialize_population()
        
        # Initialize best solution tracking
        self.best_individual = None
        self.best_fitness = float('-inf')
        
        # Evaluate initial population
        self._evaluate_population()
    
    def _initialize_population(self) -> np.ndarray:
        """Initialize population randomly within bounds."""
        population = np.random.uniform(
            self.lower_bound, self.upper_bound, 
            size=(self.population_size, self.dimension)
        )
        return population
    
    def _evaluate_population(self) -> np.ndarray:
        """Evaluate fitness for all individuals in the population."""
        fitness_values = np.zeros(self.population_size)
        
        for i in range(self.population_size):
            fitness_values[i] = self.fitness_function(self.population[i])
            
            # Update best individual if better
            if fitness_values[i] > self.best_fitness:
                self.best_fitness = fitness_values[i]
                self.best_individual = self.population[i].copy()
        
        return fitness_values
    
    def evolve(self) -> Tuple[np.ndarray, float]:
        """
        Evolve the population for one generation.
        
        Returns:
            Tuple of (best individual, best fitness) after evolution
        """
        # Evaluate fitness for all individuals
        fitness_values = self._evaluate_population()
        
        # Sort population by fitness (descending)
        sorted_indices = np.argsort(-fitness_values)
        sorted_population = self.population[sorted_indices]
        sorted_fitness = fitness_values[sorted_indices]
        
        # Calculate mean of population
        mean = np.mean(self.population, axis=0)
        
        # Create new population
        new_population = np.zeros_like(self.population)
        
        # Implement elitism - preserve top individuals
        for i in range(self.elitism):
            new_population[i] = sorted_population[i]
        
        # Generate new individuals for the rest of the population
        for i in range(self.elitism, self.population_size):
            # Get current individual
            individual = self.population[i]
            
            # Decide between exploitation and exploration
            if np.random.random() >= 0.5:  # Exploitation
                # Pick random individual
                random_idx = np.random.randint(0, self.population_size)
                random_individual = self.population[random_idx]
                
                # BMR formula
                r1 = np.random.random()
                r2 = np.random.random()
                
                new_population[i] = individual + \
                                   r1 * (self.best_individual - self.T * mean) + \
                                   r2 * (self.best_individual - random_individual)
            else:  # Exploration - random reset
                # Generate random solution within bounds
                r3 = np.random.random()
                new_population[i] = self.lower_bound + \
                                   (self.upper_bound - self.lower_bound) * r3
            
            # Ensure bounds
            new_population[i] = np.clip(
                new_population[i], self.lower_bound, self.upper_bound
            )
        
        # Replace old population
        self.population = new_population
        
        # Return best individual and fitness
        return self.best_individual, self.best_fitness
    
    def get_best_individual(self) -> np.ndarray:
        """Return the best individual found so far."""
        return self.best_individual
    
    def get_best_fitness(self) -> float:
        """Return the fitness of the best individual."""
        return self.best_fitness
    
    def get_population(self) -> np.ndarray:
        """Return the current population."""
        return self.population
    
    def get_population_statistics(self) -> Dict[str, Any]:
        """Return statistics about the current population."""
        fitness_values = np.array([
            self.fitness_function(ind) for ind in self.population
        ])
        
        return {
            "mean_fitness": np.mean(fitness_values),
            "min_fitness": np.min(fitness_values),
            "max_fitness": np.max(fitness_values),
            "std_fitness": np.std(fitness_values),
            "diversity": np.mean([
                np.linalg.norm(ind - self.best_individual) 
                for ind in self.population
            ]),
        }