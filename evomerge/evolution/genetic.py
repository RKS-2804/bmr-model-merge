"""
Genetic Algorithm Optimization.

A traditional genetic algorithm with selection, crossover, and mutation operations
for evolutionary optimization.
"""

import numpy as np
from typing import Callable, List, Tuple, Optional, Union, Dict, Any


class GeneticOptimizer:
    """
    Implements a standard genetic algorithm for optimization.
    
    This genetic algorithm uses tournament selection, uniform crossover,
    and Gaussian mutation.
    """
    
    def __init__(
        self,
        population_size: int,
        dimension: int,
        lower_bound: Union[float, List[float], np.ndarray],
        upper_bound: Union[float, List[float], np.ndarray],
        fitness_function: Callable[[np.ndarray], float],
        mutation_rate: float = 0.1,
        crossover_rate: float = 0.7,
        tournament_size: int = 3,
        elitism: int = 1,
    ):
        """
        Initialize the genetic algorithm optimizer.
        
        Args:
            population_size: Size of the population
            dimension: Dimensionality of the solution space
            lower_bound: Lower bound(s) of the search space
            upper_bound: Upper bound(s) of the search space
            fitness_function: Function to evaluate fitness of solutions
            mutation_rate: Probability of mutation per gene (default: 0.1)
            crossover_rate: Probability of crossover (default: 0.7)
            tournament_size: Number of individuals for tournament selection (default: 3)
            elitism: Number of top individuals to preserve unchanged (default: 1)
        """
        self.population_size = population_size
        self.dimension = dimension
        self.fitness_function = fitness_function
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.tournament_size = min(tournament_size, population_size)
        self.elitism = min(elitism, population_size)
        
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
    
    def _tournament_selection(self, fitness_values: np.ndarray) -> int:
        """Select an individual using tournament selection."""
        # Select random individuals for tournament
        tournament_indices = np.random.choice(
            self.population_size, self.tournament_size, replace=False
        )
        
        # Find winner (highest fitness)
        winner_idx = tournament_indices[
            np.argmax(fitness_values[tournament_indices])
        ]
        
        return winner_idx
    
    def _crossover(self, parent1: np.ndarray, parent2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Perform uniform crossover between two parents."""
        # Generate random binary mask
        mask = np.random.random(self.dimension) < 0.5
        
        # Create offspring
        offspring1 = np.where(mask, parent1, parent2)
        offspring2 = np.where(mask, parent2, parent1)
        
        return offspring1, offspring2
    
    def _mutation(self, individual: np.ndarray) -> np.ndarray:
        """Apply Gaussian mutation to an individual."""
        # Determine which genes to mutate
        mutation_mask = np.random.random(self.dimension) < self.mutation_rate
        
        # Apply mutation
        if np.any(mutation_mask):
            # Calculate mutation strength (dynamic, based on bounds)
            strength = (self.upper_bound - self.lower_bound) * 0.1
            
            # Generate Gaussian noise
            noise = np.random.normal(0, 1, self.dimension) * strength * mutation_mask
            
            # Apply mutation
            mutant = individual + noise
            
            # Ensure bounds
            mutant = np.clip(mutant, self.lower_bound, self.upper_bound)
            
            return mutant
        else:
            return individual
    
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
        
        # Create new population
        new_population = np.zeros_like(self.population)
        
        # Implement elitism - preserve top individuals
        for i in range(self.elitism):
            new_population[i] = sorted_population[i]
        
        # Fill the rest of the population with offspring
        i = self.elitism
        while i < self.population_size:
            # Select parents
            parent1_idx = self._tournament_selection(fitness_values)
            parent2_idx = self._tournament_selection(fitness_values)
            
            # Get parents
            parent1 = self.population[parent1_idx]
            parent2 = self.population[parent2_idx]
            
            # Perform crossover with probability
            if np.random.random() < self.crossover_rate:
                offspring1, offspring2 = self._crossover(parent1, parent2)
            else:
                offspring1, offspring2 = parent1.copy(), parent2.copy()
            
            # Perform mutation
            offspring1 = self._mutation(offspring1)
            offspring2 = self._mutation(offspring2)
            
            # Add offspring to new population
            new_population[i] = offspring1
            i += 1
            
            # Add second offspring if there's room
            if i < self.population_size:
                new_population[i] = offspring2
                i += 1
        
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