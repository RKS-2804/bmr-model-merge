"""Population management for evolutionary model merging."""

import os
import uuid
import json
import random
import logging
from typing import Dict, List, Any, Optional, Tuple, Callable, Union
import numpy as np
from dataclasses import dataclass, field, asdict
from datetime import datetime

logger = logging.getLogger(__name__)

@dataclass
class Individual:
    """
    Represents an individual model in the evolutionary process.
    
    Each individual contains information about model merging parameters,
    including which models to merge and with what weights.
    """
    
    # Unique identifier for this individual
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    
    # Generation this individual was created in
    generation: int = 0
    
    # Dictionary of model paths and their merge weights
    model_config: Dict[str, Any] = field(default_factory=dict)
    
    # Merging parameters dictionary
    merge_params: Dict[str, Any] = field(default_factory=dict)
    
    # Path to where model weights are stored (if available)
    model_path: Optional[str] = None
    
    # Fitness scores (higher is better)
    fitness: Dict[str, float] = field(default_factory=dict)
    
    # Overall fitness score (computed from individual metrics)
    overall_fitness: float = 0.0
    
    # When this individual was created
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    
    # Whether this individual has been evaluated
    evaluated: bool = False
    
    def __lt__(self, other):
        """
        Compare individuals by overall fitness (for sorting).
        """
        return self.overall_fitness < other.overall_fitness
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert individual to a dictionary for serialization.
        """
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Individual':
        """
        Create an individual from a dictionary.
        """
        return cls(**data)


class Population:
    """
    Manages a population of model individuals for evolutionary optimization.
    
    This class handles selection, tracking of individuals across generations,
    and persistence of the population state.
    """
    
    def __init__(
        self,
        size: int = 10,
        checkpoint_dir: str = "./checkpoints",
        maximize_fitness: bool = True, 
        elitism_pct: float = 0.2,
        tournament_size: int = 3,
        random_seed: Optional[int] = None
    ):
        """
        Initialize a new population.
        
        Args:
            size: Size of the population
            checkpoint_dir: Directory for saving/loading checkpoints
            maximize_fitness: Whether higher fitness values are better
            elitism_pct: Percentage of top individuals to preserve 
            tournament_size: Number of individuals in tournament selection
            random_seed: Optional seed for random number generator
        """
        self.size = size
        self.individuals: List[Individual] = []
        self.generation = 0
        self.checkpoint_dir = checkpoint_dir
        self.maximize_fitness = maximize_fitness
        self.elitism_pct = elitism_pct
        self.tournament_size = tournament_size
        self.history: Dict[int, Dict[str, Any]] = {}
        
        # Set random seed if provided
        if random_seed is not None:
            random.seed(random_seed)
            np.random.seed(random_seed)
    
    def add_individual(self, individual: Individual) -> None:
        """
        Add an individual to the population.
        
        Args:
            individual: The individual to add
        """
        self.individuals.append(individual)
    
    def get_individual(self, id: str) -> Optional[Individual]:
        """
        Get an individual by ID.
        
        Args:
            id: The ID of the individual to retrieve
            
        Returns:
            The individual if found, None otherwise
        """
        for ind in self.individuals:
            if ind.id == id:
                return ind
        return None
    
    def sort_by_fitness(self) -> None:
        """
        Sort individuals by overall fitness score.
        """
        self.individuals.sort(
            key=lambda ind: ind.overall_fitness,
            reverse=self.maximize_fitness
        )
    
    def select_parents(self, n_parents: int = 2) -> List[Individual]:
        """
        Select individuals to become parents through tournament selection.
        
        Args:
            n_parents: Number of parents to select
            
        Returns:
            List of selected parent individuals
        """
        parents = []
        for _ in range(n_parents):
            # Tournament selection
            tournament = random.sample(self.individuals, min(self.tournament_size, len(self.individuals)))
            
            if self.maximize_fitness:
                winner = max(tournament, key=lambda ind: ind.overall_fitness)
            else:
                winner = min(tournament, key=lambda ind: ind.overall_fitness)
                
            parents.append(winner)
        
        return parents
    
    def select_elite(self) -> List[Individual]:
        """
        Select elite individuals to preserve in the next generation.
        
        Returns:
            List of elite individuals
        """
        self.sort_by_fitness()
        n_elite = max(1, int(self.size * self.elitism_pct))
        return self.individuals[:n_elite]
    
    def update_statistics(self) -> Dict[str, Any]:
        """
        Update and return population statistics.
        
        Returns:
            Dictionary of population statistics
        """
        if not self.individuals:
            return {}
            
        fitness_values = [ind.overall_fitness for ind in self.individuals]
        
        stats = {
            "generation": self.generation,
            "population_size": len(self.individuals),
            "best_fitness": max(fitness_values) if self.maximize_fitness else min(fitness_values),
            "mean_fitness": sum(fitness_values) / len(fitness_values),
            "stddev_fitness": np.std(fitness_values),
            "best_individual_id": self.individuals[0].id if self.individuals else None,
            "timestamp": datetime.now().isoformat()
        }
        
        self.history[self.generation] = stats
        return stats
    
    def get_best_individual(self) -> Optional[Individual]:
        """
        Get the individual with the best fitness score.
        
        Returns:
            The best individual, or None if population is empty
        """
        if not self.individuals:
            return None
            
        self.sort_by_fitness()
        return self.individuals[0]
    
    def save_checkpoint(self, path: Optional[str] = None) -> str:
        """
        Save the current population state to a checkpoint file.
        
        Args:
            path: Optional specific path to save to
            
        Returns:
            Path to the saved checkpoint file
        """
        if path is None:
            os.makedirs(self.checkpoint_dir, exist_ok=True)
            path = os.path.join(
                self.checkpoint_dir, 
                f"population_gen{self.generation}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            )
            
        data = {
            "generation": self.generation,
            "size": self.size,
            "maximize_fitness": self.maximize_fitness,
            "elitism_pct": self.elitism_pct,
            "tournament_size": self.tournament_size,
            "individuals": [ind.to_dict() for ind in self.individuals],
            "history": self.history,
            "timestamp": datetime.now().isoformat()
        }
        
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
            
        logger.info(f"Saved population checkpoint to {path}")
        return path
    
    @classmethod
    def load_checkpoint(cls, path: str) -> 'Population':
        """
        Load a population from a checkpoint file.
        
        Args:
            path: Path to the checkpoint file
            
        Returns:
            The loaded Population
        """
        with open(path, "r") as f:
            data = json.load(f)
            
        pop = cls(
            size=data["size"],
            maximize_fitness=data["maximize_fitness"],
            elitism_pct=data["elitism_pct"],
            tournament_size=data["tournament_size"]
        )
        
        pop.generation = data["generation"]
        pop.individuals = [Individual.from_dict(ind_data) for ind_data in data["individuals"]]
        pop.history = data["history"]
        
        logger.info(f"Loaded population checkpoint from {path}, generation {pop.generation}")
        return pop
        
    def next_generation(self) -> None:
        """
        Advance to the next generation.
        """
        self.generation += 1
        for ind in self.individuals:
            ind.generation = self.generation