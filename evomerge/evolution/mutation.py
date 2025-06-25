"""Mutation operations for evolutionary model merging."""

import os
import yaml
import random
import logging
import tempfile
from typing import Dict, List, Any, Optional, Tuple, Union

import torch
import numpy as np
from mergekit import merge
from mergekit.merge_recipe import MergeRecipe

from .population import Individual

logger = logging.getLogger(__name__)

def mutate_model(
    individual: Individual,
    mutation_method: str = "parameter_noise",
    mutation_rate: float = 0.1,
    mutation_strength: float = 0.01,
    output_dir: str = None,
    **kwargs
) -> Individual:
    """
    Apply mutation to an individual model.
    
    Args:
        individual: Individual to mutate
        mutation_method: Method of mutation ('parameter_noise', 'layer_dropout', 'recipe_params')
        mutation_rate: Probability of mutating each parameter
        mutation_strength: Strength of mutation effect
        output_dir: Directory to save the mutated model
        **kwargs: Additional arguments specific to mutation method
        
    Returns:
        A new Individual representing the mutated model
    """
    if output_dir is None:
        output_dir = tempfile.mkdtemp(prefix="evomerge_mutated_")
        
    # Create a copy of the individual for mutation
    mutated = Individual(
        generation=individual.generation + 1,
        model_config=individual.model_config.copy(),
        merge_params=individual.merge_params.copy()
    )
    
    # Determine the mutation method to use
    if mutation_method == "parameter_noise":
        recipe = _create_noise_mutation_recipe(individual, mutation_rate, mutation_strength, **kwargs)
    elif mutation_method == "layer_dropout":
        recipe = _create_layer_dropout_recipe(individual, mutation_rate, **kwargs)
    elif mutation_method == "recipe_params":
        recipe = _mutate_recipe_params(individual, mutation_strength, **kwargs)
    else:
        raise ValueError(f"Unknown mutation method: {mutation_method}")
    
    # Save mutation recipe
    recipe_path = os.path.join(output_dir, "mutation_recipe.yaml")
    with open(recipe_path, 'w') as f:
        yaml.dump(recipe, f)
    
    # Store mutation parameters
    mutated.merge_params.update({
        "mutation_method": mutation_method,
        "mutation_rate": mutation_rate,
        "mutation_strength": mutation_strength,
        "parent_id": individual.id,
        "recipe_path": recipe_path
    })
    
    # Update model path
    mutated.model_path = output_dir
    mutated.model_config["base_model"] = output_dir
    
    return mutated


def _create_noise_mutation_recipe(
    individual: Individual, 
    mutation_rate: float,
    mutation_strength: float,
    target_layers: Optional[List[str]] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Create a recipe for adding random noise to model parameters.
    
    Args:
        individual: Individual to mutate
        mutation_rate: Probability of mutating each parameter
        mutation_strength: Standard deviation of the noise
        target_layers: Optional list of layer patterns to target
        
    Returns:
        Mutation recipe dictionary
    """
    model_path = individual.model_config.get("base_model", individual.model_path)
    
    # Default to all layers if none specified
    if target_layers is None:
        # This regex captures all model parameters
        target_layers = [".*"]
    
    # Create base recipe
    recipe = {
        "models": {
            "base": {
                "path": model_path,
                "type": "standard"
            }
        },
        "merge": [],
        "tokenizer_source": "base"
    }
    
    # Add noise mutation operations for each layer pattern
    for layer_pattern in target_layers:
        recipe["merge"].append({
            "op": "gaussian_noise",
            "model": "base",
            "regex": layer_pattern,
            "prob": mutation_rate,
            "sigma": mutation_strength
        })
    
    return recipe


def _create_layer_dropout_recipe(
    individual: Individual, 
    dropout_rate: float,
    layers_to_consider: Optional[List[str]] = None,
    substitute_model_path: Optional[str] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Create a recipe for randomly dropping out model layers.
    
    Args:
        individual: Individual to mutate
        dropout_rate: Probability of dropping out each layer
        layers_to_consider: Optional list of layer patterns to consider
        substitute_model_path: Path to model to use for dropped layers
        
    Returns:
        Mutation recipe dictionary
    """
    model_path = individual.model_config.get("base_model", individual.model_path)
    
    # If no substitute model provided, use one from the config if available
    if substitute_model_path is None and "parent_models" in individual.model_config:
        parent_models = individual.model_config["parent_models"]
        if len(parent_models) > 0:
            substitute_model_path = parent_models[0]  # Use first parent
    
    # If still no substitute, we can't do layer dropout
    if substitute_model_path is None:
        logger.warning("Cannot perform layer dropout mutation without substitute model")
        return _create_noise_mutation_recipe(individual, 0.05, 0.001)
    
    # Default layer patterns if none provided
    if layers_to_consider is None:
        # This covers typical transformer layers
        layers_to_consider = [
            ".*transformer.h.[0-9]+.*",  # For transformer models
            ".*encoder.layer.[0-9]+.*",  # For encoder models
            ".*decoder.layer.[0-9]+.*"   # For decoder models
        ]
    
    # Create base recipe
    recipe = {
        "models": {
            "base": {
                "path": model_path,
                "type": "standard"
            },
            "substitute": {
                "path": substitute_model_path,
                "type": "standard"
            }
        },
        "merge": [],
        "tokenizer_source": "base"
    }
    
    # For each layer pattern, create a potential layer dropout operation
    for layer_pattern in layers_to_consider:
        # Only apply if random number is below dropout rate
        if random.random() < dropout_rate:
            recipe["merge"].append({
                "op": "use_layers_from",
                "from_model": "substitute", 
                "regex": layer_pattern
            })
    
    # If no layers were selected for dropout, add a small noise mutation instead
    if len(recipe["merge"]) == 0:
        recipe["merge"].append({
            "op": "gaussian_noise",
            "model": "base",
            "regex": ".*",
            "prob": 0.05,
            "sigma": 0.001
        })
    
    return recipe


def _mutate_recipe_params(
    individual: Individual, 
    mutation_strength: float,
    **kwargs
) -> Dict[str, Any]:
    """
    Mutate the parameters in an existing merge recipe.
    
    Args:
        individual: Individual to mutate
        mutation_strength: Strength of mutation effect
        
    Returns:
        Mutated recipe dictionary
    """
    # Get the original recipe if available
    if "recipe_path" in individual.merge_params and os.path.exists(individual.merge_params["recipe_path"]):
        with open(individual.merge_params["recipe_path"], 'r') as f:
            recipe = yaml.safe_load(f)
    else:
        # Create a simple recipe if none exists
        model_path = individual.model_config.get("base_model", individual.model_path)
        recipe = {
            "models": {
                "base": {
                    "path": model_path,
                    "type": "standard"
                }
            },
            "merge": [
                {
                    "op": "gaussian_noise",
                    "model": "base",
                    "regex": ".*",
                    "prob": 0.05, 
                    "sigma": 0.001
                }
            ],
            "tokenizer_source": "base"
        }
    
    # Mutate numeric parameters in the recipe
    if "merge" in recipe:
        for operation in recipe["merge"]:
            for key, value in operation.items():
                # Only mutate numeric values
                if isinstance(value, (int, float)) and key not in ["op", "from_model", "to_model"]:
                    # Apply mutation with specified strength
                    if key in ["prob", "sigma", "weight"]:  # Special handling for probabilities
                        # For probabilities, ensure we stay in [0, 1]
                        delta = random.uniform(-mutation_strength, mutation_strength)
                        operation[key] = max(0, min(1, value + delta))
                    else:
                        # For other numeric values, apply relative mutation
                        factor = 1 + random.uniform(-mutation_strength, mutation_strength)
                        operation[key] = value * factor
    
    return recipe


def apply_mutation(
    model_path: str,
    recipe: Dict[str, Any],
    output_dir: str
) -> str:
    """
    Apply a mutation to a model using mergekit.
    
    Args:
        model_path: Path to the model to mutate
        recipe: Mutation recipe
        output_dir: Directory to save the mutated model
        
    Returns:
        Path to the mutated model
    """
    # Ensure the model path is set correctly
    for model_key, model_info in recipe["models"].items():
        if model_info["path"] == "base_model_placeholder":
            recipe["models"][model_key]["path"] = model_path
    
    # Create a merge config
    merge_config = MergeRecipe.parse_obj(recipe).to_merge_config()
    merge_config.out_path = output_dir
    
    # Execute the mutation
    logger.info(f"Applying mutation to model: {model_path} -> {output_dir}")
    merge.run_merge(merge_config)
    logger.info(f"Mutation completed: {output_dir}")
    
    return output_dir