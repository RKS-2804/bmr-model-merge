"""Crossover operations for evolutionary model merging."""

import os
import yaml
import random
import logging
import tempfile
from typing import Dict, List, Any, Optional, Tuple, Union

import torch
import numpy as np
from mergekit import merge
from mergekit.merge_recipe import MergeRecipe, ModelSliceNode, MergeLayer, ModelMergeConfig

from .population import Individual

logger = logging.getLogger(__name__)

def crossover_models(
    parent1: Individual, 
    parent2: Individual, 
    crossover_method: str = "uniform",
    crossover_prob: float = 0.5,
    output_dir: str = None,
    temp_dir: str = None,
    **kwargs
) -> Individual:
    """
    Perform crossover between two parent models to create a child model.
    
    Args:
        parent1: First parent individual
        parent2: Second parent individual
        crossover_method: Method of crossover ('uniform', 'single_point', 'sliced')
        crossover_prob: Probability of inheriting from parent1 (for uniform)
        output_dir: Directory to save the merged model (if None, will use temp dir)
        temp_dir: Directory for temporary files
        **kwargs: Additional arguments specific to crossover method
        
    Returns:
        A new Individual representing the child model
    """
    if output_dir is None:
        output_dir = tempfile.mkdtemp(prefix="evomerge_")
    
    # Create a new individual for the child
    child = Individual(
        generation=max(parent1.generation, parent2.generation) + 1,
        model_config={},
        merge_params={}
    )
    
    # Copy basic params from parents
    for key in ["model_type", "tokenizer_path", "model_family"]:
        if key in parent1.model_config:
            child.model_config[key] = parent1.model_config[key]
            
    # Determine the crossover method to use
    if crossover_method == "uniform":
        recipe = _uniform_crossover(parent1, parent2, crossover_prob, **kwargs)
    elif crossover_method == "single_point":
        recipe = _single_point_crossover(parent1, parent2, **kwargs)
    elif crossover_method == "sliced":
        recipe = _sliced_crossover(parent1, parent2, **kwargs)
    elif crossover_method == "weighted":
        recipe = _weighted_crossover(parent1, parent2, **kwargs)
    else:
        raise ValueError(f"Unknown crossover method: {crossover_method}")
    
    # Create merge configuration
    merge_config = _create_merge_config(recipe, output_dir)
    
    # Save merge recipe for reference
    recipe_path = os.path.join(output_dir, "merge_recipe.yaml")
    with open(recipe_path, 'w') as f:
        yaml.dump(recipe, f)
    
    # Store merge parameters for reproduction
    child.merge_params = {
        "crossover_method": crossover_method,
        "crossover_prob": crossover_prob,
        "parent1_id": parent1.id,
        "parent2_id": parent2.id,
        "recipe_path": recipe_path
    }
    
    # Update model configuration
    child.model_config.update({
        "base_model": output_dir,
        "parent_models": [
            parent1.model_config.get("base_model", parent1.model_path),
            parent2.model_config.get("base_model", parent2.model_path)
        ]
    })
    
    # Set the model path
    child.model_path = output_dir
    
    return child


def _uniform_crossover(
    parent1: Individual, 
    parent2: Individual, 
    crossover_prob: float = 0.5,
    layer_regex: Optional[str] = None
) -> Dict[str, Any]:
    """
    Create a uniform crossover recipe between two models.
    
    In uniform crossover, each layer has a probability (crossover_prob) of 
    coming from parent1, otherwise it comes from parent2.
    
    Args:
        parent1: First parent individual
        parent2: Second parent individual
        crossover_prob: Probability of inheriting a layer from parent1
        layer_regex: Optional regex to filter which layers to consider
        
    Returns:
        Merge recipe dictionary
    """
    model1_path = parent1.model_config.get("base_model", parent1.model_path)
    model2_path = parent2.model_config.get("base_model", parent2.model_path)
    
    # Get model metadata (this is a simplified approach; in practice, you'd
    # need to inspect the actual model architecture)
    model_type = parent1.model_config.get("model_type", "standard")
    
    # Define the layers pattern based on model type
    if layer_regex is None:
        if "vlm" in model_type.lower() or "visual" in model_type.lower():
            # For vision-language models, use a pattern that includes vision-related layers
            layer_pattern = ".*"
        else:
            # For standard language models
            layer_pattern = ".*"
    else:
        layer_pattern = layer_regex
    
    # Create a merge recipe with uniform crossover
    recipe = {
        "models": {
            "model1": {
                "path": model1_path,
                "type": "standard"
            },
            "model2": {
                "path": model2_path,
                "type": "standard"
            }
        },
        "merge": [
            {
                "op": "binarize",
                "prob": crossover_prob,
                "models": ["model1", "model2"],
                "regex": layer_pattern
            }
        ],
        "tokenizer_source": "model1"
    }
    
    return recipe


def _single_point_crossover(
    parent1: Individual, 
    parent2: Individual,
    crossover_point: Optional[float] = None
) -> Dict[str, Any]:
    """
    Create a single-point crossover recipe between two models.
    
    In single-point crossover, layers before the crossover point come from
    parent1, and layers after come from parent2.
    
    Args:
        parent1: First parent individual
        parent2: Second parent individual
        crossover_point: Point at which to switch from parent1 to parent2 (0-1)
        
    Returns:
        Merge recipe dictionary
    """
    model1_path = parent1.model_config.get("base_model", parent1.model_path)
    model2_path = parent2.model_config.get("base_model", parent2.model_path)
    
    # If no crossover point specified, choose randomly
    if crossover_point is None:
        crossover_point = random.random()
    
    # Create a merge recipe with single-point crossover
    recipe = {
        "models": {
            "model1": {
                "path": model1_path,
                "type": "standard"
            },
            "model2": {
                "path": model2_path,
                "type": "standard"
            }
        },
        "merge": [
            {
                "op": "use_layers",
                "from_model": "model1",
                "from_layer_prefix": "0",
                "to_layer_prefix": str(crossover_point)
            },
            {
                "op": "use_layers",
                "from_model": "model2",
                "from_layer_prefix": str(crossover_point),
                "to_layer_prefix": "1"
            }
        ],
        "tokenizer_source": "model1"
    }
    
    return recipe


def _sliced_crossover(
    parent1: Individual, 
    parent2: Individual,
    slices: Optional[List[float]] = None,
    n_slices: int = 4
) -> Dict[str, Any]:
    """
    Create a sliced crossover recipe between two models.
    
    In sliced crossover, the model is divided into multiple slices,
    alternating between parent1 and parent2.
    
    Args:
        parent1: First parent individual
        parent2: Second parent individual
        slices: List of slice points (0-1)
        n_slices: Number of slices if not explicitly provided
        
    Returns:
        Merge recipe dictionary
    """
    model1_path = parent1.model_config.get("base_model", parent1.model_path)
    model2_path = parent2.model_config.get("base_model", parent2.model_path)
    
    # If no slices specified, create evenly spaced ones
    if slices is None:
        slices = [i / n_slices for i in range(1, n_slices)]
    
    # Ensure we include start and end points
    all_points = [0] + slices + [1]
    
    # Create a merge recipe with sliced crossover
    recipe = {
        "models": {
            "model1": {
                "path": model1_path,
                "type": "standard"
            },
            "model2": {
                "path": model2_path,
                "type": "standard"
            }
        },
        "merge": [],
        "tokenizer_source": "model1"
    }
    
    # Alternate between parents for each slice
    current_model = "model1"
    for i in range(len(all_points) - 1):
        recipe["merge"].append({
            "op": "use_layers",
            "from_model": current_model,
            "from_layer_prefix": str(all_points[i]),
            "to_layer_prefix": str(all_points[i+1])
        })
        # Switch to the other parent
        current_model = "model2" if current_model == "model1" else "model1"
    
    return recipe


def _weighted_crossover(
    parent1: Individual, 
    parent2: Individual,
    weights: Optional[Dict[str, List[float]]] = None,
    weight_factor: Optional[float] = None
) -> Dict[str, Any]:
    """
    Create a weighted crossover recipe between two models.
    
    In weighted crossover, each layer is a weighted combination of 
    the corresponding layers from both parents.
    
    Args:
        parent1: First parent individual
        parent2: Second parent individual
        weights: Dictionary mapping layer patterns to weight lists [parent1_weight, parent2_weight]
        weight_factor: If provided, will use parent fitness to determine weights
        
    Returns:
        Merge recipe dictionary
    """
    model1_path = parent1.model_config.get("base_model", parent1.model_path)
    model2_path = parent2.model_config.get("base_model", parent2.model_path)
    
    # If weight factor provided, compute weights based on fitness
    if weight_factor is not None and parent1.overall_fitness > 0 and parent2.overall_fitness > 0:
        p1_norm = parent1.overall_fitness / (parent1.overall_fitness + parent2.overall_fitness)
        p2_norm = 1 - p1_norm
        
        # Apply the weight factor to make it more or less extreme
        p1_weight = p1_norm ** weight_factor
        p2_weight = p2_norm ** weight_factor
        
        # Renormalize
        total = p1_weight + p2_weight
        p1_weight /= total
        p2_weight /= total
        
        default_weights = [p1_weight, p2_weight]
    else:
        # Default to equal weights
        default_weights = [0.5, 0.5]
    
    # Use provided weights or defaults
    if weights is None:
        weights = {".*": default_weights}
    
    # Create a merge recipe with weighted crossover
    recipe = {
        "models": {
            "model1": {
                "path": model1_path,
                "type": "standard"
            },
            "model2": {
                "path": model2_path,
                "type": "standard"
            }
        },
        "merge": [],
        "tokenizer_source": "model1"
    }
    
    # Add merging operations for each pattern
    for pattern, weight_pair in weights.items():
        recipe["merge"].append({
            "op": "linear",
            "models": ["model1", "model2"],
            "weights": weight_pair,
            "regex": pattern
        })
    
    return recipe


def _create_merge_config(recipe: Dict[str, Any], output_dir: str) -> ModelMergeConfig:
    """
    Convert a recipe dictionary to a mergekit ModelMergeConfig.
    
    Args:
        recipe: Merge recipe dictionary
        output_dir: Output directory for the merged model
        
    Returns:
        ModelMergeConfig object for mergekit
    """
    # This is a simplified version; in practice, you would need to
    # properly convert the recipe to mergekit format
    config = MergeRecipe.parse_obj(recipe).to_merge_config()
    config.out_path = output_dir
    return config


def execute_merge(config: ModelMergeConfig) -> str:
    """
    Execute a model merge operation using mergekit.
    
    Args:
        config: ModelMergeConfig object
        
    Returns:
        Path to the merged model
    """
    logger.info(f"Starting model merge: {config.out_path}")
    merge.run_merge(config)
    logger.info(f"Model merge completed: {config.out_path}")
    return config.out_path