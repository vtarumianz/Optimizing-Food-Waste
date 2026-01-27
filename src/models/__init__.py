"""
Mathematical Models for Food Waste Optimization

- MarkovChainModel: Models behavioral state transitions
- SIRModel: Models spread of food waste awareness
"""

from .markov_chain import MarkovChainModel
from .sir_model import SIRModel

__all__ = ["MarkovChainModel", "SIRModel"]
