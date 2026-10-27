"""Frequent pattern extraction for multiple disease detection."""
from .apriori import apriori
from .classifier import AssociativeClassifier
from .fpgrowth import fpgrowth
from .rules import Rule, generate_rules, prune_redundant

__all__ = ["apriori", "fpgrowth", "generate_rules", "prune_redundant", "Rule", "AssociativeClassifier"]
__version__ = "1.0.0"
