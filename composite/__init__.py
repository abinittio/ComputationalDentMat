"""ComputationalDentMat — physics-based dental composite property predictor."""

from .components import ALL_COMPONENTS, RESINS, FILLERS, COUPLING_AGENTS, PHOTOINITIATORS, get_density, list_components
from .physics import (
    mass_to_volume_fractions,
    rule_of_mixtures_density,
    estimate_polymerisation_shrinkage,
    cured_density,
    mass_required,
    filler_volume_fraction,
    classify_composite,
)
from .bayesian import monte_carlo_density, mass_uncertainty, MIXING_PRIORS

__all__ = [
    "ALL_COMPONENTS", "RESINS", "FILLERS", "COUPLING_AGENTS", "PHOTOINITIATORS",
    "get_density", "list_components",
    "mass_to_volume_fractions", "rule_of_mixtures_density",
    "estimate_polymerisation_shrinkage", "cured_density",
    "mass_required", "filler_volume_fraction", "classify_composite",
    "monte_carlo_density", "mass_uncertainty", "MIXING_PRIORS",
]
