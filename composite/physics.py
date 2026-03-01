"""
Physics engine for dental composite property prediction.

Core model: Rule of Mixtures (Voigt model)
  ρ_composite = Σ (φᵢ · ρᵢ)      [volume-fraction weighted]

Mass fraction → volume fraction conversion:
  φᵢ = (wᵢ / ρᵢ) / Σ (wⱼ / ρⱼ)

Post-cure corrections:
  ρ_cured = ρ_composite × (1 + ΔV_shrinkage)   # density increases on cure
  mass_needed (g) = ρ_cured × V_cavity (cm³)
"""

import numpy as np


def mass_to_volume_fractions(composition: dict[str, float],
                              densities: dict[str, float]) -> dict[str, float]:
    """
    Convert mass fractions (wt%) to volume fractions.

    Parameters
    ----------
    composition : {component_name: wt_fraction}   (must sum to 1.0)
    densities   : {component_name: density g/cm³}

    Returns
    -------
    {component_name: volume_fraction}
    """
    # partial volumes
    partial = {k: composition[k] / densities[k] for k in composition}
    total = sum(partial.values())
    return {k: v / total for k, v in partial.items()}


def rule_of_mixtures_density(composition: dict[str, float],
                              densities: dict[str, float]) -> float:
    """
    Theoretical density by rule of mixtures (volume-weighted).

    Parameters
    ----------
    composition : {component: wt_fraction}  (must sum ≈ 1)
    densities   : {component: g/cm³}

    Returns
    -------
    ρ_theoretical in g/cm³
    """
    vol_fracs = mass_to_volume_fractions(composition, densities)
    return sum(vol_fracs[k] * densities[k] for k in vol_fracs)


def estimate_polymerisation_shrinkage(composition: dict[str, float],
                                      shrinkage_per_wt: dict[str, float]) -> float:
    """
    Estimate volumetric polymerisation shrinkage (%).

    Uses linear mixing of per-monomer shrinkage contributions weighted by
    resin mass fraction. Filler dilutes shrinkage proportionally.

    Reference: Braga et al., Dent Mater 2005 (linear additivity of C=C groups)

    Returns
    -------
    shrinkage_vol_percent  (e.g. 2.4 means 2.4% volumetric shrinkage)
    """
    total = 0.0
    for component, wt_frac in composition.items():
        s = shrinkage_per_wt.get(component, 0.0)
        total += wt_frac * s
    return total * 100.0   # as %


def cured_density(rho_theoretical: float, shrinkage_pct: float) -> float:
    """
    Density of cured composite.

    Polymerisation shrinkage reduces volume → increases density:
        ρ_cured = ρ_uncured / (1 − ΔV/V)

    Parameters
    ----------
    rho_theoretical : uncured theoretical density (g/cm³)
    shrinkage_pct   : volumetric shrinkage (%)

    Returns
    -------
    ρ_cured in g/cm³
    """
    frac = shrinkage_pct / 100.0
    return rho_theoretical / (1.0 - frac)


def mass_required(cavity_volume_mm3: float, density_g_cm3: float) -> float:
    """
    Mass of composite needed to fill a cavity.

    Parameters
    ----------
    cavity_volume_mm3 : cavity volume in mm³
    density_g_cm3     : density of cured composite

    Returns
    -------
    mass in grams
    """
    volume_cm3 = cavity_volume_mm3 / 1000.0
    return density_g_cm3 * volume_cm3


def filler_volume_fraction(composition: dict[str, float],
                            densities: dict[str, float],
                            filler_names: list[str]) -> float:
    """
    Total filler volume fraction — key parameter correlating with
    mechanical properties (strength ∝ φ_filler for well-bonded fillers).
    """
    vol_fracs = mass_to_volume_fractions(composition, densities)
    return sum(vol_fracs.get(f, 0.0) for f in filler_names)


def classify_composite(filler_wt_total: float,
                        particle_size_um: float | None) -> str:
    """
    Classify composite type based on filler loading and particle size.
    Based on Ferracane 2011 classification (Dent Mater).
    """
    if filler_wt_total >= 0.85:
        return "Packable / condensable composite (heavy-filled)"
    elif filler_wt_total >= 0.77:
        if particle_size_um is not None and particle_size_um <= 0.1:
            return "Nanofilled composite"
        elif particle_size_um is not None and particle_size_um <= 1.0:
            return "Nanohybrid composite"
        else:
            return "Microhybrid composite"
    elif filler_wt_total >= 0.65:
        return "Flowable composite (mid-fill)"
    elif filler_wt_total >= 0.45:
        return "Low-viscosity / flowable composite"
    else:
        return "Experimental / unfilled resin"
