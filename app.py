"""
ComputationalDentMat — Dental Composite Property Predictor
==========================================================
Physics-based prediction of cured composite density and mass requirement.

Inputs : component ratios (wt%), cavity volume, mixing method
Outputs: density prediction with Bayesian confidence intervals,
         mass required ± uncertainty, polymerisation shrinkage estimate,
         composite classification
"""

import numpy as np
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from composite import (
    ALL_COMPONENTS, RESINS, FILLERS, COUPLING_AGENTS, PHOTOINITIATORS,
    list_components,
    mass_to_volume_fractions,
    rule_of_mixtures_density,
    estimate_polymerisation_shrinkage,
    cured_density,
    mass_required,
    filler_volume_fraction,
    classify_composite,
    monte_carlo_density,
    mass_uncertainty,
    MIXING_PRIORS,
)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ComputationalDentMat",
    page_icon="🦷",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
  .main-title { font-size:2.1rem; font-weight:800; color:#1e3a8a; }
  .subtitle   { font-size:1rem; color:#64748b; margin-bottom:1.2rem; }
  .card { background:#eff6ff; border-left:4px solid #3b82f6;
          border-radius:10px; padding:14px 18px; margin-bottom:10px; }
  .warn { background:#fff7ed; border-left:4px solid #f97316;
          border-radius:8px; padding:10px 14px; font-size:0.9rem; }
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-title">🦷 ComputationalDentMat</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="subtitle">Physics-based prediction of dental composite density '
    'and fill-mass with Bayesian uncertainty quantification.</p>',
    unsafe_allow_html=True,
)

# ── Example formulations ──────────────────────────────────────────────────────
EXAMPLES = {
    "Filtek Z250 (3M) — microhybrid": {
        "components": {
            "BisGMA":                          0.09,
            "UDMA":                            0.07,
            "BisEMA":                          0.08,
            "Barium aluminosilicate glass":    0.60,
            "Zirconia (ZrO₂)":                0.04,
            "Quartz (SiO₂)":                  0.10,
            "γ-MPS silane":                    0.015,
            "Camphorquinone (CQ)":             0.005,
        },
        "particle_size_um": 0.6,
    },
    "Filtek Supreme Ultra (3M) — nanofilled": {
        "components": {
            "BisGMA":                          0.10,
            "BisEMA":                          0.10,
            "UDMA":                            0.05,
            "Zirconia-silica (ZrO₂/SiO₂)":   0.595,
            "Fumed silica (Aerosil)":          0.13,
            "γ-MPS silane":                    0.015,
            "Camphorquinone (CQ)":             0.005,
            "DMAEMA co-initiator":             0.005,
        },
        "particle_size_um": 0.02,
    },
    "Experimental low-shrinkage (BisEMA/DCDMA)": {
        "components": {
            "BisEMA":                          0.15,
            "DCDMA":                           0.15,
            "Barium aluminosilicate glass":    0.60,
            "Fumed silica (Aerosil)":          0.06,
            "γ-MPS silane":                    0.02,
            "Lucirin TPO":                     0.005,
            "Camphorquinone (CQ)":             0.005,
            "DMAEMA co-initiator":             0.005,
        },
        "particle_size_um": 0.7,
    },
}

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Quick Load")
    example_choice = st.selectbox("Load example", ["— custom input —"] + list(EXAMPLES.keys()))

    st.divider()
    st.markdown("### Cavity & Process")
    cavity_vol = st.number_input("Cavity volume (mm³)", value=50.0, min_value=1.0,
                                  step=5.0, help="Typical Class I: 20–80 mm³")
    mixing_method = st.selectbox("Mixing method", list(MIXING_PRIORS.keys()),
                                  index=1, help="Affects void content prior")
    particle_size = st.number_input("Mean filler particle size (µm)",
                                     value=0.6, min_value=0.01, step=0.1,
                                     help="Used for composite classification only")

    st.divider()
    st.markdown("### Composition (wt fractions)")
    st.caption("Must sum to 1.0. Add components then adjust sliders.")

    run = st.button("▶  Calculate", type="primary", use_container_width=True)

# ── Component input ───────────────────────────────────────────────────────────
if example_choice != "— custom input —":
    preset = EXAMPLES[example_choice]
    init_components = preset["components"]
    particle_size = preset["particle_size_um"]
else:
    init_components = {
        "BisGMA":                       0.20,
        "TEGDMA":                       0.05,
        "Barium aluminosilicate glass": 0.72,
        "γ-MPS silane":                 0.02,
        "Camphorquinone (CQ)":          0.01,
    }

st.markdown("#### Formulation Input")
st.caption("Adjust wt fractions below (must sum to 1.0).")

col_left, col_right = st.columns([2, 1])

with col_left:
    n_comp = st.number_input("Number of components", min_value=2, max_value=12,
                               value=len(init_components), step=1)

    component_names = list(ALL_COMPONENTS.keys())
    composition = {}
    init_keys = list(init_components.keys())
    init_vals = list(init_components.values())

    cols = st.columns(2)
    for i in range(int(n_comp)):
        default_name = init_keys[i] if i < len(init_keys) else component_names[0]
        default_val  = float(init_vals[i]) if i < len(init_vals) else 0.05
        with cols[i % 2]:
            name = st.selectbox(f"Component {i+1}", component_names,
                                 index=component_names.index(default_name)
                                 if default_name in component_names else 0,
                                 key=f"comp_{i}")
            val  = st.number_input(f"wt fraction", value=default_val,
                                    min_value=0.0, max_value=1.0, step=0.01,
                                    key=f"val_{i}", format="%.4f")
            composition[name] = val

with col_right:
    total_wt = sum(composition.values())
    delta = abs(total_wt - 1.0)
    if delta < 0.001:
        st.success(f"✓ Total wt fraction: {total_wt:.4f}")
    else:
        st.error(f"✗ Total: {total_wt:.4f} (must = 1.000, Δ = {delta:.4f})")

    st.markdown("##### Component info")
    for name in list(composition.keys())[:5]:
        info = ALL_COMPONENTS.get(name, {})
        st.markdown(f"**{name}**")
        st.caption(f"ρ = {info.get('density','?')} g/cm³ · {info.get('group','')}")
        st.caption(info.get('desc', ''))

if not run:
    st.info("👈 Set composition in the sidebar + above, then click **Calculate**.")
    st.stop()

# ── Validate ──────────────────────────────────────────────────────────────────
if delta > 0.02:
    st.error("Weight fractions must sum to 1.0 (±0.02). Please adjust.")
    st.stop()

# Normalise to exactly 1
total = sum(composition.values())
composition = {k: v / total for k, v in composition.items()}

# ── Extract densities & shrinkage data ────────────────────────────────────────
densities = {k: ALL_COMPONENTS[k]["density"] for k in composition if k in ALL_COMPONENTS}
shrinkage_map = {k: ALL_COMPONENTS[k].get("shrinkage_per_wt", 0.0)
                 for k in composition if k in ALL_COMPONENTS}

missing = [k for k in composition if k not in ALL_COMPONENTS]
if missing:
    st.warning(f"Unknown components (skipped): {missing}")
    composition = {k: v for k, v in composition.items() if k in ALL_COMPONENTS}

filler_names = list_components("Filler")

# ── Compute ───────────────────────────────────────────────────────────────────
rho_theory    = rule_of_mixtures_density(composition, densities)
shrink_pct    = estimate_polymerisation_shrinkage(composition, shrinkage_map)
rho_cured_det = cured_density(rho_theory, shrink_pct)
mass_det      = mass_required(cavity_vol, rho_cured_det)

vol_fracs     = mass_to_volume_fractions(composition, densities)
phi_filler    = filler_volume_fraction(composition, densities, filler_names)
filler_wt     = sum(composition.get(f, 0.0) for f in filler_names)
comp_type     = classify_composite(filler_wt, particle_size)

mc            = monte_carlo_density(rho_theory, shrink_pct, densities,
                                     filler_names, mixing_method)
mass_mc       = mass_uncertainty(mc, cavity_vol)

# ── Metric cards ──────────────────────────────────────────────────────────────
st.divider()
m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Theoretical ρ",    f"{rho_theory:.4f} g/cm³",  help="Rule of mixtures (uncured)")
m2.metric("Predicted ρ (cured)", f"{mc['mean']:.4f} g/cm³", delta=f"±{mc['std']:.4f}")
m3.metric("Shrinkage (est.)", f"{shrink_pct:.2f}%",       help="Volumetric polymerisation shrinkage")
m4.metric("Filler vol %",     f"{phi_filler*100:.1f}%",   help="Volume fraction of inorganic filler")
m5.metric("Mass needed",      f"{mass_mc['mean_g']:.4f} g",
          delta=f"95% CI: {mass_mc['ci_95_lo_g']:.4f}–{mass_mc['ci_95_hi_g']:.4f}")

st.markdown(f'<div class="card"><b>Composite type:</b> {comp_type}</div>', unsafe_allow_html=True)

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["📊 Composition Breakdown", "📈 Uncertainty Distribution", "🔬 Detailed Results"])

# ── Tab 1: Composition ────────────────────────────────────────────────────────
with tab1:
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("#### Mass Fractions (wt%)")
        import pandas as pd
        rows = []
        for comp, wf in composition.items():
            info = ALL_COMPONENTS[comp]
            rows.append({
                "Component":  comp,
                "Group":      info["group"],
                "wt %":       f"{wf*100:.2f}",
                "vol %":      f"{vol_fracs.get(comp, 0)*100:.2f}",
                "ρ (g/cm³)":  info["density"],
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    with col_b:
        fig, axes = plt.subplots(1, 2, figsize=(9, 4))
        fig.patch.set_facecolor("white")

        COLOURS = ["#3b82f6","#ef4444","#f59e0b","#8b5cf6","#22c55e","#ec4899",
                   "#06b6d4","#84cc16","#f43f5e","#a855f7","#14b8a6","#fb923c"]

        labels_wt = list(composition.keys())
        vals_wt   = [composition[k]*100 for k in labels_wt]
        axes[0].pie(vals_wt, labels=[l[:14] for l in labels_wt],
                    colors=COLOURS[:len(labels_wt)], autopct="%1.1f%%",
                    startangle=90, textprops={"fontsize": 7})
        axes[0].set_title("Mass fractions (wt%)", fontweight="bold", fontsize=9)

        labels_vf = list(vol_fracs.keys())
        vals_vf   = [vol_fracs[k]*100 for k in labels_vf]
        axes[1].pie(vals_vf, labels=[l[:14] for l in labels_vf],
                    colors=COLOURS[:len(labels_vf)], autopct="%1.1f%%",
                    startangle=90, textprops={"fontsize": 7})
        axes[1].set_title("Volume fractions (vol%)", fontweight="bold", fontsize=9)

        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

# ── Tab 2: MC distribution ────────────────────────────────────────────────────
with tab2:
    st.markdown("#### Monte Carlo Density Distribution (N=10,000)")
    st.caption(f"Mixing method: **{mixing_method}** · Cavity: **{cavity_vol:.0f} mm³**")

    fig2, axes2 = plt.subplots(1, 3, figsize=(13, 4))
    fig2.patch.set_facecolor("white")

    ax_d, ax_s, ax_m = axes2

    # Density histogram
    ax_d.hist(mc["density_samples"], bins=80, color="#3b82f6", alpha=0.75, edgecolor="white")
    ax_d.axvline(mc["mean"],      color="#1e3a8a", lw=2,   label=f"Mean {mc['mean']:.4f}")
    ax_d.axvline(mc["ci_95_lo"],  color="#ef4444", lw=1.5, ls="--", label="95% CI")
    ax_d.axvline(mc["ci_95_hi"],  color="#ef4444", lw=1.5, ls="--")
    ax_d.axvline(rho_theory,      color="#f59e0b", lw=1.5, ls=":", label=f"Theoretical {rho_theory:.4f}")
    ax_d.set(xlabel="Density (g/cm³)", ylabel="Count", title="Cured Density")
    ax_d.legend(fontsize=7.5); ax_d.grid(alpha=0.25)

    # Shrinkage histogram
    ax_s.hist(mc["shrinkage_samples"], bins=60, color="#f59e0b", alpha=0.75, edgecolor="white")
    ax_s.axvline(shrink_pct, color="#92400e", lw=2, label=f"Deterministic {shrink_pct:.2f}%")
    ax_s.set(xlabel="Shrinkage (%)", ylabel="Count", title="Shrinkage Distribution")
    ax_s.legend(fontsize=7.5); ax_s.grid(alpha=0.25)

    # Mass histogram
    ax_m.hist(mass_mc["mass_samples"], bins=80, color="#22c55e", alpha=0.75, edgecolor="white")
    ax_m.axvline(mass_mc["mean_g"],       color="#14532d", lw=2,   label=f"Mean {mass_mc['mean_g']:.4f}g")
    ax_m.axvline(mass_mc["ci_95_lo_g"],   color="#ef4444", lw=1.5, ls="--", label="95% CI")
    ax_m.axvline(mass_mc["ci_95_hi_g"],   color="#ef4444", lw=1.5, ls="--")
    ax_m.set(xlabel="Mass (g)", ylabel="Count", title=f"Mass Required ({cavity_vol:.0f} mm³ cavity)")
    ax_m.legend(fontsize=7.5); ax_m.grid(alpha=0.25)

    plt.tight_layout()
    st.pyplot(fig2)
    plt.close(fig2)

# ── Tab 3: Detailed results ───────────────────────────────────────────────────
with tab3:
    c1, c2 = st.columns(2)

    with c1:
        st.markdown("#### Density Summary")
        st.markdown(f"**Rule-of-mixtures (uncured):** {rho_theory:.4f} g/cm³")
        st.markdown(f"**Deterministic cured density:** {rho_cured_det:.4f} g/cm³")
        st.markdown(f"**Bayesian mean:**               {mc['mean']:.4f} ± {mc['std']:.4f} g/cm³")
        st.markdown(f"**68% CI:**  {mc['ci_68_lo']:.4f} – {mc['ci_68_hi']:.4f} g/cm³")
        st.markdown(f"**95% CI:**  {mc['ci_95_lo']:.4f} – {mc['ci_95_hi']:.4f} g/cm³")

        st.markdown("#### Shrinkage")
        st.markdown(f"**Estimated vol. shrinkage:** {shrink_pct:.2f}%")
        st.markdown('<div class="warn">⚠ Shrinkage estimated from monomer contributions. '
                    'Verify experimentally for final formulation.</div>', unsafe_allow_html=True)

    with c2:
        st.markdown(f"#### Mass Required — {cavity_vol:.0f} mm³ cavity")
        st.markdown(f"**Deterministic:** {mass_det:.4f} g")
        st.markdown(f"**Bayesian mean:** {mass_mc['mean_g']:.4f} ± {mass_mc['std_g']:.4f} g")
        st.markdown(f"**95% CI:**  {mass_mc['ci_95_lo_g']:.4f} – {mass_mc['ci_95_hi_g']:.4f} g")

        st.markdown("#### Filler Analysis")
        st.markdown(f"**Total filler wt%:** {filler_wt*100:.1f}%")
        st.markdown(f"**Total filler vol%:** {phi_filler*100:.1f}%")
        st.markdown(f"**Type:** {comp_type}")

# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "ComputationalDentMat · Physics-based dental composite predictor · "
    "[github.com/abinittio](https://github.com/abinittio) · "
    "Research use only — not clinical advice"
)
