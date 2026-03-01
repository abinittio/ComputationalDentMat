"""
Dental composite component database.

Densities sourced from:
  - Manufacturer TDS / SDS sheets
  - Sabbagh et al., Dent Mater 2004 (resin monomers)
  - Antonucci & Stansbury, ACS Symp 1997 (BisGMA systems)
  - Ilie & Hickel, Clin Oral Invest 2011 (filler characterisation)

All densities in g/cm³.
"""

# ── Resin monomers ────────────────────────────────────────────────────────────
RESINS = {
    "BisGMA":   {"density": 1.15, "shrinkage_per_wt": 0.038,
                 "desc": "Bisphenol A-glycidyl methacrylate — high viscosity base monomer"},
    "TEGDMA":   {"density": 1.08, "shrinkage_per_wt": 0.067,
                 "desc": "Triethylene glycol dimethacrylate — reactive diluent, high shrinkage"},
    "UDMA":     {"density": 1.10, "shrinkage_per_wt": 0.041,
                 "desc": "Urethane dimethacrylate — lower shrinkage than BisGMA"},
    "BisEMA":   {"density": 1.12, "shrinkage_per_wt": 0.021,
                 "desc": "Bisphenol A ethoxylated dimethacrylate — reduced shrinkage diluent"},
    "HEMA":     {"density": 1.07, "shrinkage_per_wt": 0.082,
                 "desc": "2-Hydroxyethyl methacrylate — hydrophilic, high shrinkage"},
    "DCDMA":    {"density": 1.13, "shrinkage_per_wt": 0.018,
                 "desc": "Dicarbonate dimethacrylate — low-shrinkage experimental monomer"},
    "TMPTMA":   {"density": 1.06, "shrinkage_per_wt": 0.071,
                 "desc": "Trimethylolpropane trimethacrylate — trifunctional crosslinker"},
}

# ── Inorganic fillers ─────────────────────────────────────────────────────────
FILLERS = {
    "Quartz (SiO₂)":                 {"density": 2.65,
                                       "desc": "Crystalline silica; ground to 0.1–10 µm"},
    "Barium aluminosilicate glass":  {"density": 2.80,
                                       "desc": "Radiopaque Ba glass; most common filler ~0.4–5 µm"},
    "Strontium aluminosilicate glass": {"density": 3.05,
                                        "desc": "Radiopaque Sr glass; used in nanohybrid composites"},
    "Borosilicate glass":            {"density": 2.23,
                                       "desc": "Low-density glass; used in flowable composites"},
    "Zirconia (ZrO₂)":              {"density": 5.68,
                                       "desc": "High radiopacity, nanocluster applications"},
    "Zirconia-silica (ZrO₂/SiO₂)": {"density": 4.00,
                                       "desc": "Zirconia–silica nanoclusters (e.g. 3M Filtek Supreme)"},
    "Alumina (Al₂O₃)":             {"density": 3.97,
                                       "desc": "High hardness filler; abrasion resistance"},
    "Hydroxyapatite":               {"density": 3.16,
                                       "desc": "Bioactive filler; remineralisation applications"},
    "Prepolymerised filler (PPF)":  {"density": 1.80,
                                       "desc": "Ground resin composite; reduces net shrinkage"},
    "Fumed silica (Aerosil)":       {"density": 2.20,
                                       "desc": "Nanofiller 7–40 nm; rheology modifier"},
}

# ── Coupling / surface treatment agents ──────────────────────────────────────
COUPLING_AGENTS = {
    "γ-MPS silane":  {"density": 1.04,
                      "desc": "γ-Methacryloxypropyltrimethoxysilane — standard filler silane"},
    "APTES silane":  {"density": 0.95,
                      "desc": "3-Aminopropyltriethoxysilane"},
    "VTMS silane":   {"density": 0.97,
                      "desc": "Vinyltrimethoxysilane — hydrolysis-resistant"},
}

# ── Photoinitiator / co-initiator systems ────────────────────────────────────
PHOTOINITIATORS = {
    "Camphorquinone (CQ)":  {"density": 1.00,
                              "desc": "Classical visible-light initiator (468 nm); 0.2–1 wt%"},
    "DMAEMA co-initiator":  {"density": 0.93,
                              "desc": "Dimethylaminoethyl methacrylate; used with CQ"},
    "Lucirin TPO":          {"density": 1.18,
                              "desc": "Acylphosphine oxide; UV/violet light (385–410 nm)"},
    "Ivocerin":             {"density": 1.12,
                              "desc": "Germanium-based initiator (Ivoclar); violet LED curing"},
}

# ── Convenience lookup ────────────────────────────────────────────────────────
ALL_COMPONENTS: dict[str, dict] = {}
for _group, _items in [
    ("Resin", RESINS),
    ("Filler", FILLERS),
    ("Coupling agent", COUPLING_AGENTS),
    ("Photoinitiator", PHOTOINITIATORS),
]:
    for name, props in _items.items():
        ALL_COMPONENTS[name] = {**props, "group": _group}


def get_density(component_name: str) -> float | None:
    """Return density (g/cm³) for a named component, or None if not found."""
    entry = ALL_COMPONENTS.get(component_name)
    return entry["density"] if entry else None


def list_components(group: str | None = None) -> list[str]:
    """Return component names, optionally filtered by group."""
    if group is None:
        return list(ALL_COMPONENTS.keys())
    return [k for k, v in ALL_COMPONENTS.items() if v["group"] == group]
