#!/usr/bin/env python3
"""
NEXUS — High-Power Rocketry Simulation Suite v3.0
═══════════════════════════════════════════════════
Comprehensive internal ballistics, structural, and propellant chemistry
analysis platform for solid rocket motor engineering.

Architecture
────────────
  1. Chemical / Material Input Layer    — stoichiometry, thermal data, safety
  2. Geometric Regression Engine        — multi-segment BATES, mm-by-mm burn
  3. Internal Ballistics Solver         — QSS + Saint-Robert + 2-phase η
  4. Structural Margin Module           — Lamé / thin-wall, Von Mises, SF
  5. Data Export                        — .ENG thrust curve + .JSON config
  6. Knowledge Base                     — textbook equations in LaTeX
  7. AI Engineering Assistant           — context-aware Anthropic chatbot

Usage
─────
  pip install streamlit plotly numpy pandas anthropic
  streamlit run ballistics_app.py

References
──────────
  Sutton & Biblarz (2016)  Rocket Propulsion Elements, 9th ed.
  Kubota (2007)            Propellants and Explosives, 2nd ed.
  Nakka, R. (2023)         Experimental Rocketry website
  Kuo & Summerfield (1984) Fundamentals of Solid-Propellant Combustion
"""

# ═══════════════════════════════════════════════════════════════════════════
# IMPORTS
# ═══════════════════════════════════════════════════════════════════════════
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple
import json
import math
import random
from datetime import datetime

try:
    import anthropic
    _ANTHROPIC_OK = True
except ImportError:
    _ANTHROPIC_OK = False

# ═══════════════════════════════════════════════════════════════════════════
# PHYSICAL CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════
G0     = 9.80665      # m/s²  standard gravity
R_UNIV = 8.31446      # J/(mol·K)
MPA    = 1_000_000.0  # Pa per MPa

# Plotly dark sci-fi template
_BG  = '#03080f'
_SRF = '#0a1628'
_GRD = '#142239'
_CYN = '#00d4ff'
_ORG = '#ff6b35'
_GRN = '#39ff14'
_RED = '#ff2d55'
_TXT = '#c9d1d9'
_DIM = '#8b949e'
_PRP = '#a78bfa'

# Pre-computed rgba fill variants (alpha=0.08)
_FILL = {
    '#00d4ff': 'rgba(0,212,255,0.08)',
    '#ff6b35': 'rgba(255,107,53,0.08)',
    '#39ff14': 'rgba(57,255,20,0.08)',
    '#a78bfa': 'rgba(167,139,250,0.08)',
}

NEXUS_LAYOUT = dict(
    paper_bgcolor=_BG, plot_bgcolor=_SRF,
    font=dict(color=_TXT, family='Share Tech Mono, monospace', size=11),
    margin=dict(l=50, r=20, t=40, b=40),
    legend=dict(bgcolor='rgba(10,22,40,0.8)', bordercolor=_GRD, borderwidth=1),
)
NEXUS_AXIS = dict(
    gridcolor=_GRD, zerolinecolor='#1a2d4a',
    linecolor='#1a2d4a', tickcolor=_DIM, title_font_color=_DIM,
)


# ═══════════════════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class PropellantData:
    name:         str
    abbr:         str
    composition:  Dict[str, float]   # component: mass fraction
    rho:          float              # density  kg/m³
    Tf:           float              # adiabatic flame temperature  K
    gamma:        float              # specific heat ratio Cp/Cv
    MW:           float              # mean molecular weight of products  g/mol
    cstar:        float              # characteristic velocity  m/s
    Isp_vac:      float              # vacuum Isp  s
    Isp_sl:       float              # sea-level Isp  s
    burn_a:       float              # Saint-Robert a-coefficient  mm/s·MPa^-n
    burn_n:       float              # pressure exponent  (must be < 1)
    P_min:        float              # valid pressure range — low   MPa
    P_max:        float              # valid pressure range — high  MPa
    # Stoichiometric / chemical
    OF_ratio:     float              # oxidizer-to-fuel mass ratio
    OB_pct:       float              # oxygen balance  %
    equiv_ratio:  float              # equivalence ratio  φ
    # Thermal safety
    cure_T:       str
    cure_t:       str
    decomp_onset: float              # °C, first significant decomposition
    decomp_crit:  float              # °C, CRITICAL / auto-ignition
    crit_label:   str
    process_note: str
    refs:         str


@dataclass
class CasingMaterial:
    name:    str
    E_GPa:   float    # Young's modulus  GPa
    Sy_MPa:  float    # yield strength   MPa
    Su_MPa:  float    # ultimate tensile MPa
    rho:     float    # density  kg/m³
    comment: str


# ═══════════════════════════════════════════════════════════════════════════
# PROPELLANT DATABASE
# ═══════════════════════════════════════════════════════════════════════════

PROPELLANTS: Dict[str, PropellantData] = {
    'KNSB': PropellantData(
        name='Potassium Nitrate / Sorbitol (65/35)',
        abbr='KNSB',
        composition={'KNO₃': 0.65, 'Sorbitol C₆H₁₄O₆': 0.35},
        rho=1879, Tf=1720, gamma=1.131, MW=38.90,
        cstar=889, Isp_vac=165, Isp_sl=130,
        burn_a=8.26, burn_n=0.319, P_min=1.0, P_max=10.0,
        OF_ratio=1.857, OB_pct=-10.4, equiv_ratio=1.12,
        cure_T='95–115 °C (melt-cast, solidification only)',
        cure_t='0.5–2 h',
        decomp_onset=339.0, decomp_crit=400.0,
        crit_label='400 °C → KNO₃ decomposes: 2KNO₃ → 2KNO₂ + O₂',
        process_note='Melt-cast system. No chemical curative required. Process at ≤200 °C. Store in cool, dry, static-safe environment.',
        refs='Nakka (2023); Kubota (2007); Sutton & Biblarz (2016)',
    ),
    'KNSU': PropellantData(
        name='Potassium Nitrate / Sucrose (65/35)',
        abbr='KNSU',
        composition={'KNO₃': 0.65, 'Sucrose C₁₂H₂₂O₁₁': 0.35},
        rho=1889, Tf=1733, gamma=1.134, MW=38.50,
        cstar=908, Isp_vac=164, Isp_sl=128,
        burn_a=7.96, burn_n=0.321, P_min=1.0, P_max=10.0,
        OF_ratio=1.857, OB_pct=-12.1, equiv_ratio=1.14,
        cure_T='160–185 °C (near sucrose melt point 186 °C)',
        cure_t='0.25–1 h',
        decomp_onset=300.0, decomp_crit=380.0,
        crit_label='380 °C — thermal runaway onset',
        process_note='Higher melt temp than KNSB. Use electric heating only; no open flame. Lower decomposition onset than KNSB — extra caution required.',
        refs='Nakka (2023); Sutton & Biblarz (2016)',
    ),
    'APCP_STD': PropellantData(
        name='AP/HTPB/Al Standard APCP (70/18/12)',
        abbr='APCP-STD',
        composition={'NH₄ClO₄ bimodal': 0.70, 'HTPB R-45': 0.18, 'Al powder': 0.12},
        rho=1786, Tf=3350, gamma=1.200, MW=29.00,
        cstar=1578, Isp_vac=242, Isp_sl=210,
        burn_a=5.13, burn_n=0.350, P_min=2.0, P_max=15.0,
        OF_ratio=3.89, OB_pct=+28.1, equiv_ratio=0.87,
        cure_T='50–70 °C (HTPB/IPDI; NCO:OH ≈ 0.85–0.90)',
        cure_t='72–168 h (3–7 days at 60 °C)',
        decomp_onset=240.0, decomp_crit=300.0,
        crit_label='~300 °C — AP/Al matrix auto-ignition',
        process_note='⚠ Chemically-cured thermosetting composite. NEVER heat above 90 °C. Mixed propellant is sensitive to friction, impact, and heat. Handle only in approved facilities.',
        refs='Sutton & Biblarz (2016); Kuo & Summerfield (1984)',
    ),
    'APCP_NOAL': PropellantData(
        name='AP/HTPB Metal-Free (80/20)',
        abbr='APCP-NoAl',
        composition={'NH₄ClO₄ bimodal': 0.80, 'HTPB R-45': 0.20},
        rho=1690, Tf=2800, gamma=1.220, MW=26.00,
        cstar=1470, Isp_vac=215, Isp_sl=186,
        burn_a=4.20, burn_n=0.330, P_min=2.0, P_max=14.0,
        OF_ratio=4.00, OB_pct=+18.3, equiv_ratio=0.92,
        cure_T='50–70 °C (HTPB/IPDI)',
        cure_t='72–168 h',
        decomp_onset=240.0, decomp_crit=300.0,
        crit_label='~300 °C — AP decomposition / auto-ignition',
        process_note='Metal-free composite. Cleaner exhaust signature; lower specific impulse than Al-loaded formulation. Same cure schedule as APCP-STD.',
        refs='Sutton & Biblarz (2016)',
    ),
    'GAP_AP': PropellantData(
        name='GAP/AP Energetic Composite (65/20/10/5)',
        abbr='GAP-AP',
        composition={'NH₄ClO₄': 0.65, 'GAP diol': 0.20, 'TMETN': 0.10, 'Al': 0.05},
        rho=1760, Tf=3100, gamma=1.220, MW=28.50,
        cstar=1530, Isp_vac=255, Isp_sl=226,
        burn_a=6.00, burn_n=0.380, P_min=3.0, P_max=18.0,
        OF_ratio=3.25, OB_pct=+14.6, equiv_ratio=0.91,
        cure_T='40–60 °C (GAP azide diol + isocyanate curative)',
        cure_t='96–240 h',
        decomp_onset=220.0, decomp_crit=270.0,
        crit_label='270 °C — CRITICAL: azide (-N₃) decomposition (lower than HTPB systems)',
        process_note='⚠ ELEVATED HAZARD: GAP contains energetic azide groups (-N₃). Critical threshold significantly lower than HTPB. Requires specialist certified facility. Not recommended for amateur use.',
        refs='Kubota (2007); Frankel et al. AIAA-92-3461',
    ),
}

CASING_MATERIALS: Dict[str, CasingMaterial] = {
    '6061-T6': CasingMaterial(
        name='Aluminum 6061-T6',
        E_GPa=68.9, Sy_MPa=276, Su_MPa=310, rho=2700,
        comment='Standard aerospace alloy. Widely used in HPR motor casings. Excellent machinability.',
    ),
    '304_SS': CasingMaterial(
        name='Stainless Steel 304',
        E_GPa=193.0, Sy_MPa=207, Su_MPa=515, rho=8000,
        comment='Good corrosion resistance. High density. Suitable for test hardware and static fire fixtures.',
    ),
    '4130_Steel': CasingMaterial(
        name='4130 Chromoly Steel (normalized)',
        E_GPa=205.0, Sy_MPa=435, Su_MPa=670, rho=7850,
        comment='High strength. Used by commercial HPR casing manufacturers (e.g., AeroTech).',
    ),
    'CF_Epoxy': CasingMaterial(
        name='Carbon Fiber / Epoxy (filament wound)',
        E_GPa=70.0, Sy_MPa=900, Su_MPa=1500, rho=1550,
        comment='Highest specific strength. Research/EX motors. Note: brittle failure mode — conservative SF required.',
    ),
    'Ti_6Al4V': CasingMaterial(
        name='Titanium Ti-6Al-4V',
        E_GPa=113.8, Sy_MPa=880, Su_MPa=950, rho=4430,
        comment='Excellent specific strength. Expensive; used in high-performance research motors.',
    ),
}


# ═══════════════════════════════════════════════════════════════════════════
# BATES GRAIN GEOMETRY  (Geometric Regression Engine)
# ═══════════════════════════════════════════════════════════════════════════

class BATESGrain:
    """
    Multi-segment BATES (Ballistic Test & Evaluation System) grain geometry.

    Per-segment burn surface:
        Ab_seg = π · Di · L  +  2 · (π/4) · (Do² − Di²)
                  [core]           [two annular end faces]

    Regression per timestep Δt at local burn rate r (m/s):
        ri(t+Δt) = ri(t) + r·Δt       [core burns outward]
        L(t+Δt)  = L(t)  − 2r·Δt      [both ends recede]

    Burnout when:  ri ≥ ro  OR  L ≤ 0

    Port-to-throat area ratio (erosive burning risk):
        J = Ap / At  (flag if J < 2.0)

    Ref: Sutton & Biblarz §13; Nakka Grain Design Guide (2023)
    """

    def __init__(self, ro_m: float, ri_m: float, L_m: float, n_segments: int = 1):
        if ri_m >= ro_m:
            raise ValueError(f'Inner radius ri={ri_m*1e3:.2f}mm must be < ro={ro_m*1e3:.2f}mm')
        if L_m <= 0:
            raise ValueError('Grain segment length must be positive')
        if n_segments < 1:
            raise ValueError('Number of segments must be ≥ 1')
        self.ro    = ro_m
        self.ri0   = ri_m
        self.L0    = L_m
        self.n_seg = n_segments

    @property
    def Do(self) -> float:
        return 2.0 * self.ro

    @property
    def Di0(self) -> float:
        return 2.0 * self.ri0

    def burn_area(self, ri: float, L: float) -> float:
        """Total burning surface area (m²) across all n_seg segments."""
        L  = max(0.0, L)
        ri = min(ri, self.ro)
        Di, Do = 2 * ri, 2 * self.ro
        A_core = math.pi * Di * L
        A_ends = 2.0 * (math.pi / 4.0) * (Do**2 - Di**2)
        return max(0.0, A_core + A_ends) * self.n_seg

    def volume(self, ri: float, L: float) -> float:
        """Remaining propellant volume (m³) across all segments."""
        return math.pi * (self.ro**2 - ri**2) * max(0.0, L) * self.n_seg

    def initial_mass(self, rho: float) -> float:
        """Initial propellant mass (kg)."""
        return self.volume(self.ri0, self.L0) * rho

    def port_to_throat(self, ri: float, At: float) -> float:
        """Port-to-throat ratio J = Ap / At.  Flag if J < 2.0 (erosive risk)."""
        return (math.pi * ri**2) / At if At > 0 else float('inf')

    def regress(self, ri: float, L: float, r: float, dt: float) -> Tuple[float, float, bool]:
        """Advance grain geometry by dt.  Returns (ri_new, L_new, burned_out)."""
        ri_new  = ri + r * dt
        L_new   = max(0.0, L - 2.0 * r * dt)
        done    = (ri_new >= self.ro) or (L_new <= 1e-9)
        return ri_new, L_new, done


# ═══════════════════════════════════════════════════════════════════════════
# INTERNAL BALLISTICS SOLVER
# ═══════════════════════════════════════════════════════════════════════════

class InternalBallistics:
    """
    Quasi-Steady-State (QSS) internal ballistics solver.

    Key equations
    ─────────────
    Vandenkerckhove function:
        Γ = √[ γ · (2/(γ+1))^((γ+1)/(γ-1)) ]

    Saint-Robert burn rate law:
        r  = a · Pc^n          [P in MPa, r in mm/s → convert: ×1e-3]

    QSS chamber pressure (from mass balance ṁ_gen = ṁ_exit):
        Pc = (ρ_p · a_SI · c*_eff · Kn)^(1/(1-n))

    Choked nozzle mass flow:
        ṁ  = Pc · At · Γ / √(Tf · Rsp)

    Thrust (sea-level):
        F  = Cf_sl · Pc · At

    Two-phase flow efficiency factors:
        η_c*  — c-star efficiency (mixing/combustion losses): 0.92–0.98
        η_DP  — two-phase dispersion penalty (Al particles):  0.95–0.99
        c*_eff = η_c* · c*_theory
        Isp_eff = η_DP · Isp_theory (propagated through Cf)

    Ref: Sutton & Biblarz §3, §13; Turner (2009) Rocket & Spacecraft Propulsion §6
    """

    def __init__(self, prop: PropellantData, grain: BATESGrain,
                 At_m2: float, Ae_m2: float,
                 eta_cstar: float = 0.95, eta_DP: float = 0.97):
        self.prop      = prop
        self.grain     = grain
        self.At        = At_m2
        self.Ae        = Ae_m2
        self.eta_cstar = eta_cstar
        self.eta_DP    = eta_DP

        self.R_sp      = R_UNIV / (prop.MW * 1e-3)           # J/(kg·K)
        g              = prop.gamma
        self.Gamma     = math.sqrt(g * (2.0/(g+1))**((g+1)/(g-1)))
        self.cstar_eff = prop.cstar * eta_cstar
        self.Cf_sl     = (prop.Isp_sl  * G0 * eta_DP) / self.cstar_eff
        self.Cf_vac    = (prop.Isp_vac * G0 * eta_DP) / self.cstar_eff

    def saint_robert(self, Pc_Pa: float) -> float:
        """Burn rate (m/s), clamped to valid pressure range."""
        Pc_MPa = max(self.prop.P_min, min(self.prop.P_max, Pc_Pa / MPA))
        return self.prop.burn_a * (Pc_MPa ** self.prop.burn_n) * 1e-3

    def qss_pressure(self, Ab: float) -> float:
        """
        QSS chamber pressure (Pa).
        Unit transform for a:  a_SI [m/s·Pa^-n] = a [mm/s·MPa^-n] × 1e-3 × MPA^n
        Stability condition:   burn_n < 1  (n ≥ 1 → CATO)
        """
        n = self.prop.burn_n
        if n >= 1.0:
            return float('nan')
        a_SI = self.prop.burn_a * 1e-3 / (MPA ** n)
        Kn   = Ab / self.At
        return (self.prop.rho * a_SI * self.cstar_eff * Kn) ** (1.0 / (1.0 - n))

    def mass_flow(self, Pc_Pa: float) -> float:
        """Choked nozzle ṁ (kg/s): ṁ = Pc·At·Γ / √(Tf·Rsp)"""
        return (Pc_Pa * self.At * self.Gamma) / math.sqrt(self.prop.Tf * self.R_sp)

    def run(self, dt_s: float = 5e-4, max_time: float = 600.0) -> Dict:
        """
        Execute QSS time-marching simulation.

        Returns dict with time-series arrays and scalar performance metrics.
        """
        ri, L = self.grain.ri0, self.grain.L0
        t_l, Pc_l, F_l, Ab_l, Kn_l = [], [], [], [], []
        r_l, Isp_l, mdot_l, Jpt_l  = [], [], [], []

        t = 0.0
        while t < max_time:
            Ab = self.grain.burn_area(ri, L)
            if Ab < 1e-10:
                break
            Pc = self.qss_pressure(Ab)
            if not math.isfinite(Pc) or Pc <= 0:
                break

            r    = self.saint_robert(Pc)
            mdot = self.mass_flow(Pc)
            F    = self.Cf_sl * Pc * self.At
            Isp  = F / (mdot * G0) if mdot > 1e-12 else 0.0
            Kn   = Ab / self.At
            Jpt  = self.grain.port_to_throat(ri, self.At)

            t_l.append(t);    Pc_l.append(Pc);   F_l.append(F)
            Ab_l.append(Ab);  Kn_l.append(Kn);   r_l.append(r * 1e3)
            Isp_l.append(Isp); mdot_l.append(mdot); Jpt_l.append(Jpt)

            ri, L, done = self.grain.regress(ri, L, r, dt_s)
            if done:
                break
            t += dt_s

        if not t_l:
            raise RuntimeError('Simulation produced no output. Check grain geometry and nozzle throat area.')

        t_a = np.array(t_l);  F_a = np.array(F_l);  Pc_a = np.array(Pc_l)
        It  = float(np.trapezoid(F_a, t_a))
        m_p = self.grain.initial_mass(self.prop.rho)

        # Burn profile type (from Kn trend)
        kn = np.array(Kn_l)
        dkn = kn[-1] - kn[0]
        if abs(dkn) / (kn[0] + 1e-9) < 0.05:
            profile = 'Neutral'
        else:
            profile = 'Progressive' if dkn > 0 else 'Regressive'

        return {
            't': t_a, 'Pc': Pc_a, 'F': F_a,
            'Ab': np.array(Ab_l), 'Kn': kn,
            'r': np.array(r_l), 'Isp': np.array(Isp_l),
            'mdot': np.array(mdot_l), 'Jpt': np.array(Jpt_l),
            'burn_time':      float(t_a[-1]),
            'total_impulse':  It,
            'avg_thrust':     float(np.mean(F_a)),
            'max_thrust':     float(np.max(F_a)),
            'avg_Pc_MPa':     float(np.mean(Pc_a)) / MPA,
            'max_Pc_MPa':     float(np.max(Pc_a)) / MPA,
            'avg_Isp':        float(np.mean(Isp_l)),
            'avg_r':          float(np.mean(r_l)),
            'init_Kn':        float(kn[0]),
            'max_Kn':         float(np.max(kn)),
            'min_Jpt':        float(np.min(Jpt_l)),
            'prop_mass_kg':   m_p,
            'motor_class':    motor_class(It),
            'profile':        profile,
            'cstar_eff':      self.cstar_eff,
            'Cf_sl':          self.Cf_sl,
            'Gamma':          self.Gamma,
            'R_sp':           self.R_sp,
        }


# ═══════════════════════════════════════════════════════════════════════════
# STRUCTURAL ANALYSIS MODULE
# ═══════════════════════════════════════════════════════════════════════════

class StructuralAnalysis:
    """
    Pressure vessel analysis for solid motor casings.

    Thin-wall (t < 0.1·ri):
        σ_h  = Pc·ri / t                      (hoop)
        σ_a  = Pc·ri / (2t)                   (axial)

    Thick-wall Lamé equations (t ≥ 0.1·ri), evaluated at inner radius:
        σ_h  = Pc·(ro² + ri²) / (ro² − ri²)
        σ_a  = Pc·ri²         / (ro² − ri²)

    Von Mises equivalent stress:
        σ_VM = √(σ_h² − σ_h·σ_a + σ_a²)

    Safety factors:
        SF_yield = Sy  / σ_VM    (HPR recommendation: ≥ 4)
        SF_ult   = Su  / σ_VM

    Required wall thickness for SF_target (Lamé solution):
        ro_req = ri · √[(Sy/SF + Pc) / (Sy/SF − Pc)]
        t_req  = ro_req − ri

    Ref: Shigley's Machine Design §3-14; Roark's Formulas §13
    """

    def __init__(self, material: CasingMaterial, ri_m: float, t_m: float):
        self.mat = material
        self.ri  = ri_m
        self.t   = t_m
        self.ro  = ri_m + t_m

    @property
    def is_thin_wall(self) -> bool:
        return self.t < 0.1 * self.ri

    def hoop_stress(self, Pc_Pa: float) -> float:
        if self.is_thin_wall:
            return Pc_Pa * self.ri / self.t
        return Pc_Pa * (self.ro**2 + self.ri**2) / (self.ro**2 - self.ri**2)

    def axial_stress(self, Pc_Pa: float) -> float:
        if self.is_thin_wall:
            return Pc_Pa * self.ri / (2.0 * self.t)
        return Pc_Pa * self.ri**2 / (self.ro**2 - self.ri**2)

    def von_mises(self, Pc_Pa: float) -> float:
        s_h = self.hoop_stress(Pc_Pa)
        s_a = self.axial_stress(Pc_Pa)
        return math.sqrt(s_h**2 - s_h * s_a + s_a**2)

    def safety_factor_yield(self, Pc_Pa: float) -> float:
        sv = self.von_mises(Pc_Pa)
        return (self.mat.Sy_MPa * MPA) / sv if sv > 0 else float('inf')

    def safety_factor_ult(self, Pc_Pa: float) -> float:
        sv = self.von_mises(Pc_Pa)
        return (self.mat.Su_MPa * MPA) / sv if sv > 0 else float('inf')

    def required_thickness(self, Pc_Pa: float, SF_target: float = 4.0) -> float:
        """Minimum wall thickness (m) using Lamé solution for SF_target."""
        allow = (self.mat.Sy_MPa * MPA) / SF_target
        if allow <= Pc_Pa:
            return float('inf')
        ratio = math.sqrt((allow + Pc_Pa) / (allow - Pc_Pa))
        return self.ri * (ratio - 1.0)

    def burst_pressure(self) -> float:
        """Estimated burst pressure (Pa) from ultimate strength."""
        Su_Pa = self.mat.Su_MPa * MPA
        if self.is_thin_wall:
            return Su_Pa * self.t / self.ri
        return Su_Pa * (self.ro**2 - self.ri**2) / (self.ro**2 + self.ri**2)

    def analyze(self, Pc_Pa: float) -> Dict:
        """Return full structural margin report dict."""
        return {
            'wall_type':      'Thin-wall (t/ri < 0.1)' if self.is_thin_wall else 'Thick-wall (Lamé equations)',
            't_ratio':        self.t / self.ri,
            'hoop_MPa':       self.hoop_stress(Pc_Pa) / MPA,
            'axial_MPa':      self.axial_stress(Pc_Pa) / MPA,
            'vm_MPa':         self.von_mises(Pc_Pa) / MPA,
            'SF_yield':       self.safety_factor_yield(Pc_Pa),
            'SF_ult':         self.safety_factor_ult(Pc_Pa),
            'SF_ok':          self.safety_factor_yield(Pc_Pa) >= 2.0,
            't_req_SF4_mm':   self.required_thickness(Pc_Pa, 4.0) * 1e3,
            't_req_SF2_mm':   self.required_thickness(Pc_Pa, 2.0) * 1e3,
            'burst_MPa':      self.burst_pressure() / MPA,
            'Sy_MPa':         self.mat.Sy_MPa,
            'Su_MPa':         self.mat.Su_MPa,
        }


# ═══════════════════════════════════════════════════════════════════════════
# FLIGHT TRAJECTORY SIMULATOR
# ═══════════════════════════════════════════════════════════════════════════

class FlightTrajectory:
    """1D rocket flight simulation along a fixed launch angle using ISA atmosphere."""

    def __init__(self, t_arr, F_arr, prop_mass_kg, dry_mass_kg, Cd, body_diam_m,
                 launch_angle_deg=90):
        self.t_arr = np.array(t_arr, dtype=float)
        self.F_arr = np.array(F_arr, dtype=float)
        self.prop_mass_kg = float(prop_mass_kg)
        self.dry_mass_kg = float(dry_mass_kg)
        self.Cd = float(Cd)
        self.ref_area = math.pi * (float(body_diam_m) / 2.0) ** 2
        self.burn_time = float(self.t_arr[-1] - self.t_arr[0])
        self.sin_angle = math.sin(math.radians(float(launch_angle_deg)))

    def _isa_density(self, h):
        T0, P0, R, L, g = 288.15, 101325.0, 287.058, 0.0065, G0
        h = max(h, 0.0)
        if h <= 11000.0:
            T = T0 - L * h
            P = P0 * (T / T0) ** (g / (L * R))
        else:
            T11 = T0 - L * 11000.0
            P11 = P0 * (T11 / T0) ** (g / (L * R))
            T = T11
            P = P11 * math.exp(-g * (h - 11000.0) / (R * T))
        return P / (R * T)

    def _thrust(self, t):
        if t < self.t_arr[0] or t > self.t_arr[-1]:
            return 0.0
        return float(np.interp(t, self.t_arr, self.F_arr))

    def _prop_mass(self, t):
        if t <= self.t_arr[0]: return self.prop_mass_kg
        if t >= self.t_arr[-1]: return 0.0
        return self.prop_mass_kg * (1.0 - (t - self.t_arr[0]) / self.burn_time)

    def run(self, dt=0.05):
        alt, vel, t = 0.0, 0.0, 0.0
        t_h, alt_h, vel_h = [t], [alt], [vel]
        burnout_alt, burnout_time, apogee_time, past_apogee = None, None, None, False
        burn_end = float(self.t_arr[-1])

        while True:
            m = self.dry_mass_kg + self._prop_mass(t)
            F = self._thrust(t)
            rho = self._isa_density(alt)
            drag = 0.5 * rho * self.Cd * self.ref_area * vel * abs(vel)
            a = (F - drag) / m - G0 * self.sin_angle
            if burnout_alt is None and t >= burn_end:
                burnout_alt, burnout_time = alt, t
            if not past_apogee and vel < 0.0 and t > 0.1:
                past_apogee, apogee_time = True, t
            vel += a * dt
            alt += vel * self.sin_angle * dt
            t += dt
            if past_apogee and alt <= 0.0:
                alt = 0.0
                t_h.append(t); alt_h.append(alt); vel_h.append(vel)
                break
            if t > 600.0: break
            t_h.append(t); alt_h.append(alt); vel_h.append(vel)

        max_alt = max(alt_h)
        max_vel = max(abs(v) for v in vel_h)
        if apogee_time is None:
            apogee_time = t_h[alt_h.index(max_alt)]
        if burnout_alt is None:
            burnout_alt, burnout_time = 0.0, 0.0
        coast = apogee_time - burnout_time if burnout_time else 0.0
        return {
            't': np.array(t_h), 'altitude_m': np.array(alt_h), 'velocity_ms': np.array(vel_h),
            'max_altitude_m': max_alt, 'max_velocity_ms': max_vel,
            'burnout_altitude_m': burnout_alt, 'burnout_time_s': burnout_time,
            'time_to_apogee_s': apogee_time, 'flight_time_s': t_h[-1],
            'coast_time_s': coast,
        }


# ═══════════════════════════════════════════════════════════════════════════
# UTILITY FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════

def apply_nozzle_erosion(At_initial_m2: float, t_s: float, erosion_rate_mm2_per_s: float) -> float:
    return At_initial_m2 + erosion_rate_mm2_per_s * 1e-6 * t_s


def burn_rate_at_temp(prop_burn_a, prop_burn_n, Pc_Pa, T_propellant_C, T_ref_C=20.0, sigma_p=0.003):
    r_ref = prop_burn_a * (Pc_Pa / MPA) ** prop_burn_n * 1e-3
    return r_ref * math.exp(sigma_p * (T_propellant_C - T_ref_C))


def render_validation_warnings(ro_mm, ri_mm, L_mm, n_seg, Dt_mm, De_mm, wall_mm, prop, mat_key):
    w = []
    if ri_mm >= ro_mm:
        w.append('FATAL: Core radius ri >= outer radius ro — impossible geometry.')
    if ri_mm / ro_mm < 0.20:
        w.append('Very thick web (ri/ro < 0.2) — strongly progressive burn, Kn spike likely.')
    if ri_mm / ro_mm > 0.75:
        w.append('Very thin web (ri/ro > 0.75) — short burn, near-burnout erosive risk.')
    if L_mm / (2 * ri_mm) > 8:
        w.append(f'Long grain: L/D_port = {L_mm/(2*ri_mm):.1f} > 8 — high erosive burning risk, check J ratio.')
    if De_mm <= Dt_mm:
        w.append('Nozzle exit diameter <= throat diameter — impossible geometry.')
    if De_mm / Dt_mm > 6:
        w.append(f'Very high expansion ratio ε = {(De_mm/Dt_mm)**2:.1f} — likely over-expanded at sea level.')
    if wall_mm < 1.5:
        w.append(f'Wall thickness {wall_mm:.2f} mm is dangerously thin. Aim for SF ≥ 4.')
    if n_seg * L_mm > 600:
        w.append(f'Total grain length {n_seg*L_mm:.0f} mm > 600 mm — verify structural integrity.')
    if prop.burn_n >= 0.90:
        w.append(f'Burn exponent n = {prop.burn_n} is near the instability limit (n ≥ 1). CATO risk.')
    return w


def motor_class(It_Ns: float) -> str:
    thresholds = [
        (0,'sub-A'),(1.25,'A'),(2.5,'B'),(5,'C'),(10,'D'),
        (20,'E'),(40,'F'),(80,'G'),(160,'H'),(320,'I'),
        (640,'J'),(1280,'K'),(2560,'L'),(5120,'M'),
        (10240,'N'),(20480,'O'),
    ]
    cls = 'O+'
    for lo, label in reversed(thresholds):
        if It_Ns >= lo:
            cls = label
            break
    return cls


def generate_eng(res: Dict, prop: PropellantData, grain: BATESGrain,
                 At_m2: float, motor_name: str = 'NEXUS') -> str:
    """
    Standard RASP .ENG thrust curve (compatible with OpenRocket, RASAero II).
    Ref: http://www.thrustcurve.org/info/raspformat.html
    """
    diam_mm = grain.ro * 2 * 1e3
    len_mm  = grain.L0 * 1e3 * grain.n_seg
    prop_kg = grain.initial_mass(prop.rho)
    tot_kg  = prop_kg * 1.85   # estimate structural mass fraction

    t_a, F_a = res['t'], res['F']
    if len(t_a) > 200:
        idx = np.round(np.linspace(0, len(t_a)-1, 200)).astype(int)
        t_a, F_a = t_a[idx], F_a[idx]

    lines = [
        f'; ──────────────────────────────────────────────────────',
        f'; NEXUS Rocketry Simulation Suite — Thrust Curve Export',
        f'; Generated : {datetime.now().strftime("%Y-%m-%d %H:%M UTC")}',
        f'; Propellant: {prop.name}',
        f'; Grain     : {grain.n_seg}×BATES  ro={grain.ro*1e3:.2f}mm  ri={grain.ri0*1e3:.2f}mm  L={grain.L0*1e3:.2f}mm/seg',
        f'; Nozzle    : At={At_m2*1e6:.3f}mm²  ε={res.get("Cf_sl",0)/res.get("Cf_sl",1):.3f}',
        f'; a={prop.burn_a}  n={prop.burn_n}  c*_theory={prop.cstar}m/s  c*_eff={res.get("cstar_eff",prop.cstar):.1f}m/s',
        f'; ρ_p={prop.rho}kg/m³  Tf={prop.Tf}K  γ={prop.gamma}  MW={prop.MW}g/mol',
        f'; Γ={res.get("Gamma",0):.5f}  Cf_sl={res.get("Cf_sl",0):.5f}',
        f'; ─── Performance Summary ───────────────────────────────',
        f'; Class     : {res["motor_class"]}',
        f'; It        : {res["total_impulse"]:.2f} N·s',
        f'; Fmax      : {res["max_thrust"]:.1f} N',
        f'; Favg      : {res["avg_thrust"]:.1f} N',
        f'; tb        : {res["burn_time"]:.3f} s',
        f'; Pc_max    : {res["max_Pc_MPa"]:.3f} MPa',
        f'; Isp_avg   : {res["avg_Isp"]:.1f} s',
        f'; Profile   : {res["profile"]}',
        f'; mp        : {res["prop_mass_kg"]*1e3:.1f} g',
        f'; ⚠ THEORETICAL SIMULATION ONLY — Verify experimentally before flight',
        f'; Refs: {prop.refs}',
        f'; ──────────────────────────────────────────────────────',
        f'{motor_name} {diam_mm:.1f} {len_mm:.1f} 0 {prop_kg:.4f} {tot_kg:.4f} NEXUS-SIM',
    ]
    for t, F in zip(t_a, F_a):
        lines.append(f'   {t:.4f} {F:.3f}')
    lines.append(f'   {res["burn_time"]+0.005:.4f} 0.000')
    lines.append(';')
    return '\n'.join(lines)


def generate_json(res: Dict, prop: PropellantData, grain: BATESGrain,
                  struct_report: Optional[Dict],
                  At_m2: float, Ae_m2: float,
                  eta_cstar: float, eta_DP: float,
                  motor_name: str) -> str:
    """Comprehensive simulation configuration and results in JSON format."""
    cfg = {
        'metadata': {
            'generator': 'NEXUS High-Power Rocketry Simulation Suite v3.0',
            'generated': datetime.now().isoformat(),
            'motor_name': motor_name,
            'refs': prop.refs,
        },
        'propellant': {
            'name': prop.name, 'abbreviation': prop.abbr,
            'composition': prop.composition,
            'density_kg_m3': prop.rho, 'flame_temp_K': prop.Tf,
            'gamma': prop.gamma, 'MW_products_g_mol': prop.MW,
            'cstar_theoretical_m_s': prop.cstar,
            'Isp_vac_s': prop.Isp_vac, 'Isp_sl_s': prop.Isp_sl,
            'burn_a_mm_s_MPan': prop.burn_a, 'burn_n': prop.burn_n,
            'valid_Pc_range_MPa': [prop.P_min, prop.P_max],
            'OF_ratio': prop.OF_ratio, 'OB_pct': prop.OB_pct,
            'equivalence_ratio': prop.equiv_ratio,
            'cure_temperature': prop.cure_T,
            'cure_time': prop.cure_t,
            'decomp_onset_C': prop.decomp_onset,
            'decomp_critical_C': prop.decomp_crit,
        },
        'grain': {
            'type': 'Multi-segment BATES', 'n_segments': grain.n_seg,
            'outer_radius_mm': grain.ro * 1e3, 'inner_radius_mm': grain.ri0 * 1e3,
            'segment_length_mm': grain.L0 * 1e3,
            'total_length_mm': grain.L0 * grain.n_seg * 1e3,
            'initial_propellant_mass_g': grain.initial_mass(prop.rho) * 1e3,
        },
        'nozzle': {
            'throat_area_mm2': At_m2 * 1e6,
            'exit_area_mm2': Ae_m2 * 1e6,
            'expansion_ratio': Ae_m2 / At_m2 if At_m2 > 0 else None,
            'throat_diameter_mm': 2.0 * math.sqrt(At_m2 / math.pi) * 1e3,
        },
        'efficiency': {
            'eta_cstar': eta_cstar, 'eta_two_phase': eta_DP,
            'combined_eta': eta_cstar * eta_DP,
            'cstar_effective_m_s': res.get('cstar_eff', prop.cstar),
            'Gamma_Vandenkerckhove': res.get('Gamma', 0),
            'Cf_sl': res.get('Cf_sl', 0),
        },
        'performance': {
            'motor_class': res['motor_class'],
            'burn_time_s': res['burn_time'],
            'total_impulse_Ns': res['total_impulse'],
            'avg_thrust_N': res['avg_thrust'],
            'max_thrust_N': res['max_thrust'],
            'avg_chamber_pressure_MPa': res['avg_Pc_MPa'],
            'max_chamber_pressure_MPa': res['max_Pc_MPa'],
            'avg_Isp_s': res['avg_Isp'],
            'avg_burn_rate_mm_s': res['avg_r'],
            'propellant_mass_kg': res['prop_mass_kg'],
            'initial_Kn': res['init_Kn'],
            'max_Kn': res['max_Kn'],
            'min_port_to_throat': res['min_Jpt'],
            'burn_profile': res['profile'],
        },
        'structural': struct_report or {},
        'disclaimers': [
            '⚠ DISCLAIMER: This simulation is for academic and theoretical purposes only.',
            'All values are QSS approximations. Actual performance may differ significantly.',
            'Processing energetic propellants requires ATF licensing and certified facilities.',
            'HPR activities must comply with NAR/TRA certification and NFPA 1127.',
            'Verify all structural margins with independent qualified engineering analysis.',
            'The authors assume no liability for use of this simulation output.',
        ]
    }
    return json.dumps(cfg, indent=2, default=str)


# ═══════════════════════════════════════════════════════════════════════════
# PLOTTING
# ═══════════════════════════════════════════════════════════════════════════

def plot_overview(res: Dict, prop_name: str) -> go.Figure:
    t = res['t']
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=['Chamber Pressure (MPa)', 'Thrust (N)',
                        'Specific Impulse (s)', 'Burn Rate (mm/s)'],
        horizontal_spacing=0.12, vertical_spacing=0.15,
    )
    traces = [
        (res['Pc'] / MPA, _CYN, 'Pc (MPa)', 1, 1),
        (res['F'],         _ORG, 'F (N)',    1, 2),
        (res['Isp'],       _GRN, 'Isp (s)',  2, 1),
        (res['r'],         '#a78bfa', 'r (mm/s)', 2, 2),
    ]
    for y, col, name, row, c in traces:
        fig.add_trace(go.Scatter(
            x=t, y=y, name=name, mode='lines',
            line=dict(color=col, width=2),
            fill='tozeroy', fillcolor=_FILL.get(col, 'rgba(0,212,255,0.08)'),
        ), row=row, col=c)

    fig.update_layout(**NEXUS_LAYOUT, title_text=f'<b>NEXUS — {prop_name} | Overview</b>',
                      title_font=dict(color=_CYN, size=14), showlegend=False)
    for i in range(1, 5):
        r, c = ((i-1)//2)+1, ((i-1)%2)+1
        fig.update_xaxes(title_text='Time (s)', **NEXUS_AXIS, row=r, col=c)
        fig.update_yaxes(**NEXUS_AXIS, row=r, col=c)
    return fig


def plot_grain(res: Dict) -> go.Figure:
    t = res['t']
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=['Burning Surface Area Ab (cm²)', 'Klemmung Coefficient Kn = Ab/At'],
    )
    fig.add_trace(go.Scatter(
        x=t, y=res['Ab'] * 1e4, name='Ab', mode='lines',
        line=dict(color=_CYN, width=2.5), fill='tozeroy',
        fillcolor=_FILL[_CYN],
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=t, y=res['Kn'], name='Kn', mode='lines',
        line=dict(color=_ORG, width=2.5), fill='tozeroy',
        fillcolor=_FILL[_ORG],
    ), row=1, col=2)

    for c in (1, 2):
        fig.update_xaxes(title_text='Time (s)', **NEXUS_AXIS, row=1, col=c)
        fig.update_yaxes(**NEXUS_AXIS, row=1, col=c)

    fig.update_layout(**NEXUS_LAYOUT,
                      title_text='<b>NEXUS — Grain Regression Dynamics</b>',
                      title_font=dict(color=_CYN, size=14), showlegend=False)
    return fig


def plot_saint_robert(prop: PropellantData) -> go.Figure:
    P = np.linspace(prop.P_min, prop.P_max, 300)
    r = prop.burn_a * P**prop.burn_n
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=P, y=r, mode='lines',
        name=f'r = {prop.burn_a} · Pc^{prop.burn_n}',
        line=dict(color=_GRN, width=3),
    ))
    fig.update_layout(**NEXUS_LAYOUT,
                      title_text=f'<b>Saint-Robert Burn Rate Law — {prop.abbr}</b>',
                      title_font=dict(color=_CYN, size=14),
                      xaxis=dict(title='Chamber Pressure Pc (MPa)', **NEXUS_AXIS),
                      yaxis=dict(title='Burn Rate r (mm/s)', **NEXUS_AXIS))
    return fig


def plot_structural(struct_arr: List[Tuple], peak_Pc: float) -> go.Figure:
    """Bar chart comparing hoop/von Mises stress vs yield strength across materials."""
    mats   = [s[0] for s in struct_arr]
    vm     = [s[1]['vm_MPa']  for s in struct_arr]
    sy     = [s[1]['Sy_MPa']  for s in struct_arr]
    sf     = [s[1]['SF_yield'] for s in struct_arr]
    colors = [_GRN if s >= 2.0 else _RED for s in sf]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name='Von Mises σ_VM', x=mats, y=vm,
        marker_color=[_CYN]*len(mats), opacity=0.85,
    ))
    fig.add_trace(go.Bar(
        name='Yield Strength Sy', x=mats, y=sy,
        marker_color=[_ORG]*len(mats), opacity=0.65,
    ))
    # Safety factor annotation
    for i, (m, s) in enumerate(zip(mats, sf)):
        fig.add_annotation(
            x=m, y=max(vm[i], sy[i]) + 20,
            text=f'SF={s:.2f}', showarrow=False,
            font=dict(color=colors[i], size=11, family='Share Tech Mono'),
        )
    fig.update_layout(**NEXUS_LAYOUT,
                      title_text=f'<b>Structural Margin — Peak Pc = {peak_Pc:.2f} MPa</b>',
                      title_font=dict(color=_CYN, size=14),
                      barmode='group',
                      xaxis=dict(title='Casing Material', **NEXUS_AXIS),
                      yaxis=dict(title='Stress (MPa)', **NEXUS_AXIS))
    return fig


def plot_flight(flight_res: dict) -> go.Figure:
    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=('Altitude (m)', 'Velocity (m/s)'),
                        horizontal_spacing=0.12)
    t = flight_res['t']
    alt = flight_res['altitude_m']
    vel = flight_res['velocity_ms']
    t_bo = flight_res.get('burnout_time_s')
    t_ap = flight_res.get('time_to_apogee_s')
    alt_ap = flight_res.get('max_altitude_m')

    fig.add_trace(go.Scatter(x=t, y=alt, mode='lines', name='Altitude',
                             line=dict(color=_CYN, width=2),
                             fill='tozeroy', fillcolor='rgba(0,212,255,0.08)'), row=1, col=1)
    fig.add_trace(go.Scatter(x=t, y=vel, mode='lines', name='Velocity',
                             line=dict(color=_ORG, width=2)), row=1, col=2)
    if t_bo:
        for col in (1, 2):
            fig.add_vline(x=t_bo, line=dict(color=_ORG, width=1.5, dash='dash'),
                          annotation_text='Burnout' if col == 1 else '',
                          annotation_position='top right',
                          annotation_font=dict(color=_ORG, size=10), row=1, col=col)
    if t_ap and alt_ap:
        fig.add_trace(go.Scatter(x=[t_ap], y=[alt_ap], mode='markers+text',
                                 name='Apogee', marker=dict(color=_GRN, size=10, symbol='diamond'),
                                 text=['Apogee'], textposition='top center',
                                 textfont=dict(color=_GRN, size=10)), row=1, col=1)
    fig.update_xaxes(**NEXUS_AXIS, title_text='Time (s)')
    fig.update_yaxes(**NEXUS_AXIS, title_text='Altitude (m)', row=1, col=1)
    fig.update_yaxes(**NEXUS_AXIS, title_text='Velocity (m/s)', row=1, col=2)
    fig.update_layout(**NEXUS_LAYOUT,
                      title_text='<b>NEXUS — Flight Trajectory</b>',
                      title_font=dict(color=_CYN, size=14))
    return fig


def plot_sensitivity(results_list: list, param_label: str, param_vals: list,
                     metric_key: str, metric_label: str) -> go.Figure:
    n = len(param_vals)
    center = n // 2
    vals = [r.get(metric_key, 0) for r in results_list]
    colors = [_CYN if i == center else 'rgba(167,139,250,0.35)' for i in range(n)]
    fig = go.Figure(go.Bar(
        x=[str(v) for v in param_vals], y=vals,
        marker=dict(color=colors, line=dict(color=[_CYN if i == center else _PRP for i in range(n)], width=1.5)),
        hovertemplate=f'{param_label}: %{{x}}<br>{metric_label}: %{{y:.2f}}<extra></extra>',
    ))
    fig.update_layout(**NEXUS_LAYOUT,
                      title_text=f'<b>Sensitivity: {metric_label} vs {param_label}</b>',
                      title_font=dict(color=_CYN, size=14),
                      xaxis=dict(title=param_label, **NEXUS_AXIS),
                      yaxis=dict(title=metric_label, **NEXUS_AXIS),
                      showlegend=False)
    return fig


def plot_grain_cross_section(ro_mm: float, ri_mm: float, n_steps: int = 6) -> go.Figure:
    theta = np.linspace(0, 2 * math.pi, 360)
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ro_mm * cos_t, y=ro_mm * sin_t, mode='lines',
                             fill='toself', fillcolor='rgba(160,160,160,0.25)',
                             line=dict(color='#aaaaaa', width=2),
                             name=f'Propellant (ro={ro_mm:.1f} mm)'))
    radii = np.linspace(ri_mm, ro_mm * 0.97, n_steps)
    for i, r in enumerate(radii):
        frac = i / max(n_steps - 1, 1)
        c_r, c_g, c_b = int(frac * 80), int(212 - frac * 180), int(255 - frac * 120)
        alpha = 0.9 - frac * 0.55
        color = f'rgba({c_r},{c_g},{c_b},{alpha:.2f})'
        label = (f'Core initial (ri={ri_mm:.1f} mm)' if i == 0 else
                 f'Near burnout (r≈{r:.1f} mm)' if i == n_steps - 1 else
                 f'Step {i+1} (r={r:.1f} mm)')
        fig.add_trace(go.Scatter(x=r * cos_t, y=r * sin_t, mode='lines',
                                 line=dict(color=color, width=2.5 if i == 0 else 1.2,
                                           dash='solid' if i == 0 else 'dot'),
                                 name=label))
    pad = ro_mm * 1.3
    fig.update_layout(**{**NEXUS_LAYOUT,
                         'title': dict(text=f'BATES Grain Cross-Section  ·  ro={ro_mm:.1f} mm  ri={ri_mm:.1f} mm',
                                       font=dict(color=_CYN, size=13)),
                         'xaxis': dict(range=[-pad, pad], scaleanchor='y', scaleratio=1,
                                       showgrid=False, zeroline=False, showticklabels=False),
                         'yaxis': dict(range=[-pad, pad], showgrid=False, zeroline=False, showticklabels=False),
                         'height': 460,
                         'legend': dict(x=1.02, y=0.98, font=dict(size=9),
                                        bgcolor='rgba(0,0,0,0)', bordercolor='rgba(255,255,255,0.1)', borderwidth=1)})
    return fig


def plot_overlay(overlay_sims: list) -> go.Figure:
    fig = go.Figure()
    for sim in overlay_sims:
        label = f"{sim['label']} — {sim['motor_class']} ({sim['total_impulse']:.1f} N·s)"
        fig.add_trace(go.Scatter(x=sim['t'], y=sim['F'], mode='lines', name=label,
                                 line=dict(color=sim['color'], width=2),
                                 hovertemplate='<b>%{fullData.name}</b><br>Time: %{x:.3f}s<br>Thrust: %{y:.1f}N<extra></extra>'))
    fig.update_layout(**NEXUS_LAYOUT,
                      title_text='<b>Thrust Curve Overlay</b>', title_font=dict(color=_CYN, size=14),
                      xaxis=dict(title='Time (s)', **NEXUS_AXIS),
                      yaxis=dict(title='Thrust (N)', **NEXUS_AXIS))
    return fig


def plot_temp_sensitivity(prop: 'PropellantData') -> go.Figure:
    T_vals = [-10.0, 20.0, 50.0]
    colors = {-10.0: _PRP, 20.0: _CYN, 50.0: _ORG}
    labels = {-10.0: 'Cold (−10 °C)', 20.0: 'Nominal (20 °C)', 50.0: 'Hot (50 °C)'}
    P_Pa = np.linspace(prop.P_min * MPA, prop.P_max * MPA, 200)
    fig = go.Figure()
    for T in T_vals:
        rates = [burn_rate_at_temp(prop.burn_a, prop.burn_n, P, T) * 1e3 for P in P_Pa]
        fig.add_trace(go.Scatter(x=P_Pa / MPA, y=rates, mode='lines',
                                 name=labels[T], line=dict(color=colors[T], width=2,
                                                           dash='solid' if T == 20.0 else 'dash')))
    fig.update_layout(**NEXUS_LAYOUT,
                      title_text='<b>Temperature Sensitivity — Burn Rate</b>',
                      title_font=dict(color=_CYN, size=14),
                      xaxis=dict(title='Chamber Pressure (MPa)', **NEXUS_AXIS),
                      yaxis=dict(title='Burn Rate (mm/s)', **NEXUS_AXIS))
    return fig


def store_sim_overlay(res: dict, prop_abbr: str, label: str) -> None:
    _OVERLAY_COLORS = [_CYN, _ORG, _GRN, _PRP]
    if 'overlay_sims' not in st.session_state:
        st.session_state.overlay_sims = []
    sims = st.session_state.overlay_sims
    entry = {
        'label': label, 'prop_abbr': prop_abbr,
        't': res['t'], 'F': res['F'], 'Pc': res['Pc'], 'Kn': res['Kn'],
        'motor_class': res['motor_class'],
        'total_impulse': res['total_impulse'],
        'burn_time': res['burn_time'],
        'color': _OVERLAY_COLORS[len(sims) % len(_OVERLAY_COLORS)],
    }
    if len(sims) >= 4:
        sims.pop(0)
    sims.append(entry)


def render_overlay_controls() -> None:
    sims = st.session_state.get('overlay_sims', [])
    if not sims:
        st.caption('No simulations stored. Run a simulation and click "Add to Overlay".')
        return
    for sim in sims:
        c1, c2 = st.columns([0.04, 0.96])
        with c1:
            st.markdown(f"<div style='width:12px;height:12px;border-radius:3px;background:{sim['color']};margin-top:5px'></div>",
                        unsafe_allow_html=True)
        with c2:
            st.markdown(f"**{sim['label']}** — {sim['motor_class']}, {sim['total_impulse']:.1f} N·s, "
                        f"burn {sim['burn_time']:.2f}s ({sim['prop_abbr']})")
    st.caption(f"{len(sims)}/4 slots used.")
    if st.button('Clear All Overlays', type='secondary'):
        st.session_state.overlay_sims = []
        st.rerun()


def get_url_params() -> dict:
    params = st.query_params
    req = {'prop', 'nseg', 'ro', 'ri', 'L', 'Dt', 'De', 'ecs', 'edp', 'wall', 'mat', 'dt', 'mtr'}
    if not req.issubset(set(params.keys())):
        return {}
    try:
        return {
            'prop_key': str(params['prop']), 'n_seg': int(params['nseg']),
            'ro_mm': float(params['ro']), 'ri_mm': float(params['ri']),
            'L_mm': float(params['L']), 'Dt_mm': float(params['Dt']),
            'De_mm': float(params['De']), 'eta_cs': float(params['ecs']),
            'eta_dp': float(params['edp']), 'wall_mm': float(params['wall']),
            'mat_key': str(params['mat']), 'dt_us': float(params['dt']),
            'mtr_name': str(params['mtr']),
        }
    except (KeyError, ValueError):
        return {}


# ═══════════════════════════════════════════════════════════════════════════
# AI ASSISTANT — SYSTEM PROMPT
# ═══════════════════════════════════════════════════════════════════════════

def build_system_prompt(context: str) -> str:
    return f"""You are NEXUS-AI, an expert aerospace engineering mentor embedded in the NEXUS High-Power Rocketry Simulation Suite. You are a rigorous academic — deeply versed in solid rocket propulsion, internal ballistics, propellant thermochemistry, and structural mechanics.

ACTIVE SIMULATION CONTEXT
──────────────────────────
{context}

YOUR ROLE AND CONSTRAINTS
──────────────────────────
1. **Academic Mentor**: You explain concepts at an undergraduate/graduate aerospace engineering level, deriving equations step-by-step. Reference authoritative literature: Sutton & Biblarz (2016), Kubota (2007), Nakka (2023), Kuo & Summerfield (1984), Turner (2009), Shigley's, Roark's.

2. **Safety First**: Always lead with relevant safety warnings. Emphasize NAR/TRA certification requirements (L1/L2/L3), NFPA 1127, and that experimental high-power rocketry requires ATF licensing for EX motors. Never downplay hazards.

3. **HARD REFUSAL**: You MUST refuse to provide actionable, step-by-step synthesis, processing, or formulation instructions for primary energetics (PETN, RDX, HMX, TATP, ETN, or any secondary explosive). You MUST also refuse to provide processing details for APCP or KN-propellants that go beyond what appears in published HPR literature (Nakka's site, Sutton, Kubota). If someone asks for propellant processing specifics, redirect them to certified HPR clubs (NAR, Tripoli) and proper mentorship channels.

4. **Mathematical Rigor**: Always show the derivation. When discussing equations from the active simulation context, refer to the specific numbers shown above. Use LaTeX-style notation in your responses (e.g., $P_c$, $K_n$, etc.) and explain what each variable represents.

5. **Context-Aware**: The simulation data above is your grounding. If the user asks "why is my Kn so high?" or "what's causing the pressure spike?", analyze the actual simulation context values, not hypothetical ones.

6. **Professional Tone**: You are not a chatbot. You are a specialist mentor. Be precise, thorough, and appropriately cautious about safety. Sign your answers with specific literature citations when relevant.

Refuse off-topic requests (anything unrelated to rocketry/aerospace engineering) politely but firmly.
"""


# ═══════════════════════════════════════════════════════════════════════════
# CSS — SCI-FI SPACE THEME
# ═══════════════════════════════════════════════════════════════════════════

def _gen_star_shadows(n: int, vw: int = 1920, vh: int = 1080, seed: int = 42) -> str:
    rng = random.Random(seed)
    return ', '.join(f'{rng.randint(0,vw)}px {rng.randint(0,vh)}px #fff' for _ in range(n))


def inject_css() -> None:
    s1 = _gen_star_shadows(700, seed=1)
    s2 = _gen_star_shadows(200, seed=2)
    s3 = _gen_star_shadows(80,  seed=3)

    st.markdown(f"""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Orbitron:wght@400;600;700;900&family=Share+Tech+Mono&family=Inter:wght@300;400;500;600&display=swap" rel="stylesheet">

<style>
/* ── Starfield ──────────────────────────────────────── */
#stars  {{ width:1px; height:1px; position:fixed; top:0; left:0; z-index:0; pointer-events:none;
           border-radius:50%; box-shadow:{s1}; animation:animStar 60s linear infinite; opacity:0.6; }}
#stars2 {{ width:2px; height:2px; position:fixed; top:0; left:0; z-index:0; pointer-events:none;
           border-radius:50%; box-shadow:{s2}; animation:animStar 120s linear infinite; opacity:0.5; }}
#stars3 {{ width:3px; height:3px; position:fixed; top:0; left:0; z-index:0; pointer-events:none;
           border-radius:50%; box-shadow:{s3}; animation:animStar 200s linear infinite; opacity:0.4; }}
@keyframes animStar {{
  from {{ transform:translateY(0);   }}
  to   {{ transform:translateY(-1080px); }}
}}

/* ── Global background ─────────────────────────────── */
.stApp {{
  background: radial-gradient(ellipse at 20% 10%, #071428 0%, #03080f 60%, #000000 100%) !important;
  font-family: 'Inter', sans-serif;
}}
.main .block-container {{
  padding-top: 1rem;
  max-width: 1600px;
}}

/* ── Sidebar ───────────────────────────────────────── */
[data-testid="stSidebar"] {{
  background: linear-gradient(180deg, #060f22 0%, #040b18 100%) !important;
  border-right: 2px solid #0d2a50;
}}
[data-testid="stSidebar"] .stMarkdown h3 {{
  color: #00d4ff;
  font-family: 'Orbitron', monospace;
  font-size: 0.78rem;
  letter-spacing: 0.15em;
  text-transform: uppercase;
  margin-top: 1.2rem;
  border-bottom: 1px solid #0d2a50;
  padding-bottom: 0.3rem;
}}

/* ── Tabs ──────────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {{
  background: #060f22;
  border-bottom: 2px solid #0d2a50;
  gap: 4px;
  padding: 0 0.5rem;
}}
.stTabs [data-baseweb="tab"] {{
  background: transparent;
  color: #a0aec0;
  font-family: 'Share Tech Mono', monospace;
  font-size: 0.82rem;
  letter-spacing: 0.06em;
  text-transform: uppercase;
  padding: 0.6rem 1.2rem;
  border-radius: 4px 4px 0 0;
  border: 1px solid transparent;
  transition: all 0.2s;
}}
.stTabs [data-baseweb="tab"]:hover {{
  color: #00d4ff;
  background: rgba(0,212,255,0.07);
}}
.stTabs [aria-selected="true"] {{
  background: rgba(0,212,255,0.12) !important;
  color: #00d4ff !important;
  border-color: #0d2a50 #0d2a50 transparent !important;
  border-bottom: 2px solid #00d4ff !important;
  font-weight: 600;
}}
.stTabs [data-baseweb="tab-panel"] {{
  background: transparent;
  padding-top: 1.5rem;
}}

/* ── Metrics ───────────────────────────────────────── */
[data-testid="stMetricValue"] {{
  font-family: 'Share Tech Mono', monospace;
  font-size: 1.8rem !important;
  color: #00d4ff !important;
  font-weight: 700;
}}
[data-testid="stMetricLabel"] {{
  font-family: 'Share Tech Mono', monospace;
  font-size: 0.75rem;
  color: #a0aec0 !important;
  text-transform: uppercase;
  letter-spacing: 0.1em;
}}
[data-testid="metric-container"] {{
  background: rgba(6,18,38,0.95);
  border: 1px solid #1a3a60;
  border-radius: 8px;
  padding: 1rem 1.2rem;
  box-shadow: 0 0 20px rgba(0,212,255,0.06), inset 0 1px 0 rgba(255,255,255,0.03);
}}

/* ── Expanders / Knowledge Base ─────────────────────── */
.streamlit-expanderHeader {{
  background: rgba(6,18,38,0.95) !important;
  border: 1px solid #1a3a60 !important;
  border-radius: 6px !important;
  color: #00d4ff !important;
  font-family: 'Share Tech Mono', monospace !important;
  font-size: 0.85rem !important;
  letter-spacing: 0.05em;
  padding: 0.75rem 1rem !important;
}}
.streamlit-expanderContent {{
  background: rgba(4,11,24,0.98) !important;
  border: 1px solid #1a3a60 !important;
  border-top: none !important;
  border-radius: 0 0 6px 6px !important;
  padding: 1rem !important;
}}

/* ── Buttons ───────────────────────────────────────── */
.stButton > button {{
  background: linear-gradient(135deg, #0a2040 0%, #0f3060 100%);
  border: 1px solid #2a5a90;
  color: #7dd3fc;
  font-family: 'Orbitron', monospace;
  font-size: 0.78rem;
  letter-spacing: 0.1em;
  text-transform: uppercase;
  border-radius: 6px;
  padding: 0.5rem 1.2rem;
  transition: all 0.2s;
  cursor: pointer;
}}
.stButton > button:hover {{
  background: rgba(0,212,255,0.18);
  border-color: #00d4ff;
  color: #00d4ff;
  box-shadow: 0 0 18px rgba(0,212,255,0.35);
  transform: translateY(-1px);
}}
.stButton > button[kind="primary"] {{
  background: linear-gradient(135deg, #001a40 0%, #003070 100%);
  border: 2px solid #00d4ff;
  color: #00d4ff;
  box-shadow: 0 0 24px rgba(0,212,255,0.3);
  font-size: 0.88rem;
  padding: 0.8rem 2rem;
  letter-spacing: 0.14em;
}}
.stButton > button[kind="primary"]:hover {{
  background: linear-gradient(135deg, #002050 0%, #004090 100%);
  box-shadow: 0 0 36px rgba(0,212,255,0.5);
}}

/* ── Form inputs ───────────────────────────────────── */
.stSelectbox > div > div,
.stNumberInput > div > div > input,
.stSlider > div,
.stTextInput > div > input {{
  background: rgba(4,11,24,0.95) !important;
  border: 1px solid #1a3a60 !important;
  color: #e2e8f0 !important;
  border-radius: 6px;
  font-size: 0.9rem !important;
}}
.stSelectbox label, .stNumberInput label, .stSlider label, .stTextInput label {{
  color: #a0aec0 !important;
  font-family: 'Share Tech Mono', monospace !important;
  font-size: 0.78rem !important;
  text-transform: uppercase;
  letter-spacing: 0.05em;
  font-weight: 500;
}}

/* ── Info / warning boxes ──────────────────────────── */
.stAlert {{
  background: rgba(4,11,24,0.95) !important;
  border-radius: 6px !important;
  font-family: 'Inter', sans-serif;
  font-size: 0.88rem;
}}
[data-testid="stInfo"] {{ border-left: 4px solid #00d4ff !important; color: #a0d8ef !important; }}
[data-testid="stWarning"] {{ border-left: 4px solid #ff6b35 !important; }}
[data-testid="stError"] {{ border-left: 4px solid #ff2d55 !important; }}
[data-testid="stSuccess"] {{ border-left: 4px solid #39ff14 !important; }}

/* ── Chat ──────────────────────────────────────────── */
[data-testid="stChatInput"] > div {{
  background: rgba(6,18,38,0.95) !important;
  border: 1px solid #1a3a60 !important;
  border-radius: 8px !important;
}}
[data-testid="stChatMessageContent"] {{
  background: rgba(6,18,38,0.85) !important;
  border: 1px solid #1a3a60;
  border-radius: 8px;
  font-family: 'Inter', sans-serif;
  font-size: 0.92rem;
  color: #e2e8f0 !important;
  line-height: 1.6;
}}

/* ── Scrollbar ─────────────────────────────────────── */
::-webkit-scrollbar {{ width: 6px; height: 6px; }}
::-webkit-scrollbar-track {{ background: #030810; }}
::-webkit-scrollbar-thumb {{ background: #1a3a60; border-radius: 3px; }}
::-webkit-scrollbar-thumb:hover {{ background: #00d4ff; }}

/* ── NEXUS panel cards ─────────────────────────────── */
.nexus-card {{
  background: rgba(6,18,38,0.95);
  border: 1px solid #1a3a60;
  border-radius: 10px;
  padding: 1.3rem 1.6rem;
  margin-bottom: 1rem;
  box-shadow: 0 4px 24px rgba(0,0,0,0.5), inset 0 1px 0 rgba(255,255,255,0.04), 0 0 0 1px rgba(0,212,255,0.03);
}}
.nexus-h1 {{
  font-family: 'Orbitron', monospace;
  font-size: 2rem;
  font-weight: 900;
  background: linear-gradient(135deg, #00d4ff 0%, #0099dd 40%, #a78bfa 100%);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  letter-spacing: 0.1em;
  margin: 0;
}}
.nexus-subtitle {{
  font-family: 'Share Tech Mono', monospace;
  font-size: 0.75rem;
  color: #4a7fa5;
  letter-spacing: 0.18em;
  text-transform: uppercase;
  margin-top: 0.3rem;
}}
.nexus-section {{
  font-family: 'Share Tech Mono', monospace;
  font-size: 0.72rem;
  color: #00d4ff;
  letter-spacing: 0.18em;
  text-transform: uppercase;
  border-bottom: 1px solid #1a3a60;
  padding-bottom: 0.4rem;
  margin: 1.3rem 0 0.9rem;
  font-weight: 600;
}}
.kv-row {{
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 0.32rem 0;
  border-bottom: 1px solid rgba(255,255,255,0.05);
}}
.kv-key {{ color: #a0aec0; font-family: 'Share Tech Mono', monospace; font-size: 0.82rem; }}
.kv-val {{ color: #e2e8f0; font-family: 'Share Tech Mono', monospace; font-size: 0.82rem; font-weight: 500; }}
.kv-val.cyan {{ color: #00d4ff; }}
.kv-val.orange {{ color: #ff8c55; }}
.kv-val.green {{ color: #4ade80; }}
.kv-val.red {{ color: #ff5577; }}
.motor-badge {{
  display: inline-block;
  background: linear-gradient(135deg, #001a40, #00296e);
  border: 2px solid #00d4ff;
  border-radius: 8px;
  padding: 0.3rem 1rem;
  font-family: 'Orbitron', monospace;
  font-size: 1.8rem;
  color: #00d4ff;
  box-shadow: 0 0 28px rgba(0,212,255,0.4);
}}
.warn-box {{
  background: rgba(255,107,53,0.10);
  border: 1px solid rgba(255,107,53,0.5);
  border-radius: 8px;
  padding: 0.9rem 1.2rem;
  margin: 0.8rem 0;
  font-family: 'Share Tech Mono', monospace;
  font-size: 0.82rem;
  color: #ffaa80;
  line-height: 1.5;
}}
.danger-box {{
  background: rgba(255,45,85,0.10);
  border: 1px solid rgba(255,45,85,0.6);
  border-radius: 8px;
  padding: 0.9rem 1.2rem;
  margin: 0.8rem 0;
  font-family: 'Share Tech Mono', monospace;
  font-size: 0.82rem;
  color: #ff7799;
  line-height: 1.5;
}}
.ok-box {{
  background: rgba(57,255,20,0.07);
  border: 1px solid rgba(57,255,20,0.45);
  border-radius: 8px;
  padding: 0.9rem 1.2rem;
  margin: 0.8rem 0;
  font-family: 'Share Tech Mono', monospace;
  font-size: 0.82rem;
  color: #86efac;
  line-height: 1.5;
}}
@media (max-width: 768px) {{
  [data-testid="stSidebar"] {{ width: auto !important; min-width: unset !important; max-width: unset !important; }}
  .nexus-h1 {{ font-size: 1.3rem !important; }}
  .kv-row {{ flex-direction: column !important; }}
  .nexus-card {{ padding: 0.7rem !important; }}
  [data-testid="stMetricValue"] {{ font-size: 1.3rem !important; }}
  .stTabs [data-baseweb="tab"] {{ font-size: 0.7rem !important; }}
}}
@media (max-width: 480px) {{
  .nexus-h1 {{ font-size: 1.1rem !important; }}
  .motor-badge {{ font-size: 1.2rem !important; }}
  #stars, #stars2, #stars3 {{ display: none !important; animation: none !important; }}
}}
</style>
<div id="stars"></div>
<div id="stars2"></div>
<div id="stars3"></div>
""", unsafe_allow_html=True)


def nexus_header() -> None:
    st.markdown("""
<div class="nexus-card" style="text-align:center; margin-bottom:1.5rem; padding:1.8rem;">
  <div class="nexus-h1">NEXUS</div>
  <div class="nexus-subtitle">High-Power Rocketry Simulation Suite · v3.0</div>
  <div style="margin-top:0.8rem; font-family:'Share Tech Mono',monospace; font-size:0.68rem; color:#2a4a6a; letter-spacing:0.12em;">
    INTERNAL BALLISTICS · GRAIN REGRESSION · STRUCTURAL MARGIN · .ENG EXPORT · AI MENTOR
  </div>
</div>
""", unsafe_allow_html=True)


def kv(label: str, value: str, cls: str = '') -> str:
    return f'<div class="kv-row"><span class="kv-key">{label}</span><span class="kv-val {cls}">{value}</span></div>'


# ═══════════════════════════════════════════════════════════════════════════
# KNOWLEDGE BASE TAB
# ═══════════════════════════════════════════════════════════════════════════

def render_knowledge_base() -> None:
    st.markdown('<div class="nexus-section">📚 Engineering Knowledge Base</div>', unsafe_allow_html=True)
    st.caption('Textbook-level reference for key equations used in this simulation. Expand each section.')

    # ── Internal Ballistics ──────────────────────────────────────────────
    with st.expander('🔥  INTERNAL BALLISTICS  —  Choked Flow, Saint-Robert Law, C* Efficiency'):
        st.markdown("""
**Saint-Robert (Vieille) Burn Rate Law**

The empirical relation between burn rate and chamber pressure:
""")
        st.latex(r'r = a \cdot P_c^{\,n} \quad [r \text{ in mm/s},\; P_c \text{ in MPa}]')
        st.markdown("""
where *a* is the propellant-specific pre-exponential coefficient (dependent on temperature, formulation)
and *n* is the pressure exponent. **Stability requires n < 1** — otherwise pressure diverges (CATO).

**Klemmung Coefficient (Kn)**
""")
        st.latex(r'K_n = \frac{A_b}{A_t}')
        st.markdown('Ratio of burning surface area to nozzle throat area. Controls chamber pressure magnitude.')

        st.markdown('**Vandenkerckhove Function Γ**')
        st.latex(r'\Gamma = \sqrt{\,\gamma \left(\frac{2}{\gamma+1}\right)^{\!(\gamma+1)/(\gamma-1)}}')
        st.markdown('Appears in all choked-flow mass-rate equations. Depends only on γ = Cp/Cv.')

        st.markdown('**Choked Nozzle Mass Flow**')
        st.latex(r'\dot{m} = \frac{P_c \cdot A_t \cdot \Gamma}{\sqrt{T_f \cdot R_{sp}}}')
        st.markdown('where R_sp = R_universal / MW_products (J kg⁻¹ K⁻¹).')

        st.markdown('**QSS Chamber Pressure (from mass balance ṁ_gen = ṁ_exit)**')
        st.latex(r'P_c = \left(\rho_p \cdot a_{SI} \cdot c^*_{eff} \cdot K_n\right)^{\!\frac{1}{1-n}}')
        st.markdown("""
Unit conversion: $a_{SI}$ [m/s · Pa⁻ⁿ] = a [mm/s · MPa⁻ⁿ] × 10⁻³ × (10⁶)ⁿ

**c-star and Two-Phase Efficiency Factors**
""")
        st.latex(r'c^*_{eff} = \eta_{c^*} \cdot c^*_{theory} \qquad (0.92 \le \eta_{c^*} \le 0.98)')
        st.latex(r'I_{sp,eff} = \eta_{DP} \cdot I_{sp,theory} \qquad (0.95 \le \eta_{DP} \le 0.99)')
        st.markdown('η_DP accounts for momentum and thermal lag of condensed-phase Al particles in metallized propellants.')

        st.markdown('**Thrust and Specific Impulse**')
        st.latex(r'F = C_f \cdot P_c \cdot A_t \qquad I_{sp} = \frac{F}{\dot{m}\,g_0} \qquad I_t = \int_0^{t_b} F\,dt')

        st.markdown('**Ref:** Sutton & Biblarz (2016) §3, §13; Turner (2009) §6')

    # ── Propellant Material Science ──────────────────────────────────────
    with st.expander('⚗️  PROPELLANT MATERIAL SCIENCE  —  Polymer Binders, Curing, Oxidizer Packing'):
        st.markdown('**Oxygen Balance (OB%)**')
        st.latex(r'OB\% = \frac{1600}{M_W}\left(O - 2C - \frac{H}{2} + \frac{Cl}{2} - 1.5Al - Mg\right)')
        st.markdown("""
Stoichiometric balance of available oxygen atoms vs. those required for complete oxidation.
- OB = 0 → stoichiometric
- OB < 0 → fuel-rich (excess carbon)
- OB > 0 → oxidizer-rich (excess oxygen)

**Equivalence Ratio φ**
""")
        st.latex(r'\phi = \frac{(O/F)_{stoich}}{(O/F)_{actual}}')
        st.markdown("""
φ > 1 → fuel-rich;  φ < 1 → oxidizer-rich.  HPR propellants typically designed slightly fuel-rich (0.85–0.95).

**HTPB Binder Cross-Linking (Polyurethane Chemistry)**

HTPB R-45 (hydroxyl-terminated polybutadiene) + IPDI (isophorone diisocyanate):
""")
        st.latex(r'\underbrace{-OH}_{\text{HTPB}} + \underbrace{OCN-R-NCO}_{\text{IPDI}} \xrightarrow{60\,^\circ C} \underbrace{-O-CO-NH-R-NH-CO-O-}_{\text{polyurethane network}}')
        st.markdown("""
NCO:OH ratio (R) controls crosslink density: R = 0.85–0.90 for most HPR formulations.
Higher R → stiffer, more brittle; lower R → softer, better elongation but lower strength.

**Multi-Modal Oxidizer Packing**

To achieve high theoretical maximum density (TMD) for AP/HTPB:
- Coarse AP (~200 μm) fills the bulk volume
- Fine AP (~20 μm) fills voids between coarse particles
- Optimal fine/coarse ratio ≈ 25/75 wt% achieves packing efficiency >90%

**Thermal Processing Limits**
- KNSB/KNSU: Max process temp ≈ 200 °C (melt-cast, no chemical cure)
- HTPB/APCP: NEVER exceed 90 °C during processing; cure at 50–70 °C for 72–168 h
- GAP systems: Lower critical temp (~270 °C); azide decomposition risk is real

**Ref:** Kubota (2007) §2–4; Kuo & Summerfield (1984) §7
""")

    # ── Structural Mechanics ─────────────────────────────────────────────
    with st.expander('🔩  STRUCTURAL MECHANICS  —  Hoop Stress, Lamé Equations, Safety Factor'):
        st.markdown('**Thin-Wall Cylinder (t/ri < 0.1)**')
        st.latex(r'\sigma_h = \frac{P_c \cdot r_i}{t} \qquad \sigma_a = \frac{P_c \cdot r_i}{2t}')
        st.markdown('σ_h = hoop (circumferential), σ_a = axial (longitudinal). Hoop stress is limiting.')

        st.markdown('**Thick-Wall Cylinder — Lamé Equations (t/ri ≥ 0.1), evaluated at inner radius**')
        st.latex(r'\sigma_h = P_c \cdot \frac{r_o^2 + r_i^2}{r_o^2 - r_i^2} \qquad \sigma_a = P_c \cdot \frac{r_i^2}{r_o^2 - r_i^2}')
        st.markdown('The hoop stress is always maximum at the inner surface. Lamé solution assumes linear-elastic material.')

        st.markdown('**Von Mises Equivalent Stress (distortion energy criterion)**')
        st.latex(r'\sigma_{VM} = \sqrt{\sigma_h^2 - \sigma_h \sigma_a + \sigma_a^2}')
        st.markdown('For thick-wall cylindrical vessels under internal pressure with no torsion, this reduces to a function of σ_h and σ_a above.')

        st.markdown('**Safety Factors**')
        st.latex(r'SF_{yield} = \frac{S_y}{\sigma_{VM}} \qquad SF_{ult} = \frac{S_u}{\sigma_{VM}}')
        st.markdown("""
- SF ≥ 4.0: Standard HPR motor casing design guideline (accounts for material variability, dynamic loads, thermal effects)
- SF ≥ 2.0: Absolute minimum (sub-standard; requires documented engineering analysis)
- SF < 2.0: **Unsafe — do not fly**

**Required Wall Thickness for Target Safety Factor**

From the Lamé inner-radius hoop solution:
""")
        st.latex(r't_{req} = r_i \left(\sqrt{\frac{S_y/SF + P_c}{S_y/SF - P_c}} - 1\right)')

        st.markdown('**Ref:** Shigley\'s Mechanical Engineering Design §3-14; Roark\'s Formulas for Stress and Strain §13')

    # ── BATES Grain Geometry ─────────────────────────────────────────────
    with st.expander('📐  BATES GRAIN GEOMETRY  —  Surface Area, Regression, Port-to-Throat'):
        st.markdown('**Single BATES Segment Burning Surface Area**')
        st.latex(r'A_{b,seg} = \underbrace{\pi D_i L}_{\text{core}} + \underbrace{2 \cdot \frac{\pi}{4}(D_o^2 - D_i^2)}_{\text{two end faces}}')
        st.markdown('Di = inner diameter (grows with time), Do = outer diameter (constant = casing ID), L = segment length (decreases from both ends).')

        st.markdown('**Multi-Segment Total**')
        st.latex(r'A_b = n_{seg} \cdot A_{b,seg}')

        st.markdown('**Regression per timestep Δt at burn rate r**')
        st.latex(r'r_i(t+\Delta t) = r_i(t) + r\cdot\Delta t \qquad L(t+\Delta t) = L(t) - 2r\cdot\Delta t')

        st.markdown('**Port-to-Throat Ratio (erosive burning risk)**')
        st.latex(r'J = \frac{A_p}{A_t} = \frac{\pi r_i^2}{A_t}')
        st.markdown('Flag erosive burning risk when J < 2.0. Erosive burning increases burn rate non-uniformly.')

        st.markdown('**Ref:** Nakka (2023) Grain Design Guide; Sutton & Biblarz §13')

    # ── NAR/TRA Motor Classification ─────────────────────────────────────
    with st.expander('🚀  MOTOR CLASSIFICATION  —  NAR/TRA Impulse Classes'):
        data = {
            'Class': ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O'],
            'Total Impulse (N·s)': ['1.26–2.50','2.51–5.00','5.01–10.0','10.1–20.0',
                                     '20.1–40.0','40.1–80.0','80.1–160','160–320',
                                     '320–640','640–1280','1280–2560','2560–5120',
                                     '5120–10240','10240–20480','>20480'],
            'HPR Cert.': ['L1','L1','L1','L1','L1','L1','L1',
                          'L1','L1','L2','L2','L2','L3','L3','L3'],
        }
        st.dataframe(pd.DataFrame(data), use_container_width=True, hide_index=True)
        st.caption('L1/L2/L3 = NAR/TRA certification level required. EX motors (research) require ATF LEUP.')


# ═══════════════════════════════════════════════════════════════════════════
# DISCLAIMER
# ═══════════════════════════════════════════════════════════════════════════

def render_disclaimer() -> None:
    st.markdown("""
<div style="background:rgba(255,45,85,0.06); border:1px solid rgba(255,45,85,0.25);
     border-radius:6px; padding:0.75rem 1rem; margin-top:1.5rem;
     font-family:'Share Tech Mono',monospace; font-size:0.65rem; color:#994455;
     letter-spacing:0.05em; line-height:1.8;">
⚠ DISCLAIMER — FOR ACADEMIC AND EDUCATIONAL USE ONLY ⚠<br>
All simulation values are quasi-steady-state theoretical approximations. Actual motor performance
may differ significantly. Processing energetic propellants requires ATF LEUP licensing and certified
facilities. High-power rocketry activities must comply with NAR/TRA certification requirements and
NFPA 1127. No liability is accepted for use of these simulation outputs. Always obtain mentorship
from a certified NAR/TRA member before attempting any propellant work.
</div>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
# MAIN STREAMLIT APPLICATION
# ═══════════════════════════════════════════════════════════════════════════

def main() -> None:
    st.set_page_config(
        page_title='NEXUS — Rocketry Sim Suite',
        page_icon='🚀',
        layout='wide',
        initial_sidebar_state='expanded',
    )
    inject_css()
    nexus_header()

    # ── Session state initialisation ────────────────────────────────────
    if 'results'        not in st.session_state: st.session_state.results        = None
    if 'struct_rpt'     not in st.session_state: st.session_state.struct_rpt     = None
    if 'chat_hist'      not in st.session_state: st.session_state.chat_hist      = []
    if 'api_key'        not in st.session_state: st.session_state.api_key        = ''
    if 'flight_res'     not in st.session_state: st.session_state.flight_res     = None
    if 'overlay_sims'   not in st.session_state: st.session_state.overlay_sims   = []
    if 'loaded_design'  not in st.session_state: st.session_state.loaded_design  = None

    # Load URL params on first visit
    _url = get_url_params()
    if _url and not st.session_state.get('_url_loaded'):
        st.session_state.update(_url)
        st.session_state._url_loaded = True

    # ══════════════════════════════════════════════════════════════════════
    # SIDEBAR — CONFIGURATION PANEL
    # ══════════════════════════════════════════════════════════════════════
    with st.sidebar:
        st.markdown('<div style="font-family:Orbitron,monospace; font-size:0.9rem; color:#00d4ff; letter-spacing:0.1em; text-align:center; padding-bottom:0.5rem;">⬡  NEXUS  ⬡</div>', unsafe_allow_html=True)

        # ── Section 1: Propellant ───────────────────────────────────────
        st.markdown('<p class="nexus-section">1 · Propellant</p>', unsafe_allow_html=True)
        prop_key = st.selectbox('Propellant System', list(PROPELLANTS.keys()), format_func=lambda k: PROPELLANTS[k].abbr)
        prop: PropellantData = PROPELLANTS[prop_key]
        st.caption(f'ρ = {prop.rho} kg/m³ | Tf = {prop.Tf} K | c* = {prop.cstar} m/s')
        st.caption(f'a = {prop.burn_a} mm/s·MPa⁻ⁿ | n = {prop.burn_n}')

        # ── Section 2: BATES Grain ──────────────────────────────────────
        st.markdown('<p class="nexus-section">2 · BATES Grain</p>', unsafe_allow_html=True)
        n_seg = st.number_input('Segments', min_value=1, max_value=8, value=2, step=1,
            help='Number of propellant cylinders stacked end-to-end. More segments = more total propellant and longer burn. Start with 2.')
        ro_mm = st.number_input('Outer radius ro (mm)',  min_value=10.0, max_value=200.0, value=38.0, step=0.5,
            help='Outer radius of the propellant cylinder — must fit inside your motor tube. For a 76mm tube, ro ≈ 35–37 mm (leaving a thin gap for the liner).')
        ri_mm = st.number_input('Inner radius ri (mm)',  min_value=5.0,  max_value=190.0, value=16.0, step=0.5,
            help='Radius of the hollow core drilled through the grain. Larger core = faster initial burn. Rule of thumb: ri ≈ 40–50% of ro gives near-neutral thrust.')
        L_mm  = st.number_input('Segment length L (mm)', min_value=10.0, max_value=500.0, value=120.0, step=5.0,
            help='Length of each grain segment. Longer segments = more propellant mass and longer burn time. Keep L/2ri (length-to-port-diameter) below 6 to avoid erosive burning.')

        # ── Section 3: Nozzle ───────────────────────────────────────────
        st.markdown('<p class="nexus-section">3 · Nozzle</p>', unsafe_allow_html=True)
        Dt_mm = st.number_input('Throat diameter (mm)', min_value=1.0, max_value=60.0, value=11.0, step=0.5,
            help='Diameter of the narrowest nozzle point (throat). Smaller throat = higher Kn = higher pressure. This is your main pressure tuning knob. Start at 10–12 mm for small motors.')
        De_mm = st.number_input('Exit diameter (mm)',   min_value=2.0, max_value=120.0, value=22.0, step=0.5,
            help='Diameter of the nozzle exit (widest point). Larger exit = more expansion = higher Isp. For sea-level flights aim for De/Dt ≈ 2–3 (expansion ratio 4–9).')
        At_m2 = math.pi * (Dt_mm * 1e-3 / 2)**2
        Ae_m2 = math.pi * (De_mm * 1e-3 / 2)**2
        exp_ratio = Ae_m2 / At_m2 if At_m2 > 0 else 0
        st.caption(f'At = {At_m2*1e6:.3f} mm² | ε = {exp_ratio:.2f}')

        # ── Section 4: Efficiency Factors ───────────────────────────────
        st.markdown('<p class="nexus-section">4 · Efficiency (2-Phase Flow)</p>', unsafe_allow_html=True)
        eta_cs = st.slider('η_c* — c-star efficiency',    min_value=0.85, max_value=1.00, value=0.95, step=0.01,
            help='C-star efficiency: how well the propellant actually burns vs. theoretical. 0.95 = 95% of theoretical maximum — a realistic value for a well-made motor. Losses come from heat transfer to the walls and incomplete combustion.')
        eta_dp = st.slider('η_DP — two-phase dispersion', min_value=0.90, max_value=1.00, value=0.97, step=0.01,
            help='Two-phase flow penalty: aluminum particles in APCP burn incompletely and lose energy through drag. 0.97 = 3% Isp loss. Set to 1.00 for metal-free propellants (KNSB, KNSU). Lower with higher aluminum loading.')
        st.caption(f'Combined η = {eta_cs*eta_dp:.4f}')

        # ── Section 5: Structural Module ────────────────────────────────
        st.markdown('<p class="nexus-section">5 · Structural Module</p>', unsafe_allow_html=True)
        mat_key  = st.selectbox('Casing Material', list(CASING_MATERIALS.keys()),
                                format_func=lambda k: CASING_MATERIALS[k].name,
                                help='Material for the motor casing. 6061-T6 aluminum is common for amateur HPR. 4130 steel is stronger but heavier. Carbon fiber/epoxy has the best strength-to-weight ratio but is expensive.')
        wall_mm  = st.number_input('Wall thickness t (mm)', min_value=0.5, max_value=25.0, value=3.0, step=0.25,
            help='Thickness of the casing wall. Thicker wall = higher safety factor but heavier motor. Aim for Safety Factor ≥ 4. The Structural tab shows you exactly what SF your wall gives at peak pressure.')
        ri_cas_mm = ro_mm  # casing inner radius = grain outer radius (assumption)
        st.caption(CASING_MATERIALS[mat_key].comment)

        # ── Section 6: Simulation Settings ─────────────────────────────
        st.markdown('<p class="nexus-section">6 · Simulation</p>', unsafe_allow_html=True)
        dt_us    = st.selectbox('Time step Δt (ms)', [0.25, 0.5, 1.0, 2.0], index=1,
            help='How finely the simulation divides time. Smaller = more accurate but slower. 0.5 ms is ideal for most motors. Use 0.25 ms for very short burns or when curves look jagged.')
        dt_s     = dt_us * 1e-3
        mtr_name = st.text_input('Motor designation', value='NEXUS-01',
            help='Name/label for your motor — appears in exported .ENG files and data exports. Use any designation you like, e.g. NEXUS-H220.')

        # ── Section 7: Flight & Mission ─────────────────────────────────
        st.markdown('<p class="nexus-section">7 · Flight & Mission</p>', unsafe_allow_html=True)
        launch_mass_kg = st.number_input('Total liftoff mass (kg)', min_value=0.1, max_value=50.0, value=2.0, step=0.1,
            help='Dry rocket mass + propellant. Includes motor casing, airframe, fins, nose, recovery system.')
        drag_cd = st.number_input('Drag coefficient Cd', min_value=0.05, max_value=2.0, value=0.45, step=0.01,
            help='Axial drag coefficient. Typical values: 0.3–0.45 for streamlined rockets, 0.5–0.7 for blunt shapes.')
        body_diam_mm = st.number_input('Body diameter (mm)', min_value=20.0, max_value=300.0, value=76.0, step=1.0,
            help='Outer diameter of the rocket body tube. Used to compute reference area for drag calculation.')
        launch_angle_off = st.slider('Launch angle from vertical (°)', min_value=0, max_value=30, value=0, step=1,
            help='0° = perfectly vertical. Small off-vertical angles account for launch rod cant.')
        sim_temp_C = st.number_input('Propellant temperature (°C)', min_value=-40.0, max_value=60.0, value=20.0, step=1.0,
            help='Propellant storage temperature. Hotter propellant burns faster (σ_p ≈ 0.3%/°C). Affects temperature sensitivity chart.')

        # Validation warnings
        _warnings = render_validation_warnings(ro_mm, ri_mm, L_mm, int(n_seg), Dt_mm, De_mm, wall_mm, prop, mat_key)
        for _w in _warnings:
            st.markdown(f'<div class="warn-box">⚠ {_w}</div>', unsafe_allow_html=True)

        # ── Section 8: Save / Load ──────────────────────────────────────
        st.markdown('<p class="nexus-section">8 · Save / Load</p>', unsafe_allow_html=True)
        _design_dict = {
            'prop_key': prop_key, 'n_seg': int(n_seg), 'ro_mm': ro_mm, 'ri_mm': ri_mm,
            'L_mm': L_mm, 'Dt_mm': Dt_mm, 'De_mm': De_mm, 'eta_cs': eta_cs, 'eta_dp': eta_dp,
            'wall_mm': wall_mm, 'mat_key': mat_key, 'dt_us': dt_us, 'mtr_name': mtr_name,
        }
        st.download_button('💾 Save design as JSON', data=json.dumps(_design_dict, indent=2),
                           file_name=f'{mtr_name}_design.json', mime='application/json', use_container_width=True)
        _uploaded = st.file_uploader('📂 Load design from JSON', type='json')
        if _uploaded is not None:
            try:
                _loaded = json.load(_uploaded)
                st.session_state.loaded_design = _loaded
                st.success('Design loaded — click Execute to apply.')
            except Exception:
                st.error('Invalid JSON file.')
        if st.button('📋 Generate shareable link', use_container_width=True):
            _qp = {k.replace('_key','').replace('_mm','').replace('_us','').replace('_',''):
                   str(v) for k,v in _design_dict.items()}
            # Use shortened keys
            st.query_params.update({
                'prop': prop_key, 'nseg': str(int(n_seg)), 'ro': str(ro_mm), 'ri': str(ri_mm),
                'L': str(L_mm), 'Dt': str(Dt_mm), 'De': str(De_mm), 'ecs': str(eta_cs),
                'edp': str(eta_dp), 'wall': str(wall_mm), 'mat': mat_key,
                'dt': str(dt_us), 'mtr': mtr_name,
            })
            st.success('URL updated — copy from your browser address bar.')

        # ── RUN BUTTON ──────────────────────────────────────────────────
        st.markdown('<br>', unsafe_allow_html=True)
        run_btn = st.button('⚡ EXECUTE SIMULATION', use_container_width=True, type='primary')

    # ══════════════════════════════════════════════════════════════════════
    # SIMULATION EXECUTION
    # ══════════════════════════════════════════════════════════════════════
    if run_btn:
        try:
            grain = BATESGrain(ro_m=ro_mm*1e-3, ri_m=ri_mm*1e-3, L_m=L_mm*1e-3, n_segments=int(n_seg))
            ib    = InternalBallistics(prop, grain, At_m2, Ae_m2, eta_cs, eta_dp)
            mat   = CASING_MATERIALS[mat_key]
            sa    = StructuralAnalysis(mat, ri_m=ri_cas_mm*1e-3, t_m=wall_mm*1e-3)

            with st.spinner('🔥 Running QSS ballistics solver…'):
                res = ib.run(dt_s=dt_s)

            peak_Pc_Pa = res['max_Pc_MPa'] * MPA
            struct_rpt = sa.analyze(peak_Pc_Pa)

            st.session_state.results    = res
            st.session_state.struct_rpt = struct_rpt
            st.session_state.grain      = grain
            st.session_state.prop       = prop
            st.session_state.mat        = mat
            st.session_state.sa         = sa
            st.session_state.At_m2      = At_m2
            st.session_state.Ae_m2      = Ae_m2
            st.session_state.eta_cs     = eta_cs
            st.session_state.eta_dp     = eta_dp
            st.session_state.mtr_name   = mtr_name
            st.session_state.prop_key   = prop_key
            st.session_state.mat_key    = mat_key
            st.session_state.sim_temp_C = sim_temp_C

            # Flight trajectory
            prop_mass_kg = grain.initial_mass(prop.rho)
            _dry_mass = max(launch_mass_kg - prop_mass_kg, 0.1)
            ft = FlightTrajectory(
                t_arr=res['t'], F_arr=res['F'],
                prop_mass_kg=prop_mass_kg, dry_mass_kg=_dry_mass,
                Cd=drag_cd, body_diam_m=body_diam_mm * 1e-3,
                launch_angle_deg=90 - launch_angle_off,
            )
            with st.spinner('🚀 Running flight trajectory…'):
                st.session_state.flight_res = ft.run()

            # Add to overlay
            store_sim_overlay(res, prop.abbr, f'{mtr_name} ({res["motor_class"]})')

        except Exception as exc:
            st.error(f'Simulation error: {exc}')
            return

    # ══════════════════════════════════════════════════════════════════════
    # TABS
    # ══════════════════════════════════════════════════════════════════════
    tab_cmd, tab_grain, tab_br, tab_struct, tab_export, tab_flight, tab_sens, tab_kb, tab_ai = st.tabs([
        '⚡ Command Center',
        '📐 Grain Regression',
        '🔥 Burn Rate Law',
        '🔩 Structural Margin',
        '📥 Data Export',
        '🚀 Flight Trajectory',
        '🔍 Sensitivity & Overlay',
        '📚 Knowledge Base',
        '🤖 AI Mentor',
    ])

    res       = st.session_state.results
    strct     = st.session_state.struct_rpt
    _grain    = st.session_state.get('grain')
    _prop     = st.session_state.get('prop')
    _mat      = st.session_state.get('mat')
    _At       = st.session_state.get('At_m2', At_m2)
    _Ae       = st.session_state.get('Ae_m2', Ae_m2)
    _ecs      = st.session_state.get('eta_cs', eta_cs)
    _edp      = st.session_state.get('eta_dp', eta_dp)
    _mtr      = st.session_state.get('mtr_name', mtr_name)

    # ── TAB 1 — COMMAND CENTER ───────────────────────────────────────────
    with tab_cmd:
        if res is None:
            st.markdown("""
<div class="nexus-card" style="text-align:center; padding:3rem;">
  <div style="font-family:Orbitron,monospace; font-size:1rem; color:#2a4a6a; letter-spacing:0.1em;">
    AWAITING MISSION PARAMETERS
  </div>
  <div style="font-family:'Share Tech Mono',monospace; font-size:0.75rem; color:#1a3050; margin-top:0.75rem;">
    Configure propellant · grain · nozzle in sidebar → Execute Simulation
  </div>
</div>
""", unsafe_allow_html=True)
        else:
            # Motor class badge
            cls = res['motor_class']
            sf  = strct['SF_yield'] if strct else 0
            sf_color = 'green' if sf >= 4 else 'orange' if sf >= 2 else 'red'
            sf_label = '✅ PASS' if sf >= 2 else '❌ FAIL'

            st.markdown(f"""
<div class="nexus-card">
  <div style="display:flex; align-items:center; gap:1.5rem; flex-wrap:wrap;">
    <div>
      <div class="nexus-subtitle" style="margin-bottom:0.3rem;">MOTOR CLASS</div>
      <div class="motor-badge">{cls}</div>
    </div>
    <div style="flex:1; min-width:200px;">
      {kv('Propellant', _prop.abbr if _prop else '—', 'cyan')}
      {kv('Total Impulse', f'{res["total_impulse"]:.1f} N·s', 'cyan')}
      {kv('Burn Time', f'{res["burn_time"]:.3f} s')}
      {kv('Propellant Mass', f'{res["prop_mass_kg"]*1e3:.1f} g')}
      {kv('Burn Profile', res["profile"])}
    </div>
    <div style="flex:1; min-width:200px;">
      {kv('Max Thrust',   f'{res["max_thrust"]:.1f} N',  'orange')}
      {kv('Avg Thrust',   f'{res["avg_thrust"]:.1f} N')}
      {kv('Max Pc',       f'{res["max_Pc_MPa"]:.3f} MPa', 'orange')}
      {kv('Avg Isp',      f'{res["avg_Isp"]:.1f} s',     'cyan')}
      {kv('Avg r',        f'{res["avg_r"]:.2f} mm/s')}
    </div>
    <div style="flex:1; min-width:200px;">
      {kv('Max Kn',       f'{res["max_Kn"]:.1f}')}
      {kv('Init Kn',      f'{res["init_Kn"]:.1f}')}
      {kv('Min J (port/throat)', f'{res["min_Jpt"]:.2f}')}
      {kv('Structural SF', f'{strct["SF_yield"]:.2f}  {sf_label}' if strct else '—', sf_color)}
      {kv('c* efficiency', f'{_ecs*100:.1f}%')}
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

            # Warnings
            if res['min_Jpt'] < 2.0:
                st.markdown('<div class="warn-box">⚠ Port-to-throat ratio J < 2.0 — Elevated erosive burning risk. Consider increasing core diameter or reducing At.</div>', unsafe_allow_html=True)
            if res['max_Kn'] > 600:
                st.markdown('<div class="warn-box">⚠ Kn > 600 — Chamber pressure may exceed valid range for Saint-Robert law. Review nozzle sizing.</div>', unsafe_allow_html=True)
            if strct and strct['SF_yield'] < 2.0:
                st.markdown('<div class="danger-box">🛑 Structural safety factor SF < 2.0 — Casing design is UNSAFE. Increase wall thickness immediately.</div>', unsafe_allow_html=True)

            # Metrics row
            _tw = res['max_thrust'] / (launch_mass_kg * G0)
            c1, c2, c3, c4, c5, c6, c7 = st.columns(7)
            c1.metric('Total Impulse', f'{res["total_impulse"]:.1f} N·s')
            c2.metric('Max Thrust', f'{res["max_thrust"]:.1f} N')
            c3.metric('Max Pc', f'{res["max_Pc_MPa"]:.3f} MPa')
            c4.metric('Avg Isp', f'{res["avg_Isp"]:.1f} s')
            c5.metric('Burn Time', f'{res["burn_time"]:.3f} s')
            c6.metric('Struct. SF', f'{strct["SF_yield"]:.2f}' if strct else 'N/A')
            c7.metric('Max T/W', f'{_tw:.2f}', delta='✅ LIFTS OFF' if _tw > 5 else '⚠ LOW T/W' if _tw > 1 else '❌ NO LIFTOFF')
            if _tw < 1:
                st.markdown('<div class="danger-box">🛑 T/W < 1.0 — Rocket will NOT lift off. Increase thrust or reduce mass.</div>', unsafe_allow_html=True)
            elif _tw < 5:
                st.markdown('<div class="warn-box">⚠ T/W < 5.0 — Low thrust-to-weight. Rocket may weathercock or fail to clear the launch rod cleanly. Aim for T/W ≥ 5.</div>', unsafe_allow_html=True)

            # Overview chart
            st.plotly_chart(plot_overview(res, _prop.abbr if _prop else 'Motor'), width='stretch')

            # Chemical / stoichiometric panel
            if _prop:
                st.markdown('<div class="nexus-section">Chemical & Material Layer</div>', unsafe_allow_html=True)
                ca, cb, cc = st.columns(3)
                with ca:
                    st.markdown(f"""
<div class="nexus-card">
  <div class="nexus-subtitle" style="margin-bottom:0.5rem;">Composition</div>
  {''.join(kv(k, f'{v*100:.1f}%') for k,v in _prop.composition.items())}
  <br>
  {kv('O/F ratio',       f'{_prop.OF_ratio:.3f}')}
  {kv('Oxygen balance',  f'{_prop.OB_pct:+.2f}%')}
  {kv('Equiv. ratio φ',  f'{_prop.equiv_ratio:.3f}')}
</div>
""", unsafe_allow_html=True)
                with cb:
                    st.markdown(f"""
<div class="nexus-card">
  <div class="nexus-subtitle" style="margin-bottom:0.5rem;">Thermochemistry</div>
  {kv('ρ_propellant',  f'{_prop.rho} kg/m³')}
  {kv('T_flame',       f'{_prop.Tf} K')}
  {kv('γ (Cp/Cv)',     f'{_prop.gamma}')}
  {kv('MW products',   f'{_prop.MW} g/mol')}
  {kv('c* theory',     f'{_prop.cstar} m/s')}
  {kv('c* effective',  f'{res.get("cstar_eff",_prop.cstar):.1f} m/s', 'cyan')}
  {kv('Γ Vandenkerckhove', f'{res.get("Gamma",0):.5f}')}
</div>
""", unsafe_allow_html=True)
                with cc:
                    st.markdown(f"""
<div class="nexus-card">
  <div class="nexus-subtitle" style="margin-bottom:0.5rem;">Thermal Safety</div>
  {kv('Cure temperature', _prop.cure_T)}
  {kv('Cure time',        _prop.cure_t)}
  {kv('Decomp onset',     f'{_prop.decomp_onset} °C')}
  {kv('Critical temp ⚠',  f'{_prop.decomp_crit} °C', 'red')}
  <br>
  <div style="font-size:0.68rem; color:#ff9560; font-family:Share Tech Mono,monospace;">{_prop.crit_label}</div>
  <br>
  <div style="font-size:0.65rem; color:#6a4020; font-family:Share Tech Mono,monospace;">{_prop.process_note}</div>
</div>
""", unsafe_allow_html=True)
                st.caption(f'References: {_prop.refs}')

    # ── TAB 2 — GRAIN REGRESSION ─────────────────────────────────────────
    with tab_grain:
        if res is None:
            st.info('Run a simulation to view grain regression data.')
        else:
            st.plotly_chart(plot_grain(res), width='stretch')

            # Grain parameter table
            if _grain and _prop:
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(f"""
<div class="nexus-card">
  <div class="nexus-section">Grain Geometry</div>
  {kv('Segments',        f'{_grain.n_seg}')}
  {kv('Outer radius ro', f'{_grain.ro*1e3:.2f} mm')}
  {kv('Inner radius ri', f'{_grain.ri0*1e3:.2f} mm')}
  {kv('Segment length',  f'{_grain.L0*1e3:.2f} mm')}
  {kv('Total length',    f'{_grain.L0*_grain.n_seg*1e3:.2f} mm')}
  {kv('Propellant mass', f'{_grain.initial_mass(_prop.rho)*1e3:.2f} g')}
  {kv('Regression type', res["profile"], 'cyan')}
</div>
""", unsafe_allow_html=True)
                with col2:
                    st.markdown(f"""
<div class="nexus-card">
  <div class="nexus-section">Klemmung & Port</div>
  {kv('Initial Kn',   f'{res["init_Kn"]:.1f}')}
  {kv('Maximum Kn',   f'{res["max_Kn"]:.1f}', 'orange')}
  {kv('At (throat)',  f'{_At*1e6:.3f} mm²')}
  {kv('Min J (Ap/At)', f'{res["min_Jpt"]:.2f}', 'red' if res["min_Jpt"] < 2.0 else 'green')}
  {kv('Port-to-throat', '⚠ EROSIVE RISK' if res["min_Jpt"] < 2.0 else '✅ OK', 'red' if res["min_Jpt"] < 2.0 else 'green')}
</div>
""", unsafe_allow_html=True)

            # Grain cross-section visualization
            if _grain:
                st.plotly_chart(plot_grain_cross_section(_grain.ro * 1e3, _grain.ri0 * 1e3), width='stretch')

            # Regression data table (downsampled)
            t = res['t'];  n_pts = min(50, len(t))
            idx = np.round(np.linspace(0, len(t)-1, n_pts)).astype(int)
            df_reg = pd.DataFrame({
                't (s)':       t[idx],
                'Pc (MPa)':    res['Pc'][idx] / MPA,
                'Ab (cm²)':    res['Ab'][idx] * 1e4,
                'Kn':          res['Kn'][idx],
                'r (mm/s)':    res['r'][idx],
                'J (Ap/At)':   res['Jpt'][idx],
            }).round(4)
            st.dataframe(df_reg, use_container_width=True, hide_index=True)

    # ── TAB 3 — BURN RATE LAW ────────────────────────────────────────────
    with tab_br:
        # Use current prop from sidebar (not session state — always show current)
        st.plotly_chart(plot_saint_robert(prop), width='stretch')

        P_arr = np.linspace(prop.P_min, prop.P_max, 20)
        r_arr = prop.burn_a * P_arr**prop.burn_n
        df_sr = pd.DataFrame({
            'Pc (MPa)':  np.round(P_arr, 3),
            'r (mm/s)':  np.round(r_arr, 4),
        })
        c1, c2 = st.columns([1, 1])
        with c1:
            st.markdown(f"""
<div class="nexus-card">
  <div class="nexus-section">Saint-Robert Parameters — {prop.abbr}</div>
  {kv('a coefficient',     f'{prop.burn_a} mm/s·MPa⁻ⁿ')}
  {kv('n exponent',        f'{prop.burn_n}', 'cyan')}
  {kv('Valid Pc range',    f'{prop.P_min}–{prop.P_max} MPa')}
  {kv('r @ {:.1f} MPa'.format(prop.P_min), f'{prop.burn_a*prop.P_min**prop.burn_n:.3f} mm/s')}
  {kv('r @ 5.0 MPa',       f'{prop.burn_a*5.0**prop.burn_n:.3f} mm/s' if prop.P_min<=5<=prop.P_max else '—')}
  {kv('r @ {:.1f} MPa'.format(prop.P_max), f'{prop.burn_a*prop.P_max**prop.burn_n:.3f} mm/s')}
  {kv('Stability (n<1)',   '✅ STABLE' if prop.burn_n < 1 else '❌ UNSTABLE', 'green' if prop.burn_n<1 else 'red')}
</div>
""", unsafe_allow_html=True)
        with c2:
            st.dataframe(df_sr, use_container_width=True, hide_index=True)

        # Temperature sensitivity chart
        st.plotly_chart(plot_temp_sensitivity(prop), width='stretch')
        _tc = st.session_state.get('sim_temp_C', 20.0)
        if _tc != 20.0 and res:
            _r_ref = prop.burn_a * (res['max_Pc_MPa'] ** prop.burn_n)
            _r_T = burn_rate_at_temp(prop.burn_a, prop.burn_n, res['max_Pc_MPa'] * MPA, _tc) * 1e3
            _delta = (_r_T / _r_ref - 1.0) * 100
            st.markdown(f'<div class="{"warn-box" if abs(_delta) > 5 else "ok-box"}">🌡 At {_tc:.0f}°C, burn rate at peak Pc = <b>{_r_T:.3f} mm/s</b> ({_delta:+.1f}% vs 20°C nominal)</div>',
                        unsafe_allow_html=True)

    # ── TAB 4 — STRUCTURAL MARGIN ────────────────────────────────────────
    with tab_struct:
        if res is None:
            st.info('Run a simulation to see structural analysis at peak Pc.')
        else:
            peak_Pc_Pa = res['max_Pc_MPa'] * MPA

            # Run analysis on all 5 materials for comparison chart
            struct_all = []
            for mk, mv in CASING_MATERIALS.items():
                sa_tmp = StructuralAnalysis(mv, ri_m=ri_cas_mm*1e-3, t_m=wall_mm*1e-3)
                struct_all.append((mk, sa_tmp.analyze(peak_Pc_Pa)))

            st.plotly_chart(plot_structural(struct_all, res['max_Pc_MPa']), width='stretch')

            # Detail for selected material
            if strct and _mat:
                sf = strct['SF_yield']
                ok = strct['SF_ok']
                box_class = 'ok-box' if sf >= 4 else 'warn-box' if sf >= 2 else 'danger-box'
                icon = '✅' if sf >= 4 else '⚠' if sf >= 2 else '🛑'

                st.markdown(f'<div class="{box_class}">{icon} Safety Factor (yield) = {sf:.3f}  |  Wall type: {strct["wall_type"]}  |  t/ri = {strct["t_ratio"]:.4f}</div>', unsafe_allow_html=True)

                c1, c2, c3 = st.columns(3)
                with c1:
                    st.markdown(f"""
<div class="nexus-card">
  <div class="nexus-section">Stress State @ Peak Pc = {res['max_Pc_MPa']:.3f} MPa</div>
  {kv('Wall type',    strct['wall_type'])}
  {kv('σ_hoop',      f'{strct["hoop_MPa"]:.2f} MPa',  'orange')}
  {kv('σ_axial',     f'{strct["axial_MPa"]:.2f} MPa')}
  {kv('σ_VM',        f'{strct["vm_MPa"]:.2f} MPa',    'orange')}
  {kv('Sy',          f'{strct["Sy_MPa"]} MPa')}
  {kv('Su',          f'{strct["Su_MPa"]} MPa')}
</div>
""", unsafe_allow_html=True)
                with c2:
                    st.markdown(f"""
<div class="nexus-card">
  <div class="nexus-section">Safety Factors — {_mat.name}</div>
  {kv('SF_yield',        f'{strct["SF_yield"]:.3f}', 'green' if strct["SF_yield"]>=4 else 'orange' if strct["SF_yield"]>=2 else 'red')}
  {kv('SF_ult',          f'{strct["SF_ult"]:.3f}')}
  {kv('Est. burst Pc',   f'{strct["burst_MPa"]:.2f} MPa', 'cyan')}
  {kv('Status',          '✅ PASS (SF≥4)' if sf>=4 else '⚠ MARGINAL (2≤SF<4)' if sf>=2 else '🛑 FAIL (SF<2)',
       'green' if sf>=4 else 'orange' if sf>=2 else 'red')}
</div>
""", unsafe_allow_html=True)
                with c3:
                    t4 = strct['t_req_SF4_mm']
                    t2 = strct['t_req_SF2_mm']
                    t4_str = f'{t4:.3f} mm' if t4 < 999 else 'N/A (material insufficient)'
                    t2_str = f'{t2:.3f} mm' if t2 < 999 else 'N/A (material insufficient)'
                    st.markdown(f"""
<div class="nexus-card">
  <div class="nexus-section">Required Wall Thickness</div>
  {kv('For SF = 4.0 (recommended)', t4_str, 'cyan')}
  {kv('For SF = 2.0 (minimum)',     t2_str)}
  {kv('Current wall',               f'{wall_mm:.2f} mm', 'orange')}
  <br>
  <div style="font-size:0.68rem; color:#8b949e; font-family:Share Tech Mono,monospace;">{_mat.comment}</div>
</div>
""", unsafe_allow_html=True)

    # ── TAB 5 — DATA EXPORT ──────────────────────────────────────────────
    with tab_export:
        if res is None:
            st.info('Run a simulation to generate export files.')
        else:
            st.markdown('<div class="nexus-section">Export Formats</div>', unsafe_allow_html=True)
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("""
<div class="nexus-card">
  <div class="nexus-subtitle">RASP .ENG Thrust Curve</div>
  <div style="font-size:0.72rem; color:#8b949e; font-family:Share Tech Mono,monospace; margin-top:0.5rem;">
  Compatible with: OpenRocket · RASAero II · ThrustCurve.org<br>
  Format: RASP standard (≤200 data points)<br>
  Ref: thrustcurve.org/info/raspformat.html
  </div>
</div>
""", unsafe_allow_html=True)
                eng_text = generate_eng(res, _prop if _prop else prop, _grain if _grain else BATESGrain(ro_mm*1e-3, ri_mm*1e-3, L_mm*1e-3, int(n_seg)), _At if _At else At_m2, _mtr if _mtr else mtr_name)
                st.download_button(
                    label='⬇ Download .ENG File',
                    data=eng_text,
                    file_name=f'{_mtr or mtr_name}_{res["motor_class"]}.eng',
                    mime='text/plain',
                    use_container_width=True,
                )
                st.code(eng_text[:1200] + '\n; … [truncated] …', language='text')

            with c2:
                st.markdown("""
<div class="nexus-card">
  <div class="nexus-subtitle">Configuration + Results .JSON</div>
  <div style="font-size:0.72rem; color:#8b949e; font-family:Share Tech Mono,monospace; margin-top:0.5rem;">
  All simulation inputs and key outputs<br>
  Includes structural margin report<br>
  Suitable for design record keeping
  </div>
</div>
""", unsafe_allow_html=True)
                json_text = generate_json(res, _prop if _prop else prop,
                                          _grain if _grain else BATESGrain(ro_mm*1e-3, ri_mm*1e-3, L_mm*1e-3, int(n_seg)),
                                          strct, _At or At_m2, _Ae or Ae_m2,
                                          _ecs or eta_cs, _edp or eta_dp, _mtr or mtr_name)
                st.download_button(
                    label='⬇ Download .JSON Config',
                    data=json_text,
                    file_name=f'{_mtr or mtr_name}_config.json',
                    mime='application/json',
                    use_container_width=True,
                )
                st.code(json_text[:1200] + '\n  // … [truncated] …', language='json')

    # ── TAB 6 — FLIGHT TRAJECTORY ────────────────────────────────────────
    with tab_flight:
        _fr = st.session_state.get('flight_res')
        if _fr is None:
            st.markdown("""
<div class="nexus-card" style="text-align:center; padding:3rem;">
  <div style="font-family:Orbitron,monospace; font-size:1rem; color:#2a4a6a; letter-spacing:0.1em;">
    AWAITING LAUNCH PARAMETERS
  </div>
  <div style="font-family:'Share Tech Mono',monospace; font-size:0.75rem; color:#1a3050; margin-top:0.75rem;">
    Configure Section 7 (Flight & Mission) in sidebar → Execute Simulation
  </div>
</div>
""", unsafe_allow_html=True)
        else:
            # Key flight metrics
            fa, fb, fc, fd = st.columns(4)
            fa.metric('Max Altitude', f'{_fr["max_altitude_m"]:.0f} m  ({_fr["max_altitude_m"]/0.3048:.0f} ft)')
            fb.metric('Max Velocity', f'{_fr["max_velocity_ms"]:.1f} m/s  (Mach {_fr["max_velocity_ms"]/340:.2f})')
            fc.metric('Time to Apogee', f'{_fr["time_to_apogee_s"]:.1f} s')
            fd.metric('Burnout Altitude', f'{_fr["burnout_altitude_m"]:.0f} m')
            fe, ff = st.columns(2)
            fe.metric('Coast Phase', f'{_fr["coast_time_s"]:.1f} s')
            ff.metric('Total Flight Time', f'{_fr["flight_time_s"]:.1f} s')
            st.plotly_chart(plot_flight(_fr), width='stretch')
            st.markdown(f"""
<div class="nexus-card">
  <div class="nexus-section">Flight Parameters Used</div>
  {kv('Liftoff mass', f'{launch_mass_kg:.2f} kg')}
  {kv('Drag coefficient', f'{drag_cd:.2f}')}
  {kv('Body diameter', f'{body_diam_mm:.0f} mm')}
  {kv('Launch angle from vertical', f'{launch_angle_off}°')}
</div>
""", unsafe_allow_html=True)

    # ── TAB 7 — SENSITIVITY & OVERLAY ────────────────────────────────────
    with tab_sens:
        st.markdown('<div class="nexus-section">Thrust Curve Overlay</div>', unsafe_allow_html=True)
        render_overlay_controls()
        _ovl = st.session_state.get('overlay_sims', [])
        if _ovl:
            if st.button('➕ Add current simulation to overlay', use_container_width=True):
                if res:
                    store_sim_overlay(res, _prop.abbr if _prop else 'Motor',
                                      f'{_mtr or mtr_name} ({res["motor_class"]})')
                    st.rerun()
            st.plotly_chart(plot_overlay(_ovl), width='stretch')
        else:
            if res:
                if st.button('➕ Add current simulation to overlay', use_container_width=True):
                    store_sim_overlay(res, _prop.abbr if _prop else 'Motor',
                                      f'{_mtr or mtr_name} ({res["motor_class"]})')
                    st.rerun()

        st.markdown('<div class="nexus-section">Throat Diameter Sensitivity</div>', unsafe_allow_html=True)
        if res is None:
            st.info('Run a simulation first to enable sensitivity analysis.')
        else:
            _dt_range = st.slider('Throat Δ range (mm)', min_value=0.5, max_value=5.0, value=2.0, step=0.5)
            _n_steps = 5
            _dt_vals = [round(Dt_mm - _dt_range + i * 2 * _dt_range / (_n_steps - 1), 2) for i in range(_n_steps)]
            _sens_results = []
            with st.spinner('Running sensitivity sweep…'):
                for _dv in _dt_vals:
                    if _dv <= 0:
                        _sens_results.append({'max_Pc_MPa': 0, 'avg_Isp': 0, 'total_impulse': 0, 'burn_time': 0})
                        continue
                    try:
                        _at_v = math.pi * (_dv * 1e-3 / 2) ** 2
                        _ae_v = math.pi * (De_mm * 1e-3 / 2) ** 2
                        _grain_v = BATESGrain(ro_mm * 1e-3, ri_mm * 1e-3, L_mm * 1e-3, int(n_seg))
                        _ib_v = InternalBallistics(prop, _grain_v, _at_v, _ae_v, eta_cs, eta_dp)
                        _res_v = _ib_v.run(dt_s=dt_s)
                        _sens_results.append(_res_v)
                    except Exception:
                        _sens_results.append({'max_Pc_MPa': 0, 'avg_Isp': 0, 'total_impulse': 0, 'burn_time': 0})

            _metric_sel = st.selectbox('Metric', ['max_Pc_MPa', 'avg_Isp', 'total_impulse', 'burn_time'],
                                       format_func=lambda k: {'max_Pc_MPa': 'Max Pressure (MPa)',
                                                               'avg_Isp': 'Avg Isp (s)',
                                                               'total_impulse': 'Total Impulse (N·s)',
                                                               'burn_time': 'Burn Time (s)'}[k])
            st.plotly_chart(plot_sensitivity(_sens_results, 'Throat Diameter (mm)', _dt_vals,
                                            _metric_sel, {'max_Pc_MPa': 'Max Pressure (MPa)',
                                                          'avg_Isp': 'Avg Isp (s)',
                                                          'total_impulse': 'Total Impulse (N·s)',
                                                          'burn_time': 'Burn Time (s)'}[_metric_sel]),
                            width='stretch')

    # ── TAB 8 — KNOWLEDGE BASE ───────────────────────────────────────────
    with tab_kb:
        render_knowledge_base()

    # ── TAB 9 — AI EXPERT (built-in, no API key needed) ─────────────────
    with tab_ai:
        render_ai_tab(res, strct, _prop, _grain, _At, _Ae, _ecs, _edp, prop, n_seg, ro_mm, ri_mm, L_mm, At_m2, Ae_m2, eta_cs, eta_dp, wall_mm, mat_key)

    # Footer disclaimer
    render_disclaimer()


# ═══════════════════════════════════════════════════════════════════════════
# BUILT-IN ROCKETRY EXPERT KNOWLEDGE BASE  (no API key required)
# ═══════════════════════════════════════════════════════════════════════════

_KB: list = [
    {
        "tags": ["c star","c*","cstar","characteristic velocity","c-star","what is c"],
        "title": "C* — Characteristic Velocity",
        "answer": """**C\\* (C-star) — Characteristic Velocity** is the single best number for measuring how energetically your propellant burns, independent of the nozzle design.

**Plain English:** Imagine two rockets with identical nozzles. The one whose propellant produces hotter, lighter combustion gases will push more mass out faster — that's a higher C\\*. It tells you "how good is the propellant itself?"

**Formula:**
> C\\* = (Chamber Pressure × Throat Area) / Mass Flow Rate
> C\\* = √(R·Tf / γ) / Γ

**What the numbers mean:**
- KNSB (sugar rocket): ~889 m/s — decent for a hobby propellant
- APCP standard: ~1578 m/s — much more energetic
- Liquid hydrogen/oxygen: ~2300+ m/s — top tier

**C\\* Efficiency (η\\_c\\*)** in the sidebar (default 0.95) accounts for real-world losses: incomplete mixing, heat loss to the casing, and combustion instability. A value of 0.95 means your motor achieves 95% of the theoretical maximum — typical for well-made amateur motors.

**Ref:** Sutton & Biblarz (2016) §3.3"""
    },
    {
        "tags": ["dispersion","eta dp","two phase","two-phase","particle","aluminum","al particle","η_dp","phase flow","what is dispersion"],
        "title": "η_DP — Two-Phase Dispersion Loss",
        "answer": """**η\\_DP — Two-Phase Dispersion Penalty** accounts for the fact that burning aluminum particles (in APCP) don't behave like a perfect gas.

**Plain English:** When aluminum powder burns inside the motor, it creates tiny molten droplets. These droplets are heavier than gas molecules — they can't accelerate as fast through the nozzle, and they carry heat with them that doesn't get converted to thrust. η\\_DP captures how much performance you lose because of this.

**Why it matters:**
- Metal-free propellants (APCP-NoAl, KNSB): set η\\_DP = 1.00 (no penalty — no particles)
- Standard APCP with 12% Al: η\\_DP ≈ 0.97 (lose ~3% of theoretical Isp)
- Higher Al loading or finer particles → lower η\\_DP

**Default value of 0.97** means 97% of theoretical Isp is achieved — 3% lost to particle drag and thermal lag.

**Combined efficiency** = η\\_c\\* × η\\_DP. At defaults (0.95 × 0.97 = 0.9215), you get ~92% of theoretical performance — realistic for a well-built amateur motor.

**Ref:** Kuo & Summerfield (1984) §5; Sutton & Biblarz §13.5"""
    },
    {
        "tags": ["kn","klemmung","klemung","ab/at","burn area","surface area ratio","what is kn","klemung coefficient"],
        "title": "Kn — Klemmung Coefficient",
        "answer": """**Kn (Klemmung Coefficient)** = Burning Surface Area ÷ Throat Area (Ab / At)

**Plain English:** Kn tells you how "pressurized" your motor will be. A bigger burning surface relative to the throat means more gas trying to escape through a smaller hole — higher chamber pressure.

**Why it matters:**
- Higher Kn → Higher chamber pressure → Higher burn rate → Even higher pressure (feedback loop)
- This is why n < 1 is required for stability — if n ≥ 1, the feedback becomes runaway (CATO = catastrophic failure)
- Typical safe operating range: Kn = 150–300 for sugar propellants, up to 400–500 for APCP

**Burn profiles from Kn trend:**
- Kn stays flat → Neutral burn (constant thrust) — ideal
- Kn increases → Progressive burn (thrust increases over time)
- Kn decreases → Regressive burn (thrust decreases over time)

**Rule of thumb:** If your Kn > 600, check your nozzle — the pressure may exceed the propellant's valid range.

**Ref:** Nakka (2023); Sutton & Biblarz §13"""
    },
    {
        "tags": ["isp","specific impulse","impulse","efficiency","seconds","what is isp","specific impulse mean"],
        "title": "Isp — Specific Impulse",
        "answer": """**Isp (Specific Impulse)** is the "miles-per-gallon" of rocket propellants — how much thrust you get per unit of propellant consumed.

**Formula:**
> Isp = Thrust / (Mass Flow Rate × g₀)   [units: seconds]

**Plain English:** If your motor has Isp = 165 s, it means 1 kg of propellant produces 165 N of thrust for 1 second, OR 1650 N for 0.1 seconds. Higher = better efficiency.

**Typical values:**
- KNSB (sugar): ~130–165 s (sea level to vacuum)
- APCP standard: ~210–242 s
- Liquid LOX/Kerosene (Falcon 9): ~282–311 s
- Liquid LOX/LH2 (Space Shuttle Main Engine): ~366–453 s

**Vacuum vs Sea-Level Isp:** Vacuum Isp is always higher because there's no atmosphere pushing back against the exhaust. The difference depends on nozzle expansion ratio.

**Ref:** Sutton & Biblarz §2"""
    },
    {
        "tags": ["bates","grain","segments","segment","core","hollow cylinder","grain geometry","what is bates","grain design"],
        "title": "BATES Grain — What It Is",
        "answer": """**BATES** stands for **Ballistic Test and Evaluation System** — it's the most common grain geometry for amateur and research solid rocket motors.

**Plain English:** A BATES grain is a cylinder of propellant with a hole drilled through the middle. When it burns, the hole gets bigger from the inside out AND the ends recede inward simultaneously.

**The 3 dimensions you set:**
- **Outer radius (ro):** How thick the cylinder is — limited by your motor tube diameter
- **Inner radius (ri):** How big the starting hole is — controls initial Kn and burn rate
- **Length (L):** How long each segment is — more length = more propellant = longer burn

**Multiple segments:** Instead of one long grain, you use 2–4 shorter segments with small gaps between them. This prevents the grain from cracking under thermal stress, and the gaps expose the end faces for burning.

**Rule of thumb:** ri/ro ratio of 0.4–0.6 gives close to neutral burn. Smaller core (lower ri) = progressive burn. Larger core (higher ri) = regressive burn.

**Ref:** Nakka (2023) Grain Design Guide; Sutton & Biblarz §13"""
    },
    {
        "tags": ["saint robert","burn rate","vieille","a coefficient","n exponent","pressure exponent","burn law","what is a","what is n","burn rate law"],
        "title": "Saint-Robert Burn Rate Law (a and n)",
        "answer": """**Saint-Robert Law** (also called Vieille's Law) describes how fast your propellant surface burns at a given chamber pressure:

> **r = a · Pc^n**   (r in mm/s, Pc in MPa)

**What 'a' means:** The base burn rate at 1 MPa pressure. Higher 'a' = faster burning propellant overall. KNSB has a = 8.26, meaning at 1 MPa it burns at 8.26 mm/s.

**What 'n' means (pressure exponent):** How sensitive the burn rate is to pressure changes.
- n = 0.3: A 10% pressure increase → only 3% faster burn rate (stable, good)
- n = 0.9: A 10% pressure increase → 9% faster burn rate (risky, nearly unstable)
- n ≥ 1.0: **CATO territory** — pressure runaway, motor explodes

**CRITICAL STABILITY RULE: n must be less than 1.0**

Typical safe values: n = 0.30–0.40 for sugar propellants, 0.33–0.40 for APCP.

**The chart on the Burn Rate tab** shows exactly how burn rate changes across the valid pressure range for your selected propellant.

**Ref:** Sutton & Biblarz §11.2; Kubota (2007) §3"""
    },
    {
        "tags": ["throat","nozzle","throat diameter","at","throat area","nozzle throat","what is throat","choked flow"],
        "title": "Nozzle Throat — What It Does",
        "answer": """**The nozzle throat** is the narrowest point of the nozzle — this is where the magic happens.

**Plain English:** Hot combustion gases accelerate as they're squeezed through the throat. At the throat, the flow reaches exactly the speed of sound (Mach 1). After the throat, it continues expanding and accelerating to supersonic speeds through the diverging section.

**Why throat size matters enormously:**
- Smaller throat → higher Kn → higher chamber pressure → faster burn rate
- Bigger throat → lower pressure → slower burn, lower thrust
- The throat is the single most powerful tuning knob on your motor

**Throat diameter selection rule of thumb:**
> At = Ab_initial / target_Kn

If you want Kn ≈ 200 and your grain has 50 cm² of burning surface, you need At ≈ 50/200 = 0.25 cm² → throat diameter ≈ 5.6 mm.

**Erosion:** Nozzle throats erode during firing (especially graphite nozzles), which increases throat area, lowers Kn and pressure over time. This is why real burn profiles often show decreasing pressure late in the burn.

**Ref:** Sutton & Biblarz §3.3, §15"""
    },
    {
        "tags": ["exit","expansion ratio","exit diameter","ae","expansion","diverging","epsilon","nozzle exit","what is exit"],
        "title": "Nozzle Exit & Expansion Ratio (ε)",
        "answer": """**Expansion Ratio (ε)** = Exit Area / Throat Area = (De/Dt)²

**Plain English:** After gases go supersonic at the throat, they expand in the diverging section. A larger exit area gives the gases more room to expand, extracting more energy as thrust. But there's an optimal size — too big and you "over-expand" (exhaust pressure drops below ambient, reducing thrust).

**Typical values:**
- Simple amateur motors: ε = 4–8
- High-altitude/vacuum optimized: ε = 20–100+
- Sea-level optimal (maximize thrust at ground level): ε ≈ 7–12 for most propellants

**How to pick:** For HPR flights below 30,000 ft, ε = 6–10 is a good starting point. The simulator uses sea-level Isp, so your expansion ratio is baked into the Cf value.

**The caption below Exit Diameter** shows you the current ε — aim for 4–12 for typical amateur motors.

**Ref:** Sutton & Biblarz §3.4"""
    },
    {
        "tags": ["port throat","port to throat","j ratio","erosive","erosive burning","j<2","port","what is j","erosion burning"],
        "title": "Port-to-Throat Ratio (J) & Erosive Burning",
        "answer": """**Port-to-Throat Ratio J** = Core Port Area / Throat Area = (π·ri²) / At

**Plain English:** The "port" is the hollow channel (hole) running through the grain. If this channel is too narrow relative to the throat, combustion gases rush through it too fast — scrubbing the burning surface and making it burn even faster than the pressure alone predicts. This is **erosive burning**.

**The danger of J < 2.0:**
When J drops below 2, gas velocity in the core is so high it creates local pressure and temperature spikes. The burn rate on the upstream end of the grain increases unpredictably, spiking chamber pressure — potentially to failure.

**What to do if J < 2.0 warning appears:**
1. Increase the core diameter (larger ri)
2. Reduce the throat area (larger throat diameter) — wait, that INCREASES Kn, careful
3. Use fewer/shorter segments to keep the core area large relative to throat

**Rule:** J ≥ 2.0 at all times during the burn. The simulator tracks the minimum J across the entire burn and flags it if it goes below 2.

**Ref:** Nakka (2023); Sutton & Biblarz §13.4"""
    },
    {
        "tags": ["chamber pressure","pc","pressure","mpa","chamber","what is pc","chamber pressure mean"],
        "title": "Chamber Pressure (Pc)",
        "answer": """**Chamber Pressure (Pc)** is the pressure inside the combustion chamber while the motor is firing.

**Plain English:** This is how hard the burning propellant is pushing in every direction — against the casing walls, against the nozzle, and out the throat. Higher pressure generally means more thrust, but also more stress on the motor.

**Typical ranges:**
- KNSB/KNSU sugar motors: 1–10 MPa (145–1450 psi)
- APCP amateur motors: 2–15 MPa (290–2175 psi)
- Professional solid motors: 5–25 MPa

**Why Max Pc matters:**
1. **Structural:** The casing must survive peak pressure. The Structural tab calculates safety factors based on max Pc.
2. **Burn rate validity:** Each propellant has a valid pressure range (P\\_min to P\\_max in the database). Operating outside this range makes the Saint-Robert law inaccurate.
3. **Nozzle erosion:** Higher pressure increases throat erosion rate.

**To lower Pc:** Increase throat size, decrease Kn, or use fewer grain segments.
**To raise Pc:** Decrease throat size, increase Kn.

**Ref:** Sutton & Biblarz §3, §13"""
    },
    {
        "tags": ["total impulse","impulse","newton second","ns","motor class","class","what class","h motor","i motor","motor classification"],
        "title": "Total Impulse & Motor Classification",
        "answer": """**Total Impulse (It)** = area under the thrust-vs-time curve = Thrust × Burn Time (approximately)
Units: Newton-seconds (N·s)

**Plain English:** This is the total "kick" your motor delivers. A motor that burns 100 N for 2 seconds and one that burns 50 N for 4 seconds both have the same total impulse (200 N·s) — same letter class.

**NAR/TRA Motor Classes:**

| Class | Total Impulse (N·s) | Cert. Needed |
|-------|---------------------|--------------|
| F | 40–80 | L1 |
| G | 80–160 | L1 |
| H | 160–320 | L1 |
| I | 320–640 | L1 |
| J | 640–1,280 | L2 |
| K | 1,280–2,560 | L2 |
| L | 2,560–5,120 | L2 |
| M | 5,120–10,240 | L3 |

Each class is exactly double the previous. A "full H" motor = 320 N·s exactly.

**Research/EX motors** (motors you make yourself) require an ATF LEUP (Low Explosives User Permit) in the US.

**Ref:** NAR Motor Classification Standards; NFPA 1127"""
    },
    {
        "tags": ["safety factor","sf","yield","structural","hoop stress","von mises","wall thickness","burst","casing","what is sf","what is safety factor"],
        "title": "Safety Factor & Structural Analysis",
        "answer": """**Safety Factor (SF)** = Material Strength / Actual Stress

**Plain English:** If SF = 4, your casing is 4× stronger than it needs to be to survive the peak pressure. SF = 1 means you're exactly at the breaking point — any variation and it fails.

**Recommended minimum SF values for HPR casings:**
- SF ≥ 4.0 ✅ — Standard design guideline (accounts for material variability, dynamic loading, thermal effects)
- SF 2.0–4.0 ⚠️ — Marginal. Requires engineering analysis and justification
- SF < 2.0 🛑 — **DO NOT FLY.** Casing will likely fail.

**Hoop Stress** is the stress trying to split the cylinder lengthwise (like a soda can bursting). It's always higher than axial stress for a cylinder under internal pressure — this is what usually fails first.

**Von Mises Stress** combines hoop and axial stresses into one number representing the "effective" stress for predicting yielding. It's what's compared against yield strength (Sy).

**What to do if SF is too low:**
1. Increase wall thickness (most effective)
2. Choose a stronger material (4130 steel or CF/epoxy instead of aluminum)
3. Reduce chamber pressure (larger throat)

**Ref:** Shigley's Mechanical Engineering Design §3-14"""
    },
    {
        "tags": ["knsb","knsu","sugar","potassium nitrate","sorbitol","sucrose","kn propellant","candy propellant","what is knsb","what is knsu"],
        "title": "KNSB & KNSU — Sugar Propellants",
        "answer": """**KNSB** (65% Potassium Nitrate / 35% Sorbitol) and **KNSU** (65% KNO₃ / 35% Sucrose) are the classic "candy propellants" — the starting point for most amateur rocketry.

**Why beginners use them:**
- No special chemicals beyond hobby-grade KNO₃ and food sugar
- Melt-cast process (like making fudge) — no chemical curing needed
- Relatively low processing temperatures (95–185°C)
- Well-documented in Richard Nakka's extensive online research

**KNSB vs KNSU:**
- KNSB uses sorbitol (a sugar alcohol): melts at ~95°C, more forgiving to process
- KNSU uses sucrose (table sugar): melts at ~186°C, slightly higher performance but trickier
- Both deliver Isp ~130–165 s (sea level to vacuum) — lower than APCP but accessible

**CRITICAL SAFETY:**
- Process at ≤200°C only. Higher temperatures risk auto-ignition (KNO₃ decomposes at 400°C)
- Never use open flame heating — electric heat only
- Store in cool, dry, static-safe environment
- Work in small batches only

**These are Class 1.3C or 1.4C explosives** — check local regulations before processing.

**Ref:** Nakka (2023) rocketry.burri.to; Sutton & Biblarz"""
    },
    {
        "tags": ["apcp","ammonium perchlorate","htpb","composite propellant","aluminum","standard apcp","what is apcp"],
        "title": "APCP — Ammonium Perchlorate Composite Propellant",
        "answer": """**APCP** (Ammonium Perchlorate Composite Propellant) is the standard propellant for commercial HPR motors (AeroTech, Cesaroni, etc.) and advanced experimental rocketry.

**Composition (typical):**
- 70% NH₄ClO₄ (ammonium perchlorate) — oxidizer, provides oxygen for combustion
- 18% HTPB (hydroxyl-terminated polybutadiene) — rubber binder/fuel
- 12% Aluminum powder — fuel, increases flame temperature dramatically (+800°C vs metal-free)

**Why it's better than sugar:**
- Isp ~210–242 s vs ~130–165 s for KNSB — roughly 50% more efficient
- Much higher flame temperature (3350 K vs 1720 K) = more energy per gram
- Can be formulated for a wide range of burn rates

**Why it's harder:**
- Chemically cured thermosetting system — must cure at 50–70°C for 3–7 DAYS
- NEVER heat above 90°C during processing — fire/explosion risk
- Mixed propellant is sensitive to friction, impact, and static
- Requires ATF LEUP and certified facility to process legally in the US
- NOT for beginners

**The white smoke** from APCP motors is aluminum oxide (Al₂O₃) — the combustion product of aluminum burning.

**Ref:** Sutton & Biblarz §12; Kuo & Summerfield (1984)"""
    },
    {
        "tags": ["neutral","progressive","regressive","burn profile","thrust profile","flat burn","what is neutral","progressive burn"],
        "title": "Burn Profiles — Neutral, Progressive, Regressive",
        "answer": """**Burn Profile** describes how thrust changes over the burn duration, determined by how the burning surface area (and thus Kn) changes with time.

**Neutral Burn** (flat thrust):
- Burning surface area stays roughly constant throughout the burn
- Kn stays flat → constant chamber pressure → constant thrust
- Most desirable for flight stability and predictable performance
- Achieved by balancing core growth (increases Ab) vs. end recession (decreases Ab)
- BATES grains with ri/ro ≈ 0.5 tend to be close to neutral

**Progressive Burn** (rising thrust):
- Burning surface increases over time → Kn rises → pressure and thrust increase
- Gives a slow start and a powerful finish
- Useful for staged motors or when you want max thrust near end of burn
- Risk: late pressure spike must still be within structural limits

**Regressive Burn** (falling thrust):
- Burning surface decreases over time → Kn falls → pressure and thrust decrease
- Common in end-burning grains
- Often seen in larger ri/ro ratios where the grain quickly becomes all end-burning

**For flight stability:** Neutral or slightly progressive burns are preferred — you want consistent thrust during the boost phase."""
    },
    {
        "tags": ["time step","dt","simulation accuracy","timestep","time resolution","what is time step","0.5 ms"],
        "title": "Time Step (Δt) — Simulation Accuracy",
        "answer": """**Time Step (Δt)** is how finely the simulation divides time when calculating the burn progression.

**Plain English:** The simulator steps forward in tiny time increments, recalculating pressure, thrust, and burn rate at each step. Smaller steps = more accurate result, but takes longer to compute.

**The options:**
- **0.25 ms** — Most accurate. Use for final design verification or when you see jagged curves. Can be slow for long burns.
- **0.5 ms** *(default)* — Good balance of accuracy and speed. Fine for most designs.
- **1.0 ms** — Faster, slightly less accurate. OK for initial exploration.
- **2.0 ms** — Quickest. Use only for rough estimates or very long burn motors.

**When to use finer steps:**
- If your thrust curve looks "stepped" or choppy instead of smooth
- Short, fast-burning motors (< 0.5 s burn time)
- When peak pressure values seem inconsistent between runs

**For most HPR motors:** 0.5 ms is perfectly adequate."""
    },
    {
        "tags": ["total impulse","burn time","avg thrust","max thrust","results","output","what does","reading results","understand results"],
        "title": "Reading the Simulation Results",
        "answer": """**How to read the Command Center results panel:**

**Motor Class** — The NAR/TRA letter (A through O+) based on total impulse. Determines what certification you need to fly it.

**Total Impulse (N·s)** — Total "kick" delivered. This is the primary number for motor classification.

**Burn Time (s)** — How long the motor fires from ignition to burnout.

**Max Thrust / Avg Thrust (N)** — Peak and average thrust force. 1 N = 0.225 lbf. To lift a rocket, avg thrust should exceed rocket weight × 5 (5:1 thrust-to-weight ratio minimum).

**Max Pc (MPa)** — Peak chamber pressure. Check this against your structural safety factor. 1 MPa ≈ 145 psi.

**Avg Isp (s)** — Average specific impulse — propellant efficiency.

**Avg r (mm/s)** — Average burn rate of the propellant surface.

**Max Kn** — Peak Klemmung. Should stay below ~500 for sugar propellants.

**Min J (port/throat)** — Should stay above 2.0 to avoid erosive burning.

**Struct. SF** — Structural safety factor. Green ✅ = SF ≥ 4, Orange ⚠️ = SF 2–4, Red 🛑 = SF < 2 (unsafe)."""
    },
    {
        "tags": ["decomp","decomposition","temperature","safe temperature","processing temperature","heat","critical temp","auto ignition"],
        "title": "Decomposition Temperatures & Thermal Safety",
        "answer": """**Every propellant has two critical temperature thresholds:**

**Decomp Onset Temperature** — The temperature where the propellant begins to decompose (release gas and energy) without an ignition source. You feel it as faint heating or gas release.

**Critical Temperature** — The point where thermal runaway begins. At or above this temperature, the decomposition becomes self-sustaining and accelerates to auto-ignition.

**The propellants ranked by how forgiving they are:**
1. **KNSB** — Onset 339°C, Critical 400°C. Most forgiving. Widest safety margin.
2. **KNSU** — Onset 300°C, Critical 380°C. Slightly less margin, higher processing temp needed.
3. **APCP** — Onset 240°C, Critical 300°C. Never exceed 90°C during processing — far below onset, but exothermic cure reaction adds heat.
4. **GAP-AP** — Onset 220°C, Critical 270°C. ⚠️ Energetic azide groups. Specialist facility only.

**Golden rules:**
- Always use electric heating (no open flames)
- Never leave propellant unattended while heating
- Work in small batches (< 500g for beginners)
- Keep a large water source nearby
- No metal tools that could create sparks

**Ref:** Kubota (2007) §2; NFPA 1127"""
    },
    {
        "tags": ["eng file","eng","openrocket","rasp","thrust curve","export","download","file format","eng format"],
        "title": ".ENG File — Exporting to OpenRocket",
        "answer": """**The .ENG file** is the standard RASP format for sharing thrust curves between simulation software.

**What it is:** A text file containing a time-vs-thrust table that flight simulation programs (OpenRocket, RASAero II, ThrustCurve.org) read to predict your rocket's flight trajectory.

**How to use it:**
1. Run your simulation in NEXUS
2. Go to the **Data Export** tab
3. Click **Download .ENG File**
4. In **OpenRocket**: File → Open → select your .eng file, or put it in the `thrustcurves` folder in your OpenRocket directory
5. Your motor will appear in the motor database under "User Defined"

**The file contains:**
- Motor designation, diameter, length, propellant mass, total mass
- Up to 200 time-thrust data points (downsampled if needed)
- Comments with all simulation parameters

⚠️ **Important:** This is a theoretical simulation. Real motor performance will differ. Always static-fire test before flight and record your actual thrust curve.

**Ref:** thrustcurve.org/info/raspformat.html"""
    },
    {
        "tags": ["sigma","hoop","axial","stress","lame","thin wall","thick wall","ri","ro","wall","pressure vessel"],
        "title": "Hoop Stress, Axial Stress & Lamé Equations",
        "answer": """**Hoop Stress (σ\\_h)** — The stress trying to burst the cylinder open along its length (like a hotdog casing splitting). This is ALWAYS the largest stress in a pressurized cylinder.

**Axial Stress (σ\\_a)** — The stress trying to push the end caps off the cylinder. Always half of hoop stress for thin-wall cylinders.

**Thin-wall vs Thick-wall:**
- If wall thickness t < 10% of inner radius → **thin-wall equations** (simpler, slightly optimistic)
- If t ≥ 10% of inner radius → **Lamé thick-wall equations** (more conservative, more accurate for stubby motor casings)

**Simple thin-wall formula:**
> σ\\_hoop = Pc × ri / t

So if chamber pressure is 5 MPa, inner radius is 38 mm, and wall is 3 mm:
σ\\_hoop = 5 × 38/3 = 63.3 MPa

Compare this to your material's yield strength (Sy):
- 6061-T6 aluminum: Sy = 276 MPa → SF = 276/63.3 = **4.36 ✅**

**Ref:** Shigley's §3-14; Roark's §13"""
    },
    {
        "tags": ["gamma","specific heat ratio","cp cv","gamma ratio","what is gamma","specific heat"],
        "title": "γ (Gamma) — Specific Heat Ratio",
        "answer": """**γ (gamma)** = Cp / Cv = ratio of specific heat at constant pressure to specific heat at constant volume

**Plain English:** γ describes how the combustion gas behaves when it expands. A higher γ means the gas releases more energy as it expands — more thrust per unit of expansion.

**Why it appears in rocket equations:**
- In the Vandenkerckhove function Γ (capital gamma) which appears in choked flow calculations
- In the isentropic expansion equations for nozzle design
- In the optimal expansion ratio calculation

**Typical values:**
- Diatomic gases (N₂, O₂): γ = 1.4
- Combustion products of solid propellants: γ = 1.13–1.25 (lower because larger, more complex molecules)
- KNSB products: γ = 1.131 (mostly K₂CO₃, CO₂, CO, N₂, H₂O)
- APCP products: γ = 1.20

**For simulation purposes:** γ is built into the propellant database and used automatically. You don't need to set it — it's listed in the propellant info panel."""
    },
    {
        "tags": ["of ratio","oxidizer fuel","oxygen balance","equivalence ratio","phi","stoichiometric","rich lean","fuel rich"],
        "title": "O/F Ratio, Oxygen Balance & Equivalence Ratio",
        "answer": """**O/F Ratio** (Oxidizer-to-Fuel) = mass of oxidizer / mass of fuel in the propellant mix.

**Plain English:** How much oxidizer do you have for every gram of fuel? Too much oxidizer = oxygen-rich (wasteful, cooler). Too much fuel = fuel-rich (unburned carbon, also wasteful). Stoichiometric = perfect balance for complete combustion.

**Oxygen Balance (OB%)** tells you if the propellant has excess or deficit oxygen:
- OB = 0%: Exactly the right amount of oxygen for complete combustion
- OB > 0%: Oxidizer-rich (excess O₂) → less efficient, but cleaner exhaust
- OB < 0%: Fuel-rich (carbon-rich exhaust → black smoke)

**Equivalence Ratio (φ)**:
- φ = 1.0: Stoichiometric (perfect balance)
- φ > 1.0: Fuel-rich
- φ < 1.0: Oxidizer-rich

**Why HPR propellants run slightly fuel-rich (φ ≈ 0.85–0.95):**
Running slightly rich reduces the mean molecular weight of combustion products, which actually increases Isp even though combustion is incomplete. There's an optimal point that's slightly fuel-rich for most propellants.

**Ref:** Sutton & Biblarz §5; Kubota (2007) §2"""
    },
    {
        "tags": ["how do i","where do i start","beginner","new","first time","getting started","help","confused","dont understand","don't understand"],
        "title": "Getting Started — How to Use NEXUS",
        "answer": """**Welcome! Here's how to run your first simulation:**

**Step 1 — Pick a propellant (sidebar, Section 1)**
Start with **KNSB** — it's the most forgiving sugar propellant. The caption below shows key properties.

**Step 2 — Set your grain geometry (sidebar, Section 2)**
- Segments: Start with 2
- Outer radius: Match your motor tube (e.g., 38 mm for a 76mm tube)
- Inner radius: Start at ~40% of outer radius (e.g., 15 mm for 38 mm outer)
- Length: 100–150 mm per segment is typical

**Step 3 — Set your nozzle (sidebar, Section 3)**
- Throat diameter: Start with 10–12 mm — you'll tune this
- Exit diameter: 2× the throat diameter (gives expansion ratio ~4)

**Step 4 — Leave efficiency sliders at defaults (Section 4)**
η\\_c\\* = 0.95, η\\_DP = 0.97 are realistic starting values.

**Step 5 — Pick a casing material and wall thickness (Section 5)**
6061-T6 aluminum at 3 mm wall is a common starting point.

**Step 6 — Click ⚡ EXECUTE SIMULATION**
Read the results on the Command Center tab. Check:
- Is SF (safety factor) ≥ 4? If not, increase wall thickness.
- Is Min J ≥ 2.0? If not, increase inner radius.
- Is the motor class what you expected?

**Then iterate** — adjust throat size to hit your target Kn/pressure, adjust grain dimensions for desired burn time and total impulse."""
    },
]


# ── Propellant startup parameter recommendations ────────────────────────────
_PROP_PARAMS = {
    'KNSB': {
        'name': 'KNSB (Potassium Nitrate / Sorbitol)',
        'segments': 2, 'ro_mm': 32, 'ri_mm': 13, 'L_mm': 100,
        'Dt_mm': 10, 'De_mm': 20, 'wall_mm': 3.0, 'material': '6061-T6',
        'eta_cs': 0.92, 'eta_dp': 1.00,
        'target_kn': '150–250', 'target_pc': '2–7 MPa',
        'notes': (
            "KNSB is the best starting propellant — melt-cast at 95–115 °C, no curing needed.\n\n"
            "**Why these numbers:**\n"
            "- ri/ro ≈ 0.41 gives a nearly neutral burn profile\n"
            "- Dt = 10 mm gives Kn ≈ 200 at initial geometry — safe and stable\n"
            "- 3 mm Al 6061-T6 wall gives SF ≈ 3.5–4.5 at typical KNSB pressures\n"
            "- η_DP = 1.00 because KNSB has no metal particles\n\n"
            "**Expected results:** ~H or I class motor, 2–5 s burn, ~130 N average thrust."
        ),
    },
    'KNSU': {
        'name': 'KNSU (Potassium Nitrate / Sucrose)',
        'segments': 2, 'ro_mm': 32, 'ri_mm': 13, 'L_mm': 100,
        'Dt_mm': 10, 'De_mm': 20, 'wall_mm': 3.0, 'material': '6061-T6',
        'eta_cs': 0.92, 'eta_dp': 1.00,
        'target_kn': '150–250', 'target_pc': '2–7 MPa',
        'notes': (
            "KNSU behaves very similarly to KNSB but needs a higher processing temperature (160–185 °C). "
            "Use identical starting parameters to KNSB.\n\n"
            "**Key difference:** Lower decomposition onset (300 °C vs 339 °C for KNSB) — be extra "
            "careful with heat control during casting."
        ),
    },
    'APCP_STD': {
        'name': 'APCP Standard (AP/HTPB/Al 70/18/12)',
        'segments': 2, 'ro_mm': 38, 'ri_mm': 16, 'L_mm': 120,
        'Dt_mm': 11, 'De_mm': 22, 'wall_mm': 3.5, 'material': '4130_Steel',
        'eta_cs': 0.95, 'eta_dp': 0.97,
        'target_kn': '200–350', 'target_pc': '4–12 MPa',
        'notes': (
            "APCP delivers ~50% higher Isp than sugar propellants but requires chemical curing "
            "(3–7 days at 60 °C) and a certified facility.\n\n"
            "**Why these numbers:**\n"
            "- Larger grain (38 mm outer radius) because APCP is used in larger amateur motors\n"
            "- 4130 Steel casing recommended — higher pressures need stronger material\n"
            "- η_DP = 0.97 accounts for the 12% aluminum particle loss\n"
            "- η_c\\* = 0.95 is typical for well-mixed HTPB-based propellant\n\n"
            "**Expected results:** ~J or K class motor, 2–4 s burn, 300–600 N average thrust."
        ),
    },
    'APCP_NOAL': {
        'name': 'APCP Metal-Free (AP/HTPB 80/20)',
        'segments': 2, 'ro_mm': 38, 'ri_mm': 16, 'L_mm': 120,
        'Dt_mm': 11, 'De_mm': 22, 'wall_mm': 3.5, 'material': '6061-T6',
        'eta_cs': 0.95, 'eta_dp': 1.00,
        'target_kn': '200–350', 'target_pc': '4–12 MPa',
        'notes': (
            "APCP-NoAl (metal-free) has a cleaner, smoke-reduced exhaust — no aluminum oxide smoke. "
            "Isp is slightly lower than aluminized APCP but still much better than sugar.\n\n"
            "**Why η_DP = 1.00:** No aluminum particles means no two-phase flow loss.\n"
            "**Why 6061-T6 instead of steel:** Lower chamber pressures and no aluminum "
            "abrasion on the nozzle allows lighter aluminum casings.\n\n"
            "**Good use cases:** Drone/UAV propulsion where smoke signatures matter; "
            "research motors where clean exhaust is needed for diagnostics."
        ),
    },
    'GAP_AP': {
        'name': 'GAP-AP Energetic Composite',
        'segments': 2, 'ro_mm': 38, 'ri_mm': 15, 'L_mm': 110,
        'Dt_mm': 10, 'De_mm': 22, 'wall_mm': 4.0, 'material': '4130_Steel',
        'eta_cs': 0.94, 'eta_dp': 0.98,
        'target_kn': '200–400', 'target_pc': '5–15 MPa',
        'notes': (
            "⚠️ **GAP-AP is NOT for beginners.** This is a specialist energetic propellant "
            "with a critical temperature of only 270 °C (much lower than HTPB systems).\n\n"
            "**Requires:** Certified facility, ATF LEUP, experienced mentorship.\n\n"
            "**Why it's used:** Highest burn rate among the options — used when compact "
            "high-thrust motors are needed. The azide (-N₃) groups in GAP make it "
            "self-oxidizing, allowing extreme performance in small volumes.\n\n"
            "**Start with KNSB or APCP-STD first.** Only move to GAP-AP after extensive "
            "experience with less hazardous systems."
        ),
    },
}

# ── Intent classifier ────────────────────────────────────────────────────────
_INTENTS = {
    'recommend_params': [
        'recommend', 'suggest', 'good formula', 'what formula', 'what parameter',
        'what value', 'get started', 'starting point', 'where to start', 'input',
        'what should i', 'settings for', 'setup for', 'configure', 'values for',
        'parameters for', 'how to set up', 'what to enter', 'good setting',
        'initial', 'defaults for', 'starting config',
    ],
    'troubleshoot': [
        'why is', 'too high', 'too low', 'wrong', 'error', 'problem', 'issue',
        'not working', 'fix', 'crashing', 'fail', 'bad', 'weird', 'unexpected',
        'increase', 'decrease', 'improve', 'lower my', 'raise my', 'reduce',
    ],
    'compare': [
        'difference between', 'vs', 'versus', 'compare', 'better', 'which is',
        'should i use', 'pick between',
    ],
    'explain': [
        'what is', 'what are', 'explain', 'define', 'tell me about', 'describe',
        'how does', 'why does', 'what does', 'mean', 'stands for',
    ],
}

_PROP_ALIASES = {
    'knsb': 'KNSB', 'kn sb': 'KNSB', 'potassium nitrate sorbitol': 'KNSB', 'sugar rocket': 'KNSB',
    'knsu': 'KNSU', 'kn su': 'KNSU', 'potassium nitrate sucrose': 'KNSB',
    'apcp std': 'APCP_STD', 'apcp standard': 'APCP_STD', 'apcp-std': 'APCP_STD',
    'apcp': 'APCP_STD', 'standard apcp': 'APCP_STD', 'aluminized apcp': 'APCP_STD',
    'apcp noal': 'APCP_NOAL', 'apcp-noal': 'APCP_NOAL', 'metal free': 'APCP_NOAL',
    'metal-free': 'APCP_NOAL', 'no aluminum': 'APCP_NOAL', 'apcp no al': 'APCP_NOAL',
    'gap': 'GAP_AP', 'gap ap': 'GAP_AP', 'gap-ap': 'GAP_AP', 'energetic': 'GAP_AP',
}


def _detect_intent(q: str) -> str:
    for intent, phrases in _INTENTS.items():
        if any(p in q for p in phrases):
            return intent
    return 'explain'


def _detect_propellant(q: str) -> str | None:
    for alias, key in _PROP_ALIASES.items():
        if alias in q:
            return key
    return None


def _recommend_answer(prop_key: str | None, active_prop_abbr: str = '') -> dict:
    """Generate a parameter recommendation answer for a given propellant."""
    # If no propellant detected in query, use the active one
    if prop_key is None:
        reverse = {'KNSB': 'KNSB', 'KNSU': 'KNSU', 'APCP-STD': 'APCP_STD',
                   'APCP-NoAl': 'APCP_NOAL', 'GAP-AP': 'GAP_AP'}
        prop_key = reverse.get(active_prop_abbr, 'KNSB')

    p = _PROP_PARAMS.get(prop_key)
    if not p:
        p = _PROP_PARAMS['KNSB']

    answer = f"""Here are the **recommended starting parameters** for **{p['name']}**:

---

**Sidebar Section 2 — BATES Grain:**
| Parameter | Value | Why |
|-----------|-------|-----|
| Segments | {p['segments']} | Good balance of total propellant vs. thermal cracking risk |
| Outer radius (ro) | **{p['ro_mm']} mm** | Fits standard motor tube diameter |
| Inner radius (ri) | **{p['ri_mm']} mm** | ri/ro ≈ {p['ri_mm']/p['ro_mm']:.2f} → near-neutral burn profile |
| Segment length (L) | **{p['L_mm']} mm** | L/ri ≈ {p['L_mm']/p['ri_mm']:.1f} → below erosive burning threshold |

**Sidebar Section 3 — Nozzle:**
| Parameter | Value | Why |
|-----------|-------|-----|
| Throat diameter | **{p['Dt_mm']} mm** | Gives Kn ≈ {p['target_kn']} (stable range) |
| Exit diameter | **{p['De_mm']} mm** | Expansion ratio ε = {(p['De_mm']/p['Dt_mm'])**2:.1f} → good sea-level Isp |

**Sidebar Section 4 — Efficiency:**
| Parameter | Value |
|-----------|-------|
| η_c\\* | {p['eta_cs']:.2f} |
| η_DP | {p['eta_dp']:.2f} |

**Sidebar Section 5 — Structural:**
| Parameter | Value |
|-----------|-------|
| Casing material | {p['material']} |
| Wall thickness | {p['wall_mm']} mm |

**Target operating range:** Kn = {p['target_kn']} | Pc = {p['target_pc']}

---

{p['notes']}

---
**Next step:** Enter these values in the sidebar and click **⚡ EXECUTE SIMULATION**. Then check the Command Center tab — if Safety Factor < 4, increase wall thickness. If Min J < 2, increase inner radius (ri)."""

    return {"title": f"Recommended Starting Parameters — {p['name']}", "answer": answer}


def _troubleshoot_answer(q: str, res: dict | None, strct: dict | None) -> dict | None:
    """Generate troubleshooting answer from live sim data if available."""
    if res is None:
        return None  # fall through to KB search

    lines = []
    prop_name = ''

    if 'kn' in q or 'klemmung' in q or 'pressure' in q:
        kn = res['max_Kn']
        pc = res['max_Pc_MPa']
        if 'high' in q or kn > 400:
            lines.append(f"**Your Max Kn = {kn:.1f}** (target: 150–350 for sugar, 200–400 for APCP)\n\n")
            lines.append("**To lower Kn and pressure:**\n")
            lines.append("1. **Increase throat diameter** — even +1 mm makes a big difference\n")
            lines.append("2. **Reduce number of segments** — fewer segments = less burning surface\n")
            lines.append("3. **Reduce segment length** — shorter grains burn less area simultaneously\n")
            lines.append("4. **Increase inner radius (ri)** — larger core = less solid propellant per segment\n\n")
            lines.append(f"Current Max Pc = **{pc:.2f} MPa**. Safe upper limit for your propellant: check the P_max in the propellant caption.")
        elif 'low' in q or kn < 100:
            lines.append(f"**Your Max Kn = {kn:.1f}** — this is low, which means low chamber pressure and low thrust.\n\n")
            lines.append("**To increase Kn:**\n")
            lines.append("1. **Decrease throat diameter** — smaller throat → higher Kn\n")
            lines.append("2. **Add more segments** or increase segment length\n")
            lines.append("3. **Decrease inner radius (ri)** — smaller core → more solid propellant burning area\n")
        else:
            lines.append(f"**Your Max Kn = {kn:.1f}** | Max Pc = **{pc:.2f} MPa** — these look reasonable.\n\n")
            lines.append("Kn 150–350 is the typical target for stable operation. Your burn is in the expected range.")

    elif 'sf' in q or 'safety' in q or 'structural' in q or 'wall' in q:
        if strct:
            sf = strct['SF_yield']
            vm = strct['vm_MPa']
            t4 = strct['t_req_SF4_mm']
            lines.append(f"**Your Safety Factor (yield) = {sf:.2f}** | Von Mises stress = {vm:.1f} MPa\n\n")
            if sf < 2:
                lines.append(f"🛑 **CRITICAL — DO NOT FLY.** SF < 2 means the casing will likely fail at peak pressure.\n\n")
                lines.append(f"**Fix:** Increase wall thickness to at least **{t4:.2f} mm** (for SF = 4.0), or switch to a stronger material like 4130 Steel or Carbon Fiber/Epoxy.")
            elif sf < 4:
                lines.append(f"⚠️ **Marginal** — SF between 2 and 4. Minimum for flight is SF ≥ 4 per HPR guidelines.\n\n")
                lines.append(f"**Recommended wall thickness for SF = 4:** **{t4:.2f} mm**\n")
                lines.append(f"Currently using: {strct.get('t_ratio', 0)*100:.1f}% of inner radius. Increase wall thickness.")
            else:
                lines.append(f"✅ **Structural margin is good.** SF = {sf:.2f} ≥ 4.0 — casing is safe at peak pressure.")

    elif 'isp' in q or 'thrust' in q or 'impulse' in q or 'class' in q:
        lines.append(f"**Your motor:** Class **{res['motor_class']}** | It = {res['total_impulse']:.1f} N·s | Fmax = {res['max_thrust']:.1f} N | Isp = {res['avg_Isp']:.1f} s | tb = {res['burn_time']:.3f} s\n\n")
        if 'increase' in q or 'improve' in q or 'higher' in q:
            lines.append("**To increase total impulse (larger motor class):**\n")
            lines.append("1. Add more grain segments or increase segment length\n")
            lines.append("2. Use a higher-Isp propellant (APCP gives ~50% more Isp than KNSB)\n")
            lines.append("3. Increase outer radius (more propellant mass per segment)\n\n")
            lines.append("**To increase average thrust (same burn time, more force):**\n")
            lines.append("1. Increase Kn (decrease throat diameter)\n")
            lines.append("2. Switch to faster-burning propellant (higher 'a' coefficient)\n")

    if lines:
        return {"title": "Troubleshooting — Your Simulation", "answer": ''.join(lines)}
    return None


def _kb_search(query: str, active_prop_abbr: str = '', res: dict | None = None, strct: dict | None = None) -> dict:
    """Smart multi-intent search over the knowledge base."""
    q = query.lower().strip()

    # 1. Detect intent and propellant
    intent = _detect_intent(q)
    prop_key = _detect_propellant(q)

    # 2. Parameter recommendation intent → answer directly
    if intent == 'recommend_params':
        return _recommend_answer(prop_key, active_prop_abbr)

    # 3. Troubleshooting with live sim data
    if intent == 'troubleshoot' and res is not None:
        ans = _troubleshoot_answer(q, res, strct)
        if ans:
            return ans

    # 4. KB search with smarter scoring
    best, best_score = None, 0
    words = [w for w in q.split() if len(w) > 2]
    for entry in _KB:
        score = 0
        # Tag match (phrase-in-query)
        for tag in entry["tags"]:
            if tag in q:
                score += len(tag.split())  # longer tag phrases score higher
        # Word match in title
        title_lower = entry["title"].lower()
        for word in words:
            if word in title_lower:
                score += 3
        # Word match in answer snippet (first 200 chars)
        answer_snippet = entry["answer"][:200].lower()
        for word in words:
            if word in answer_snippet:
                score += 1
        if score > best_score:
            best, best_score = entry, score

    if best and best_score >= 2:
        return best

    # 5. Fallback with helpful menu
    return {
        "title": "Ask me anything about rocketry!",
        "answer": f"""I didn't catch exactly what you meant by *"{query}"* — here are things I know well:

**⚡ Quick starts — just ask:**
- *"Recommend parameters for KNSB"* — I'll give you exact sidebar values to enter
- *"Recommend parameters for APCP-NoAl"* — propellant-specific starting configs
- *"Why is my Kn too high?"* — I'll analyze your live simulation
- *"How do I increase thrust?"* — troubleshooting with your current numbers

**📖 Concepts I can explain:**
`c-star` · `Isp` · `Kn / klemmung` · `burn rate` · `a and n coefficients` · `BATES grain` · `KNSB` · `APCP` · `dispersion (η_DP)` · `port-to-throat ratio` · `safety factor` · `hoop stress` · `expansion ratio` · `motor class` · `total impulse` · `neutral burn` · `progressive burn` · `decomposition temperature` · `.ENG file` · `gamma` · `O/F ratio`

**🔧 Troubleshooting — just describe the problem:**
*"My safety factor is too low"* · *"Kn keeps spiking"* · *"burn time too short"* · *"pressure too high"*
"""
    }


def render_ai_tab(res, strct, _prop, _grain, _At, _Ae, _ecs, _edp, prop, n_seg, ro_mm, ri_mm, L_mm, At_m2, Ae_m2, eta_cs, eta_dp, wall_mm, mat_key) -> None:
    active_prop = (_prop or prop)
    active_prop_abbr = active_prop.abbr if active_prop else ''

    st.markdown('<div class="nexus-section">🤖 NEXUS Expert — Ask Anything</div>', unsafe_allow_html=True)
    st.markdown(f"""
<div class="nexus-card" style="padding:0.9rem 1.2rem; margin-bottom:1rem; border-color:#1a3a60;">
<div style="font-family:Share Tech Mono,monospace; font-size:0.8rem; color:#a0aec0; line-height:1.7;">
  💬 <strong style="color:#00d4ff;">Ask in plain English</strong> — no API key, no internet needed. I know your simulation live.<br>
  Try: <em>"Recommend parameters for APCP-NoAl"</em> · <em>"Why is my Kn too high?"</em> · <em>"What is c-star?"</em> · <em>"How do I improve my safety factor?"</em>
</div>
</div>
""", unsafe_allow_html=True)

    # Live sim context banner
    if res:
        sf_col = "🟢" if strct and strct['SF_yield'] >= 4 else "🟡" if strct and strct['SF_yield'] >= 2 else "🔴"
        st.markdown(f"""
<div style="background:rgba(0,212,255,0.06); border:1px solid #1a3a60; border-radius:8px; padding:0.7rem 1.1rem; margin-bottom:1rem; font-family:Share Tech Mono,monospace; font-size:0.78rem; color:#a0d8ef;">
📊 <strong>Active sim:</strong> {active_prop_abbr} · Class <strong style="color:#00d4ff">{res['motor_class']}</strong> · It = {res['total_impulse']:.1f} N·s · Pc_max = {res['max_Pc_MPa']:.2f} MPa · Isp = {res['avg_Isp']:.1f} s · Kn_max = {res['max_Kn']:.0f} · {sf_col} SF = {strct['SF_yield']:.2f if strct else '—'}
</div>
""", unsafe_allow_html=True)

    # Quick question buttons — two rows, propellant-aware
    st.markdown('<div style="font-family:Share Tech Mono,monospace; font-size:0.72rem; color:#4a7fa5; margin-bottom:0.5rem; letter-spacing:0.05em;">QUICK QUESTIONS:</div>', unsafe_allow_html=True)
    suggestions = [
        f"Recommend parameters for {active_prop_abbr}",
        "What is c-star?",
        "What is Kn?",
        "Why is my safety factor low?",
        "What is dispersion (η_DP)?",
        "Explain burn rate law",
        "What is BATES grain?",
        "How do I increase total impulse?",
        "How do I read results?",
    ]
    q_cols = st.columns(3)
    for i, sug in enumerate(suggestions):
        if q_cols[i % 3].button(sug, key=f'sug_{i}'):
            st.session_state.chat_hist.append({'role': 'user', 'content': sug})
            result = _kb_search(sug, active_prop_abbr, res, strct)
            answer = f"### {result['title']}\n\n{result['answer']}"
            answer += _inject_sim_context(sug, res, strct, active_prop, _grain, _At or At_m2, _ecs or eta_cs, _edp or eta_dp) if res else ''
            st.session_state.chat_hist.append({'role': 'assistant', 'content': answer})
            st.rerun()

    # Chat history
    for msg in st.session_state.chat_hist:
        with st.chat_message(msg['role']):
            st.markdown(msg['content'])

    # Chat input
    user_input = st.chat_input('Ask about your simulation, parameters, rocketry concepts, or troubleshooting…')
    if user_input:
        st.session_state.chat_hist.append({'role': 'user', 'content': user_input})
        with st.chat_message('user'):
            st.markdown(user_input)

        result = _kb_search(user_input, active_prop_abbr, res, strct)
        answer = f"### {result['title']}\n\n{result['answer']}"
        if res:
            answer += _inject_sim_context(user_input, res, strct, active_prop, _grain, _At or At_m2, _ecs or eta_cs, _edp or eta_dp)

        with st.chat_message('assistant'):
            st.markdown(answer)
        st.session_state.chat_hist.append({'role': 'assistant', 'content': answer})

    if st.session_state.chat_hist:
        if st.button('Clear chat', key='clear_chat'):
            st.session_state.chat_hist = []
            st.rerun()


def _inject_sim_context(query: str, res: dict, strct: dict, prop, grain, At_m2: float, eta_cs: float, eta_dp: float) -> str:
    """Append live simulation numbers to an answer when relevant."""
    if res is None:
        return ''
    q = query.lower()
    lines = []
    if any(w in q for w in ['kn','klemmung','pressure','high','low','why','my','current']):
        lines.append(f"\n\n---\n**📊 Your current sim:** Max Kn = **{res['max_Kn']:.1f}** | Avg Pc = **{res['avg_Pc_MPa']:.2f} MPa** | Max Pc = **{res['max_Pc_MPa']:.2f} MPa**")
    if any(w in q for w in ['safety','sf','structural','wall','casing','burst']):
        if strct:
            lines.append(f"\n\n---\n**📊 Your current sim:** SF = **{strct['SF_yield']:.2f}** | σ_VM = **{strct['vm_MPa']:.1f} MPa** | Wall needed for SF=4: **{strct['t_req_SF4_mm']:.2f} mm**")
    if any(w in q for w in ['isp','thrust','impulse','class','burn time','result','how am i doing']):
        lines.append(f"\n\n---\n**📊 Your current sim:** Class **{res['motor_class']}** | It = **{res['total_impulse']:.1f} N·s** | Fmax = **{res['max_thrust']:.1f} N** | Isp = **{res['avg_Isp']:.1f} s** | tb = **{res['burn_time']:.3f} s**")
    return ''.join(lines)




# ═══════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    main()
