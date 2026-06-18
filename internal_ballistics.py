#!/usr/bin/env python3
"""
================================================================================
  SOLID PROPELLANT INTERNAL BALLISTICS SIMULATION TOOL
  Version 1.0 — For Educational & Theoretical Use Only
================================================================================

  A comprehensive simulation framework for solid rocket motor static-fire
  analysis using established thermodynamic and ballistic literature data.

  KEY PHYSICS IMPLEMENTED
  -----------------------
  * Saint-Robert (Vieille) burn rate law:  r = a · Pc^n
  * Quasi-steady-state chamber pressure:   Pc = (ρ · a · c* · Kn)^(1/(1-n))
  * Thrust:                                F  = Cf · Pc · At
  * Specific Impulse:                      Isp = F / (ṁ · g₀)
  * Total Impulse:                         It = ∫F dt

  GRAIN GEOMETRIES
  ----------------
  * BATES  — hollow cylinder, burns inner core + both annular ends
  * End-Burning — cigarette-style, constant cross-section, neutral burn

  REFERENCES
  ----------
  [1] Sutton, G.P. & Biblarz, O. (2016). Rocket Propulsion Elements, 9th ed.
      John Wiley & Sons, Hoboken, NJ.
  [2] Kuo, K.K. & Summerfield, M. (Eds.) (1984). Fundamentals of Solid
      Propellant Combustion. AIAA Progress in Astronautics, Vol. 90.
  [3] Kubota, N. (2007). Propellants and Explosives: Thermochemical Aspects
      of Combustion, 2nd ed. Wiley-VCH, Weinheim.
  [4] Nakka, R.A. (2023). Solid Propellant Rocket Motor Design and Testing.
      [Online] https://www.nakka-rocketry.net
  [5] Brown, R.S. et al. (1981). Solid Propellant Combustion at High
      Pressures. AIAA Journal 19(9).
  [6] NFPA 1125 (2019). Code for the Manufacture of Model Rocket and High
      Power Rocket Motors. NFPA, Quincy, MA.

================================================================================
  ⚠  SAFETY DISCLAIMER — ENERGETIC MATERIALS
================================================================================

  This software is provided for EDUCATIONAL, THEORETICAL, and ACADEMIC
  MODELING purposes ONLY.

  The thermal processing and handling of energetic materials — including
  propellant mixing, casting, and curing — are INHERENTLY HAZARDOUS and:

    • Must be performed ONLY by properly trained and licensed personnel
    • Require compliance with all applicable federal, state/local laws
    • Require appropriate licensing (ATF Low Explosives Permit or equivalent
      national authority authorization)
    • Must follow established industry codes (NFPA 1125, TRA, NAR)
    • Require appropriate PPE and explosion-rated processing facilities
    • Must NEVER be performed in residential or uncontrolled environments

  The decomposition temperatures listed in this tool are CRITICAL SAFETY
  LIMITS. Exceeding them during processing can cause spontaneous ignition,
  deflagration, or detonation resulting in fire, explosion, serious injury,
  or death.

  The authors and distributors of this software assume NO LIABILITY for
  any use, misuse, or application of the information contained herein.
================================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from dataclasses import dataclass, field
from typing import Tuple, Dict, Optional
import textwrap
import warnings
import sys

# ─────────────────────────────────────────────────────────────
# PHYSICAL CONSTANTS
# ─────────────────────────────────────────────────────────────
G0        = 9.80665        # Standard gravity (m/s²) — NIST
R_UNIV    = 8.31446        # Universal gas constant (J/mol·K)
PA_TO_MPA = 1e-6           # Pa → MPa
MPA_TO_PA = 1e6            # MPa → Pa
ATM_PA    = 101325.0       # 1 atm in Pa


# ═══════════════════════════════════════════════════════════════
#  PROPELLANT DATABASE
# ═══════════════════════════════════════════════════════════════

@dataclass
class PropellantData:
    """
    Consolidated propellant thermophysical and ballistic properties.
    All values from peer-reviewed literature (see references above).
    """
    # Identification
    name:          str
    abbreviation:  str
    composition:   Dict[str, float]   # {ingredient: mass fraction}

    # Physical
    density:       float    # Bulk propellant density (kg/m³)

    # Thermochemical (CEA/ProPEP at ~7 MPa unless noted)
    T_flame:       float    # Adiabatic flame temperature (K)
    gamma:         float    # Effective ratio of specific heats
    MW_products:   float    # Mean MW of combustion products (g/mol)
    c_star:        float    # Characteristic velocity c* (m/s)
    Isp_vac:       float    # Theoretical vacuum Isp, optimum expansion (s)
    Isp_sl:        float    # Theoretical sea-level Isp, optimum expansion (s)

    # Saint-Robert burn rate law:  r [mm/s] = a · Pc[MPa]^n
    burn_a:        float    # Coefficient a  (mm/s · MPa⁻ⁿ)
    burn_n:        float    # Pressure exponent n (dimensionless)
    P_range_MPa:   Tuple[float, float]  # Valid pressure range (MPa)

    # Thermal processing (curing / casting)
    cure_T_C:      Tuple[float, float]  # Cure temperature range °C [min, max]
    cure_t_hr:     Tuple[float, float]  # Cure duration range, hours [min, max]
    decomp_onset:  float    # DSC decomposition onset temperature (°C)
    decomp_crit:   float    # CRITICAL thermal runaway threshold — NEVER EXCEED (°C)
    process_note:  str      # Processing guidance

    # References
    refs: list = field(default_factory=list)


# ─────────────────────────────────────────────────────────────
#  Propellant definitions
# ─────────────────────────────────────────────────────────────

PROPELLANTS: Dict[str, PropellantData] = {

# ── Sugar/KN Systems ──────────────────────────────────────────
'KNSB': PropellantData(
    name         = "Potassium Nitrate / Sorbitol (KNSB 65/35)",
    abbreviation = "KNSB",
    composition  = {'KNO₃': 0.65, 'Sorbitol (C₆H₁₄O₆)': 0.35},
    density      = 1879.0,   # kg/m³; TMD = 1909; porous cast ~1879 [4]
    T_flame      = 1720.0,   # K; ProPEP/NASA-CEA at 3.5 MPa [4]
    gamma        = 1.131,    # Effective γ for K-CO₂-N₂-CO-H₂O mix [4]
    MW_products  = 38.90,    # g/mol; bulk effective (K₂CO₃, CO₂, N₂, CO, H₂O) [4]
    c_star       = 889.0,    # m/s [4]
    Isp_vac      = 165.0,    # s; optimum expansion, vacuum [4]
    Isp_sl       = 130.0,    # s; at 0.101 MPa ambient [4]
    burn_a       = 8.26,     # mm/s·MPa⁻ⁿ; Nakka empirical [4]
    burn_n       = 0.319,    # [4]; range 0.30–0.34 in literature
    P_range_MPa  = (1.0, 10.0),
    cure_T_C     = (95.0, 115.0),   # Sorbitol mp ≈ 95°C; cast at 95–115°C
    cure_t_hr    = (0.5, 2.0),      # Solidification; no chemical cure
    decomp_onset = 339.0,           # KNO₃ active decomp onset (DSC) [3]
    decomp_crit  = 400.0,           # KNO₃ → KNO₂ + ½O₂ — catastrophic [3]
    process_note = (
        "MELT-CAST system. Heat mixture to 95–115 °C until homogeneous; cast into "
        "pre-heated mold; cool slowly to prevent cracking. NO chemical curative "
        "required. ⚠ Keep temperature BELOW 200 °C at all times. Avoid contamination "
        "with fuels, sulfur, or metals other than those in formulation. Process in small "
        "batches (<50 g) inside an explosion-rated enclosure with blast shielding."
    ),
    refs=['[4] Nakka 2023', '[3] Kubota 2007', '[1] Sutton & Biblarz 2016'],
),

'KNSU': PropellantData(
    name         = "Potassium Nitrate / Sucrose (KNSU 65/35)",
    abbreviation = "KNSU",
    composition  = {'KNO₃': 0.65, 'Sucrose (C₁₂H₂₂O₁₁)': 0.35},
    density      = 1889.0,   # kg/m³ [4]
    T_flame      = 1733.0,   # K [4]
    gamma        = 1.134,
    MW_products  = 38.50,
    c_star       = 908.0,
    Isp_vac      = 164.0,
    Isp_sl       = 128.0,
    burn_a       = 7.96,     # mm/s·MPa⁻ⁿ [4]
    burn_n       = 0.321,    # [4]
    P_range_MPa  = (1.0, 10.0),
    cure_T_C     = (160.0, 185.0),  # Sucrose mp/decomp ~186°C — care required
    cure_t_hr    = (0.25, 1.0),
    decomp_onset = 300.0,           # Sucrose caramelization/decomp [3]
    decomp_crit  = 380.0,
    process_note = (
        "MELT-CAST system. Higher processing temperature than KNSB due to sucrose mp. "
        "Process at 160–185 °C. Mixture darkening (caramelization) near upper limit is "
        "acceptable. ⚠ Sucrose decomposes >186 °C — hold below 200 °C strictly. "
        "Use electric heating only; never open flame."
    ),
    refs=['[4] Nakka 2023', '[1] Sutton & Biblarz 2016'],
),

# ── AP Composite Systems ──────────────────────────────────────
'APCP_STD': PropellantData(
    name         = "AP/HTPB/Al Standard APCP (70/18/12)",
    abbreviation = "APCP-STD",
    composition  = {'AP': 0.70, 'HTPB (R-45M)': 0.18, 'Al (atomized)': 0.12},
    density      = 1786.0,   # kg/m³; measured TMD ~1800–1830; casting voids [1]
    T_flame      = 3350.0,   # K; CEA at 6.9 MPa [1]
    gamma        = 1.200,    # Effective two-phase (gas+Al₂O₃) [1,2]
    MW_products  = 29.00,    # g/mol; effective (incl. condensed Al₂O₃) [1]
    c_star       = 1578.0,   # m/s; Sutton Table 13-4 [1]
    Isp_vac      = 242.0,    # s; Sutton Table 13-3 [1]
    Isp_sl       = 210.0,
    burn_a       = 5.13,     # mm/s·MPa⁻ⁿ; Sutton Table 13-2 mid [1]
    burn_n       = 0.350,    # Typical bimodal AP dist. [1,2]
    P_range_MPa  = (2.0, 15.0),
    cure_T_C     = (50.0, 70.0),    # HTPB/IPDI optimal cure 60 °C [1,3]
    cure_t_hr    = (72.0, 168.0),   # 3–7 days at 60 °C for full crosslink [1]
    decomp_onset = 240.0,           # AP low-T decomp DSC exotherm [3]
    decomp_crit  = 300.0,           # Mixed APCP auto-ignition 280–320 °C [2]
    process_note = (
        "CHEMICALLY CURED thermosetting composite. Mix binder system (HTPB + "
        "plasticizer + IPDI curative, NCO:OH = 0.85–0.90) with solid fill "
        "(AP bimodal + Al). Cure at 60 °C for 72–120 h under vacuum. "
        "⚠ Mixed propellant is FRICTION, IMPACT, and HEAT SENSITIVE. "
        "⚠ AP decomposes exothermically above 240 °C; MIXED propellant ignition "
        "threshold is LOWER (~280 °C) due to Al and Fe₂O₃ catalysis. "
        "NEVER heat cured or uncured mix above 90 °C. Follow NFPA 1125."
    ),
    refs=['[1] Sutton & Biblarz 2016', '[2] Kuo & Summerfield 1984',
          '[3] Kubota 2007'],
),

'APCP_NOAL': PropellantData(
    name         = "AP/HTPB (No Metal) 80/20",
    abbreviation = "APCP-NoAl",
    composition  = {'AP': 0.80, 'HTPB (R-45M)': 0.20},
    density      = 1690.0,
    T_flame      = 2800.0,   # K; CEA at 6.9 MPa
    gamma        = 1.220,
    MW_products  = 26.00,    # g/mol; HCl, CO₂, H₂O, N₂
    c_star       = 1470.0,
    Isp_vac      = 215.0,
    Isp_sl       = 186.0,
    burn_a       = 4.20,     # mm/s·MPa⁻ⁿ [1]
    burn_n       = 0.330,    # [1]
    P_range_MPa  = (2.0, 14.0),
    cure_T_C     = (50.0, 70.0),
    cure_t_hr    = (72.0, 168.0),
    decomp_onset = 240.0,
    decomp_crit  = 300.0,
    process_note = (
        "Metal-free composite; cleaner exhaust (no Al₂O₃ smoke), lower Isp. "
        "Same cure schedule as aluminized APCP. Same thermal hazards apply. "
        "Suitable for research motors where optical transparency is needed."
    ),
    refs=['[1] Sutton & Biblarz 2016', '[2] Kuo & Summerfield 1984'],
),

# ── Energetic Binder Systems ──────────────────────────────────
'GAP_AP': PropellantData(
    name         = "GAP/AP Energetic Composite (20/60 + 15% TMETN + 5% Al)",
    abbreviation = "GAP-AP",
    composition  = {'GAP': 0.20, 'AP': 0.60, 'TMETN': 0.15, 'Al': 0.05},
    density      = 1760.0,
    T_flame      = 3100.0,
    gamma        = 1.220,
    MW_products  = 28.50,
    c_star       = 1530.0,
    Isp_vac      = 255.0,    # Higher than HTPB due to energetic binder [2,3]
    Isp_sl       = 226.0,
    burn_a       = 6.00,     # mm/s·MPa⁻ⁿ; range 5–8 per formulation [2,3]
    burn_n       = 0.380,    # Higher n; GAP azide contribution [3]
    P_range_MPa  = (3.0, 18.0),
    cure_T_C     = (40.0, 60.0),    # GAP cures at lower T than HTPB
    cure_t_hr    = (96.0, 240.0),   # 4–10 days; GAP cures slowly
    decomp_onset = 220.0,           # GAP azide decomp onset DSC [3]
    decomp_crit  = 270.0,           # ⚠ LOWER than HTPB systems — azide hazard
    process_note = (
        "⚠ ELEVATED HAZARD: GAP binder contains energetic azide groups (-N₃). "
        "Critical decomp threshold (270 °C) is LOWER than HTPB systems. "
        "GAP/AP mixtures may auto-ignite at 230–250 °C. Requires specialized "
        "facilities and training beyond standard APCP. Cure at 50–55 °C for "
        "minimum 96 h. Avoid all shock and friction during uncured processing."
    ),
    refs=['[2] Kuo & Summerfield 1984', '[3] Kubota 2007',
          'Frankel et al. (1992) AIAA-92-3461'],
),

'AP_HMX': PropellantData(
    name         = "AP/HTPB/HMX Nitramine Composite (60/12/16 + 12% Al)",
    abbreviation = "AP-HMX",
    composition  = {'AP': 0.60, 'HTPB': 0.12, 'HMX': 0.16, 'Al': 0.12},
    density      = 1800.0,
    T_flame      = 3200.0,
    gamma        = 1.210,
    MW_products  = 28.80,
    c_star       = 1555.0,
    Isp_vac      = 258.0,
    Isp_sl       = 228.0,
    burn_a       = 7.50,     # Higher due to HMX [2]
    burn_n       = 0.400,
    P_range_MPa  = (3.0, 20.0),
    cure_T_C     = (50.0, 65.0),
    cure_t_hr    = (96.0, 168.0),
    decomp_onset = 258.0,    # HMX β→δ transition 187 °C; decomp ~258 °C [3]
    decomp_crit  = 280.0,
    process_note = (
        "⚠ EXPLOSIVE HAZARD: HMX is a secondary explosive (UN Class 1.1). "
        "Requires ATF Class A explosives manufacturing license and licensed "
        "storage. DO NOT process outside of licensed explosive facilities. "
        "HMX is impact/friction sensitive at fine particle sizes."
    ),
    refs=['[1] Sutton & Biblarz 2016', '[2] Kuo & Summerfield 1984',
          '[3] Kubota 2007'],
),

}  # end PROPELLANTS dict


# ═══════════════════════════════════════════════════════════════
#  GRAIN GEOMETRY
# ═══════════════════════════════════════════════════════════════

class BATESGrain:
    """
    BATES (Ballistic Test and Evaluation System) hollow-cylinder grain.

    Burns on:
      • Inner cylindrical surface (core)
      • Both annular end faces
    Outer diameter is INHIBITED (does not regress).

    Burn surface area:
      Ab = π·Di·L  +  2·(π/4)·(Do² − Di²)
         = [inner core]  +  [two annular ends]

    Regression per time step Δt:
      Di  →  Di + 2·r·Δt     (inner surface burns outward)
      L   →  L  − 2·r·Δt    (both ends shorten by r each)

    Grain burnout when Di ≥ Do  OR  L ≤ 0.

    Reference: Nakka [4]; Sutton [1] Ch. 13.
    """
    def __init__(self, ro_m: float, ri_m: float, L_m: float):
        assert ri_m < ro_m, "Inner radius must be less than outer radius."
        self.ro = ro_m
        self.ri0 = ri_m
        self.L0  = L_m

    def burn_area(self, ri: float, L: float) -> float:
        """Instantaneous burning surface area (m²)."""
        A_core = np.pi * (2*ri) * L
        A_ends = 2 * (np.pi/4) * ((2*self.ro)**2 - (2*ri)**2)
        return max(0.0, A_core + A_ends)

    def volume(self, ri: float, L: float) -> float:
        """Remaining propellant volume (m³)."""
        return np.pi * (self.ro**2 - ri**2) * L

    def regress(self, ri, L, r, dt):
        ri_new = ri + r * dt
        L_new  = max(0.0, L - 2*r*dt)
        done   = (ri_new >= self.ro) or (L_new <= 0.0)
        return ri_new, L_new, done

    @property
    def initial_burn_area(self): return self.burn_area(self.ri0, self.L0)
    @property
    def initial_volume(self): return self.volume(self.ri0, self.L0)
    @property
    def label(self):
        return (f"BATES  ro={self.ro*1e3:.1f} mm  "
                f"ri={self.ri0*1e3:.1f} mm  L={self.L0*1e3:.1f} mm")


class EndBurningGrain:
    """
    End-burning (cigarette) grain.
    Burns only on one flat face → constant area → neutral thrust profile.

    Reference: Sutton [1] Ch. 13.
    """
    def __init__(self, radius_m: float, length_m: float):
        self.R  = radius_m
        self.L0 = length_m

    def burn_area(self) -> float:
        return np.pi * self.R**2

    def volume(self, L: float) -> float:
        return np.pi * self.R**2 * L

    def regress(self, L, r, dt):
        L_new = max(0.0, L - r*dt)
        done  = (L_new <= 0.0)
        return L_new, done

    @property
    def initial_burn_area(self): return self.burn_area()
    @property
    def initial_volume(self): return self.volume(self.L0)
    @property
    def label(self):
        return (f"End-Burning  R={self.R*1e3:.1f} mm  "
                f"L={self.L0*1e3:.1f} mm")


# ═══════════════════════════════════════════════════════════════
#  INTERNAL BALLISTICS SIMULATOR
# ═══════════════════════════════════════════════════════════════

class InternalBallistics:
    """
    Quasi-Steady-State (QSS) solid rocket motor simulation.

    PRESSURE MODEL
    ──────────────
    From steady-state mass balance  ṁ_gen = ṁ_exit:

        ρ_p · r · Ab  =  (Pc · At) / c*                    (1)

    Substituting Saint-Robert law  r = a · Pc^n :

        ρ_p · a · Pc^n · Ab  =  Pc · At / c*
        Pc^(1−n)  =  ρ_p · a · c* · (Ab/At)               (2)
        Pc  =  (ρ_p · a · c* · Kn)^(1/(1−n))              (3)

    where  Kn = Ab/At  is the Klemmung coefficient.

    Stability: n < 1 required; n ≥ 1 → Pc diverges (chuffing/CATO risk).

    THRUST
    ──────
    F = Cf · Pc · At                                        (4)

    where Cf (sea-level) is back-calculated from literature Isp data:
        Cf,sl = Isp_sl · g₀ / c*

    SPECIFIC IMPULSE
    ────────────────
    Isp = F / (ṁ · g₀)                                    (5)

    MASS FLOW (nozzle, isentropic)
    ──────────────────────────────
    ṁ = Pc · At · Γ / √(Tc · R_sp)                        (6)

    where Γ = √[γ · (2/(γ+1))^((γ+1)/(γ−1))]   (Vandenkerckhove function)
    and   R_sp = R_univ / M_products  (specific gas constant, J/kg·K)

    Reference: Sutton [1] Ch. 3; Nakka [4].
    """

    def __init__(
        self,
        prop:  PropellantData,
        grain,                    # BATESGrain or EndBurningGrain
        At_m2: float,             # Nozzle throat area (m²)
        Ae_m2: float,             # Nozzle exit area (m²)
        Pa:    float = ATM_PA,    # Ambient pressure (Pa)
        dt:    float = 0.0005,    # Time step (s)
    ):
        self.prop  = prop
        self.grain = grain
        self.At    = At_m2
        self.Ae    = Ae_m2
        self.Pa    = Pa
        self.dt    = dt
        self.eps   = Ae_m2 / At_m2            # Expansion ratio
        self.R_sp  = R_UNIV / (prop.MW_products * 1e-3)  # J/(kg·K)

        # Sea-level thrust coefficient from literature Isp
        self.Cf_sl = (prop.Isp_sl * G0) / prop.c_star

        # Vandenkerckhove function Γ
        g = prop.gamma
        self.Gamma = np.sqrt(g * (2/(g+1))**((g+1)/(g-1)))

    # ── Core equations ─────────────────────────────────────────

    def saint_robert(self, Pc_Pa: float) -> float:
        """Linear burn rate r (m/s) via Saint-Robert's law."""
        Pc_MPa = np.clip(Pc_Pa * PA_TO_MPA,
                         self.prop.P_range_MPa[0],
                         self.prop.P_range_MPa[1])
        return self.prop.burn_a * (Pc_MPa ** self.prop.burn_n) * 1e-3  # m/s

    def qss_pressure(self, Ab: float) -> float:
        """
        Quasi-steady-state chamber pressure (Pa) from Eq. (3).
        Convert a [mm/s·MPa⁻ⁿ] → [m/s·Pa⁻ⁿ] for SI consistency.
        """
        n  = self.prop.burn_n
        if n >= 1.0:
            warnings.warn(
                f"n = {n:.3f} ≥ 1 → motor is PRESSURE-UNSTABLE. "
                "Pc will diverge. Check formulation.", RuntimeWarning
            )
            return np.nan
        a_SI = self.prop.burn_a * 1e-3 * (MPA_TO_PA ** n)
        Kn   = Ab / self.At
        return (self.prop.density * a_SI * self.prop.c_star * Kn) ** (1.0/(1.0-n))

    def thrust_isp_mdot(self, Pc_Pa: float):
        """Return F (N), Isp (s), ṁ (kg/s)."""
        F    = self.Cf_sl * Pc_Pa * self.At
        mdot = (Pc_Pa * self.At * self.Gamma /
                np.sqrt(self.prop.T_flame * self.R_sp))
        Isp  = F / (mdot * G0) if mdot > 0 else 0.0
        return F, Isp, mdot

    # ── Main simulation loop ────────────────────────────────────

    def run(self) -> dict:
        """Execute the time-step simulation. Returns results dict."""

        # Initialise grain state
        if isinstance(self.grain, BATESGrain):
            ri, L = self.grain.ri0, self.grain.L0
        else:
            L_eb = self.grain.L0

        t_list   = []
        Pc_list  = []
        F_list   = []
        Isp_list = []
        r_list   = []
        Kn_list  = []
        Ab_list  = []
        mdot_list= []

        t       = 0.0
        burnout = False

        while not burnout and t < 600.0:

            # Burning surface area
            if isinstance(self.grain, BATESGrain):
                Ab = self.grain.burn_area(ri, L)
            else:
                Ab = self.grain.burn_area()

            if Ab <= 1e-9:
                break

            # Chamber pressure
            Pc = self.qss_pressure(Ab)
            if Pc <= 0 or np.isnan(Pc):
                break

            # Burn rate
            r = self.saint_robert(Pc)   # m/s

            # Thrust / Isp / mass flow
            F, Isp, mdot = self.thrust_isp_mdot(Pc)

            # Record
            t_list.append(t)
            Pc_list.append(Pc)
            F_list.append(F)
            Isp_list.append(Isp)
            r_list.append(r * 1e3)       # mm/s
            Kn_list.append(Ab / self.At)
            Ab_list.append(Ab)
            mdot_list.append(mdot)

            # Regress grain
            if isinstance(self.grain, BATESGrain):
                ri, L, burnout = self.grain.regress(ri, L, r, self.dt)
            else:
                L_eb, burnout = self.grain.regress(L_eb, r, self.dt)

            t += self.dt

        # Convert to arrays
        t_a   = np.array(t_list)
        Pc_a  = np.array(Pc_list)
        F_a   = np.array(F_list)
        Isp_a = np.array(Isp_list)
        r_a   = np.array(r_list)
        Kn_a  = np.array(Kn_list)
        Ab_a  = np.array(Ab_list)

        if len(t_a) == 0:
            raise RuntimeError("Simulation produced no data. Check grain/nozzle sizing.")

        It       = float(np.trapz(F_a, t_a))   # total impulse
        m_prop   = self.grain.initial_volume * self.prop.density

        return dict(
            time=t_a, Pc=Pc_a, F=F_a, Isp=Isp_a,
            burn_rate=r_a, Kn=Kn_a, Ab=Ab_a,
            burn_time    = t_a[-1],
            total_impulse= It,
            avg_thrust   = float(np.mean(F_a)),
            avg_Pc       = float(np.mean(Pc_a)),
            max_Pc       = float(np.max(Pc_a)),
            avg_Isp      = float(np.mean(Isp_a)),
            avg_r        = float(np.mean(r_a)),
            prop_mass_g  = m_prop * 1000.0,
        )


# ═══════════════════════════════════════════════════════════════
#  PLOTTING
# ═══════════════════════════════════════════════════════════════

_STYLE = {
    'bg'     : '#0d1117',
    'panel'  : '#161b22',
    'border' : '#30363d',
    'text'   : '#c9d1d9',
    'muted'  : '#8b949e',
    'blue'   : '#58a6ff',
    'green'  : '#3fb950',
    'amber'  : '#e3b341',
    'red'    : '#f85149',
    'purple' : '#ce93d8',
    'teal'   : '#80cbc4',
}


def _style_ax(ax):
    s = _STYLE
    ax.set_facecolor(s['panel'])
    for sp in ax.spines.values():
        sp.set_color(s['border'])
    ax.tick_params(colors=s['muted'], labelsize=8.5)
    ax.xaxis.label.set_color(s['muted'])
    ax.yaxis.label.set_color(s['muted'])
    ax.title.set_color(s['text'])


def plot_simulation(res: dict, prop: PropellantData,
                    grain, At_m2: float, Ae_m2: float,
                    outfile: str = 'ballistics_results.png'):
    """Full 6-panel simulation results figure."""
    s = _STYLE
    plt.style.use('dark_background')

    fig = plt.figure(figsize=(17, 10), facecolor=s['bg'])
    gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.48, wspace=0.38)

    ax_F   = fig.add_subplot(gs[0, :2])   # Thrust vs time
    ax_Pc  = fig.add_subplot(gs[1, :2])   # Pressure vs time
    ax_r   = fig.add_subplot(gs[2, :2])   # Saint-Robert curve
    ax_Kn  = fig.add_subplot(gs[0, 2])    # Kn vs time
    ax_Isp = fig.add_subplot(gs[1, 2])    # Isp vs time
    ax_txt = fig.add_subplot(gs[2, 2])    # Summary text

    for ax in [ax_F, ax_Pc, ax_r, ax_Kn, ax_Isp, ax_txt]:
        _style_ax(ax)

    t  = res['time']
    Pc = res['Pc'] * PA_TO_MPA  # → MPa
    F  = res['F']
    r  = res['burn_rate']
    Kn = res['Kn']
    Isp= res['Isp']

    # ── Thrust ───────────────────────────────────────
    ax_F.fill_between(t, F, alpha=0.22, color=s['blue'])
    ax_F.plot(t, F, color=s['blue'], lw=2)
    ax_F.axhline(res['avg_thrust'], color=s['blue'], lw=0.9, ls='--', alpha=0.5)
    ax_F.set_xlabel('Time (s)');  ax_F.set_ylabel('Thrust (N)')
    ax_F.set_title(
        f"Thrust Curve   ·   Total Impulse = {res['total_impulse']:.2f} N·s")
    ax_F.text(0.98, 0.90, f"avg = {res['avg_thrust']:.1f} N",
              transform=ax_F.transAxes, ha='right', fontsize=8.5, color=s['blue'])

    # ── Chamber Pressure ─────────────────────────────
    ax_Pc.fill_between(t, Pc, alpha=0.18, color=s['amber'])
    ax_Pc.plot(t, Pc, color=s['amber'], lw=2)
    ax_Pc.axhline(res['avg_Pc']*PA_TO_MPA, color=s['amber'], lw=0.9, ls='--', alpha=0.5)
    ax_Pc.set_xlabel('Time (s)');  ax_Pc.set_ylabel('Chamber Pressure (MPa)')
    ax_Pc.set_title(
        f"Chamber Pressure   ·   Peak Pc = {res['max_Pc']*PA_TO_MPA:.2f} MPa")
    ax_Pc.text(0.98, 0.90, f"avg = {res['avg_Pc']*PA_TO_MPA:.2f} MPa",
               transform=ax_Pc.transAxes, ha='right', fontsize=8.5, color=s['amber'])

    # ── Saint-Robert Burn Rate ────────────────────────
    P_plot = np.linspace(*prop.P_range_MPa, 300)
    r_plot = prop.burn_a * P_plot**prop.burn_n
    ax_r.plot(P_plot, r_plot, color=s['green'], lw=2.5,
              label=f"r = {prop.burn_a}·Pc^{prop.burn_n}")
    ax_r.scatter(Pc, r, c=s['blue'], s=3, alpha=0.35, zorder=4,
                 label='Simulation points')
    ax_r.set_xlabel('Chamber Pressure (MPa)')
    ax_r.set_ylabel('Burn Rate (mm/s)')
    ax_r.set_title("Saint-Robert's Law:  r = a · Pc ⁿ")
    ax_r.legend(fontsize=8, facecolor='#21262d',
                edgecolor=s['border'], labelcolor=s['text'])

    # ── Klemmung Kn ──────────────────────────────────
    ax_Kn.plot(t, Kn, color=s['purple'], lw=2)
    ax_Kn.set_xlabel('Time (s)');  ax_Kn.set_ylabel('Kn = Ab / At')
    ax_Kn.set_title('Klemmung (Kn)')

    # ── Specific Impulse ─────────────────────────────
    ax_Isp.plot(t, Isp, color=s['teal'], lw=2)
    ax_Isp.axhline(res['avg_Isp'], color=s['teal'], lw=0.9, ls='--', alpha=0.5)
    ax_Isp.set_xlabel('Time (s)');  ax_Isp.set_ylabel('Isp (s)')
    ax_Isp.set_title('Specific Impulse')
    ax_Isp.text(0.98, 0.90, f"avg = {res['avg_Isp']:.1f} s",
                transform=ax_Isp.transAxes, ha='right', fontsize=8.5, color=s['teal'])

    # ── Summary ───────────────────────────────────────
    ax_txt.axis('off')
    It_cls = _motor_class(res['total_impulse'])
    summary = (
        "── RESULTS SUMMARY ──────────────\n\n"
        f"Propellant  {prop.abbreviation}\n"
        f"Grain       {grain.label}\n\n"
        f"Burn time   {res['burn_time']:.3f} s\n"
        f"Total Imp   {res['total_impulse']:.1f} N·s  [{It_cls}]\n"
        f"Avg Thrust  {res['avg_thrust']:.1f} N\n"
        f"Max Pc      {res['max_Pc']*PA_TO_MPA:.3f} MPa\n"
        f"            ({res['max_Pc']/ATM_PA:.1f} atm)\n"
        f"Avg Isp     {res['avg_Isp']:.1f} s\n"
        f"Avg r       {res['avg_r']:.2f} mm/s\n"
        f"Prop mass   {res['prop_mass_g']:.1f} g\n\n"
        f"Nozzle\n"
        f"  At = {At_m2*1e6:.2f} mm²\n"
        f"  Ae = {Ae_m2*1e6:.2f} mm²\n"
        f"  ε  = {Ae_m2/At_m2:.2f}"
    )
    ax_txt.text(0.05, 0.97, summary, transform=ax_txt.transAxes,
                va='top', fontsize=8.5, family='monospace',
                color=s['text'], linespacing=1.55)

    fig.suptitle(
        f"Solid Rocket Motor — Internal Ballistics Simulation\n{prop.name}",
        fontsize=13, color='#ffffff', fontweight='bold', y=1.00
    )

    plt.savefig(outfile, dpi=150, bbox_inches='tight', facecolor=s['bg'])
    print(f"\n  ✓  Saved: {outfile}")
    plt.show()


def plot_burn_rate_all(outfile: str = 'burn_rate_comparison.png'):
    """Log-scale burn rate comparison for all propellants."""
    s = _STYLE
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(11, 6), facecolor=s['bg'])
    _style_ax(ax)

    colors = [s['blue'], s['green'], s['amber'], s['red'], s['purple'], s['teal']]
    P      = np.linspace(0.5, 20, 600)

    for (key, prop), col in zip(PROPELLANTS.items(), colors):
        # Clip to valid range
        P_valid = P[(P >= prop.P_range_MPa[0]) & (P <= prop.P_range_MPa[1])]
        r_valid = prop.burn_a * P_valid**prop.burn_n
        ax.semilogy(P_valid, r_valid, color=col, lw=2.2,
                    label=f"{prop.abbreviation}: a={prop.burn_a}, n={prop.burn_n}")

    ax.set_xlabel('Chamber Pressure (MPa)')
    ax.set_ylabel('Burn Rate r (mm/s) — log scale')
    ax.set_title("Saint-Robert's Law Comparison — All Propellants in Database",
                 fontweight='bold')
    ax.legend(fontsize=9, facecolor='#21262d',
              edgecolor=s['border'], labelcolor=s['text'])
    ax.grid(True, alpha=0.12, color=s['border'], which='both')

    plt.tight_layout()
    plt.savefig(outfile, dpi=150, bbox_inches='tight', facecolor=s['bg'])
    print(f"  ✓  Saved: {outfile}")
    plt.show()


def _motor_class(It: float) -> str:
    """Return NAR/TRA motor class letter for a given total impulse (N·s)."""
    classes = [
        (0, 'sub-A'), (1.25, 'A'), (2.5, 'B'), (5, 'C'), (10, 'D'),
        (20, 'E'), (40, 'F'), (80, 'G'), (160, 'H'), (320, 'I'),
        (640, 'J'), (1280, 'K'), (2560, 'L'), (5120, 'M'),
        (10240, 'N'), (20480, 'O'),
    ]
    cls = 'O+'
    for lim, name in reversed(classes):
        if It >= lim:
            cls = name
            break
    return cls


# ═══════════════════════════════════════════════════════════════
#  MATERIAL SCIENCE REFERENCE PRINTER
# ═══════════════════════════════════════════════════════════════

def print_material_reference(key: str):
    """Print formatted material science reference for one propellant."""
    p = PROPELLANTS[key]
    W = 72

    def hr(ch='═'): print(ch * W)
    def sec(title): print(f"\n  ✦  {title}")

    hr()
    print(f"  MATERIAL SCIENCE REFERENCE: {p.name}")
    hr()

    sec("COMPOSITION")
    for comp, frac in p.composition.items():
        print(f"     {comp:<35s} {frac*100:5.1f} wt%")

    sec("THERMOPHYSICAL PROPERTIES")
    rows = [
        ("Density ρ",                f"{p.density:.1f} kg/m³  ({p.density/1000:.3f} g/cm³)"),
        ("Adiabatic flame temp Tc",  f"{p.T_flame:.0f} K   ({p.T_flame-273.15:.0f} °C)"),
        ("Ratio of sp. heats γ",     f"{p.gamma:.3f}"),
        ("Mean product MW",          f"{p.MW_products:.2f} g/mol"),
        ("Characteristic vel c*",    f"{p.c_star:.0f} m/s"),
        ("Isp (vacuum, opt. exp.)",  f"{p.Isp_vac:.0f} s"),
        ("Isp (sea-level, opt.)",    f"{p.Isp_sl:.0f} s"),
    ]
    for label, val in rows:
        print(f"     {label:<32s} {val}")

    sec("SAINT-ROBERT BURN RATE  —  r = a · Pc^n   [Pc in MPa, r in mm/s]")
    print(f"     a (coefficient)              {p.burn_a:.3f} mm/s · MPa⁻ⁿ")
    print(f"     n (pressure exponent)        {p.burn_n:.3f}  {'⚠ n≥1 UNSTABLE' if p.burn_n>=1 else ''}")
    print(f"     Valid Pc range               {p.P_range_MPa[0]:.1f} – {p.P_range_MPa[1]:.1f} MPa")
    print(f"\n     Sample burn rates:")
    for P_s in [1.0, 2.0, 3.5, 6.9, 10.0]:
        if p.P_range_MPa[0] <= P_s <= p.P_range_MPa[1]:
            r_s = p.burn_a * P_s**p.burn_n
            print(f"       Pc = {P_s:5.1f} MPa  →  r = {r_s:6.2f} mm/s")

    sec("THERMAL PROCESSING PROPERTIES")
    print(f"     Optimal process temp         {p.cure_T_C[0]:.0f} – {p.cure_T_C[1]:.0f} °C")
    print(f"     Typical cure / set time      {p.cure_t_hr[0]:.1f} – {p.cure_t_hr[1]:.1f} hours")
    print(f"     Decomposition onset (DSC)    {p.decomp_onset:.0f} °C   [exotherm begins]")
    print(f"     ⚠  CRITICAL THRESHOLD:       {p.decomp_crit:.0f} °C   ← NEVER EXCEED")

    sec("PROCESSING NOTES")
    for line in textwrap.wrap(p.process_note, width=W-6):
        print(f"     {line}")

    sec("REFERENCES")
    for ref in p.refs:
        print(f"     {ref}")

    print()
    hr('─')
    print("  ⚠  SAFETY DISCLAIMER")
    hr('─')
    disclaimer = (
        "This reference data is provided solely for EDUCATIONAL and THEORETICAL "
        "modeling. Fabrication of solid propellants requires proper licensing "
        "(ATF or equivalent), certified facilities, trained personnel, and "
        "compliance with NFPA 1125, TRA, and NAR safety codes. The decomposition "
        "temperatures listed above are CRITICAL LIMITS — exceeding them during "
        "mixing, casting, or curing can cause spontaneous ignition, deflagration, "
        "or detonation. The authors assume NO LIABILITY for use of this information."
    )
    for line in textwrap.wrap(disclaimer, width=W-4):
        print(f"  {line}")
    hr()


# ═══════════════════════════════════════════════════════════════
#  MAIN — INTERACTIVE SESSION
# ═══════════════════════════════════════════════════════════════

def _ask(prompt: str, default):
    """Prompt with default; return converted value."""
    raw = input(f"  {prompt} [{default}]: ").strip()
    if raw == '':
        return default
    try:
        return type(default)(raw)
    except ValueError:
        return default


def main():
    s = _STYLE
    print("\n" + "█"*72)
    print("  SOLID PROPELLANT INTERNAL BALLISTICS SIMULATION TOOL")
    print("  Educational / Theoretical Use Only — See Module Docstring")
    print("█"*72)
    print()

    # ── 1. Select propellant ───────────────────────────────────
    keys = list(PROPELLANTS.keys())
    print("  Available propellants:")
    for i, k in enumerate(keys, 1):
        print(f"  [{i}]  {PROPELLANTS[k].abbreviation:12s}  {PROPELLANTS[k].name}")

    choice = _ask("Select propellant (number)", 1)
    try:
        prop_key = keys[int(choice)-1]
    except (ValueError, IndexError):
        prop_key = 'KNSB'
    prop = PROPELLANTS[prop_key]

    # ── 2. Print material reference ────────────────────────────
    print()
    print_material_reference(prop_key)

    # ── 3. Grain geometry ─────────────────────────────────────
    print("\n  Grain Geometry:")
    print("  [1]  BATES         — hollow cylinder, inner core + both end faces")
    print("  [2]  End-Burning   — flat face only, constant area, neutral burn")
    geom = _ask("Select geometry (number)", 1)

    print(f"\n  ── {prop.abbreviation} Motor Parameters ──")

    if str(geom) == '2':
        R_mm = _ask("Grain radius   (mm)", 25.0)
        L_mm = _ask("Grain length   (mm)", 80.0)
        grain = EndBurningGrain(R_mm*1e-3, L_mm*1e-3)
    else:
        ro_mm = _ask("Outer radius   (mm)", 30.0)
        ri_mm = _ask("Inner radius   (mm)", 12.0)
        L_mm  = _ask("Grain length   (mm)", 75.0)
        grain = BATESGrain(ro_mm*1e-3, ri_mm*1e-3, L_mm*1e-3)

    # ── 4. Nozzle ─────────────────────────────────────────────
    At_mm2 = _ask("Throat area    (mm²)", 45.0)
    Ae_mm2 = _ask("Exit area      (mm²)", 180.0)
    dt_ms  = _ask("Time step      (ms)", 0.5)

    At = float(At_mm2) * 1e-6
    Ae = float(Ae_mm2) * 1e-6
    dt = float(dt_ms)  * 1e-3

    # ── 5. Run simulation ──────────────────────────────────────
    print(f"\n  Running simulation …  (dt = {dt_ms} ms)")
    sim = InternalBallistics(prop, grain, At, Ae, dt=dt)
    res = sim.run()

    # ── 6. Print summary ───────────────────────────────────────
    print("\n" + "═"*72)
    print("  SIMULATION RESULTS SUMMARY")
    print("═"*72)
    print(f"  Propellant:           {prop.name}")
    print(f"  Grain:                {grain.label}")
    print(f"  Throat area At:       {At*1e6:.2f} mm²")
    print(f"  Exit area   Ae:       {Ae*1e6:.2f} mm²  (ε = {Ae/At:.2f})")
    print(f"  Propellant mass:      {res['prop_mass_g']:.2f} g")
    print()
    print(f"  Burn time:            {res['burn_time']:.3f} s")
    print(f"  Total Impulse:        {res['total_impulse']:.2f} N·s  "
          f"[{_motor_class(res['total_impulse'])}-class]")
    print(f"  Average Thrust:       {res['avg_thrust']:.2f} N")
    print(f"  Peak Chamber Pres.:   {res['max_Pc']*PA_TO_MPA:.3f} MPa  "
          f"({res['max_Pc']/ATM_PA:.1f} atm)")
    print(f"  Avg Chamber Pres.:    {res['avg_Pc']*PA_TO_MPA:.3f} MPa")
    print(f"  Average Isp:          {res['avg_Isp']:.1f} s")
    print(f"  Average burn rate:    {res['avg_r']:.2f} mm/s")
    print(f"  Initial Kn:           {grain.initial_burn_area/At:.1f}")

    # ── 7. Plots ──────────────────────────────────────────────
    print()
    plot_burn_rate_all()
    plot_simulation(res, prop, grain, At, Ae)

    print("\n  Simulation complete ✓")
    print("═"*72)


if __name__ == '__main__':
    main()
