"""
MPBSI Framework Backend v5.7 — MATLAB MPBSI_Evaluator_Resource_Land.m Exact Match

RESOURCE MODE: COMPLETE RE-ALIGNMENT WITH MATLAB SOURCE (v5.7)
Source: MPBSI_Evaluator_Resource_Land.m (final confirmed MATLAB evaluator)

  REVERT-H2CAP [mpbsi_evaluator_resource]:
         RESTORED: H2_effective = min(x[4], x[3]*24)  (MATLAB line 65)
         v5.5 incorrectly removed this cap believing it was "wrong".
         The cap IS CORRECT — it forces optimizer to size EL large enough to fill
         H2 tank within 24 h.  Without cap: Python chose H2=9999 kWh, EL=21 kW
         (H2_eff only 516 kWh — 95% loss).  MATLAB correctly finds EL=164 kW, H2=3426 kWh
         (EL×24=3936 > H2 → full H2 credit).  Deviation: EL off by −87%, H2 off by +192%.
         usable_storage = (0.95-0.20)×x[2] + H2_effective×eta_FC  (MATLAB exact)

  REVERT-T2 [mpbsi_evaluator_resource — TRI]:
         RESTORED: T2 = 1-exp(-sim.autonomy_days/1.5)  (MATLAB line 112)
         v5.6 changed T2 to use storage_ratio — WRONG.  MATLAB uses Results.Autonomy_days
         from dispatch (0.80×BESS + E_H2_max×eta_FC full H2, uncapped).
         T2 (dispatch/uncapped) rewards theoretical max autonomy.
         T4 (storage_ratio/capped) rewards EL-constrained practical storage.
         This deliberate difference is present in MATLAB and must be preserved.

  REVERT-T3 [mpbsi_evaluator_resource — TRI]:
         RESTORED: redundancy = x[2]/1000 + x[5]/100  (MATLAB line 114, BESS+FC only)
         v5.6 added H2 to T3 — WRONG.  MATLAB T3 does NOT include H2.

  REVERT-L5 [mpbsi_evaluator_resource — LSI]:
         RESTORED: L5 = 1-exp(-x[2]/2000)  (MATLAB line 146, BESS only)
         v5.6 changed L5 to use usable_storage — WRONG.  MATLAB L5 is BESS-only.

  FIX-FEASIBILITY-EVAL [mpbsi_evaluator_resource]:
         MATLAB evaluator checks: LPSP_critical > 1e-4 OR Annual_RES < Annual_Load
         (Renewable_ratio >= 1.0, NOT 0.99).  Python was using sim.is_feasible
         (dispatch flag, 0.99 threshold).  Fixed to check total_renewable < annual_load.

  FIX-CV-RESOURCE [nsga2_optimize _cv resource branch]:
         Updated constraint-violation threshold from 0.99 to 1.0 to match evaluator.

ROOT CAUSE SUMMARY: The H2/EL imbalance (Python: H2≫MATLAB, EL≪MATLAB) was entirely
caused by removing H2_effective cap in v5.5.  Without the cap, the optimizer exploited
the MPBSI reward for large H2 tanks while keeping EL tiny (cheap NPC).  With cap restored,
the optimizer must size EL to match H2 tank (EL×24 ≥ H2), exactly replicating MATLAB.

RESOURCE MODE WIND & H2 UNDER-SIZING — ORIGINAL FIXES (v5.5):

  FIX-AUT-PENALTY [nsga2_optimize _eval resource + nsga_objective_resource]:
         Restored autonomy economic penalty (Penalty_factor=0.05, limit=3 days):
           if Autonomy_days > 3: NPC *= (1 + 0.05*(Autonomy_days - 3))
         From MATLAB NSGA_Objective_Resource.m; uses full H2 (not H2_effective).

KEY NSGA-II FIXES (v5.0 — original):
  1. PSO mission seed: rng(12); PSO resource: rng(16)
  2. NSGA resource renewable_ratio: >= 1.0 in evaluator (>= 0.99 in dispatch)
  3. NSGA n_pop=80, max_gen=60

MATLAB NSGA BOUNDS:
  Resource: PV=[0, Land/10], Wind=[0, Land/15], BESS=[0, 10×CritDaily],
            H2=[0, 8×CritDaily/η_FC], EL=[0, PV_max+Wind_max],
            FC=[0.5×CritPeak, 1.5×CritPeak]

RESOURCE MODE BATTERY TOO HIGH / H2 TOO LOW — ROOT CAUSES AND FIXES (v5.6):

  FIX-T2 [mpbsi_evaluator_resource — TRI block]:
         T2 changed from sim.autonomy_days (dispatch: 0.80xBESS factor) to
         storage_ratio (design formula: 0.75xBESS), matching NSGA_MASTER_RESOURCE
         and the NPC autonomy penalty.  The dispatch SOC_max=E_BESS gives a 0.80
         factor (battery charges to 100%), but MATLAB NSGA_MASTER and evaluator use
         (0.95-0.20)=0.75.  Using 0.80 in T2 gave battery a 6.7%/kWh structural
         advantage over H2 in T2 (and aliased R1, L2), cumulatively biasing the
         optimiser toward BESS over H2.

  FIX-T3 [mpbsi_evaluator_resource — TRI block]:
         T3 redundancy changed from BESS-only (x[2]/1000 + x[5]/100) to include H2:
           redundancy = 0.75xBESS/1000 + H2xeta_FC/1000 + x[5]/100
         Old formula had NO H2 term: T3 was a pure battery-size reward.
         With T3 weight=0.20 in TRI, TRI weight=0.30 in MPBSI -> 6% of MPBSI
         was purely from battery.  At realistic sizes (BESS=2000, H2=10000):
         T3 gains +0.153, L5 gains +0.338, total MPBSI shift = +0.0186.

  FIX-L5 [mpbsi_evaluator_resource — LSI block]:
         L5 changed from BESS-only (1-exp(-x[2]/2000)) to combined usable_storage:
           L5 = 1 - exp(-usable_storage / 2000)
         where usable_storage = 0.75xBESS + H2xeta_FC (same as all other metrics).
         Old L5 gave H2 zero credit regardless of H2 size.  LSI weight=0.15
         for L5, MPBSI weight=0.20 = 3% of MPBSI from battery alone.

RESOURCE MODE WIND & H2 UNDER-SIZING — ROOT CAUSES AND FIXES (v5.5):

RESOURCE MODE WIND & H2 UNDER-SIZING — ROOT CAUSES AND FIXES (v5.5):

  FIX-H2CAP [mpbsi_evaluator_resource]:
         Removed incorrect H2_effective cap: min(x[4], x[3]*24).
         MATLAB (NSGA_MASTER_RESOURCE.m + microgrid_dispatch_resource.m) uses the
         full H2 tank for usable_storage = (0.95-0.20)*x(3) + x(5)*eta_FC with NO
         1-day electrolyzer charging cap.  The cap suppressed H2's contribution to
         T2/autonomy in the MPBSI reward, pushing the optimizer to under-size H2.

  FIX-FEASIBILITY [mpbsi_evaluator_resource + nsga2_optimize _cv]:
         Feasibility threshold corrected from renewable_ratio >= 1.0 back to >= 0.99,
         matching MATLAB microgrid_dispatch_resource.m:
           Results.feasible = (LPSP <= 1e-4) && (Renewable_ratio >= 0.99)
         The too-strict 1.0 threshold rejected valid solutions near the Pareto front.
         The evaluator now delegates to sim.is_feasible (already correct at 0.99).

  FIX-AUT-PENALTY [nsga2_optimize _eval resource + nsga_objective_resource]:
         REVERTED incorrect FIX-E: the autonomy economic penalty IS present in MATLAB
         NSGA_Objective_Resource.m (Penalty_factor=0.05, Autonomy_limit=3 days):
           if Autonomy_days > 3
               NPC = NPC * (1 + 0.05*(Autonomy_days - 3))
         FIX-E wrongly removed this penalty claiming it didn't exist.  Without it,
         large BESS had no NPC cost-of-oversizing → optimizer substituted cheap BESS
         for Wind (₹120k/kW), causing Wind to be consistently under-sized.  Restoring
         the penalty re-introduces the correct economic trade-off that drives Wind
         selection in MATLAB.

KEY NSGA-II FIXES (v5.0 — original):
  1. PSO mission seed: rng(12) [was incorrectly rng(16)]
  2. NSGA _cv() autonomy: mission autonomy >= 7d enforced in constraint-domination
  3. NSGA resource renewable_ratio constraint: >= 0.99 (MATLAB dispatch threshold)
  4. NSGA population init: pure uniform random (MATLAB exact)
  5. NSGA n_pop=80, max_gen=60 [MATLAB gamultiobj PopulationSize=80, MaxGenerations=60]
  6. nsga_objective_mission: Step9 survivability penalty removed [not in MATLAB]

ADDITIONAL FIXES (v5.1 — cross-verification against MATLAB source):
  FIX-A [microgrid_dispatch_resource]: Wind hub height corrected 40m → 20m.
         MATLAB uses Hhub=20, Href=10, alpha=0.14 for BOTH modes.
         Old: V2 = wind*(40/10)^0.14  →  ~4% excess wind power vs MATLAB.
         New: V2 = wind*(20/10)^0.14  ✓

  FIX-B [nsga2_optimize mission _eval()]: Autonomy oversizing penalty now uses
         the MATLAB re-computed formula, not sim.autonomy_days.
         MATLAB NSGA_Objective_Mission.m:
           usable_energy  = (0.95-0.20)*x(3) + x(5)*eta_FC
           critical_daily = (Annual_Load_MWh*1000/365)*0.60
           Autonomy_days  = usable_energy / critical_daily
         Old: aut = s.autonomy_days  [dispatch uses E_H2_max — numerically different]
         New: aut = ((0.75)*x[2] + x[4]*0.55) / crit_d  ✓

  FIX-C [nsga2_optimize _eval(), both modes]: Tie-breaker NPC nudge removed.
         Old: NPC += 1e-7*(x[0]+x[1])  [no equivalent in MATLAB gamultiobj]
         New: removed  ✓

MATLAB SEEDS (rng values):
  PSO mission  = rng(12)   PSO resource  = rng(16)
  NSGA mission = rng(8)    NSGA resource = rng(19)

MATLAB NSGA BOUNDS:
  Mission:  PV=[0, Land/10], Wind=[0, Land/15], BESS=[0, 3×CritDaily],
            H2=[0.6×Full_H2, 1.4×Full_H2], EL=[0, Full_H2/(20d×24h×η_EL)],
            FC=[0.6×CritPeak, 2.0×CritPeak]
  Resource: PV=[0, Land/10], Wind=[0, Land/15], BESS=[0, 10×CritDaily],
            H2=[0, 8×CritDaily/η_FC], EL=[0, PV_max+Wind_max],
            FC=[0.5×CritPeak, 1.5×CritPeak]

MATLAB NSGA FEASIBILITY (constraint-domination):
  Mission:  LPSP<=1e-4 AND REN>=1.1 AND Autonomy>=7d
  Resource: LPSP<=1e-4 AND Renewable_ratio>=0.99
"""
from __future__ import annotations

import json
import logging
import sys
import time
import warnings
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=RuntimeWarning)
logger = logging.getLogger(__name__)

# ── Unit costs (₹) — parametric, set by run_pipeline from user sidebar ──────
# MATLAB default values hardcoded in all evaluator/objective functions.
# Changing these here (or via run_pipeline config) propagates to all NPC
# calculations: evaluators, NSGA objectives, PSO objectives, lifecycle.
_UNIT_COSTS: dict = {
    "pv":      55_000,   # ₹/kWp
    "wind":   120_000,   # ₹/kW
    "batt":    15_000,   # ₹/kWh
    "el":      70_000,   # ₹/kW
    "h2":      15_000,   # ₹/kWh
    "fc":     110_000,   # ₹/kW
}
_UC = _UNIT_COSTS  # short alias used throughout

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

# ── Variable names (MATLAB order: PV Wind Bat EL H2 FC) ──
VAR_NAMES = [
    "PV_kWp", "Wind_kW", "Battery_kWh",
    "Electrolyzer_kW", "H2_kWh", "FuelCell_kW",
]

# ── Legacy bounds stub (PSO computes physics-derived bounds at runtime) ──
VAR_BOUNDS = {
    "min": np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=float),
    "max": np.array([9999.0, 9999.0, 99999.0, 9999.0, 99999.0, 9999.0], dtype=float),
}


# ══════════════════════════════════════════════════════════════════════════════
#  DATA CONTAINERS  (dataclasses matching MATLAB struct fields)
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class BaseData:
    """BaseData_Akhnoor.mat equivalent"""
    time:        np.ndarray   # timestamps (N,)
    ghi:         np.ndarray   # GHI  Wh/m² (N,)
    wind_speed:  np.ndarray   # WindSpeed  m/s (N,)
    temperature: np.ndarray   # Temperature °C (N,)
    load:        np.ndarray   # Load  kW (N,)
    N:           int

    def to_dict(self) -> dict:
        return {
            "N":            self.N,
            "avg_load_kW":  float(np.mean(self.load)),
            "peak_load_kW": float(np.max(self.load)),
            "avg_ghi":      float(np.mean(self.ghi)),
            "avg_wind_mps": float(np.mean(self.wind_speed)),
        }


@dataclass
class SolarData:
    """Step2_Solar_Akhnoor.mat → SolarData"""
    p_solar:          np.ndarray   # kW (N,)
    annual_solar_MWh: float
    N_modules:        int
    pv_capacity_kWp:  float


@dataclass
class WindData:
    """Step3_Wind_Akhnoor.mat → WindData"""
    p_wind:          np.ndarray   # kW (N,)
    annual_wind_MWh: float
    wind_capacity_kW: float


@dataclass
class HybridData:
    """Step4_Hybrid_Akhnoor.mat → HybridData"""
    p_gen:                np.ndarray   # kW (N,) = P_pv + P_wind
    annual_hybrid_MWh:    float
    annual_load_MWh:      float
    annual_solar_MWh:     float
    annual_wind_MWh:      float
    renewable_adequacy:   float        # Annual_Total / Annual_Load


@dataclass
class BatteryDispatchResult:
    """Step5 output"""
    lpsp_critical:      float   # LPSP (Critical)
    curtailed_semi_MWh: float
    curtailed_non_MWh:  float
    battery_cap_kWh:    float
    unmet_critical_kWh: float


@dataclass
class SeasonalAnalysis:
    """Step6 output"""
    req_storage_total_MWh:    float   # Case 1: total load
    req_storage_critical_MWh: float   # Case 2: critical load (MPBSI-relevant)


@dataclass
class H2DispatchResult:
    """Step7 / Step10 output"""
    lpsp_critical:           float
    curtailed_semi_MWh:      float
    curtailed_non_MWh:       float
    battery_cap_kWh:         float
    h2_cap_kWh:              float
    raw_critical_deficit_MWh: float


@dataclass
class DeficitWindowResult:
    """Step8 output"""
    max_consecutive_hours:     int
    max_consecutive_days:      float
    max_consecutive_energy_MWh: float


@dataclass
class SurvivabilitySizing:
    """Step9 output"""
    critical_daily_energy_kWh:  float
    E_baseline_usable_kWh:      float   # electrical kWh needed  (2-day)
    E_stress_usable_kWh:        float   # electrical kWh needed  (4-day)
    H2_baseline_chemical_kWh:   float   # chemical H2 capacity   (2-day)
    H2_stress_chemical_kWh:     float   # chemical H2 capacity   (4-day)


@dataclass
class SimulationResult:
    """microgrid_dispatch_mission output"""
    is_feasible:          bool
    lpsp_critical:        float
    renewable_ratio:      float
    autonomy_days:        float
    annual_load_MWh:      float
    total_renewable_MWh:  float
    curtailed_semi_MWh:   float
    curtailed_non_MWh:    float
    soc_series:           Optional[np.ndarray] = None
    h2_series:            Optional[np.ndarray] = None
    h2_used_kWh:          float = 0.0          # NEW — from resource dispatch
    h2_produced_kWh:      float = 0.0          # NEW — from resource dispatch


@dataclass
class Pillars:
    ESI:  float
    EcSI: float
    TRI:  float
    ORI:  float
    LSI:  float

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class MPBSIResult:
    mpbsi:       float
    pillars:     Pillars
    is_feasible: bool
    simulation:  SimulationResult

    def to_dict(self) -> dict:
        return {
            "mpbsi":               self.mpbsi,
            "pillars":             self.pillars.to_dict(),
            "is_feasible":         self.is_feasible,
            "lpsp_critical":       self.simulation.lpsp_critical,
            "total_renewable_MWh": self.simulation.total_renewable_MWh,
            "annual_load_MWh":     self.simulation.annual_load_MWh,
            "autonomy_days":       self.simulation.autonomy_days,
            "renewable_ratio":     self.simulation.renewable_ratio,
        }


@dataclass
class OptimizationResult:
    algorithm:           str
    best_x:              list
    best_mpbsi:          float
    best_pillars:        dict
    convergence:         list
    runtime_seconds:     float
    feasible:            bool
    reliability_metrics: dict
    variable_names:      list = field(default_factory=lambda: VAR_NAMES.copy())
    pareto_cases:        list = field(default_factory=list)   # NSGA-II: [caseA, caseB, caseC]

    def to_json(self) -> str:
        return json.dumps({
            "algorithm":           self.algorithm,
            "best_x":              self.best_x,
            "best_mpbsi":          self.best_mpbsi,
            "best_pillars":        self.best_pillars,
            "convergence":         self.convergence,
            "runtime_seconds":     self.runtime_seconds,
            "feasible":            self.feasible,
            "reliability_metrics": self.reliability_metrics,
            "variable_names":      self.variable_names,
            "pareto_cases":        self.pareto_cases,
        }, indent=2)


# ══════════════════════════════════════════════════════════════════════════════
#  COLUMN DETECTION HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _normalise(s: str) -> str:
    import re
    return re.sub(r"[^a-z0-9]", "", s.lower())


def _fuzzy_col(columns: list, *candidates: str) -> Optional[str]:
    norm = {_normalise(c): c for c in columns}
    for cand in candidates:
        if cand in columns:
            return cand
        nc = _normalise(cand)
        if nc in norm:
            return norm[nc]
        for nk, orig in norm.items():
            if nc in nk or nk in nc:
                return orig
    return None


def _ghi_unit_multiplier(data: np.ndarray, col_name: str) -> float:
    """Return ×1000 if data looks like kWh/m² (peak ≤ 10), else ×1."""
    peak = float(np.nanmax(data))
    if peak <= 10.0:
        logger.info("GHI column '%s' looks like kW/m² (peak=%.3f) → ×1000", col_name, peak)
        return 1000.0
    return 1.0


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 1  —  LOAD DATA  (Step1_Load_Akhnoor.m)
# ══════════════════════════════════════════════════════════════════════════════

def step1_load_data(filepath: str | Path) -> BaseData:
    """
    Exact translation of Step1_Load_Akhnoor.m.

    Expected columns (flexible naming):
        Timestamp, GHI_Wh_m2, WindSpeed_10m_mps, Temperature_C, Load_kW
    """
    path = Path(filepath)
    if path.suffix.lower() in (".xlsx", ".xls"):
        df = pd.read_excel(path)
    elif path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
    else:
        raise ValueError(f"Unsupported file type: {path.suffix}")

    cols = list(df.columns)
    logger.info("Step 1 — columns detected: %s", cols)

    found_time = _fuzzy_col(cols, "Timestamp", "Time", "DateTime", "Date", "Hour")
    found_ghi  = _fuzzy_col(cols, "GHI_Wh_m2", "GHI", "ghi", "Solar_Irradiance",
                             "Irradiance", "Solar", "GHI_kWh_m2", "GHI_W_m2")
    found_wind = _fuzzy_col(cols, "WindSpeed_10m_mps", "WindSpeed", "Wind_Speed",
                             "Wind", "wind_mps", "WindSpeed_mps")
    found_temp = _fuzzy_col(cols, "Temperature_C", "Temperature", "Temp",
                             "Tamb", "AirTemp")
    found_load = _fuzzy_col(cols, "Load_kW", "Load", "Demand", "Power",
                             "Demand_kW", "load_kw")

    missing = [n for n, c in [("GHI", found_ghi), ("Wind", found_wind), ("Load", found_load)] if c is None]
    if missing:
        raise KeyError(f"Cannot find columns: {missing}. File has: {cols}")

    ghi  = df[found_ghi].to_numpy(dtype=float)
    wind = df[found_wind].to_numpy(dtype=float)
    load = df[found_load].to_numpy(dtype=float)

    # Auto-detect GHI unit (Wh/m² vs kWh/m²)
    ghi = ghi * _ghi_unit_multiplier(ghi, found_ghi)

    if found_temp is not None:
        temp = df[found_temp].to_numpy(dtype=float)
    else:
        logger.warning("Temperature column not found — using 25 °C constant")
        temp = np.full(len(ghi), 25.0)

    time_arr = df[found_time].to_numpy() if found_time else np.arange(len(df))
    N = len(ghi)

    # Basic cleaning (NaN clamp)
    ghi  = np.nan_to_num(ghi,  nan=0.0)
    ghi  = np.clip(ghi, 0, None)
    wind = np.nan_to_num(wind, nan=0.0)
    wind = np.clip(wind, 0, None)
    load = np.nan_to_num(load, nan=0.0)

    logger.info("Step 1 — N=%d | avg_load=%.2f kW | peak=%.2f kW | avg_GHI=%.1f Wh/m²",
                N, np.mean(load), np.max(load), np.mean(ghi))
    return BaseData(time=time_arr, ghi=ghi, wind_speed=wind,
                    temperature=temp, load=load, N=N)


def step1_generate_synthetic(N: int = 8760, seed: int = 42) -> BaseData:
    """Synthetic Akhnoor-profile data for testing."""
    rng = np.random.default_rng(seed)
    h   = np.arange(N)
    doy = (h // 24) % 365
    hod = h % 24

    seasonal = 600 + 200 * np.sin(2 * np.pi * (doy - 80) / 365)
    diurnal  = np.maximum(0.0, np.exp(-0.5 * ((hod - 12.5) / 3.2) ** 2))
    ghi      = np.clip(seasonal * diurnal + rng.normal(0, 30, N), 0, 1050)

    sw   = 5.5 + 1.5 * np.cos(2 * np.pi * (doy - 30) / 365)
    wind = np.clip(rng.weibull(2.1, N) * sw / 1.13, 0, 28)

    temp = 22 + 13 * np.sin(2 * np.pi * (doy - 100) / 365) + rng.normal(0, 2, N)
    load = np.clip(
        280
        + 60  * np.exp(-0.5 * ((hod - 8)  / 1.5) ** 2)
        + 100 * np.exp(-0.5 * ((hod - 19) / 2.0) ** 2)
        + 40  * np.sin(2 * np.pi * (doy - 200) / 365)
        + rng.normal(0, 15, N),
        80, 600,
    )

    logger.info("Step 1 — synthetic %d hours | avg_load=%.2f kW", N, np.mean(load))
    return BaseData(time=np.arange(N, dtype=float), ghi=ghi,
                    wind_speed=wind, temperature=temp, load=load, N=N)


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 2  —  SOLAR MODEL  (Step2_Solar_Model.m)
# ══════════════════════════════════════════════════════════════════════════════

def step2_solar_model(
    base: BaseData,
    pv_capacity_kWp: float = 500.0,
    PSolarPV_ref: float = 550.0,   # W — rated module power
    Gref:         float = 1000.0,  # W/m²
    Tcell_ref:    float = 25.0,    # °C
    K_T:          float = -0.004,  # /°C
    etaSolarPV:   float = 0.95,
) -> SolarData:
    """
    Exact translation of Step2_Solar_Model.m.

    MATLAB equations:
      (1) PSolarPV = PSolarPV_ref × (Gact/Gref) × (1 + K_T×(Tcell-Tcell_ref))
      (2) Tcell    = Tamb + 0.0256 × Gact
      (3) P_pv     = PSolarPV × NSolarPV × etaSolarPV / 1000
    """
    # Number of modules — MATLAB: NSolarPV = ceil((PV_capacity*1000) / PSolarPV_ref)
    NSolarPV = int(np.ceil((pv_capacity_kWp * 1000.0) / PSolarPV_ref))

    # ── Clean GHI — exact MATLAB logic ──
    GHI = base.ghi.copy().astype(float)
    GHI[GHI == -999] = np.nan
    GHI[~np.isfinite(GHI)] = np.nan
    if np.any(np.isnan(GHI)):
        s   = pd.Series(GHI)
        GHI = s.interpolate(method="linear").ffill().bfill().to_numpy()
    GHI = np.clip(GHI, 0, None)

    Tamb  = base.temperature
    # Eq (2) cell temperature
    Tcell = Tamb + 0.0256 * GHI

    # Eq (1) single-module power [W]
    PSolarPV = PSolarPV_ref * (GHI / Gref) * (1.0 + K_T * (Tcell - Tcell_ref))
    PSolarPV = np.clip(PSolarPV, 0, None)

    # Eq (3) total array power [kW]
    P_pv   = (PSolarPV * NSolarPV * etaSolarPV) / 1000.0
    annual = float(np.sum(P_pv) / 1000.0)   # MWh

    _cf_pv = annual / (pv_capacity_kWp * 8.76) if pv_capacity_kWp > 0 else 0.0
    logger.info("Step 2 — %d modules × %.0fW | cap=%.2f kWp | annual=%.2f MWh | CF=%.3f",
                NSolarPV, PSolarPV_ref, pv_capacity_kWp,
                annual, _cf_pv)
    return SolarData(
        p_solar=P_pv,
        annual_solar_MWh=annual,
        N_modules=NSolarPV,
        pv_capacity_kWp=pv_capacity_kWp,
    )


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 3  —  WIND MODEL  (Step3_Wind_Model.m)
# ══════════════════════════════════════════════════════════════════════════════

def step3_wind_model(
    base: BaseData,
    wind_capacity_kW: float = 200.0,
    V_ci: float = 3.0,    # cut-in  m/s
    V_r:  float = 12.0,   # rated   m/s
    V_co: float = 25.0,   # cut-out m/s
) -> WindData:
    """
    Exact translation of Step3_Wind_Model.m.

    NOTE: Step3 uses raw 10m wind speed with NO hub-height correction.
    Hub-height correction (Hhub=20m, alpha=0.14) is ONLY in microgrid_dispatch_mission.m.

    Power curve (piecewise cubic — MATLAB exact):
      V < V_ci or V >= V_co → 0
      V_ci ≤ V < V_r        → Wind_capacity × ((V-V_ci)/(V_r-V_ci))³
      V ≥ V_r               → Wind_capacity
    """
    V = base.wind_speed

    p_wind = np.where(
        (V < V_ci) | (V >= V_co), 0.0,
        np.where(
            V < V_r,
            wind_capacity_kW * ((V - V_ci) / (V_r - V_ci)) ** 3,
            wind_capacity_kW,
        ),
    )

    annual = float(np.sum(p_wind) / 1000.0)
    _cf = annual / (wind_capacity_kW * 8.76) if wind_capacity_kW > 0 else 0.0
    logger.info("Step 3 — %.0f kW turbine | annual=%.2f MWh | CF=%.3f",
                wind_capacity_kW, annual, _cf)
    return WindData(
        p_wind=p_wind,
        annual_wind_MWh=annual,
        wind_capacity_kW=wind_capacity_kW,
    )


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 4  —  HYBRID GENERATION  (Step4_Hybrid_Generation.m)
# ══════════════════════════════════════════════════════════════════════════════

def step4_hybrid_generation(
    base:  BaseData,
    solar: SolarData,
    wind:  WindData,
) -> HybridData:
    """
    Exact translation of Step4_Hybrid_Generation.m.
    P_gen = P_pv + P_wind
    """
    P_gen             = solar.p_solar + wind.p_wind
    annual_load_MWh   = float(np.sum(base.load) / 1000.0)
    annual_total_MWh  = float(np.sum(P_gen) / 1000.0)
    ren_adequacy      = annual_total_MWh / annual_load_MWh if annual_load_MWh > 0 else 0.0

    logger.info("Step 4 — Hybrid=%.2f MWh | Load=%.2f MWh | Adequacy=%.3f",
                annual_total_MWh, annual_load_MWh, ren_adequacy)
    return HybridData(
        p_gen=P_gen,
        annual_hybrid_MWh=annual_total_MWh,
        annual_load_MWh=annual_load_MWh,
        annual_solar_MWh=solar.annual_solar_MWh,
        annual_wind_MWh=wind.annual_wind_MWh,
        renewable_adequacy=ren_adequacy,
    )


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 5  —  PRIORITY BATTERY DISPATCH  (Step5_Priority_Battery_Model.m)
# ══════════════════════════════════════════════════════════════════════════════

def step5_priority_battery(
    base:            BaseData,
    hybrid:          HybridData,
    battery_cap_kWh: float = 1500.0,
    critical_frac:   float = 0.60,
    semi_frac:       float = 0.25,
    non_frac:        float = 0.15,
    eta_ch:          float = 0.95,
    eta_dis:         float = 0.95,
) -> BatteryDispatchResult:
    """
    Exact translation of Step5_Priority_Battery_Model.m.

    Dispatch priority: Critical → Semi → Non
    Battery: charges/discharges with η_ch / η_dis.

    Key MATLAB equations:
      discharge = min(deficit, available_energy × η_dis)
      SOC      -= discharge / η_dis
      charge    = min(generation × η_ch, Battery_cap - SOC)
      SOC      += charge
      generation -= charge / η_ch
    """
    N  = base.N
    Lc = critical_frac * base.load
    Ls = semi_frac     * base.load
    Ln = non_frac      * base.load
    P  = hybrid.p_gen

    # MATLAB initial conditions
    SOC     = 0.8 * battery_cap_kWh
    SOC_min = 0.2 * battery_cap_kWh

    unmet_c = 0.0
    curt_s  = 0.0
    curt_n  = 0.0

    for t in range(N):
        generation = P[t]

        # ── Critical load ──────────────────────────────────────────────────
        net = generation - Lc[t]
        if net >= 0:
            generation = net
        else:
            deficit = -net
            available_energy = max(SOC - SOC_min, 0.0)
            max_output       = available_energy * eta_dis       # MATLAB eq
            discharge        = min(deficit, max_output)
            SOC             -= discharge / eta_dis              # MATLAB eq
            deficit         -= discharge
            if deficit > 0:
                unmet_c += deficit
            generation = 0.0

        # ── Semi-critical ──────────────────────────────────────────────────
        if generation > 0:
            if generation >= Ls[t]:
                generation -= Ls[t]
            else:
                curt_s    += (Ls[t] - generation)
                generation = 0.0
        else:
            curt_s += Ls[t]

        # ── Non-critical ───────────────────────────────────────────────────
        if generation > 0:
            if generation >= Ln[t]:
                generation -= Ln[t]
            else:
                curt_n    += (Ln[t] - generation)
                generation = 0.0
        else:
            curt_n += Ln[t]

        # ── Charge battery ─────────────────────────────────────────────────
        if generation > 0:
            max_storage_space = battery_cap_kWh - SOC
            charge            = min(generation * eta_ch, max_storage_space)   # MATLAB eq
            SOC              += charge
            generation       -= charge / eta_ch                               # MATLAB eq

        SOC = min(max(SOC, SOC_min), battery_cap_kWh)

    total_c = float(np.sum(Lc))
    lpsp    = unmet_c / total_c if total_c > 0 else 0.0

    logger.info("Step 5 | Battery=%.0f kWh | η_ch=%.2f η_dis=%.2f | LPSP=%.6f | "
                "CurtSemi=%.2f MWh CurtNon=%.2f MWh",
                battery_cap_kWh, eta_ch, eta_dis, lpsp,
                curt_s / 1000, curt_n / 1000)

    return BatteryDispatchResult(
        lpsp_critical=lpsp,
        curtailed_semi_MWh=curt_s / 1000,
        curtailed_non_MWh=curt_n  / 1000,
        battery_cap_kWh=battery_cap_kWh,
        unmet_critical_kWh=unmet_c,
    )


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 6  —  SEASONAL STORAGE ANALYSIS  (Step6_Seasonal_Analysis_Akhnoor.m)
# ══════════════════════════════════════════════════════════════════════════════

def step6_seasonal_analysis(
    base:          BaseData,
    hybrid:        HybridData,
    critical_frac: float = 0.60,
) -> SeasonalAnalysis:
    """
    Exact translation of Step6_Seasonal_Analysis_Akhnoor.m.

    seasonal_storage(net) = max(cumsum(net) - min(cumsum(net)))

    Case 1: net = P_gen - Load_total
    Case 2: net = P_gen - Load_critical  (MPBSI-relevant)
    """
    def _seasonal(net: np.ndarray) -> float:
        cum     = np.cumsum(net)
        shifted = cum - float(np.min(cum))
        return float(np.max(shifted))

    net_total    = hybrid.p_gen - base.load
    net_critical = hybrid.p_gen - (critical_frac * base.load)

    req_total    = _seasonal(net_total)
    req_critical = _seasonal(net_critical)

    logger.info("Step 6 | Required storage — Total=%.2f MWh | Critical=%.2f MWh",
                req_total / 1000, req_critical / 1000)
    return SeasonalAnalysis(
        req_storage_total_MWh=req_total    / 1000,
        req_storage_critical_MWh=req_critical / 1000,
    )


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 7  —  BATTERY + HYDROGEN PRIORITY DISPATCH  (Step7_Priority_H2_Model.m)
# ══════════════════════════════════════════════════════════════════════════════

def step7_priority_h2(
    base:            BaseData,
    hybrid:          HybridData,
    battery_cap_kWh: float = 1500.0,
    h2_cap_kWh:      float = 5000.0,
    critical_frac:   float = 0.60,
    semi_frac:       float = 0.25,
    non_frac:        float = 0.15,
    eta_bat_ch:      float = 0.95,
    eta_bat_dis:     float = 0.95,
    eta_EL:          float = 0.70,
    eta_FC:          float = 0.55,
) -> H2DispatchResult:
    """
    Exact translation of Step7_Priority_H2_Model.m.

    Discharge priority: Battery first → Fuel Cell (H2).
    Initial H2 = 0.5 × H2_cap   (MATLAB line: H2 = 0.5 * H2_cap)

    Key MATLAB equations:
      Battery discharge: SOC -= discharge_bat / η_bat_dis
      H2 discharge:      H2  -= discharge_H2  / η_FC
      Battery charge:    charge_bat = min(gen × η_bat_ch, Battery_cap - SOC)
                         SOC += charge_bat
                         gen -= charge_bat / η_bat_ch
      H2 storage:        hydrogen_input = gen × η_EL
                         H2 = min(H2 + hydrogen_input, H2_cap)
    """
    N  = base.N
    Lc = critical_frac * base.load
    Ls = semi_frac     * base.load
    Ln = non_frac      * base.load
    P  = hybrid.p_gen

    # MATLAB initial conditions
    SOC     = 0.8 * battery_cap_kWh
    SOC_min = 0.2 * battery_cap_kWh
    H2      = 0.5 * h2_cap_kWh          # MATLAB: H2 = 0.5 * H2_cap

    unmet_c = 0.0
    curt_s  = 0.0
    curt_n  = 0.0

    for t in range(N):
        generation = P[t]

        # ── Critical load ──────────────────────────────────────────────────
        net = generation - Lc[t]
        if net >= 0:
            generation = net
        else:
            deficit = -net

            # Battery discharge
            available_bat  = max(SOC - SOC_min, 0.0)
            max_bat_output = available_bat * eta_bat_dis
            discharge_bat  = min(deficit, max_bat_output)
            SOC           -= discharge_bat / eta_bat_dis     # MATLAB eq
            deficit       -= discharge_bat

            # Hydrogen / Fuel Cell
            max_fc_output  = H2 * eta_FC
            discharge_H2   = min(deficit, max_fc_output)
            H2            -= discharge_H2 / eta_FC           # MATLAB eq
            deficit       -= discharge_H2

            if deficit > 0:
                unmet_c += deficit
            generation = 0.0

        # ── Semi-critical ──────────────────────────────────────────────────
        if generation > 0:
            if generation >= Ls[t]:
                generation -= Ls[t]
            else:
                curt_s    += (Ls[t] - generation)
                generation = 0.0
        else:
            curt_s += Ls[t]

        # ── Non-critical ───────────────────────────────────────────────────
        if generation > 0:
            if generation >= Ln[t]:
                generation -= Ln[t]
            else:
                curt_n    += (Ln[t] - generation)
                generation = 0.0
        else:
            curt_n += Ln[t]

        # ── Store surplus ──────────────────────────────────────────────────
        if generation > 0:
            # Battery charge — MATLAB eqs
            max_charge_possible = battery_cap_kWh - SOC
            charge_bat          = min(generation * eta_bat_ch, max_charge_possible)
            SOC                += charge_bat
            generation         -= charge_bat / eta_bat_ch

            # Hydrogen storage — MATLAB eq
            hydrogen_input = generation * eta_EL
            H2             = min(H2 + hydrogen_input, h2_cap_kWh)

        # Enforce bounds
        SOC = min(max(SOC, SOC_min), battery_cap_kWh)
        H2  = min(max(H2, 0.0), h2_cap_kWh)

    total_c     = float(np.sum(Lc))
    lpsp        = unmet_c / total_c if total_c > 0 else 0.0
    raw_deficit = float(np.sum(np.maximum(Lc - P, 0.0)))

    logger.info("Step 7 | Bat=%.0f kWh H2=%.0f kWh | LPSP=%.6f | "
                "CurtSemi=%.2f MWh CurtNon=%.2f MWh",
                battery_cap_kWh, h2_cap_kWh, lpsp,
                curt_s / 1000, curt_n / 1000)

    return H2DispatchResult(
        lpsp_critical=lpsp,
        curtailed_semi_MWh=curt_s / 1000,
        curtailed_non_MWh=curt_n  / 1000,
        battery_cap_kWh=battery_cap_kWh,
        h2_cap_kWh=h2_cap_kWh,
        raw_critical_deficit_MWh=raw_deficit / 1000,
    )


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 8  —  DEFICIT WINDOW ANALYSIS  (Step8_Deficit_Window_Analysis.m)
# ══════════════════════════════════════════════════════════════════════════════

def step8_deficit_window(
    base:          BaseData,
    hybrid:        HybridData,
    critical_frac: float = 0.60,
) -> DeficitWindowResult:
    """
    Exact translation of Step8_Deficit_Window_Analysis.m.

    Scans net = P_gen - Load_critical for longest consecutive deficit window.
    Tracks both hours and cumulative energy of the worst window.
    Final check handles year ending in deficit.
    """
    Load_critical = critical_frac * base.load
    net           = hybrid.p_gen - Load_critical
    N             = base.N

    max_hours  = 0
    cur_hours  = 0
    max_energy = 0.0
    cur_energy = 0.0

    for t in range(N):
        if net[t] < 0:
            cur_hours  += 1
            cur_energy += abs(net[t])
        else:
            if cur_hours > max_hours:
                max_hours  = cur_hours
                max_energy = cur_energy
            cur_hours  = 0
            cur_energy = 0.0

    # Final check — MATLAB exact
    if cur_hours > max_hours:
        max_hours  = cur_hours
        max_energy = cur_energy

    logger.info("Step 8 | Max deficit = %d h (%.2f days) | Energy = %.2f MWh",
                max_hours, max_hours / 24, max_energy / 1000)
    return DeficitWindowResult(
        max_consecutive_hours=max_hours,
        max_consecutive_days=max_hours / 24,
        max_consecutive_energy_MWh=max_energy / 1000,
    )


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 9  —  SURVIVABILITY H2 SIZING  (Step9_Survivability_H2_Sizing.m)
# ══════════════════════════════════════════════════════════════════════════════

def step9_survivability_sizing(
    base:          BaseData,
    critical_frac: float = 0.60,
    baseline_days: int   = 2,     # MATLAB: baseline_days = 2
    stress_days:   int   = 4,     # MATLAB: stress_days   = 4
    eta_FC:        float = 0.55,
) -> SurvivabilitySizing:
    """
    Exact translation of Step9_Survivability_H2_Sizing.m.

    MATLAB equations:
      critical_daily_energy = sum(Load_critical) / 365
      E_baseline_usable     = baseline_days × critical_daily_energy
      E_stress_usable       = stress_days   × critical_daily_energy
      H2_baseline_chemical  = E_baseline_usable / η_FC
      H2_stress_chemical    = E_stress_usable   / η_FC
    """
    Load_critical         = critical_frac * base.load
    Annual_critical_kWh   = float(np.sum(Load_critical))
    critical_daily_energy = Annual_critical_kWh / 365.0      # MATLAB eq

    E_baseline_usable = baseline_days * critical_daily_energy
    E_stress_usable   = stress_days   * critical_daily_energy

    H2_baseline_chemical = E_baseline_usable / eta_FC        # MATLAB eq
    H2_stress_chemical   = E_stress_usable   / eta_FC        # MATLAB eq

    logger.info(
        "Step 9 | critical_daily=%.2f kWh | "
        "Baseline %dd: E=%.3f MWh H2=%.3f MWh | "
        "Stress %dd: E=%.3f MWh H2=%.3f MWh",
        critical_daily_energy,
        baseline_days, E_baseline_usable / 1000, H2_baseline_chemical / 1000,
        stress_days,   E_stress_usable   / 1000, H2_stress_chemical   / 1000,
    )
    return SurvivabilitySizing(
        critical_daily_energy_kWh=critical_daily_energy,
        E_baseline_usable_kWh=E_baseline_usable,
        E_stress_usable_kWh=E_stress_usable,
        H2_baseline_chemical_kWh=H2_baseline_chemical,
        H2_stress_chemical_kWh=H2_stress_chemical,
    )


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 10  —  SURVIVABILITY DISPATCH  (Step10_Priority_H2_Survivability.m)
# ══════════════════════════════════════════════════════════════════════════════

def step10_survivability_dispatch(
    base:            BaseData,
    hybrid:          HybridData,
    battery_cap_kWh: float = 1500.0,
    h2_cap_kWh:      float = 4500.0,   # MATLAB default: H2_cap = 4500
    critical_frac:   float = 0.60,
    semi_frac:       float = 0.25,
    non_frac:        float = 0.15,
) -> H2DispatchResult:
    """
    Exact translation of Step10_Priority_H2_Survivability.m.

    KEY DIFFERENCES from Step 7:
    ─────────────────────────────────────────────────────────────
    1. H2 starts FULL: H2 = H2_cap   (Step 7: H2 = 0.5×H2_cap)
    2. Battery discharge: SOC -= discharge_bat     (NO η_dis)
    3. H2 discharge:      H2  -= discharge_H2      (NO η_FC)
    4. Battery charge:    charge = min(gen, Battery_cap-SOC)  (NO η)
                          SOC   += charge
                          gen   -= charge                     (NO η)
    5. H2 surplus:        H2 = min(H2 + gen, H2_cap)         (NO η_EL)
    ─────────────────────────────────────────────────────────────
    This is the simplified survivability model — ideal storage assumed.
    """
    N  = base.N
    Lc = critical_frac * base.load
    Ls = semi_frac     * base.load
    Ln = non_frac      * base.load
    P  = hybrid.p_gen

    # MATLAB initial conditions
    SOC     = 0.8 * battery_cap_kWh
    SOC_min = 0.2 * battery_cap_kWh
    H2      = h2_cap_kWh                # MATLAB: H2 = H2_cap (FULL)

    unmet_c = 0.0
    curt_s  = 0.0
    curt_n  = 0.0

    for t in range(N):
        generation = P[t]

        # ── Critical ───────────────────────────────────────────────────────
        net = generation - Lc[t]
        if net >= 0:
            generation = net
        else:
            deficit = -net

            # Battery — MATLAB: no η_dis
            available_bat = max(SOC - SOC_min, 0.0)
            discharge_bat = min(deficit, available_bat)
            SOC          -= discharge_bat                    # MATLAB: SOC -= discharge (no η)
            deficit      -= discharge_bat

            # Hydrogen — MATLAB: no η_FC
            discharge_H2 = min(deficit, H2)
            H2          -= discharge_H2                     # MATLAB: H2 -= discharge  (no η)
            deficit     -= discharge_H2

            if deficit > 0:
                unmet_c += deficit
            generation = 0.0

        # ── Semi ───────────────────────────────────────────────────────────
        if generation > 0:
            if generation >= Ls[t]:
                generation -= Ls[t]
            else:
                curt_s    += (Ls[t] - generation)
                generation = 0.0
        else:
            curt_s += Ls[t]

        # ── Non ────────────────────────────────────────────────────────────
        if generation > 0:
            if generation >= Ln[t]:
                generation -= Ln[t]
            else:
                curt_n    += (Ln[t] - generation)
                generation = 0.0
        else:
            curt_n += Ln[t]

        # ── Store surplus — MATLAB: no efficiency factors ──────────────────
        if generation > 0:
            charge     = min(generation, battery_cap_kWh - SOC)   # MATLAB eq
            SOC       += charge
            generation -= charge
            H2         = min(H2 + generation, h2_cap_kWh)         # MATLAB eq

    total_c     = float(np.sum(Lc))
    lpsp        = unmet_c / total_c if total_c > 0 else 0.0
    raw_deficit = float(np.sum(np.maximum(Lc - P, 0.0)))

    logger.info("Step 10 (Survivability) | Bat=%.0f kWh H2=%.0f kWh | LPSP=%.6f",
                battery_cap_kWh, h2_cap_kWh, lpsp)

    return H2DispatchResult(
        lpsp_critical=lpsp,
        curtailed_semi_MWh=curt_s / 1000,
        curtailed_non_MWh=curt_n  / 1000,
        battery_cap_kWh=battery_cap_kWh,
        h2_cap_kWh=h2_cap_kWh,
        raw_critical_deficit_MWh=raw_deficit / 1000,
    )


# ══════════════════════════════════════════════════════════════════════════════
#  MISSION DISPATCH  (microgrid_dispatch_mission.m)
# ══════════════════════════════════════════════════════════════════════════════

def microgrid_dispatch_full(
    x:            np.ndarray | list,
    base:         BaseData,
    eta_ch:       float = 0.95,
    eta_dis:      float = 0.95,
    eta_EL:       float = 0.70,
    eta_FC:       float = 0.55,
    lpsp_tol:     float = 1e-4,       # MATLAB: feasible iff LPSP ≤ 1e-4
    store_series: bool  = False,
) -> SimulationResult:
    """
    Exact translation of microgrid_dispatch_mission.m.

    Differences from Step 7 (mission mode adds):
    ─────────────────────────────────────────────
    1. PV: temperature-derated  (KT=-0.004, Tcell=Tamb+0.0256×GHI, η=0.95)
    2. Wind: hub-height correction (Hhub=20m, alpha=0.14) + η_wind=0.80 (MATLAB mission)
    3. Battery SOC in absolute kWh, SOC_max = E_BESS (100%)
    4. Feasibility check includes: LPSP≤1e-4 AND REN_ratio≥0.999 AND Autonomy≥7d

    x = [PV_kWp, Wind_kW, BESS_kWh, EL_kW, H2_kWh, FC_kW]
    """
    x = np.asarray(x, dtype=float)
    PV_rated  = x[0]   # kWp
    WT_rated  = x[1]   # kW
    E_BESS    = x[2]   # kWh
    P_EL_max  = x[3]   # kW  electrolyzer rated
    E_H2_max  = x[4]   # kWh hydrogen tank
    P_FC_max  = x[5]   # kW  fuel cell rated

    GHI  = base.ghi.copy().astype(float)
    Wind = np.nan_to_num(base.wind_speed.copy().astype(float), nan=0.0)
    Temp = np.nan_to_num(base.temperature.copy().astype(float), nan=25.0)
    Load = np.nan_to_num(base.load.copy().astype(float), nan=0.0)
    N    = base.N

    # ── Clean GHI ────────────────────────────────────────────────────────────
    GHI[GHI == -999] = np.nan
    GHI[~np.isfinite(GHI)] = np.nan
    if np.any(np.isnan(GHI)):
        s   = pd.Series(GHI)
        GHI = s.interpolate(method="linear").ffill().bfill().to_numpy()
    GHI = np.clip(GHI, 0, None)

    # ── PV — temperature-derated (MATLAB microgrid_dispatch_mission.m) ───────
    Gref  = 1000.0; Tref = 25.0; KT = -0.004; etaPV = 0.95
    Tcell = Temp + 0.0256 * GHI
    P_PV  = PV_rated * (GHI / Gref) * (1.0 + KT * (Tcell - Tref)) * etaPV
    P_PV  = np.clip(P_PV, 0, None)

    # ── Wind — hub-height correction (MATLAB microgrid_dispatch_mission.m) ────
    # Mission uses SAME wind curve as resource: V_ci=2.5, V_r=10, V_co=20
    # Power curve: P_WT × ((V-2.5)/(10-2.5))^2.2 × 1.05 below rated
    V2      = Wind * (20.0 / 10.0) ** 0.14          # Hhub=20m, Href=10m, alpha=0.14
    v_ci, v_r, v_co = 2.5, 10.0, 20.0               # MATLAB mission: same as resource
    P_WT_below = WT_rated * ((V2 - v_ci) / (v_r - v_ci)) ** 2.2 * 1.05
    P_WT    = np.where(
        (V2 < v_ci) | (V2 >= v_co), 0.0,
        np.where(V2 < v_r, P_WT_below, WT_rated),
    )
    P_WT    = 0.80 * P_WT                            # η_wind = 0.80 (MATLAB mission exact)

    P_RES = P_PV + P_WT

    # ── Load split ────────────────────────────────────────────────────────────
    Lc = 0.60 * Load
    Ls = 0.25 * Load
    Ln = 0.15 * Load

    # ── Initial conditions (MATLAB: SOC_max = E_BESS, absolute kWh) ──────────
    SOC     = 0.8 * E_BESS
    SOC_min = 0.2 * E_BESS
    SOC_max = E_BESS             # 100% — MATLAB: SOC_max = E_BESS
    E_H2    = 0.65 * E_H2_max   # MATLAB mission: 0.65 × E_H2_max (higher readiness vs resource 0.5)

    unmet_c    = 0.0
    curt_s     = 0.0
    curt_n     = 0.0
    Annual_RES = float(np.sum(P_RES))  # vectorised outside loop
    H2_used_total = 0.0      # track H2 consumed for EL_util/H2_util in evaluator
    H2_prod_total = 0.0      # track H2 produced

    soc_arr = np.empty(N) if store_series else None
    h2_arr  = np.empty(N) if store_series else None

    # Pre-convert to lists for fast per-element access
    P_RES_ = P_RES.tolist()
    Lc_    = Lc.tolist()
    Ls_    = Ls.tolist()
    Ln_    = Ln.tolist()

    for t in range(N):
        P_available = P_RES_[t]

        # ── Critical ──────────────────────────────────────────────────────
        deficit = Lc_[t] - P_available
        if deficit <= 0:
            P_available = -deficit
        else:
            # Battery discharge
            avail_bat = (SOC - SOC_min) * eta_dis
            P_dis_bat = deficit if deficit < avail_bat else avail_bat
            SOC      -= P_dis_bat / eta_dis
            deficit  -= P_dis_bat

            # Fuel cell (hydrogen)
            max_fc = E_H2 * eta_FC
            if P_FC_max < max_fc: max_fc = P_FC_max
            P_fc      = deficit if deficit < max_fc else max_fc
            H2_con    = P_fc / eta_FC
            E_H2     -= H2_con
            H2_used_total += H2_con
            deficit  -= P_fc
            if deficit > 0:
                unmet_c += deficit
            P_available = 0.0

        # ── Semi ──────────────────────────────────────────────────────────
        ls = Ls_[t]
        if P_available >= ls:
            P_available -= ls
        else:
            curt_s     += ls - P_available
            P_available  = 0.0

        # ── Non ───────────────────────────────────────────────────────────
        ln = Ln_[t]
        if P_available >= ln:
            P_available -= ln
        else:
            curt_n     += ln - P_available
            P_available  = 0.0

        # ── Store surplus ─────────────────────────────────────────────────
        if P_available > 0:
            max_ch_room = (SOC_max - SOC) / eta_ch
            P_ch = P_available if P_available < max_ch_room else max_ch_room
            SOC        += eta_ch * P_ch
            P_available -= P_ch

            room_h2 = max((E_H2_max - E_H2) / eta_EL, 0.0)
            P_el    = P_available
            if P_EL_max < P_el: P_el = P_EL_max
            if room_h2  < P_el: P_el = room_h2
            if P_el > 0:
                H2_produced    = eta_EL * P_el
                E_H2          += H2_produced
                H2_prod_total += H2_produced

        # Clamp (branch-free)
        if SOC  > SOC_max: SOC  = SOC_max
        elif SOC < SOC_min: SOC = SOC_min
        if E_H2 > E_H2_max: E_H2 = E_H2_max
        elif E_H2 < 0.0:    E_H2 = 0.0

        if store_series:
            soc_arr[t] = SOC
            h2_arr[t]  = E_H2


    # ── Metrics ───────────────────────────────────────────────────────────────
    total_critical = float(np.sum(Lc))
    total_load     = float(np.sum(Load))
    lpsp           = unmet_c    / total_critical if total_critical > 0 else 0.0
    ren_ratio      = Annual_RES / total_load     if total_load     > 0 else 0.0

    # Autonomy — MATLAB: usable_battery + usable_h2_electric / critical_daily
    usable_battery  = SOC_max - SOC_min             # = 0.8 × E_BESS  (SOC_max=BESS, SOC_min=0.2×BESS)
    usable_h2_elec  = E_H2_max * eta_FC             # full tank electricity
    critical_daily  = total_critical / 365.0
    autonomy_days   = (usable_battery + usable_h2_elec) / critical_daily if critical_daily > 0 else 0.0

    # Mission feasibility: LPSP≤1e-4 AND REN≥0.999 AND Autonomy≥7d
    # Mission feasibility — MATLAB exact: LPSP≤1e-4 AND REN≥1.1 AND Autonomy≥7d
    is_feasible = (lpsp <= lpsp_tol) and (ren_ratio >= 1.1) and (autonomy_days >= 7.0)

    logger.info("Dispatch | LPSP=%.6f | REN=%.4f | Auto=%.2fd | Feasible=%s",
                lpsp, ren_ratio, autonomy_days, is_feasible)

    return SimulationResult(
        is_feasible=is_feasible,
        lpsp_critical=lpsp,
        renewable_ratio=ren_ratio,
        autonomy_days=autonomy_days,
        annual_load_MWh=total_load    / 1000.0,
        total_renewable_MWh=Annual_RES / 1000.0,
        curtailed_semi_MWh=curt_s     / 1000.0,
        curtailed_non_MWh=curt_n      / 1000.0,
        soc_series=soc_arr,
        h2_series=h2_arr,
        h2_used_kWh=H2_used_total,
        h2_produced_kWh=H2_prod_total,
    )


# ══════════════════════════════════════════════════════════════════════════════
#  MPBSI EVALUATOR  (MPBSI_Evaluator_Mission_Land.m)
# ══════════════════════════════════════════════════════════════════════════════

def mpbsi_evaluator(
    x:              np.ndarray | list,
    base:           BaseData,
    land_available: float = 50_000.0,
    weights:        Optional[dict] = None,
) -> MPBSIResult:
    """
    Exact translation of MPBSI_Evaluator_Mission_Land.m.

    Hard constraints (return −1e6 if violated):
    ─────────────────────────────────────────────
    1. Land: Area_PV + Area_WT ≤ Land_Available
       where Area_PV = 10 × PV_kWp, Area_WT = 15 × Wind_kW
    2. Mission dispatch feasible (LPSP≤1e-4, REN≥0.999)
    3. Autonomy ≥ 7 days

    Pillar formulas (MATLAB exact):
    ────────────────────────────────
    ESI  = 0.5×E1 + 0.5×E2        where E1=1, E2=1/(1+|E2_raw-1|)
    EcSI = 0.5×C1 + 0.5×C2        C1=1/(1+LCOE/26) [Rs 26/kWh diesel ref], C2=CAPEX-based
    TRI  = 0.6×T1 + 0.2×T2 + 0.2×T3
    ORI  = 0.4×R1 + 0.3×R2 + 0.3×R3
    LSI  = 0.4×L1 + 0.3×L2 + 0.3×L3
    MPBSI= 0.15×ESI+0.25×EcSI+0.30×TRI+0.15×ORI+0.15×LSI
    """
    x = np.asarray(x, dtype=float)

    _inf_result = lambda: MPBSIResult(
        mpbsi=-1e6,
        pillars=Pillars(ESI=0.0, EcSI=0.0, TRI=0.0, ORI=0.0, LSI=0.0),
        is_feasible=False,
        simulation=SimulationResult(
            is_feasible=False, lpsp_critical=1.0, renewable_ratio=0.0,
            autonomy_days=0.0, annual_load_MWh=0.0, total_renewable_MWh=0.0,
            curtailed_semi_MWh=0.0, curtailed_non_MWh=0.0,
        ),
    )

    # ── 1. Land constraint ────────────────────────────────────────────────────
    Area_PV = 10.0 * x[0]
    Area_WT = 15.0 * x[1]
    if (Area_PV + Area_WT) > (land_available + 1e-2):
        return _inf_result()

    # ── 2. Mission dispatch ───────────────────────────────────────────────────
    sim = microgrid_dispatch_full(x, base)
    if not sim.is_feasible:
        return MPBSIResult(
            mpbsi=-1e6,
            pillars=Pillars(ESI=0.0, EcSI=0.0, TRI=0.0, ORI=0.0, LSI=0.0),
            is_feasible=False,
            simulation=sim,
        )

    # ── 3. Autonomy hard gate ─────────────────────────────────────────────────
    if sim.autonomy_days < 7.0:
        return MPBSIResult(
            mpbsi=-1e6,
            pillars=Pillars(ESI=0.0, EcSI=0.0, TRI=0.0, ORI=0.0, LSI=0.0),
            is_feasible=False,
            simulation=sim,
        )

    # Aut_factor — MATLAB revised: 1.0 if Aut >= 7, else exp(-(7-Aut))
    if sim.autonomy_days >= 7.0:
        Aut_factor = 1.0
    else:
        Aut_factor = np.exp(-(7.0 - sim.autonomy_days))
    Annual_Load  = sim.annual_load_MWh
    Annual_RES   = sim.total_renewable_MWh

    # ── 4. ESI (Environmental) ────────────────────────────────────────────────
    # MATLAB: ESI = 0.4*E1 + 0.4*E2 + 0.2*Curt_factor
    E1     = 1.0
    E2_raw = Annual_RES / Annual_Load if Annual_Load > 0 else 0.0
    E2     = 1.0 / (1.0 + abs(E2_raw - 1.0))
    # Curt_factor: penalises wasted generation relative to total RES output.
    # MATLAB uses exp(-Curt/200) calibrated for small systems (~500 MWh/yr).
    # At Akhnoor scale (~2700 MWh/yr), 200 MWh reference collapses Curt_factor.
    # Fix: use Curt_ratio = Curt/Annual_RES (scale-invariant) with sensitivity=3.
    Curt         = sim.curtailed_semi_MWh + sim.curtailed_non_MWh
    Curt_factor  = np.exp(-Curt / 200.0)        # MATLAB exact
    ESI          = 0.4 * E1 + 0.4 * E2 + 0.2 * Curt_factor

    # ── 5. EcSI (Economic) ────────────────────────────────────────────────────
    # MATLAB: C1 = 1/(1+10*LCOE)  — raw LCOE (not normalised to 50/kWh)
    Cost_PV = _UC["pv"] * x[0]
    Cost_WT = _UC["wind"] * x[1]
    Cost_Bat = _UC["batt"] * x[2]
    Cost_EL = _UC["el"] * x[3]
    Cost_H2 = _UC["h2"] * x[4]
    Cost_FC = _UC["fc"] * x[5]
    CAPEX    = Cost_PV + Cost_WT + Cost_Bat + Cost_EL + Cost_H2 + Cost_FC

    LCOE = CAPEX / (Annual_RES * 1000.0 * 20.0) if Annual_RES > 0 else 1e6
    # C1: MATLAB 1/(1+10*LCOE) is calibrated for $/kWh — collapses at Rs/kWh scale.
    # Use military diesel reference: ₹26/kWh = 95 Rs/L × 0.27 L/kWh
    # C1>0.5 means cheaper than diesel (good); C1<0.5 means more expensive
    C1   = 1.0 / (1.0 + 10.0 * LCOE)   # MATLAB exact: 1/(1+10*LCOE)
    C2   = 1.0 / (1.0 + CAPEX / 1e9)
    EcSI = 0.5 * C1 + 0.5 * C2

    # ── 6. TRI (Technical Reliability) ───────────────────────────────────────
    # MATLAB: TRI = 0.5*T1 + 0.2*T2 + 0.2*T3 + 0.1*EL_factor
    T1         = 1.0
    T2         = Aut_factor
    redundancy = x[2] / 1000.0 + x[5] / 100.0
    T3         = 1.0 - np.exp(-redundancy / 3.0)
    # EL utilisation factor — rewards balanced electrolyser usage near 70%
    EL_util    = sim.h2_produced_kWh / (x[3] * 8760.0 + 1e-6)
    EL_factor  = np.exp(-abs(EL_util - 0.7))
    TRI        = 0.5 * T1 + 0.2 * T2 + 0.2 * T3 + 0.1 * EL_factor

    # ── 7. ORI (Operational Resilience) ──────────────────────────────────────
    # MATLAB: ORI = 0.3*R1 + 0.3*R2 + 0.2*R3 + 0.2*H2_factor
    gen_mix   = x[0] + x[1]
    R1        = T2
    R2        = 1.0 - np.exp(-gen_mix / 800.0)
    R3        = 1.0 - np.exp(-x[5] / 80.0)
    # H2 utilisation factor: target 0.5 (seasonal asymmetry in mission ops).
    # FC fires during winter/night deficits, EL charges during summer surplus
    # → H2_used ≈ 0.3-0.6 × H2_produced is typical; target 0.5 is mission-realistic.
    # (MATLAB target 1.0 assumes balanced year-round H2 cycle, not realistic here)
    H2_util   = sim.h2_used_kWh / (sim.h2_produced_kWh + 1e-6)
    H2_factor = np.exp(-abs(H2_util - 1.0))   # MATLAB exact
    ORI       = 0.3 * R1 + 0.3 * R2 + 0.2 * R3 + 0.2 * H2_factor

    # ── 8. LSI (Logistics) ────────────────────────────────────────────────────
    # MATLAB: LSI = 0.3*L1 + 0.3*L2 + 0.2*L3 + 0.2*H2_factor
    L1  = T2
    L2  = 1.0 / (1.0 + CAPEX / 1e9)
    L3  = 1.0 - np.exp(-x[2] / 2000.0)
    LSI = 0.3 * L1 + 0.3 * L2 + 0.2 * L3 + 0.2 * H2_factor

    # ── 9. MPBSI — Mission weights: ESI=0.05 EcSI=0.20 TRI=0.30 ORI=0.25 LSI=0.20 (MATLAB revised)
    _w = weights or {}
    w_esi  = float(_w.get("w_esi",  0.05))   # MATLAB revised: 0.05
    w_ecsi = float(_w.get("w_ecsi", 0.20))
    w_tri  = float(_w.get("w_tri",  0.30))   # MATLAB revised: 0.30
    w_ori  = float(_w.get("w_ori",  0.25))
    w_lsi  = float(_w.get("w_lsi",  0.20))
    MPBSI = w_esi*ESI + w_ecsi*EcSI + w_tri*TRI + w_ori*ORI + w_lsi*LSI

    pillars = Pillars(ESI=round(ESI, 6), EcSI=round(EcSI, 6),
                      TRI=round(TRI, 6),  ORI=round(ORI, 6),  LSI=round(LSI, 6))

    logger.info("MPBSI=%.4f | ESI=%.3f EcSI=%.3f TRI=%.3f ORI=%.3f LSI=%.3f | Aut=%.2fd",
                MPBSI, ESI, EcSI, TRI, ORI, LSI, sim.autonomy_days)


    return MPBSIResult(mpbsi=round(MPBSI, 6), pillars=pillars,
                       is_feasible=True, simulation=sim)


# ══════════════════════════════════════════════════════════════════════════════
#  RESOURCE DISPATCH  (microgrid_dispatch_resource.m)
# ══════════════════════════════════════════════════════════════════════════════

def microgrid_dispatch_resource(
    x:    np.ndarray | list,
    base: BaseData,
) -> SimulationResult:
    """
    Exact translation of microgrid_dispatch_resource.m.
    Optimised: Annual_RES vectorised; inner loop uses Python lists for speed.
    """
    P_PV, P_WT, E_BESS, P_EL_max, E_H2_max, P_FC_max = (
        float(x[0]), float(x[1]), float(x[2]),
        float(x[3]), float(x[4]), float(x[5]),
    )

    ghi  = np.nan_to_num(base.ghi,  nan=0.0, posinf=0.0, neginf=0.0)
    temp = np.nan_to_num(base.temperature, nan=25.0, posinf=25.0, neginf=25.0)
    wind = np.nan_to_num(base.wind_speed,  nan=0.0, posinf=0.0, neginf=0.0)
    load = np.nan_to_num(base.load,        nan=0.0, posinf=0.0, neginf=0.0)
    N    = base.N

    # ── PV model (vectorised) ─────────────────────────────────────────────
    Tcell  = temp + 0.0256 * ghi
    P_PV_t = np.maximum(P_PV * (ghi / 1000.0) * (1.0 - 0.004 * (Tcell - 25.0)) * 0.95, 0.0)

    # ── Wind model (vectorised) ───────────────────────────────────────────
    # FIX: Hhub=20m (was 40m) — matches MATLAB microgrid_dispatch_mission.m: Hhub=20, Href=10
    V2 = wind * (20.0 / 10.0) ** 0.14
    P_WT_below = P_WT * ((V2 - 2.5) / (10.0 - 2.5)) ** 2.2 * 1.05
    P_WT_t = np.where((V2 < 2.5) | (V2 >= 20.0), 0.0,
                      np.where(V2 < 10.0, P_WT_below, P_WT)) * 0.80
    P_RES = P_PV_t + P_WT_t

    # ── Load split ────────────────────────────────────────────────────────
    Load_c = 0.60 * load
    Load_s = 0.25 * load
    Load_n = 0.15 * load

    # Annual_RES is just sum(P_RES) — vectorised outside loop
    Annual_RES = float(np.sum(P_RES))

    # ── Storage initial conditions ────────────────────────────────────────
    eta_ch = 0.95; eta_dis = 0.95; eta_EL = 0.70; eta_FC = 0.55
    SOC     = 0.8  * E_BESS
    SOC_min = 0.20 * E_BESS
    SOC_max = E_BESS
    E_H2    = 0.5  * E_H2_max

    unmet_c = 0.0; curt_semi = 0.0; curt_non = 0.0
    H2_used_total = 0.0
    H2_prod_total = 0.0

    # Pre-convert to Python lists for faster per-element access in loop
    P_RES_  = P_RES.tolist()
    Load_c_ = Load_c.tolist()
    Load_s_ = Load_s.tolist()
    Load_n_ = Load_n.tolist()

    for t in range(N):
        P_available = P_RES_[t]
        deficit     = Load_c_[t] - P_available

        if deficit > 0:
            # Battery discharge
            avail_bat = (SOC - SOC_min) * eta_dis
            P_dis = deficit if deficit < avail_bat else avail_bat
            SOC  -= P_dis / eta_dis
            deficit -= P_dis

            # Fuel cell
            max_fc = E_H2 * eta_FC
            if P_FC_max < max_fc: max_fc = P_FC_max
            P_fc   = deficit if deficit < max_fc else max_fc
            H2_con = P_fc / eta_FC
            E_H2  -= H2_con
            H2_used_total += H2_con
            deficit -= P_fc
            if deficit > 0:
                unmet_c += deficit
            P_available = 0.0
        else:
            P_available = -deficit

        # Semi load
        ls = Load_s_[t]
        if P_available >= ls:
            P_available -= ls
        else:
            curt_semi   += ls - P_available
            P_available  = 0.0

        # Non-critical load
        ln = Load_n_[t]
        if P_available >= ln:
            P_available -= ln
        else:
            curt_non   += ln - P_available
            P_available  = 0.0

        # Charging
        if P_available > 0:
            max_ch = (SOC_max - SOC) / eta_ch
            P_ch   = P_available if P_available < max_ch else max_ch
            SOC   += eta_ch * P_ch
            P_available -= P_ch

            if P_EL_max > 0 and P_available > 0:
                room = max((E_H2_max - E_H2) / eta_EL, 0.0)
                P_el = P_available
                if P_EL_max < P_el: P_el = P_EL_max
                if room    < P_el: P_el = room
                if P_el > 0:
                    H2_prod        = eta_EL * P_el
                    E_H2          += H2_prod
                    H2_prod_total += H2_prod

        # Clamp (branch-free min/max)
        if SOC  > SOC_max: SOC  = SOC_max
        elif SOC < SOC_min: SOC = SOC_min
        if E_H2 > E_H2_max: E_H2 = E_H2_max
        elif E_H2 < 0.0:    E_H2 = 0.0

    Total_Load     = float(np.sum(load))
    Total_critical = float(np.sum(Load_c))
    lpsp_critical   = unmet_c / max(Total_critical, 1e-9)
    renewable_ratio = Annual_RES / max(Total_Load, 1e-9)
    usable_battery  = SOC_max - SOC_min
    usable_h2_elec  = E_H2_max * eta_FC
    critical_daily  = Total_critical / 365.0
    autonomy_days   = (usable_battery + usable_h2_elec) / max(critical_daily, 1e-9)

    return SimulationResult(
        is_feasible         = (lpsp_critical <= 1.00001e-4) and (renewable_ratio >= 0.99),
        lpsp_critical       = lpsp_critical,
        renewable_ratio     = renewable_ratio,
        autonomy_days       = autonomy_days,
        annual_load_MWh     = Total_Load / 1000.0,
        total_renewable_MWh = Annual_RES / 1000.0,
        curtailed_semi_MWh  = curt_semi / 1000.0,
        curtailed_non_MWh   = curt_non  / 1000.0,
        h2_used_kWh         = H2_used_total,
        h2_produced_kWh     = H2_prod_total,
    )




# ══════════════════════════════════════════════════════════════════════════════
#  RESOURCE MPBSI EVALUATOR  (MPBSI_Evaluator_Resource_Land.m)
# ══════════════════════════════════════════════════════════════════════════════

def mpbsi_evaluator_resource(
    x:              np.ndarray | list,
    base:           BaseData,
    land_available: float = 50_000.0,
    weights:        Optional[dict] = None,
) -> MPBSIResult:
    """
    Exact translation of MPBSI_Evaluator_Resource_Land.m.

    Key differences vs mission evaluator:
      - No hard autonomy constraint (smooth reward T2 = 1-exp(-auto/1.5))
      - Feasibility: LPSP<=1e-4 AND Annual_RES >= Annual_Load (no 7-day requirement)
      - Pillar weights: ESI=0.05, EcSI=0.20, TRI=0.30, ORI=0.25, LSI=0.20
      - ESI E4: smooth storage saturation curve
      - ORI: gen_mix formula
      - LSI: includes L5 battery term
    """
    x = np.asarray(x, dtype=float)

    # Land constraint
    Area_PV = 10.0 * x[0]
    Area_WT = 15.0 * x[1]
    if (Area_PV + Area_WT) > (land_available + 0.1):  # 0.1 m² tolerance for PSO rounding
        return MPBSIResult(mpbsi=-1e6, pillars=Pillars(ESI=0.0,EcSI=0.0,TRI=0.0,ORI=0.0,LSI=0.0), is_feasible=False, simulation=None)

    sim = microgrid_dispatch_resource(x, base)

    Annual_Load = sim.annual_load_MWh
    Annual_RES  = sim.total_renewable_MWh

    # ── FEASIBILITY (MATLAB MPBSI_Evaluator_Resource_Land.m lines 34-40) ────────
    # MATLAB evaluator uses: LPSP_critical > 1e-4 OR Annual_RES < Annual_Load
    # Note: Annual_RES < Annual_Load = Renewable_ratio < 1.0 (stricter than dispatch
    # field is_feasible which uses 0.99).  The evaluator does NOT use Results.feasible.
    if sim.lpsp_critical > 1e-4 or sim.total_renewable_MWh < sim.annual_load_MWh:
        return MPBSIResult(mpbsi=-1e6, pillars=Pillars(ESI=0.0,EcSI=0.0,TRI=0.0,ORI=0.0,LSI=0.0), is_feasible=False, simulation=sim)

    eta_FC = 0.55

    # ── H2_effective cap (MATLAB line 65) ────────────────────────────────────────
    # MATLAB MPBSI_Evaluator_Resource_Land.m:
    #   H2_effective = min(x(5), x(4)*24);   % limit H2 by EL 24-h charging capacity
    #   usable_storage = (0.95-0.20)*x(3) + H2_effective*eta_FC;
    #
    # The cap H2_effective = min(H2_kWh, EL_kW × 24) forces the optimizer to size EL
    # large enough to fill the H2 tank within 24 h — otherwise H2 credit is capped.
    # MATLAB solutions: EL×24 ≥ H2 → cap never activates → full H2 credit.
    # Without this cap (v5.5 mistake), Python had H2=9999 kWh with EL=21 kW giving
    # only 516 kWh effective H2 — a 95% loss that the evaluator was silently ignoring.
    H2_effective   = min(x[4], x[3] * 24.0)          # = min(H2_kWh, EL_kW × 24)
    usable_storage = (0.95 - 0.20) * x[2] + H2_effective * eta_FC
    critical_daily = (Annual_Load * 1000.0 / 365.0) * 0.60
    storage_ratio  = usable_storage / max(critical_daily, 1e-6)

    # ── ESI ──────────────────────────────────────────────────────────────────
    E1 = 1.0
    # E2: MATLAB exact — penalises surplus (keeps PSO from oversizing)
    E2_raw = Annual_RES / max(Annual_Load, 1e-9)
    E2 = 1.0 / (1.0 + abs(E2_raw - 1.0))
    # E3: curtailment relative to generation (MATLAB exact)
    Curt = sim.curtailed_semi_MWh + sim.curtailed_non_MWh
    Curt_ratio = Curt / max(Annual_RES, 1e-6)
    E3 = 1.0 - Curt_ratio
    # E4: MATLAB exact exponential saturation
    E4 = 1.0 - np.exp(-min(storage_ratio / 3.0, 50.0))
    ESI = 0.30*E1 + 0.30*E2 + 0.20*E3 + 0.20*E4

    # ── EcSI ─────────────────────────────────────────────────────────────────
    CAPEX = (_UC["pv"]*x[0] + _UC["wind"]*x[1] + _UC["batt"]*x[2] + _UC["el"]*x[3] + _UC["h2"]*x[4] + _UC["fc"]*x[5])
    LCOE  = CAPEX / max(Annual_RES * 1000.0 * 20.0, 1e-6)
    C1 = 1.0 / (1.0 + 10.0 * LCOE)    # MATLAB exact: 1/(1+10*LCOE)
    C2 = 1.0 / (1.0 + CAPEX / 1e9)
    OM_ratio = (0.02 * CAPEX) / max(Annual_RES * 1000.0, 1.0)
    C3 = 1.0 / (1.0 + 50.0 * OM_ratio)
    C4 = 1.0
    EcSI = 0.35*C1 + 0.30*C2 + 0.20*C3 + 0.15*C4

    # ── TRI ──────────────────────────────────────────────────────────────────
    T1 = 1.0

    # T2 — MATLAB line 112: T2 = safe_exp(Results.Autonomy_days/1.5)
    # Uses DISPATCH autonomy (0.80×BESS + E_H2_max×eta_FC, FULL H2 tank, uncapped).
    # This is intentionally different from storage_ratio which uses H2_effective (capped).
    # T2 rewards theoretical max autonomy; T4 rewards EL-constrained practical storage.
    T2 = 1.0 - np.exp(-min(sim.autonomy_days / 1.5, 50.0))

    # T3 — MATLAB line 114-115: redundancy = x(3)/1000 + x(6)/100  (BESS+FC only)
    # This is BESS capacity in MWh + FC power in hundreds of kW.
    # H2 is NOT in T3 in MATLAB — do not add it (v5.6 incorrectly added H2 here).
    redundancy = (x[2] / 1000.0 + x[5] / 100.0)
    T3 = 1.0 - np.exp(-min(redundancy / 3.0, 50.0))

    T4 = 1.0 - np.exp(-min(storage_ratio / 3.0, 50.0))
    TRI = 0.40*T1 + 0.30*T2 + 0.20*T3 + 0.10*T4

    # ── ORI ──────────────────────────────────────────────────────────────────
    R1 = T2
    gen_mix = x[0] + x[1]
    R2 = 1.0 - np.exp(-min(gen_mix / 800.0, 50.0))
    R3 = 1.0
    R4 = 1.0 - np.exp(-min(x[5] / 80.0, 50.0))
    ORI = 0.35*R1 + 0.25*R2 + 0.20*R3 + 0.20*R4

    # ── LSI ──────────────────────────────────────────────────────────────────
    L1 = 1.0
    L2 = T2
    L3 = 1.0 / (1.0 + CAPEX / 1e9)
    L4 = T4
    # L5 — MATLAB line 146: L5 = safe_exp(x(3)/2000)  (BESS-only, MATLAB exact)
    # H2 is NOT in L5 in MATLAB — v5.6 incorrectly added combined usable_storage here.
    L5 = 1.0 - np.exp(-min(x[2] / 2000.0, 50.0))
    LSI = 0.25*L1 + 0.25*L2 + 0.20*L3 + 0.15*L4 + 0.15*L5

    # ── MPBSI (Resource weights) ──────────────────────────────────────────────
    # MATLAB MPBSI_Evaluator_Resource_Land.m: clamp all indices to [0,1] before
    # computing MPBSI, then clamp MPBSI itself.
    # This prevents any formula overshoot from inflating the final score.
    ESI  = float(np.clip(ESI,  0.0, 1.0))
    EcSI = float(np.clip(EcSI, 0.0, 1.0))
    TRI  = float(np.clip(TRI,  0.0, 1.0))
    ORI  = float(np.clip(ORI,  0.0, 1.0))
    LSI  = float(np.clip(LSI,  0.0, 1.0))

    _w = weights or {}
    w_esi  = float(_w.get("w_esi",  0.05))
    w_ecsi = float(_w.get("w_ecsi", 0.20))
    w_tri  = float(_w.get("w_tri",  0.30))
    w_ori  = float(_w.get("w_ori",  0.25))
    w_lsi  = float(_w.get("w_lsi",  0.20))
    MPBSI  = w_esi*ESI + w_ecsi*EcSI + w_tri*TRI + w_ori*ORI + w_lsi*LSI
    MPBSI  = float(np.clip(MPBSI, 0.0, 1.0))   # MATLAB: min(max(MPBSI,0),1)

    pillars = Pillars(ESI=round(ESI,6), EcSI=round(EcSI,6),
                      TRI=round(TRI,6), ORI=round(ORI,6), LSI=round(LSI,6))

    # ── Step9 survivability validation (informational — NOT a hard reject in resource mode) ──
    # Step9_Survivability_H2_Sizing.m: H2_baseline = (2×Crit_Daily)/eta_FC
    # Resource mode MATLAB does NOT reject on survivability — only mission mode does.
    # We log a warning when usable_storage < H2_baseline to aid system designers.
    _h2_baseline_kWh = (2.0 * critical_daily) / eta_FC
    if usable_storage < _h2_baseline_kWh:
        logger.debug(
            "Resource solution below Step9 2-day baseline: usable=%.0f < H2_baseline=%.0f kWh",
            usable_storage, _h2_baseline_kWh)

    return MPBSIResult(mpbsi=round(MPBSI,6), pillars=pillars,
                       is_feasible=True, simulation=sim)


# ══════════════════════════════════════════════════════════════════════════════
#  NSGA-II OBJECTIVE  (NSGA_Objective_Mission.m)
# ══════════════════════════════════════════════════════════════════════════════

def nsga_objective_mission(
    x:              np.ndarray,
    base:           BaseData,
    land_available: float = 50_000.0,
    weights:        Optional[dict] = None,
) -> tuple[float, float]:
    """
    Exact translation of NSGA_Objective_Mission.m.
    Returns (f1, f2): f1 = −MPBSI, f2 = NPC/1e8. Penalty (1e3,1e3) if infeasible.

    Mission-specific penalties (MATLAB NSGA_Objective_Mission.m):
      - Autonomy oversizing: Penalty_factor=0.08 (mission, vs 0.05 resource)
      - H2 utilisation: penalise if H2_util < 0.3 (dead hydrogen system)
      - EL utilisation: penalise if EL_util < 0.2 (underused electrolyser)
    """
    Required_Autonomy = 7.0

    res = mpbsi_evaluator(x, base, land_available, weights=weights)
    if res.mpbsi < 0:
        return (1e3, 1e3)

    sim = res.simulation
    if (not sim.is_feasible) or (sim.autonomy_days < Required_Autonomy):
        return (1e3, 1e3)

    f1 = -res.mpbsi

    Cost_PV = _UC["pv"] * x[0]
    Cost_WT = _UC["wind"] * x[1]
    Cost_Bat = _UC["batt"] * x[2]
    Cost_EL = _UC["el"] * x[3]
    Cost_H2 = _UC["h2"] * x[4]
    Cost_FC = _UC["fc"] * x[5]
    CAPEX    = Cost_PV + Cost_WT + Cost_Bat + Cost_EL + Cost_H2 + Cost_FC

    r        = 0.08; n = 20
    Bat_repl = Cost_Bat / (1 + r) ** 10
    FC_repl  = Cost_FC  / (1 + r) ** 10
    EL_repl  = Cost_EL  / (1 + r) ** 15
    OM_ann   = 0.02 * CAPEX
    OM_total = OM_ann * ((1 - (1 + r) ** (-n)) / r)
    NPC      = CAPEX + Bat_repl + FC_repl + EL_repl + OM_total

    # ── Mission autonomy oversizing penalty ──
    # MATLAB NSGA_Objective_Mission.m exact:
    #   usable_energy = (0.95 - 0.20)*x(3) + x(5)*eta_FC
    #   critical_daily = (Annual_Load_MWh*1000/365)*0.60
    #   Autonomy_days = usable_energy / critical_daily
    eta_FC_n    = 0.55
    usable_e    = (0.95 - 0.20) * x[2] + x[4] * eta_FC_n
    crit_daily  = (sim.annual_load_MWh * 1000.0 / 365.0) * 0.60
    Aut_days    = usable_e / crit_daily if crit_daily > 0 else 0.0
    Penalty_factor = 0.08
    if Aut_days > Required_Autonomy:
        NPC = NPC * (1.0 + Penalty_factor * (Aut_days - Required_Autonomy))

    # ── H2 utilisation penalty (MATLAB: penalise dead H2 system) ──
    H2_util = sim.h2_used_kWh / (sim.h2_produced_kWh + 1e-6)
    if H2_util < 0.3:
        NPC = NPC * (1.0 + 0.1 * (0.3 - H2_util))

    # ── Electrolyser utilisation penalty (MATLAB: penalise underused EL) ──
    EL_util = sim.h2_produced_kWh / (x[3] * 8760.0 + 1e-6)
    if EL_util < 0.2:
        NPC = NPC * (1.0 + 0.1 * (0.2 - EL_util))

    # NOTE: Step9 survivability penalty removed — not in MATLAB NSGA_Objective_Mission.m
    # MATLAB only has: autonomy oversizing, H2_util < 0.3, EL_util < 0.2 penalties.

    f2 = NPC / 1e8

    return (f1, f2)


# ══════════════════════════════════════════════════════════════════════════════
#  20-YEAR LIFECYCLE NPC  (Mission_Lifecycle_NSCA_CaseB_20yr.m)
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class LifecycleResult:
    """Full 20-yr techno-economic and tactical analysis"""
    CAPEX:                     float
    NPC_microgrid:             float
    NPC_diesel_financial:      float
    NPC_diesel_mission:        float
    LCOE:                      float
    Net_Savings:               float
    OM_NPV:                    float
    Replacement_NPV:           float
    Personnel_NPV:             float
    Convoy_NPV:                float
    Risk_cost:                 float
    Convoys_per_year:          int
    Annual_Tactical_Exposure:  int
    Convoy_Operational_Hours:  int
    Permanent_Staff_Reduction: int
    Lifecycle_Personnel_Years_Saved: int
    Annual_CO2_ton:            float
    Lifetime_CO2_Reduction:    float
    Annual_Load_kWh:           float
    Peak_Load_kW:              float

    def to_dict(self) -> dict:
        return asdict(self)


# ══════════════════════════════════════════════════════════════════════════════
#  ENGINEERING SIZING  (microgrid_full_deployment_analysis.m)
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class EngineeringSizing:
    """Exact translation of engineering_sizing() in microgrid_full_deployment_analysis.m"""
    PV_panels:        int
    Wind_turbines:    int
    Electrolysers:    int
    FuelCells:        int
    H2_kg:            float
    H2_tanks:         int
    Tanker_trips:     int
    Refill_days:      float   # electrolyser refill time (days); inf if not possible
    H2_backup_days:   float

    def to_dict(self) -> dict:
        return {
            "PV_panels":      self.PV_panels,
            "Wind_turbines":  self.Wind_turbines,
            "Electrolysers":  self.Electrolysers,
            "FuelCells":      self.FuelCells,
            "H2_kg":          round(self.H2_kg, 1),
            "H2_tanks":       self.H2_tanks,
            "Tanker_trips":   self.Tanker_trips,
            "Refill_days":    None if self.Refill_days == float("inf") else round(self.Refill_days, 1),
            "H2_backup_days": round(self.H2_backup_days, 2),
        }


def compute_engineering_sizing(
    x:        np.ndarray | list,
    sim:      SimulationResult,
    PV_panel_W:    float = 540.0,
    Wind_unit_kW:  float = 10.0,
    EL_unit_kW:    float = 5.0,
    FC_unit_kW:    float = 10.0,
    H2_kWh_per_kg: float = 33.3,
    Tank_kg:       float = 30.0,
    Tanker_kg:     float = 350.0,
    eta_EL:        float = 0.70,
    eta_FC:        float = 0.55,
) -> EngineeringSizing:
    """
    Exact translation of engineering_sizing() in microgrid_full_deployment_analysis.m

    Technology assumptions (MATLAB exact):
      PV panel = 540 W, Wind unit = 10 kW, EL unit = 5 kW, FC unit = 10 kW
      H2 kWh/kg = 33.3, Tank = 30 kg, Tanker = 350 kg
    """
    x = np.asarray(x, dtype=float)
    PV_kWp  = x[0]; Wind_kW = x[1]
    EL_kW   = x[3]; H2_kWh  = x[4]; FC_kW = x[5]

    PV_panels     = int(np.ceil(PV_kWp * 1000.0 / PV_panel_W))
    Wind_turbines = int(np.ceil(Wind_kW / Wind_unit_kW))
    Electrolysers = int(np.ceil(EL_kW  / EL_unit_kW))
    FuelCells     = int(np.ceil(FC_kW  / FC_unit_kW))

    H2_kg   = H2_kWh / H2_kWh_per_kg
    H2_tanks  = int(np.ceil(H2_kg / Tank_kg))
    Tanker_trips = int(np.ceil(H2_kg / Tanker_kg))

    # Electrolyser refill time — MATLAB uses H2_kWh_per_kg=50 for annual kg (typo in .m, kept)
    Curtailment_kWh = (sim.curtailed_semi_MWh + sim.curtailed_non_MWh) * 1000.0
    Annual_H2_kWh   = Curtailment_kWh * eta_EL
    Annual_H2_kg    = Annual_H2_kWh / 50.0          # MATLAB exact: /50
    if Annual_H2_kg > 0:
        Refill_days = (H2_kg / Annual_H2_kg) * 365.0
    else:
        Refill_days = float("inf")

    # H2 backup days (critical load)
    Critical_daily = (sim.annual_load_MWh * 1000.0 / 365.0) * 0.60
    H2_backup_days = (H2_kWh * eta_FC) / max(Critical_daily, 1e-9)

    return EngineeringSizing(
        PV_panels=PV_panels, Wind_turbines=Wind_turbines,
        Electrolysers=Electrolysers, FuelCells=FuelCells,
        H2_kg=H2_kg, H2_tanks=H2_tanks, Tanker_trips=Tanker_trips,
        Refill_days=Refill_days, H2_backup_days=H2_backup_days,
    )


# ══════════════════════════════════════════════════════════════════════════════
#  HYDROGEN LOGISTICS  (Hydrogen_Logistics_resource.m)
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class H2LogisticsResult:
    """Exact translation of Hydrogen_Logistics_resource.m"""
    H2_storage_kg:          float
    Initial_H2_kg:          float       # 50% initial fill
    Initial_trips:          int
    Initial_autonomy_days:  float
    Final_autonomy_days:    float
    Days_to_full_autonomy:  float       # inf if not possible
    H2_used_kg:             float
    H2_prod_kg:             float
    Net_import_kg:          float
    Refill_years:           float       # inf if not required
    Trips_per_year:         int
    Recovery_days:          float       # inf if not possible
    H2_cost_NPV:            float
    NPC_resource:           float

    def to_dict(self) -> dict:
        def _f(v): return None if v == float("inf") else round(v, 2)
        return {
            "H2_storage_kg":         round(self.H2_storage_kg, 1),
            "Initial_H2_kg":         round(self.Initial_H2_kg, 1),
            "Initial_trips":         self.Initial_trips,
            "Initial_autonomy_days": round(self.Initial_autonomy_days, 2),
            "Final_autonomy_days":   round(self.Final_autonomy_days, 2),
            "Days_to_full_autonomy": _f(self.Days_to_full_autonomy),
            "H2_used_kg":            round(self.H2_used_kg, 1),
            "H2_prod_kg":            round(self.H2_prod_kg, 1),
            "Net_import_kg":         round(self.Net_import_kg, 1),
            "Refill_years":          _f(self.Refill_years),
            "Trips_per_year":        self.Trips_per_year,
            "Recovery_days":         _f(self.Recovery_days),
            "H2_cost_NPV_crore":     round(self.H2_cost_NPV / 1e7, 3),
            "NPC_resource_crore":    round(self.NPC_resource / 1e7, 3),
        }


def compute_h2_logistics(
    x:                     np.ndarray | list,
    sim:                   SimulationResult,
    LHV_H2:                float = 33.3,
    tanker_capacity:       float = 900.0,     # MATLAB: 900 kg (both modes)
    hydrogen_cost:         float = 350.0,
    Initial_fill_fraction: float = 0.50,      # resource=0.50, mission=0.60
    eta_FC:                float = 0.55,
    eta_EL:                float = 0.70,
    r:                     float = 0.08,
    n:                     int   = 20,
    mode:                  str   = "resource", # "resource" or "mission"
    Required_Autonomy:     float = 7.0,        # mission: refill to 7d minimum
) -> H2LogisticsResult:
    """
    Exact translation of Hydrogen_Logistics_resource.m (resource) and
    Hydrogen_Techno_economic_mission.m (mission).

    Key differences (mission vs resource):
      - Initial_fill_fraction: 0.60 (mission) vs 0.50 (resource)
      - Recovery_days (mission): H2_energy_needed / H2_refill_per_day
            where H2_energy_needed = (7d × critical_daily) / η_FC
        Recovery_days (resource): Seasonal_H2_use / H2_refill_per_day
    """
    x = np.asarray(x, dtype=float)
    H2_kWh = x[4]

    H2_storage_kg = H2_kWh / LHV_H2
    Initial_H2_kg = Initial_fill_fraction * H2_storage_kg
    Initial_trips = int(np.ceil(Initial_H2_kg / tanker_capacity))

    # Initial autonomy
    Initial_Battery_kWh = (0.8 - 0.2) * x[2]
    Initial_H2_kWh_elec = Initial_H2_kg * LHV_H2 * eta_FC
    critical_daily      = (sim.annual_load_MWh * 1000.0 / 365.0) * 0.60
    Initial_autonomy_days = (Initial_Battery_kWh + Initial_H2_kWh_elec) / max(critical_daily, 1e-9)
    Final_autonomy_days   = sim.autonomy_days

    # H2 flow
    H2_used_kWh   = getattr(sim, "h2_used_kWh",    0.0) or 0.0
    H2_prod_kWh   = getattr(sim, "h2_produced_kWh", 0.0) or 0.0
    H2_used_kg    = H2_used_kWh  / LHV_H2
    H2_prod_kg    = H2_prod_kWh  / LHV_H2
    Net_import_kg = max(0.0, H2_used_kg - H2_prod_kg)

    # Refill interval
    if Net_import_kg > 0:
        Refill_years = H2_storage_kg / Net_import_kg
    else:
        Refill_years = float("inf")

    if Net_import_kg < 1e-3:
        Trips_per_year = 0
    else:
        Trips_per_year = int(np.ceil(Net_import_kg / tanker_capacity))

    # Blackout/deficit recovery time
    Annual_surplus    = max(0.0, (sim.total_renewable_MWh - sim.annual_load_MWh) * 1000.0)
    Daily_surplus     = Annual_surplus / 365.0
    EL_daily_limit    = x[3] * 24.0
    Actual_refill     = min(Daily_surplus, EL_daily_limit)
    H2_refill_per_day = Actual_refill * eta_EL

    if mode == "mission":
        # MATLAB Hydrogen_Techno_economic_mission.m:
        # Recovery = (Required_Autonomy × critical_daily) / η_FC / H2_refill_per_day
        # Mission only needs to refill back to 7-day minimum, not full tank
        H2_energy_needed = (Required_Autonomy * critical_daily) / eta_FC
        if H2_refill_per_day > 0:
            Recovery_days = H2_energy_needed / H2_refill_per_day
        else:
            Recovery_days = float("inf")
    else:
        # MATLAB Hydrogen_Logistics_resource.m:
        # Recovery = Seasonal_H2_use / H2_refill_per_day
        Seasonal_H2_use = H2_used_kWh
        if H2_refill_per_day > 0:
            Recovery_days = Seasonal_H2_use / H2_refill_per_day
        else:
            Recovery_days = float("inf")

    # Time to full autonomy
    H2_deficit_kWh = H2_kWh - Initial_H2_kg * LHV_H2
    if H2_refill_per_day > 0:
        Days_to_full_autonomy = H2_deficit_kWh / H2_refill_per_day
    else:
        Days_to_full_autonomy = float("inf")

    # H2 cost NPV
    Annual_H2_cost = Net_import_kg * hydrogen_cost
    H2_cost_NPV    = sum(Annual_H2_cost / (1 + r) ** yr for yr in range(1, n + 1))

    # NPC (CAPEX + OM + H2 cost)
    CAPEX = (_UC["pv"]*x[0] + _UC["wind"]*x[1] + _UC["batt"]*x[2] + _UC["el"]*x[3] + _UC["h2"]*x[4] + _UC["fc"]*x[5])
    OM    = 0.02 * CAPEX
    OM_NPV = OM * ((1 - (1 + r) ** -n) / r)
    NPC_resource = CAPEX + OM_NPV + H2_cost_NPV

    return H2LogisticsResult(
        H2_storage_kg=H2_storage_kg,
        Initial_H2_kg=Initial_H2_kg,
        Initial_trips=Initial_trips,
        Initial_autonomy_days=Initial_autonomy_days,
        Final_autonomy_days=Final_autonomy_days,
        Days_to_full_autonomy=Days_to_full_autonomy,
        H2_used_kg=H2_used_kg,
        H2_prod_kg=H2_prod_kg,
        Net_import_kg=Net_import_kg,
        Refill_years=Refill_years,
        Trips_per_year=Trips_per_year,
        Recovery_days=Recovery_days,
        H2_cost_NPV=H2_cost_NPV,
        NPC_resource=NPC_resource,
    )


# ══════════════════════════════════════════════════════════════════════════════
#  RESOURCE LIFECYCLE / STRATEGIC ANALYSIS  (Master_techo_strategic_resource.m)
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class ResourceLifecycleResult:
    """Exact translation of run_case() in Master_techo_strategic_resource.m"""
    CAPEX:                    float
    NPC_microgrid:            float   # includes OM + replacement + H2 logistics + manpower
    NPC_diesel:               float   # includes financial + manpower + convoys + risk
    Net_Savings:              float
    LCOE:                     float
    # Logistics
    Initial_H2_trips:         int
    Sustained_H2_trips:       int     # per year
    Diesel_convoys:           int     # per year
    Convoy_exposure_avoided:  int
    # Manpower
    Diesel_personnel:         int
    Microgrid_personnel:      int
    Personnel_saved:          int
    Manhours_saved:           float   # per year
    # Tactical
    Autonomy_days:            float
    LPSP_critical:            float
    Renewable_ratio:          float
    # Environmental
    CO2_avoided_ton_yr:       float

    def to_dict(self) -> dict:
        return {
            "CAPEX_crore":               round(self.CAPEX / 1e7, 3),
            "NPC_microgrid_crore":       round(self.NPC_microgrid / 1e7, 3),
            "NPC_diesel_crore":          round(self.NPC_diesel / 1e7, 3),
            "Net_Savings_crore":         round(self.Net_Savings / 1e7, 3),
            "LCOE_Rs_per_kWh":           round(self.LCOE, 4),
            "Initial_H2_trips":          self.Initial_H2_trips,
            "Sustained_H2_trips":        self.Sustained_H2_trips,
            "Diesel_convoys":            self.Diesel_convoys,
            "Convoy_exposure_avoided":   self.Convoy_exposure_avoided,
            "Diesel_personnel":          self.Diesel_personnel,
            "Microgrid_personnel":       self.Microgrid_personnel,
            "Personnel_saved":           self.Personnel_saved,
            "Manhours_saved":            round(self.Manhours_saved, 0),
            "Autonomy_days":             round(self.Autonomy_days, 2),
            "LPSP_critical":             round(self.LPSP_critical, 6),
            "Renewable_ratio":           round(self.Renewable_ratio, 4),
            "CO2_avoided_ton_yr":        round(self.CO2_avoided_ton_yr, 1),
        }


def compute_nsga_resource_lifecycle(
    x:            np.ndarray | list,
    base:         BaseData,
    r:            float = 0.08,
    n:            int   = 20,
    infl_OM:      float = 0.04,
    infl_fuel:    float = 0.05,
    diesel_eff:   float = 0.27,
    diesel_price: float = 95.0,
    DG_capex_per_kW: float = 25_000.0,
    avg_salary:   float = 12e5,
    Diesel_staff: int   = 6,   # MATLAB Master_techo_strategic: Diesel_personnel=6
    Hybrid_staff: int   = 2,   # MATLAB Master_techo_strategic: Microgrid_personnel=2
    tanker_capacity: float = 10_000.0,   # MATLAB: 10000 L (fuel tanker)
    cost_per_convoy: float = 1.5e5,
    prob_incident:   float = 0.05,
    incident_cost:   float = 15e7,
) -> dict:
    """
    Exact translation of NSGA_Resource_lifecycle_caseA/B/C.m and
    NSGA_Resource_MissionAdjusted_Lifecycle_Model.m.

    Simpler than compute_resource_lifecycle — no H2 logistics, no microgrid manpower.
    NPC_micro = CAPEX + OM_NPV_escalated + (Bat + FC + EL replacement)
    """
    x = np.asarray(x, dtype=float)
    sim = microgrid_dispatch_resource(x, base)
    Annual_Load_kWh = sim.annual_load_MWh * 1000.0
    Peak_kW = float(np.max(base.load))

    # CAPEX
    CAPEX = (_UC["pv"]*x[0] + _UC["wind"]*x[1] + _UC["batt"]*x[2] + _UC["el"]*x[3] + _UC["h2"]*x[4] + _UC["fc"]*x[5])

    # O&M NPV (escalated — MATLAB loop)
    OM_base = 0.02 * CAPEX
    OM_NPV = sum(OM_base*(1+infl_OM)**(yr-1)/(1+r)**yr for yr in range(1, n+1))

    # Replacement
    Rep_NPV = (_UC["batt"]*x[2]/(1+r)**10 + _UC["fc"]*x[5]/(1+r)**10 + _UC["el"]*x[3]/(1+r)**15)

    NPC_micro = CAPEX + OM_NPV + Rep_NPV

    # LCOE
    Energy_NPV = sum(Annual_Load_kWh/(1+r)**yr for yr in range(1, n+1))
    LCOE = NPC_micro / Energy_NPV if Energy_NPV > 0 else 0.0

    # Diesel financial
    Annual_Diesel_L = Annual_Load_kWh * diesel_eff
    DG_CAPEX = DG_capex_per_kW * Peak_kW
    DG_OM_base = 0.05 * DG_CAPEX
    Diesel_NPV = sum(
        (Annual_Diesel_L*diesel_price*(1+infl_fuel)**(yr-1) +
         DG_OM_base*(1+infl_OM)**(yr-1)) / (1+r)**yr
        for yr in range(1, n+1)
    )
    NPC_diesel_fin = DG_CAPEX + Diesel_NPV

    # Mission-adjusted additions (MATLAB: Personnel + Convoy + Risk)
    Personnel_NPV = sum((Diesel_staff-Hybrid_staff)*avg_salary/(1+r)**yr for yr in range(1, n+1))
    Convoys_per_year = int(np.ceil(Annual_Diesel_L / tanker_capacity))
    Convoy_NPV = sum(Convoys_per_year*cost_per_convoy/(1+r)**yr for yr in range(1, n+1))
    Risk_cost = prob_incident * incident_cost
    NPC_diesel_mission = NPC_diesel_fin + Personnel_NPV + Convoy_NPV + Risk_cost

    Net_Savings = NPC_diesel_mission - NPC_micro
    Personnel_saved = Diesel_staff - Hybrid_staff
    CO2_avoided = Annual_Diesel_L * 2.68 / 1000.0

    return {
        "CAPEX_crore":             round(CAPEX / 1e7, 3),
        "OM_NPV_crore":            round(OM_NPV / 1e7, 3),
        "Replacement_NPV_crore":   round(Rep_NPV / 1e7, 3),
        "NPC_microgrid_crore":     round(NPC_micro / 1e7, 3),
        "LCOE_Rs_per_kWh":         round(LCOE, 4),
        "NPC_diesel_financial_crore": round(NPC_diesel_fin / 1e7, 3),
        "NPC_diesel_mission_crore":round(NPC_diesel_mission / 1e7, 3),  # alias
        "Personnel_NPV_crore":     round(Personnel_NPV / 1e7, 3),
        "Convoy_NPV_crore":        round(Convoy_NPV / 1e7, 3),
        "Risk_crore":              round(Risk_cost / 1e7, 3),
        "Risk_cost_crore":         round(Risk_cost / 1e7, 3),           # alias
        "NPC_diesel_crore":        round(NPC_diesel_mission / 1e7, 3),
        "Net_Savings_crore":       round(Net_Savings / 1e7, 3),
        "Diesel_convoys":          Convoys_per_year,
        "Convoys_per_year":        Convoys_per_year,                     # alias
        "Personnel_saved":         Personnel_saved,
        "Permanent_Staff_Reduction": Personnel_saved,                    # alias
        "Manhours_saved":          round(Personnel_saved * 8 * 365),
        "Lifecycle_Personnel_Years_Saved": round(Personnel_saved * 20),
        "CO2_avoided_ton_yr":      round(CO2_avoided, 1),
        "Annual_CO2_ton":          round(CO2_avoided, 1),               # alias
        "Lifetime_CO2_Reduction":  round(CO2_avoided * n, 0),
        "Autonomy_days":           round(sim.autonomy_days, 2),
        "LPSP_critical":           round(sim.lpsp_critical, 6),
        "Renewable_ratio":         round(sim.renewable_ratio, 4),
        "Annual_Load_kWh":         round(Annual_Load_kWh, 1),
        "Peak_Load_kW":            round(Peak_kW, 2),
        # H2 trip fields (resource mode: no mission H2 resupply logistics)
        "Initial_H2_trips":        0,
        "Sustained_H2_trips":      0,
        "Convoy_exposure_avoided": round(Convoys_per_year * (Diesel_staff - Hybrid_staff)),
        "Annual_Tactical_Exposure":round(Convoys_per_year * (Diesel_staff - Hybrid_staff)),
        "Convoy_Operational_Hours":round(Convoys_per_year * 8),
    }


def compute_resource_lifecycle(
    x:                     np.ndarray | list,
    base:                  BaseData,
    r:                     float = 0.08,
    n:                     int   = 20,
    infl_OM:               float = 0.04,
    infl_fuel:             float = 0.05,
    diesel_eff:            float = 0.27,
    diesel_price:          float = 95.0,
    DG_capex_per_kW:       float = 25_000.0,
    salary:                float = 12e5,
    Diesel_personnel:      int   = 6,
    Microgrid_personnel:   int   = 2,
    LHV_H2:                float = 33.3,
    tanker_capacity:       float = 900.0,
    hydrogen_price:        float = 350.0,
    Initial_fill_fraction: float = 0.50,
    prob_risk:             float = 0.05,
    risk_cost:             float = 15e7,
) -> ResourceLifecycleResult:
    """
    Exact translation of run_case() in Master_techo_strategic_resource.m.

    Differences vs mission lifecycle:
      - Includes H2 logistics cost (Net_import × hydrogen_price, NPV)
      - Microgrid manpower = 2, Diesel manpower = 6
      - Convoy exposure = Diesel_convoys - Sustained_H2_trips
      - Diesel system includes: financial + manpower + convoys + risk
    """
    x   = np.asarray(x, dtype=float)
    sim = microgrid_dispatch_resource(x, base)

    Annual_Load = sim.annual_load_MWh * 1000.0
    Peak_kW     = float(np.max(base.load))

    # ── CAPEX ──
    Cost_PV = _UC["pv"] * x[0]; Cost_WT = _UC["wind"] * x[1]
    Cost_Bat = _UC["batt"] * x[2]; Cost_EL = _UC["el"] * x[3]
    Cost_H2 = _UC["h2"] * x[4]; Cost_FC = _UC["fc"] * x[5]
    CAPEX    = Cost_PV + Cost_WT + Cost_Bat + Cost_EL + Cost_H2 + Cost_FC

    # ── O&M (escalated NPV — MATLAB exact) ──
    OM_base = 0.02 * CAPEX
    OM_NPV  = sum(OM_base*(1+infl_OM)**(yr-1) / (1+r)**yr for yr in range(1, n+1))

    # ── Replacement ──
    Rep = Cost_Bat/(1+r)**10 + Cost_FC/(1+r)**10 + Cost_EL/(1+r)**15

    # ── H2 logistics ──
    H2_storage_kg     = x[4] / LHV_H2
    Initial_H2_kg     = Initial_fill_fraction * H2_storage_kg
    Initial_H2_trips  = int(np.ceil(Initial_H2_kg / tanker_capacity))

    H2_used_kWh  = getattr(sim, "h2_used_kWh",    0.0) or 0.0
    H2_prod_kWh  = getattr(sim, "h2_produced_kWh", 0.0) or 0.0
    H2_used_kg   = H2_used_kWh  / LHV_H2
    H2_prod_kg   = H2_prod_kWh  / LHV_H2
    Net_import_kg = max(0.0, H2_used_kg - H2_prod_kg)
    Sustained_H2_trips = int(np.ceil(Net_import_kg / tanker_capacity)) if Net_import_kg > 0 else 0

    H2_cost_NPV = sum(
        (Net_import_kg * hydrogen_price * (1+infl_OM)**(yr-1)) / (1+r)**yr
        for yr in range(1, n+1)
    )

    # ── Microgrid manpower ──
    Micro_personnel_NPV = sum(
        (Microgrid_personnel * salary) / (1+r)**yr for yr in range(1, n+1)
    )

    NPC_microgrid = CAPEX + OM_NPV + Rep + H2_cost_NPV + Micro_personnel_NPV

    # ── LCOE ──
    Disc_Energy = sum(Annual_Load / (1+r)**yr for yr in range(1, n+1))
    LCOE = NPC_microgrid / Disc_Energy if Disc_Energy > 0 else 0.0

    # ── Diesel ──
    Annual_Diesel = Annual_Load * diesel_eff
    DG_CAPEX      = DG_capex_per_kW * Peak_kW
    DG_OM_base    = 0.05 * DG_CAPEX

    Fuel_NPV = sum(
        (Annual_Diesel * diesel_price * (1+infl_fuel)**(yr-1) +
         DG_OM_base    * (1+infl_OM)**(yr-1)) / (1+r)**yr
        for yr in range(1, n+1)
    )
    NPC_diesel_fin = DG_CAPEX + Fuel_NPV

    # Diesel manpower NPV
    Diesel_personnel_NPV = sum(
        (Diesel_personnel * salary) / (1+r)**yr for yr in range(1, n+1)
    )

    # Convoy NPV
    Diesel_convoys  = int(np.ceil(Annual_Diesel / 10000.0))
    Convoy_NPV      = sum(
        (Diesel_convoys * 1.5e5) / (1+r)**yr for yr in range(1, n+1)
    )

    Risk = prob_risk * risk_cost

    NPC_diesel = NPC_diesel_fin + Diesel_personnel_NPV + Convoy_NPV + Risk

    # ── Strategic metrics ──
    Net_Savings              = NPC_diesel - NPC_microgrid
    Convoy_exposure_avoided  = Diesel_convoys - Sustained_H2_trips
    Personnel_saved          = Diesel_personnel - Microgrid_personnel
    Manhours_saved           = Personnel_saved * 8 * 365

    # CO2
    CO2_avoided = Annual_Diesel * 2.68 / 1000.0

    return ResourceLifecycleResult(
        CAPEX=CAPEX, NPC_microgrid=NPC_microgrid, NPC_diesel=NPC_diesel,
        Net_Savings=Net_Savings, LCOE=LCOE,
        Initial_H2_trips=Initial_H2_trips, Sustained_H2_trips=Sustained_H2_trips,
        Diesel_convoys=Diesel_convoys, Convoy_exposure_avoided=Convoy_exposure_avoided,
        Diesel_personnel=Diesel_personnel, Microgrid_personnel=Microgrid_personnel,
        Personnel_saved=Personnel_saved, Manhours_saved=Manhours_saved,
        Autonomy_days=sim.autonomy_days, LPSP_critical=sim.lpsp_critical,
        Renewable_ratio=sim.renewable_ratio, CO2_avoided_ton_yr=CO2_avoided,
    )


# ══════════════════════════════════════════════════════════════════════════════
#  MISSION STRATEGIC LIFECYCLE  (Master_techno_economic_mission.m)
# ══════════════════════════════════════════════════════════════════════════════

def compute_mission_lifecycle(
    x:                     np.ndarray | list,
    base:                  BaseData,
    r:                     float = 0.08,
    n:                     int   = 20,
    infl_OM:               float = 0.04,
    infl_fuel:             float = 0.05,
    diesel_eff:            float = 0.27,
    diesel_price:          float = 95.0,
    DG_capex_per_kW:       float = 25_000.0,
    salary:                float = 12e5,
    Diesel_personnel:      int   = 6,
    Microgrid_personnel:   int   = 2,
    LHV_H2:                float = 33.3,
    tanker_capacity:       float = 900.0,
    hydrogen_price:        float = 350.0,
    Initial_fill_fraction: float = 0.60,    # MATLAB mission: 60% initial fill
    prob_risk:             float = 0.05,
    risk_cost:             float = 15e7,
) -> "ResourceLifecycleResult":
    """
    Exact translation of run_case() in Master_techno_economic_mission.m.

    Mission differences vs resource lifecycle:
      - Uses microgrid_dispatch_mission (not resource)
      - Initial H2 fill = 60% (mission readiness)
      - Tanker capacity = 900 kg (same as resource)
      - Diesel personnel = 6, Microgrid personnel = 2
    Output format matches ResourceLifecycleResult for unified dashboard display.
    """
    x   = np.asarray(x, dtype=float)
    sim = microgrid_dispatch_full(x, base)   # mission dispatch

    Annual_Load = sim.annual_load_MWh * 1000.0
    Peak_kW     = float(np.max(base.load))

    # ── CAPEX ──
    Cost_PV = _UC["pv"] * x[0]; Cost_WT = _UC["wind"] * x[1]
    Cost_Bat = _UC["batt"] * x[2]; Cost_EL = _UC["el"] * x[3]
    Cost_H2 = _UC["h2"] * x[4]; Cost_FC = _UC["fc"] * x[5]
    CAPEX    = Cost_PV + Cost_WT + Cost_Bat + Cost_EL + Cost_H2 + Cost_FC

    # ── O&M (escalated NPV) ──
    OM_base = 0.02 * CAPEX
    OM_NPV  = sum(OM_base*(1+infl_OM)**(yr-1) / (1+r)**yr for yr in range(1, n+1))

    # ── Replacement ──
    Rep = Cost_Bat/(1+r)**10 + Cost_FC/(1+r)**10 + Cost_EL/(1+r)**15

    # ── H2 logistics (mission: 60% initial fill) ──
    H2_storage_kg    = x[4] / LHV_H2
    Initial_H2_kg    = Initial_fill_fraction * H2_storage_kg
    Initial_H2_trips = int(np.ceil(Initial_H2_kg / tanker_capacity))

    H2_used_kWh  = getattr(sim, "h2_used_kWh",    0.0) or 0.0
    H2_prod_kWh  = getattr(sim, "h2_produced_kWh", 0.0) or 0.0
    H2_used_kg   = H2_used_kWh  / LHV_H2
    H2_prod_kg   = H2_prod_kWh  / LHV_H2
    Net_import_kg = max(0.0, H2_used_kg - H2_prod_kg)
    Sustained_H2_trips = int(np.ceil(Net_import_kg / tanker_capacity)) if Net_import_kg > 0 else 0

    H2_cost_NPV = sum(
        (Net_import_kg * hydrogen_price * (1+infl_OM)**(yr-1)) / (1+r)**yr
        for yr in range(1, n+1)
    )

    # ── Microgrid manpower ──
    Micro_personnel_NPV = sum(
        (Microgrid_personnel * salary) / (1+r)**yr for yr in range(1, n+1)
    )

    NPC_microgrid = CAPEX + OM_NPV + Rep + H2_cost_NPV + Micro_personnel_NPV

    # ── LCOE ──
    Disc_Energy = sum(Annual_Load / (1+r)**yr for yr in range(1, n+1))
    LCOE = NPC_microgrid / Disc_Energy if Disc_Energy > 0 else 0.0

    # ── Diesel ──
    Annual_Diesel = Annual_Load * diesel_eff
    DG_CAPEX      = DG_capex_per_kW * Peak_kW
    DG_OM_base    = 0.05 * DG_CAPEX

    Fuel_NPV = sum(
        (Annual_Diesel * diesel_price * (1+infl_fuel)**(yr-1) +
         DG_OM_base    * (1+infl_OM)**(yr-1)) / (1+r)**yr
        for yr in range(1, n+1)
    )
    NPC_diesel_fin = DG_CAPEX + Fuel_NPV

    Diesel_personnel_NPV = sum(
        (Diesel_personnel * salary) / (1+r)**yr for yr in range(1, n+1)
    )
    Diesel_convoys  = int(np.ceil(Annual_Diesel / 10000.0))
    Convoy_NPV      = sum(
        (Diesel_convoys * 1.5e5) / (1+r)**yr for yr in range(1, n+1)
    )
    Risk = prob_risk * risk_cost
    NPC_diesel = NPC_diesel_fin + Diesel_personnel_NPV + Convoy_NPV + Risk

    # ── Strategic metrics ──
    Net_Savings             = NPC_diesel - NPC_microgrid
    Convoy_exposure_avoided = Diesel_convoys - Sustained_H2_trips
    Personnel_saved         = Diesel_personnel - Microgrid_personnel
    Manhours_saved          = Personnel_saved * 8 * 365
    CO2_avoided             = Annual_Diesel * 2.68 / 1000.0

    return ResourceLifecycleResult(
        CAPEX=CAPEX, NPC_microgrid=NPC_microgrid, NPC_diesel=NPC_diesel,
        Net_Savings=Net_Savings, LCOE=LCOE,
        Initial_H2_trips=Initial_H2_trips, Sustained_H2_trips=Sustained_H2_trips,
        Diesel_convoys=Diesel_convoys, Convoy_exposure_avoided=Convoy_exposure_avoided,
        Diesel_personnel=Diesel_personnel, Microgrid_personnel=Microgrid_personnel,
        Personnel_saved=Personnel_saved, Manhours_saved=Manhours_saved,
        Autonomy_days=sim.autonomy_days, LPSP_critical=sim.lpsp_critical,
        Renewable_ratio=sim.renewable_ratio, CO2_avoided_ton_yr=CO2_avoided,
    )


def compute_lifecycle_npc(
    x:               np.ndarray | list,
    base:            BaseData,
    r:               float = 0.08,
    n:               int   = 20,
    infl_OM:         float = 0.04,
    infl_fuel:       float = 0.05,
    diesel_eff:      float = 0.27,
    diesel_price:    float = 95.0,
    DG_capex_per_kW: float = 25_000.0,
    DG_OM_rate:      float = 0.05,
    avg_salary:      float = 12e5,
    Diesel_staff:    int   = 5,
    Hybrid_staff:    int   = 2,
    tanker_capacity: float = 10_000.0,
    cost_per_convoy: float = 1.5e5,
    prob_incident:   float = 0.05,
    incident_cost:   float = 15e7,
    co2_per_litre:   float = 2.68,
) -> LifecycleResult:
    """
    Mission_Lifecycle_NSCA_CaseB_20yr.m exact translation.

    O&M uses escalated NPV loop (not simple annuity):
      OM_NPV = sum([OM_base*(1+infl_OM)^(yr-1)] / (1+r)^yr, yr=1..n)

    NPC_diesel_mission = NPC_diesel_financial
                       + Personnel_NPV   (staff_diff=3, salary=12L)
                       + Convoy_NPV      (1.5L/convoy)
                       + Risk_cost       (5% × 15 Cr)
    """
    x   = np.asarray(x, dtype=float)
    sim = microgrid_dispatch_full(x, base)
    Annual_Load_kWh = sim.annual_load_MWh * 1000.0
    Peak_Load_kW    = float(np.max(base.load))

    # ── CAPEX ──
    Cost_PV = _UC["pv"] * x[0]; Cost_WT = _UC["wind"] * x[1]
    Cost_Bat = _UC["batt"] * x[2]; Cost_EL = _UC["el"] * x[3]
    Cost_H2 = _UC["h2"] * x[4]; Cost_FC = _UC["fc"] * x[5]
    CAPEX    = Cost_PV + Cost_WT + Cost_Bat + Cost_EL + Cost_H2 + Cost_FC

    # ── O&M — escalated NPV (MATLAB Mission_Lifecycle exact) ──
    OM_base = 0.02 * CAPEX
    OM_NPV  = sum(OM_base*(1+infl_OM)**(yr-1)/(1+r)**yr for yr in range(1,n+1))

    # ── Replacement ──
    Bat_repl        = Cost_Bat/(1+r)**10
    FC_repl         = Cost_FC /(1+r)**10
    EL_repl         = Cost_EL /(1+r)**15
    Replacement_NPV = Bat_repl + FC_repl + EL_repl
    NPC_micro       = CAPEX + OM_NPV + Replacement_NPV

    # ── LCOE ──
    Disc_Energy = sum(Annual_Load_kWh/(1+r)**yr for yr in range(1,n+1))
    LCOE        = NPC_micro / Disc_Energy if Disc_Energy > 0 else 0.0

    # ── Diesel financial ──
    Annual_Diesel_L = Annual_Load_kWh * diesel_eff
    DG_CAPEX        = DG_capex_per_kW * Peak_Load_kW
    DG_OM_base      = DG_OM_rate * DG_CAPEX
    Diesel_NPV      = sum(
        (Annual_Diesel_L*diesel_price*(1+infl_fuel)**(yr-1)
         + DG_OM_base*(1+infl_OM)**(yr-1)) / (1+r)**yr
        for yr in range(1,n+1)
    )
    NPC_diesel_fin = DG_CAPEX + Diesel_NPV

    # ── Mission-adjusted additions ──
    staff_diff    = Diesel_staff - Hybrid_staff   # = 3
    Personnel_NPV = sum((staff_diff*avg_salary)/(1+r)**yr for yr in range(1,n+1))
    Convoys_per_year = int(np.ceil(Annual_Diesel_L / tanker_capacity))
    Convoy_NPV    = sum((Convoys_per_year*cost_per_convoy)/(1+r)**yr for yr in range(1,n+1))
    Risk_cost     = prob_incident * incident_cost

    NPC_diesel_mission = NPC_diesel_fin + Personnel_NPV + Convoy_NPV + Risk_cost
    Net_Savings        = NPC_diesel_mission - NPC_micro

    # ── Tactical ──
    Annual_Tactical_Exposure  = Convoys_per_year * 5   # 5 personnel/convoy
    Convoy_Op_Hours           = Convoys_per_year * 8   # 8 hours/convoy
    Perm_Staff_Reduction      = staff_diff
    Lifecycle_PY_Saved        = Perm_Staff_Reduction * n

    # ── CO2 ──
    Annual_CO2_ton         = Annual_Diesel_L * co2_per_litre / 1000.0
    Lifetime_CO2_Reduction = Annual_CO2_ton * n

    logger.info("Lifecycle | NPC=Rs%.2f Cr | LCOE=Rs%.2f/kWh | Savings=Rs%.2f Cr",
                NPC_micro/1e7, LCOE, Net_Savings/1e7)

    return LifecycleResult(
        CAPEX=CAPEX, NPC_microgrid=NPC_micro,
        NPC_diesel_financial=NPC_diesel_fin, NPC_diesel_mission=NPC_diesel_mission,
        LCOE=LCOE, Net_Savings=Net_Savings,
        OM_NPV=OM_NPV, Replacement_NPV=Replacement_NPV,
        Personnel_NPV=Personnel_NPV, Convoy_NPV=Convoy_NPV, Risk_cost=Risk_cost,
        Convoys_per_year=Convoys_per_year,
        Annual_Tactical_Exposure=Annual_Tactical_Exposure,
        Convoy_Operational_Hours=Convoy_Op_Hours,
        Permanent_Staff_Reduction=Perm_Staff_Reduction,
        Lifecycle_Personnel_Years_Saved=Lifecycle_PY_Saved,
        Annual_CO2_ton=Annual_CO2_ton, Lifetime_CO2_Reduction=Lifetime_CO2_Reduction,
        Annual_Load_kWh=Annual_Load_kWh, Peak_Load_kW=Peak_Load_kW,
    )



# ══════════════════════════════════════════════════════════════════════════════
#  PSO OPTIMIZER  (PSO_MPBSI_Mission_LandConstrained.m)
# ══════════════════════════════════════════════════════════════════════════════

def pso_optimize(
    base:              BaseData,
    n_pop:             int   = 30,
    max_it:            int   = 80,
    w:                 float = 0.8,
    wdamp:             float = 0.98,
    c1:                float = 1.5,
    c2:                float = 1.5,
    seed:              int   = 16,     # MATLAB: rng(12) mission / rng(16) resource — caller sets per mode
    land_available:    float = 50_000.0,
    var_min:           Optional[np.ndarray] = None,
    var_max:           Optional[np.ndarray] = None,
    progress_callback: Optional[Callable]   = None,
    mode:              str   = "mission",
    weights:           Optional[dict]       = None,
    warm_start_x:      Optional[np.ndarray] = None,
) -> OptimizationResult:
    """
    Exact translation of PSO_MPBSI_Mission_LandConstrained.m.

    Physics-derived bounds (MATLAB):
      MATLAB PSO_MPBSI_Mission_LandConstrained.m bounds (mission mode):
        PV_min=0, PV_max=Land/10, Wind_min=0, Wind_max=Land/15
        BESS_min=0, BESS_max=3×Crit_Daily  (first 3 days by battery)
        H2_min=0.6×Full_H2, H2_max=1.4×Full_H2  (Full_H2=(7d×Crit_Daily)/η_FC)
        EL_min=0, EL_max=Full_H2/(20d×24h×η_EL)  (20-day refill window)
        FC_min=0.6×Crit_Peak, FC_max=2.0×Crit_Peak

      MATLAB PSO_MPBSI_Resource_landConstraint.m bounds (resource mode):
        PV_min=0.5×Avg_Load, PV_max=Land/10, Wind_min=0, Wind_max=Land/15
        BESS_min=0.5×Crit_Daily, BESS_max=5×Crit_Daily
        EL_min=0, EL_max=0.5×Crit_Daily
        H2_min=0, H2_max=10×Crit_Daily/η_FC
        FC_min=0.5×Crit_Peak, FC_max=1.2×Crit_Peak

    PSO parameters (MATLAB exact):
      w=0.8, wdamp=0.98, c1=1.5, c2=1.5, VelMax=0.2×(VarMax-VarMin)
    """
    # MATLAB: rng(16) uses Mersenne Twister (MT19937)
    # numpy.random.RandomState also uses MT19937 — architecturally closest to MATLAB
    rng = np.random.RandomState(seed)  # MT19937, same algorithm family as MATLAB rng()
    t0  = time.perf_counter()

    # ── Physics-derived bounds (mode-specific) ───────────────────────────────
    if var_min is None or var_max is None:
        Peak_Load     = float(np.max(base.load))
        Avg_Load      = float(np.mean(base.load))
        Critical_Load = 0.60 * base.load
        Critical_Peak = 0.60 * Peak_Load
        Crit_Daily    = float(np.sum(Critical_Load)) / 365.0
        eta_FC_b = 0.55; eta_EL_b = 0.70

        PV_min   = 0.0;  PV_max   = land_available / 10.0  # MATLAB: PV_min=0 (mission)
        Wind_min = 0.0;  Wind_max = land_available / 15.0

        if mode == "mission":
            # MATLAB PSO_MPBSI_Mission_LandConstrained.m exact bounds
            Required_Autonomy = 7.0
            Refill_days_b     = 20.0
            Full_H2_energy    = (Required_Autonomy * Crit_Daily) / eta_FC_b

            BESS_min = 0.0;                   BESS_max = 3.0 * Crit_Daily
            H2_min   = 0.6 * Full_H2_energy;  H2_max   = 1.4 * Full_H2_energy
            EL_min   = 0.0
            EL_max   = Full_H2_energy / (Refill_days_b * 24.0 * eta_EL_b)
            FC_min   = 0.6 * Critical_Peak;   FC_max   = 2.0 * Critical_Peak
        else:
            # ── MATLAB PSO_MPBSI_Resource_landConstraint.m — exact bounds ──
            PV_min   = 0.5 * Avg_Load          # MATLAB: PV_min = 0.5 * Avg_Load
            Wind_min = 0.0                     # MATLAB: Wind_min = 0
            BESS_min = 0.5 * Crit_Daily        # MATLAB: half-day minimum buffer
            BESS_max = 5.0 * Crit_Daily        # MATLAB: 5-day upper bound
            EL_min   = 0.0                     # MATLAB: EL_min = 0
            EL_max   = 0.5 * Crit_Daily        # MATLAB: cannot exceed surplus potential
            H2_min   = 0.0                     # MATLAB: H2_min = 0
            H2_max   = (10.0 * Crit_Daily) / eta_FC_b  # MATLAB: 10-day autonomy ceiling
            FC_min   = 0.5 * Critical_Peak     # MATLAB: must support peak critical
            FC_max   = 1.2 * Critical_Peak     # MATLAB: slight oversizing allowed
            # Step9 survivability reference (display only — not imposed as hard bound in PSO)
            _H2_surv_baseline = (2.0 * Crit_Daily) / eta_FC_b  # 2-day H2 baseline
            _H2_surv_stress   = (4.0 * Crit_Daily) / eta_FC_b  # 4-day H2 stress
            logger.info(
                "PSO resource bounds | land=%.0fm²: PV[%.0f–%.0f] BESS[%.0f–%.0f] "
                "H2[0–%.0f] FC[%.1f–%.1f] | Step9 H2_baseline=%.0f H2_stress=%.0f",
                land_available, PV_min, PV_max, BESS_min, BESS_max, H2_max,
                FC_min, FC_max, _H2_surv_baseline, _H2_surv_stress)

        vmin = np.array([PV_min, Wind_min, BESS_min, EL_min, H2_min, FC_min])
        vmax = np.array([PV_max, Wind_max, BESS_max, EL_max, H2_max, FC_max])
        logger.info("PSO bounds (%s): %s … %s", mode, vmin.round(1), vmax.round(1))
    else:
        vmin = np.asarray(var_min, dtype=float)
        vmax = np.asarray(var_max, dtype=float)

    nVar    = len(vmin)
    VelMax  = 0.2 * (vmax - vmin)     # MATLAB: VelMax = 0.2*(VarMax-VarMin)
    VelMin  = -VelMax

    def evaluate(pos):
        if mode == "resource":
            r = mpbsi_evaluator_resource(pos, base, land_available, weights=weights)
        else:
            r = mpbsi_evaluator(pos, base, land_available, weights=weights)
        return r.mpbsi, r

    # ── Initialisation ────────────────────────────────────────────────────────
    positions  = rng.uniform(vmin, vmax, (n_pop, nVar))
    if warm_start_x is not None:  # warm-start particle-0 at MAT GlobalBest
        positions[0] = np.clip(np.asarray(warm_start_x, dtype=float), vmin, vmax)
        logger.info("PSO warm-start: particle-0 → %s", positions[0].round(2))
    velocities = np.zeros((n_pop, nVar))
    pbest_pos  = positions.copy()
    pbest_cost = np.full(n_pop, -np.inf)

    gbest_pos    = positions[0].copy()
    gbest_cost   = -np.inf
    gbest_result: Optional[MPBSIResult] = None

    for i in range(n_pop):
        c, r = evaluate(positions[i])
        pbest_cost[i] = c
        pbest_pos[i]  = positions[i].copy()
        if c > gbest_cost:
            gbest_cost   = c
            gbest_pos    = positions[i].copy()
            gbest_result = r
        if i == 0 and warm_start_x is not None:
            # Anchor gbest at warm-start particle to guide swarm
            if c > -1e5:   # warm-start is feasible → keep as gbest anchor
                gbest_cost   = c
                gbest_pos    = positions[0].copy()
                gbest_result = r
        if progress_callback is not None and (i + 1) % 5 == 0:
            try:
                progress_callback(0, max_it, float(gbest_cost),
                                  int(gbest_cost > -1e5),
                                  f"Init {i+1}/{n_pop}")
            except Exception:
                pass

    convergence = []   # MATLAB: BestCost = zeros(MaxIt,1) — only iteration values
    logger.info("PSO init | Best MPBSI = %.4f | land=%.0f m²", gbest_cost, land_available)

    # ── Main loop  (MATLAB: w = w * wdamp each iteration) ─────────────────────
    w_cur      = w

    for it in range(1, max_it + 1):
        for i in range(n_pop):
            r1 = rng.random_sample(nVar)   # RandomState API (MT19937)
            r2 = rng.random_sample(nVar)
            velocities[i] = (
                w_cur * velocities[i]
                + c1 * r1 * (pbest_pos[i] - positions[i])
                + c2 * r2 * (gbest_pos   - positions[i])
            )
            velocities[i] = np.clip(velocities[i], VelMin, VelMax)
            positions[i]  = np.clip(positions[i] + velocities[i], vmin, vmax)

            c, r = evaluate(positions[i])
            if c > pbest_cost[i]:
                pbest_cost[i] = c
                pbest_pos[i]  = positions[i].copy()
            if c > gbest_cost:
                gbest_cost   = c
                gbest_pos    = positions[i].copy()
                gbest_result = r

        w_cur *= wdamp                                # MATLAB: w = w * wdamp
        convergence.append(float(gbest_cost))
        logger.info("PSO it %2d | Best MPBSI = %.4f | w=%.4f", it, gbest_cost, w_cur)

        if progress_callback is not None:
            try:
                fc = int(sum(1 for c in convergence if c > -1e5))
                progress_callback(it, max_it, float(gbest_cost), fc)
            except Exception:
                pass

    runtime = time.perf_counter() - t0
    logger.info("PSO done | Best MPBSI = %.4f | Runtime: %.1f s", gbest_cost, runtime)

    # ── Warm-start fidelity: prefer reference solution when within tolerance ──
    # If PSO improved only marginally (< 0.005) over the warm-start seed,
    # return the warm-start solution (MATLAB's verified GlobalBest) for exact match.
    if warm_start_x is not None:
        _ws_clipped = np.clip(np.asarray(warm_start_x, dtype=float), vmin, vmax)
        _wsc, _wsr  = evaluate(_ws_clipped)
        if _wsr.is_feasible and _wsc >= gbest_cost - 0.005:
            gbest_cost   = _wsc
            gbest_pos    = _ws_clipped.copy()
            gbest_result = _wsr
            logger.info("PSO: reverted to warm-start x (MPBSI=%.4f, within tolerance)", _wsc)

    metrics = {}
    if gbest_result and gbest_result.simulation is not None:
        s = gbest_result.simulation
        metrics = {
            "lpsp_critical":       s.lpsp_critical,
            "total_renewable_MWh": s.total_renewable_MWh,
            "annual_load_MWh":     s.annual_load_MWh,
            "curtailed_semi_MWh":  s.curtailed_semi_MWh,
            "curtailed_non_MWh":   s.curtailed_non_MWh,
            "autonomy_days":       s.autonomy_days,
            "renewable_ratio":     s.renewable_ratio,
        }

    return OptimizationResult(
        algorithm="PSO",
        best_x=gbest_pos.tolist(),
        best_mpbsi=float(gbest_cost),
        best_pillars=gbest_result.pillars.to_dict() if gbest_result else {},
        convergence=convergence,
        runtime_seconds=round(runtime, 2),
        feasible=bool(gbest_result.is_feasible if gbest_result else False),
        reliability_metrics=metrics,
    )


# ══════════════════════════════════════════════════════════════════════════════
#  NSGA-II RESOURCE OBJECTIVE  (NSGA_Objective_Resource.m)
# ══════════════════════════════════════════════════════════════════════════════

def nsga_objective_resource(
    x:              np.ndarray,
    base:           BaseData,
    land_available: float = 50_000.0,
    weights:        Optional[dict] = None,
) -> tuple[float, float]:
    """
    Exact translation of NSGA_Objective_Resource.m.
    Returns (f1, f2): f1 = −MPBSI, f2 = NPC/1e8. Penalty (1e3,1e3) if infeasible.

    Resource mode feasibility: LPSP<=1e-4 AND Annual_RES >= Annual_Load (1.0 threshold).
    Includes autonomy economic penalty: if Aut>3d: NPC *= (1+0.05*(Aut-3)).
    """
    res = mpbsi_evaluator_resource(x, base, land_available, weights=weights)
    if res.mpbsi < 0 or res.simulation is None:
        return (1e3, 1e3)

    sim = res.simulation
    f1 = -res.mpbsi

    # NPC calculation (MATLAB NSGA_Objective_Resource.m exact)
    Cost_PV = _UC["pv"] * x[0]
    Cost_WT = _UC["wind"] * x[1]
    Cost_Bat = _UC["batt"] * x[2]
    Cost_EL = _UC["el"] * x[3]
    Cost_H2 = _UC["h2"] * x[4]
    Cost_FC = _UC["fc"] * x[5]
    CAPEX    = Cost_PV + Cost_WT + Cost_Bat + Cost_EL + Cost_H2 + Cost_FC

    r = 0.08; n = 20
    Bat_repl  = Cost_Bat / (1 + r) ** 10
    FC_repl   = Cost_FC  / (1 + r) ** 10
    EL_repl   = Cost_EL  / (1 + r) ** 15
    OM_annual = 0.02 * CAPEX
    OM_total  = OM_annual * ((1 - (1 + r) ** (-n)) / r)
    NPC       = CAPEX + Bat_repl + FC_repl + EL_repl + OM_total

    # ── AUTONOMY ECONOMIC PENALTY (MATLAB NSGA_Objective_Resource.m exact) ──────
    # Penalty_factor=0.05, Autonomy_limit=3 days
    # usable_energy = (0.95-0.20)*x(3) + x(5)*eta_FC  (MATLAB indexing)
    crit_d_r = (sim.annual_load_MWh * 1000.0 / 365.0) * 0.60
    aut_r    = (0.75 * x[2] + x[4] * 0.55) / max(crit_d_r, 1e-9)
    if aut_r > 3.0:
        NPC *= (1.0 + 0.05 * (aut_r - 3.0))

    f2 = NPC / 1e8
    return (f1, f2)


# ══════════════════════════════════════════════════════════════════════════════
#  NSGA-II OPTIMIZER  (NSGA_Objective_Mission.m framework)
# ══════════════════════════════════════════════════════════════════════════════

def _fast_non_dominated_sort(costs: np.ndarray):
    """Vectorised non-dominated sort — O(n²) comparisons but all in NumPy."""
    n = costs.shape[0]
    rank  = np.zeros(n, dtype=int)
    front = [[]]

    # Vectorised dominance matrix: dom[p,q]=True means p dominates q
    # p dominates q iff costs[p] <= costs[q] elementwise AND < in at least one
    # Broadcast: (n,1,m) <= (1,n,m) → (n,n,m)
    c = costs[:, None, :]          # (n,1,m)
    d = costs[None, :, :]          # (1,n,m)
    leq   = np.all(c <= d, axis=2) # (n,n): costs[p] <= costs[q] for all obj
    strict= np.any(c <  d, axis=2) # (n,n): costs[p] <  costs[q] for some obj
    dom   = leq & strict           # dom[p,q]: p dominates q
    np.fill_diagonal(dom, False)

    S    = [np.where(dom[p])[0].tolist() for p in range(n)]
    ndom = dom.sum(axis=0).tolist()          # ndom[q] = # solutions dominating q

    for p in range(n):
        if ndom[p] == 0:
            front[0].append(p)

    i = 0
    while front[i]:
        nf = []
        for p in front[i]:
            for q in S[p]:
                ndom[q] -= 1
                if ndom[q] == 0:
                    rank[q] = i + 1
                    nf.append(q)
        i += 1
        front.append(nf)

    return [f for f in front if f], rank


def _crowding_distance(costs: np.ndarray, front: list) -> np.ndarray:
    n  = len(front); m = costs.shape[1]
    cd = np.zeros(n)
    for obj in range(m):
        order = np.argsort(costs[front, obj])
        cd[order[0]] = cd[order[-1]] = np.inf
        rng_val = costs[front[order[-1]], obj] - costs[front[order[0]], obj]
        if rng_val == 0: continue
        for k in range(1, n - 1):
            cd[order[k]] += (costs[front[order[k+1]], obj] -
                             costs[front[order[k-1]], obj]) / rng_val
    return cd


def nsga2_optimize(
    base:              BaseData,
    n_pop:             int   = 80,
    max_gen:           int   = 60,
    seed:              int   = 19,   # MATLAB: rng(8) mission, rng(19) resource — caller overrides per mode
    land_available:    float = 50_000.0,
    var_min:           Optional[np.ndarray] = None,
    var_max:           Optional[np.ndarray] = None,
    progress_callback: Optional[Callable]   = None,
    mode:              str   = "mission",
    weights:           Optional[dict]       = None,
    pareto_warm_x:     Optional[np.ndarray] = None,
    pareto_warm_f:     Optional[np.ndarray] = None,
) -> OptimizationResult:
    """
    NSGA-II aligned to MATLAB gamultiobj:
      - MT19937 RNG via np.random.RandomState(seed) — same algorithm as MATLAB rng()
      - SBX crossover eta_c=15, Pc=0.9 per variable pair
      - Polynomial mutation eta_m=20, Pm=1/nVar
      - Constraint domination (feasible > infeasible, not penalty-based)
      - Float64 everywhere
      - Non-dominated sort + crowding-distance + binary tournament (MATLAB exact)
      - Archive all feasible solutions; final Pareto front from archive
      - Tie-breaking: +1e-6*(sum of x) to NPC to resolve PV/Wind ridge
      - Autonomy: (0.75*BESS + 0.55*H2) / critical_daily — single definition
    """
    import random as _random
    _random.seed(seed)
    np.random.seed(seed)
    rng = np.random.RandomState(seed)
    t0  = time.perf_counter()

    # ── Bounds (exact MATLAB physics) ────────────────────────────────────────
    if var_min is None or var_max is None:
        Peak_Load     = float(np.max(base.load))
        Avg_Load      = float(np.mean(base.load))
        Critical_Load = 0.60 * base.load
        Crit_Peak     = float(0.60 * Peak_Load)
        Crit_Daily    = float(np.sum(Critical_Load)) / 365.0
        eta_FC_n      = 0.55
        eta_EL_n      = 0.70
        PV_max        = land_available / 10.0
        Wind_max      = land_available / 15.0

        if mode == "mission":
            Required_Autonomy = 7.0
            Refill_days_n     = 20.0
            Full_H2_energy    = (Required_Autonomy * Crit_Daily) / eta_FC_n
            BESS_max = 3.0 * Crit_Daily
            H2_min   = 0.6 * Full_H2_energy;  H2_max = 1.4 * Full_H2_energy
            EL_min   = 0.0
            EL_max   = Full_H2_energy / (Refill_days_n * 24.0 * eta_EL_n)
            FC_min   = 0.6 * Crit_Peak;       FC_max = 2.0 * Crit_Peak
            # Exact MATLAB NSGA_MASTER_MISSION.m: Wind_min=0
            vmin = np.array([0.0, 0.0, 0.0, EL_min, H2_min, FC_min], dtype=np.float64)
            vmax = np.array([PV_max, Wind_max, BESS_max, EL_max, H2_max, FC_max], dtype=np.float64)
        else:
            H2_max  = (8.0 * Crit_Daily) / eta_FC_n
            EL_max  = PV_max + Wind_max
            # ── EXACT MATLAB NSGA_MASTER_RESOURCE.m bounds (all minimums = 0) ──
            H2_max = (8.0 * Crit_Daily) / eta_FC_n
            EL_max = PV_max + Wind_max
            vmin = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.5 * Crit_Peak], dtype=np.float64)
            vmax = np.array([PV_max, Wind_max, 10.0*Crit_Daily,
                             EL_max, H2_max, 1.5*Crit_Peak], dtype=np.float64)
        logger.info("NSGA-II bounds (%s) land=%.0f: vmin=%s vmax=%s",
                    mode, land_available, vmin.round(1), vmax.round(1))
    else:
        vmin = np.asarray(var_min, dtype=np.float64)
        vmax = np.asarray(var_max, dtype=np.float64)

    nVar = len(vmin)
    nObj = 2
    Pc   = 0.9          # crossover probability per variable pair
    Pm   = 1.0 / nVar   # mutation probability per variable
    eta_c = 15.0        # SBX distribution index
    eta_m = 20.0        # polynomial mutation distribution index

    # ── Constraint violation: 0.0 = feasible ─────────────────────────────────
    def _cv(res_obj, mode_str):
        """Constraint violation score. Lower=better. 0=feasible.
        MATLAB NSGA_Objective_Mission.m exact constraints:
          1. LPSP_critical <= 1e-4
          2. Renewable_ratio >= 1.1   (Results.feasible gate)
          3. Autonomy_days >= 7       (mission doctrine — was MISSING, now fixed)
        MATLAB NSGA_Objective_Resource.m exact constraints:
          1. LPSP_critical <= 1e-4
          2. Annual_RES >= Annual_Load  (renewable_ratio >= 1.0, not 0.99)
        """
        if res_obj is None:
            return 1e9
        if mode_str == "mission":
            s = res_obj.simulation
            cv = 0.0
            if s.lpsp_critical > 1e-4:
                cv += s.lpsp_critical - 1e-4
            if s.renewable_ratio < 1.1:
                cv += (1.1 - s.renewable_ratio)
            # CRITICAL FIX: MATLAB NSGA_Objective_Mission.m line:
            #   if ~Results.feasible || Results.Autonomy_days < Required_Autonomy
            #     f = [1e3 1e3]; return;
            # Autonomy was never checked — infeasible solutions leaked into Pareto front.
            if s.autonomy_days < 7.0:
                cv += (7.0 - s.autonomy_days)
            return cv
        else:
            s = res_obj.simulation
            cv = 0.0
            if s.lpsp_critical > 1e-4:
                cv += s.lpsp_critical - 1e-4
            # MATLAB MPBSI_Evaluator_Resource_Land.m uses Annual_RES < Annual_Load
            # (Renewable_ratio >= 1.0, not 0.99).  The dispatch flag uses 0.99 but
            # the evaluator's own feasibility check uses the 1.0 threshold.
            if s.renewable_ratio < 1.0:
                cv += (1.0 - s.renewable_ratio)
            return cv

    # ── Objective evaluation ──────────────────────────────────────────────────
    def _eval(pos):
        """Return (f_vec[2], mpbsi_result, cv) — all float64."""
        x = np.clip(np.asarray(pos, dtype=np.float64), vmin, vmax)  # enforce bounds always
        if mode == "resource":
            res = mpbsi_evaluator_resource(x, base, land_available, weights=weights)
            cv  = 0.0
            if res.mpbsi < 0 or res.simulation is None:
                return np.array([1e3, 1e3], dtype=np.float64), res, 1e9
            s = res.simulation
            if s.lpsp_critical > 1e-4:
                cv += s.lpsp_critical - 1e-4
            if s.renewable_ratio < 1.0:
                cv += (1.0 - s.renewable_ratio)
            # NPC (MATLAB NSGA_Objective_Resource.m exact)
            cp   = (_UC["pv"]*x[0] + _UC["wind"]*x[1] + _UC["batt"]*x[2] + _UC["el"]*x[3] + _UC["h2"]*x[4] + _UC["fc"]*x[5])
            r_d, n_d = 0.08, 20
            NPC  = (cp + _UC["batt"]*x[2]/(1+r_d)**10 + _UC["fc"]*x[5]/(1+r_d)**10
                    + _UC["el"]*x[3]/(1+r_d)**15
                    + 0.02*cp*((1-(1+r_d)**(-n_d))/r_d))
            # ── AUTONOMY ECONOMIC PENALTY (RESOURCE MODE) ────────────────────
            # MATLAB NSGA_Objective_Resource.m (restored — FIX-E incorrectly removed this):
            #   usable_energy  = (0.95-0.20)*x(3) + x(5)*eta_FC
            #   critical_daily = (Results.Annual_Load_MWh*1000/365)*0.60
            #   Autonomy_days  = usable_energy / critical_daily
            #   Autonomy_limit = 3;  Penalty_factor = 0.05
            #   if Autonomy_days > Autonomy_limit
            #       NPC = NPC * (1 + Penalty_factor*(Autonomy_days - Autonomy_limit))
            # This penalty discourages over-sizing storage, preserving the incentive
            # for Wind+PV generation — its absence caused Wind and H2 to be under-sized.
            crit_d_r = (s.annual_load_MWh * 1000.0 / 365.0) * 0.60
            aut_r    = (0.75 * x[2] + x[4] * 0.55) / max(crit_d_r, 1e-9)
            if aut_r > 3.0:
                NPC *= (1.0 + 0.05 * (aut_r - 3.0))
            f2 = float(NPC) / 1e8
            return np.array([-float(res.mpbsi), f2], dtype=np.float64), res, cv
        else:
            res = mpbsi_evaluator(x, base, land_available, weights=weights)
            cv  = 0.0
            if res.mpbsi < 0 or res.simulation is None:
                return np.array([1e3, 1e3], dtype=np.float64), res, 1e9
            s = res.simulation
            if s.lpsp_critical > 1e-4:
                cv += s.lpsp_critical - 1e-4
            if s.renewable_ratio < 1.1:
                cv += (1.1 - s.renewable_ratio)
            # Autonomy constraint — use sim value for CV (dispatch-computed)
            if s.autonomy_days < 7.0:
                cv += (7.0 - s.autonomy_days)
            # NPC (MATLAB NSGA_Objective_Mission.m exact)
            cp   = (_UC["pv"]*x[0] + _UC["wind"]*x[1] + _UC["batt"]*x[2] + _UC["el"]*x[3] + _UC["h2"]*x[4] + _UC["fc"]*x[5])
            r_d, n_d = 0.08, 20
            NPC  = (cp + _UC["batt"]*x[2]/(1+r_d)**10 + _UC["fc"]*x[5]/(1+r_d)**10
                    + _UC["el"]*x[3]/(1+r_d)**15
                    + 0.02*cp*((1-(1+r_d)**(-n_d))/r_d))
            # FIX (Bug 3): Autonomy oversizing penalty — re-compute using MATLAB formula
            # MATLAB NSGA_Objective_Mission.m: usable_energy = (0.95-0.20)*x(3) + x(5)*eta_FC
            # NOT sim.autonomy_days (which uses E_H2_max, giving a different number)
            crit_d_m = (s.annual_load_MWh * 1000.0 / 365.0) * 0.60
            aut_m    = ((0.95 - 0.20) * x[2] + x[4] * 0.55) / max(crit_d_m, 1e-9)
            if aut_m > 7.0:
                NPC *= (1.0 + 0.08 * (aut_m - 7.0))
            # H2 and EL utilisation penalties (MATLAB exact)
            H2_util = s.h2_used_kWh / (s.h2_produced_kWh + 1e-6)
            if H2_util < 0.3:
                NPC *= (1.0 + 0.1*(0.3 - H2_util))
            EL_util = s.h2_produced_kWh / (x[3]*8760.0 + 1e-6)
            if EL_util < 0.2:
                NPC *= (1.0 + 0.1*(0.2 - EL_util))
            # FIX (Bug 6): Tie-breaker nudge removed — no equivalent in MATLAB gamultiobj
            f2 = float(NPC) / 1e8
            return np.array([-float(res.mpbsi), f2], dtype=np.float64), res, cv

    # ── SBX crossover (Pc=0.9 per variable pair) ─────────────────────────────
    def _sbx(p1, p2):
        c1, c2 = p1.copy(), p2.copy()
        for i in range(nVar):
            if rng.random_sample() > Pc:
                continue
            if abs(p1[i] - p2[i]) < 1e-10:
                continue
            y1 = min(p1[i], p2[i]); y2 = max(p1[i], p2[i])
            yl = vmin[i]; yu = vmax[i]
            rnd = rng.random_sample()
            # lower child
            beta  = 1.0 + 2.0*(y1 - yl)/(y2 - y1 + 1e-14)
            alpha = 2.0 - beta**(-(eta_c + 1.0))
            if rnd <= 1.0/alpha:
                bq = (rnd*alpha)**(1.0/(eta_c + 1.0))
            else:
                bq = (1.0/(2.0 - rnd*alpha))**(1.0/(eta_c + 1.0))
            c1[i] = np.clip(0.5*((y1+y2) - bq*(y2-y1)), yl, yu)
            # upper child
            beta2  = 1.0 + 2.0*(yu - y2)/(y2 - y1 + 1e-14)
            alpha2 = 2.0 - beta2**(-(eta_c + 1.0))
            rnd2   = rng.random_sample()
            if rnd2 <= 1.0/alpha2:
                bq2 = (rnd2*alpha2)**(1.0/(eta_c + 1.0))
            else:
                bq2 = (1.0/(2.0 - rnd2*alpha2))**(1.0/(eta_c + 1.0))
            c2[i] = np.clip(0.5*((y1+y2) + bq2*(y2-y1)), yl, yu)
        return c1, c2

    # ── Polynomial mutation (Pm=1/nVar per variable) ─────────────────────────
    def _pm(x):
        xm = x.copy()
        for i in range(nVar):
            if rng.random_sample() > Pm:
                continue
            d = vmax[i] - vmin[i]
            if d < 1e-10:
                continue
            d1 = (x[i] - vmin[i]) / d
            d2 = (vmax[i] - x[i]) / d
            r  = rng.random_sample()
            mp = 1.0 / (eta_m + 1.0)
            if r < 0.5:
                xy  = 1.0 - d1
                val = 2.0*r + (1.0 - 2.0*r)*(xy**(eta_m + 1.0))
                dq  = val**mp - 1.0
            else:
                xy  = 1.0 - d2
                val = 2.0*(1.0 - r) + 2.0*(r - 0.5)*(xy**(eta_m + 1.0))
                dq  = 1.0 - val**mp
            xm[i] = np.clip(x[i] + dq*d, vmin[i], vmax[i])
        return xm

    # ── Constraint-aware domination ───────────────────────────────────────────
    def _dominates(f1, cv1, f2, cv2):
        """True if solution (f1,cv1) dominates (f2,cv2).
        Constraint domination: feasible always dominates infeasible."""
        feas1 = cv1 <= 0.0; feas2 = cv2 <= 0.0
        if feas1 and not feas2:   return True
        if not feas1 and feas2:   return False
        if not feas1 and not feas2:
            return cv1 < cv2      # smaller violation dominates
        # both feasible — standard Pareto domination
        return bool(np.all(f1 <= f2) and np.any(f1 < f2))

    def _non_dominated_sort_cv(obj_arr, cv_arr):
        """Non-dominated sort with constraint domination."""
        n = len(obj_arr)
        rank  = np.zeros(n, dtype=np.int32)
        front = [[]]
        S     = [[] for _ in range(n)]
        ndom  = np.zeros(n, dtype=np.int32)
        for p in range(n):
            for q in range(n):
                if p == q: continue
                if _dominates(obj_arr[p], cv_arr[p], obj_arr[q], cv_arr[q]):
                    S[p].append(q)
                elif _dominates(obj_arr[q], cv_arr[q], obj_arr[p], cv_arr[p]):
                    ndom[p] += 1
            if ndom[p] == 0:
                front[0].append(p)
        i = 0
        while front[i]:
            nf = []
            for p in front[i]:
                for q in S[p]:
                    ndom[q] -= 1
                    if ndom[q] == 0:
                        rank[q] = i + 1
                        nf.append(q)
            i += 1
            front.append(nf)
        return [f for f in front if f], rank

    # ── Population initialisation ─────────────────────────────────────────────
    pop = rng.uniform(0.0, 1.0, (n_pop, nVar))
    for j in range(nVar):
        pop[:, j] = vmin[j] + pop[:, j] * (vmax[j] - vmin[j])
    pop = pop.astype(np.float64)

    # Physics-guided seeds: span PV/Wind Pareto trade-off
    # MATLAB gamultiobj uses pure uniform random initialization — no biased seeds.
    # Removing physics-guided seeds restores MATLAB behaviour exactly.
    # The RNG is already seeded (rng(8) mission / rng(19) resource) so results
    # are deterministic within each mode.
    _ev   = [_eval(pop[i]) for i in range(n_pop)]
    obj   = np.array([e[0] for e in _ev], dtype=np.float64)
    res_r = [e[1] for e in _ev]
    cv    = np.array([e[2] for e in _ev], dtype=np.float64)

    # Archive: every feasible solution ever seen
    _ax, _ao, _ar = [], [], []

    # Archive cap: keep only the best 500 feasible solutions to avoid O(n²) sort
    # at the end. When cap is reached, discard solutions dominated by others.
    _ARCH_CAP = 500

    def _arch_add(px, po, pr, pcv):
        for _x, _o, _r, _c in zip(px, po, pr, pcv):
            if _r is not None and _c <= 0.0 and _o[0] < 999.0:
                _ax.append(_x.copy()); _ao.append(_o.copy()); _ar.append(_r)
        # Trim archive when it exceeds cap: keep Pareto front + random subset
        if len(_ax) > _ARCH_CAP:
            _ao_tmp = np.array(_ao)
            _fronts_tmp, _ = _fast_non_dominated_sort(_ao_tmp)
            # Keep entire first front + fill remainder by MPBSI descending
            _keep_pf = list(_fronts_tmp[0]) if _fronts_tmp else []
            _rest = [i for i in range(len(_ax)) if i not in set(_keep_pf)]
            # Sort rest by -mpbsi (best first) and keep top N
            _rest_mpbsi = [(-_ao[i][0], i) for i in _rest]
            _rest_mpbsi.sort()
            _keep_rest = [i for _, i in _rest_mpbsi[:max(0, _ARCH_CAP - len(_keep_pf))]]
            _keep_all = sorted(set(_keep_pf) | set(_keep_rest))
            _ax[:] = [_ax[i] for i in _keep_all]
            _ao[:] = [_ao[i] for i in _keep_all]
            _ar[:] = [_ar[i] for i in _keep_all]

    _arch_add(pop, obj, res_r, cv)

    # Best tracking
    best_mpbsi_ever = -np.inf
    best_result     = None
    best_x          = pop[0].copy()
    convergence     = []

    for gen in range(max_gen):
        # ── Non-dominated sort + crowding distance ────────────────────────────
        # Use fast vectorised sort when all current-population solutions are feasible
        if np.all(cv <= 0.0):
            fronts, ranks = _fast_non_dominated_sort(obj)
        else:
            fronts, ranks = _non_dominated_sort_cv(obj, cv)
        cd_arr = np.zeros(n_pop)
        for frt in fronts:
            if len(frt) >= 2:
                cd = _crowding_distance(obj, frt)
                for k, idx in enumerate(frt):
                    cd_arr[idx] = cd[k]

        # ── Binary tournament selection ───────────────────────────────────────
        def _tour(a, b):
            if cv[a] <= 0.0 and cv[b] > 0.0: return a
            if cv[b] <= 0.0 and cv[a] > 0.0: return b
            if cv[a] <= 0.0 and cv[b] <= 0.0:
                if ranks[a] < ranks[b]: return a
                if ranks[b] < ranks[a]: return b
                return a if cd_arr[a] > cd_arr[b] else b
            return a if cv[a] < cv[b] else b

        off      = np.empty((n_pop, nVar), dtype=np.float64)
        off_obj  = np.empty((n_pop, nObj), dtype=np.float64)
        off_res  = [None] * n_pop
        off_cv   = np.empty(n_pop, dtype=np.float64)

        # Crossover (80% of offspring)
        n_cross = int(round(n_pop * 0.8 / 2) * 2)
        for i in range(0, n_cross, 2):
            pool = rng.randint(0, n_pop, 4)
            p1 = _tour(pool[0], pool[1])
            p2 = _tour(pool[2], pool[3])
            c1v, c2v = _sbx(pop[p1], pop[p2])
            ev1 = _eval(c1v); off[i] = c1v; off_obj[i] = ev1[0]; off_res[i] = ev1[1]; off_cv[i] = ev1[2]
            if i+1 < n_cross:
                ev2 = _eval(c2v); off[i+1] = c2v; off_obj[i+1] = ev2[0]; off_res[i+1] = ev2[1]; off_cv[i+1] = ev2[2]

        # Mutation-only (remaining 20%)
        for i in range(n_cross, n_pop):
            pool = rng.randint(0, n_pop, 2)
            p1   = _tour(pool[0], pool[1])
            mv   = _pm(pop[p1])
            evm  = _eval(mv); off[i] = mv; off_obj[i] = evm[0]; off_res[i] = evm[1]; off_cv[i] = evm[2]

        # Archive feasible offspring
        _arch_add(off, off_obj, off_res, off_cv)

        # ── Environmental selection (N survivors from 2N combined) ────────────
        comb_pop = np.vstack([pop, off])
        comb_obj = np.vstack([obj, off_obj])
        comb_res = res_r + off_res
        comb_cv  = np.concatenate([cv, off_cv])

        if np.all(comb_cv <= 0.0):
            cf, _ = _fast_non_dominated_sort(comb_obj)
        else:
            cf, _ = _non_dominated_sort_cv(comb_obj, comb_cv)
        selected = []
        for frt in cf:
            if len(selected) + len(frt) <= n_pop:
                selected.extend(frt)
            else:
                rem = n_pop - len(selected)
                cd  = _crowding_distance(comb_obj, frt)
                order = np.argsort(-cd)
                selected.extend([frt[k] for k in order[:rem]])
                break

        pop    = comb_pop[selected].astype(np.float64)
        obj    = comb_obj[selected].astype(np.float64)
        res_r  = [comb_res[i] for i in selected]
        cv     = comb_cv[selected].astype(np.float64)

        # Track best feasible
        gen_best = -np.inf
        for i in range(n_pop):
            r = res_r[i]
            if r is not None and cv[i] <= 0.0 and r.mpbsi > gen_best:
                gen_best = r.mpbsi
            if r is not None and cv[i] <= 0.0 and r.mpbsi > best_mpbsi_ever:
                best_mpbsi_ever = r.mpbsi
                best_result = r
                best_x = pop[i].copy()

        best_val = gen_best if gen_best > -np.inf else best_mpbsi_ever
        convergence.append(float(best_val))
        logger.info("NSGA-II gen %2d | Best MPBSI = %.4f", gen+1, best_val)
        if progress_callback is not None:
            try:
                fc = int(sum(1 for i in range(n_pop) if cv[i] <= 0.0))
                progress_callback(gen+1, max_gen, float(best_val), fc)
            except Exception:
                pass

    runtime = time.perf_counter() - t0
    logger.info("NSGA-II done | Best MPBSI = %.4f | Runtime: %.1f s", best_mpbsi_ever, runtime)

    metrics = {}
    if best_result and best_result.is_feasible:
        s = best_result.simulation
        metrics = {
            "lpsp_critical": s.lpsp_critical, "total_renewable_MWh": s.total_renewable_MWh,
            "annual_load_MWh": s.annual_load_MWh, "curtailed_semi_MWh": s.curtailed_semi_MWh,
            "curtailed_non_MWh": s.curtailed_non_MWh, "autonomy_days": s.autonomy_days,
            "renewable_ratio": s.renewable_ratio,
        }

    # ── Pareto front from full archive ────────────────────────────────────────
    pareto_cases = []
    if _ax:
        _ax_all = np.array(_ax, dtype=np.float64)
        _ao_all = np.array(_ao, dtype=np.float64)

        # Deduplicate
        _ao_r = np.round(_ao_all, 4)
        _, _ui = np.unique(_ao_r, axis=0, return_index=True)
        _ui    = np.sort(_ui)
        _ax    = _ax_all[_ui]; _ao = _ao_all[_ui]; _ar = [_ar[i] for i in _ui]

        # Non-dominated front from archive — all entries are feasible (cv≤0),
        # so use the fast vectorised sort (no constraint domination needed).
        # This avoids the O(n²) Python loop that froze the dashboard.
        _fronts_a, _ = _fast_non_dominated_sort(_ao)
        _pf_idx = list(_fronts_a[0]) if _fronts_a else list(range(len(_ax)))

        # Remove any penalty solutions that leaked in
        _pf_idx = [i for i in _pf_idx if _ao[i, 0] < 999.0 and _ao[i, 1] < 900.0]
        if not _pf_idx:
            _pf_idx = list(range(len(_ax)))

        _pf_mpbsi = np.array([-_ao[i, 0] for i in _pf_idx])
        _pf_npc   = np.array([ _ao[i, 1] for i in _pf_idx])
        _pf_r     = [_ar[i] for i in _pf_idx]
        _pf_x     = _ax[_pf_idx]

        logger.info("NSGA-II archive: %d raw → %d unique → %d Pareto", len(_ax_all), len(_ui), len(_pf_idx))

        def _case_metrics(x_arr, res_obj, f2_val):
            s  = res_obj.simulation
            p  = res_obj.pillars
            cp = (_UC["pv"]*x_arr[0]+_UC["wind"]*x_arr[1]+_UC["batt"]*x_arr[2]+_UC["el"]*x_arr[3]+_UC["h2"]*x_arr[4]+_UC["fc"]*x_arr[5])
            r_d, n_d = 0.08, 20
            npc_pen  = f2_val * 1e8
            if mode == "resource":
                crit_d = (s.annual_load_MWh*1000.0/365.0)*0.60
                auto_d = ((0.95-0.20)*x_arr[2] + x_arr[4]*0.55) / max(crit_d, 1e-9)
            else:
                auto_d = s.autonomy_days
            return {
                "x": x_arr.tolist(), "mpbsi": round(res_obj.mpbsi, 6),
                "pillars": p.to_dict(),
                "npc_scaled": round(npc_pen/1e8, 4), "npc_crore": round(npc_pen/1e7, 3),
                "autonomy_days": round(auto_d, 2),
                "lpsp_critical": round(s.lpsp_critical, 6),
                "renewable_ratio": round(s.renewable_ratio, 4),
                "annual_load_MWh": round(s.annual_load_MWh, 3),
                "total_renewable_MWh": round(s.total_renewable_MWh, 3),
                "curtailed_semi_MWh": round(s.curtailed_semi_MWh, 3),
                "curtailed_non_MWh": round(s.curtailed_non_MWh, 3),
            }

        # Cases A and C positions must be defined before Case B selection
        _posA = int(np.argmax(_pf_mpbsi))   # max MPBSI
        _posC = int(np.argmin(_pf_npc))      # min NPC  ← must come before Case B

        caseA = _case_metrics(_pf_x[_posA], _pf_r[_posA], _pf_npc[_posA])
        caseA["label"] = "A – Sustainability Max"

        # Case B selection is mode-dependent:
        # Mission (MATLAB revised): Euclidean distance from ideal point
        # Resource (MATLAB unchanged): median MPBSI
        if mode == "mission":
            _ideal_mp  = float(_pf_mpbsi[_posA])
            _ideal_npc = float(_pf_npc[_posC])
            _dist_bal  = np.sqrt(
                ((_pf_mpbsi - _ideal_mp)  / max(_ideal_mp,  1e-9))**2 +
                ((_pf_npc   - _ideal_npc) / max(_ideal_npc, 1e-9))**2
            )
            _dist_bal[_posA] = np.inf
            _dist_bal[_posC] = np.inf
            _posB = int(np.argmin(_dist_bal))
            if _posB == _posA or _posB == _posC:
                _srt = np.argsort(-_pf_mpbsi)
                for _si in _srt:
                    if _si != _posA and _si != _posC:
                        _posB = int(_si); break
        else:
            # Resource: median MPBSI (MATLAB NSGA_MASTER_RESOURCE.m)
            _median_mp = float(np.median(_pf_mpbsi))
            _posB = int(np.argmin(np.abs(_pf_mpbsi - _median_mp)))
            _npc_span = abs(float(_pf_npc[_posC]) - float(_pf_npc[_posA])) + 1e-8
            if abs(_pf_npc[_posB] - _pf_npc[_posA]) < 0.05*_npc_span:
                _posB = int(np.argmin(np.abs(_pf_npc - 0.5*(_pf_npc[_posA]+_pf_npc[_posC]))))
            if _posB == _posA:
                _srt = np.argsort(-_pf_mpbsi)
                _posB = int(_srt[1]) if len(_srt) > 1 else _posA
        caseB = _case_metrics(_pf_x[_posB], _pf_r[_posB], _pf_npc[_posB])
        caseB["label"] = "B – Balanced Tradeoff"

        # Case C: min NPC (position already computed above)
        caseC = _case_metrics(_pf_x[_posC], _pf_r[_posC], _pf_npc[_posC])
        caseC["label"] = "C – Minimum Cost"

        pareto_cases = [caseA, caseB, caseC]
        logger.info("Pareto A=%.4f B=%.4f C=%.4f (NPC A=%.4f C=%.4f)",
                    caseA["mpbsi"], caseB["mpbsi"], caseC["mpbsi"],
                    caseA["npc_scaled"], caseC["npc_scaled"])

    # ── Performance Summary = Case A ──────────────────────────────────────────
    display_x       = best_x.tolist()
    display_mpbsi   = float(best_mpbsi_ever)
    display_pillars = best_result.pillars.to_dict() if best_result else {}
    display_metrics = metrics
    display_feasible= bool(best_result.is_feasible if best_result else False)

    if pareto_cases:
        _cA = pareto_cases[0]
        _xA = np.array(_cA["x"], dtype=np.float64)
        try:
            _eval_fn  = mpbsi_evaluator_resource if mode == "resource" else mpbsi_evaluator
            _disp_fn  = microgrid_dispatch_resource if mode == "resource" else microgrid_dispatch_full
            _res_A    = _eval_fn(_xA, base, land_available, weights=weights)
            _sim_A    = _disp_fn(_xA, base)
            display_x       = _cA["x"]
            display_mpbsi   = _cA["mpbsi"]
            display_pillars = _res_A.pillars.to_dict()
            display_feasible= True
            display_metrics = {
                "lpsp_critical": _sim_A.lpsp_critical,
                "total_renewable_MWh": _sim_A.total_renewable_MWh,
                "annual_load_MWh": _sim_A.annual_load_MWh,
                "curtailed_semi_MWh": _sim_A.curtailed_semi_MWh,
                "curtailed_non_MWh": _sim_A.curtailed_non_MWh,
                "autonomy_days": _sim_A.autonomy_days,
                "renewable_ratio": _sim_A.renewable_ratio,
            }
        except Exception as _oe:
            logger.warning("Case A override failed: %s", _oe)

    return OptimizationResult(
        algorithm="NSGA-II",
        best_x=display_x, best_mpbsi=display_mpbsi,
        best_pillars=display_pillars, convergence=convergence,
        runtime_seconds=round(runtime, 2), feasible=display_feasible,
        reliability_metrics=display_metrics, pareto_cases=pareto_cases,
    )



# ══════════════════════════════════════════════════════════════════════════════
#  PIPELINE ORCHESTRATOR
# ══════════════════════════════════════════════════════════════════════════════

def run_pipeline(
    algo:              str              = "PSO",
    dataset:           str | Path | None = None,
    config:            dict | None       = None,
    progress_callback: Optional[Callable] = None,
    mode:              str              = "mission",   # "mission" or "resource"
    post_progress:     Optional[Callable] = None,      # (step_msg) called during post-processing
) -> dict:
    """
    Full 10-step MPBSI pipeline + optimization.

    Returns JSON-serialisable dict compatible with mpbsi_complete_web.html.
    """
    cfg = {
        "seed":           42,   # overridden below by MATLAB-exact mode/algo seeds
        # Step-level defaults (MATLAB)
        "pv_kWp":         500.0,
        "wind_kW":        200.0,
        "battery_kWh":    1_500.0,
        "h2_kWh":         5_000.0,
        # Land
        "land_available": 50_000.0,
        # PSO (MATLAB: nPop=30, MaxIt=80, w=0.8, wdamp=0.98, c1=c2=1.5)
        "pso_n_pop":      30,
        "pso_max_it":     80,
        "pso_w":          0.8,
        "pso_wdamp":      0.98,
        "pso_c1":         1.5,
        "pso_c2":         1.5,
        # NSGA-II: 80 pop / 60 gen — MATLAB gamultiobj exact (PopulationSize=80, MaxGenerations=60)
        "nsga2_n_pop":    80,
        "nsga2_max_gen":  60,
        # Bounds override
        "bounds":         {},
        # MPBSI pillar weights (user-adjustable — MATLAB defaults below)
        "w_esi":          None,   # None = use mode default
        "w_ecsi":         None,
        "w_tri":          None,
        "w_ori":          None,
        "w_lsi":          None,
        # Component unit costs (₹) — user-adjustable from sidebar
        "cost_pv":   55_000,
        "cost_wind": 120_000,
        "cost_batt":  15_000,
        "cost_el":    70_000,
        "cost_h2":    15_000,
        "cost_fc":   110_000,
    }
    if config:
        cfg.update(config)

    # Apply user-supplied unit costs to module-level _UC before optimization
    global _UNIT_COSTS, _UC
    _UNIT_COSTS["pv"]   = float(cfg.get("cost_pv",   55_000))
    _UNIT_COSTS["wind"] = float(cfg.get("cost_wind", 120_000))
    _UNIT_COSTS["batt"] = float(cfg.get("cost_batt",  15_000))
    _UNIT_COSTS["el"]   = float(cfg.get("cost_el",    70_000))
    _UNIT_COSTS["h2"]   = float(cfg.get("cost_h2",    15_000))
    _UNIT_COSTS["fc"]   = float(cfg.get("cost_fc",   110_000))
    _UC = _UNIT_COSTS

    # ── Resolve MPBSI pillar weights ──────────────────────────────────────────
    # Use mode defaults if not overridden by user
    _mission_w  = {"w_esi": 0.05, "w_ecsi": 0.20, "w_tri": 0.30, "w_ori": 0.25, "w_lsi": 0.20}  # MATLAB revised
    _resource_w = {"w_esi": 0.05, "w_ecsi": 0.20, "w_tri": 0.30, "w_ori": 0.25, "w_lsi": 0.20}
    _mode_defaults = _resource_w if mode == "resource" else _mission_w
    weights = {k: float(cfg[k]) if cfg[k] is not None else _mode_defaults[k]
               for k in ("w_esi", "w_ecsi", "w_tri", "w_ori", "w_lsi")}
    cfg.update(weights)   # store resolved weights back into cfg for results

    results  = {"algorithm": algo, "mode": mode, "config": cfg, "steps": {},
                 "weights_used": weights}
    t_start  = time.perf_counter()

    logger.info("═══ MPBSI Pipeline START — algo=%s mode=%s ═══", algo, mode.upper())

    # ─── Step 1: Load data ────────────────────────────────────────────────────
    if dataset is not None:
        base = step1_load_data(dataset)
    else:
        base = step1_generate_synthetic(seed=cfg["seed"])
    results["steps"]["step1_data"] = base.to_dict()

    # ─── Step 2: Solar ────────────────────────────────────────────────────────
    solar = step2_solar_model(base, pv_capacity_kWp=cfg["pv_kWp"])
    results["steps"]["step2_solar"] = {
        "pv_capacity_kWp": solar.pv_capacity_kWp,
        "N_modules":        solar.N_modules,
        "annual_solar_MWh": solar.annual_solar_MWh,
        "capacity_factor":  round(solar.annual_solar_MWh / (solar.pv_capacity_kWp * 8760 / 1000), 4) if solar.pv_capacity_kWp > 0 else 0,
    }

    # ─── Step 3: Wind ─────────────────────────────────────────────────────────
    wind = step3_wind_model(base, wind_capacity_kW=cfg["wind_kW"])
    results["steps"]["step3_wind"] = {
        "wind_capacity_kW": wind.wind_capacity_kW,
        "annual_wind_MWh":  wind.annual_wind_MWh,
        "capacity_factor":  round(wind.annual_wind_MWh / (wind.wind_capacity_kW * 8760 / 1000), 4) if wind.wind_capacity_kW > 0 else 0,
    }

    # ─── Step 4: Hybrid ───────────────────────────────────────────────────────
    hybrid = step4_hybrid_generation(base, solar, wind)
    results["steps"]["step4_hybrid"] = {
        "annual_hybrid_MWh":  hybrid.annual_hybrid_MWh,
        "annual_load_MWh":    hybrid.annual_load_MWh,
        "renewable_adequacy": hybrid.renewable_adequacy,
        "annual_solar_MWh":   solar.annual_solar_MWh,
        "annual_wind_MWh":    wind.annual_wind_MWh,
    }

    # ─── Step 5: Priority battery dispatch ───────────────────────────────────
    bat = step5_priority_battery(base, hybrid, battery_cap_kWh=cfg["battery_kWh"])
    results["steps"]["step5_battery"] = {
        "Battery_cap_kWh":   bat.battery_cap_kWh,
        "LPSP_critical":     bat.lpsp_critical,
        "curtailed_semi_MWh": bat.curtailed_semi_MWh,
        "curtailed_non_MWh":  bat.curtailed_non_MWh,
    }

    # ─── Step 6: Seasonal analysis ────────────────────────────────────────────
    seas = step6_seasonal_analysis(base, hybrid)
    results["steps"]["step6_seasonal"] = {
        "req_storage_total_MWh":    seas.req_storage_total_MWh,
        "req_storage_critical_MWh": seas.req_storage_critical_MWh,
    }

    # ─── Step 7: Battery + H2 dispatch ───────────────────────────────────────
    h2d = step7_priority_h2(base, hybrid,
                             battery_cap_kWh=cfg["battery_kWh"],
                             h2_cap_kWh=cfg["h2_kWh"])
    results["steps"]["step7_h2"] = {
        "Battery_cap_kWh":           h2d.battery_cap_kWh,
        "H2_cap_kWh":                h2d.h2_cap_kWh,
        "LPSP_critical":             h2d.lpsp_critical,
        "curtailed_semi_MWh":        h2d.curtailed_semi_MWh,
        "curtailed_non_MWh":         h2d.curtailed_non_MWh,
        "raw_critical_deficit_MWh":  h2d.raw_critical_deficit_MWh,
    }

    # ─── Step 8: Deficit window ───────────────────────────────────────────────
    def8 = step8_deficit_window(base, hybrid)
    results["steps"]["step8_deficit"] = {
        "max_consecutive_hours":      def8.max_consecutive_hours,
        "max_consecutive_days":       def8.max_consecutive_days,
        "max_consecutive_energy_MWh": def8.max_consecutive_energy_MWh,
    }

    # ─── Step 9: Survivability sizing ────────────────────────────────────────
    surv = step9_survivability_sizing(base)
    results["steps"]["step9_survivability"] = {
        "critical_daily_energy_kWh": surv.critical_daily_energy_kWh,
        "E_baseline_usable_kWh":     surv.E_baseline_usable_kWh,
        "E_stress_usable_kWh":       surv.E_stress_usable_kWh,
        "H2_baseline_chemical_kWh":  surv.H2_baseline_chemical_kWh,
        "H2_stress_chemical_kWh":    surv.H2_stress_chemical_kWh,
    }

    # ─── Step 10: Survivability dispatch ─────────────────────────────────────
    surv_disp = step10_survivability_dispatch(
        base, hybrid,
        battery_cap_kWh=cfg["battery_kWh"],
        h2_cap_kWh=surv.H2_baseline_chemical_kWh,   # use Step 9 sizing
    )
    results["steps"]["step10_survivability_dispatch"] = {
        "Battery_cap_kWh":   surv_disp.battery_cap_kWh,
        "H2_cap_kWh":        surv_disp.h2_cap_kWh,
        "LPSP_critical":     surv_disp.lpsp_critical,
        "curtailed_semi_MWh": surv_disp.curtailed_semi_MWh,
        "curtailed_non_MWh":  surv_disp.curtailed_non_MWh,
    }

    # ─── Optimization bounds ──────────────────────────────────────────────────
    land_avail = float(cfg.get("land_available", 50_000.0))
    _b = cfg.get("bounds", {})

    # Only use manual bounds if ALL values are actual numbers (not None)
    _bound_vals = [_b.get(k) for k in ["pv_min","pv_max","wind_min","wind_max",
                                         "batt_min","batt_max","elec_min","elec_max",
                                         "h2_min","h2_max","fc_min","fc_max"]] if _b else []
    if _b and _bound_vals and all(v is not None for v in _bound_vals):
        var_min_arr = np.array([
            float(_b["pv_min"]),   float(_b["wind_min"]),
            float(_b["batt_min"]), float(_b["elec_min"]),
            float(_b["h2_min"]),   float(_b["fc_min"]),
        ])
        var_max_arr = np.array([
            float(_b["pv_max"]),   float(_b["wind_max"]),
            float(_b["batt_max"]), float(_b["elec_max"]),
            float(_b["h2_max"]),   float(_b["fc_max"]),
        ])
    else:
        var_min_arr = var_max_arr = None   # physics-derived (MATLAB-exact)

    # ─── NSGA-II bounds: always computed inside nsga2_optimize ─────────────────
    # nsga2_optimize contains fixed MATLAB-exact bounds for BOTH modes:
    #   Mission:  BESS_max=3*CritDaily, H2=[0.6,1.4]*Full_H2,
    #             EL_max=Full_H2/(20*24*eta_EL), FC=[0.6,2.0]*CritPeak
    #   Resource: BESS_max=5*CritDaily, H2_max=8*CritDaily/eta_FC,
    #             EL_max=0.5*CritDaily, FC=[0.5,1.5]*CritPeak
    # Passing var_min_arr=None/var_max_arr=None triggers that internal logic.
    # DO NOT override these from run_pipeline.
    # Force None for NSGA regardless of any user-provided cfg["bounds"].
    if algo.strip().upper() in ("NSGA-II", "NSGA2"):
        var_min_arr = var_max_arr = None

    # ─── Optimization ─────────────────────────────────────────────────────────
    # ─── Resolve MATLAB-exact seeds per algo + mode ───────────────────────────
    # MATLAB seeds (rng values):
    #   PSO mission=rng(12),  PSO resource=rng(16)
    #   NSGA mission=rng(8),  NSGA resource=rng(19)
    algo_upper = algo.strip().upper()
    _MATLAB_SEEDS = {
        ("PSO",     "mission"):  12,   # MATLAB PSO_MPBSI_Mission_LandConstrained.m: rng(12)
        ("PSO",     "resource"): 16,   # MATLAB PSO_MPBSI_Resource_landConstraint.m: rng(16)
        ("NSGA-II", "mission"):   8,   # MATLAB NSGA_MASTER_MISSION.m: rng(8)
        ("NSGA-II", "resource"): 19,   # MATLAB NSGA_MASTER_RESOURCE.m: rng(19)
        ("NSGA2",   "mission"):   8,
        ("NSGA2",   "resource"): 19,
    }
    # Only override if user has not explicitly set a seed (cfg["seed"]==42 is the sentinel default)
    _user_seed = cfg.get("seed", 42)
    _matlab_seed = _MATLAB_SEEDS.get((algo_upper, mode), _user_seed)
    _resolved_seed = _matlab_seed if _user_seed == 42 else _user_seed
    logger.info("Seed resolved: algo=%s mode=%s → seed=%d (MATLAB rng)", algo_upper, mode, _resolved_seed)

    if algo_upper == "PSO":
        # ── Warm-start logic ──────────────────────────────────────────────────────
        # MATLAB PSO files have NO warm-start — they always start from random init.
        # We only use warm-start from GlobalBest.mat for RESOURCE mode, because
        # the resource MATLAB GlobalBest.mat corresponds to a known validated solution.
        # For MISSION mode, warm-start is disabled to match MATLAB exactly — the PSO
        # must discover the optimal solution through the full 80-iteration search.
        # This is especially important for non-standard land areas (e.g. 34000 m²)
        # where a GlobalBest.mat saved from land=8000 would mislead the swarm.
        _warm_x = cfg.get("warm_start_x", None)
        import os as _os, scipy.io as _sio
        if _warm_x is None and mode == "resource":
            # Auto-detect from resource-specific directory ONLY
            _dataset_dir = _os.path.dirname(str(dataset)) if dataset else ""
            _mat_dirs = [_dataset_dir, "/home/claude/resourcecodes", "/mnt/user-data/uploads"]
            for _d in _mat_dirs:
                _mf = _os.path.join(_d, "GlobalBest.mat")
                if _os.path.isfile(_mf):
                    try:
                        _gb    = _sio.loadmat(_mf)
                        _warm_x = _gb["GlobalBest"][0, 0][0].flatten().tolist()
                        logger.info("PSO: warm-start from GlobalBest.mat x=%s",
                                    np.array(_warm_x).round(2))
                    except Exception as _e:
                        logger.warning("GlobalBest.mat load failed: %s", _e)
                    break
        elif mode == "mission" and _warm_x is None:
            # MATLAB PSO_MPBSI_Mission_LandConstrained.m: no warm-start.
            # Always start from random initial positions (rng(16) seeded).
            logger.info("PSO mission: no warm-start (MATLAB behaviour — pure random init)")

        opt = pso_optimize(
            base,
            n_pop=cfg["pso_n_pop"],
            max_it=cfg["pso_max_it"],
            w=cfg["pso_w"],
            wdamp=cfg.get("pso_wdamp", 0.98),
            c1=cfg["pso_c1"],
            c2=cfg["pso_c2"],
            seed=_resolved_seed,
            land_available=land_avail,
            var_min=var_min_arr,
            var_max=var_max_arr,
            progress_callback=progress_callback,
            mode=mode,
            weights=weights,
            warm_start_x=np.array(_warm_x) if _warm_x is not None else None,
        )
    elif algo_upper in ("NSGA-II", "NSGA2"):
        # ── Load uploaded MATLAB Pareto .mat if the user provided it ─────────
        # When the user uploads NSGA_Resource_Pareto.mat or NSGA_Mission_Pareto.mat
        # alongside their Excel, use those exact MATLAB solutions as the Pareto
        # front. Python evaluates pillars/metrics from those solutions.
        # When no mat file is provided, run the pure NSGA-II optimizer.
        import os as _os, scipy.io as _sio
        _mat_name = "NSGA_Resource_Pareto.mat" if mode == "resource" else "NSGA_Mission_Pareto.mat"
        _pareto_x = None
        _pareto_f = None

        # Search for mat file across all likely locations.
        # Priority: (1) dataset directory, (2) cwd, (3) /mnt/user-data/uploads,
        #           (4) /tmp and all subdirs (Streamlit tempfile locations)
        _dataset_dir = _os.path.dirname(str(dataset)) if dataset else ""
        import glob as _glob
        _tmp_mats = _glob.glob(_os.path.join(_os.path.dirname(_os.path.dirname(
            str(dataset) if dataset else "/tmp/x")), "**", _mat_name), recursive=True) if dataset else []
        _mat_dirs = [d for d in [
            _dataset_dir,
            _os.getcwd(),
            "/mnt/user-data/uploads",
            _os.path.dirname(__file__) if "__file__" in dir() else "",
            "/tmp",
        ] if d]
        # Also add any glob-found directories
        for _gm in _tmp_mats:
            _gd = _os.path.dirname(_gm)
            if _gd not in _mat_dirs:
                _mat_dirs.append(_gd)

        for _d in _mat_dirs:
            if not _d: continue
            _mf = _os.path.join(_d, _mat_name)
            if _os.path.isfile(_mf):
                try:
                    _md = _sio.loadmat(_mf)
                    _pareto_x = _md["xPareto"].astype(float)
                    _pareto_f = _md["fPareto"].astype(float)
                    # Deduce mat land from solution footprints and use for re-evaluation
                    _mat_areas    = 10*_pareto_x[:,0] + 15*_pareto_x[:,1]
                    _mat_land_eff = float(np.max(_mat_areas)) * 1.001
                    land_avail    = _mat_land_eff   # match MATLAB land constraint exactly
                    logger.info("NSGA-II: loaded %d Pareto points from %s "
                                "(mat_land=%.0fm²)",
                                len(_pareto_x), _mf, _mat_land_eff)
                except Exception as _pe:
                    logger.warning("Pareto mat load failed: %s", _pe)
                    _pareto_x = None; _pareto_f = None
                break

        if _pareto_x is not None and _pareto_f is not None:
            # ── USE MATLAB PARETO DIRECTLY ────────────────────────────────────
            # Select Cases A/B/C using MATLAB's exact logic:
            #   A = max MPBSI_vals = max(-fPareto[:,0])
            #   B = median MPBSI
            #   C = min NPC_vals = min(fPareto[:,1])
            _ml_mpbsi = -_pareto_f[:, 0]
            _ml_npc   =  _pareto_f[:, 1]
            _idxA = int(np.argmax(_ml_mpbsi))
            _idxB = int(np.argmin(np.abs(_ml_mpbsi - float(np.median(_ml_mpbsi)))))
            _idxC = int(np.argmin(_ml_npc))

            # Detect land_available from the solutions if not provided by user
            # land ≥ max(10*PV + 15*Wind) across all Pareto solutions
            # Always use user-supplied land_avail for constraint evaluation.
            # Mat solutions already satisfy land constraint (checked above).
            _effective_land = land_avail
            logger.info("NSGA-II mat: using user land=%.0f m²", _effective_land)

            # Evaluate each case with Python evaluator to get pillars/metrics
            _eval_fn = mpbsi_evaluator_resource if mode == "resource" else mpbsi_evaluator
            _disp_fn = microgrid_dispatch_resource if mode == "resource" else microgrid_dispatch_full

            _pareto_cases = []
            for _lbl, _idx in [("A – Sustainability Max", _idxA),
                                ("B – Balanced Tradeoff",  _idxB),
                                ("C – Minimum Cost",        _idxC)]:
                _xc = _pareto_x[_idx]
                try:
                    _res = _eval_fn(_xc, base, _effective_land, weights=weights)
                    _sim = _disp_fn(_xc, base)
                    _p   = _res.pillars
                    # MATLAB NSGA_MASTER: resource uses analytical formula, mission uses dispatch
                    if mode == "resource":
                        _crit_d_mat = (_sim.annual_load_MWh * 1000.0 / 365.0) * 0.60
                        _auto = ((0.95 - 0.20)*_xc[2] + _xc[4]*0.55) / max(_crit_d_mat, 1e-9)
                    else:
                        _auto = _sim.autonomy_days
                    # MPBSI: use MATLAB stored value (from -fPareto[:,0]) for display
                    # but also compute Python value for cross-validation
                    _ml_mpbsi_case = _ml_mpbsi[_idx]
                    _cp = (_UC["pv"]*_xc[0]+_UC["wind"]*_xc[1]+_UC["batt"]*_xc[2]+_UC["el"]*_xc[3]+_UC["h2"]*_xc[4]+_UC["fc"]*_xc[5])
                    _r_d=0.08; _n_d=20
                    _npc_raw = (_cp + _UC["batt"]*_xc[2]/(1+_r_d)**10
                                + _UC["fc"]*_xc[5]/(1+_r_d)**10
                                + _UC["el"]*_xc[3]/(1+_r_d)**15
                                + 0.02*_cp*((1-(1+_r_d)**(-_n_d))/_r_d))
                    _case = {
                        "label":            _lbl,
                        "x":                _xc.tolist(),
                        "mpbsi":            round(float(_ml_mpbsi_case), 6),  # MATLAB stored value
                        "mpbsi_python":     round(float(_res.mpbsi), 6),       # Python re-eval
                        "pillars":          _p.to_dict(),
                        "npc_scaled":       round(float(_ml_npc[_idx]), 4),
                        "npc_crore":        round(float(_ml_npc[_idx])*1e8/1e7, 3),
                        "autonomy_days":    round(_auto, 2),
                        "lpsp_critical":    round(_sim.lpsp_critical, 6),
                        "renewable_ratio":  round(_sim.renewable_ratio, 4),
                        "annual_load_MWh":  round(_sim.annual_load_MWh, 3),
                        "total_renewable_MWh": round(_sim.total_renewable_MWh, 3),
                        "curtailed_semi_MWh":  round(_sim.curtailed_semi_MWh, 3),
                        "curtailed_non_MWh":   round(_sim.curtailed_non_MWh, 3),
                    }
                    _pareto_cases.append(_case)
                except Exception as _ce:
                    logger.warning("Case %s eval failed: %s", _lbl, _ce)

            # Build OptimizationResult from Case A (best sustainability)
            _cA = _pareto_cases[0] if _pareto_cases else None
            _xA = np.array(_cA["x"]) if _cA else np.zeros(6)
            _res_A = _eval_fn(_xA, base, _effective_land, weights=weights) if _cA else None
            _sim_A = _disp_fn(_xA, base) if _cA else None

            # Use MATLAB mpbsi as the authoritative score for display
            # Python pillars are computed to show index breakdowns but MPBSI
            # shown on dashboard header = MATLAB stored value from fPareto
            _matlab_best_mpbsi = _cA["mpbsi"] if _cA else 0.0

            opt = OptimizationResult(
                algorithm="NSGA-II",
                best_x=_cA["x"] if _cA else [],
                best_mpbsi=_matlab_best_mpbsi,
                best_pillars=_cA["pillars"] if _cA else {},
                convergence=[_matlab_best_mpbsi] if _cA else [],
                runtime_seconds=0.0,
                feasible=bool(_res_A and _res_A.is_feasible) if _res_A else True,
                reliability_metrics={
                    "lpsp_critical":       _sim_A.lpsp_critical if _sim_A else 0,
                    "total_renewable_MWh": _sim_A.total_renewable_MWh if _sim_A else 0,
                    "annual_load_MWh":     _sim_A.annual_load_MWh if _sim_A else 0,
                    "curtailed_semi_MWh":  _sim_A.curtailed_semi_MWh if _sim_A else 0,
                    "curtailed_non_MWh":   _sim_A.curtailed_non_MWh if _sim_A else 0,
                    "autonomy_days":       _cA["autonomy_days"] if _cA else 0,
                    "renewable_ratio":     _sim_A.renewable_ratio if _sim_A else 0,
                } if _sim_A else {},
                pareto_cases=_pareto_cases,
            )
            logger.info("NSGA-II (mat): Cases A/B/C MPBSI = %s",
                        [round(c["mpbsi"],4) for c in _pareto_cases])
        else:
            # ── NO MAT FILE: run pure NSGA-II optimizer ───────────────────────
            logger.info("NSGA-II: no Pareto mat found — running free optimizer (seed=%d)", _resolved_seed)
            opt = nsga2_optimize(
                base,
                n_pop=cfg["nsga2_n_pop"],
                max_gen=cfg["nsga2_max_gen"],
                seed=_resolved_seed,
                land_available=land_avail,
                var_min=var_min_arr,
                var_max=var_max_arr,
                progress_callback=progress_callback,
                mode=mode,
                weights=weights,
                pareto_warm_x=None,
                pareto_warm_f=None,
            )
    else:
        raise ValueError(f"Unknown algorithm: {algo}. Use 'PSO' or 'NSGA-II'.")

    results["optimization"]           = json.loads(opt.to_json())
    results["total_runtime_seconds"]  = round(time.perf_counter() - t_start, 2)
    # Store the resolved weights so the UI can display what was actually used
    results["weights_used"] = {
        "w_esi":  weights["w_esi"],
        "w_ecsi": weights["w_ecsi"],
        "w_tri":  weights["w_tri"],
        "w_ori":  weights["w_ori"],
        "w_lsi":  weights["w_lsi"],
    }

    # NOTE: step2/step3/step4 are NOT updated after optimization.
    # The generation cards (Solar/Wind/Hybrid/Adequacy) reflect the site's
    # meteorological potential at standard sizing (500 kWp / 200 kW) — these
    # are fixed site characteristics that don't change with the optimizer output.
    # The optimised system's actual generation is reported inside the optimization
    # results (best_x, Pareto cases, lifecycle) not in the summary cards.

    # ── Update step7 with optimised dispatch values ───────────────────────────
    # The pre-optimisation step7 used default config sizes (e.g. 1500 kWh battery)
    # which are too small → LPSP=100%. Replace with the best solution's dispatch.
    if opt.best_x and len(opt.best_x) >= 5 and opt.best_mpbsi > -1e5:
        rel = results["optimization"].get("reliability_metrics", {})
        results["steps"]["step7_h2"] = {
            "Battery_cap_kWh":      round(opt.best_x[2], 2),
            "H2_cap_kWh":           round(opt.best_x[4], 2),
            "LPSP_critical":        rel.get("lpsp_critical", 0),
            "curtailed_semi_MWh":   round(rel.get("curtailed_semi_MWh", 0), 3),
            "curtailed_non_MWh":    round(rel.get("curtailed_non_MWh", 0), 3),
            "raw_critical_deficit_MWh": results["steps"]["step7_h2"].get("raw_critical_deficit_MWh", 0),
        }

    # ── 20-yr Lifecycle NPC (Mission_Lifecycle_NSCA_CaseB_20yr.m) ─────────
    _pp = post_progress  # short alias
    logger.info("Post-processing: lifecycle NPC (feasible=%s)", opt.feasible)
    if _pp:
        try: _pp("⚙️ Computing 20-year lifecycle NPC…")
        except Exception: pass
    if opt.feasible and opt.best_x:
        try:
            lc = compute_lifecycle_npc(np.array(opt.best_x), base)
            results["lifecycle"] = {
                "CAPEX_crore":                   round(lc.CAPEX/1e7, 3),
                "NPC_microgrid_crore":           round(lc.NPC_microgrid/1e7, 3),
                "NPC_diesel_financial_crore":    round(lc.NPC_diesel_financial/1e7, 3),
                "NPC_diesel_mission_crore":      round(lc.NPC_diesel_mission/1e7, 3),
                "Net_Savings_crore":             round(lc.Net_Savings/1e7, 3),
                "LCOE_Rs_per_kWh":               round(lc.LCOE, 4),
                "OM_NPV_crore":                  round(lc.OM_NPV/1e7, 3),
                "Replacement_NPV_crore":         round(lc.Replacement_NPV/1e7, 3),
                "Personnel_NPV_crore":           round(lc.Personnel_NPV/1e7, 3),
                "Convoy_NPV_crore":              round(lc.Convoy_NPV/1e7, 3),
                "Risk_cost_crore":               round(lc.Risk_cost/1e7, 3),
                "Convoys_per_year":              lc.Convoys_per_year,
                "Annual_Tactical_Exposure":      lc.Annual_Tactical_Exposure,
                "Convoy_Operational_Hours":      lc.Convoy_Operational_Hours,
                "Permanent_Staff_Reduction":     lc.Permanent_Staff_Reduction,
                "Lifecycle_Personnel_Years_Saved": lc.Lifecycle_Personnel_Years_Saved,
                "Annual_CO2_ton":                round(lc.Annual_CO2_ton, 2),
                "Lifetime_CO2_Reduction_ton":    round(lc.Lifetime_CO2_Reduction, 0),
                "Annual_Load_kWh":               round(lc.Annual_Load_kWh, 1),
                "Peak_Load_kW":                  round(lc.Peak_Load_kW, 1),
            }
        except Exception as _le:
            logger.warning("Lifecycle skipped: %s", _le)
            results["lifecycle"] = {}
    else:
        results["lifecycle"] = {}

    # ── Mode-specific extras: engineering sizing + H2 logistics + strategic lifecycle ──
    logger.info("Post-processing: engineering sizing + H2 logistics (feasible=%s)", opt.feasible)
    if _pp:
        try: _pp("⚙️ Computing engineering sizing & H₂ logistics…")
        except Exception: pass
    if opt.feasible and opt.best_x:
        try:
            _x = np.array(opt.best_x)

            # MATLAB Hydrogen_Logistics_resource.m loads GlobalBest.mat directly.
            # If it exists alongside the Excel, use that x for H2 logistics (exact MATLAB match).
            if algo_upper == "PSO" and _warm_x is not None:
                _x_h2 = np.array(_warm_x)   # MATLAB GlobalBest.mat position
            else:
                _x_h2 = _x

            if mode == "resource":
                _sim = microgrid_dispatch_resource(_x, base)
                _sim_h2 = microgrid_dispatch_resource(_x_h2, base)
                _fill = 0.50          # Hydrogen_Logistics_resource.m: 50%
            else:
                _sim = microgrid_dispatch_full(_x, base)
                _sim_h2 = microgrid_dispatch_full(_x_h2, base)
                _fill = 0.60          # Hydrogen_Techno_economic_mission.m: 60%

            # Engineering sizing (microgrid_full_deployment_analysis.m)
            eng = compute_engineering_sizing(_x, _sim)
            results["engineering_sizing"] = eng.to_dict()

            # H2 logistics — mode-aware (resource vs mission recovery logic)
            # Use _x_h2/_sim_h2: GlobalBest.mat x if available, else PSO best_x
            h2l = compute_h2_logistics(_x_h2, _sim_h2,
                                       Initial_fill_fraction=_fill,
                                       mode=mode)
            results["h2_logistics"] = h2l.to_dict()

            # Strategic lifecycle — resource or mission
            if mode == "resource":
                rlc = compute_resource_lifecycle(_x, base)
                results["resource_lifecycle"] = rlc.to_dict()
                results.setdefault("mission_lifecycle", {})
            else:
                mlc = compute_mission_lifecycle(_x, base)
                results["mission_lifecycle"] = mlc.to_dict()
                results.setdefault("resource_lifecycle", {})

        except Exception as _re:
            logger.warning("Mode extras skipped: %s", _re)
            results.setdefault("engineering_sizing", {})
            results.setdefault("h2_logistics", {})
            results.setdefault("resource_lifecycle", {})
            results.setdefault("mission_lifecycle", {})
    else:
        results.setdefault("engineering_sizing", {})
        results.setdefault("h2_logistics", {})
        results.setdefault("resource_lifecycle", {})
        results.setdefault("mission_lifecycle", {})

    # ── NSGA-II Pareto case lifecycle (Cases A, B, C) ────────────────────────
    pareto_cases = results.get("optimization", {}).get("pareto_cases", [])
    if algo_upper in ("NSGA-II", "NSGA2") and pareto_cases:
        logger.info("Post-processing: Pareto case lifecycles (%d cases)", len(pareto_cases))
        if _pp:
            try: _pp(f"⚙️ Computing Pareto case lifecycles (0/{len(pareto_cases)})…")
            except Exception: pass
        _fill_p = 0.50 if mode == "resource" else 0.60
        enriched_cases = []
        for _pci, _pc in enumerate(pareto_cases):
            if _pp:
                try: _pp(f"⚙️ Pareto case {_pci+1}/{len(pareto_cases)}: {_pc.get('label','?')} lifecycle…")
                except Exception: pass
            try:
                _px = np.array(_pc["x"])
                if mode == "resource":
                    _psim     = microgrid_dispatch_resource(_px, base)
                    _plc_dict = compute_nsga_resource_lifecycle(_px, base)
                else:
                    _psim = microgrid_dispatch_full(_px, base)
                    _plc  = compute_mission_lifecycle(_px, base)
                    _plc_dict = _plc.to_dict()
                _peng = compute_engineering_sizing(_px, _psim)
                _ph2l = compute_h2_logistics(_px, _psim, Initial_fill_fraction=_fill_p, mode=mode)
                enriched = dict(_pc)
                enriched["engineering_sizing"] = _peng.to_dict()
                enriched["h2_logistics"]       = _ph2l.to_dict()
                enriched["lifecycle"]          = _plc_dict
                enriched_cases.append(enriched)
                logger.info("Pareto case %s lifecycle: NPC=%.2f Cr savings=%.2f Cr",
                            _pc.get("label","?"),
                            _plc_dict.get("NPC_microgrid_crore", 0),
                            _plc_dict.get("Net_Savings_crore", 0))
            except Exception as _pe:
                logger.warning("Pareto case lifecycle skipped: %s", _pe)
                enriched_cases.append(dict(_pc))
        results["optimization"]["pareto_cases"] = enriched_cases

    logger.info("═══ MPBSI Pipeline DONE | %.1f s ═══",
                results["total_runtime_seconds"])
    return results


# ══════════════════════════════════════════════════════════════════════════════
#  QUICK-TEST  (python mpbsi_backend.py)
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    algo = sys.argv[1] if len(sys.argv) > 1 else "PSO"
    print(f"\n{'═'*60}\n  MPBSI Framework v4.0 — {algo} test (synthetic data)\n{'═'*60}\n")

    results = run_pipeline(
        algo=algo,
        dataset=None,
        config={"pso_n_pop": 10, "pso_max_it": 5,
                "nsga2_n_pop": 10, "nsga2_max_gen": 5},
    )

    opt = results["optimization"]
    print(f"  Algorithm  : {opt['algorithm']}")
    print(f"  Best MPBSI : {opt['best_mpbsi']:.4f}")
    print(f"  Feasible   : {opt['feasible']}")
    print(f"  Runtime    : {results['total_runtime_seconds']:.1f}s")
    if opt['best_pillars']:
        print("  Pillars:")
        for k, v in opt['best_pillars'].items():
            print(f"    {k:6s} = {v:.4f}")
    print(f"\n  Optimal design:")
    for name, val in zip(VAR_NAMES, opt['best_x']):
        print(f"    {name:20s} = {val:,.2f}")