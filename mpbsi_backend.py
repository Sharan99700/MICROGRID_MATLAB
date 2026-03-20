"""
MPBSI Framework Backend v4.0
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

    logger.info("Step 2 — %d modules × %.0fW | cap=%.2f kWp | annual=%.2f MWh | CF=%.3f",
                NSolarPV, PSolarPV_ref, pv_capacity_kWp,
                annual, annual / (pv_capacity_kWp * 8.76))
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
    Hub-height correction (Hhub=40m, alpha=0.14) is ONLY in microgrid_dispatch_mission.m.

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
    logger.info("Step 3 — %.0f kW turbine | annual=%.2f MWh | CF=%.3f",
                wind_capacity_kW, annual, annual / (wind_capacity_kW * 8.76))
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
    2. Wind: hub-height correction (Hhub=40m, alpha=0.14) + η_wind=0.90
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

    # ── Wind — hub-height correction + η_wind (MATLAB mission only) ──────────
    V2      = Wind * (40.0 / 10.0) ** 0.14          # Hhub=40, Href=10, alpha=0.14
    v_ci, v_r, v_co = 3.0, 12.0, 25.0
    P_WT    = np.where(
        (V2 < v_ci) | (V2 >= v_co), 0.0,
        np.where(V2 < v_r,
                 WT_rated * ((V2 - v_ci) / (v_r - v_ci)) ** 3,
                 WT_rated),
    )
    P_WT    = 0.90 * P_WT                            # η_wind = 0.90

    P_RES = P_PV + P_WT

    # ── Load split ────────────────────────────────────────────────────────────
    Lc = 0.60 * Load
    Ls = 0.25 * Load
    Ln = 0.15 * Load

    # ── Initial conditions (MATLAB: SOC_max = E_BESS, absolute kWh) ──────────
    SOC     = 0.8 * E_BESS
    SOC_min = 0.2 * E_BESS
    SOC_max = E_BESS             # 100% — MATLAB: SOC_max = E_BESS
    E_H2    = 0.5 * E_H2_max    # MATLAB: E_H2 = 0.5 * E_H2_max

    unmet_c    = 0.0
    curt_s     = 0.0
    curt_n     = 0.0
    Annual_RES = 0.0

    soc_arr = np.empty(N) if store_series else None
    h2_arr  = np.empty(N) if store_series else None

    for t in range(N):
        P_available  = P_RES[t]
        Annual_RES  += P_available

        # ── Critical ──────────────────────────────────────────────────────
        deficit = Lc[t] - P_available
        if deficit <= 0:
            P_available = -deficit
        else:
            # Battery discharge
            max_bat_out = (SOC - SOC_min) * eta_dis
            P_dis_bat   = min(deficit, max_bat_out)
            SOC        -= P_dis_bat / eta_dis
            deficit    -= P_dis_bat

            # Fuel cell (hydrogen)
            max_fc_energy = E_H2 * eta_FC
            max_fc        = min(P_FC_max, max_fc_energy)
            P_fc          = min(deficit, max_fc)
            E_H2         -= P_fc / eta_FC
            deficit      -= P_fc

            if deficit > 0:
                unmet_c += deficit
            P_available = 0.0

        # ── Semi ──────────────────────────────────────────────────────────
        if P_available >= Ls[t]:
            P_available -= Ls[t]
        else:
            curt_s     += (Ls[t] - P_available)
            P_available  = 0.0

        # ── Non ───────────────────────────────────────────────────────────
        if P_available >= Ln[t]:
            P_available -= Ln[t]
        else:
            curt_n     += (Ln[t] - P_available)
            P_available  = 0.0

        # ── Store surplus ─────────────────────────────────────────────────
        if P_available > 0:
            max_ch_room = (SOC_max - SOC) / eta_ch
            P_ch        = min(P_available, max_ch_room)
            SOC        += eta_ch * P_ch
            P_available -= P_ch

            max_h2_room = (E_H2_max - E_H2) / eta_EL
            P_el        = min(P_available, P_EL_max, max_h2_room)
            E_H2       += eta_EL * P_el

        SOC  = min(max(SOC,  SOC_min), SOC_max)
        E_H2 = min(max(E_H2, 0.0),    E_H2_max)

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

    # MATLAB feasibility: LPSP≤1e-4 AND REN≥0.999 AND Autonomy≥7
    is_feasible = (lpsp <= lpsp_tol) and (ren_ratio >= 0.999) and (autonomy_days >= 7.0)

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
    EcSI = 0.5×C1 + 0.5×C2        C1=1/(1+LCOE/50) [Rs 50/kWh ref], C2=CAPEX-based
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

    Aut_factor   = 1.0          # autonomy treated as compliance only
    Annual_Load  = sim.annual_load_MWh
    Annual_RES   = sim.total_renewable_MWh

    # ── 4. ESI (Environmental) ────────────────────────────────────────────────
    E1     = 1.0
    # E2: MATLAB exact — penalises both deficit AND surplus.
    # E2 = 1 only when RES = Load exactly; penalises over-generation,
    # which keeps PSO from oversizing PV beyond what demand needs.
    E2_raw = Annual_RES / Annual_Load if Annual_Load > 0 else 0.0
    E2     = 1.0 / (1.0 + abs(E2_raw - 1.0))
    ESI    = 0.5 * E1 + 0.5 * E2

    # ── 5. EcSI (Economic) ────────────────────────────────────────────────────
    Cost_PV  = 55_000  * x[0]
    Cost_WT  = 120_000 * x[1]
    Cost_Bat = 15_000  * x[2]   # MATLAB: 15000 ₹/kWh
    Cost_EL  = 70_000  * x[3]
    Cost_H2  = 15_000  * x[4]   # MATLAB: 15000 ₹/kWh
    Cost_FC  = 110_000 * x[5]
    CAPEX    = Cost_PV + Cost_WT + Cost_Bat + Cost_EL + Cost_H2 + Cost_FC

    LCOE = CAPEX / (Annual_RES * 1000.0 * 20.0) if Annual_RES > 0 else 1e6
    # C1: LCOE cost-competitiveness index.
    # Original MATLAB uses 10×LCOE (calibrated for $/kWh, ~0.05–0.15).
    # Akhnoor costs are in Rs/kWh (~20–60 Rs/kWh) so the multiplier is
    # normalised to Rs 50/kWh reference (÷50 = equivalent of ×0.02).
    # This restores C1 as an active cost-signal — without it C1≈0.002
    # making EcSI a dead pillar and the PSO blind to LCOE differences.
    C1   = 1.0 / (1.0 + LCOE / 50.0)   # normalised to Rs 50/kWh ref
    C2   = 1.0 / (1.0 + CAPEX / 1e9)
    EcSI = 0.5 * C1 + 0.5 * C2

    # ── 6. TRI (Technical Reliability) ───────────────────────────────────────
    T1         = 1.0           # LPSP gate already passed
    T2         = Aut_factor    # = 1.0
    # T3: MATLAB exact — continuous gradient for PSO optimisation.
    # 1-exp(-redundancy/3) never fully saturates so PSO always has incentive
    # to improve storage/FC, driving better balanced designs.
    redundancy = x[2] / 1000.0 + x[5] / 100.0
    T3         = 1.0 - np.exp(-redundancy / 3.0)
    TRI        = 0.6 * T1 + 0.2 * T2 + 0.2 * T3

    # ── 7. ORI (Operational Resilience) ──────────────────────────────────────
    gen_mix = x[0] + x[1]
    R1      = T2
    R2      = 1.0 - np.exp(-gen_mix / 800.0)
    R3      = 1.0 - np.exp(-x[5] / 80.0)
    ORI     = 0.4 * R1 + 0.3 * R2 + 0.3 * R3

    # ── 8. LSI (Logistics) ────────────────────────────────────────────────────
    L1  = T2
    L2  = 1.0 / (1.0 + CAPEX / 1e9)
    L3  = 1.0 - np.exp(-x[2] / 2000.0)
    LSI = 0.4 * L1 + 0.3 * L2 + 0.3 * L3

    # ── 9. MPBSI ──────────────────────────────────────────────────────────────
    # ── MPBSI final score — use user weights or MATLAB defaults ──────────────
    _w = weights or {}
    w_esi  = float(_w.get("w_esi",  0.15))
    w_ecsi = float(_w.get("w_ecsi", 0.25))
    w_tri  = float(_w.get("w_tri",  0.30))
    w_ori  = float(_w.get("w_ori",  0.15))
    w_lsi  = float(_w.get("w_lsi",  0.15))
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

    Key differences vs mission dispatch:
      - Wind: hub-height correction applied (Hhub=40m, alpha=0.14)
      - Autonomy: theoretical (usable_storage / critical_daily), not simulated
      - Feasibility: no hard autonomy threshold (smooth reward instead)
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

    # PV model (same as mission)
    Gref = 1000.0; KT = -0.004; etaPV = 0.95
    Tcell  = temp + 0.0256 * ghi
    P_PV_t = P_PV * (ghi / Gref) * (1 + KT * (Tcell - 25.0)) * etaPV
    P_PV_t = np.maximum(P_PV_t, 0.0)

    # Wind model — Resource uses hub-height correction (Hhub=40m, alpha=0.14)
    Href = 10.0; Hhub = 40.0; alpha = 0.14
    V2   = wind * (Hhub / Href) ** alpha
    V_ci = 3.0; V_r = 12.0; V_co = 25.0
    P_WT_t = np.where(
        (V2 < V_ci) | (V2 >= V_co), 0.0,
        np.where(V2 < V_r, P_WT * ((V2 - V_ci) / (V_r - V_ci)) ** 3, P_WT),
    )
    P_WT_t = 0.90 * P_WT_t  # mechanical + generator efficiency

    P_RES = P_PV_t + P_WT_t

    # Load split
    Load_c = 0.60 * load
    Load_s = 0.25 * load
    Load_n = 0.15 * load

    # Storage params
    eta_ch = 0.95; eta_dis = 0.95; eta_EL = 0.70; eta_FC = 0.55
    SOC     = 0.8  * E_BESS
    SOC_min = 0.20 * E_BESS
    SOC_max = E_BESS
    E_H2    = 0.5  * E_H2_max

    unmet_c = 0.0; curt_semi = 0.0; curt_non = 0.0
    Annual_RES = 0.0                                   # MATLAB: initialise to 0, accumulate in loop

    for t in range(N):
        FC_used     = False
        P_available = P_RES[t]
        Annual_RES += P_available                       # MATLAB: Annual_RES += P_available

        # ── Critical load ─────────────────────────────────────────────────────
        deficit = Load_c[t] - P_available               # MATLAB: deficit = Load_c(t) - P_available

        if deficit > 0:
            # Battery discharge
            max_bat = (SOC - SOC_min) * eta_dis
            P_dis   = min(deficit, max_bat)
            SOC    -= P_dis / eta_dis
            deficit -= P_dis

            # Fuel cell dispatch
            max_fc_energy = E_H2 * eta_FC
            max_fc  = min(P_FC_max, max_fc_energy)
            P_fc    = min(deficit, max_fc)
            if P_fc > 0:
                FC_used = True
            E_H2   -= P_fc / eta_FC
            deficit -= P_fc
            if deficit > 0:
                unmet_c += deficit
            P_available = 0.0
        else:
            P_available = -deficit                      # surplus after critical

        # ── Semi load ─────────────────────────────────────────────────────────
        if P_available >= Load_s[t]:
            P_available -= Load_s[t]
        else:
            curt_semi   += Load_s[t] - P_available
            P_available  = 0.0

        # ── Non-critical load ─────────────────────────────────────────────────
        if P_available >= Load_n[t]:
            P_available -= Load_n[t]
        else:
            curt_non   += Load_n[t] - P_available
            P_available = 0.0

        # ── Charging (battery first, then H2 if FC not used) ──────────────────
        if P_available > 0:
            # Battery charge — MATLAB: max_ch=(SOC_max-SOC)/eta_ch; P_ch=min(P_avail,max_ch)
            max_ch = (SOC_max - SOC) / eta_ch
            P_ch   = min(P_available, max_ch)
            SOC   += eta_ch * P_ch
            P_available -= P_ch                         # MATLAB: P_available -= P_ch

            # Electrolyzer only if FC not used — MATLAB: if ~FC_used
            if not FC_used and P_available > 0:
                max_h2_room = (E_H2_max - E_H2) / eta_EL
                P_el  = min(P_available, P_EL_max, max_h2_room)
                E_H2 += eta_EL * P_el
                P_available -= P_el                     # MATLAB: P_available -= P_el

        SOC  = float(np.clip(SOC,  SOC_min, SOC_max))
        E_H2 = float(np.clip(E_H2, 0.0,    E_H2_max))

    Total_Load     = float(np.sum(load))
    Total_critical = float(np.sum(Load_c))

    lpsp_critical  = unmet_c / max(Total_critical, 1e-9)
    renewable_ratio = Annual_RES / max(Total_Load, 1e-9)

    # Theoretical autonomy (Resource mode — not simulated)
    usable_battery = (SOC_max - SOC_min)
    usable_h2_elec = E_H2_max * eta_FC
    critical_daily = Total_critical / 365.0
    autonomy_days  = (usable_battery + usable_h2_elec) / max(critical_daily, 1e-9)

    return SimulationResult(
        is_feasible         = (lpsp_critical <= 1e-4) and (renewable_ratio >= 0.999),
        lpsp_critical       = lpsp_critical,
        renewable_ratio     = renewable_ratio,
        autonomy_days       = autonomy_days,
        annual_load_MWh     = Total_Load / 1000.0,
        total_renewable_MWh = Annual_RES / 1000.0,
        curtailed_semi_MWh  = curt_semi / 1000.0,
        curtailed_non_MWh   = curt_non  / 1000.0,
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
    if (Area_PV + Area_WT) > (land_available + 1e-2):
        return MPBSIResult(mpbsi=-1e6, pillars=Pillars(ESI=0.0,EcSI=0.0,TRI=0.0,ORI=0.0,LSI=0.0), is_feasible=False, simulation=None)

    sim = microgrid_dispatch_resource(x, base)

    Annual_Load = sim.annual_load_MWh
    Annual_RES  = sim.total_renewable_MWh

    # Feasibility (Resource mode — no autonomy hard limit)
    if sim.lpsp_critical > 1e-4 or Annual_RES < Annual_Load:
        return MPBSIResult(mpbsi=-1e6, pillars=Pillars(ESI=0.0,EcSI=0.0,TRI=0.0,ORI=0.0,LSI=0.0), is_feasible=False, simulation=sim)

    eta_FC = 0.55
    usable_storage = (0.95 - 0.20) * x[2] + x[4] * eta_FC
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
    E4 = 1.0 - np.exp(-storage_ratio / 3.0)
    ESI = 0.30*E1 + 0.30*E2 + 0.20*E3 + 0.20*E4

    # ── EcSI ─────────────────────────────────────────────────────────────────
    CAPEX = (55000*x[0] + 120000*x[1] + 15000*x[2] +
             70000*x[3] + 15000*x[4] + 110000*x[5])
    LCOE  = CAPEX / max(Annual_RES * 1000.0 * 20.0, 1e-6)
    C1 = 1.0 / (1.0 + LCOE / 50.0)    # normalised to Rs 50/kWh ref
    C2 = 1.0 / (1.0 + CAPEX / 1e9)
    OM_ratio = (0.02 * CAPEX) / max(Annual_RES * 1000.0, 1.0)
    C3 = 1.0 / (1.0 + 50.0 * OM_ratio)
    C4 = 1.0
    EcSI = 0.35*C1 + 0.30*C2 + 0.20*C3 + 0.15*C4

    # ── TRI ──────────────────────────────────────────────────────────────────
    T1 = 1.0
    # T2/T3/T4: MATLAB exact exponential saturations — continuous PSO gradient
    T2 = 1.0 - np.exp(-sim.autonomy_days / 1.5)
    redundancy = (x[2] / 1000.0 + x[5] / 100.0)
    T3 = 1.0 - np.exp(-redundancy / 3.0)
    T4 = 1.0 - np.exp(-storage_ratio / 3.0)
    TRI = 0.40*T1 + 0.30*T2 + 0.20*T3 + 0.10*T4

    # ── ORI ──────────────────────────────────────────────────────────────────
    R1 = T2
    gen_mix = x[0] + x[1]
    R2 = 1.0 - np.exp(-gen_mix / 800.0)
    R3 = 1.0
    R4 = 1.0 - np.exp(-x[5] / 80.0)
    ORI = 0.35*R1 + 0.25*R2 + 0.20*R3 + 0.20*R4

    # ── LSI ──────────────────────────────────────────────────────────────────
    L1 = 1.0
    L2 = T2
    L3 = 1.0 / (1.0 + CAPEX / 1e9)
    L4 = T4
    L5 = 1.0 - np.exp(-x[2] / 2000.0)
    LSI = 0.25*L1 + 0.25*L2 + 0.20*L3 + 0.15*L4 + 0.15*L5

    # ── MPBSI (Resource weights) ──────────────────────────────────────────────
    # ESI:0.05  EcSI:0.20  TRI:0.30  ORI:0.25  LSI:0.20
    # ── MPBSI final score — use user weights or MATLAB Resource defaults ──────
    _w = weights or {}
    w_esi  = float(_w.get("w_esi",  0.05))
    w_ecsi = float(_w.get("w_ecsi", 0.20))
    w_tri  = float(_w.get("w_tri",  0.30))
    w_ori  = float(_w.get("w_ori",  0.25))
    w_lsi  = float(_w.get("w_lsi",  0.20))
    MPBSI = w_esi*ESI + w_ecsi*EcSI + w_tri*TRI + w_ori*ORI + w_lsi*LSI

    pillars = Pillars(ESI=round(ESI,6), EcSI=round(EcSI,6),
                      TRI=round(TRI,6), ORI=round(ORI,6), LSI=round(LSI,6))
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
    """
    Required_Autonomy = 7.0

    res = mpbsi_evaluator(x, base, land_available, weights=weights)
    if res.mpbsi < 0:
        return (1e3, 1e3)

    sim = res.simulation
    if (not sim.is_feasible) or (sim.autonomy_days < Required_Autonomy):
        return (1e3, 1e3)

    f1 = -res.mpbsi

    Cost_PV  = 55_000  * x[0]
    Cost_WT  = 120_000 * x[1]
    Cost_Bat = 15_000  * x[2]
    Cost_EL  = 70_000  * x[3]
    Cost_H2  = 15_000  * x[4]
    Cost_FC  = 110_000 * x[5]
    CAPEX    = Cost_PV + Cost_WT + Cost_Bat + Cost_EL + Cost_H2 + Cost_FC

    r        = 0.08; n = 20
    Bat_repl = Cost_Bat / (1 + r) ** 10
    FC_repl  = Cost_FC  / (1 + r) ** 10
    EL_repl  = Cost_EL  / (1 + r) ** 15
    OM_ann   = 0.02 * CAPEX
    OM_total = OM_ann * ((1 - (1 + r) ** (-n)) / r)
    NPC      = CAPEX + Bat_repl + FC_repl + EL_repl + OM_total

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
    Cost_PV  = 55_000  * x[0]; Cost_WT  = 120_000 * x[1]
    Cost_Bat = 15_000  * x[2]; Cost_EL  = 70_000  * x[3]
    Cost_H2  = 15_000  * x[4]; Cost_FC  = 110_000 * x[5]
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
    seed:              int   = 42,
    land_available:    float = 50_000.0,
    var_min:           Optional[np.ndarray] = None,
    var_max:           Optional[np.ndarray] = None,
    progress_callback: Optional[Callable]   = None,
    mode:              str   = "mission",
    weights:           Optional[dict]       = None,
) -> OptimizationResult:
    """
    Exact translation of PSO_MPBSI_Mission_LandConstrained.m.

    Physics-derived bounds (MATLAB):
      PV_min  = 0.5 × Avg_Load,   PV_max  = Land/10
      Wind_min= 0,                 Wind_max= Land/15
      BESS_min= 0.5×Crit_Daily,   BESS_max= 5×Crit_Daily
      EL_min  = 0,                 EL_max  = 0.5×(PV_max+Wind_max)
      H2_min  = 0,                 H2_max  = 10×Crit_Daily/η_FC
      FC_min  = 0.5×Crit_Peak,    FC_max  = 1.2×Crit_Peak

    PSO parameters (MATLAB exact):
      w=0.8, wdamp=0.98, c1=1.5, c2=1.5, VelMax=0.2×(VarMax-VarMin)
    """
    rng = np.random.default_rng(seed)
    t0  = time.perf_counter()

    # ── Physics-derived bounds ────────────────────────────────────────────────
    if var_min is None or var_max is None:
        Peak_Load     = float(np.max(base.load))
        Avg_Load      = float(np.mean(base.load))
        Critical_Load = 0.60 * base.load
        Critical_Peak = 0.60 * Peak_Load
        Crit_Daily    = float(np.sum(Critical_Load)) / 365.0

        PV_min   = 0.5 * Avg_Load;         PV_max   = land_available / 10.0
        Wind_min = 0.0;                     Wind_max = land_available / 15.0
        BESS_min = 0.5 * Crit_Daily;       BESS_max = 5.0 * Crit_Daily
        EL_min   = 0.0;                     EL_max   = 0.5 * (PV_max + Wind_max)
        H2_min   = 0.0;                     H2_max   = (10.0 * Crit_Daily) / 0.55
        FC_min   = 0.5 * Critical_Peak;    FC_max   = 1.2 * Critical_Peak

        vmin = np.array([PV_min, Wind_min, BESS_min, EL_min, H2_min, FC_min])
        vmax = np.array([PV_max, Wind_max, BESS_max, EL_max, H2_max, FC_max])
        logger.info("PSO bounds (physics-derived): %s … %s", vmin.round(1), vmax.round(1))
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
        if progress_callback is not None and (i + 1) % 5 == 0:
            try:
                progress_callback(0, max_it, float(gbest_cost),
                                  int(gbest_cost > -1e5),
                                  f"Init {i+1}/{n_pop}")
            except Exception:
                pass

    convergence = [float(gbest_cost)]
    logger.info("PSO init | Best MPBSI = %.4f | land=%.0f m²", gbest_cost, land_available)

    # ── Main loop  (MATLAB: w = w * wdamp each iteration) ─────────────────────
    w_cur      = w

    for it in range(1, max_it + 1):
        for i in range(n_pop):
            r1 = rng.random(nVar)
            r2 = rng.random(nVar)
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
#  NSGA-II OPTIMIZER  (NSGA_Objective_Mission.m framework)
# ══════════════════════════════════════════════════════════════════════════════

def _fast_non_dominated_sort(costs: np.ndarray):
    n = costs.shape[0]
    S     = [[] for _ in range(n)]
    ndom  = np.zeros(n, dtype=int)
    rank  = np.zeros(n, dtype=int)
    front = [[]]
    for p in range(n):
        for q in range(n):
            if p == q: continue
            if np.all(costs[p] <= costs[q]) and np.any(costs[p] < costs[q]):
                S[p].append(q)
            elif np.all(costs[q] <= costs[p]) and np.any(costs[q] < costs[p]):
                ndom[p] += 1
        if ndom[p] == 0:
            rank[p] = 0; front[0].append(p)
    i = 0
    while front[i]:
        nf = []
        for p in front[i]:
            for q in S[p]:
                ndom[q] -= 1
                if ndom[q] == 0:
                    rank[q] = i + 1; nf.append(q)
        i += 1; front.append(nf)
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
    seed:              int   = 42,
    land_available:    float = 50_000.0,
    var_min:           Optional[np.ndarray] = None,
    var_max:           Optional[np.ndarray] = None,
    progress_callback: Optional[Callable]   = None,
    mode:              str   = "mission",
    weights:           Optional[dict]       = None,
) -> OptimizationResult:
    """
    NSGA-II with objectives from NSGA_Objective_Mission.m:
      f1 = −MPBSI   (maximise)
      f2 = NPC/1e8  (minimise)
    """
    rng = np.random.default_rng(seed)
    t0  = time.perf_counter()

    if var_min is None or var_max is None:
        # NSGA_MASTER_MISSION.m bounds (wider than PSO — more Pareto diversity)
        Peak_Load     = float(np.max(base.load))
        Critical_Load = 0.60 * base.load
        Crit_Peak     = 0.60 * Peak_Load
        Crit_Daily    = float(np.sum(Critical_Load)) / 365.0
        PV_max   = land_available / 10.0
        Wind_max = land_available / 15.0
        # MATLAB exact: BESS_max=20x, EL_max=PV+Wind, H2_max=20x/eta, FC_max=2x
        vmin = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.5 * Crit_Peak])
        vmax = np.array([PV_max, Wind_max, 20.0*Crit_Daily,
                         PV_max+Wind_max, 20.0*Crit_Daily/0.55, 2.0*Crit_Peak])
        logger.info("NSGA-II bounds (NSGA_MASTER): BESS_max=%.0f H2_max=%.0f FC_max=%.0f",
                    vmax[2], vmax[4], vmax[5])
    else:
        vmin = np.asarray(var_min, dtype=float)
        vmax = np.asarray(var_max, dtype=float)

    nVar = len(vmin); nObj = 2

    def eval_multi(pos):
        if mode == "resource":
            r = mpbsi_evaluator_resource(pos, base, land_available, weights=weights)
            f1 = -r.mpbsi if r.mpbsi > -1e5 else 1e3
            f2 = (55000*pos[0] + 120000*pos[1] + 15000*pos[2] +
                  70000*pos[3] + 15000*pos[4] + 110000*pos[5]) / 1e8
            if r.mpbsi < -1e5: f2 = 1e3
        else:
            f1, f2 = nsga_objective_mission(pos, base, land_available, weights=weights)
        return np.array([f1, f2])

    def eval_mpbsi(pos):
        if mode == "resource":
            r = mpbsi_evaluator_resource(pos, base, land_available, weights=weights)
        else:
            r = mpbsi_evaluator(pos, base, land_available, weights=weights)
        return r

    pop      = rng.uniform(vmin, vmax, (n_pop, nVar))
    obj_vals = np.array([eval_multi(pop[i]) for i in range(n_pop)])
    mpbsi_r  = [eval_mpbsi(pop[i]) for i in range(n_pop)]

    convergence     = []
    best_mpbsi_ever = -np.inf
    best_result     = None
    best_x          = pop[0].copy()

    def _sbx(p1, p2, eta=15.0):
        u = rng.random(nVar)
        b = np.where(u < 0.5,
                     (2*u)**(1/(eta+1)),
                     (1/(2*(1-u)))**(1/(eta+1)))
        c1v = np.clip(0.5*((1+b)*p1 + (1-b)*p2), vmin, vmax)
        c2v = np.clip(0.5*((1-b)*p1 + (1+b)*p2), vmin, vmax)
        return c1v, c2v

    def _mutate(x, eta=20.0):
        xm = x.copy(); pm = 1.0/nVar
        for j in range(nVar):
            if rng.random() < pm:
                u = rng.random()
                d = (2*u)**(1/(eta+1))-1 if u<0.5 else 1-(2*(1-u))**(1/(eta+1))
                xm[j] = np.clip(xm[j]+d*(vmax[j]-vmin[j]), vmin[j], vmax[j])
        return xm

    for gen in range(max_gen):
        fronts, ranks = _fast_non_dominated_sort(obj_vals)
        cd_arr = np.zeros(n_pop)
        for f in fronts:
            if len(f) >= 2:
                cd = _crowding_distance(obj_vals, f)
                for k, idx in enumerate(f):
                    cd_arr[idx] = cd[k]

        offspring     = np.empty((n_pop, nVar))
        off_obj       = np.empty((n_pop, nObj))
        off_r         = [None] * n_pop

        for i in range(0, n_pop, 2):
            pool = rng.integers(0, n_pop, 4)
            def tour(a, b):
                return a if ranks[a] < ranks[b] or (ranks[a]==ranks[b] and cd_arr[a]>cd_arr[b]) else b
            p1 = tour(pool[0], pool[1])
            p2 = tour(pool[2], pool[3])
            c1v, c2v = _sbx(pop[p1], pop[p2])
            c1v = _mutate(c1v); c2v = _mutate(c2v)
            offspring[i]   = c1v
            off_obj[i]     = eval_multi(c1v)
            off_r[i]       = eval_mpbsi(c1v)
            if i+1 < n_pop:
                offspring[i+1] = c2v
                off_obj[i+1]   = eval_multi(c2v)
                off_r[i+1]     = eval_mpbsi(c2v)

        comb_pop = np.vstack([pop, offspring])
        comb_obj = np.vstack([obj_vals, off_obj])
        comb_r   = mpbsi_r + off_r

        fronts_c, _ = _fast_non_dominated_sort(comb_obj)
        selected = []
        for f in fronts_c:
            if len(selected) + len(f) <= n_pop:
                selected.extend(f)
            else:
                rem = n_pop - len(selected)
                cd  = _crowding_distance(comb_obj, f)
                order = np.argsort(-cd)
                selected.extend([f[k] for k in order[:rem]])
                break

        pop       = comb_pop[selected]
        obj_vals  = comb_obj[selected]
        mpbsi_r   = [comb_r[i] for i in selected]

        gen_best = -np.inf
        for i in range(n_pop):
            r = mpbsi_r[i]
            if r and r.is_feasible and r.mpbsi > gen_best:
                gen_best = r.mpbsi
            if r and r.is_feasible and r.mpbsi > best_mpbsi_ever:
                best_mpbsi_ever = r.mpbsi
                best_result     = r
                best_x          = pop[i].copy()

        best_val = gen_best if gen_best > -np.inf else best_mpbsi_ever
        convergence.append(float(best_val))
        logger.info("NSGA-II gen %2d | Best MPBSI = %.4f", gen+1, best_val)

        if progress_callback is not None:
            try:
                fc = int(sum(1 for r in mpbsi_r if r and r.is_feasible))
                progress_callback(gen+1, max_gen, float(best_val), fc)
            except Exception:
                pass

    runtime = time.perf_counter() - t0
    logger.info("NSGA-II done | Best MPBSI = %.4f | Runtime: %.1f s",
                best_mpbsi_ever, runtime)

    metrics = {}
    if best_result and best_result.is_feasible:
        s = best_result.simulation
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
        algorithm="NSGA-II",
        best_x=best_x.tolist(),
        best_mpbsi=float(best_mpbsi_ever),
        best_pillars=best_result.pillars.to_dict() if best_result else {},
        convergence=convergence,
        runtime_seconds=round(runtime, 2),
        feasible=bool(best_result.is_feasible if best_result else False),
        reliability_metrics=metrics,
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
) -> dict:
    """
    Full 10-step MPBSI pipeline + optimization.

    Returns JSON-serialisable dict compatible with mpbsi_complete_web.html.
    """
    cfg = {
        "seed":           42,
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
        # NSGA-II
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
    }
    if config:
        cfg.update(config)

    # ── Resolve MPBSI pillar weights ──────────────────────────────────────────
    # Use mode defaults if not overridden by user
    _mission_w  = {"w_esi": 0.15, "w_ecsi": 0.25, "w_tri": 0.30, "w_ori": 0.15, "w_lsi": 0.15}
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

    if _b and all(k in _b for k in ["pv_min", "pv_max"]):
        var_min_arr = np.array([
            float(_b.get("pv_min",   0)), float(_b.get("wind_min", 0)),
            float(_b.get("batt_min", 0)), float(_b.get("elec_min", 0)),
            float(_b.get("h2_min",   0)), float(_b.get("fc_min",   0)),
        ])
        var_max_arr = np.array([
            float(_b.get("pv_max",   9999)), float(_b.get("wind_max", 9999)),
            float(_b.get("batt_max", 99999)), float(_b.get("elec_max", 9999)),
            float(_b.get("h2_max",   99999)), float(_b.get("fc_max",   9999)),
        ])
    else:
        var_min_arr = var_max_arr = None   # physics-derived

    # ─── Optimization ─────────────────────────────────────────────────────────
    algo_upper = algo.strip().upper()

    if algo_upper == "PSO":
        opt = pso_optimize(
            base,
            n_pop=cfg["pso_n_pop"],
            max_it=cfg["pso_max_it"],
            w=cfg["pso_w"],
            wdamp=cfg.get("pso_wdamp", 0.98),
            c1=cfg["pso_c1"],
            c2=cfg["pso_c2"],
            seed=cfg["seed"],
            land_available=land_avail,
            var_min=var_min_arr,
            var_max=var_max_arr,
            progress_callback=progress_callback,
            mode=mode,
            weights=weights,
        )
    elif algo_upper in ("NSGA-II", "NSGA2"):
        opt = nsga2_optimize(
            base,
            n_pop=cfg["nsga2_n_pop"],
            max_gen=cfg["nsga2_max_gen"],
            seed=cfg["seed"],
            land_available=land_avail,
            var_min=var_min_arr,
            var_max=var_max_arr,
            progress_callback=progress_callback,
            mode=mode,
            weights=weights,
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
    print("═"*60)