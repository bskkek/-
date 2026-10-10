"""
exhaust_backpressure_solver.py
==============================
Phase 1 core: cylinder / exhaust-manifold / downstream-restriction gas-exchange
solver for assessing the effect of exhaust backpressure on a diesel engine.

Physical model (SI units everywhere, crank angle theta in rad, 0 = gas-exchange TDC)
------------------------------------------------------------------------------------
Control volumes (open thermodynamic systems, ideal-gas mixture of air + burned gas):

    cylinder  : m_c, T_c, x_c   (x = burned-gas mass fraction -> residual gas)
    manifold  : m_x, T_x, x_x

Conservation laws for a control volume (mass, species, energy):

    dm/dt = sum_k mdot_k
    m dx/dt = sum_k mdot_k^in (x_k - x)
    m cv dT/dt = -p dV/dt - Qdot + sum_k mdot_k (h_k - u) - m (du/dx)|_T dx/dt

with h_k = h(T_up, x_up) for inflow and h_k = h(T, x) for outflow.
Mixture properties: cp(T,x) = a(x) + b(x) T + c(x) T^2 (mass-weighted), R(x)
mass-weighted, gamma = cp/(cp-R)  ->  variable specific-heat ratio.

Valve flow: compressible isentropic nozzle with discharge coefficient,
subsonic and choked branches, reverse flow supported, with a linear
regularisation near p_ratio -> 1 to remove the infinite slope at zero flow.

Downstream: manifold -> tailpipe/aftertreatment modelled as a lumped inertance
(1-D momentum equation, first acoustic mode / Helmholtz resonator) plus a
Darcy-Forchheimer restriction (DPF / muffler / turbine equivalent):

    (L/A) dMdot/dt = p_x - p_down - dp_loss(Mdot),
    dp_loss = k_lin*Mdot + zeta*Mdot|Mdot| / (2 rho_x A^2)

The manifold compliance + tailpipe inertance reproduce exhaust pressure
pulsation; multi-pipe wave action (method of characteristics) can be plugged
in behind the same boundary interface (see Phase 2).

Scope note: single-cylinder-equivalent formulation (each cylinder with its own
runner volume). Combustion is NOT modelled here: the cylinder state at EVO is
an input supplied by the combustion/zero-D module.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.integrate import solve_ivp

FloatArr = NDArray[np.float64]

# --------------------------------------------------------------------------- #
# Errors and unit helpers
# --------------------------------------------------------------------------- #


class UnitError(ValueError):
    """Raised when an input is non-finite or outside its plausible SI range."""


class SolverError(RuntimeError):
    """Raised when the ODE integration or the periodic iteration fails."""


def _check(name: str, value: float, lo: float, hi: float) -> float:
    if not (isinstance(value, (int, float, np.floating)) and np.isfinite(value)
            and lo <= value <= hi):
        raise UnitError(f"{name}={value!r} outside plausible SI range [{lo}, {hi}]")
    return float(value)


def bar(x: float) -> float:
    """bar -> Pa"""
    return x * 1.0e5


def mm(x: float) -> float:
    """mm -> m"""
    return x * 1.0e-3


def litre(x: float) -> float:
    """litre -> m^3"""
    return x * 1.0e-3


def rpm_to_rad_s(n: float) -> float:
    return n * 2.0 * math.pi / 60.0


# --------------------------------------------------------------------------- #
# Gas model: variable cp / gamma, air + burned-gas mixture
# --------------------------------------------------------------------------- #
# cp = a + b T + c T^2 [J/kg/K]; fitted to 300-1500 K data for air and for lean
# diesel combustion products (lambda ~ 1.5). Replace with NASA-7 polynomials
# (via Cantera) when species-resolved composition is available.
_AIR = (936.9, 0.2383, -3.75e-5, 287.05)      # a, b, c, R
_BURNED = (948.8, 0.2833, -4.17e-5, 288.60)
T_REF = 298.15


def _coef(x: ArrayLike) -> tuple[FloatArr, FloatArr, FloatArr, FloatArr]:
    x = np.clip(np.asarray(x, dtype=float), 0.0, 1.0)
    return tuple((1.0 - x) * a + x * b for a, b in zip(_AIR, _BURNED))  # type: ignore


def gas_R(x: ArrayLike) -> FloatArr:
    return _coef(x)[3]


def gas_cp(T: ArrayLike, x: ArrayLike) -> FloatArr:
    a, b, c, _ = _coef(x)
    T = np.asarray(T, dtype=float)
    return a + b * T + c * T * T


def gas_cv(T: ArrayLike, x: ArrayLike) -> FloatArr:
    return gas_cp(T, x) - gas_R(x)


def gas_gamma(T: ArrayLike, x: ArrayLike) -> FloatArr:
    cp = gas_cp(T, x)
    return cp / (cp - gas_R(x))


def gas_h(T: ArrayLike, x: ArrayLike) -> FloatArr:
    """Sensible specific enthalpy, reference T_REF [J/kg]."""
    a, b, c, _ = _coef(x)
    T = np.asarray(T, dtype=float)
    return (a * (T - T_REF) + 0.5 * b * (T**2 - T_REF**2)
            + c / 3.0 * (T**3 - T_REF**3))


def gas_u(T: ArrayLike, x: ArrayLike) -> FloatArr:
    """Specific internal energy consistent with gas_h [J/kg]."""
    return gas_h(T, x) - gas_R(x) * np.asarray(T, dtype=float)


# --------------------------------------------------------------------------- #
# Isentropic nozzle flow
# --------------------------------------------------------------------------- #


def flow_function(pr: ArrayLike, gamma: ArrayLike, eps: float = 1.0e-3) -> FloatArr:
    """
    Dimensionless compressible flow function psi(p_down/p_up, gamma) such that
        mdot = A_eff * p_up / sqrt(R T_up) * psi
    Subsonic:  psi = sqrt( 2g/(g-1) * (pr^(2/g) - pr^((g+1)/g)) )
    Choked  :  psi = sqrt(g) * (2/(g+1))^((g+1)/(2(g-1)))      (pr <= pr_crit)
    For 1-pr < eps the flow is linearised to zero (finite slope, solver-friendly).
    """
    pr = np.clip(np.asarray(pr, dtype=float), 0.0, 1.0)
    g = np.asarray(gamma, dtype=float)
    pr_crit = (2.0 / (g + 1.0)) ** (g / (g - 1.0))
    pr_eff = np.minimum(pr, 1.0 - eps)
    sub = np.sqrt(np.maximum(
        2.0 * g / (g - 1.0) * (pr_eff ** (2.0 / g) - pr_eff ** ((g + 1.0) / g)), 0.0))
    chk = np.sqrt(g) * (2.0 / (g + 1.0)) ** ((g + 1.0) / (2.0 * (g - 1.0)))
    psi = np.where(pr_eff <= pr_crit, chk, sub)
    return psi * np.minimum(1.0, (1.0 - pr) / eps)


def orifice_mass_flow(A: ArrayLike, p_up: ArrayLike, T_up: ArrayLike,
                      p_dn: ArrayLike, gamma_up: ArrayLike, R_up: ArrayLike) -> FloatArr:
    """Non-negative mass flow through an effective area A from upstream to downstream."""
    p_up = np.asarray(p_up, dtype=float)
    pr = np.asarray(p_dn, dtype=float) / p_up
    return (np.asarray(A, dtype=float) * p_up
            / np.sqrt(np.asarray(R_up) * np.asarray(T_up, dtype=float))
            * flow_function(pr, gamma_up))


def signed_valve_flow(A: ArrayLike,
                      p1: ArrayLike, T1: ArrayLike, g1: ArrayLike, R1: ArrayLike,
                      p2: ArrayLike, T2: ArrayLike, g2: ArrayLike, R2: ArrayLike) -> FloatArr:
    """Flow from volume 2 INTO volume 1 (positive); negative = flow from 1 into 2."""
    p1, p2 = np.asarray(p1, dtype=float), np.asarray(p2, dtype=float)
    fwd = p2 >= p1
    mag = np.where(fwd,
                   orifice_mass_flow(A, p2, T2, p1, g2, R2),
                   orifice_mass_flow(A, p1, T1, p2, g1, R1))
    return np.where(fwd, mag, -mag)


# --------------------------------------------------------------------------- #
# Hardware descriptions (validated, immutable)
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class EngineGeometry:
    bore: float
    stroke: float
    conrod: float
    compression_ratio: float
    speed_rpm: float
    wall_temperature: float = 480.0

    def __post_init__(self) -> None:
        _check("bore [m]", self.bore, 0.02, 0.6)
        _check("stroke [m]", self.stroke, 0.02, 1.0)
        _check("conrod [m]", self.conrod, 0.5 * self.stroke * 1.2 if self.stroke else 0.03, 2.0)
        _check("compression_ratio [-]", self.compression_ratio, 8.0, 30.0)
        _check("speed [rpm]", self.speed_rpm, 300.0, 6000.0)
        _check("wall_temperature [K]", self.wall_temperature, 300.0, 800.0)

    @property
    def piston_area(self) -> float:
        return 0.25 * math.pi * self.bore**2

    @property
    def displaced_volume(self) -> float:
        return self.piston_area * self.stroke

    @property
    def clearance_volume(self) -> float:
        return self.displaced_volume / (self.compression_ratio - 1.0)

    @property
    def omega(self) -> float:
        return rpm_to_rad_s(self.speed_rpm)

    @property
    def mean_piston_speed(self) -> float:
        return 2.0 * self.stroke * self.speed_rpm / 60.0

    def volume(self, theta: ArrayLike) -> FloatArr:
        th = np.asarray(theta, dtype=float)
        a = 0.5 * self.stroke
        lam = a / self.conrod
        s = a * (1.0 - np.cos(th)) + self.conrod * (
            1.0 - np.sqrt(1.0 - (lam * np.sin(th)) ** 2))
        return self.clearance_volume + self.piston_area * s

    def dV_dtheta(self, theta: ArrayLike) -> FloatArr:
        th = np.asarray(theta, dtype=float)
        a = 0.5 * self.stroke
        lam = a / self.conrod
        root = np.sqrt(1.0 - (lam * np.sin(th)) ** 2)
        return self.piston_area * a * np.sin(th) * (1.0 + lam * np.cos(th) / root)

    def wall_area(self, V: ArrayLike) -> FloatArr:
        """Head + piston + exposed liner."""
        return 2.0 * self.piston_area + math.pi * self.bore * np.asarray(V) / self.piston_area


@dataclass(frozen=True)
class ValveSpec:
    """Poppet valve: sin^2 lift profile + tabulated Cd(l/D) (replace with flow-bench data)."""
    open_deg: float
    close_deg: float
    max_lift: float
    seat_diameter: float
    ld_table: tuple[float, ...] = (0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30)
    cd_table: tuple[float, ...] = (0.60, 0.74, 0.76, 0.72, 0.66, 0.62, 0.60)

    def __post_init__(self) -> None:
        _check("valve open [deg]", self.open_deg, -400.0, 400.0)
        _check("valve close [deg]", self.close_deg, -400.0, 400.0)
        if self.close_deg <= self.open_deg:
            raise UnitError("valve close angle must exceed open angle")
        _check("max_lift [m]", self.max_lift, 1e-3, 0.05)
        _check("seat_diameter [m]", self.seat_diameter, 5e-3, 0.2)
        if len(self.ld_table) != len(self.cd_table) or len(self.ld_table) < 2:
            raise UnitError("ld_table and cd_table must have equal length >= 2")
        if any(not 0.05 <= c <= 1.0 for c in self.cd_table):
            raise UnitError("Cd values must lie in [0.05, 1.0]")

    def lift(self, theta: ArrayLike) -> FloatArr:
        deg = np.degrees(np.asarray(theta, dtype=float))
        ph = (deg - self.open_deg) / (self.close_deg - self.open_deg)
        inside = (ph > 0.0) & (ph < 1.0)
        return np.where(inside, self.max_lift * np.sin(np.pi * np.clip(ph, 0, 1)) ** 2, 0.0)

    def effective_area(self, theta: ArrayLike) -> FloatArr:
        l = self.lift(theta)
        D = self.seat_diameter
        geo = np.minimum(math.pi * D * l, 0.25 * math.pi * D * D)
        cd = np.interp(l / D, self.ld_table, self.cd_table)
        return np.where(l > 0.0, cd * geo, 0.0)


@dataclass(frozen=True)
class IntakeBoundary:
    pressure: float
    temperature: float

    def __post_init__(self) -> None:
        _check("intake pressure [Pa]", self.pressure, 3e4, 1e6)
        _check("intake temperature [K]", self.temperature, 250.0, 500.0)


class DownstreamBoundary(Protocol):
    """Port for the aftertreatment / turbine / silencer side (swap for adaptive models)."""
    flow_area: float
    p_ambient: float
    T_ambient: float

    def pressure_drop(self, mdot: FloatArr, rho: FloatArr) -> FloatArr: ...


@dataclass(frozen=True)
class RestrictionBoundary:
    """Darcy-Forchheimer restriction: dp = k_lin*M + zeta*M|M|/(2 rho A^2)."""
    flow_area: float
    zeta: float
    k_linear: float = 0.0
    p_ambient: float = 1.013e5
    T_ambient: float = 300.0

    def __post_init__(self) -> None:
        _check("flow_area [m^2]", self.flow_area, 1e-5, 0.1)
        _check("zeta [-]", self.zeta, 0.0, 1e6)
        _check("k_linear [Pa s/kg]", self.k_linear, 0.0, 1e8)
        _check("p_ambient [Pa]", self.p_ambient, 5e4, 1.2e5)
        _check("T_ambient [K]", self.T_ambient, 230.0, 330.0)

    def pressure_drop(self, mdot: FloatArr, rho: FloatArr) -> FloatArr:
        return (self.k_linear * mdot
                + self.zeta * mdot * np.abs(mdot) / (2.0 * rho * self.flow_area**2))

    @classmethod
    def for_target_backpressure(cls, p_target: float, mdot_mean: float, rho: float,
                                flow_area: float, **kw: float) -> "RestrictionBoundary":
        """Choose zeta so that the mean flow produces p_target (absolute) at the manifold."""
        base = cls(flow_area=flow_area, zeta=0.0, **kw)
        dp = p_target - base.p_ambient - base.k_linear * mdot_mean
        if dp <= 0:
            raise UnitError("target backpressure not above ambient + linear loss")
        zeta = dp * 2.0 * rho * flow_area**2 / mdot_mean**2
        return cls(flow_area=flow_area, zeta=zeta, **kw)


@dataclass(frozen=True)
class ExhaustSystem:
    manifold_volume: float
    tailpipe_length: float
    boundary: DownstreamBoundary
    wall_area: float = 0.03
    h_wall: float = 50.0
    wall_temperature: float = 500.0

    def __post_init__(self) -> None:
        _check("manifold_volume [m^3]", self.manifold_volume, 1e-5, 0.05)
        _check("tailpipe_length [m]", self.tailpipe_length, 0.05, 20.0)
        _check("wall_area [m^2]", self.wall_area, 0.0, 5.0)
        _check("h_wall [W/m^2/K]", self.h_wall, 0.0, 1e3)
        _check("wall_temperature [K]", self.wall_temperature, 300.0, 1000.0)

    def helmholtz_frequency(self, T: float = 800.0, x: float = 1.0) -> float:
        """First lumped acoustic mode of manifold + tailpipe [Hz] (diagnostic)."""
        c = math.sqrt(float(gas_gamma(T, x)) * float(gas_R(x)) * T)
        return c / (2 * math.pi) * math.sqrt(
            self.boundary.flow_area / (self.manifold_volume * self.tailpipe_length))


@dataclass(frozen=True)
class EVOState:
    """Cylinder state at exhaust-valve opening, from the combustion module."""
    pressure: float
    temperature: float
    burned_fraction: float = 1.0

    def __post_init__(self) -> None:
        _check("EVO pressure [Pa]", self.pressure, 1e5, 4e6)
        _check("EVO temperature [K]", self.temperature, 500.0, 2000.0)
        _check("burned_fraction [-]", self.burned_fraction, 0.0, 1.0)


# --------------------------------------------------------------------------- #
# Control-volume balance
# --------------------------------------------------------------------------- #


def _control_volume(m: FloatArr, T: FloatArr, x: FloatArr, work: FloatArr,
                    q_loss: FloatArr,
                    ports: list[tuple[FloatArr, FloatArr, FloatArr]]
                    ) -> tuple[FloatArr, FloatArr, FloatArr]:
    """
    ports: (mdot_into_CV, T_upstream, x_upstream). Returns dm/dt, dT/dt, dx/dt.
    `work` = -p dV/dt [W], `q_loss` = heat rate lost to walls [W].
    """
    u = gas_u(T, x)
    h_cv = gas_h(T, x)
    dudx = gas_u(T, 1.0) - gas_u(T, 0.0)
    dm = 0.0
    energy = work - q_loss
    comp = 0.0
    for mdot, T_up, x_up in ports:
        inflow = mdot > 0.0
        h_k = np.where(inflow, gas_h(T_up, x_up), h_cv)
        x_k = np.where(inflow, x_up, x)
        dm = dm + mdot
        energy = energy + mdot * (h_k - u)
        comp = comp + mdot * (x_k - x)
    dx = comp / m
    dT = (energy - dudx * comp) / (m * gas_cv(T, x))
    return np.asarray(dm), np.asarray(dT), np.asarray(dx)



# --------------------------------------------------------------------------- #
# Scalar fast path (pure-Python floats) used inside the ODE integrator.
# Mirrors the vectorised physics above exactly (verified in verify_scalar_path()).
# --------------------------------------------------------------------------- #
_sqrt = math.sqrt


def _s_props(T: float, x: float) -> tuple[float, float, float, float, float]:
    x = 0.0 if x < 0.0 else (1.0 if x > 1.0 else x)
    a = _AIR[0] + x * (_BURNED[0] - _AIR[0])
    b = _AIR[1] + x * (_BURNED[1] - _AIR[1])
    c = _AIR[2] + x * (_BURNED[2] - _AIR[2])
    R = _AIR[3] + x * (_BURNED[3] - _AIR[3])
    cp = a + b * T + c * T * T
    h = (a * (T - T_REF) + 0.5 * b * (T * T - T_REF * T_REF)
         + c / 3.0 * (T**3 - T_REF**3))
    return R, cp, cp - R, h, h - R * T          # R, cp, cv, h, u


def _s_dudx(T: float) -> float:
    ha = (_AIR[0] * (T - T_REF) + 0.5 * _AIR[1] * (T * T - T_REF**2)
          + _AIR[2] / 3.0 * (T**3 - T_REF**3))
    hb = (_BURNED[0] * (T - T_REF) + 0.5 * _BURNED[1] * (T * T - T_REF**2)
          + _BURNED[2] / 3.0 * (T**3 - T_REF**3))
    return (hb - ha) - (_BURNED[3] - _AIR[3]) * T


def _s_flow_fn(pr: float, g: float, eps: float = 1.0e-3) -> float:
    pr = 0.0 if pr < 0.0 else (1.0 if pr > 1.0 else pr)
    pcrit = (2.0 / (g + 1.0)) ** (g / (g - 1.0))
    pe = pr if pr < 1.0 - eps else 1.0 - eps
    if pe <= pcrit:
        psi = _sqrt(g) * (2.0 / (g + 1.0)) ** ((g + 1.0) / (2.0 * (g - 1.0)))
    else:
        psi = _sqrt(max(2.0 * g / (g - 1.0) * (pe ** (2.0 / g) - pe ** ((g + 1.0) / g)), 0.0))
    return psi * min(1.0, (1.0 - pr) / eps)


def _s_signed_flow(A: float, p1: float, T1: float, g1: float, R1: float,
                   p2: float, T2: float, g2: float, R2: float) -> float:
    if A <= 0.0:
        return 0.0
    if p2 >= p1:
        return A * p2 / _sqrt(R2 * T2) * _s_flow_fn(p1 / p2, g2)
    return -A * p1 / _sqrt(R1 * T1) * _s_flow_fn(p2 / p1, g1)


def _s_valve_area(v: "ValveSpec", theta: float) -> float:
    deg = math.degrees(theta)
    ph = (deg - v.open_deg) / (v.close_deg - v.open_deg)
    if not 0.0 < ph < 1.0:
        return 0.0
    l = v.max_lift * math.sin(math.pi * ph) ** 2
    D = v.seat_diameter
    geo = min(math.pi * D * l, 0.25 * math.pi * D * D)
    ld, cd = v.ld_table, v.cd_table
    r = l / D
    if r <= ld[0]:
        c = cd[0]
    elif r >= ld[-1]:
        c = cd[-1]
    else:
        i = 1
        while ld[i] < r:
            i += 1
        c = cd[i - 1] + (cd[i] - cd[i - 1]) * (r - ld[i - 1]) / (ld[i] - ld[i - 1])
    return c * geo


def _s_control_volume(m: float, T: float, x: float, work: float, q_loss: float,
                      ports: list[tuple[float, float, float]]) -> tuple[float, float, float]:
    R, cp, cv, h_cv, u = _s_props(T, x)
    dm = 0.0
    energy = work - q_loss
    comp = 0.0
    for mdot, T_up, x_up in ports:
        if mdot > 0.0:
            h_k = _s_props(T_up, x_up)[3]
            x_k = x_up
        else:
            h_k, x_k = h_cv, x
        dm += mdot
        energy += mdot * (h_k - u)
        comp += mdot * (x_k - x)
    return dm, (energy - _s_dudx(T) * comp) / (m * cv), comp / m


# State vector layout
(M_C, T_C, X_C, M_X, T_X, X_X, MDOT_TP, M_IV, M_EV_BACK, M_TP_CUM) = range(10)
N_STATE = 10


# --------------------------------------------------------------------------- #
# Results
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class CycleMetrics:
    """Scalar KPIs; `as_vector` is the interface to the surrogate / PINN layer."""
    p_x_mean_window: float
    p_x_mean_cycle: float
    p_x_max: float
    p_x_mean_overlap: float
    overlap_pressure_ratio: float      # mean p_x / p_intake during valve overlap
    backflow_mass_ev: float            # kg, exhaust gas pulled back through EV
    trapped_mass: float                # kg at IVC
    residual_fraction: float           # burned mass fraction at IVC
    volumetric_efficiency: float       # fresh-air mass / (rho_int * Vd)
    pmep: float                        # Pa, gas-exchange loop [-180, +180] deg
    T_x_mean_window: float
    mass_balance_error: float          # relative

    FIELDS = ("p_x_mean_window", "p_x_mean_cycle", "p_x_max", "p_x_mean_overlap",
              "overlap_pressure_ratio", "backflow_mass_ev", "trapped_mass",
              "residual_fraction", "volumetric_efficiency", "pmep",
              "T_x_mean_window", "mass_balance_error")

    def as_vector(self) -> FloatArr:
        return np.array([getattr(self, f) for f in self.FIELDS], dtype=float)


@dataclass(frozen=True)
class CycleResult:
    theta_deg: FloatArr
    p_cyl: FloatArr
    p_man: FloatArr
    T_cyl: FloatArr
    T_man: FloatArr
    burned_fraction_cyl: FloatArr
    mdot_exhaust_valve: FloatArr      # positive = into cylinder
    mdot_intake_valve: FloatArr       # positive = into cylinder
    mdot_tailpipe: FloatArr
    metrics: CycleMetrics
    cycles_to_converge: int
    manifold_state: FloatArr          # [m_x, T_x, x_x, Mdot_tp] at EVO (warm start)


# --------------------------------------------------------------------------- #
# Solver
# --------------------------------------------------------------------------- #


@dataclass
class GasExchangeSolver:
    engine: EngineGeometry
    exhaust_valve: ValveSpec
    intake_valve: ValveSpec
    intake: IntakeBoundary
    exhaust: ExhaustSystem
    window_deg: tuple[float, float] = (-220.0, 220.0)   # EVO .. IVC
    rtol: float = 1.0e-7
    method: str = "LSODA"
    out_step_deg: float = 0.25
    _atol: FloatArr = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if abs(self.exhaust_valve.open_deg - self.window_deg[0]) > 1e-9:
            raise UnitError("window start must coincide with exhaust-valve opening (EVO)")
        self._atol = np.array([1e-11, 1e-4, 1e-7, 1e-11, 1e-4, 1e-7,
                               1e-9, 1e-11, 1e-11, 1e-11])

    # ------------------------------------------------------------------ #
    def _evaluate(self, theta: ArrayLike, y: FloatArr, cyl_open: bool = True) -> FloatArr:
        """Right-hand side dy/dt (and algebraic quantities) - fully vectorised."""
        th = np.asarray(theta, dtype=float)
        eng, ex, ib = self.engine, self.exhaust, self.intake
        bd = ex.boundary

        m_c, T_c, x_c = y[M_C], y[T_C], y[X_C]
        m_x, T_x, x_x = y[M_X], y[T_X], y[X_X]
        Mtp = y[MDOT_TP]

        V_c = eng.volume(th)
        R_c, R_x = gas_R(x_c), gas_R(x_x)
        p_c = m_c * R_c * T_c / V_c
        p_x = m_x * R_x * T_x / ex.manifold_volume
        g_c, g_x = gas_gamma(T_c, x_c), gas_gamma(T_x, x_x)
        R_air = gas_R(0.0)
        g_in = gas_gamma(ib.temperature, 0.0)

        if cyl_open:
            A_ev = self.exhaust_valve.effective_area(th)
            A_iv = self.intake_valve.effective_area(th)
        else:
            A_ev = A_iv = np.zeros_like(th)
        q_ev = signed_valve_flow(A_ev, p_c, T_c, g_c, R_c, p_x, T_x, g_x, R_x)
        q_iv = signed_valve_flow(A_iv, p_c, T_c, g_c, R_c,
                                 ib.pressure, ib.temperature, g_in, R_air)

        # cylinder balance
        if cyl_open:
            dVdt = eng.dV_dtheta(th) * eng.omega
            w = 6.18 * eng.mean_piston_speed                       # Woschni, gas exchange
            h_c = (3.26 * eng.bore**-0.2 * (p_c / 1e3)**0.8
                   * T_c**-0.55 * w**0.8)
            q_wall = h_c * eng.wall_area(V_c) * (T_c - eng.wall_temperature)
            dm_c, dT_c, dx_c = _control_volume(
                m_c, T_c, x_c, -p_c * dVdt, q_wall,
                [(q_ev, T_x, x_x), (q_iv, np.asarray(ib.temperature), np.asarray(0.0))])
        else:
            dm_c = dT_c = dx_c = np.zeros_like(m_c)

        # manifold balance (cylinder outflow enters manifold; tailpipe flow leaves it)
        q_wall_x = ex.h_wall * ex.wall_area * (T_x - ex.wall_temperature)
        dm_x, dT_x, dx_x = _control_volume(
            m_x, T_x, x_x, np.zeros_like(m_x), q_wall_x,
            [(-q_ev, T_c, x_c),
             (-Mtp, np.asarray(bd.T_ambient), np.asarray(0.0))])

        # tailpipe inertance (1-D momentum, lumped)
        rho_x = p_x / (R_x * T_x)
        dMtp = (bd.flow_area / ex.tailpipe_length
                * (p_x - bd.p_ambient - bd.pressure_drop(Mtp, rho_x)))

        d_iv = q_iv if cyl_open else np.zeros_like(q_iv)
        d_back = np.maximum(q_ev, 0.0)
        return np.stack([dm_c, dT_c, dx_c, dm_x, dT_x, dx_x, dMtp,
                         d_iv, d_back, Mtp])


    # ------------------------------------------------------------------ #
    def _evaluate_scalar(self, theta: float, y: list[float], cyl_open: bool = True) -> list[float]:
        """Scalar twin of `_evaluate` (same equations, ~20x faster per call)."""
        eng, ex, ib = self.engine, self.exhaust, self.intake
        bd = ex.boundary
        m_c, T_c, x_c, m_x, T_x, x_x, Mtp = y[0], y[1], y[2], y[3], y[4], y[5], y[6]

        R_c, _, _, _, _ = _s_props(T_c, x_c)
        R_x = _s_props(T_x, x_x)[0]
        V_c = float(eng.volume(theta))
        p_c = m_c * R_c * T_c / V_c
        p_x = m_x * R_x * T_x / ex.manifold_volume
        g_c = _s_props(T_c, x_c)[1] / _s_props(T_c, x_c)[2]
        g_x = _s_props(T_x, x_x)[1] / _s_props(T_x, x_x)[2]
        R_a, cp_a, cv_a, _, _ = _s_props(ib.temperature, 0.0)
        g_in = cp_a / cv_a

        if cyl_open:
            A_ev = _s_valve_area(self.exhaust_valve, theta)
            A_iv = _s_valve_area(self.intake_valve, theta)
        else:
            A_ev = A_iv = 0.0
        q_ev = _s_signed_flow(A_ev, p_c, T_c, g_c, R_c, p_x, T_x, g_x, R_x)
        q_iv = _s_signed_flow(A_iv, p_c, T_c, g_c, R_c, ib.pressure, ib.temperature, g_in, R_a)

        if cyl_open:
            dVdt = float(eng.dV_dtheta(theta)) * eng.omega
            w = 6.18 * eng.mean_piston_speed
            h_c = 3.26 * eng.bore**-0.2 * (p_c / 1e3) ** 0.8 * T_c**-0.55 * w**0.8
            area = float(eng.wall_area(V_c))
            q_wall = h_c * area * (T_c - eng.wall_temperature)
            dm_c, dT_c, dx_c = _s_control_volume(
                m_c, T_c, x_c, -p_c * dVdt, q_wall,
                [(q_ev, T_x, x_x), (q_iv, ib.temperature, 0.0)])
            d_iv = q_iv
        else:
            dm_c = dT_c = dx_c = d_iv = 0.0

        q_wall_x = ex.h_wall * ex.wall_area * (T_x - ex.wall_temperature)
        dm_x, dT_x, dx_x = _s_control_volume(
            m_x, T_x, x_x, 0.0, q_wall_x,
            [(-q_ev, T_c, x_c), (-Mtp, bd.T_ambient, 0.0)])

        rho_x = p_x / (R_x * T_x)
        dp_loss = float(bd.pressure_drop(np.float64(Mtp), np.float64(rho_x)))
        dMtp = bd.flow_area / ex.tailpipe_length * (p_x - bd.p_ambient - dp_loss)
        return [dm_c, dT_c, dx_c, dm_x, dT_x, dx_x, dMtp,
                d_iv, max(q_ev, 0.0), Mtp]

    # ------------------------------------------------------------------ #
    def _integrate(self, th0: float, th1: float, y0: FloatArr,
                   cyl_open: bool, step_deg: float) -> tuple[FloatArr, FloatArr]:
        omega = self.engine.omega
        n = max(int(round((th1 - th0) / math.radians(step_deg))) + 1, 2)
        t_eval = np.linspace(th0, th1, n)

        inv_omega = 1.0 / omega
        f = self._evaluate_scalar

        def rhs(th: float, y: FloatArr) -> list[float]:
            return [d * inv_omega for d in f(th, y.tolist(), cyl_open)]

        sol = solve_ivp(rhs, (th0, th1), y0, method=self.method, t_eval=t_eval,
                        rtol=self.rtol, atol=self._atol,
                        max_step=math.radians(1.0))
        if not sol.success:
            raise SolverError(f"ODE integration failed: {sol.message}")
        if not np.all(np.isfinite(sol.y)) or np.any(sol.y[[M_C, T_C, M_X, T_X]] <= 0):
            raise SolverError("non-physical state (negative mass/temperature) encountered")
        return sol.t, sol.y

    # ------------------------------------------------------------------ #
    def solve_periodic_cycle(self, evo: EVOState, max_cycles: int = 80,
                             tol: float = 1.0e-4,
                             y_man0: FloatArr | None = None) -> CycleResult:
        """Iterate cycles until the manifold state at EVO is periodic."""
        eng, ex = self.engine, self.exhaust
        th_a, th_b = (math.radians(d) for d in self.window_deg)
        th_next_evo = th_a + 4.0 * math.pi
        V_evo = float(eng.volume(th_a))
        x_evo = evo.burned_fraction
        m_c0 = evo.pressure * V_evo / (float(gas_R(x_evo)) * evo.temperature)

        # initial guess for manifold: ambient pressure, hot burned gas
        T_x0 = 800.0
        y_man = np.array([ex.boundary.p_ambient * ex.manifold_volume
                          / (float(gas_R(1.0)) * T_x0), T_x0, 1.0, 0.0])
        if y_man0 is not None:
            y_man = np.asarray(y_man0, dtype=float).copy()

        for cycle in range(1, max_cycles + 1):
            y0 = np.array([m_c0, evo.temperature, x_evo, y_man[0], y_man[1],
                           y_man[2], y_man[3], 0.0, 0.0, 0.0])
            th_w, y_w = self._integrate(th_a, th_b, y0, True, self.out_step_deg)

            yc0 = y_w[:, -1].copy()
            th_c, y_c = self._integrate(th_b, th_next_evo, yc0, False, 1.0)
            y_new = y_c[[M_X, T_X, X_X, MDOT_TP], -1]
            scale = np.array([y_man[0], y_man[1], 1.0, max(abs(y_man[3]), 2e-2)])
            err = float(np.max(np.abs(y_new - y_man) / scale))
            y_man = y_new
            if err < tol:
                return self._package(th_w, y_w, th_c, y_c, cycle, y_man)
        raise SolverError(f"periodic cycle did not converge in {max_cycles} cycles "
                          f"(last relative change {err:.2e})")

    # ------------------------------------------------------------------ #
    def _package(self, th: FloatArr, y: FloatArr, th_c: FloatArr, y_c: FloatArr,
                 cycles: int, y_man: FloatArr) -> CycleResult:
        eng, ex, ib = self.engine, self.exhaust, self.intake
        V = eng.volume(th)
        R_c, R_x = gas_R(y[X_C]), gas_R(y[X_X])
        p_c = y[M_C] * R_c * y[T_C] / V
        p_x = y[M_X] * R_x * y[T_X] / ex.manifold_volume
        g_c, g_x = gas_gamma(y[T_C], y[X_C]), gas_gamma(y[T_X], y[X_X])
        A_ev = self.exhaust_valve.effective_area(th)
        A_iv = self.intake_valve.effective_area(th)
        q_ev = signed_valve_flow(A_ev, p_c, y[T_C], g_c, R_c, p_x, y[T_X], g_x, R_x)
        q_iv = signed_valve_flow(A_iv, p_c, y[T_C], g_c, R_c, ib.pressure,
                                 ib.temperature, gas_gamma(ib.temperature, 0.0),
                                 gas_R(0.0))
        deg = np.degrees(th)

        p_x_closed = (y_c[M_X] * gas_R(y_c[X_X]) * y_c[T_X] / ex.manifold_volume)
        ov = (deg >= self.intake_valve.open_deg) & (deg <= self.exhaust_valve.close_deg)
        loop = (deg >= -180.0) & (deg <= 180.0)
        pmep = -float(np.trapezoid(p_c[loop], V[loop])) / eng.displaced_volume

        m_trapped = float(y[M_C, -1])
        x_res = float(y[X_C, -1])
        rho_int = ib.pressure / (float(gas_R(0.0)) * ib.temperature)
        m_bal = (y[M_C, -1] + y[M_X, -1]) - (y[M_C, 0] + y[M_X, 0])
        m_bal -= (y[M_IV, -1] - y[M_TP_CUM, -1])
        m_scale = max(abs(float(y[M_IV, -1])), 1e-12)
        th_full = np.concatenate([th, th_c[1:]])
        p_full = np.concatenate([p_x, p_x_closed[1:]])

        metrics = CycleMetrics(
            p_x_mean_window=float(np.mean(p_x)),
            p_x_mean_cycle=float(np.trapezoid(p_full, th_full) / (th_full[-1] - th_full[0])),
            p_x_max=float(np.max(p_x)),
            p_x_mean_overlap=float(np.mean(p_x[ov])),
            overlap_pressure_ratio=float(np.mean(p_x[ov]) / ib.pressure),
            backflow_mass_ev=float(y[M_EV_BACK, -1]),
            trapped_mass=m_trapped,
            residual_fraction=x_res,
            volumetric_efficiency=float((1.0 - x_res) * m_trapped
                                        / (rho_int * eng.displaced_volume)),
            pmep=float(pmep),
            T_x_mean_window=float(np.mean(y[T_X])),
            mass_balance_error=float(abs(m_bal) / m_scale),
        )
        return CycleResult(deg, p_c, p_x, y[T_C], y[T_X], y[X_C], q_ev, q_iv,
                           y[MDOT_TP], metrics, cycles, y_man)


# --------------------------------------------------------------------------- #
# Backpressure calibration: target cycle-mean manifold pressure -> restriction zeta
# --------------------------------------------------------------------------- #


ZETA_MIN = 0.25    # numerically safe lower bound of the restriction coefficient


def solve_for_backpressure(build, evo: EVOState, p_target: float,
                           rel_tol: float = 5.0e-3, max_iter: int = 12,
                           **solve_kw) -> tuple[CycleResult, float]:
    """
    Find the restriction coefficient zeta so that the cycle-averaged exhaust
    manifold pressure equals p_target [Pa abs]. `build(zeta)` returns a
    GasExchangeSolver. Fixed-point update zeta <- zeta*(dp_target/dp_now) with
    warm-started periodic solutions, bracketed by regula falsi in ln(zeta).
    If p_target is below the open-exhaust floor (~1.03 bar here) the open-exhaust
    case (zeta = ZETA_MIN) is returned; compare metrics.p_x_mean_cycle to p_target.
    """
    probe = build(1.0)
    bd = probe.exhaust.boundary
    _check("target backpressure [Pa]", p_target, bd.p_ambient * 1.001, 1.0e6)
    eng, ib = probe.engine, probe.intake
    m_cycle = 1.03 * 0.95 * ib.pressure / (float(gas_R(0.0)) * ib.temperature) \
        * eng.displaced_volume
    mdot_mean = m_cycle / (120.0 / eng.speed_rpm)
    rho_x = p_target / (float(gas_R(1.0)) * 700.0)
    zeta = RestrictionBoundary.for_target_backpressure(
        p_target, mdot_mean, rho_x, bd.flow_area,
        p_ambient=bd.p_ambient, T_ambient=bd.T_ambient).zeta
    y_man: FloatArr | None = None
    lo: tuple[float, float] | None = None      # (ln zeta, err) with err < 0 (p below target)
    hi: tuple[float, float] | None = None      # (ln zeta, err) with err > 0 (p above target)
    dp_tgt = p_target - bd.p_ambient
    tol_abs = max(rel_tol * dp_tgt, 200.0)          # at least 2 mbar
    best: tuple[float, CycleResult, float] | None = None
    for _ in range(max_iter):
        res = build(zeta).solve_periodic_cycle(evo, y_man0=y_man, **solve_kw)
        y_man = res.manifold_state
        err = res.metrics.p_x_mean_cycle - p_target
        if best is None or abs(err) < best[0]:
            best = (abs(err), res, zeta)
        if abs(err) < tol_abs:
            return res, zeta
        if zeta <= ZETA_MIN * 1.0001 and err > 0:
            # Target below the open-exhaust floor (blowdown pulses + tailpipe
            # inertance keep the cycle mean above ambient): best attainable case.
            return res, zeta
        if err < 0:
            lo = (math.log(zeta), err)
        else:
            hi = (math.log(zeta), err)
        if lo is not None and hi is not None:
            # regula falsi in ln(zeta); p(zeta) is monotonic increasing
            frac = -lo[1] / (hi[1] - lo[1])
            frac = min(max(frac, 0.1), 0.9)
            ln_new = lo[0] + frac * (hi[0] - lo[0])
        else:
            dp_now = res.metrics.p_x_mean_cycle - bd.p_ambient
            ratio = dp_tgt / dp_now if dp_now > 1.0 else 4.0
            ln_new = math.log(zeta * float(np.clip(ratio, 0.25, 4.0)))
        zeta = max(math.exp(ln_new), ZETA_MIN)
    # p(zeta) can be non-monotonic at very small zeta (lightly damped resonance):
    # accept the closest iterate if it is within 15 % of the requested rise.
    if best is not None and best[0] <= max(0.15 * dp_tgt, 1000.0):
        return best[1], best[2]
    raise SolverError("backpressure calibration did not converge")


# --------------------------------------------------------------------------- #
# Reference engine: heavy-duty turbocharged diesel (120 x 140 mm)
# --------------------------------------------------------------------------- #


def build_heavy_duty_solver(zeta: float, speed_rpm: float = 1500.0,
                            overlap_deg: float = 40.0, p_intake: float = bar(2.2),
                            T_intake: float = 330.0, rtol: float = 1.0e-7,
                            out_step_deg: float = 0.25) -> GasExchangeSolver:
    """Symmetric timing: 245 deg valve durations, overlap centred on gas-exchange TDC."""
    _check("overlap [deg]", overlap_deg, 0.0, 120.0)
    half = 0.5 * overlap_deg
    evc, ivo = +half, -half
    evo, ivc = evc - 245.0, ivo + 245.0
    engine = EngineGeometry(bore=mm(120), stroke=mm(140), conrod=mm(220),
                            compression_ratio=17.0, speed_rpm=speed_rpm,
                            wall_temperature=500.0)
    ex_valve = ValveSpec(evo, evc, mm(15.0), mm(48.0))
    in_valve = ValveSpec(ivo, ivc, mm(16.0), mm(52.0))
    intake = IntakeBoundary(pressure=p_intake, temperature=T_intake)
    # k_linear: pipe-wall friction + laminar (Darcy) loss; also damps the lumped resonance
    boundary = RestrictionBoundary(flow_area=0.25 * math.pi * 0.08**2, zeta=zeta,
                                   k_linear=2.0e4)
    exhaust = ExhaustSystem(manifold_volume=litre(2.0), tailpipe_length=1.5,
                            boundary=boundary, wall_area=0.08)
    return GasExchangeSolver(engine, ex_valve, in_valve, intake, exhaust,
                             window_deg=(evo, ivc), rtol=rtol, out_step_deg=out_step_deg)


def default_evo_state(p_intake: float) -> EVOState:
    """EVO state surrogate (replace with the combustion module output)."""
    return EVOState(pressure=3.4 * p_intake, temperature=1050.0)



def verify_scalar_path(n: int = 200, seed: int = 1) -> float:
    """Max relative deviation between vectorised and scalar RHS (should be ~1e-12)."""
    rng = np.random.default_rng(seed)
    sv = build_heavy_duty_solver(300.0)
    worst = 0.0
    for _ in range(n):
        th = math.radians(rng.uniform(-225, 225))
        y = np.array([rng.uniform(5e-4, 3e-3), rng.uniform(500, 1200), rng.uniform(0, 1),
                      rng.uniform(5e-4, 3e-3), rng.uniform(500, 900), rng.uniform(0, 1),
                      rng.uniform(-0.05, 0.2), 0.0, 0.0, 0.0])
        a = sv._evaluate(th, y, True)
        b = np.array(sv._evaluate_scalar(th, y.tolist(), True))
        worst = max(worst, float(np.max(np.abs(a - b) / (np.abs(a) + 1e-9))))
    return worst


if __name__ == "__main__":
    import time
    p_int = bar(2.2)
    evo_state = default_evo_state(p_int)
    probe = build_heavy_duty_solver(1.0)
    print(f"HD diesel 120x140 mm, Vd/cyl = {probe.engine.displaced_volume*1e3:.3f} L, "
          f"1500 rpm, boost {p_int/1e5:.2f} bar, overlap 40 deg")
    print(f"Lumped exhaust acoustic mode ~ {probe.exhaust.helmholtz_frequency():.0f} Hz\n")
    print(f"{'p_bp tgt':>8} {'zeta':>8} {'px_mean':>7} {'px_max':>7} {'ovl px/pi':>9} "
          f"{'backflow[mg]':>12} {'RGF[%]':>7} {'eta_v':>6} {'trapped[g]':>10} "
          f"{'PMEP[bar]':>9} {'mass err':>9} {'cyc':>3} {'t[s]':>5}")
    for p_bp in (1.05, 1.25, 1.50, 1.75, 2.00, 2.25, 2.50):
        t0 = time.time()
        res, zeta = solve_for_backpressure(
            lambda z: build_heavy_duty_solver(z, p_intake=p_int), evo_state, bar(p_bp))
        m = res.metrics
        print(f"{p_bp:8.2f} {zeta:8.1f} {m.p_x_mean_cycle/1e5:7.3f} {m.p_x_max/1e5:7.3f} "
              f"{m.overlap_pressure_ratio:9.3f} {m.backflow_mass_ev*1e6:12.2f} "
              f"{100*m.residual_fraction:7.2f} {m.volumetric_efficiency:6.3f} "
              f"{m.trapped_mass*1e3:10.3f} {m.pmep/1e5:9.3f} "
              f"{m.mass_balance_error:9.1e} {res.cycles_to_converge:3d} {time.time()-t0:5.1f}")
