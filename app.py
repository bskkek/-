"""
app.py - Exhaust Backpressure Gas-Exchange Cockpit  (RU / EN web application)
=============================================================================
Local run :  pip install -r requirements.txt
             streamlit run app.py
Docker    :  docker build -t backpressure-cockpit . && docker run -p 8501:8501 backpressure-cockpit

All physics lives in exhaust_backpressure_solver.py; this file only presents it.
Every slider move runs the real solver (periodic limit cycle + restriction
calibration so the cycle-mean manifold pressure equals the requested target).
Results are cached per operating point.
"""
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from exhaust_backpressure_solver import (
    SolverError, UnitError, bar, build_heavy_duty_solver, default_evo_state,
    solve_for_backpressure,
)

REF_BP_BAR = 1.05      # open-exhaust reference for deltas / overlay
MIN_BP_BAR = 1.02      # lowest target accepted by the calibration (ambient 1.013)
P_AMB_BAR = 1.013

# --------------------------------------------------------------------------- #
# Internationalisation
# --------------------------------------------------------------------------- #
TR: dict[str, dict[str, str]] = {
    "en": {
        "lang_label": "Language / Язык",
        "tag": "1-D thermodynamic CAE · gas-exchange module",
        "title": "Exhaust Backpressure Cockpit",
        "sub": ("Heavy-duty turbo-diesel, 120 × 140 mm, single-cylinder-equivalent "
                "cylinder ⇄ manifold ⇄ restriction model: variable-γ gas, choked/subsonic "
                "valve flow, lumped tailpipe acoustics."),
        "op_point": "Operating point",
        "bp": "Exhaust backpressure target  [bar abs, cycle mean]",
        "rpm": "Engine speed  [rpm]",
        "ovl": "Valve overlap duration  [° CA]",
        "bc": "Boundary conditions",
        "boost": "Intake boost pressure  [bar abs]",
        "tint": "Intake temperature  [K]",
        "overlay": "Overlay {ref:.2f} bar reference",
        "disclaimer": ("Single-cylinder-equivalent gas-exchange model. The EVO state is a "
                       "surrogate (3.4 × boost, 1050 K) pending coupling to the combustion "
                       "module. Research prototype, not a calibrated product."),
        "spinner": "Solving the periodic gas-exchange cycle and calibrating the restriction…",
        "err": "The solver could not produce a valid cycle for this operating point: {e}",
        "floor": ("Requested {bp:.2f} bar cannot be matched exactly: near ambient the mean "
                  "manifold pressure is bounded by blowdown pulses and tailpipe inertance. "
                  "Showing the closest case: mean manifold pressure {ach:.3f} bar."),
        "k_mass": "Trapped mass", "k_rgf": "Residual gas fraction",
        "k_etav": "Volumetric efficiency", "k_pmep": "PMEP (pumping loss)",
        "vs_ref": "vs ref.",
        "k_pmax": "Mean / peak manifold p", "k_back": "Exhaust backflow (iEGR path)",
        "k_ratio": "Overlap p_exh / p_int", "k_misc": "Restriction ζ · cycles · f_ac",
        "x_ca": "Crank angle  [° from gas-exchange TDC]",
        "p_title": "Cylinder & manifold pressure vs crank angle",
        "p_unit": "Pressure  [bar abs]",
        "p_cyl": "In-cylinder", "p_man": "Exhaust manifold",
        "p_ref": "Manifold @ {ref:.2f} bar ref.", "intake": "intake",
        "overlap": "valve overlap",
        "f_title": "Valve mass flow  (+ = into cylinder)", "f_unit": "Mass flow  [g/s]",
        "f_iv": "Intake valve", "f_ev": "Exhaust valve", "f_tp": "Tailpipe (outflow +)",
        "f_back": "Backflow / internal EGR",
        "pv_title": "Gas-exchange P-V loop   ·   PMEP = {v:+.2f} bar ({kind})",
        "loss": "loss", "gain": "gain",
        "pv_area": "loop area = pumping work", "pv_ex": "Exhaust stroke",
        "pv_in": "Intake stroke", "pv_x": "Cylinder volume  [L]",
        "pv_y": "Cylinder pressure  [bar abs]",
        "sweep_exp": "Backpressure sweep at this speed / overlap / boost (≈20-30 s first run)",
        "sweep_btn": "Run sweep 1.05 → 3.00 bar",
        "sw_x": "Mean exhaust backpressure  [bar abs]",
        "sw_pmep": "PMEP  [bar]", "sw_rgf": "Residual gas fraction  [%]",
        "sw_etav": "Volumetric efficiency  [-]",
        "about_exp": "Model equations",
        "about": (
            "**Control volumes** (cylinder, manifold): mass, burned-gas fraction and energy "
            "balances for an open system with variable cp(T, x) and γ(T, x).  \n"
            "**Valves:** compressible isentropic nozzle with Cd(l/D); choked and subsonic "
            "branches, reverse flow supported.  \n"
            "**Downstream:** lumped inertance (L/A)·dṀ/dt = p_x − p_amb − Δp(Ṁ) with "
            "Darcy-Forchheimer restriction Δp = k·Ṁ + ζ·Ṁ|Ṁ|/(2ρA²).  \n"
            "**Heat transfer:** Woschni (gas-exchange constants).  \n"
            "**Solution:** `scipy.integrate.solve_ivp` (LSODA), periodic limit cycle over "
            "720° CA; mass-balance residual ~1e-15."),
    },
    "ru": {
        "lang_label": "Language / Язык",
        "tag": "1-D термодинамический CAE · модуль газообмена",
        "title": "Панель анализа противодавления выпуска",
        "sub": ("Тяжёлый турбодизель 120 × 140 мм, одноцилиндровый эквивалент: "
                "цилиндр ⇄ коллектор ⇄ сопротивление; газ с переменным γ, "
                "докритическое и критическое истечение через клапаны, "
                "сосредоточенная акустика выпускной трубы."),
        "op_point": "Режим работы",
        "bp": "Целевое противодавление на выпуске  [бар абс., среднее за цикл]",
        "rpm": "Частота вращения  [об/мин]",
        "ovl": "Продолжительность перекрытия клапанов  [° ПКВ]",
        "bc": "Граничные условия",
        "boost": "Давление наддува  [бар абс.]",
        "tint": "Температура на впуске  [К]",
        "overlay": "Показать эталон {ref:.2f} бар",
        "disclaimer": ("Одноцилиндровая эквивалентная модель газообмена. Состояние в момент "
                       "открытия выпускного клапана (ОВК) задано суррогатом (3,4 × наддув, "
                       "1050 К) до связи с модулем сгорания. Исследовательский прототип, "
                       "не откалиброванный продукт."),
        "spinner": "Решается периодический цикл газообмена и калибруется сопротивление…",
        "err": "Решатель не смог получить корректный цикл для этого режима: {e}",
        "floor": ("Запрошенные {bp:.2f} бар не удаётся получить точно: вблизи атмосферного давления "
                  "среднее давление в коллекторе ограничено импульсами выпуска и инерцией "
                  "выпускной трубы. Показан ближайший случай: среднее давление в "
                  "коллекторе {ach:.3f} бар."),
        "k_mass": "Захваченная масса", "k_rgf": "Доля остаточных газов",
        "k_etav": "Коэффициент наполнения", "k_pmep": "PMEP (насосные потери)",
        "vs_ref": "к эталону",
        "k_pmax": "Давление в коллекторе: среднее / пик",
        "k_back": "Заброс выпускных газов (внутр. рециркуляция)",
        "k_ratio": "Перекрытие: p_вып / p_впуск",
        "k_misc": "Сопротивление ζ · циклов · f_ак",
        "x_ca": "Угол ПКВ  [° от ВМТ газообмена]",
        "p_title": "Давление в цилиндре и выпускном коллекторе по углу ПКВ",
        "p_unit": "Давление  [бар абс.]",
        "p_cyl": "В цилиндре", "p_man": "Выпускной коллектор",
        "p_ref": "Коллектор, эталон {ref:.2f} бар", "intake": "впуск",
        "overlap": "перекрытие клапанов",
        "f_title": "Массовый расход через клапаны  (+ = в цилиндр)",
        "f_unit": "Массовый расход  [г/с]",
        "f_iv": "Впускной клапан", "f_ev": "Выпускной клапан",
        "f_tp": "Выпускная труба (выход +)",
        "f_back": "Заброс / внутренняя рециркуляция",
        "pv_title": "Петля газообмена P-V   ·   PMEP = {v:+.2f} бар ({kind})",
        "loss": "потери", "gain": "выигрыш",
        "pv_area": "площадь петли = насосная работа", "pv_ex": "Такт выпуска",
        "pv_in": "Такт впуска", "pv_x": "Объём цилиндра  [л]",
        "pv_y": "Давление в цилиндре  [бар абс.]",
        "sweep_exp": "Развёртка по противодавлению при этих оборотах / перекрытии / наддуве (≈20-30 с при первом запуске)",
        "sweep_btn": "Запустить развёртку 1,05 → 3,00 бар",
        "sw_x": "Среднее противодавление на выпуске  [бар абс.]",
        "sw_pmep": "PMEP  [бар]", "sw_rgf": "Доля остаточных газов  [%]",
        "sw_etav": "Коэффициент наполнения  [-]",
        "about_exp": "Уравнения модели",
        "about": (
            "**Контрольные объёмы** (цилиндр, коллектор): балансы массы, доли burned-газа и "
            "энергии для открытой системы с переменными cp(T, x) и γ(T, x).  \n"
            "**Клапаны:** сжимаемое изоэнтропное сопло с Cd(l/D); докритическая и "
            "критическая ветви, обратный поток учитывается.  \n"
            "**Выпускная система:** сосредоточенная инертность "
            "(L/A)·dṀ/dt = p_x − p_атм − Δp(Ṁ) с сопротивлением Дарси-Форхгеймера "
            "Δp = k·Ṁ + ζ·Ṁ|Ṁ|/(2ρA²).  \n"
            "**Теплообмен:** Вошни (константы для газообмена).  \n"
            "**Решение:** `scipy.integrate.solve_ivp` (LSODA), периодический предельный цикл "
            "за 720° ПКВ; невязка баланса массы ~1e-15."),
    },
}

C = dict(bg="#0a0e17", panel="#101826", grid="#1d2a3f", text="#d6e2f3", mute="#7d8ea8",
         cyan="#22d3ee", blue="#4f8cff", amber="#fbbf24", red="#f43f5e", green="#34d399")

st.set_page_config(page_title="Backpressure Cockpit", page_icon="⚙️", layout="wide",
                   initial_sidebar_state="expanded")

# language selector first (sidebar), so the rest of the page can use t()
with st.sidebar:
    lang_name = st.radio(TR["en"]["lang_label"], ["Русский", "English"], index=0, horizontal=True)
LANG = "ru" if lang_name == "Русский" else "en"


def t(key: str, **kw: object) -> str:
    s = TR[LANG][key]
    return s.format(**kw) if kw else s


st.markdown(f"""
<style>
  .stApp {{ background: radial-gradient(1200px 600px at 10% -10%, #14213a 0%, {C['bg']} 55%); }}
  section[data-testid="stSidebar"] {{ background:#0c1220; border-right:1px solid {C['grid']}; }}
  .hdr .tag {{ font-size:.72rem; letter-spacing:.18em; color:{C['cyan']}; text-transform:uppercase; }}
  .hdr h1 {{ font-size:1.55rem; margin:0; font-weight:650; }}
  .sub {{ color:{C['mute']}; font-size:.85rem; margin:.2rem 0 .8rem 0; }}
  div[data-testid="stMetric"] {{
      background: linear-gradient(160deg,#142036 0%,#0e1626 100%);
      border:1px solid {C['grid']}; border-left:3px solid {C['cyan']};
      border-radius:10px; padding:14px 16px; }}
  div[data-testid="stMetric"] label {{ color:{C['mute']} !important; font-size:.76rem;
      text-transform:uppercase; letter-spacing:.06em; }}
  div[data-testid="stMetricValue"] {{ font-size:1.75rem; font-weight:650; }}
  .note {{ color:{C['mute']}; font-size:.78rem; }}
</style>
<div class="hdr"><div class="tag">{t('tag')}</div><h1>{t('title')}</h1></div>
<div class="sub">{t('sub')}</div>
""", unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
# Simulation (cached, language independent)
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner=False, max_entries=96)
def simulate(bp_bar: float, rpm: float, overlap_deg: float,
             boost_bar: float, t_int: float) -> dict:
    p_int = bar(boost_bar)

    def build(zeta: float):
        return build_heavy_duty_solver(zeta, speed_rpm=rpm, overlap_deg=overlap_deg,
                                       p_intake=p_int, T_intake=t_int,
                                       rtol=1.0e-6, out_step_deg=0.5)

    target = bar(max(bp_bar, MIN_BP_BAR))
    res, zeta = solve_for_backpressure(build, default_evo_state(p_int), target, rel_tol=0.01)
    sv = build(zeta)
    th = res.theta_deg
    return dict(
        theta=th, V=sv.engine.volume(np.radians(th)),
        p_cyl=res.p_cyl, p_man=res.p_man, q_ev=res.mdot_exhaust_valve,
        q_iv=res.mdot_intake_valve, q_tp=res.mdot_tailpipe,
        zeta=zeta, cycles=res.cycles_to_converge, p_int=p_int,
        f_ac=sv.exhaust.helmholtz_frequency(), target=target,
        metrics={f: getattr(res.metrics, f) for f in res.metrics.FIELDS},
    )


# --------------------------------------------------------------------------- #
# Plot helpers
# --------------------------------------------------------------------------- #
def style(fig: go.Figure, title: str, xt: str, yt: str, height: int = 380) -> go.Figure:
    fig.update_layout(
        template="plotly_dark", height=height, title=dict(text=title, font=dict(size=14)),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#0c1422",
        font=dict(family="Inter, Segoe UI, sans-serif", color=C["text"], size=12),
        margin=dict(l=55, r=20, t=55, b=45), hovermode="x unified",
        legend=dict(orientation="h", y=1.15, x=0, bgcolor="rgba(0,0,0,0)"))
    fig.update_xaxes(title_text=xt, gridcolor=C["grid"], zeroline=False)
    fig.update_yaxes(title_text=yt, gridcolor=C["grid"], zeroline=False)
    return fig


def crank_axis(fig: go.Figure, overlap_deg: float) -> None:
    fig.update_xaxes(range=[-180, 180], tickmode="linear", tick0=-180, dtick=60)
    if overlap_deg > 0:
        fig.add_vrect(x0=-overlap_deg / 2, x1=overlap_deg / 2, fillcolor=C["amber"],
                      opacity=0.10, line_width=0, annotation_text=t("overlap"),
                      annotation_position="top left",
                      annotation_font=dict(color=C["amber"], size=11))
    fig.add_vline(x=0, line=dict(color=C["mute"], width=1, dash="dot"))


def window(d: dict, key: str) -> tuple[np.ndarray, np.ndarray]:
    m = (d["theta"] >= -180.0) & (d["theta"] <= 180.0)
    return d["theta"][m], d[key][m]


def fig_pressure(d: dict, ref: dict | None, ov: float) -> go.Figure:
    f = go.Figure()
    th, pc = window(d, "p_cyl")
    f.add_trace(go.Scatter(x=th, y=pc / 1e5, name=t("p_cyl"), line=dict(color=C["cyan"], width=2.4)))
    th, pm = window(d, "p_man")
    f.add_trace(go.Scatter(x=th, y=pm / 1e5, name=t("p_man"), line=dict(color=C["red"], width=2.4)))
    if ref is not None:
        th, pr = window(ref, "p_man")
        f.add_trace(go.Scatter(x=th, y=pr / 1e5, name=t("p_ref", ref=REF_BP_BAR),
                               line=dict(color=C["mute"], width=1.4, dash="dash")))
    f.add_hline(y=d["p_int"] / 1e5, line=dict(color=C["green"], width=1, dash="dot"),
                annotation_text=t("intake"), annotation_font=dict(color=C["green"], size=11))
    crank_axis(f, ov)
    return style(f, t("p_title"), t("x_ca"), t("p_unit"))


def fig_flows(d: dict, ov: float) -> go.Figure:
    f = go.Figure()
    th, qe = window(d, "q_ev")
    _, qi = window(d, "q_iv")
    _, qt = window(d, "q_tp")
    f.add_trace(go.Scatter(x=th, y=qi * 1e3, name=t("f_iv"), line=dict(color=C["green"], width=2.2)))
    f.add_trace(go.Scatter(x=th, y=qe * 1e3, name=t("f_ev"), line=dict(color=C["blue"], width=2.2)))
    f.add_trace(go.Scatter(x=th, y=qt * 1e3, name=t("f_tp"),
                           line=dict(color=C["mute"], width=1.2, dash="dot")))
    f.add_trace(go.Scatter(x=th, y=np.where(qe > 0.0, qe * 1e3, 0.0), name=t("f_back"),
                           line=dict(color=C["red"], width=0.5), fill="tozeroy",
                           fillcolor="rgba(244,63,94,0.55)"))
    crank_axis(f, ov)
    f.add_hline(y=0, line=dict(color=C["mute"], width=1))
    return style(f, t("f_title"), t("x_ca"), t("f_unit"))


def fig_pv(d: dict) -> go.Figure:
    f = go.Figure()
    m = (d["theta"] >= -180) & (d["theta"] <= 180)
    V, p, th = d["V"][m] * 1e3, d["p_cyl"][m] / 1e5, d["theta"][m]
    pmep = d["metrics"]["pmep"] / 1e5
    loss = pmep > 0
    f.add_trace(go.Scatter(x=V, y=p, mode="lines", name=t("pv_area"), line=dict(width=0),
                           fill="toself", hoverinfo="skip",
                           fillcolor="rgba(244,63,94,0.28)" if loss else "rgba(52,211,153,0.28)"))
    ex, it = th < 0, th >= 0
    f.add_trace(go.Scatter(x=V[ex], y=p[ex], name=t("pv_ex"), line=dict(color=C["red"], width=2.4)))
    f.add_trace(go.Scatter(x=V[it], y=p[it], name=t("pv_in"), line=dict(color=C["green"], width=2.4)))
    f.add_hline(y=d["p_int"] / 1e5, line=dict(color=C["green"], width=1, dash="dot"))
    f.add_hline(y=d["metrics"]["p_x_mean_window"] / 1e5, line=dict(color=C["red"], width=1, dash="dot"))
    f.update_layout(hovermode="closest")
    return style(f, t("pv_title", v=pmep, kind=t("loss" if loss else "gain")),
                 t("pv_x"), t("pv_y"))


def show(fig: go.Figure) -> None:
    st.plotly_chart(fig, use_container_width=True)


# --------------------------------------------------------------------------- #
# Sidebar controls
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.markdown(f"### {t('op_point')}")
    bp = st.slider(t("bp"), 1.0, 3.0, 2.0, 0.05)
    rpm = st.slider(t("rpm"), 800, 2200, 1500, 50)
    ovl = st.slider(t("ovl"), 0, 100, 40, 5)
    with st.expander(t("bc")):
        boost = st.slider(t("boost"), 1.5, 3.5, 2.2, 0.1)
        t_int = st.slider(t("tint"), 300, 360, 330, 5)
    overlay = st.checkbox(t("overlay", ref=REF_BP_BAR), value=True)
    st.markdown("---")
    st.markdown(f"<div class='note'>{t('disclaimer')}</div>", unsafe_allow_html=True)

# --------------------------------------------------------------------------- #
# Run
# --------------------------------------------------------------------------- #
try:
    with st.spinner(t("spinner")):
        cur = simulate(float(bp), float(rpm), float(ovl), float(boost), float(t_int))
        ref = simulate(REF_BP_BAR, float(rpm), float(ovl), float(boost), float(t_int))
except (SolverError, UnitError) as exc:
    st.error(t("err", e=exc))
    st.stop()

m, r = cur["metrics"], ref["metrics"]
achieved = m["p_x_mean_cycle"] / 1e5
target_bar = cur["target"] / 1e5
if abs(achieved - target_bar) > max(0.01, 0.02 * (target_bar - P_AMB_BAR)):
    st.info(t("floor", bp=bp, ach=achieved))

# --------------------------------------------------------------------------- #
# KPI cards
# --------------------------------------------------------------------------- #
vs = t("vs_ref")
k1, k2, k3, k4 = st.columns(4)
k1.metric(t("k_mass"), f"{m['trapped_mass'] * 1e3:.3f} g",
          f"{(m['trapped_mass'] - r['trapped_mass']) * 1e3:+.3f} g {vs}")
k2.metric(t("k_rgf"), f"{100 * m['residual_fraction']:.2f} %",
          f"{100 * (m['residual_fraction'] - r['residual_fraction']):+.2f} pp {vs}",
          delta_color="inverse")
k3.metric(t("k_etav"), f"{m['volumetric_efficiency']:.3f}",
          f"{m['volumetric_efficiency'] - r['volumetric_efficiency']:+.3f} {vs}")
k4.metric(t("k_pmep"), f"{m['pmep'] / 1e5:+.2f} bar",
          f"{(m['pmep'] - r['pmep']) / 1e5:+.2f} bar {vs}", delta_color="inverse")

s1, s2, s3, s4 = st.columns(4)
s1.metric(t("k_pmax"), f"{achieved:.2f} / {m['p_x_max'] / 1e5:.2f} bar")
s2.metric(t("k_back"), f"{m['backflow_mass_ev'] * 1e6:.1f} mg")
s3.metric(t("k_ratio"), f"{m['overlap_pressure_ratio']:.2f}")
s4.metric(t("k_misc"), f"{cur['zeta']:.0f} · {cur['cycles']} · {cur['f_ac']:.0f} Hz")

st.write("")

# --------------------------------------------------------------------------- #
# Plots
# --------------------------------------------------------------------------- #
show(fig_pressure(cur, ref if overlay else None, ovl))
c1, c2 = st.columns(2)
with c1:
    show(fig_flows(cur, ovl))
with c2:
    show(fig_pv(cur))

# --------------------------------------------------------------------------- #
# Sweep + model description
# --------------------------------------------------------------------------- #
with st.expander(t("sweep_exp")):
    if st.button(t("sweep_btn")):
        pts = np.linspace(1.05, 3.0, 8)
        out, prog = [], st.progress(0.0)
        for i, b in enumerate(pts):
            out.append(simulate(float(round(b, 3)), float(rpm), float(ovl),
                                float(boost), float(t_int)))
            prog.progress((i + 1) / len(pts))
        st.session_state["sweep"] = (pts, out)
    if "sweep" in st.session_state:
        pts, out = st.session_state["sweep"]
        series = [("sw_pmep", [o["metrics"]["pmep"] / 1e5 for o in out], C["red"]),
                  ("sw_rgf", [100 * o["metrics"]["residual_fraction"] for o in out], C["amber"]),
                  ("sw_etav", [o["metrics"]["volumetric_efficiency"] for o in out], C["cyan"])]
        for col, (key, ys, color) in zip(st.columns(3), series):
            f = go.Figure(go.Scatter(x=pts, y=ys, mode="lines+markers",
                                     line=dict(color=color, width=2.4)))
            style(f, t(key), t("sw_x"), t(key), 300)
            f.update_layout(hovermode="closest")
            with col:
                show(f)

with st.expander(t("about_exp")):
    st.markdown(t("about"))
