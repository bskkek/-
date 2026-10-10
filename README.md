---
title: Exhaust Backpressure Cockpit
emoji: ⚙️
colorFrom: blue
colorTo: gray
sdk: docker
app_port: 8501
pinned: false
---

# Exhaust Backpressure Cockpit / Панель анализа противодавления выпуска

**EN** — Web app (Russian / English UI) built on a physical 1-D gas-exchange solver:
cylinder ⇄ exhaust manifold ⇄ restriction, variable-γ gas, choked/subsonic valve flow,
lumped tailpipe acoustics, periodic limit-cycle solution.

**RU** — Веб-приложение (интерфейс на русском и английском) на основе физического 1-D решателя
газообмена: цилиндр ⇄ выпускной коллектор ⇄ сопротивление, газ с переменным γ,
критическое и докритическое истечение, сосредоточенная акустика, периодический предельный цикл.

## Run / Запуск
```bash
pip install -r requirements.txt
streamlit run app.py                     # http://localhost:8501
python exhaust_backpressure_solver.py    # console sweep 1.05 -> 2.50 bar
```
## Docker
```bash
docker build -t backpressure-cockpit .
docker run -p 8501:8501 backpressure-cockpit
```
## Deploy / Публикация
Streamlit Community Cloud, Hugging Face Spaces (SDK: Docker, port 8501), Render, Fly.io
or any VPS with Docker. Push `app.py`, `exhaust_backpressure_solver.py`, `requirements.txt`,
`.streamlit/config.toml` (and `Dockerfile`).

## Files
- `exhaust_backpressure_solver.py` — physics core (no UI dependencies)
- `app.py` — bilingual Streamlit UI

Status: research prototype. Single-cylinder-equivalent model; EVO state is a surrogate
until coupled to a combustion module; Cd(l/D) table and heat-transfer constants need calibration.
