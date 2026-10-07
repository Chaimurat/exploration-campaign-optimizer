# exploration-campaign-optimizer

Multi-objective scheduling of offshore exploration drilling campaigns using exact
optimisation (CP-SAT) and a genetic algorithm (GA), with an optional natural
language interface powered by Claude (Anthropic API).

## Problem

A company holds exploration licences in **block 34 of the Norwegian North Sea**
(Snorre/34/7 and Statfjord/34/10 areas). Given a portfolio of 20 exploration
targets, 3 semi-submersible rigs, and real drilling durations and DST oil rates
from SODIR, the scheduler finds the optimal assignment of wells to rigs and
drilling sequence.

Two objectives compete:

| Objective | Meaning | Want |
|---|---|---|
| **Makespan** | Total campaign duration (days) | Minimise (lower rig hire cost) |
| **Deferred discovery value** | sum(start_day × DST_rate) × oil_price (USD) | Minimise (drill high-rate wells early) |

These conflict because drilling high-value wells early often requires cross-cluster
rig moves (34/7 ↔ 34/10, ~26 km apart, ~5 days mobilisation), extending the campaign.

## Methods

### Exact solver — CP-SAT (epsilon-constraint)
Google OR-Tools CP-SAT with `AddCircuit` for sequence-dependent mobilisation times.
An epsilon-constraint sweep fixes the makespan limit and minimises deferred value,
tracing the Pareto front.

### Heuristic — Genetic Algorithm (weight sweep)
Custom GA with greedy decoder and order-crossover (OX). Nine weight combinations
`(w_makespan, w_deferred)` from (0.9, 0.1) to (0.1, 0.9) explore the Pareto front.

### LLM layer — Natural language interface (added)
A Claude Haiku call translates plain-English scheduling requests into a validated
`ConstraintSpec` (Pydantic), which is compiled into a modified instance before
the solver runs. The LLM never touches the optimisation — it only reformulates
the problem.

## Data

All data is publicly available from the **Norwegian Offshore Directorate (SODIR)**:
[factpages.sodir.no](https://factpages.sodir.no)

Download and place in `data/`:
- `wellbore_exploration_all.csv` — well names, UTM coordinates, drilling dates
- `wellbore_dst.csv` — drill stem test oil flow rates

`data/snorre_instance.json` (pre-parsed, included in repo) can be used directly
without downloading the raw CSVs.

## Installation

```bash
git clone https://github.com/YOUR_USERNAME/exploration-campaign-optimizer
cd exploration-campaign-optimizer
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS / Linux
pip install -r requirements.txt
```

For the LLM interactive mode, also install:
```bash
pip install anthropic
```

## Usage

### Standard run (CP-SAT + GA Pareto sweep)
```bash
python main.py
```
Outputs:
- `images/pareto_block34.png` — Pareto front comparison
- `images/gantt_block34.html` — interactive Gantt chart
- `output/results.csv` — numerical results

### Interactive LLM mode
```bash
python main.py --interactive --api-key sk-ant-YOUR_KEY
```
Type plain-English requests such as:
- *"Only drill Snorre wells, use 2 rigs"*
- *"Focus on wells above 1500 bbl/day, finish as fast as possible"*
- *"Lock Rig-Alpha to the Statfjord area"*

The LLM reformulates the instance; the solver runs unchanged.

## Project structure

```
exploration-campaign-optimizer/
├── main.py                    # Entry point (standard + interactive modes)
├── requirements.txt
├── data/
│   ├── snorre_parser.py       # Parses SODIR CSVs → snorre_instance.json
│   └── snorre_instance.json   # Pre-parsed instance (20 wells, 3 rigs)
└── src/
    ├── well_solver.py         # CP-SAT exact solver (epsilon-constraint)
    ├── well_ga.py             # Genetic algorithm (weight sweep)
    ├── well_visualizer.py     # Pareto front plots
    ├── well_gantt.py          # Interactive Gantt chart (Plotly)
    ├── constraint_spec.py     # Pydantic schema for LLM output
    ├── llm_formulator.py      # LLM backends (Anthropic / OpenAI)
    └── instance_builder.py    # Deterministic spec → instance compiler
```

## Results

The exact solver finds a Pareto front spanning ~460–480 days makespan and
$296M–$430M deferred discovery value. At identical makespan budgets, CP-SAT
reduces deferred value by up to **25% vs the GA**, demonstrating the value of
exact methods for constrained campaign planning.

## References

- SODIR FactPages: [factpages.sodir.no](https://factpages.sodir.no) (open licence)
- Google OR-Tools CP-SAT: [developers.google.com/optimization](https://developers.google.com/optimization)
