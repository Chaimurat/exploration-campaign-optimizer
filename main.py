"""
main.py
-------
Multi-Objective Offshore Exploration Drilling Scheduler
Block 34, Norwegian North Sea -- SODIR real data

Objectives
----------
  Makespan            -- total campaign duration (days)
  Deferred value      -- sum_w(start[w] * flow_rate[w]) * oil_price  (USD)
                         lower is better (drill high-rate wells first)

Methods
-------
  Exact:     epsilon-constraint sweep (CP-SAT)
  GA:        9-run weight sweep  (w_makespan, w_deferred)
  Heuristic: 4 greedy fixed-ordering baselines
"""

import argparse
import json
import os
import sys
import time
import csv

from src.well_solver     import solve_well_workover
from src.well_ga         import WellWorkoverGA
from src.well_heuristic  import run_all_heuristics
from src.well_visualizer import plot_comparison_pareto_fronts
from src.well_gantt      import generate_well_gantt


CAMPAIGN_LABEL = "Block 34 Exploration Campaign -- Norwegian North Sea (20 Wells, 3 Rigs)"
OIL_PRICE      = 80

INSTANCE_PATH = os.path.join(os.path.dirname(__file__), 'data', 'snorre_instance.json')
OUTPUT_DIR    = os.path.join(os.path.dirname(__file__), 'output')
IMAGES_DIR    = os.path.join(os.path.dirname(__file__), 'images')


# ---------------------------------------------------------------------------
# Pareto utilities
# ---------------------------------------------------------------------------

def filter_pareto(points):
    srt    = sorted(points, key=lambda p: (p['makespan'], p['deferred_production']))
    pareto = []
    best   = float('inf')
    for pt in srt:
        if pt['deferred_production'] < best:
            pareto.append(pt)
            best = pt['deferred_production']
    return pareto


def compute_gaps(ga_pareto, exact_pareto):
    gaps = []
    for ga_pt in ga_pareto:
        candidates = [p for p in exact_pareto
                      if p['makespan'] <= ga_pt['makespan']]
        if not candidates:
            continue
        best_exact = min(candidates, key=lambda p: p['deferred_production'])
        gap_pct = ((ga_pt['deferred_production'] - best_exact['deferred_production'])
                   / best_exact['deferred_production'] * 100)
        gaps.append({
            'ga_makespan':    ga_pt['makespan'],
            'ga_deferred':    ga_pt['deferred_production'],
            'exact_makespan': best_exact['makespan'],
            'exact_deferred': best_exact['deferred_production'],
            'gap_pct':        round(gap_pct, 2)
        })
    return gaps


def heuristic_gap(heuristic_results, exact_pareto):
    """
    For each heuristic solution, compute gap vs. best exact point
    with makespan <= heuristic makespan.
    """
    gaps = {}
    for key, h in heuristic_results.items():
        candidates = [p for p in exact_pareto
                      if p['makespan'] <= h['makespan']]
        if not candidates:
            gaps[key] = None
            continue
        best = min(candidates, key=lambda p: p['deferred_production'])
        gap  = ((h['deferred_production'] - best['deferred_production'])
                / best['deferred_production'] * 100)
        gaps[key] = round(gap, 1)
    return gaps


# ---------------------------------------------------------------------------
# Solvers
# ---------------------------------------------------------------------------

def run_exact_pareto_sweep(instance, num_steps=15, time_limit=15.0, step_days=3):
    print("\n  [Exact] epsilon-constraint sweep ({} steps x {}d, {}s/call)".format(
        num_steps, step_days, int(time_limit)))
    t0 = time.time()

    baseline = solve_well_workover(instance, time_limit=time_limit)
    if baseline is None:
        print("  ERROR: baseline returned no solution.")
        return [], [], None, 0.0

    min_ms  = baseline['makespan']
    max_cap = instance.get('max_makespan_days')
    all_pts = [baseline]

    for step in range(1, num_steps + 1):
        limit = min_ms + step * step_days
        if max_cap is not None and limit > max_cap:
            print("  [Exact] deadline {}d reached at step {}".format(max_cap, step))
            break
        result = solve_well_workover(instance, max_makespan_limit=limit,
                                     time_limit=time_limit)
        if result:
            all_pts.append(result)

    pareto  = filter_pareto(all_pts)
    elapsed = time.time() - t0
    print("  [Exact] {} Pareto points in {:.1f}s  (min makespan: {}d)".format(
        len(pareto), elapsed, min_ms))
    return pareto, all_pts, baseline['schedule'], elapsed


def run_ga_pareto_sweep(instance, generations=200):
    print("\n  [GA] weight sweep (9 runs, {} generations each)".format(generations))
    t0 = time.time()
    ga = WellWorkoverGA(instance, pop_size=80, mutation_rate=0.2)

    max_cap = instance.get('max_makespan_days')
    weight_pairs = [
        (0.9, 0.1), (0.8, 0.2), (0.7, 0.3),
        (0.6, 0.4), (0.5, 0.5), (0.4, 0.6),
        (0.3, 0.7), (0.2, 0.8), (0.1, 0.9)
    ]

    all_pts = []
    for w_ms, w_def in weight_pairs:
        res = ga.run_evolution(generations=generations,
                               w_makespan=w_ms, w_deferred=w_def)
        if res and (max_cap is None or res['makespan'] <= max_cap):
            all_pts.append(res)

    pareto  = filter_pareto(all_pts)
    elapsed = time.time() - t0
    print("  [GA] {} Pareto points in {:.1f}s".format(len(pareto), elapsed))
    return pareto, all_pts, elapsed


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def save_results(instance, exact_pareto, ga_pareto, gaps,
                 exact_elapsed, ga_elapsed, label, csv_path,
                 heuristic_results=None, h_gaps=None):
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    oil = instance['oil_price']

    avg_gap = (sum(g['gap_pct'] for g in gaps) / len(gaps)) if gaps else None
    max_gap = max((g['gap_pct'] for g in gaps), default=None)

    with open(csv_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow([])
        w.writerow(['INSTANCE: {}'.format(label)])
        w.writerow(['Metric', 'Exact (CP-SAT)', 'GA', 'Best Heuristic'])

        best_h = (min(heuristic_results.values(),
                      key=lambda h: h['deferred_production'])
                  if heuristic_results else None)

        w.writerow(['Runtime (s)',
                    round(exact_elapsed, 1), round(ga_elapsed, 1), '<1'])
        w.writerow(['Pareto points',
                    len(exact_pareto), len(ga_pareto),
                    len(heuristic_results) if heuristic_results else '--'])
        w.writerow(['Best makespan (days)',
                    min(p['makespan'] for p in exact_pareto),
                    min(p['makespan'] for p in ga_pareto),
                    min(h['makespan'] for h in heuristic_results.values()) if heuristic_results else '--'])
        w.writerow(['Min deferred cost (USD)',
                    int(min(p['deferred_production'] for p in exact_pareto) * oil),
                    int(min(p['deferred_production'] for p in ga_pareto) * oil),
                    int(best_h['deferred_cost']) if best_h else '--'])
        w.writerow(['Avg deferred gap (vs exact) (%)',
                    '--', round(avg_gap, 1) if avg_gap else '--', '--'])
        w.writerow(['Max deferred gap (vs exact) (%)',
                    '--', round(max_gap, 1) if max_gap else '--', '--'])

        w.writerow([])
        w.writerow(['EXACT PARETO FRONT'])
        w.writerow(['Makespan (days)', 'Deferred (bbl-days)', 'Deferred Cost (USD)'])
        for p in sorted(exact_pareto, key=lambda x: x['makespan']):
            w.writerow([p['makespan'],
                        int(p['deferred_production']),
                        int(p['deferred_production'] * oil)])

        w.writerow([])
        w.writerow(['GA PARETO FRONT'])
        w.writerow(['Makespan (days)', 'Deferred (bbl-days)', 'Deferred Cost (USD)'])
        for p in sorted(ga_pareto, key=lambda x: x['makespan']):
            w.writerow([p['makespan'],
                        int(p['deferred_production']),
                        int(p['deferred_production'] * oil)])

        if heuristic_results:
            w.writerow([])
            w.writerow(['HEURISTIC BASELINES'])
            w.writerow(['Strategy', 'Makespan (days)', 'Deferred Cost (USD)',
                        'Gap vs CP-SAT (%)'])
            for key, h in heuristic_results.items():
                gap_str = str(h_gaps[key]) if h_gaps and h_gaps.get(key) is not None else '--'
                w.writerow([h['label'], int(h['makespan']),
                             int(h['deferred_cost']), gap_str])

        if gaps:
            w.writerow([])
            w.writerow(['GA PER-POINT GAP'])
            w.writerow(['GA Makespan', 'GA Deferred', 'Exact Makespan',
                        'Exact Deferred', 'Gap (%)'])
            for g in gaps:
                w.writerow([g['ga_makespan'], int(g['ga_deferred']),
                             g['exact_makespan'], int(g['exact_deferred']),
                             g['gap_pct']])

    print("[SUCCESS] Results saved to: {}".format(csv_path))


# ---------------------------------------------------------------------------
# Standard run
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print(CAMPAIGN_LABEL)
    print("=" * 60)

    if not os.path.exists(INSTANCE_PATH):
        print("\n[ERROR] snorre_instance.json not found.")
        print("Run:  python data/snorre_parser.py  to generate it.")
        sys.exit(1)

    with open(INSTANCE_PATH) as f:
        instance = json.load(f)

    print("\nInstance: {} wells, {} rigs".format(
        instance['n_wells'], instance['n_rigs']))

    exact_pareto, exact_all, _, exact_t = run_exact_pareto_sweep(instance)
    ga_pareto,    ga_all,    ga_t       = run_ga_pareto_sweep(instance)

    # Heuristics
    print("\n  [Heuristic] running 4 greedy strategies...")
    t0 = time.time()
    h_results = run_all_heuristics(instance)
    h_elapsed = time.time() - t0
    for key, h in h_results.items():
        print("    {:20s}  makespan={:4.0f}d  deferred=${:.0f}M".format(
            h['label'], h['makespan'],
            h['deferred_production'] * instance['oil_price'] / 1e6))

    if not exact_pareto or not ga_pareto:
        print("[ERROR] No Pareto points found.")
        sys.exit(1)

    gaps  = compute_gaps(ga_pareto, exact_pareto)
    h_gaps = heuristic_gap(h_results, exact_pareto)

    if gaps:
        avg_g = sum(g['gap_pct'] for g in gaps) / len(gaps)
        print("\n  GA gap: avg={:.1f}%".format(avg_g))

    print("\n  Heuristic gaps vs CP-SAT:")
    for key, gap in h_gaps.items():
        print("    {:20s}  gap={} %".format(h_results[key]['label'],
                                             gap if gap is not None else '--'))

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    save_results(instance, exact_pareto, ga_pareto, gaps,
                 exact_t, ga_t,
                 label=CAMPAIGN_LABEL,
                 csv_path=os.path.join(OUTPUT_DIR, 'results.csv'),
                 heuristic_results=h_results,
                 h_gaps=h_gaps)

    os.makedirs(IMAGES_DIR, exist_ok=True)
    plot_comparison_pareto_fronts(
        exact_pareto, ga_pareto,
        exact_all=exact_all, ga_all=ga_all,
        heuristic_results=h_results,
        output_path=os.path.join(IMAGES_DIR, 'pareto_block34.png'),
        oil_price=instance['oil_price'],
        campaign_label=CAMPAIGN_LABEL
    )

    best_exact  = min(exact_pareto, key=lambda p: p['makespan'])
    gantt_sched = next(
        (p['schedule'] for p in exact_all
         if p['makespan'] == best_exact['makespan']), None)
    if gantt_sched:
        generate_well_gantt(
            gantt_sched, instance,
            output_html_path=os.path.join(IMAGES_DIR, 'gantt_block34.html'),
            title=CAMPAIGN_LABEL + " -- Optimal Schedule"
        )

    print("\n[DONE] All outputs in output/ and images/")


# ---------------------------------------------------------------------------
# Interactive (LLM) mode
# ---------------------------------------------------------------------------

def interactive_mode(api_key, provider='anthropic'):
    from src.constraint_spec  import ConstraintSpec
    from src.llm_formulator   import AnthropicBackend, OpenAIBackend, formulate
    from src.instance_builder import apply_spec, describe_instance

    print("=" * 60)
    print("LLM INTERACTIVE MODE -- Block 34 Exploration Campaign")
    print("Provider: {}  |  Type 'quit' to exit.".format(provider))
    print("=" * 60)

    with open(INSTANCE_PATH) as f:
        base_instance = json.load(f)

    backend = (OpenAIBackend(api_key=api_key) if provider == 'openai'
               else AnthropicBackend(api_key=api_key))

    while True:
        print()
        request = input("Your scheduling request: ").strip()
        if request.lower() in ('quit', 'exit', 'q'):
            break
        if not request:
            continue

        print("\n  [LLM] translating...")
        try:
            spec = formulate(request, backend)
        except ValueError as e:
            print("  [ERROR] LLM translation failed: {}".format(e))
            continue

        print("\n  LLM interpretation: {}".format(spec.explanation))
        print("  Spec: {}".format(spec.model_dump(exclude_none=True)))

        confirm = input("\n  Run solver with this spec? [y/n]: ").strip().lower()
        if confirm != 'y':
            print("  Cancelled.")
            continue

        try:
            inst = apply_spec(base_instance, spec)
        except ValueError as e:
            print("  [ERROR] Could not build instance: {}".format(e))
            continue

        print("\n  Instance: {}".format(describe_instance(inst)))

        label = "LLM: {} ({} wells, {} rigs)".format(
            spec.explanation, inst['n_wells'], inst['n_rigs'])

        exact_pareto, exact_all, _, exact_t = run_exact_pareto_sweep(inst)
        ga_pareto,    ga_all,    ga_t       = run_ga_pareto_sweep(inst)

        print("\n  [Heuristic] running baselines...")
        h_results = run_all_heuristics(inst)
        h_gaps    = heuristic_gap(h_results, exact_pareto) if exact_pareto else {}

        if not exact_pareto or not ga_pareto:
            print("  [ERROR] No solution found.")
            continue

        gaps = compute_gaps(ga_pareto, exact_pareto)
        if gaps:
            avg_g = sum(g['gap_pct'] for g in gaps) / len(gaps)
            print("  GA gap: avg={:.1f}%".format(avg_g))

        os.makedirs(OUTPUT_DIR, exist_ok=True)
        save_results(inst, exact_pareto, ga_pareto, gaps,
                     exact_t, ga_t, label=label,
                     csv_path=os.path.join(OUTPUT_DIR, 'llm_results.csv'),
                     heuristic_results=h_results, h_gaps=h_gaps)

        os.makedirs(IMAGES_DIR, exist_ok=True)
        max_ms = inst.get('max_makespan_days')
        plot_comparison_pareto_fronts(
            exact_pareto, ga_pareto,
            exact_all=exact_all, ga_all=ga_all,
            heuristic_results=h_results,
            output_path=os.path.join(IMAGES_DIR, 'pareto_llm.png'),
            oil_price=inst['oil_price'],
            campaign_label=label,
            max_makespan_days=max_ms
        )

        best_exact  = min(exact_pareto, key=lambda p: p['makespan'])
        gantt_sched = next(
            (p['schedule'] for p in exact_all
             if p['makespan'] == best_exact['makespan']), None)
        if gantt_sched:
            generate_well_gantt(
                gantt_sched, inst,
                output_html_path=os.path.join(IMAGES_DIR, 'gantt_llm.html'),
                title=label)

        print("\n  [DONE] Results in output/ and images/")

        print("\n  CP-SAT Pareto front:")
        for p in sorted(exact_pareto, key=lambda x: x['makespan']):
            print("    makespan={:3d}d  deferred=${:.0f}M".format(
                p['makespan'],
                p['deferred_production'] * inst['oil_price'] / 1e6))

        if max_ms:
            print("  [Deadline] only points <= {}d shown.".format(max_ms))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Block 34 Exploration Campaign Scheduler')
    parser.add_argument('--interactive', action='store_true')
    parser.add_argument('--api-key',  type=str, default=None)
    parser.add_argument('--provider', type=str, default='anthropic',
                        choices=['anthropic', 'openai'])
    args = parser.parse_args()

    if args.interactive:
        if not args.api_key:
            print("[ERROR] --api-key required for interactive mode.")
            sys.exit(1)
        interactive_mode(api_key=args.api_key, provider=args.provider)
    else:
        main()
