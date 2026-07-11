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
  Exact: epsilon-constraint sweep (CP-SAT)
  GA:    9-run weight sweep  (w_makespan, w_deferred) from (0.9,0.1) to (0.1,0.9)

Usage
-----
  python main.py                                    # standard run
  python main.py --interactive --api-key KEY        # LLM interactive mode
  python main.py --interactive --api-key KEY --provider openai
"""

import argparse
import json
import os
import sys
import time
import csv

from src.well_solver     import solve_well_workover
from src.well_ga         import WellWorkoverGA
from src.well_visualizer import plot_comparison_pareto_fronts
from src.well_gantt      import generate_well_gantt


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CAMPAIGN_LABEL = "Block 34 Exploration Campaign -- Norwegian North Sea (20 Wells, 3 Rigs)"
OIL_PRICE      = 80   # USD/bbl (used for display; model works in bbl-days)

INSTANCE_PATH  = os.path.join(os.path.dirname(__file__), 'data', 'snorre_instance.json')
OUTPUT_DIR     = os.path.join(os.path.dirname(__file__), 'output')
IMAGES_DIR     = os.path.join(os.path.dirname(__file__), 'images')


# ---------------------------------------------------------------------------
# Pareto utilities
# ---------------------------------------------------------------------------

def filter_pareto(points):
    """
    Return Pareto-optimal subset.
    Sort by (makespan, deferred) so ties on makespan keep the best deferred.
    """
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


# ---------------------------------------------------------------------------
# Solvers
# ---------------------------------------------------------------------------

def run_exact_pareto_sweep(instance, num_steps=15, time_limit=15.0, step_days=3):
    """
    Epsilon-constraint sweep over makespan limits.
    If instance has 'max_makespan_days', the sweep is capped at that value.
    """
    print("\n  [Exact] epsilon-constraint sweep ({} steps x {}d, {}s/call)".format(
        num_steps, step_days, int(time_limit)))

    t0 = time.time()
    baseline = solve_well_workover(instance, time_limit=time_limit)
    if baseline is None:
        print("  ERROR: baseline call returned no solution.")
        return [], [], None, 0.0

    min_ms    = baseline['makespan']
    max_cap   = instance.get('max_makespan_days')
    all_pts   = [baseline]

    for step in range(1, num_steps + 1):
        limit = min_ms + step * step_days
        if max_cap is not None and limit > max_cap:
            print("  [Exact] step {}: limit {}d exceeds deadline {}d -- stopping sweep".format(
                step, limit, max_cap))
            break
        result = solve_well_workover(instance, max_makespan_limit=limit,
                                     time_limit=time_limit)
        if result:
            all_pts.append(result)

    pareto   = filter_pareto(all_pts)
    elapsed  = time.time() - t0
    print("  [Exact] {} Pareto points in {:.1f}s  (min makespan: {}d)".format(
        len(pareto), elapsed, min_ms))

    return pareto, all_pts, baseline['schedule'], elapsed


def run_ga_pareto_sweep(instance, generations=200):
    """9-run weight sweep to approximate the Pareto front."""
    print("\n  [GA] weight sweep (9 runs, {} generations each)".format(generations))

    t0 = time.time()
    ga = WellWorkoverGA(instance, pop_size=80, mutation_rate=0.2)

    max_cap    = instance.get('max_makespan_days')
    weight_pairs = [
        (0.9, 0.1), (0.8, 0.2), (0.7, 0.3),
        (0.6, 0.4), (0.5, 0.5), (0.4, 0.6),
        (0.3, 0.7), (0.2, 0.8), (0.1, 0.9)
    ]

    all_pts = []
    for w_ms, w_def in weight_pairs:
        res = ga.run_evolution(generations=generations,
                               w_makespan=w_ms, w_deferred=w_def)
        if res:
            if max_cap is None or res['makespan'] <= max_cap:
                all_pts.append(res)

    pareto  = filter_pareto(all_pts)
    elapsed = time.time() - t0
    print("  [GA] {} Pareto points in {:.1f}s".format(len(pareto), elapsed))

    return pareto, all_pts, elapsed


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def save_results(instance, exact_pareto, ga_pareto, gaps,
                 exact_elapsed, ga_elapsed, label, csv_path):
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    oil = instance['oil_price']

    avg_gap = (sum(g['gap_pct'] for g in gaps) / len(gaps)) if gaps else None
    max_gap = max((g['gap_pct'] for g in gaps), default=None)

    with open(csv_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow([])
        w.writerow(['INSTANCE: {}'.format(label)])
        w.writerow(['Metric', 'Exact (CP-SAT)', 'GA'])
        w.writerow(['Runtime (s)',
                    round(exact_elapsed, 1), round(ga_elapsed, 1)])
        w.writerow(['Pareto points',
                    len(exact_pareto), len(ga_pareto)])
        w.writerow(['Best makespan (days)',
                    min(p['makespan'] for p in exact_pareto),
                    min(p['makespan'] for p in ga_pareto)])
        w.writerow(['Min deferred (bbl-days)',
                    int(min(p['deferred_production'] for p in exact_pareto)),
                    int(min(p['deferred_production'] for p in ga_pareto))])
        w.writerow(['Min deferred cost (USD)',
                    int(min(p['deferred_production'] for p in exact_pareto) * oil),
                    int(min(p['deferred_production'] for p in ga_pareto) * oil)])
        w.writerow(['Avg deferred gap (%)',
                    '--', round(avg_gap, 1) if avg_gap is not None else '--'])
        w.writerow(['Max deferred gap (%)',
                    '--', round(max_gap, 1) if max_gap is not None else '--'])

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

        if gaps:
            w.writerow([])
            w.writerow(['PER-POINT GAP'])
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

    # Load instance
    if not os.path.exists(INSTANCE_PATH):
        print("\n[ERROR] snorre_instance.json not found at {}".format(INSTANCE_PATH))
        print("Run:  python data/snorre_parser.py  to generate it first.")
        sys.exit(1)

    with open(INSTANCE_PATH) as f:
        instance = json.load(f)

    print("\nInstance: {} wells, {} rigs".format(
        instance['n_wells'], instance['n_rigs']))

    # Solve
    exact_pareto, exact_all, best_sched, exact_t = run_exact_pareto_sweep(instance)
    ga_pareto,    ga_all,    ga_t                = run_ga_pareto_sweep(instance)

    if not exact_pareto or not ga_pareto:
        print("[ERROR] No Pareto points found.")
        sys.exit(1)

    # Gaps
    gaps = compute_gaps(ga_pareto, exact_pareto)
    if gaps:
        avg_g = sum(g['gap_pct'] for g in gaps) / len(gaps)
        max_g = max(g['gap_pct'] for g in gaps)
        print("\n  GA gap: avg={:.1f}%  max={:.1f}%".format(avg_g, max_g))

    # Save CSV
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    save_results(instance, exact_pareto, ga_pareto, gaps,
                 exact_t, ga_t,
                 label=CAMPAIGN_LABEL,
                 csv_path=os.path.join(OUTPUT_DIR, 'results.csv'))

    # Pareto plot
    os.makedirs(IMAGES_DIR, exist_ok=True)
    plot_comparison_pareto_fronts(
        exact_pareto, ga_pareto,
        exact_all=exact_all, ga_all=ga_all,
        output_path=os.path.join(IMAGES_DIR, 'pareto_block34.png'),
        oil_price=instance['oil_price'],
        campaign_label=CAMPAIGN_LABEL
    )

    # Gantt (best CP-SAT schedule = min makespan point)
    best_exact = min(exact_pareto, key=lambda p: p['makespan'])
    gantt_sched = next(
        (p['schedule'] for p in exact_all if p['makespan'] == best_exact['makespan']),
        None
    )
    if gantt_sched:
        generate_well_gantt(
            gantt_sched, instance,
            output_html_path=os.path.join(IMAGES_DIR, 'gantt_block34.html'),
            title=CAMPAIGN_LABEL + " -- Optimal Schedule"
        )

    print("\n[DONE] All outputs written to output/ and images/")


# ---------------------------------------------------------------------------
# Interactive (LLM) mode
# ---------------------------------------------------------------------------

def interactive_mode(api_key, provider='anthropic'):
    from src.constraint_spec   import ConstraintSpec
    from src.llm_formulator    import AnthropicBackend, OpenAIBackend, formulate
    from src.instance_builder  import apply_spec, describe_instance

    print("=" * 60)
    print("LLM INTERACTIVE MODE -- Block 34 Exploration Campaign")
    print("Provider: {}".format(provider))
    print("Type 'quit' to exit.")
    print("=" * 60)

    with open(INSTANCE_PATH) as f:
        base_instance = json.load(f)

    if provider == 'openai':
        backend = OpenAIBackend(api_key=api_key)
    else:
        backend = AnthropicBackend(api_key=api_key)

    while True:
        print()
        request = input("Your scheduling request: ").strip()
        if request.lower() in ('quit', 'exit', 'q'):
            break
        if not request:
            continue

        # LLM translate
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

        # Build instance
        try:
            inst = apply_spec(base_instance, spec)
        except ValueError as e:
            print("  [ERROR] Could not build instance: {}".format(e))
            continue

        print("\n  Instance: {}".format(describe_instance(inst)))

        # Solve
        label = "LLM: {} ({} wells, {} rigs)".format(
            spec.explanation, inst['n_wells'], inst['n_rigs'])

        exact_pareto, exact_all, _, exact_t = run_exact_pareto_sweep(inst)
        ga_pareto,    ga_all,    ga_t       = run_ga_pareto_sweep(inst)

        if not exact_pareto or not ga_pareto:
            print("  [ERROR] No solution found.")
            continue

        gaps = compute_gaps(ga_pareto, exact_pareto)
        if gaps:
            avg_g = sum(g['gap_pct'] for g in gaps) / len(gaps)
            print("\n  GA gap: avg={:.1f}%".format(avg_g))

        os.makedirs(OUTPUT_DIR, exist_ok=True)
        save_results(inst, exact_pareto, ga_pareto, gaps,
                     exact_t, ga_t, label=label,
                     csv_path=os.path.join(OUTPUT_DIR, 'llm_results.csv'))

        os.makedirs(IMAGES_DIR, exist_ok=True)
        max_ms = inst.get('max_makespan_days')
        plot_comparison_pareto_fronts(
            exact_pareto, ga_pareto,
            exact_all=exact_all, ga_all=ga_all,
            output_path=os.path.join(IMAGES_DIR, 'pareto_llm.png'),
            oil_price=inst['oil_price'],
            campaign_label=label,
            max_makespan_days=max_ms
        )

        best_exact = min(exact_pareto, key=lambda p: p['makespan'])
        gantt_sched = next(
            (p['schedule'] for p in exact_all
             if p['makespan'] == best_exact['makespan']), None
        )
        if gantt_sched:
            generate_well_gantt(
                gantt_sched, inst,
                output_html_path=os.path.join(IMAGES_DIR, 'gantt_llm.html'),
                title=label
            )

        print("\n  [DONE] Results in output/ and images/")

        # Print Pareto summary
        print("\n  Pareto front (CP-SAT):")
        for p in sorted(exact_pareto, key=lambda x: x['makespan']):
            print("    makespan={:3d}d  deferred=${:.0f}M".format(
                p['makespan'],
                p['deferred_production'] * inst['oil_price'] / 1e6))

        if max_ms:
            print("\n  [Deadline] Only points <= {}d shown.".format(max_ms))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Block 34 Exploration Campaign Scheduler')
    parser.add_argument('--interactive', action='store_true',
                        help='Enable LLM interactive mode')
    parser.add_argument('--api-key',    type=str, default=None,
                        help='API key for LLM provider')
    parser.add_argument('--provider',   type=str, default='anthropic',
                        choices=['anthropic', 'openai'],
                        help='LLM provider (default: anthropic)')
    args = parser.parse_args()

    if args.interactive:
        if not args.api_key:
            print("[ERROR] --api-key required for interactive mode.")
            sys.exit(1)
        interactive_mode(api_key=args.api_key, provider=args.provider)
    else:
        main()
