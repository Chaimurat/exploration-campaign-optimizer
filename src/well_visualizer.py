"""
well_visualizer.py  (v4)
------------------
Clean Pareto plot: no per-marker text annotations (legend handles labels).
Myopic = green star, other heuristics = grey shapes.
"""

import os
import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def _to_usd(points, oil_price=80):
    return [(p['makespan'], p['deferred_production'] * oil_price) for p in points]


def _find_knee(pts):
    if len(pts) < 3:
        return 0
    x0, y0 = pts[0]
    x1, y1 = pts[-1]
    dx, dy = x1 - x0, y1 - y0
    length = math.hypot(dx, dy)
    if length == 0:
        return 0
    best_idx, best_dist = 0, -1
    for i, (x, y) in enumerate(pts):
        dist = abs(dy * x - dx * y + x1 * y0 - y1 * x0) / length
        if dist > best_dist:
            best_dist, best_idx = dist, i
    return best_idx


# One style per heuristic key (order must match STRATEGIES in well_heuristic.py)
_HEURISTIC_STYLES = {
    'flow_rate_desc': dict(marker='X', color='#888888', size=110,  label='Greedy-Rate (rate↓)'),
    'flow_rate_asc':  dict(marker='s', color='#888888', size=110,  label='Greedy-Rate (rate↑)'),
    'duration_asc':   dict(marker='^', color='#888888', size=110,  label='Greedy-SPT (dur↑)'),
    'duration_desc':  dict(marker='D', color='#888888', size=110,  label='Greedy-LPT (dur↓)'),
    'myopic':         dict(marker='*', color='#2ca02c', size=220,  label='Greedy-Myopic'),
}


def plot_comparison_pareto_fronts(exact_pareto, ga_pareto,
                                   exact_all=None, ga_all=None,
                                   heuristic_results=None,
                                   output_path="images/pareto_wellworkover.png",
                                   oil_price=80,
                                   campaign_label="Block 34 Exploration Campaign",
                                   max_makespan_days=None):

    fig, ax = plt.subplots(figsize=(12, 6.5))

    # ---- Background scatter ----
    if exact_all:
        xs, ys = zip(*_to_usd(exact_all, oil_price))
        ax.scatter(xs, ys, color='steelblue', alpha=0.10, s=18, zorder=2)
    if ga_all:
        xs, ys = zip(*_to_usd(ga_all, oil_price))
        ax.scatter(xs, ys, color='darkorange', alpha=0.10, s=18, zorder=2)

    # ---- Heuristics (plotted first so Pareto front sits on top) ----
    heuristic_handles = []
    if heuristic_results:
        for key, h in heuristic_results.items():
            st = _HEURISTIC_STYLES.get(key, dict(marker='o', color='#888888',
                                                   size=110, label=key))
            hx = h['makespan']
            hy = h['deferred_production'] * oil_price
            ax.scatter([hx], [hy],
                       color=st['color'], s=st['size'], marker=st['marker'],
                       zorder=5, edgecolors='black', linewidths=0.8,
                       alpha=0.85)
            heuristic_handles.append(
                ax.scatter([], [], color=st['color'], s=st['size'],
                           marker=st['marker'], edgecolors='black',
                           linewidths=0.8, label=st['label'])
            )

    # ---- Exact Pareto ----
    exact_pts = sorted(_to_usd(exact_pareto, oil_price), key=lambda p: p[0])
    epx, epy  = zip(*exact_pts)
    ax.plot(epx, epy, color='steelblue', linestyle='--', linewidth=1.5, zorder=6)
    ax.scatter(epx, epy, color='steelblue', s=80, facecolors='none',
               edgecolors='steelblue', linewidths=2, zorder=7)

    # ---- GA Pareto ----
    ga_pts = sorted(_to_usd(ga_pareto, oil_price), key=lambda p: p[0])
    gpx, gpy = zip(*ga_pts)
    ax.plot(gpx, gpy, color='darkorange', linestyle='--', linewidth=1.5, zorder=6)
    ax.scatter(gpx, gpy, color='darkorange', s=80, facecolors='none',
               edgecolors='darkorange', linewidths=2, zorder=7)

    # ---- Knee ----
    if len(exact_pts) >= 3:
        ki = _find_knee(exact_pts)
        kx, ky = exact_pts[ki]
        dx_k = epx[-1] - epx[0]
        dy_k_M = (max(epy) - min(epy)) / 1e6
        ax.scatter([kx], [ky], color='steelblue', s=180,
                   marker='D', zorder=8, edgecolors='navy', linewidths=1.5)
        ax.annotate(
            "Knee: +{:.0f}d saves ${:.0f}M".format(
                kx - epx[0], (max(epy) - min(epy)) / 1e6 *
                (ky - min(epy)) / (max(epy) - min(epy) + 1e-9)),
            xy=(kx, ky),
            xytext=(kx + dx_k * 0.15,
                    ky + (max(epy) - min(epy)) * 0.25),
            fontsize=9, color='navy',
            arrowprops=dict(arrowstyle='->', color='navy', lw=1.3),
            bbox=dict(boxstyle='round,pad=0.3', fc='white',
                      ec='navy', alpha=0.85)
        )

    # ---- GA gap arrow ----
    best_gap_pct, best_gap_pair = 0, None
    for gp in ga_pts:
        candidates = [(ep, ep_y) for ep, ep_y in exact_pts if ep <= gp[0] + 0.5]
        if not candidates:
            continue
        best_e = min(candidates, key=lambda c: c[1])
        if best_e[1] > 0:
            gap = (gp[1] - best_e[1]) / best_e[1] * 100
            if gap > best_gap_pct:
                best_gap_pct  = gap
                best_gap_pair = (gp, best_e)

    if best_gap_pair and best_gap_pct > 0.5:
        (gx, gy), (ex, ey_) = best_gap_pair
        ax.annotate("", xy=(ex, ey_), xytext=(gx, gy),
                    arrowprops=dict(arrowstyle='<->', color='dimgray',
                                    lw=1.5, shrinkA=4, shrinkB=4))
        ax.text(gx + (epx[-1] - epx[0]) * 0.03, (gy + ey_) / 2,
                "GA gap\n{:.0f}%".format(best_gap_pct),
                fontsize=8.5, color='dimgray', va='center',
                bbox=dict(boxstyle='round,pad=0.25', fc='white',
                          ec='dimgray', alpha=0.8))

    # ---- Deadline line ----
    if max_makespan_days is not None:
        ax.axvline(x=max_makespan_days, color='crimson',
                   linestyle=':', linewidth=1.8, zorder=5)
        ax.text(max_makespan_days + 0.3, min(epy),
                "Max: {}d".format(max_makespan_days),
                color='crimson', fontsize=8.5, va='bottom',
                bbox=dict(boxstyle='round,pad=0.2', fc='white',
                          ec='crimson', alpha=0.85))

    # ---- Legend ----
    solver_handles = [
        mpatches.Patch(facecolor='none', edgecolor='steelblue',
                       linewidth=2, label='CP-SAT (exact)'),
        mpatches.Patch(facecolor='none', edgecolor='darkorange',
                       linewidth=2, label='GA (weight sweep)'),
    ]
    ax.legend(handles=solver_handles + heuristic_handles,
              loc='upper right', frameon=True, fontsize=9)

    ax.set_title(
        "{}\nMakespan vs. Deferred Discovery Value".format(campaign_label),
        fontsize=12, fontweight='bold', pad=14
    )
    ax.set_xlabel("Makespan (campaign days)", fontsize=11)
    ax.set_ylabel("Deferred Discovery Value (USD)", fontsize=11)
    ax.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: '${:.0f}M'.format(x / 1e6)))
    ax.grid(True, linestyle=':', alpha=0.55)
    ax.set_facecolor('#fafafa')

    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print("[SUCCESS] Pareto plot saved to: {}".format(output_path))
