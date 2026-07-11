"""
well_visualizer.py
------------------
Pareto front plot for the Block 34 exploration campaign.
Features: knee annotation, GA gap arrow, optional max-makespan deadline line.
"""

import os
import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_usd(points, oil_price=80):
    return [(p['makespan'], p['deferred_production'] * oil_price) for p in points]


def _find_knee(pts):
    """
    Return the index of the 'knee' point in a sorted list of (x, y) tuples.
    Uses the perpendicular distance from the line joining the first and last point.
    """
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


# ---------------------------------------------------------------------------
# Main plot
# ---------------------------------------------------------------------------

def plot_comparison_pareto_fronts(exact_pareto, ga_pareto,
                                   exact_all=None, ga_all=None,
                                   output_path="images/pareto_wellworkover.png",
                                   oil_price=80,
                                   campaign_label="Block 34 Exploration Campaign",
                                   max_makespan_days=None):
    """
    Plot CP-SAT and GA Pareto fronts with:
      - faint background scatter of all explored points
      - knee annotation on exact front
      - GA gap double-headed arrow
      - optional vertical deadline at max_makespan_days
    """
    fig, ax = plt.subplots(figsize=(11, 6))

    # ---- Background scatter (very faint, not in legend) ----
    if exact_all:
        xs, ys = zip(*_to_usd(exact_all, oil_price))
        ax.scatter(xs, ys, color='steelblue', alpha=0.10, s=18, zorder=2)

    if ga_all:
        xs, ys = zip(*_to_usd(ga_all, oil_price))
        ax.scatter(xs, ys, color='darkorange', alpha=0.10, s=18, zorder=2)

    # ---- Exact Pareto front ----
    exact_pts = sorted(_to_usd(exact_pareto, oil_price), key=lambda p: p[0])
    epx, epy  = zip(*exact_pts)
    ax.plot(epx, epy, color='steelblue', linestyle='--', linewidth=1.5, zorder=3)
    ax.scatter(epx, epy, color='steelblue', s=80, facecolors='none',
               edgecolors='steelblue', linewidths=2, zorder=4)

    # ---- GA Pareto front ----
    ga_pts = sorted(_to_usd(ga_pareto, oil_price), key=lambda p: p[0])
    gpx, gpy = zip(*ga_pts)
    ax.plot(gpx, gpy, color='darkorange', linestyle='--', linewidth=1.5, zorder=3)
    ax.scatter(gpx, gpy, color='darkorange', s=80, facecolors='none',
               edgecolors='darkorange', linewidths=2, zorder=4)

    # ---- Knee annotation ----
    if len(exact_pts) >= 3:
        ki = _find_knee(exact_pts)
        kx, ky = exact_pts[ki]
        dx_knee = exact_pts[-1][0] - exact_pts[0][0]
        dy_knee = (exact_pts[0][1] - exact_pts[-1][1]) / 1e6  # save in $M

        ax.scatter([kx], [ky], color='steelblue', s=160,
                   marker='D', zorder=6, edgecolors='navy', linewidths=1.5)
        ax.annotate(
            "Knee: +{:,.0f}d saves ${:.0f}M".format(
                kx - exact_pts[0][0], dy_knee),
            xy=(kx, ky),
            xytext=(kx + dx_knee * 0.18, ky + (epy[-1] - epy[0]) * 0.22),
            fontsize=9, color='navy',
            arrowprops=dict(arrowstyle='->', color='navy', lw=1.3),
            bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='navy', alpha=0.85)
        )

    # ---- GA gap annotation (double-headed arrow at widest gap) ----
    # Find the GA point with the largest absolute gap vs nearest exact point
    best_gap_pct  = 0
    best_gap_pair = None
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
        mid_y = (gy + ey_) / 2
        ax.annotate(
            "", xy=(ex, ey_), xytext=(gx, gy),
            arrowprops=dict(arrowstyle='<->', color='dimgray', lw=1.5,
                            shrinkA=4, shrinkB=4)
        )
        ax.text(gx + (epx[-1] - epx[0]) * 0.03, mid_y,
                "GA gap\n{:.0f}%".format(best_gap_pct),
                fontsize=8.5, color='dimgray', va='center',
                bbox=dict(boxstyle='round,pad=0.25', fc='white', ec='dimgray', alpha=0.8))

    # ---- Optional deadline line ----
    if max_makespan_days is not None:
        ax.axvline(x=max_makespan_days, color='crimson', linestyle=':', linewidth=1.8,
                   zorder=5, label='__nolegend__')
        ax.text(max_makespan_days + 1, ax.get_ylim()[0] if ax.get_ylim()[0] > 0 else epy[0],
                "Max: {}d".format(max_makespan_days),
                color='crimson', fontsize=8.5, va='bottom',
                bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='crimson', alpha=0.85))

    # ---- Legend (two patches only) ----
    legend_handles = [
        mpatches.Patch(facecolor='none', edgecolor='steelblue',
                       linewidth=2, label='CP-SAT (exact)'),
        mpatches.Patch(facecolor='none', edgecolor='darkorange',
                       linewidth=2, label='GA (weight sweep)')
    ]
    ax.legend(handles=legend_handles, loc='upper right', frameon=True, fontsize=9)

    ax.set_title(
        "{}\nMakespan vs. Deferred Discovery Value".format(campaign_label),
        fontsize=12, fontweight='bold', pad=12
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
