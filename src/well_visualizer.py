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


def plot_knockout_comparison(base_exact, ko_results_list,
                              output_path="images/pareto_knockout.png",
                              oil_price=80,
                              campaign_label="Block 34 Exploration Campaign"):
    import os, matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    fig,ax=plt.subplots(figsize=(11,6))
    base_pts=sorted([(p["makespan"],p["deferred_production"]*oil_price) for p in base_exact],key=lambda p:p[0])
    bpx,bpy=zip(*base_pts)
    ax.plot(bpx,bpy,color="steelblue",linestyle="--",linewidth=2,zorder=4)
    ax.scatter(bpx,bpy,color="steelblue",s=90,facecolors="none",edgecolors="steelblue",linewidths=2,zorder=5)
    ko_colors=["#d62728","#ff7f0e","#9467bd"]
    handles=[mpatches.Patch(facecolor="none",edgecolor="steelblue",linewidth=2,label="Baseline (20 wells)")]
    for i,ko in enumerate(ko_results_list):
        if not ko["ko_exact_pareto"]: continue
        c=ko_colors[i%len(ko_colors)]
        pts=sorted([(p["makespan"],p["deferred_production"]*oil_price) for p in ko["ko_exact_pareto"]],key=lambda p:p[0])
        kpx,kpy=zip(*pts)
        ax.plot(kpx,kpy,color=c,linestyle="--",linewidth=1.8,zorder=4)
        ax.scatter(kpx,kpy,color=c,s=80,facecolors="none",edgecolors=c,linewidths=2,zorder=5)
        delta_str="" if ko["delta_deferred_usd"] is None else " (+${:.0f}M deferred)".format(ko["delta_deferred_usd"]/1e6)
        handles.append(mpatches.Patch(facecolor="none",edgecolor=c,linewidth=2,
            label="Knock out {}{}".format(ko["well"],delta_str)))
    ax.legend(handles=handles,loc="upper right",frameon=True,fontsize=9)
    ax.set_title("{}\nKnockout Scenario: Pareto Front Comparison".format(campaign_label),fontsize=12,fontweight="bold",pad=12)
    ax.set_xlabel("Makespan (campaign days)",fontsize=11)
    ax.set_ylabel("Deferred Discovery Value (USD)",fontsize=11)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x,_:"${:.0f}M".format(x/1e6)))
    ax.grid(True,linestyle=":",alpha=0.55); ax.set_facecolor("#fafafa")
    os.makedirs(os.path.dirname(output_path) or ".",exist_ok=True)
    plt.tight_layout(); plt.savefig(output_path,dpi=300,bbox_inches="tight"); plt.close()
    print("[SUCCESS] Knockout plot saved to: {}".format(output_path))


def plot_scaling_results(results, output_path="images/pareto_scaling.png"):
    import os, matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    sizes  = [r["n_wells"]         for r in results]
    times  = [r["solve_time_s"]    for r in results]
    points = [r["n_pareto_points"] for r in results]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.plot(sizes, times, "o-", color="steelblue", linewidth=2.2, markersize=9, zorder=5)
    for x, y in zip(sizes, times):
        ax1.annotate("{:.1f}s".format(y), (x, y),
            textcoords="offset points", xytext=(0, 10), ha="center", fontsize=9)
    ax1.set_xlabel("Number of Wells", fontsize=11)
    ax1.set_ylabel("CP-SAT Solve Time (s)", fontsize=11)
    ax1.set_title("Runtime Scaling (CP-SAT e-constraint sweep)", fontsize=12, fontweight="bold")
    ax1.set_xticks(sizes); ax1.grid(True, linestyle=":", alpha=0.55); ax1.set_facecolor("#fafafa")
    bars = ax2.bar([str(s) for s in sizes], points, color="steelblue", alpha=0.75, width=0.5, zorder=3)
    for bar, v in zip(bars, points):
        ax2.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.2, str(v),
                 ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax2.set_xlabel("Number of Wells", fontsize=11)
    ax2.set_ylabel("Pareto Points Found", fontsize=11)
    ax2.set_title("Pareto Front Density vs. Instance Size", fontsize=12, fontweight="bold")
    ax2.grid(True, linestyle=":", alpha=0.55, axis="y"); ax2.set_facecolor("#fafafa")
    plt.suptitle("Block 34 Exploration Campaign -- Norwegian North Sea\nCP-SAT Scalability Study",
        fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight"); plt.close()
    print("[SUCCESS] Scaling plot saved to:", output_path)
