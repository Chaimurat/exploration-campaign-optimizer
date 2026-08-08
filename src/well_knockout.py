"""
well_knockout.py
----------------
Simulates a well becoming undrillable (dry hole, regulatory hold, equipment failure).
Removes one well, re-indexes the instance, rebuilds mob matrix, re-runs solvers.
"""
import copy, math

def build_knockout_instance(base_instance, well_name):
    inst = copy.deepcopy(base_instance)
    remaining = [w for w in inst["wells"] if w["name"] != well_name]
    if len(remaining) == len(inst["wells"]):
        raise ValueError("Well '{}' not found.".format(well_name))
    n = len(remaining)
    names = {w["name"]: idx for idx, w in enumerate(inst["wells"])}
    for i, w in enumerate(remaining):
        w["id"] = i
    mob = [[0]*n for _ in range(n)]
    for i, wi in enumerate(remaining):
        for j, wj in enumerate(remaining):
            if i == j: continue
            if "utm_x" in wi and "utm_x" in wj:
                dx = wi["utm_x"] - wj["utm_x"]
                dy = wi["utm_y"] - wj["utm_y"]
                dist_km = math.sqrt(dx*dx + dy*dy) / 1000.0
                mob[i][j] = max(1, math.ceil(dist_km / 5.0))
            else:
                oi = names[wi["name"]]; oj = names[wj["name"]]
                mob[i][j] = base_instance["mob_time"][oi][oj]
    inst["wells"] = remaining; inst["n_wells"] = n
    inst["mob_time"] = mob; inst["knocked_out"] = well_name
    return inst

def select_knockout_wells(base_instance, optimal_schedule=None):
    wells = base_instance["wells"]; seen = set(); candidates = []
    top = max(wells, key=lambda w: w["flow_rate"])
    candidates.append({"name": top["name"],
        "reason": "highest DST rate ({:,} bbl/day)".format(top["flow_rate"])})
    seen.add(top["name"])
    if optimal_schedule:
        first = min(optimal_schedule, key=lambda s: s["start"])
        if first["well_name"] not in seen:
            candidates.append({"name": first["well_name"],
                "reason": "first drilled in optimal schedule (day {:.0f})".format(first["start"])})
            seen.add(first["well_name"])
    top_block = top["name"][:4]
    other = "34/10" if "34/7" in top_block else "34/7"
    others = [w for w in wells if w["name"].startswith(other) and w["name"] not in seen]
    if others:
        best = max(others, key=lambda w: w["flow_rate"])
        candidates.append({"name": best["name"],
            "reason": "top well in {} ({:,} bbl/day)".format(other, best["flow_rate"])})
    return candidates

def summarise_knockout(well_name, reason, base_exact, ko_exact, base_ga, ko_ga, oil_price):
    def best(p, k): return min(p, key=lambda x: x[k]) if p else None
    b_ms=best(base_exact,"makespan"); k_ms=best(ko_exact,"makespan")
    b_def=best(base_exact,"deferred_production"); k_def=best(ko_exact,"deferred_production")
    print("\n  KNOCKOUT: {} ({})".format(well_name, reason))
    print("  " + "-"*52)
    for label, bv, kv in [
        ("Best makespan (days)",
         str(b_ms["makespan"]) if b_ms else "--",
         str(k_ms["makespan"]) if k_ms else "--"),
        ("Min deferred cost",
         "${:.0f}M".format(b_def["deferred_production"]*oil_price/1e6) if b_def else "--",
         "${:.0f}M".format(k_def["deferred_production"]*oil_price/1e6) if k_def else "--"),
    ]:
        print("  {:30s}  {:>10s}  {:>10s}".format(label, bv, kv))
    delta_ms  = (k_ms["makespan"] - b_ms["makespan"]) if b_ms and k_ms else None
    delta_def = ((k_def["deferred_production"] - b_def["deferred_production"])*oil_price/1e6) if b_def and k_def else None
    if delta_ms  is not None: print("  {:30s}  {:>+10.0f}  days".format("Makespan change", delta_ms))
    if delta_def is not None: print("  {:30s}  {:>+10.1f}  $M".format("Deferred cost change", delta_def))
    return {"well": well_name, "reason": reason,
            "base_makespan": b_ms["makespan"] if b_ms else None,
            "ko_makespan":   k_ms["makespan"] if k_ms else None,
            "delta_makespan": delta_ms,
            "base_deferred_usd": b_def["deferred_production"]*oil_price if b_def else None,
            "ko_deferred_usd":   k_def["deferred_production"]*oil_price if k_def else None,
            "delta_deferred_usd": delta_def*1e6 if delta_def else None,
            "ko_exact_pareto": ko_exact, "ko_ga_pareto": ko_ga}
