
import time, copy

def subsample_instance(base_instance, n_wells):
    import copy
    inst = copy.deepcopy(base_instance)
    inst["wells"] = [dict(w) for w in base_instance["wells"][:n_wells]]
    for i, w in enumerate(inst["wells"]): w["id"] = i
    inst["n_wells"] = n_wells
    inst["mob_time"] = [
        [base_instance["mob_time"][i][j] for j in range(n_wells)]
        for i in range(n_wells)]
    return inst

def run_scaling_experiments(base_instance, sizes=None, num_steps=8, time_limit=15.0):
    from main import run_exact_pareto_sweep
    if sizes is None: sizes = [5, 10, 15, 20]
    results = []
    for n in sizes:
        print("  {:2d} wells ... ".format(n), end="", flush=True)
        inst = subsample_instance(base_instance, n)
        t0 = time.time()
        pareto, _, _, _ = run_exact_pareto_sweep(inst, num_steps=num_steps, time_limit=time_limit)
        elapsed = time.time() - t0
        best_ms = min(p["makespan"] for p in pareto) if pareto else None
        print("{:.1f}s  |  {} Pareto points  |  best makespan {}d".format(
            elapsed, len(pareto), best_ms))
        results.append({"n_wells": n, "solve_time_s": elapsed,
                         "n_pareto_points": len(pareto), "best_makespan": best_ms})
    return results
