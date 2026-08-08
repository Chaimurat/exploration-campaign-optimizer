
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import json, os as _os
from src.well_scaling import run_scaling_experiments
from src.well_visualizer import plot_scaling_results

INSTANCE_PATH = os.path.join("data", "snorre_instance.json")
SIZES      = [5, 10, 15, 20]
NUM_STEPS  = 8
TIME_LIMIT = 15.0
OUTPUT_IMG = os.path.join("images", "pareto_scaling.png")
OUTPUT_CSV = os.path.join("output", "scaling_results.csv")

with open(INSTANCE_PATH) as f: instance = json.load(f)
print("\nRunning CP-SAT scaling experiments ({} sizes x {} steps)...\n".format(len(SIZES), NUM_STEPS))
results = run_scaling_experiments(instance, sizes=SIZES, num_steps=NUM_STEPS, time_limit=TIME_LIMIT)

os.makedirs("output", exist_ok=True)
with open(OUTPUT_CSV, "w", newline="", encoding="utf-8-sig") as f:
    f.write("n_wells,solve_time_s,n_pareto_points,best_makespan_days\n")
    for r in results:
        f.write("{},{:.2f},{},{}\n".format(r["n_wells"], r["solve_time_s"],
            r["n_pareto_points"], r["best_makespan"] if r["best_makespan"] else ""))
print("\nCSV saved to:", OUTPUT_CSV)
plot_scaling_results(results, output_path=OUTPUT_IMG)
