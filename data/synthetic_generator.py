"""
synthetic_generator.py
----------------------
Generates a realistic synthetic well workover scheduling instance.

Field layout: 12 wells scattered across a North-Sea-style grid (km).
Mobilization time between wells is derived from Euclidean distance
at an assumed rig travel speed of 5 km/day (tow + rig-up overhead).

Outputs
-------
data/well_instance.json  --  all problem data in one file
Prints a human-readable summary to stdout.
"""

import numpy as np
import json
import os

# ---------------------------------------------------------------------------
# Parameters you can tweak
# ---------------------------------------------------------------------------
N_WELLS      = 12
N_RIGS       = 3
OIL_PRICE    = 80          # USD per barrel
RIG_SPEED    = 5.0         # km per day (mobilisation travel rate)
SEED         = 42
# ---------------------------------------------------------------------------


def generate_instance(n_wells=N_WELLS, n_rigs=N_RIGS,
                      oil_price=OIL_PRICE, seed=SEED):
    rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------
    # Wells: position on a 50x50 km grid, duration, flow rate
    # ------------------------------------------------------------------
    # Realistic North-Sea workover durations: 3-14 days
    # Flow rates: 400-2000 bbl/day (mix of high/low producers)
    positions_x = rng.uniform(0, 50, n_wells).round(1)
    positions_y = rng.uniform(0, 50, n_wells).round(1)
    durations   = rng.integers(3, 15, n_wells).tolist()   # days
    flow_rates  = rng.integers(400, 2001, n_wells).tolist()  # bbl/day

    wells = []
    for i in range(n_wells):
        wells.append({
            "id":        i,
            "name":      "W{:02d}".format(i + 1),
            "x_km":      float(positions_x[i]),
            "y_km":      float(positions_y[i]),
            "duration":  int(durations[i]),
            "flow_rate": int(flow_rates[i])   # barrels per day
        })

    # ------------------------------------------------------------------
    # Rigs: three rigs with different day rates
    # ------------------------------------------------------------------
    rig_names     = ["Rig-Alpha", "Rig-Beta",  "Rig-Gamma"]
    rig_day_rates = [65000,        50000,        42000]      # USD/day

    rigs = []
    for r in range(n_rigs):
        rigs.append({
            "id":       r,
            "name":     rig_names[r],
            "day_rate": rig_day_rates[r]
        })

    # ------------------------------------------------------------------
    # Mobilisation time matrix: mob_time[i][j] = days to move rig
    # from well i to well j (symmetric, diagonal = 0)
    # ceil(distance / speed), minimum 1 day even for close wells
    # ------------------------------------------------------------------
    mob_time = []
    for i in range(n_wells):
        row = []
        for j in range(n_wells):
            if i == j:
                row.append(0)
            else:
                dist = np.sqrt(
                    (positions_x[i] - positions_x[j]) ** 2 +
                    (positions_y[i] - positions_y[j]) ** 2
                )
                days = max(1, int(np.ceil(dist / RIG_SPEED)))
                row.append(days)
        mob_time.append(row)

    # ------------------------------------------------------------------
    # Bundle everything
    # ------------------------------------------------------------------
    instance = {
        "n_wells":   n_wells,
        "n_rigs":    n_rigs,
        "oil_price": oil_price,
        "wells":     wells,
        "rigs":      rigs,
        "mob_time":  mob_time
    }
    return instance


def print_summary(instance):
    wells     = instance["wells"]
    rigs      = instance["rigs"]
    mob_time  = instance["mob_time"]
    n_wells   = instance["n_wells"]

    print("=" * 60)
    print("  Synthetic Well Workover Instance")
    print("=" * 60)

    print("\n  WELLS ({})".format(n_wells))
    print("  {:<6} {:<8} {:>10} {:>12} {:>8}".format(
        "ID", "Name", "Duration", "Flow Rate", "Pos(km)"))
    print("  " + "-" * 52)
    for w in wells:
        print("  {:<6} {:<8} {:>8}d  {:>9} bbl/d  ({:.0f},{:.0f})".format(
            w["id"], w["name"], w["duration"], w["flow_rate"],
            w["x_km"], w["y_km"]))

    print("\n  RIGS ({})".format(len(rigs)))
    print("  {:<6} {:<12} {:>12}".format("ID", "Name", "Day Rate (USD)"))
    print("  " + "-" * 34)
    for r in rigs:
        print("  {:<6} {:<12} {:>12,}".format(r["id"], r["name"], r["day_rate"]))

    print("\n  MOBILISATION TIME MATRIX (days)")
    header = "       " + "".join("{:>5}".format("W{:02d}".format(i+1))
                                  for i in range(n_wells))
    print(header)
    for i, row in enumerate(mob_time):
        line = "  W{:02d}  ".format(i+1)
        line += "".join("{:>5}".format(v) for v in row)
        print(line)

    total_work = sum(w["duration"] for w in wells)
    avg_mob = sum(mob_time[i][j]
                  for i in range(n_wells)
                  for j in range(n_wells) if i != j) / (n_wells * (n_wells - 1))
    print("\n  SUMMARY")
    print("  Total workover days (all wells):  {}".format(total_work))
    print("  Average mobilisation time:        {:.1f} days".format(avg_mob))
    print("  Oil price:                        ${}/bbl".format(instance["oil_price"]))
    print("=" * 60)


if __name__ == "__main__":
    instance = generate_instance()

    out_dir = os.path.join(os.path.dirname(__file__))
    out_path = os.path.join(out_dir, "well_instance.json")
    with open(out_path, "w") as f:
        json.dump(instance, f, indent=2)

    print_summary(instance)
    print("\n  Saved to: {}".format(out_path))
