"""
volve_parser.py
---------------
Builds a well workover scheduling instance from real Equinor Volve data.

Sources
-------
- Volve production data.xlsx  : daily production per well (BORE_OIL_VOL, ON_STREAM_HRS)
- wellbore_coordinates.xlsx   : NPD surface coordinates for all Norwegian wells

What we extract
---------------
Wells        : 7 active Volve wells (5 oil producers + 2 water injectors)
Flow rates   : average daily oil production (bbl/day) during active periods
               injectors get a proxy rate = 50% of lowest producer (pressure support)
Workover dur : estimated from production data -- longest gap of zero production
               while the field was otherwise active (proxy for planned intervention)
Mobilisation : all wells on same platform (within 10m), so fixed 1-2 days per move

Output
------
data/volve_instance.json  (same format as well_instance.json)
"""

import pandas as pd
import numpy as np
import json
import os

OIL_PRICE  = 80   # USD per barrel
N_RIGS     = 2    # jackup rig + intervention vessel

# Volve active wells in order of importance
VOLVE_WELLS = ['15/9-F-12', '15/9-F-14', '15/9-F-11',
               '15/9-F-1 C', '15/9-F-15 D', '15/9-F-5', '15/9-F-4']

DATA_DIR = os.path.dirname(__file__)


def estimate_workover_duration(well_df):
    """
    Estimate workover duration from production history.
    Find the longest run of zero-production days while the well was
    nominally active (i.e., between first and last production dates).
    Clamp to [3, 14] days as realistic bounds for North Sea platform workover.
    """
    df = well_df.sort_values('DATEPRD').copy()

    # Active window: first to last day with any production
    active = df[df['BORE_OIL_VOL'] > 0]
    if active.empty:
        return 5  # default for injectors

    first = active['DATEPRD'].min()
    last  = active['DATEPRD'].max()
    window = df[(df['DATEPRD'] >= first) & (df['DATEPRD'] <= last)]

    # Find longest consecutive run of zero production days
    zero_days = (window['BORE_OIL_VOL'] == 0).astype(int)
    max_run = 0
    current = 0
    for v in zero_days:
        if v == 1:
            current += 1
            max_run = max(max_run, current)
        else:
            current = 0

    duration = max(3, min(14, max_run))
    return int(duration)


def parse_volve_instance():
    prod_path   = os.path.join(DATA_DIR, 'Volve production data.xlsx')
    coord_path  = os.path.join(DATA_DIR, 'wellbore_coordinates.xlsx')

    prod   = pd.read_excel(prod_path)
    coords = pd.read_excel(coord_path)

    # ------------------------------------------------------------------
    # Filter: Volve 15/9-F wells only (Statoil/Equinor operator)
    # ------------------------------------------------------------------
    volve_coords = coords[
        coords['Wellbore name'].str.startswith('15/9-F', na=False)
    ].copy()

    # ------------------------------------------------------------------
    # Build wells list
    # ------------------------------------------------------------------
    prod_volve = prod[prod['NPD_FIELD_NAME'] == 'VOLVE'].copy()

    # Average daily oil during ACTIVE days (ON_STREAM_HRS > 0)
    active_prod = prod_volve[prod_volve['ON_STREAM_HRS'] > 0]
    avg_rates = (active_prod.groupby('NPD_WELL_BORE_NAME')['BORE_OIL_VOL']
                 .mean().round(0))

    # Minimum producer rate (for injector proxy)
    producer_rates = avg_rates[avg_rates > 0]
    min_producer   = int(producer_rates.min()) if not producer_rates.empty else 200

    wells = []
    for i, name in enumerate(VOLVE_WELLS):
        well_data = prod_volve[prod_volve['NPD_WELL_BORE_NAME'] == name]

        # Flow rate
        if name in avg_rates.index and avg_rates[name] > 0:
            flow_rate = int(avg_rates[name])
        else:
            # Injector: proxy = 50% of lowest producer
            flow_rate = max(100, min_producer // 2)

        # Workover duration
        duration = estimate_workover_duration(well_data)

        # Surface coordinates (UTM metres)
        coord_row = volve_coords[volve_coords['Wellbore name'] == name]
        if not coord_row.empty:
            ns_utm = float(coord_row['NS UTM [m]'].iloc[0])
            ew_utm = float(coord_row['EW UTM [m]'].iloc[0])
        else:
            # Fallback: field centre
            ns_utm = 6478563.0
            ew_utm = 435050.0

        wells.append({
            "id":        i,
            "name":      name,
            "x_km":      round(ew_utm / 1000, 3),
            "y_km":      round(ns_utm / 1000, 3),
            "duration":  duration,
            "flow_rate": flow_rate
        })

    # ------------------------------------------------------------------
    # Rigs  (jackup + intervention vessel)
    # ------------------------------------------------------------------
    rigs = [
        {"id": 0, "name": "Maersk-Inspirer", "day_rate": 150000},
        {"id": 1, "name": "Intervention-Vessel", "day_rate": 75000}
    ]

    # ------------------------------------------------------------------
    # Mobilisation matrix
    # All wells within ~10m on same platform -> fixed 1-2 days per move
    # (rig-down, skid to next slot, rig-up)
    # ------------------------------------------------------------------
    n = len(wells)
    rng = np.random.default_rng(42)
    mob_time = []
    for i in range(n):
        row = []
        for j in range(n):
            if i == j:
                row.append(0)
            else:
                row.append(int(rng.integers(1, 3)))  # 1 or 2 days
        mob_time.append(row)

    instance = {
        "n_wells":   n,
        "n_rigs":    N_RIGS,
        "oil_price": OIL_PRICE,
        "wells":     wells,
        "rigs":      rigs,
        "mob_time":  mob_time
    }
    return instance


def print_volve_summary(instance):
    wells = instance['wells']
    print("=" * 60)
    print("  Volve Field -- Well Workover Instance")
    print("  Source: Equinor Volve Open Dataset")
    print("=" * 60)
    print("\n  WELLS")
    print("  {:<12} {:>10} {:>12} {:>10}".format(
        "Well", "Duration", "Flow Rate", "Type"))
    print("  " + "-" * 48)
    for w in wells:
        wtype = "Injector" if w['flow_rate'] < 300 else "Producer"
        print("  {:<12} {:>8}d  {:>9} bbl/d  {:>10}".format(
            w['name'], w['duration'], w['flow_rate'], wtype))

    print("\n  RIGS")
    for r in instance['rigs']:
        print("  {} -- ${:,}/day".format(r['name'], r['day_rate']))

    print("\n  MOBILISATION MATRIX (days between slots)")
    header = "              " + "".join(
        "{:>6}".format(w['name'].replace('15/9-F-', 'F-'))
        for w in wells)
    print(header)
    for i, row in enumerate(instance['mob_time']):
        label = "  {:12}".format(wells[i]['name'].replace('15/9-F-', 'F-'))
        print(label + "".join("{:>6}".format(v) for v in row))

    print("\n  Oil price: ${}/bbl".format(instance['oil_price']))
    print("=" * 60)


if __name__ == "__main__":
    instance = parse_volve_instance()

    out_path = os.path.join(DATA_DIR, 'volve_instance.json')
    with open(out_path, 'w') as f:
        json.dump(instance, f, indent=2)

    print_volve_summary(instance)
    print("\n  Saved to: {}".format(out_path))
