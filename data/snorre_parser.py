"""
snorre_parser.py
----------------
Parses SODIR (Norwegian Offshore Directorate) public data for block 34
exploration wells (Snorre 34/7 and Statfjord 34/10).

Data sources (download from https://factpages.sodir.no/en/wellbore/TableView/Exploration/All):
  wellbore_exploration_all.csv  -- wellbore metadata incl. coordinates & drill days
  wellbore_dst.csv              -- DST (Drill Stem Test) oil rates

Outputs:
  data/snorre_instance.json     -- 20-well, 3-rig instance for the scheduler

Mobilisation model:
  mob_days(i, j) = max(1, ceil(UTM_distance_km / 5.0))   [5 km/day semisub speed]
"""

import os
import math
import json
import pandas as pd


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DATA_DIR        = os.path.dirname(__file__)
EXPLORATION_CSV = os.path.join(DATA_DIR, 'wellbore_exploration_all.csv')
DST_CSV         = os.path.join(DATA_DIR, 'wellbore_dst.csv')
OUTPUT_JSON     = os.path.join(DATA_DIR, 'snorre_instance.json')

TARGET_BLOCKS   = ['34/7', '34/10']
WELLS_PER_BLOCK = 10
OIL_PRICE       = 80          # USD/bbl
RIG_SPEED_KM    = 5.0         # km/day (semi-submersible transit speed)
N_RIGS          = 3


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

def parse_snorre_instance():
    """
    Read SODIR CSVs, filter to block 34, pick top-10 wells per block
    by max DST oil rate, compute UTM mobilisation matrix.
    """

    # -- Load exploration data --
    exp = pd.read_csv(EXPLORATION_CSV, low_memory=False, encoding='latin-1',
                      sep=',', on_bad_lines='skip')

    # Normalise column names (SODIR sometimes uses semicolons)
    if exp.shape[1] == 1:
        exp = pd.read_csv(EXPLORATION_CSV, low_memory=False, encoding='latin-1',
                          sep=';', on_bad_lines='skip')

    # Identify key columns flexibly
    col_map = {c.lower(): c for c in exp.columns}

    def get_col(*candidates):
        for c in candidates:
            if c.lower() in col_map:
                return col_map[c.lower()]
        raise KeyError("None of {} found in columns: {}".format(candidates, list(exp.columns)[:20]))

    name_col  = get_col('wlbWellboreName', 'wellborename', 'name')
    ns_col    = get_col('wlbNsDecDeg',     'nsdecdeg',     'latitude')
    ew_col    = get_col('wlbEwDesDeg',     'ewdesdeg',     'ewdecdeg', 'longitude')
    days_col  = get_col('wlbDrillDays',    'drilldays',    'drillingdays')

    exp = exp[[name_col, ns_col, ew_col, days_col]].copy()
    exp.columns = ['name', 'lat', 'lon', 'drill_days']
    exp['name']       = exp['name'].astype(str).str.strip()
    exp['drill_days'] = pd.to_numeric(exp['drill_days'], errors='coerce')
    exp['lat']        = pd.to_numeric(exp['lat'],        errors='coerce')
    exp['lon']        = pd.to_numeric(exp['lon'],        errors='coerce')

    # Filter: block 34/7 and 34/10, valid coords and duration
    mask = (
        exp['name'].str.startswith(tuple(TARGET_BLOCKS)) &
        exp['drill_days'].notna() & (exp['drill_days'] > 0) &
        exp['lat'].notna() & exp['lon'].notna()
    )
    exp = exp[mask].copy()

    # -- Load DST data --
    dst = pd.read_csv(DST_CSV, low_memory=False, encoding='latin-1',
                      sep=',', on_bad_lines='skip')
    if dst.shape[1] == 1:
        dst = pd.read_csv(DST_CSV, low_memory=False, encoding='latin-1',
                          sep=';', on_bad_lines='skip')

    dst_col_map = {c.lower(): c for c in dst.columns}

    def get_dst_col(*candidates):
        for c in candidates:
            if c.lower() in dst_col_map:
                return dst_col_map[c.lower()]
        raise KeyError("None of {} found in DST columns: {}".format(candidates, list(dst.columns)[:20]))

    dname_col = get_dst_col('wlbName', 'wellborename', 'name')
    rate_col  = get_dst_col('dstOilRecovery', 'oilrecovery', 'oilrate',
                             'dstOilRcv', 'dstFlowRateOil', 'flowrateoil')

    dst = dst[[dname_col, rate_col]].copy()
    dst.columns = ['name', 'dst_rate']
    dst['name']     = dst['name'].astype(str).str.strip()
    dst['dst_rate'] = pd.to_numeric(dst['dst_rate'], errors='coerce')
    dst = dst[dst['dst_rate'].notna() & (dst['dst_rate'] > 0)]

    # Best DST rate per well
    dst_best = dst.groupby('name')['dst_rate'].max().reset_index()

    # -- Merge --
    merged = exp.merge(dst_best, on='name', how='inner')
    merged = merged[merged['dst_rate'] > 0].copy()

    # -- Top N per block --
    all_wells = []
    for block in TARGET_BLOCKS:
        block_wells = merged[merged['name'].str.startswith(block)].copy()
        block_wells = block_wells.sort_values('dst_rate', ascending=False)
        block_wells = block_wells.head(WELLS_PER_BLOCK)
        all_wells.append(block_wells)

    wells_df = pd.concat(all_wells).reset_index(drop=True)

    if len(wells_df) < 4:
        raise ValueError(
            "Only {} wells found after filtering. Check CSV paths and column names."
            .format(len(wells_df)))

    print("[snorre_parser] {} wells selected across blocks {}".format(
        len(wells_df), TARGET_BLOCKS))

    # -- Convert lat/lon to approximate UTM (zone 31N) --
    # Simple flat-earth approximation sufficient for ~200 km extent
    # 1 deg lat ~ 111 km, 1 deg lon ~ 111 * cos(lat) km
    ref_lat = wells_df['lat'].mean()
    cos_lat = math.cos(math.radians(ref_lat))

    wells_df['utm_x'] = wells_df['lon'] * 111000 * cos_lat   # metres east
    wells_df['utm_y'] = wells_df['lat'] * 111000              # metres north

    # -- Build mob matrix (days) --
    n = len(wells_df)
    mob_matrix = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            dx = wells_df.iloc[i]['utm_x'] - wells_df.iloc[j]['utm_x']
            dy = wells_df.iloc[i]['utm_y'] - wells_df.iloc[j]['utm_y']
            dist_km = math.sqrt(dx*dx + dy*dy) / 1000.0
            mob_matrix[i][j] = max(1, math.ceil(dist_km / RIG_SPEED_KM))

    # -- Assemble instance dict --
    wells_list = []
    name_to_id = {}
    for i, row in wells_df.iterrows():
        idx = wells_df.index.get_loc(i)
        name_to_id[row['name']] = idx
        wells_list.append({
            'id':        idx,
            'name':      row['name'],
            'duration':  int(round(row['drill_days'])),
            'flow_rate': int(round(row['dst_rate'])),
            'utm_x':     float(row['utm_x']),
            'utm_y':     float(row['utm_y'])
        })

    rigs = [
        {'id': 0, 'name': 'Rig-Alpha'},
        {'id': 1, 'name': 'Rig-Beta'},
        {'id': 2, 'name': 'Rig-Gamma'}
    ]

    instance = {
        'name':            'Block 34 Exploration Campaign -- Norwegian North Sea',
        'n_wells':         len(wells_list),
        'n_rigs':          N_RIGS,
        'wells':           wells_list,
        'rigs':            rigs,
        'mob_time':        mob_matrix,
        'oil_price':       OIL_PRICE,
        'well_name_to_id': name_to_id,
        'data_source':     'SODIR (factpages.sodir.no) -- open licence'
    }

    # Print summary
    rates    = [w['flow_rate'] for w in wells_list]
    durs     = [w['duration']  for w in wells_list]
    mob_vals = [mob_matrix[i][j]
                for i in range(n) for j in range(n) if i != j]
    print("[snorre_parser] DST rate  : {}-{} bbl/day (mean {:.0f})".format(
        min(rates), max(rates), sum(rates)/len(rates)))
    print("[snorre_parser] Duration  : {}-{} days (mean {:.0f})".format(
        min(durs), max(durs), sum(durs)/len(durs)))
    print("[snorre_parser] Mob times : {}-{} days (mean {:.1f})".format(
        min(mob_vals), max(mob_vals), sum(mob_vals)/len(mob_vals)))

    # Save
    with open(OUTPUT_JSON, 'w') as f:
        json.dump(instance, f, indent=2)
    print("[snorre_parser] Saved to: {}".format(OUTPUT_JSON))
    return instance


# ---------------------------------------------------------------------------

if __name__ == '__main__':
    parse_snorre_instance()
