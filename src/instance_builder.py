"""
instance_builder.py
-------------------
Translates a validated ConstraintSpec into a solver-ready instance dict
by filtering and re-indexing the base instance.

The solver and GA are NEVER modified -- only the instance changes.
"""

import copy
import math


def apply_spec(base_instance, spec):
    """
    Apply a ConstraintSpec to the base instance.

    Parameters
    ----------
    base_instance : dict  -- full 20-well instance loaded from snorre_instance.json
    spec          : ConstraintSpec

    Returns
    -------
    dict  -- a new instance (deep copy) ready for solve_well_workover / WellWorkoverGA
    """
    inst = copy.deepcopy(base_instance)
    wells = inst['wells']

    # 1. Filter to sub_blocks
    if spec.sub_blocks:
        wells = [w for w in wells
                 if any(w['name'].startswith(b) for b in spec.sub_blocks)]

    # 2. Filter by min_flow_rate
    if spec.min_flow_rate is not None:
        wells = [w for w in wells if w['flow_rate'] >= spec.min_flow_rate]

    # 3. Top-N wells by DST rate
    if spec.n_wells is not None:
        wells = sorted(wells, key=lambda w: w['flow_rate'], reverse=True)
        wells = wells[:spec.n_wells]

    if len(wells) < 2:
        raise ValueError(
            "Spec produced only {} well(s) -- need at least 2.".format(len(wells)))

    # 4. Re-index well IDs and rebuild mob matrix from original UTM positions
    old_names = [w['name'] for w in wells]
    for i, w in enumerate(wells):
        w['id'] = i

    # Rebuild mob matrix using original UTM coordinates stored in each well
    n = len(wells)
    inst['wells']   = wells
    inst['n_wells'] = n

    # Rebuild mob matrix only if wells were filtered (indices changed)
    if n < base_instance['n_wells']:
        mob = [[0] * n for _ in range(n)]
        name_to_base_idx = base_instance.get('well_name_to_id', {})
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                wi, wj = wells[i], wells[j]
                if 'utm_x' in wi and 'utm_x' in wj:
                    dx = wi['utm_x'] - wj['utm_x']
                    dy = wi['utm_y'] - wj['utm_y']
                    dist_km = math.sqrt(dx*dx + dy*dy) / 1000.0
                    mob[i][j] = max(1, math.ceil(dist_km / 5.0))
                else:
                    bi = name_to_base_idx.get(wi['name'], i)
                    bj = name_to_base_idx.get(wj['name'], j)
                    mob[i][j] = base_instance['mob_time'][bi][bj]
        inst['mob_time'] = mob
    # else: mob_time already correct from deep copy

    # 5. Rig count
    if spec.n_rigs is not None:
        inst['rigs']   = inst['rigs'][:spec.n_rigs]
        inst['n_rigs'] = spec.n_rigs

    # 6. Rig cluster lock -- store start_cluster hint (Gantt use only)
    if spec.rig_cluster_lock:
        for rig in inst['rigs']:
            if rig['name'] in spec.rig_cluster_lock:
                rig['start_cluster'] = spec.rig_cluster_lock[rig['name']]

    # 7. Max makespan cap -- store for use by the sweep in main.py
    if spec.max_makespan_days is not None:
        inst['max_makespan_days'] = spec.max_makespan_days

    # 8. Audit trail
    inst['llm_spec'] = spec.model_dump()

    return inst


def describe_instance(inst):
    """One-line human-readable summary of an instance."""
    blocks = sorted(set(
        w['name'].rsplit('-', 1)[0] for w in inst['wells']
    ))
    max_ms = inst.get('max_makespan_days')
    deadline_str = "  |  deadline: {}d".format(max_ms) if max_ms else ""
    return "{} wells, {} rigs, blocks: {}{}".format(
        inst['n_wells'], inst['n_rigs'], ', '.join(blocks), deadline_str
    )
