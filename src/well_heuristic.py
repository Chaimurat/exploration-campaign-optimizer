"""
well_heuristic.py
-----------------
Greedy baseline heuristics for well scheduling.

Fixed-ordering heuristics
-------------------------
  flow_rate_desc  -- highest DST rate first
  flow_rate_asc   -- lowest DST rate first
  duration_asc    -- shortest first  (SPT)
  duration_desc   -- longest first   (LPT)

Myopic greedy
-------------
  myopic  -- at each step, pick the (well, rig) pair that maximises
             flow_rate / (earliest_start + 1).
             Accounts for rig positions and mob times at every decision.
"""

STRATEGIES = {
    'flow_rate_desc': 'Greedy-Rate (rate↓)',
    'flow_rate_asc':  'Greedy-Rate (rate↑)',
    'duration_asc':   'Greedy-SPT (dur↑)',
    'duration_desc':  'Greedy-LPT (dur↓)',
    'myopic':         'Greedy-Myopic',
}


# ---------------------------------------------------------------------------
# Fixed-ordering greedy
# ---------------------------------------------------------------------------

def run_greedy_heuristic(instance, sort_by='flow_rate_desc'):
    wells     = instance['wells']
    n_wells   = instance['n_wells']
    n_rigs    = instance['n_rigs']
    mob_time  = instance['mob_time']
    oil_price = instance['oil_price']

    if sort_by == 'flow_rate_desc':
        order = sorted(range(n_wells), key=lambda w: -wells[w]['flow_rate'])
    elif sort_by == 'flow_rate_asc':
        order = sorted(range(n_wells), key=lambda w:  wells[w]['flow_rate'])
    elif sort_by == 'duration_asc':
        order = sorted(range(n_wells), key=lambda w:  wells[w]['duration'])
    elif sort_by == 'duration_desc':
        order = sorted(range(n_wells), key=lambda w: -wells[w]['duration'])
    else:
        raise ValueError("Unknown sort_by: {}".format(sort_by))

    rig_free    = [0.0] * n_rigs
    rig_last    = [-1]  * n_rigs
    start_times = [0.0] * n_wells
    rig_assign  = [-1]  * n_wells

    for well_id in order:
        best_start, best_rig = float('inf'), 0
        for r in range(n_rigs):
            mob = 0 if rig_last[r] == -1 else mob_time[rig_last[r]][well_id]
            t   = rig_free[r] + mob
            if t < best_start:
                best_start, best_rig = t, r

        start_times[well_id] = best_start
        rig_free[best_rig]   = best_start + wells[well_id]['duration']
        rig_last[best_rig]   = well_id
        rig_assign[well_id]  = best_rig

    return _build_result(instance, wells, n_wells, start_times, rig_free,
                         rig_assign, STRATEGIES[sort_by])


# ---------------------------------------------------------------------------
# Myopic greedy
# ---------------------------------------------------------------------------

def run_myopic_greedy(instance):
    """
    At each step, evaluate every (unscheduled well, rig) pair and pick the
    one that maximises  flow_rate / (earliest_start + 1).

    This accounts for current rig positions and mob times at every decision,
    unlike fixed-ordering heuristics which commit to an order upfront.
    """
    wells     = instance['wells']
    n_wells   = instance['n_wells']
    n_rigs    = instance['n_rigs']
    mob_time  = instance['mob_time']
    oil_price = instance['oil_price']

    rig_free    = [0.0] * n_rigs
    rig_last    = [-1]  * n_rigs
    start_times = [0.0] * n_wells
    rig_assign  = [-1]  * n_wells
    scheduled   = set()

    for _ in range(n_wells):
        best_score  = -1
        best_well   = -1
        best_rig    = -1
        best_start  = 0.0

        for w in range(n_wells):
            if w in scheduled:
                continue
            flow = wells[w]['flow_rate']

            for r in range(n_rigs):
                mob   = 0 if rig_last[r] == -1 else mob_time[rig_last[r]][w]
                start = rig_free[r] + mob
                score = flow / (start + 1)      # higher rate, earlier start = better

                if score > best_score:
                    best_score = score
                    best_well  = w
                    best_rig   = r
                    best_start = start

        start_times[best_well] = best_start
        rig_free[best_rig]     = best_start + wells[best_well]['duration']
        rig_last[best_rig]     = best_well
        rig_assign[best_well]  = best_rig
        scheduled.add(best_well)

    return _build_result(instance, wells, n_wells, start_times, rig_free,
                         rig_assign, STRATEGIES['myopic'])


# ---------------------------------------------------------------------------
# Shared result builder
# ---------------------------------------------------------------------------

def _build_result(instance, wells, n_wells, start_times, rig_free,
                  rig_assign, label):
    oil_price = instance['oil_price']
    makespan  = max(rig_free)
    deferred  = sum(start_times[w] * wells[w]['flow_rate']
                    for w in range(n_wells))
    schedule = []
    for w in range(n_wells):
        r = rig_assign[w]
        schedule.append({
            'well':      w,
            'well_name': wells[w]['name'],
            'rig':       r,
            'rig_name':  instance['rigs'][r]['name'],
            'start':     start_times[w],
            'end':       start_times[w] + wells[w]['duration'],
            'duration':  wells[w]['duration'],
            'flow_rate': wells[w]['flow_rate'],
        })
    return {
        'makespan':            makespan,
        'deferred_production': deferred,
        'deferred_cost':       deferred * oil_price,
        'schedule':            schedule,
        'label':               label,
    }


# ---------------------------------------------------------------------------
# Run all
# ---------------------------------------------------------------------------

def run_all_heuristics(instance):
    results = {k: run_greedy_heuristic(instance, k)
               for k in STRATEGIES if k != 'myopic'}
    results['myopic'] = run_myopic_greedy(instance)
    return results
