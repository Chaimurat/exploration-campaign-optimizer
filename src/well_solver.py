"""
well_solver.py
--------------
CP-SAT exact solver for multi-objective well workover scheduling.

Problem
-------
- n wells, each needing one rig for a fixed duration
- m rigs available in parallel
- Sequence-dependent mobilisation times: moving from well i to well j
  takes mob_time[i][j] days
- Objective 1: Makespan  (total campaign duration)
- Objective 2: Deferred production = sum_w(start[w] * flow_rate[w])
  multiplied by oil_price to get USD

Method: epsilon-constraint
--------------------------
Pass max_makespan_limit=None  -> minimise makespan (find minimum C*)
Pass max_makespan_limit=k     -> fix makespan <= k, minimise deferred production

Sequence constraints use CP-SAT's AddCircuit, which is the standard
pattern for routing / sequencing with setup times.
"""

from ortools.sat.python import cp_model


def solve_well_workover(instance, max_makespan_limit=None, time_limit=30.0):
    model = cp_model.CpModel()

    n_wells   = instance['n_wells']
    n_rigs    = instance['n_rigs']
    wells     = instance['wells']
    mob_time  = instance['mob_time']
    oil_price = instance['oil_price']

    # Worst-case horizon: all wells in series with max mobilisation gaps
    horizon = (sum(w['duration'] for w in wells)
               + n_wells * max(mob_time[i][j]
                               for i in range(n_wells)
                               for j in range(n_wells) if i != j))

    # ------------------------------------------------------------------
    # Decision variables
    # ------------------------------------------------------------------
    # start[w], end[w]: when well w begins and finishes workover
    start = [model.NewIntVar(0, horizon, 'start_{}'.format(w))
             for w in range(n_wells)]
    end   = [model.NewIntVar(0, horizon, 'end_{}'.format(w))
             for w in range(n_wells)]

    for w in range(n_wells):
        model.Add(end[w] == start[w] + wells[w]['duration'])

    # assigned[w][r]: Boolean -- rig r handles well w
    assigned = [[model.NewBoolVar('asgn_{}_{}'.format(w, r))
                 for r in range(n_rigs)]
                for w in range(n_wells)]

    # Each well is assigned to exactly one rig
    for w in range(n_wells):
        model.AddExactlyOne(assigned[w])

    # ------------------------------------------------------------------
    # Sequencing with AddCircuit (one circuit per rig)
    # Node 0 = depot, nodes 1..n_wells = wells
    #
    # Self-loop on well w  ->  well NOT on this rig
    # Depot self-loop      ->  rig handles zero wells (idle rig)
    # ------------------------------------------------------------------
    for r in range(n_rigs):
        arcs = []

        # Depot can loop to itself (idle rig)
        depot_idle = model.NewBoolVar('depot_idle_{}'.format(r))
        arcs.append((0, 0, depot_idle))

        for w in range(n_wells):
            # Self-loop = well w NOT on rig r
            skip = model.NewBoolVar('skip_{}_{}'.format(r, w))
            arcs.append((w + 1, w + 1, skip))
            model.Add(assigned[w][r] == skip.Not())

            # Depot -> well w  (w is first on rig r)
            head_arc = model.NewBoolVar('head_{}_{}'.format(r, w))
            arcs.append((0, w + 1, head_arc))

            # Well w -> depot  (w is last on rig r)
            tail_arc = model.NewBoolVar('tail_{}_{}'.format(r, w))
            arcs.append((w + 1, 0, tail_arc))

            # Well w -> well v  (w directly precedes v on rig r)
            for v in range(n_wells):
                if v == w:
                    continue
                seq_arc = model.NewBoolVar('seq_{}_{}_{}'.format(r, w, v))
                arcs.append((w + 1, v + 1, seq_arc))
                # Mobilisation: v can't start until w finishes AND rig travels
                model.Add(
                    start[v] >= end[w] + mob_time[w][v]
                ).OnlyEnforceIf(seq_arc)

        model.AddCircuit(arcs)

    # ------------------------------------------------------------------
    # Objective 1: Makespan
    # ------------------------------------------------------------------
    makespan = model.NewIntVar(0, horizon, 'makespan')
    model.AddMaxEquality(makespan, end)

    # ------------------------------------------------------------------
    # Objective 2: Deferred production (bbl-days)
    # deferred = sum_w( start[w] * flow_rate[w] )
    # Multiply by oil_price when reporting (keeps model integers small)
    # ------------------------------------------------------------------
    max_flow  = max(w['flow_rate'] for w in wells)
    sum_flow  = sum(w['flow_rate'] for w in wells)

    deferred_terms = []
    for w in range(n_wells):
        t = model.NewIntVar(0, horizon * max_flow, 'def_{}'.format(w))
        model.Add(t == wells[w]['flow_rate'] * start[w])
        deferred_terms.append(t)

    total_deferred = model.NewIntVar(0, horizon * sum_flow, 'total_deferred')
    model.Add(total_deferred == sum(deferred_terms))

    # ------------------------------------------------------------------
    # Epsilon-constraint logic
    # ------------------------------------------------------------------
    if max_makespan_limit is not None:
        model.Add(makespan <= max_makespan_limit)
        model.Minimize(total_deferred)
    else:
        model.Minimize(makespan)

    # ------------------------------------------------------------------
    # Solve
    # ------------------------------------------------------------------
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit
    status = solver.Solve(model)

    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        schedule = []
        for w in range(n_wells):
            rig_id = next(r for r in range(n_rigs)
                          if solver.Value(assigned[w][r]) == 1)
            schedule.append({
                'well':      w,
                'well_name': wells[w]['name'],
                'rig':       rig_id,
                'rig_name':  instance['rigs'][rig_id]['name'],
                'start':     solver.Value(start[w]),
                'end':       solver.Value(end[w]),
                'duration':  wells[w]['duration'],
                'flow_rate': wells[w]['flow_rate']
            })

        ms_val  = solver.Value(makespan)
        def_val = solver.Value(total_deferred)

        return {
            'makespan':            ms_val,
            'deferred_production': def_val,               # bbl-days
            'deferred_cost':       def_val * oil_price,   # USD
            'schedule':            schedule,
            'proven_optimal':      status == cp_model.OPTIMAL
        }

    return None
