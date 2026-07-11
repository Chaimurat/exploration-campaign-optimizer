"""
well_ga.py
----------
Genetic Algorithm for multi-objective well workover scheduling.

Chromosome
----------
A permutation of well IDs (length n_wells, each appearing exactly once).
This differs from the JSSP GA where each job appeared num_machines times.
Standard Order Crossover (OX) applies directly here.

Decode
------
Read chromosome left-to-right. Assign each well to whichever rig can
start it earliest, accounting for mobilisation from that rig's last well.
This always produces a feasible schedule.

Objectives
----------
  makespan           = max end time across all rigs
  deferred_production = sum_w( start[w] * flow_rate[w] )   [bbl-days]
  deferred_cost       = deferred_production * oil_price      [USD]
"""

import numpy as np


class WellWorkoverGA:

    def __init__(self, instance, pop_size=80, mutation_rate=0.2):
        self.n_wells       = instance['n_wells']
        self.n_rigs        = instance['n_rigs']
        self.wells         = instance['wells']
        self.mob_time      = instance['mob_time']
        self.oil_price     = instance['oil_price']
        self.pop_size      = pop_size
        self.mutation_rate = mutation_rate

    # ------------------------------------------------------------------
    # Chromosome generation
    # ------------------------------------------------------------------
    def generate_chromosome(self):
        """Random permutation of well IDs."""
        return np.random.permutation(self.n_wells)

    # ------------------------------------------------------------------
    # Decoder: permutation -> schedule
    # ------------------------------------------------------------------
    def decode(self, chromosome):
        """
        Greedy earliest-start decoder.

        Returns
        -------
        start_times : np.ndarray  shape (n_wells,)
        end_times   : np.ndarray  shape (n_wells,)
        rig_assign  : list[int]   rig index for each well
        """
        rig_free   = np.zeros(self.n_rigs)      # when each rig next becomes free
        rig_last   = [-1] * self.n_rigs         # last well handled by each rig (-1 = idle)
        start_times = np.zeros(self.n_wells)
        rig_assign  = [-1] * self.n_wells

        for well_id in chromosome:
            duration   = self.wells[well_id]['duration']
            best_start = float('inf')
            best_rig   = 0

            for r in range(self.n_rigs):
                if rig_last[r] == -1:
                    # Rig is idle -- no mobilisation needed
                    possible_start = 0.0
                else:
                    mob = self.mob_time[rig_last[r]][well_id]
                    possible_start = rig_free[r] + mob

                if possible_start < best_start:
                    best_start = possible_start
                    best_rig   = r

            start_times[well_id]  = best_start
            rig_free[best_rig]    = best_start + duration
            rig_last[best_rig]    = well_id
            rig_assign[well_id]   = best_rig

        end_times = np.array([start_times[w] + self.wells[w]['duration']
                               for w in range(self.n_wells)])
        return start_times, end_times, rig_assign

    # ------------------------------------------------------------------
    # Fitness evaluation
    # ------------------------------------------------------------------
    def evaluate_fitness(self, chromosome):
        """Returns (makespan, deferred_production) as floats."""
        start_times, end_times, _ = self.decode(chromosome)

        makespan  = float(np.max(end_times))
        deferred  = float(sum(
            start_times[w] * self.wells[w]['flow_rate']
            for w in range(self.n_wells)
        ))
        return makespan, deferred

    # ------------------------------------------------------------------
    # Crossover: standard Order Crossover (OX)
    # Works directly because each well ID appears exactly once.
    # ------------------------------------------------------------------
    def crossover(self, parent1, parent2):
        size       = len(parent1)
        cut1, cut2 = sorted(np.random.choice(size, 2, replace=False))

        child    = np.full(size, -1, dtype=int)
        child[cut1:cut2] = parent1[cut1:cut2]

        in_child = set(child[cut1:cut2].tolist())
        fill     = [g for g in parent2 if g not in in_child]

        positions = list(range(cut2, size)) + list(range(0, cut1))
        for pos, gene in zip(positions, fill):
            child[pos] = gene

        return child

    # ------------------------------------------------------------------
    # Mutation: swap two random positions
    # ------------------------------------------------------------------
    def mutate(self, chromosome):
        mutated = chromosome.copy()
        if np.random.rand() < self.mutation_rate:
            i, j = np.random.choice(len(mutated), 2, replace=False)
            mutated[i], mutated[j] = mutated[j], mutated[i]
        return mutated

    # ------------------------------------------------------------------
    # Evolution loop
    # ------------------------------------------------------------------
    def run_evolution(self, generations=200, w_makespan=0.5, w_deferred=0.5):
        """
        Evolve population for a given objective weighting.

        Parameters
        ----------
        w_makespan   weight for makespan objective   (0-1)
        w_deferred   weight for deferred production  (0-1)
                     w_makespan + w_deferred should sum to 1

        Returns
        -------
        dict with keys: makespan, deferred_production, deferred_cost,
                        schedule (list of dicts for Gantt chart)
        """
        population  = [self.generate_chromosome() for _ in range(self.pop_size)]
        best_score  = (float('inf'), float('inf'))
        best_chrom  = population[0]

        # Normalise objectives using rough scale estimates so weights are meaningful
        # makespan is ~40-100 days; deferred is ~100k-500k bbl-days
        # We normalise by the first-generation estimates
        norm_ms  = None
        norm_def = None

        for gen in range(generations):
            scores = [self.evaluate_fitness(c) for c in population]

            # Set normalisation from generation 0
            if gen == 0:
                ms_vals   = [s[0] for s in scores]
                def_vals  = [s[1] for s in scores]
                norm_ms   = max(ms_vals)  if max(ms_vals)  > 0 else 1.0
                norm_def  = max(def_vals) if max(def_vals) > 0 else 1.0

            for i, score in enumerate(scores):
                norm_score = (score[0] / norm_ms, score[1] / norm_def)
                weighted   = w_makespan * norm_score[0] + w_deferred * norm_score[1]

                best_norm  = (best_score[0] / norm_ms, best_score[1] / norm_def)
                best_w     = w_makespan * best_norm[0] + w_deferred * best_norm[1]

                if weighted < best_w:
                    best_score = score
                    best_chrom = population[i].copy()

            # Selection: keep top 50% by normalised weighted score
            w_scores = [(w_makespan * s[0]/norm_ms + w_deferred * s[1]/norm_def)
                        for s in scores]
            survivors = [population[idx]
                         for idx in np.argsort(w_scores)[:self.pop_size // 2]]

            # Breed next generation
            new_pop = []
            while len(new_pop) < self.pop_size:
                p1 = survivors[np.random.choice(len(survivors))]
                p2 = survivors[np.random.choice(len(survivors))]
                child = self.mutate(self.crossover(p1, p2))
                new_pop.append(child)

            population = new_pop

        # Decode best chromosome into a full schedule for Gantt output
        start_times, end_times, rig_assign = self.decode(best_chrom)
        schedule = []
        for w in range(self.n_wells):
            r = rig_assign[w]
            schedule.append({
                'well':      w,
                'well_name': self.wells[w]['name'],
                'rig':       r,
                'rig_name':  'Rig-{}'.format(['Alpha', 'Beta', 'Gamma', 'Delta'][r]),
                'start':     float(start_times[w]),
                'end':       float(end_times[w]),
                'duration':  self.wells[w]['duration'],
                'flow_rate': self.wells[w]['flow_rate']
            })

        return {
            'makespan':            best_score[0],
            'deferred_production': best_score[1],
            'deferred_cost':       best_score[1] * self.oil_price,
            'schedule':            schedule
        }
