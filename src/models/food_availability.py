"""
Food Availability Model with Waste Minimization
Lawrenceville School Food Waste Optimization Project

Models the dynamics of food service using coupled ODEs based on:
    dF/dt = -(alpha/N) * S(t) * F(t)                         (food depletion)
    dE/dt = (1-w) * alpha * S(t) * F(t) - beta * E(t)        (food being eaten)
    dW/dt = w * alpha * S(t) * F(t) + beta * E(t)            (total outflow)

Where W(t) in the original formulation captures ALL food that has left the
system (both consumed and wasted). For optimization we decompose W into:
    P(t) = plate waste   (the w fraction that is never eaten)
    C(t) = consumed food (the (1-w) fraction, after eating time elapses)
    U(t) = unused food remaining on the line = F(t)*N

True waste = P(T) + U(T)   (plate waste + food never served)
We minimize this.

State Variables:
    F(t) - food available per student [lbs/student]
    E(t) - food currently being eaten [lbs]
    P(t) - cumulative plate waste [lbs]
    C(t) - cumulative consumed food [lbs]
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize, differential_evolution
from dataclasses import dataclass
from typing import Dict, Optional, Tuple


@dataclass
class FoodAvailabilityParams:
    """
    Parameters for the food availability ODE model.

    Attributes:
        N: Total number of students
        alpha: Service efficiency rate [1/hour]
        beta: Eating rate [1/hour] (inverse of average eating time)
        waste_fraction: Fraction of taken food that becomes plate waste
        F0: Initial food per student [lbs/student]
        T: Total service period [hours]
        student_distribution: (peak_frac, mid_frac, late_frac) of students
        period_durations: Duration of each arrival period [hours]
    """
    N: int = 700
    alpha: float = 6.0
    beta: float = 2.73
    waste_fraction: float = 0.30
    F0: float = 1.3
    T: float = 2.0
    student_distribution: Tuple[float, float, float] = (0.80, 0.15, 0.05)
    period_durations: Tuple[float, float, float] = (0.333, 0.334, 0.333)

    @property
    def total_food(self) -> float:
        return self.F0 * self.N

    def S(self, t: float) -> float:
        """Student demand function S(t)."""
        d = self.period_durations
        fracs = self.student_distribution
        if t < d[0]:
            return fracs[0] * self.N
        elif t < d[0] + d[1]:
            return fracs[1] * self.N
        elif t < d[0] + d[1] + d[2]:
            return fracs[2] * self.N
        else:
            return 0.0


class FoodAvailabilityModel:
    """
    Solves the food availability ODE system and optimizes parameters
    to minimize true food waste (plate waste + unused food).

    The 4-state ODE system:
        dF/dt = -(alpha/N) * S(t) * F(t)
        dE/dt = (1-w) * alpha * S(t) * F(t)  -  beta * E(t)
        dP/dt = w * alpha * S(t) * F(t)               [plate waste]
        dC/dt = beta * E(t)                            [consumed]

    Mass balance: F(t)*N + E(t) + P(t) + C(t) = F(0)*N  for all t.
    True waste = P(T) + F(T)*N + E(T)
               = total_food - C(T)
    We minimize true waste, equivalently maximize food actually consumed.
    """

    def __init__(self, params: Optional[FoodAvailabilityParams] = None):
        self.params = params or FoodAvailabilityParams()

    def _ode_system(self, t: float, y: np.ndarray,
                    params: FoodAvailabilityParams) -> list:
        """Right-hand side of the 4-state ODE system."""
        F, E, P, C = y
        S_t = params.S(t)
        alpha = params.alpha
        N = params.N
        beta = params.beta
        w = params.waste_fraction

        food_taking_rate = alpha * S_t * F  # total lbs/hr taken from line
        # Note: dF/dt uses alpha/N so units work for per-student F

        dF_dt = -(alpha / N) * S_t * F
        dE_dt = (1 - w) * food_taking_rate - beta * E
        dP_dt = w * food_taking_rate            # plate waste
        dC_dt = beta * E                        # consumed (finished eating)

        return [dF_dt, dE_dt, dP_dt, dC_dt]

    def simulate(self, params: Optional[FoodAvailabilityParams] = None,
                 n_points: int = 1000) -> Dict[str, np.ndarray]:
        """
        Simulate the ODE system forward in time.

        Returns dict with keys: 't', 'F', 'E', 'P', 'C', 'W', 'S',
            'true_waste', 'unused_food'.
        """
        p = params or self.params
        y0 = [p.F0, 0.0, 0.0, 0.0]  # F, E, P, C

        sol = solve_ivp(
            fun=lambda t, y: self._ode_system(t, y, p),
            t_span=(0, p.T),
            y0=y0,
            method='RK45',
            t_eval=np.linspace(0, p.T, n_points),
            rtol=1e-8,
            atol=1e-10,
        )

        F = sol.y[0]
        E = sol.y[1]
        P = sol.y[2]
        C = sol.y[3]
        S_vals = np.array([p.S(t) for t in sol.t])

        # W(t) = P(t) + C(t), the original user-defined total outflow
        W = P + C
        # Unused food still on the serving line
        unused = F * p.N
        # True waste = plate waste + unused food + food still being eaten
        true_waste = P + unused + E

        return {
            't': sol.t,
            'F': F,
            'E': E,
            'P': P,           # plate waste only
            'C': C,           # consumed food
            'W': W,           # original W = P + C (total outflow)
            'S': S_vals,
            'true_waste': true_waste,  # P + unused + still-eating
            'unused_food': unused,
        }

    def compute_waste(self, params: Optional[FoodAvailabilityParams] = None) -> Dict[str, float]:
        """
        Compute all waste components at end of service.

        Returns:
            Dict with plate_waste, unused_food, still_eating,
            total_true_waste, consumed, total_food.
        """
        result = self.simulate(params, n_points=200)
        p = params or self.params
        return {
            'plate_waste': result['P'][-1],
            'unused_food': result['unused_food'][-1],
            'still_eating': result['E'][-1],
            'total_true_waste': result['true_waste'][-1],
            'consumed': result['C'][-1],
            'total_food': p.total_food,
        }

    def total_true_waste(self, params: Optional[FoodAvailabilityParams] = None) -> float:
        """
        Return total true waste at T:
            plate_waste + unused_food + food_still_being_eaten
        """
        w = self.compute_waste(params)
        return w['total_true_waste']

    # ------------------------------------------------------------------
    # Optimization: minimize true waste over controllable parameters
    # ------------------------------------------------------------------

    def _build_optimized_params(self, x: np.ndarray,
                                strategy: str) -> FoodAvailabilityParams:
        """Build params from optimization vector x for given strategy."""
        base = self.params
        if strategy == 'waste_fraction':
            return FoodAvailabilityParams(
                N=base.N, alpha=base.alpha, beta=base.beta,
                waste_fraction=x[0], F0=base.F0, T=base.T,
                student_distribution=base.student_distribution,
                period_durations=base.period_durations,
            )
        elif strategy == 'portion_control':
            return FoodAvailabilityParams(
                N=base.N, alpha=base.alpha, beta=base.beta,
                waste_fraction=base.waste_fraction, F0=x[0], T=base.T,
                student_distribution=base.student_distribution,
                period_durations=base.period_durations,
            )
        elif strategy == 'service_and_portions':
            return FoodAvailabilityParams(
                N=base.N, alpha=x[2], beta=base.beta,
                waste_fraction=x[1], F0=x[0], T=base.T,
                student_distribution=base.student_distribution,
                period_durations=base.period_durations,
            )
        elif strategy == 'full':
            peak = x[3]
            mid = x[4]
            late = max(1.0 - peak - mid, 0.0)
            return FoodAvailabilityParams(
                N=base.N, alpha=x[2], beta=base.beta,
                waste_fraction=x[1], F0=x[0], T=base.T,
                student_distribution=(peak, mid, late),
                period_durations=base.period_durations,
            )
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

    def _objective(self, x: np.ndarray, strategy: str) -> float:
        """Objective: total true waste (plate waste + unused + still eating)."""
        try:
            p = self._build_optimized_params(x, strategy)
            return self.total_true_waste(p)
        except Exception:
            return 1e6

    def optimize(self, strategy: str = 'full',
                 method: str = 'differential_evolution') -> Dict:
        """
        Minimize true food waste over controllable parameters.

        Strategies:
            'waste_fraction'       — plate waste fraction only (behavioral)
            'portion_control'      — initial food per student F0 only
            'service_and_portions' — F0, waste_fraction, alpha
            'full'                 — F0, waste_fraction, alpha, arrival distribution

        Returns:
            Dict with baseline/optimized waste breakdown, optimal params.
        """
        bounds_map = {
            'waste_fraction': [(0.05, 0.30)],
            'portion_control': [(0.5, 1.3)],
            'service_and_portions': [
                (0.5, 1.3),    # F0
                (0.05, 0.30),  # waste_fraction
                (3.0, 12.0),   # alpha
            ],
            'full': [
                (0.5, 1.3),    # F0
                (0.05, 0.30),  # waste_fraction
                (3.0, 12.0),   # alpha
                (0.3, 0.95),   # peak_frac
                (0.03, 0.5),   # mid_frac
            ],
        }

        bounds = bounds_map[strategy]
        baseline_waste = self.compute_waste()

        if method == 'differential_evolution':
            result = differential_evolution(
                self._objective,
                bounds=bounds,
                args=(strategy,),
                seed=42,
                maxiter=200,
                tol=1e-8,
                polish=True,
            )
        else:
            x0_map = {
                'waste_fraction': [self.params.waste_fraction],
                'portion_control': [self.params.F0],
                'service_and_portions': [
                    self.params.F0,
                    self.params.waste_fraction,
                    self.params.alpha,
                ],
                'full': [
                    self.params.F0,
                    self.params.waste_fraction,
                    self.params.alpha,
                    self.params.student_distribution[0],
                    self.params.student_distribution[1],
                ],
            }
            constraints = []
            if strategy == 'full':
                constraints.append({
                    'type': 'ineq',
                    'fun': lambda x: 1.0 - x[3] - x[4]
                })
            result = minimize(
                self._objective,
                x0=x0_map[strategy],
                args=(strategy,),
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 1000, 'ftol': 1e-12},
            )

        opt_params = self._build_optimized_params(result.x, strategy)
        optimized_waste = self.compute_waste(opt_params)

        return {
            'strategy': strategy,
            'baseline': baseline_waste,
            'optimized': optimized_waste,
            'reduction_lbs': baseline_waste['total_true_waste'] - optimized_waste['total_true_waste'],
            'reduction_pct': (
                (baseline_waste['total_true_waste'] - optimized_waste['total_true_waste'])
                / baseline_waste['total_true_waste'] * 100
            ),
            'optimal_params': opt_params,
            'raw_result': result,
        }

    def optimize_all_strategies(self) -> Dict[str, Dict]:
        """Run optimization for every strategy and return comparison."""
        strategies = ['waste_fraction', 'portion_control',
                      'service_and_portions', 'full']
        results = {}
        for s in strategies:
            results[s] = self.optimize(strategy=s)
        return results

    # ------------------------------------------------------------------
    # Sensitivity analysis
    # ------------------------------------------------------------------

    def sensitivity_analysis(self, param_name: str,
                             values: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Sweep a single parameter and record true waste for each value.
        """
        wastes = []
        base = self.params
        for v in values:
            p = FoodAvailabilityParams(
                N=base.N,
                alpha=base.alpha if param_name != 'alpha' else v,
                beta=base.beta if param_name != 'beta' else v,
                waste_fraction=base.waste_fraction if param_name != 'waste_fraction' else v,
                F0=base.F0 if param_name != 'F0' else v,
                T=base.T,
                student_distribution=base.student_distribution,
                period_durations=base.period_durations,
            )
            wastes.append(self.total_true_waste(p))

        return {'values': values, 'waste': np.array(wastes)}


# ======================================================================
# Convenience runner
# ======================================================================

def run_baseline_and_optimize(verbose: bool = True) -> Dict:
    """
    Run baseline simulation and all optimization strategies.
    Print a detailed report if verbose.
    """
    model = FoodAvailabilityModel()

    # --- Baseline ---
    baseline_sim = model.simulate()
    bw = model.compute_waste()

    if verbose:
        print("=" * 70)
        print(" FOOD AVAILABILITY MODEL — BASELINE")
        print("=" * 70)
        print(f"  Total food prepared:       {model.params.total_food:.1f} lbs")
        print(f"  Total students:            {model.params.N}")
        print(f"  Service period:            {model.params.T} hours")
        print(f"  Plate waste fraction:      {model.params.waste_fraction:.0%}")
        print(f"  Service rate (alpha):      {model.params.alpha:.1f} /hr")
        print(f"  Eating rate (beta):        {model.params.beta:.2f} /hr")
        print()
        print("  Waste Breakdown at T:")
        print(f"    Plate waste (P):         {bw['plate_waste']:.2f} lbs")
        print(f"    Unused food on line:     {bw['unused_food']:.2f} lbs")
        print(f"    Still being eaten:       {bw['still_eating']:.2f} lbs")
        print(f"    ---")
        print(f"    Total true waste:        {bw['total_true_waste']:.2f} lbs")
        print(f"    Food consumed:           {bw['consumed']:.2f} lbs")
        print(f"    Waste % of prepared:     "
              f"{bw['total_true_waste'] / bw['total_food'] * 100:.1f}%")
        print()

    # --- Optimization ---
    all_results = model.optimize_all_strategies()

    if verbose:
        print("=" * 70)
        print(" OPTIMIZATION RESULTS — MINIMIZING TRUE WASTE")
        print("=" * 70)
        for name, res in all_results.items():
            opt_p = res['optimal_params']
            ow = res['optimized']
            print(f"\n  Strategy: {name}")
            print(f"    Baseline true waste:  {res['baseline']['total_true_waste']:.2f} lbs")
            print(f"    Optimized true waste: {ow['total_true_waste']:.2f} lbs")
            print(f"    Reduction:            {res['reduction_lbs']:.2f} lbs "
                  f"({res['reduction_pct']:.1f}%)")
            print(f"    — Plate waste:        {ow['plate_waste']:.2f} lbs")
            print(f"    — Unused food:        {ow['unused_food']:.2f} lbs")
            print(f"    — Still eating:       {ow['still_eating']:.2f} lbs")
            print(f"    — Consumed:           {ow['consumed']:.2f} lbs")
            print(f"    Optimal F0:           {opt_p.F0:.3f} lbs/student "
                  f"({opt_p.F0 * opt_p.N:.1f} lbs total)")
            print(f"    Optimal waste_frac:   {opt_p.waste_fraction:.3f}")
            print(f"    Optimal alpha:        {opt_p.alpha:.2f}")
            print(f"    Student dist:         "
                  f"({opt_p.student_distribution[0]:.2f}, "
                  f"{opt_p.student_distribution[1]:.2f}, "
                  f"{opt_p.student_distribution[2]:.2f})")

    # --- Sensitivity ---
    sensitivity = {}
    for param, rng in [('F0', np.linspace(0.5, 1.5, 50)),
                        ('waste_fraction', np.linspace(0.05, 0.40, 50)),
                        ('alpha', np.linspace(2.0, 12.0, 50))]:
        sensitivity[param] = model.sensitivity_analysis(param, rng)

    if verbose:
        print("\n" + "=" * 70)
        print(" SENSITIVITY ANALYSIS")
        print("=" * 70)
        for param, data in sensitivity.items():
            idx_min = np.argmin(data['waste'])
            print(f"  {param}: min waste = {data['waste'][idx_min]:.2f} lbs "
                  f"at {param} = {data['values'][idx_min]:.3f}")

    return {
        'baseline_sim': baseline_sim,
        'baseline_waste': bw,
        'optimized': all_results,
        'sensitivity': sensitivity,
        'model': model,
    }


if __name__ == '__main__':
    results = run_baseline_and_optimize(verbose=True)
