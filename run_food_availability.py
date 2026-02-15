#!/usr/bin/env python3
"""
Generate food availability model plots.

Usage:
    python run_food_availability.py

Saves plots to outputs/ directory.
"""

import os
import sys
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving files

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from models.food_availability import FoodAvailabilityModel, run_baseline_and_optimize
from utils.visualization import FoodWasteVisualizer

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'outputs')
os.makedirs(OUTPUT_DIR, exist_ok=True)


def main():
    # Run full analysis
    results = run_baseline_and_optimize(verbose=True)

    model = results['model']
    viz = FoodWasteVisualizer()

    # 1. Baseline dynamics (F, E, P, S over time)
    baseline_sim = results['baseline_sim']
    viz.plot_food_availability_dynamics(
        baseline_sim,
        title="Baseline Food Availability Dynamics",
        save_path=os.path.join(OUTPUT_DIR, 'baseline_dynamics.png'),
    )
    print(f"\nSaved: {OUTPUT_DIR}/baseline_dynamics.png")

    # 2. Waste rate analysis (dP/dt over time, critical intervention point)
    waste_rate = model.compute_waste_rate()
    print("\n" + "=" * 70)
    print(" CALCULUS OPTIMIZATION — WASTE RATE ANALYSIS")
    print("=" * 70)
    print(f"  Maximum plate waste rate:   {waste_rate['max_dP_dt_per_min']:.2f} lbs/min")
    print(f"  Critical time (t_crit):     {waste_rate['t_critical_min']:.1f} minutes")
    print(f"  Students at t_crit:         {waste_rate['S_at_critical']:.0f}")
    print(f"  Food available at t_crit:   {waste_rate['F_at_critical']:.3f} lbs/student")
    print()
    print("  Interpretation:")
    print(f"    Plate waste accumulates fastest at t = {waste_rate['t_critical_min']:.1f} min.")
    print(f"    This is the critical intervention window.")
    print(f"    Strategies targeting this time would have maximum impact.")

    viz.plot_waste_rate(
        waste_rate,
        title="Waste Rate Analysis — Critical Intervention Point",
        save_path=os.path.join(OUTPUT_DIR, 'waste_rate_analysis.png'),
    )
    print(f"\nSaved: {OUTPUT_DIR}/waste_rate_analysis.png")

    # 3. Baseline vs best optimized (full strategy)
    best = results['optimized']['full']
    opt_sim = model.simulate(best['optimal_params'])
    viz.plot_baseline_vs_optimized(
        baseline_sim,
        opt_sim,
        strategy_name="Full Optimization",
        save_path=os.path.join(OUTPUT_DIR, 'baseline_vs_optimized.png'),
    )
    print(f"Saved: {OUTPUT_DIR}/baseline_vs_optimized.png")

    # 4. Sensitivity analysis
    viz.plot_sensitivity(
        results['sensitivity'],
        save_path=os.path.join(OUTPUT_DIR, 'sensitivity_analysis.png'),
    )
    print(f"Saved: {OUTPUT_DIR}/sensitivity_analysis.png")

    print(f"\nAll plots saved to {OUTPUT_DIR}/")


if __name__ == '__main__':
    main()
