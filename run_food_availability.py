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

    # 1. Baseline dynamics (F, E, P, W over time)
    baseline_sim = results['baseline_sim']
    viz.plot_food_availability_dynamics(
        baseline_sim,
        title="Baseline Food Availability Dynamics",
        save_path=os.path.join(OUTPUT_DIR, 'baseline_dynamics.png'),
    )
    print(f"\nSaved: {OUTPUT_DIR}/baseline_dynamics.png")

    # 2. Baseline vs best optimized (full strategy)
    best = results['optimized']['full']
    opt_sim = model.simulate(best['optimal_params'])
    viz.plot_baseline_vs_optimized(
        baseline_sim,
        opt_sim,
        strategy_name="Full Optimization",
        save_path=os.path.join(OUTPUT_DIR, 'baseline_vs_optimized.png'),
    )
    print(f"Saved: {OUTPUT_DIR}/baseline_vs_optimized.png")

    # 3. Sensitivity analysis
    viz.plot_sensitivity(
        results['sensitivity'],
        save_path=os.path.join(OUTPUT_DIR, 'sensitivity_analysis.png'),
    )
    print(f"Saved: {OUTPUT_DIR}/sensitivity_analysis.png")

    print(f"\nAll plots saved to {OUTPUT_DIR}/")


if __name__ == '__main__':
    main()
