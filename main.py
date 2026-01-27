#!/usr/bin/env python3
"""
Food Waste Optimization Analysis
Lawrenceville School

Main script for running food waste optimization models and generating reports.

This script combines Markov Chain and SIR models to:
1. Analyze current food waste behavior patterns
2. Model the spread of food waste awareness
3. Compare intervention strategies
4. Project savings from waste reduction
5. Generate visualizations and reports

Usage:
    python main.py [--generate-data] [--no-plots] [--output-dir OUTPUT_DIR]
"""

import argparse
import os
import sys
from datetime import datetime

import numpy as np

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from models.markov_chain import MarkovChainModel
from models.sir_model import SIRModel, SIRParameters
from utils.data_generator import FoodWasteDataGenerator
from utils.visualization import FoodWasteVisualizer


def print_header(title: str) -> None:
    """Print a formatted section header."""
    print("\n" + "=" * 70)
    print(f" {title}")
    print("=" * 70)


def run_markov_analysis(output_dir: str, show_plots: bool = True) -> dict:
    """
    Run Markov Chain model analysis.

    Args:
        output_dir: Directory to save outputs
        show_plots: Whether to display plots

    Returns:
        Dictionary with analysis results
    """
    print_header("MARKOV CHAIN MODEL ANALYSIS")
    print("\nAnalyzing food waste behavior transitions...")

    model = MarkovChainModel()

    # Print model summary
    print(model.get_model_summary())

    # Compare interventions
    print("\nRunning 52-week simulation for all interventions...")
    results = model.compare_interventions(n_steps=52)

    print("\n" + "-" * 50)
    print("INTERVENTION COMPARISON RESULTS")
    print("-" * 50)
    print(f"{'Intervention':<25} {'Initial':>10} {'Final':>10} {'Reduction':>12}")
    print("-" * 50)

    for intervention, data in results.items():
        print(f"{intervention.replace('_', ' ').title():<25} "
              f"{data['initial_waste_pct']:>10.1f}% "
              f"{data['final_waste_pct']:>10.1f}% "
              f"{data['waste_reduction_pct']:>11.1f}%")

    # Estimate savings for combined intervention
    print("\n" + "-" * 50)
    print("PROJECTED SAVINGS (Combined Intervention)")
    print("-" * 50)

    current_waste = results['baseline']['initial_waste_pct']
    projected_waste = results['combined']['final_waste_pct']

    savings = model.estimate_food_savings(
        n_students=820,
        daily_food_per_student_lbs=1.1,
        current_waste_pct=current_waste,
        projected_waste_pct=projected_waste,
        days_per_week=7
    )

    print(f"Weekly food saved: {savings['weekly_food_saved_lbs']:,.1f} lbs")
    print(f"Annual food saved: {savings['annual_food_saved_lbs']:,.1f} lbs")
    print(f"Weekly cost savings: ${savings['weekly_cost_savings']:,.2f}")
    print(f"Annual cost savings: ${savings['annual_cost_savings']:,.2f}")

    # Generate visualizations
    if show_plots:
        print("\nGenerating Markov Chain visualizations...")
        viz = FoodWasteVisualizer()

        # State evolution comparison
        fig1 = viz.plot_markov_intervention_comparison(
            results,
            metric='waste',
            title='Food Waste Reduction by Intervention Strategy',
            save_path=os.path.join(output_dir, 'markov_intervention_comparison.png')
        )

        # Transition matrix heatmap for combined intervention
        combined_matrix = model.get_intervention_matrix('combined')
        fig2 = viz.plot_transition_matrix_heatmap(
            combined_matrix,
            model.state_names,
            title='Transition Probabilities (Combined Intervention)',
            save_path=os.path.join(output_dir, 'markov_transition_matrix.png')
        )

        print(f"  Saved: markov_intervention_comparison.png")
        print(f"  Saved: markov_transition_matrix.png")

    return {
        'model': model,
        'results': results,
        'savings': savings
    }


def run_sir_analysis(output_dir: str, show_plots: bool = True) -> dict:
    """
    Run SIR model analysis.

    Args:
        output_dir: Directory to save outputs
        show_plots: Whether to display plots

    Returns:
        Dictionary with analysis results
    """
    print_header("SIR MODEL ANALYSIS")
    print("\nModeling the spread of food waste awareness...")

    model = SIRModel(total_population=820)

    # Print model summary
    print(model.get_model_summary())

    # Run baseline simulation
    print("\nRunning 52-week baseline simulation...")
    baseline_sim = model.simulate(t_max=52)

    # Run simulation with campaigns
    print("Running simulation with awareness campaigns...")
    campaign_params = SIRParameters(
        beta=0.3,
        gamma=0.05,
        campaign_boost=0.2
    )

    campaign_schedule = [
        (0, 4),    # Week 0-4: Start of year campaign
        (18, 22),  # Week 18-22: Second semester kickoff
        (36, 40),  # Week 36-40: Late year push
    ]

    campaign_sim = model.simulate(
        t_max=52,
        params=campaign_params,
        campaign_schedule=campaign_schedule
    )

    # Analyze optimal campaign timing
    print("\nFinding optimal campaign timing...")
    optimal = model.find_optimal_campaign_timing(
        campaign_duration=4,
        n_campaigns=4
    )

    print("\n" + "-" * 50)
    print("SIR MODEL RESULTS")
    print("-" * 50)
    print(f"\nBaseline (no campaigns):")
    print(f"  Final awareness: {baseline_sim['awareness_pct'][-1]:.1f}%")
    print(f"\nWith awareness campaigns:")
    print(f"  Final awareness: {campaign_sim['awareness_pct'][-1]:.1f}%")
    print(f"\nOptimal campaign strategy: {optimal['best_strategy'].replace('_', ' ').title()}")
    print(f"  Expected final awareness: {optimal['best_final_awareness']:.1f}%")

    print("\nRecommended campaign schedule:")
    for i, (start, end) in enumerate(optimal['best_schedule']):
        print(f"  Campaign {i+1}: Week {start:.0f} - Week {end:.0f}")

    # Estimate waste reduction from awareness
    waste_estimate = model.estimate_waste_reduction(campaign_sim)
    initial_waste = waste_estimate['average_waste_pct'][0]
    final_waste = waste_estimate['average_waste_pct'][-1]

    print(f"\nProjected waste reduction:")
    print(f"  Initial average waste: {initial_waste:.1f}%")
    print(f"  Final average waste: {final_waste:.1f}%")
    print(f"  Reduction: {initial_waste - final_waste:.1f} percentage points")

    # Generate visualizations
    if show_plots:
        print("\nGenerating SIR visualizations...")
        viz = FoodWasteVisualizer()

        # SIR dynamics
        fig1 = viz.plot_sir_dynamics(
            campaign_sim,
            title='SIR Model: Food Waste Awareness Spread (With Campaigns)',
            save_path=os.path.join(output_dir, 'sir_dynamics.png')
        )

        # Campaign comparison
        fig2 = viz.plot_sir_campaign_comparison(
            optimal,
            title='Awareness Campaign Timing Optimization',
            save_path=os.path.join(output_dir, 'sir_campaign_comparison.png')
        )

        # Waste reduction timeline
        fig3 = viz.plot_waste_reduction_timeline(
            waste_estimate,
            title='Projected Food Waste Reduction (SIR Model)',
            save_path=os.path.join(output_dir, 'sir_waste_reduction.png')
        )

        print(f"  Saved: sir_dynamics.png")
        print(f"  Saved: sir_campaign_comparison.png")
        print(f"  Saved: sir_waste_reduction.png")

    return {
        'model': model,
        'baseline_sim': baseline_sim,
        'campaign_sim': campaign_sim,
        'optimal_campaigns': optimal,
        'waste_estimate': waste_estimate
    }


def generate_sample_data(output_dir: str) -> None:
    """
    Generate sample food waste data.

    Args:
        output_dir: Directory to save data files
    """
    print_header("GENERATING SAMPLE DATA")
    print("\nCreating synthetic food waste data for Lawrenceville School...")

    generator = FoodWasteDataGenerator(seed=42)

    # Generate one semester of data
    start_date = datetime(2024, 9, 5)
    print(f"\nGenerating Fall 2024 semester data (18 weeks)...")

    # Without intervention
    data_baseline = generator.generate_semester_data(
        start_date,
        n_weeks=18,
        intervention_start_week=None
    )

    # With intervention
    data_intervention = generator.generate_semester_data(
        start_date,
        n_weeks=18,
        intervention_start_week=4,
        intervention_type='awareness'
    )

    # Save data
    data_dir = os.path.join(output_dir, '..', 'data')
    os.makedirs(data_dir, exist_ok=True)

    generator.export_to_csv(data_baseline, os.path.join(data_dir, 'baseline'))
    generator.export_to_csv(data_intervention, os.path.join(data_dir, 'intervention'))

    # Print statistics
    stats_baseline = generator.get_statistics_summary(data_baseline)
    stats_intervention = generator.get_statistics_summary(data_intervention)

    print("\n" + "-" * 50)
    print("DATA GENERATION SUMMARY")
    print("-" * 50)
    print(f"\nBaseline Data:")
    print(f"  Total meals: {stats_baseline['total_meals_served']:,}")
    print(f"  Total food taken: {stats_baseline['total_food_taken_lbs']:,.1f} lbs")
    print(f"  Total food wasted: {stats_baseline['total_food_wasted_lbs']:,.1f} lbs")
    print(f"  Overall waste rate: {stats_baseline['overall_waste_percentage']:.1f}%")

    print(f"\nWith Intervention:")
    print(f"  Total meals: {stats_intervention['total_meals_served']:,}")
    print(f"  Total food taken: {stats_intervention['total_food_taken_lbs']:,.1f} lbs")
    print(f"  Total food wasted: {stats_intervention['total_food_wasted_lbs']:,.1f} lbs")
    print(f"  Overall waste rate: {stats_intervention['overall_waste_percentage']:.1f}%")

    waste_reduction = (stats_baseline['total_food_wasted_lbs'] -
                       stats_intervention['total_food_wasted_lbs'])
    print(f"\nFood saved with intervention: {waste_reduction:,.1f} lbs")

    print(f"\nData files saved to: {data_dir}/")
    print("  - baseline_raw.csv")
    print("  - baseline_daily_summary.csv")
    print("  - baseline_weekly_summary.csv")
    print("  - intervention_raw.csv")
    print("  - intervention_daily_summary.csv")
    print("  - intervention_weekly_summary.csv")


def generate_comprehensive_report(
    markov_results: dict,
    sir_results: dict,
    output_dir: str
) -> None:
    """
    Generate a comprehensive analysis report.

    Args:
        markov_results: Results from Markov analysis
        sir_results: Results from SIR analysis
        output_dir: Directory to save report
    """
    print_header("GENERATING COMPREHENSIVE REPORT")

    report_path = os.path.join(output_dir, 'analysis_report.txt')

    with open(report_path, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("LAWRENCEVILLE SCHOOL FOOD WASTE OPTIMIZATION ANALYSIS\n")
        f.write("=" * 70 + "\n")
        f.write(f"\nReport generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

        f.write("\n" + "=" * 70 + "\n")
        f.write("EXECUTIVE SUMMARY\n")
        f.write("=" * 70 + "\n")

        baseline_waste = markov_results['results']['baseline']['initial_waste_pct']
        combined_waste = markov_results['results']['combined']['final_waste_pct']
        waste_reduction = baseline_waste - combined_waste

        f.write(f"""
This analysis uses two complementary mathematical models to optimize
food waste reduction at Lawrenceville School:

1. MARKOV CHAIN MODEL
   - Models individual behavior transitions between waste categories
   - Compares effectiveness of different intervention strategies
   - Projects long-term behavior changes

2. SIR (SUSCEPTIBLE-INFORMED-REFORMED) MODEL
   - Models spread of food waste awareness through social influence
   - Optimizes timing of awareness campaigns
   - Predicts community-wide adoption rates

KEY FINDINGS:
- Current estimated average food waste: {baseline_waste:.1f}%
- Projected waste with combined interventions: {combined_waste:.1f}%
- Expected reduction: {waste_reduction:.1f} percentage points ({waste_reduction/baseline_waste*100:.1f}% improvement)
- Estimated annual food savings: {markov_results['savings']['annual_food_saved_lbs']:,.0f} lbs
- Estimated annual cost savings: ${markov_results['savings']['annual_cost_savings']:,.0f}

RECOMMENDED ACTIONS:
1. Implement combined intervention strategy (awareness + portion control +
   peer pressure + incentives)
2. Schedule awareness campaigns at semester starts
3. Track student behavior transitions weekly
4. Target initial 50% awareness threshold for self-sustaining change
""")

        f.write("\n" + "=" * 70 + "\n")
        f.write("DETAILED MARKOV CHAIN ANALYSIS\n")
        f.write("=" * 70 + "\n")

        f.write("\nIntervention Comparison (52-week projection):\n")
        f.write("-" * 50 + "\n")
        f.write(f"{'Intervention':<25} {'Final Waste':>12} {'Reduction':>12}\n")
        f.write("-" * 50 + "\n")

        for intervention, data in markov_results['results'].items():
            f.write(f"{intervention.replace('_', ' ').title():<25} "
                   f"{data['final_waste_pct']:>11.1f}% "
                   f"{data['waste_reduction_pct']:>11.1f}%\n")

        f.write("\n" + "=" * 70 + "\n")
        f.write("DETAILED SIR MODEL ANALYSIS\n")
        f.write("=" * 70 + "\n")

        r0 = sir_results['model'].compute_basic_reproduction_number()
        threshold = sir_results['model'].compute_herd_immunity_threshold()

        f.write(f"""
Basic Reproduction Number (R0): {r0:.2f}
  - R0 > 1 means awareness will spread
  - Current R0 indicates {'positive' if r0 > 1 else 'negative'} outlook

Self-Sustaining Threshold: {threshold*100:.1f}% of students aware
  - Once this threshold is reached, awareness becomes self-sustaining
  - Campaigns should focus on reaching this critical mass

Optimal Campaign Strategy: {sir_results['optimal_campaigns']['best_strategy'].replace('_', ' ').title()}
  - Expected final awareness: {sir_results['optimal_campaigns']['best_final_awareness']:.1f}%

Recommended Campaign Schedule:
""")
        for i, (start, end) in enumerate(sir_results['optimal_campaigns']['best_schedule']):
            f.write(f"  Campaign {i+1}: Week {start:.0f} to Week {end:.0f}\n")

        f.write("\n" + "=" * 70 + "\n")
        f.write("RECOMMENDATIONS FOR IMPLEMENTATION\n")
        f.write("=" * 70 + "\n")

        f.write("""
PHASE 1: BASELINE MEASUREMENT (Weeks 1-2)
- Install food waste tracking systems
- Establish current waste baseline
- Identify high-waste meal types and times

PHASE 2: AWARENESS CAMPAIGN (Weeks 3-6)
- Launch educational posters and announcements
- Share waste statistics with students
- Partner with environmental clubs

PHASE 3: STRUCTURAL INTERVENTIONS (Weeks 7-10)
- Introduce smaller plate sizes
- Add portion guidance signage
- Implement trayless dining trials

PHASE 4: SOCIAL INCENTIVES (Weeks 11+)
- Create inter-house waste reduction competitions
- Recognize low-waste students/tables
- Share progress updates regularly

MONITORING:
- Track waste weekly using Markov state categories
- Monitor awareness spread using surveys
- Adjust interventions based on model predictions
""")

        f.write("\n" + "=" * 70 + "\n")
        f.write("END OF REPORT\n")
        f.write("=" * 70 + "\n")

    print(f"Report saved to: {report_path}")


def main():
    """Main entry point for the analysis."""
    parser = argparse.ArgumentParser(
        description='Lawrenceville School Food Waste Optimization Analysis'
    )
    parser.add_argument(
        '--generate-data',
        action='store_true',
        help='Generate sample food waste data'
    )
    parser.add_argument(
        '--no-plots',
        action='store_true',
        help='Skip plot generation'
    )
    parser.add_argument(
        '--output-dir',
        default='outputs',
        help='Directory for output files (default: outputs)'
    )

    args = parser.parse_args()

    # Create output directory
    output_dir = os.path.join(os.path.dirname(__file__), args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    print("\n" + "=" * 70)
    print(" LAWRENCEVILLE SCHOOL FOOD WASTE OPTIMIZATION")
    print(" Using Markov Chain and SIR Models")
    print("=" * 70)
    print(f"\nOutput directory: {output_dir}")

    # Generate sample data if requested
    if args.generate_data:
        generate_sample_data(output_dir)

    # Run analyses
    show_plots = not args.no_plots
    markov_results = run_markov_analysis(output_dir, show_plots)
    sir_results = run_sir_analysis(output_dir, show_plots)

    # Generate comprehensive dashboard
    if show_plots:
        print_header("GENERATING DASHBOARD")
        viz = FoodWasteVisualizer()
        fig = viz.create_dashboard(
            markov_results['results'],
            sir_results['campaign_sim'],
            save_path=os.path.join(output_dir, 'dashboard.png')
        )
        print("Dashboard saved to: dashboard.png")

    # Generate report
    generate_comprehensive_report(markov_results, sir_results, output_dir)

    print_header("ANALYSIS COMPLETE")
    print(f"\nAll outputs saved to: {output_dir}/")
    print("\nFiles generated:")
    for f in sorted(os.listdir(output_dir)):
        print(f"  - {f}")


if __name__ == "__main__":
    main()
