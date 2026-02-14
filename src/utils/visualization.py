"""
Visualization Tools for Food Waste Analysis
Lawrenceville School Food Waste Optimization Project

Provides comprehensive visualization capabilities for:
- Markov Chain state distributions
- SIR model dynamics
- Food waste trends and comparisons
- Intervention effectiveness
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from typing import Dict, List, Optional, Tuple
import warnings


# Set style for all plots
plt.style.use('seaborn-v0_8-whitegrid')


class FoodWasteVisualizer:
    """
    Visualization tools for food waste analysis at Lawrenceville School.

    Provides methods to visualize:
    - Markov chain state transitions and distributions
    - SIR model epidemic curves
    - Waste reduction over time
    - Intervention comparisons
    - Cost savings projections
    """

    def __init__(self, figsize: Tuple[int, int] = (10, 6)):
        """
        Initialize the visualizer.

        Args:
            figsize: Default figure size for plots
        """
        self.figsize = figsize
        self.colors = {
            'high_waste': '#e74c3c',      # Red
            'medium_waste': '#f39c12',    # Orange
            'low_waste': '#3498db',       # Blue
            'minimal_waste': '#27ae60',   # Green
            'susceptible': '#95a5a6',     # Gray
            'informed': '#e74c3c',        # Red
            'reformed': '#27ae60',        # Green
            'baseline': '#7f8c8d',        # Dark gray
            'intervention': '#2ecc71'     # Bright green
        }

    def plot_markov_state_evolution(
        self,
        history: np.ndarray,
        state_names: List[str],
        title: str = "Food Waste Behavior Distribution Over Time",
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot the evolution of state distribution over time (stacked area).

        Args:
            history: Array of shape (n_steps, n_states)
            state_names: Names of the states
            title: Plot title
            save_path: Optional path to save the figure

        Returns:
            matplotlib Figure object
        """
        fig, ax = plt.subplots(figsize=self.figsize)

        weeks = np.arange(len(history))
        colors = [
            self.colors['high_waste'],
            self.colors['medium_waste'],
            self.colors['low_waste'],
            self.colors['minimal_waste']
        ]

        ax.stackplot(weeks, history.T, labels=state_names, colors=colors, alpha=0.8)

        ax.set_xlabel('Weeks', fontsize=12)
        ax.set_ylabel('Proportion of Students', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(loc='upper right', fontsize=10)
        ax.set_xlim(0, len(history) - 1)
        ax.set_ylim(0, 1)

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight')

        return fig

    def plot_markov_intervention_comparison(
        self,
        results: Dict[str, Dict],
        metric: str = 'waste',
        title: str = "Intervention Comparison - Food Waste Reduction",
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Compare different intervention strategies.

        Args:
            results: Dictionary from MarkovChainModel.compare_interventions()
            metric: 'waste' or 'distribution'
            title: Plot title
            save_path: Optional path to save the figure

        Returns:
            matplotlib Figure object
        """
        fig, ax = plt.subplots(figsize=self.figsize)

        # Define colors for each intervention
        intervention_colors = {
            'baseline': '#7f8c8d',
            'awareness_campaign': '#3498db',
            'portion_control': '#9b59b6',
            'peer_pressure': '#e67e22',
            'incentive_program': '#1abc9c',
            'combined': '#27ae60'
        }

        for intervention, data in results.items():
            history = data['history']
            weeks = np.arange(len(history))

            if metric == 'waste':
                # Calculate expected waste for each time point
                waste_pcts = np.array([40.0, 22.0, 10.0, 3.0])
                values = np.dot(history, waste_pcts)
                ylabel = 'Average Food Waste (%)'
            else:
                # Plot minimal waste proportion
                values = history[:, 3]  # Minimal waste state
                ylabel = 'Proportion in Minimal Waste Category'

            color = intervention_colors.get(intervention, '#000000')
            label = intervention.replace('_', ' ').title()
            ax.plot(weeks, values, label=label, color=color, linewidth=2)

        ax.set_xlabel('Weeks', fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(loc='upper right' if metric != 'waste' else 'lower left', fontsize=10)
        ax.set_xlim(0, max(len(data['history']) for data in results.values()) - 1)

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight')

        return fig

    def plot_sir_dynamics(
        self,
        simulation: Dict[str, np.ndarray],
        title: str = "SIR Model: Food Waste Awareness Spread",
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot SIR model dynamics.

        Args:
            simulation: Output from SIRModel.simulate()
            title: Plot title
            save_path: Optional path to save the figure

        Returns:
            matplotlib Figure object
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        time = simulation['time']
        S = simulation['S']
        I = simulation['I']
        R = simulation['R']
        N = S + I + R

        # Left plot: Absolute numbers
        ax1 = axes[0]
        ax1.plot(time, S, label='Unaware (S)', color=self.colors['susceptible'], linewidth=2)
        ax1.plot(time, I, label='Active (I)', color=self.colors['informed'], linewidth=2)
        ax1.plot(time, R, label='Reformed (R)', color=self.colors['reformed'], linewidth=2)
        ax1.fill_between(time, 0, S, alpha=0.2, color=self.colors['susceptible'])
        ax1.fill_between(time, S, S + I, alpha=0.2, color=self.colors['informed'])
        ax1.fill_between(time, S + I, N, alpha=0.2, color=self.colors['reformed'])

        ax1.set_xlabel('Weeks', fontsize=12)
        ax1.set_ylabel('Number of Students', fontsize=12)
        ax1.set_title('Population Dynamics', fontsize=12, fontweight='bold')
        ax1.legend(loc='center right', fontsize=10)
        ax1.set_xlim(0, time[-1])
        ax1.set_ylim(0, N[0] * 1.05)

        # Right plot: Awareness percentage
        ax2 = axes[1]
        awareness_pct = (I + R) / N * 100
        reformed_pct = R / N * 100

        ax2.plot(time, awareness_pct, label='Total Aware (I+R)',
                 color='#2c3e50', linewidth=2)
        ax2.plot(time, reformed_pct, label='Reformed Only (R)',
                 color=self.colors['reformed'], linewidth=2, linestyle='--')

        ax2.axhline(y=50, color='gray', linestyle=':', alpha=0.7, label='50% Threshold')
        ax2.fill_between(time, 0, awareness_pct, alpha=0.2, color='#2c3e50')

        ax2.set_xlabel('Weeks', fontsize=12)
        ax2.set_ylabel('Percentage of Students (%)', fontsize=12)
        ax2.set_title('Awareness Progress', fontsize=12, fontweight='bold')
        ax2.legend(loc='lower right', fontsize=10)
        ax2.set_xlim(0, time[-1])
        ax2.set_ylim(0, 105)

        plt.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight')

        return fig

    def plot_sir_campaign_comparison(
        self,
        campaign_results: Dict,
        title: str = "Campaign Timing Optimization",
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Compare different campaign timing strategies.

        Args:
            campaign_results: Output from SIRModel.find_optimal_campaign_timing()
            title: Plot title
            save_path: Optional path to save the figure

        Returns:
            matplotlib Figure object
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        strategy_colors = {
            'evenly_spaced': '#3498db',
            'front_loaded': '#e74c3c',
            'semester_starts': '#27ae60'
        }

        # Left plot: Awareness curves
        ax1 = axes[0]
        for name, data in campaign_results['strategies'].items():
            sim = data['simulation']
            color = strategy_colors.get(name, '#000000')
            label = name.replace('_', ' ').title()
            ax1.plot(sim['time'], sim['awareness_pct'], label=label,
                    color=color, linewidth=2)

        ax1.axhline(y=50, color='gray', linestyle=':', alpha=0.7)
        ax1.set_xlabel('Weeks', fontsize=12)
        ax1.set_ylabel('Awareness (%)', fontsize=12)
        ax1.set_title('Awareness Spread by Strategy', fontsize=12, fontweight='bold')
        ax1.legend(loc='lower right', fontsize=10)
        ax1.set_xlim(0, sim['time'][-1])
        ax1.set_ylim(0, 100)

        # Right plot: Final metrics bar chart
        ax2 = axes[1]
        strategies = list(campaign_results['strategies'].keys())
        final_awareness = [campaign_results['strategies'][s]['final_awareness_pct']
                         for s in strategies]
        colors = [strategy_colors.get(s, '#000000') for s in strategies]

        bars = ax2.bar([s.replace('_', '\n').title() for s in strategies],
                      final_awareness, color=colors, alpha=0.8)

        # Highlight best strategy
        best_idx = strategies.index(campaign_results['best_strategy'])
        bars[best_idx].set_edgecolor('gold')
        bars[best_idx].set_linewidth(3)

        ax2.set_ylabel('Final Awareness (%)', fontsize=12)
        ax2.set_title('Final Awareness by Strategy', fontsize=12, fontweight='bold')
        ax2.set_ylim(0, 100)

        # Add value labels
        for bar, val in zip(bars, final_awareness):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f'{val:.1f}%', ha='center', va='bottom', fontsize=10)

        plt.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight')

        return fig

    def plot_waste_reduction_timeline(
        self,
        waste_data: Dict[str, np.ndarray],
        title: str = "Projected Food Waste Reduction",
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot food waste reduction over time.

        Args:
            waste_data: Dictionary with 'time', 'average_waste_pct', etc.
            title: Plot title
            save_path: Optional path to save the figure

        Returns:
            matplotlib Figure object
        """
        fig, ax = plt.subplots(figsize=self.figsize)

        time = waste_data['time']
        waste = waste_data['average_waste_pct']

        # Plot waste over time
        ax.plot(time, waste, color='#e74c3c', linewidth=2, label='Average Waste %')
        ax.fill_between(time, waste, alpha=0.3, color='#e74c3c')

        # Add reference lines
        initial_waste = waste[0]
        final_waste = waste[-1]
        ax.axhline(y=initial_waste, color='gray', linestyle='--', alpha=0.5,
                   label=f'Initial: {initial_waste:.1f}%')
        ax.axhline(y=final_waste, color='green', linestyle='--', alpha=0.5,
                   label=f'Final: {final_waste:.1f}%')

        ax.set_xlabel('Weeks', fontsize=12)
        ax.set_ylabel('Average Food Waste (%)', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(loc='upper right', fontsize=10)
        ax.set_xlim(0, time[-1])
        ax.set_ylim(0, max(waste) * 1.1)

        # Add annotation for reduction
        reduction = initial_waste - final_waste
        ax.annotate(
            f'Reduction: {reduction:.1f}%\n({reduction/initial_waste*100:.1f}% improvement)',
            xy=(time[-1]*0.7, (initial_waste + final_waste)/2),
            fontsize=11,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        )

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight')

        return fig

    def plot_savings_projection(
        self,
        savings_data: Dict[str, float],
        n_years: int = 5,
        title: str = "Projected Food and Cost Savings",
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot cumulative savings projection.

        Args:
            savings_data: Dictionary from MarkovChainModel.estimate_food_savings()
            n_years: Number of years to project
            title: Plot title
            save_path: Optional path to save the figure

        Returns:
            matplotlib Figure object
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        years = np.arange(1, n_years + 1)
        annual_food_saved = savings_data['annual_food_saved_lbs']
        annual_cost_saved = savings_data['annual_cost_savings']

        cumulative_food = np.cumsum([annual_food_saved] * n_years)
        cumulative_cost = np.cumsum([annual_cost_saved] * n_years)

        # Left plot: Food savings
        ax1 = axes[0]
        ax1.bar(years, cumulative_food, color='#27ae60', alpha=0.8)
        ax1.set_xlabel('Year', fontsize=12)
        ax1.set_ylabel('Cumulative Food Saved (lbs)', fontsize=12)
        ax1.set_title('Food Waste Prevented', fontsize=12, fontweight='bold')

        for i, val in enumerate(cumulative_food):
            ax1.text(years[i], val + 100, f'{val:,.0f}', ha='center', fontsize=10)

        # Right plot: Cost savings
        ax2 = axes[1]
        ax2.bar(years, cumulative_cost, color='#3498db', alpha=0.8)
        ax2.set_xlabel('Year', fontsize=12)
        ax2.set_ylabel('Cumulative Savings ($)', fontsize=12)
        ax2.set_title('Estimated Cost Savings', fontsize=12, fontweight='bold')

        for i, val in enumerate(cumulative_cost):
            ax2.text(years[i], val + 100, f'${val:,.0f}', ha='center', fontsize=10)

        plt.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight')

        return fig

    def plot_transition_matrix_heatmap(
        self,
        matrix: np.ndarray,
        state_names: List[str],
        title: str = "Behavior Transition Probabilities",
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot transition matrix as a heatmap.

        Args:
            matrix: Transition probability matrix
            state_names: Names of states
            title: Plot title
            save_path: Optional path to save the figure

        Returns:
            matplotlib Figure object
        """
        fig, ax = plt.subplots(figsize=(8, 6))

        im = ax.imshow(matrix, cmap='YlGnBu', vmin=0, vmax=1)

        # Add colorbar
        cbar = ax.figure.colorbar(im, ax=ax)
        cbar.ax.set_ylabel('Transition Probability', rotation=-90, va="bottom")

        # Set ticks and labels
        short_names = ['High', 'Medium', 'Low', 'Minimal']
        ax.set_xticks(np.arange(len(short_names)))
        ax.set_yticks(np.arange(len(short_names)))
        ax.set_xticklabels(short_names)
        ax.set_yticklabels(short_names)

        ax.set_xlabel('To State', fontsize=12)
        ax.set_ylabel('From State', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')

        # Add text annotations
        for i in range(len(short_names)):
            for j in range(len(short_names)):
                text = ax.text(j, i, f'{matrix[i, j]:.2f}',
                              ha='center', va='center',
                              color='white' if matrix[i, j] > 0.5 else 'black')

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight')

        return fig

    def create_dashboard(
        self,
        markov_results: Dict,
        sir_results: Dict,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Create a comprehensive dashboard combining all visualizations.

        Args:
            markov_results: Results from MarkovChainModel
            sir_results: Results from SIRModel
            save_path: Optional path to save the figure

        Returns:
            matplotlib Figure object
        """
        fig = plt.figure(figsize=(16, 12))

        # Create grid
        gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.25)

        # 1. Markov state evolution (top left)
        ax1 = fig.add_subplot(gs[0, 0])
        baseline_history = markov_results['baseline']['history']
        combined_history = markov_results['combined']['history']
        weeks = np.arange(len(baseline_history))

        waste_pcts = np.array([40.0, 22.0, 10.0, 3.0])
        baseline_waste = np.dot(baseline_history, waste_pcts)
        combined_waste = np.dot(combined_history, waste_pcts)

        ax1.plot(weeks, baseline_waste, label='No Intervention',
                color=self.colors['baseline'], linewidth=2)
        ax1.plot(weeks, combined_waste, label='Combined Intervention',
                color=self.colors['intervention'], linewidth=2)
        ax1.set_xlabel('Weeks')
        ax1.set_ylabel('Average Waste %')
        ax1.set_title('Markov Model: Waste Reduction', fontweight='bold')
        ax1.legend()

        # 2. SIR dynamics (top right)
        ax2 = fig.add_subplot(gs[0, 1])
        time = sir_results['time']
        ax2.plot(time, sir_results['S'], label='Unaware', color=self.colors['susceptible'])
        ax2.plot(time, sir_results['I'], label='Active', color=self.colors['informed'])
        ax2.plot(time, sir_results['R'], label='Reformed', color=self.colors['reformed'])
        ax2.set_xlabel('Weeks')
        ax2.set_ylabel('Students')
        ax2.set_title('SIR Model: Awareness Spread', fontweight='bold')
        ax2.legend()

        # 3. Intervention comparison (middle left)
        ax3 = fig.add_subplot(gs[1, 0])
        interventions = ['baseline', 'awareness_campaign', 'portion_control',
                        'peer_pressure', 'incentive_program', 'combined']
        final_waste = [markov_results[i]['final_waste_pct'] for i in interventions]
        colors = ['#7f8c8d', '#3498db', '#9b59b6', '#e67e22', '#1abc9c', '#27ae60']
        bars = ax3.barh([i.replace('_', ' ').title() for i in interventions],
                       final_waste, color=colors)
        ax3.set_xlabel('Final Average Waste %')
        ax3.set_title('Intervention Effectiveness', fontweight='bold')
        ax3.invert_yaxis()

        # 4. Awareness growth (middle right)
        ax4 = fig.add_subplot(gs[1, 1])
        awareness = sir_results['awareness_pct']
        ax4.fill_between(time, 0, awareness, alpha=0.3, color='#27ae60')
        ax4.plot(time, awareness, color='#27ae60', linewidth=2)
        ax4.axhline(y=50, color='gray', linestyle='--', alpha=0.5)
        ax4.set_xlabel('Weeks')
        ax4.set_ylabel('% Students Aware')
        ax4.set_title('Awareness Progress', fontweight='bold')
        ax4.set_ylim(0, 100)

        # 5. Key metrics summary (bottom spanning both columns)
        ax5 = fig.add_subplot(gs[2, :])
        ax5.axis('off')

        # Calculate key metrics
        initial_waste = markov_results['baseline']['initial_waste_pct']
        final_waste_baseline = markov_results['baseline']['final_waste_pct']
        final_waste_combined = markov_results['combined']['final_waste_pct']
        waste_reduction = initial_waste - final_waste_combined
        pct_improvement = waste_reduction / initial_waste * 100

        final_awareness = sir_results['awareness_pct'][-1]
        final_reformed = sir_results['R'][-1] / (sir_results['S'][-1] +
                        sir_results['I'][-1] + sir_results['R'][-1]) * 100

        # Create summary text
        summary = (
            f"{'KEY METRICS SUMMARY':^80}\n"
            f"{'='*80}\n\n"
            f"MARKOV CHAIN MODEL\n"
            f"  Initial Average Waste: {initial_waste:.1f}%\n"
            f"  Final Waste (No Intervention): {final_waste_baseline:.1f}%\n"
            f"  Final Waste (Combined Intervention): {final_waste_combined:.1f}%\n"
            f"  Waste Reduction: {waste_reduction:.1f} percentage points "
            f"({pct_improvement:.1f}% improvement)\n\n"
            f"SIR MODEL\n"
            f"  Final Awareness Level: {final_awareness:.1f}%\n"
            f"  Students with Reformed Habits: {final_reformed:.1f}%\n\n"
            f"PROJECTED ANNUAL IMPACT (820 students)\n"
            f"  Estimated Food Saved: ~{820 * 1.1 * 7 * 52 * waste_reduction/100:.0f} lbs/year\n"
            f"  Estimated Cost Savings: ~${820 * 1.1 * 7 * 52 * waste_reduction/100 * 3:.0f}/year"
        )

        ax5.text(0.5, 0.5, summary, transform=ax5.transAxes, fontsize=11,
                verticalalignment='center', horizontalalignment='center',
                fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.3))

        plt.suptitle('Lawrenceville School Food Waste Optimization Dashboard',
                    fontsize=16, fontweight='bold', y=0.98)

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight')

        return fig


    # ------------------------------------------------------------------
    # Food Availability Model Plots
    # ------------------------------------------------------------------

    def plot_food_availability_dynamics(
        self,
        simulation: Dict[str, np.ndarray],
        title: str = "Food Availability Model Dynamics",
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot F(t), E(t), W(t) time series from the food availability ODE model.

        Args:
            simulation: Dict with 't', 'F', 'E', 'W', 'S' arrays.
            title: Plot title.
            save_path: Optional path to save the figure.

        Returns:
            matplotlib Figure object.
        """
        t = simulation['t']
        t_min = t * 60  # convert hours to minutes

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # F(t) — food available per student
        ax = axes[0, 0]
        ax.plot(t_min, simulation['F'], color='#3498db', linewidth=2)
        ax.set_ylabel('Food per Student [lbs]')
        ax.set_title('F(t) — Food Available', fontweight='bold')
        ax.set_xlabel('Minutes')

        # E(t) — food being eaten
        ax = axes[0, 1]
        ax.plot(t_min, simulation['E'], color='#e67e22', linewidth=2)
        ax.set_ylabel('Food Being Eaten [lbs]')
        ax.set_title('E(t) — Food in Process of Being Eaten', fontweight='bold')
        ax.set_xlabel('Minutes')

        # W(t) — cumulative waste
        ax = axes[1, 0]
        ax.plot(t_min, simulation['W'], color='#e74c3c', linewidth=2)
        ax.set_ylabel('Cumulative Waste [lbs]')
        ax.set_title('W(t) — Total Accumulated Waste', fontweight='bold')
        ax.set_xlabel('Minutes')

        # S(t) — student demand
        ax = axes[1, 1]
        ax.plot(t_min, simulation['S'], color='#27ae60', linewidth=2, drawstyle='steps-post')
        ax.set_ylabel('Students Demanding Food')
        ax.set_title('S(t) — Student Arrival Demand', fontweight='bold')
        ax.set_xlabel('Minutes')

        plt.suptitle(title, fontsize=14, fontweight='bold', y=1.01)
        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight')

        return fig

    def plot_baseline_vs_optimized(
        self,
        baseline: Dict[str, np.ndarray],
        optimized: Dict[str, np.ndarray],
        strategy_name: str = "Optimized",
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Compare baseline and optimized W(t) trajectories.

        Args:
            baseline: Simulation dict from baseline run.
            optimized: Simulation dict from optimized run.
            strategy_name: Label for the optimized scenario.
            save_path: Optional path to save the figure.

        Returns:
            matplotlib Figure object.
        """
        t_min = baseline['t'] * 60

        fig, axes = plt.subplots(1, 3, figsize=(16, 5))

        # W(t) comparison
        ax = axes[0]
        ax.plot(t_min, baseline['W'], label='Baseline', color='#e74c3c', linewidth=2)
        ax.plot(t_min, optimized['W'], label=strategy_name, color='#27ae60',
                linewidth=2, linestyle='--')
        ax.set_xlabel('Minutes')
        ax.set_ylabel('Waste [lbs]')
        ax.set_title('W(t) — Waste Comparison', fontweight='bold')
        ax.legend()

        # F(t) comparison
        ax = axes[1]
        ax.plot(t_min, baseline['F'], label='Baseline', color='#3498db', linewidth=2)
        ax.plot(t_min, optimized['F'], label=strategy_name, color='#2ecc71',
                linewidth=2, linestyle='--')
        ax.set_xlabel('Minutes')
        ax.set_ylabel('Food per Student [lbs]')
        ax.set_title('F(t) — Food Available', fontweight='bold')
        ax.legend()

        # E(t) comparison
        ax = axes[2]
        ax.plot(t_min, baseline['E'], label='Baseline', color='#e67e22', linewidth=2)
        ax.plot(t_min, optimized['E'], label=strategy_name, color='#1abc9c',
                linewidth=2, linestyle='--')
        ax.set_xlabel('Minutes')
        ax.set_ylabel('Food Being Eaten [lbs]')
        ax.set_title('E(t) — Eating', fontweight='bold')
        ax.legend()

        plt.suptitle(f'Baseline vs {strategy_name}', fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight')

        return fig

    def plot_sensitivity(
        self,
        sensitivity_data: Dict[str, Dict[str, np.ndarray]],
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot sensitivity analysis: waste vs each parameter.

        Args:
            sensitivity_data: Dict mapping param name -> {'values', 'waste'}.
            save_path: Optional path to save the figure.

        Returns:
            matplotlib Figure object.
        """
        n = len(sensitivity_data)
        fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))
        if n == 1:
            axes = [axes]

        labels = {
            'F0': 'Initial Food per Student [lbs]',
            'waste_fraction': 'Plate Waste Fraction',
            'alpha': 'Service Efficiency α [1/hr]',
            'beta': 'Eating Rate β [1/hr]',
        }
        colors = ['#3498db', '#e74c3c', '#27ae60', '#9b59b6']

        for ax, (param, data), color in zip(axes, sensitivity_data.items(), colors):
            ax.plot(data['values'], data['waste'], color=color, linewidth=2)
            idx_min = np.argmin(data['waste'])
            ax.axvline(data['values'][idx_min], color='gray', linestyle='--', alpha=0.5)
            ax.scatter([data['values'][idx_min]], [data['waste'][idx_min]],
                       color=color, s=80, zorder=5)
            ax.set_xlabel(labels.get(param, param), fontsize=11)
            ax.set_ylabel('Total Waste W(T) [lbs]', fontsize=11)
            ax.set_title(f'Sensitivity to {param}', fontweight='bold')

        plt.suptitle('Sensitivity Analysis — Total Waste', fontsize=14,
                     fontweight='bold', y=1.02)
        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight')

        return fig


if __name__ == "__main__":
    # Demo visualization
    print("Running visualization demo...")

    # Import models for demo
    import sys
    sys.path.insert(0, '/home/user/Optimizing-Food-Waste/src')
    from models.markov_chain import MarkovChainModel
    from models.sir_model import SIRModel

    # Create models and run simulations
    markov = MarkovChainModel()
    sir = SIRModel()

    markov_results = markov.compare_interventions(n_steps=52)
    sir_results = sir.simulate(t_max=52)

    # Create visualizer and generate plots
    viz = FoodWasteVisualizer()

    # Create dashboard
    fig = viz.create_dashboard(markov_results, sir_results)
    plt.savefig('/home/user/Optimizing-Food-Waste/outputs/dashboard.png', dpi=150)
    print("Dashboard saved to outputs/dashboard.png")

    plt.show()
