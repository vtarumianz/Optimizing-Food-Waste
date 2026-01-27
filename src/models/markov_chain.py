"""
Markov Chain Model for Food Waste Behavior Analysis
Lawrenceville School Food Waste Optimization Project

This model captures the transition of student food waste behaviors across
discrete states. Students can move between waste states based on various
factors including awareness campaigns, peer influence, and dining hall policies.

States:
- HIGH_WASTE: Students wasting > 30% of food taken
- MEDIUM_WASTE: Students wasting 15-30% of food taken
- LOW_WASTE: Students wasting 5-15% of food taken
- MINIMAL_WASTE: Students wasting < 5% of food taken
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from enum import Enum


class WasteState(Enum):
    """Food waste behavior states for students"""
    HIGH_WASTE = 0      # >30% waste
    MEDIUM_WASTE = 1    # 15-30% waste
    LOW_WASTE = 2       # 5-15% waste
    MINIMAL_WASTE = 3   # <5% waste


class MarkovChainModel:
    """
    Markov Chain Model for analyzing and predicting food waste behavior
    transitions among students at Lawrenceville School.

    The model uses a discrete-time Markov chain where each state represents
    a food waste behavior category. Transition probabilities can be adjusted
    to simulate the effects of interventions.

    Attributes:
        states: List of possible waste states
        n_states: Number of states in the model
        transition_matrix: Matrix of transition probabilities P[i,j] = P(j|i)
        state_names: Human-readable state names
        current_distribution: Current distribution of students across states
    """

    def __init__(self, transition_matrix: Optional[np.ndarray] = None):
        """
        Initialize the Markov Chain model.

        Args:
            transition_matrix: Optional custom transition matrix. If None,
                              uses default baseline probabilities.
        """
        self.states = list(WasteState)
        self.n_states = len(self.states)
        self.state_names = [
            "High Waste (>30%)",
            "Medium Waste (15-30%)",
            "Low Waste (5-15%)",
            "Minimal Waste (<5%)"
        ]

        if transition_matrix is not None:
            self._validate_transition_matrix(transition_matrix)
            self.transition_matrix = transition_matrix
        else:
            self.transition_matrix = self._get_baseline_matrix()

        # Initialize with estimated Lawrenceville distribution
        # Based on typical dining hall waste patterns
        self.current_distribution = np.array([0.25, 0.35, 0.25, 0.15])

    def _validate_transition_matrix(self, matrix: np.ndarray) -> None:
        """Validate that matrix is a proper stochastic matrix."""
        if matrix.shape != (self.n_states, self.n_states):
            raise ValueError(f"Matrix must be {self.n_states}x{self.n_states}")
        if not np.allclose(matrix.sum(axis=1), 1.0):
            raise ValueError("Rows must sum to 1 (stochastic matrix)")
        if np.any(matrix < 0) or np.any(matrix > 1):
            raise ValueError("All probabilities must be between 0 and 1")

    def _get_baseline_matrix(self) -> np.ndarray:
        """
        Get the baseline transition matrix representing current behavior
        without any interventions.

        The matrix represents weekly transition probabilities.
        Students tend to stay in their current state with some drift.
        """
        # Baseline: students mostly stay in current state
        # Small probability of improving or regressing
        return np.array([
            # To: HIGH   MEDIUM  LOW     MINIMAL
            [0.70,  0.20,   0.08,   0.02],  # From HIGH
            [0.15,  0.60,   0.20,   0.05],  # From MEDIUM
            [0.05,  0.15,   0.60,   0.20],  # From LOW
            [0.02,  0.08,   0.20,   0.70],  # From MINIMAL
        ])

    def get_intervention_matrix(self, intervention_type: str) -> np.ndarray:
        """
        Get modified transition matrix after applying an intervention.

        Args:
            intervention_type: Type of intervention:
                - 'awareness_campaign': Educational posters and announcements
                - 'portion_control': Smaller plate sizes, portion guidance
                - 'peer_pressure': Public waste tracking, social comparison
                - 'incentive_program': Rewards for low-waste behavior
                - 'combined': All interventions together

        Returns:
            Modified transition matrix with improved transition probabilities
        """
        if intervention_type == 'awareness_campaign':
            # Moderate improvement in downward transitions
            return np.array([
                [0.55,  0.30,   0.12,   0.03],
                [0.10,  0.50,   0.30,   0.10],
                [0.03,  0.12,   0.55,   0.30],
                [0.01,  0.05,   0.15,   0.79],
            ])

        elif intervention_type == 'portion_control':
            # Good improvement - helps people take less initially
            return np.array([
                [0.50,  0.32,   0.14,   0.04],
                [0.08,  0.47,   0.33,   0.12],
                [0.02,  0.10,   0.53,   0.35],
                [0.01,  0.04,   0.12,   0.83],
            ])

        elif intervention_type == 'peer_pressure':
            # Strong improvement through social influence
            return np.array([
                [0.45,  0.35,   0.15,   0.05],
                [0.08,  0.42,   0.35,   0.15],
                [0.02,  0.08,   0.50,   0.40],
                [0.01,  0.03,   0.10,   0.86],
            ])

        elif intervention_type == 'incentive_program':
            # Strong improvement through positive reinforcement
            return np.array([
                [0.40,  0.38,   0.17,   0.05],
                [0.06,  0.40,   0.38,   0.16],
                [0.02,  0.06,   0.47,   0.45],
                [0.01,  0.02,   0.08,   0.89],
            ])

        elif intervention_type == 'combined':
            # Maximum improvement with all strategies
            return np.array([
                [0.30,  0.40,   0.22,   0.08],
                [0.04,  0.30,   0.42,   0.24],
                [0.01,  0.04,   0.40,   0.55],
                [0.00,  0.01,   0.05,   0.94],
            ])

        else:
            raise ValueError(f"Unknown intervention type: {intervention_type}")

    def simulate_steps(
        self,
        n_steps: int,
        initial_distribution: Optional[np.ndarray] = None,
        transition_matrix: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Simulate the evolution of student distribution over time.

        Args:
            n_steps: Number of time steps (weeks) to simulate
            initial_distribution: Starting distribution. If None, uses current.
            transition_matrix: Matrix to use. If None, uses default.

        Returns:
            Array of shape (n_steps + 1, n_states) with distribution at each step
        """
        if initial_distribution is None:
            initial_distribution = self.current_distribution.copy()
        if transition_matrix is None:
            transition_matrix = self.transition_matrix

        history = np.zeros((n_steps + 1, self.n_states))
        history[0] = initial_distribution

        current = initial_distribution.copy()
        for t in range(n_steps):
            current = current @ transition_matrix
            history[t + 1] = current

        return history

    def compute_steady_state(
        self,
        transition_matrix: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Compute the steady-state (stationary) distribution.

        The steady state represents the long-term distribution of students
        across waste categories if the transition probabilities remain constant.

        Args:
            transition_matrix: Matrix to analyze. If None, uses default.

        Returns:
            Steady-state probability distribution
        """
        if transition_matrix is None:
            transition_matrix = self.transition_matrix

        # Solve (P^T - I) * pi = 0 with sum(pi) = 1
        A = transition_matrix.T - np.eye(self.n_states)
        A = np.vstack([A, np.ones(self.n_states)])
        b = np.zeros(self.n_states + 1)
        b[-1] = 1

        # Least squares solution
        steady_state, _, _, _ = np.linalg.lstsq(A, b, rcond=None)

        # Normalize to ensure sum = 1 (numerical precision)
        steady_state = steady_state / steady_state.sum()

        return steady_state

    def compute_expected_waste(
        self,
        distribution: np.ndarray,
        waste_percentages: Optional[np.ndarray] = None
    ) -> float:
        """
        Compute expected food waste percentage given a distribution.

        Args:
            distribution: Distribution of students across states
            waste_percentages: Average waste % for each state.
                              Defaults to [40, 22, 10, 3]

        Returns:
            Expected waste percentage
        """
        if waste_percentages is None:
            # Midpoint estimates for each category
            waste_percentages = np.array([40.0, 22.0, 10.0, 3.0])

        return np.dot(distribution, waste_percentages)

    def compute_convergence_time(
        self,
        target_waste_pct: float,
        transition_matrix: Optional[np.ndarray] = None,
        initial_distribution: Optional[np.ndarray] = None,
        max_steps: int = 520  # 10 years in weeks
    ) -> int:
        """
        Compute time to reach a target average waste percentage.

        Args:
            target_waste_pct: Target average waste percentage
            transition_matrix: Matrix to use. If None, uses default.
            initial_distribution: Starting distribution. If None, uses current.
            max_steps: Maximum steps to simulate

        Returns:
            Number of steps to reach target, or -1 if not achieved
        """
        if initial_distribution is None:
            initial_distribution = self.current_distribution.copy()
        if transition_matrix is None:
            transition_matrix = self.transition_matrix

        current = initial_distribution.copy()
        for t in range(max_steps):
            waste = self.compute_expected_waste(current)
            if waste <= target_waste_pct:
                return t
            current = current @ transition_matrix

        return -1  # Target not reached

    def compare_interventions(
        self,
        n_steps: int = 52,  # 1 year
        interventions: Optional[List[str]] = None
    ) -> Dict[str, Dict]:
        """
        Compare different intervention strategies.

        Args:
            n_steps: Number of weeks to simulate
            interventions: List of intervention types to compare.
                          If None, compares all available.

        Returns:
            Dictionary with results for each intervention
        """
        if interventions is None:
            interventions = [
                'baseline', 'awareness_campaign', 'portion_control',
                'peer_pressure', 'incentive_program', 'combined'
            ]

        results = {}

        for intervention in interventions:
            if intervention == 'baseline':
                matrix = self.transition_matrix
            else:
                matrix = self.get_intervention_matrix(intervention)

            # Simulate
            history = self.simulate_steps(n_steps, transition_matrix=matrix)

            # Compute metrics
            initial_waste = self.compute_expected_waste(history[0])
            final_waste = self.compute_expected_waste(history[-1])
            steady_state = self.compute_steady_state(matrix)
            steady_waste = self.compute_expected_waste(steady_state)

            results[intervention] = {
                'history': history,
                'initial_waste_pct': initial_waste,
                'final_waste_pct': final_waste,
                'waste_reduction_pct': initial_waste - final_waste,
                'steady_state': steady_state,
                'steady_state_waste_pct': steady_waste,
                'final_distribution': history[-1]
            }

        return results

    def estimate_food_savings(
        self,
        n_students: int,
        daily_food_per_student_lbs: float,
        current_waste_pct: float,
        projected_waste_pct: float,
        days_per_week: int = 7
    ) -> Dict[str, float]:
        """
        Estimate food savings from waste reduction.

        Args:
            n_students: Number of students in dining hall
            daily_food_per_student_lbs: Average food taken per student per day
            current_waste_pct: Current average waste percentage
            projected_waste_pct: Projected waste percentage after intervention
            days_per_week: Days dining hall operates

        Returns:
            Dictionary with savings estimates
        """
        weekly_food_total = n_students * daily_food_per_student_lbs * days_per_week

        current_waste = weekly_food_total * (current_waste_pct / 100)
        projected_waste = weekly_food_total * (projected_waste_pct / 100)
        weekly_savings = current_waste - projected_waste

        # Estimate cost savings (assume $3/lb average food cost)
        cost_per_lb = 3.0

        return {
            'weekly_food_saved_lbs': weekly_savings,
            'annual_food_saved_lbs': weekly_savings * 52,
            'weekly_cost_savings': weekly_savings * cost_per_lb,
            'annual_cost_savings': weekly_savings * 52 * cost_per_lb,
            'waste_reduction_pct_points': current_waste_pct - projected_waste_pct
        }

    def get_model_summary(self) -> str:
        """Get a summary of the current model state."""
        steady = self.compute_steady_state()
        current_waste = self.compute_expected_waste(self.current_distribution)
        steady_waste = self.compute_expected_waste(steady)

        summary = [
            "=" * 60,
            "MARKOV CHAIN MODEL SUMMARY - FOOD WASTE BEHAVIOR",
            "=" * 60,
            "",
            "Current Student Distribution:",
        ]

        for i, (name, prob) in enumerate(zip(self.state_names, self.current_distribution)):
            summary.append(f"  {name}: {prob*100:.1f}%")

        summary.extend([
            f"\nCurrent Expected Waste: {current_waste:.1f}%",
            "",
            "Steady-State Distribution (No Intervention):",
        ])

        for i, (name, prob) in enumerate(zip(self.state_names, steady)):
            summary.append(f"  {name}: {prob*100:.1f}%")

        summary.extend([
            f"\nSteady-State Expected Waste: {steady_waste:.1f}%",
            "",
            "Transition Matrix (Weekly Probabilities):",
        ])

        header = "           " + "  ".join([f"{s:>8}" for s in ["HIGH", "MEDIUM", "LOW", "MINIMAL"]])
        summary.append(header)

        row_labels = ["HIGH", "MEDIUM", "LOW", "MINIMAL"]
        for i, row in enumerate(self.transition_matrix):
            row_str = f"{row_labels[i]:>10} " + "  ".join([f"{p:8.3f}" for p in row])
            summary.append(row_str)

        summary.append("=" * 60)

        return "\n".join(summary)


if __name__ == "__main__":
    # Demo usage
    model = MarkovChainModel()
    print(model.get_model_summary())

    print("\n" + "=" * 60)
    print("INTERVENTION COMPARISON (52 weeks)")
    print("=" * 60)

    results = model.compare_interventions(n_steps=52)

    for intervention, data in results.items():
        print(f"\n{intervention.upper().replace('_', ' ')}:")
        print(f"  Final waste: {data['final_waste_pct']:.1f}%")
        print(f"  Reduction: {data['waste_reduction_pct']:.1f} percentage points")
        print(f"  Steady-state waste: {data['steady_state_waste_pct']:.1f}%")
