"""
SIR Model for Food Waste Awareness Spread
Lawrenceville School Food Waste Optimization Project

This adapts the classic epidemiological SIR (Susceptible-Infected-Recovered) model
to simulate how food waste awareness and behavior changes spread through the
student population via peer influence and awareness campaigns.

Compartments:
- S (Susceptible/Unaware): Students not actively thinking about food waste
- I (Informed/Active): Students actively reducing food waste, spreading awareness
- R (Reformed/Habitual): Students who have adopted permanent low-waste habits

The "infection" here is positive - we WANT it to spread!
"""

import numpy as np
from scipy.integrate import odeint
from typing import Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass


@dataclass
class SIRParameters:
    """
    Parameters for the SIR food waste awareness model.

    Attributes:
        beta: Transmission rate - how quickly awareness spreads through contact
              Higher beta = more effective peer influence/word of mouth
        gamma: Recovery rate - rate at which actively engaged students
               transition to habitual low-waste behavior
        mu: Birth/enrollment rate (new students joining)
        nu: Death/graduation rate (students leaving)
        campaign_boost: Additional transmission from awareness campaigns
    """
    beta: float = 0.3      # Peer influence transmission rate
    gamma: float = 0.05    # Rate of habit formation (weekly)
    mu: float = 0.0        # Enrollment rate (can model semester starts)
    nu: float = 0.0        # Graduation rate
    campaign_boost: float = 0.0  # Extra transmission from campaigns


class SIRModel:
    """
    SIR Model for simulating the spread of food waste awareness and
    behavior change through the Lawrenceville School student population.

    Unlike disease modeling where we want to stop spread, here we want
    to MAXIMIZE the spread of awareness and good habits.

    The model tracks three populations:
    - S: Unaware/unconcerned students (susceptible to becoming aware)
    - I: Actively engaged students reducing waste and influencing others
    - R: Students with established low-waste habits (Reformed)

    Key insight: High-waste behavior is the "endemic state" we start with,
    and awareness/good behavior is what we want to spread.
    """

    def __init__(
        self,
        total_population: int = 820,  # Lawrenceville approximate enrollment
        params: Optional[SIRParameters] = None
    ):
        """
        Initialize the SIR model.

        Args:
            total_population: Total number of students
            params: Model parameters. If None, uses defaults.
        """
        self.N = total_population
        self.params = params if params else SIRParameters()

        # Initial conditions: mostly unaware, small seed of aware students
        self.S0 = 0.90 * self.N  # 90% unaware
        self.I0 = 0.08 * self.N  # 8% actively engaged
        self.R0 = 0.02 * self.N  # 2% habitual low-waste

    def _sir_derivatives(
        self,
        y: np.ndarray,
        t: float,
        params: SIRParameters,
        campaign_active: bool = False
    ) -> List[float]:
        """
        Compute derivatives for the SIR system.

        dS/dt = mu*N - beta*S*I/N - nu*S
        dI/dt = beta*S*I/N - gamma*I - nu*I
        dR/dt = gamma*I - nu*R

        Args:
            y: Current state [S, I, R]
            t: Current time
            params: Model parameters
            campaign_active: Whether awareness campaign is active

        Returns:
            Derivatives [dS/dt, dI/dt, dR/dt]
        """
        S, I, R = y
        N = S + I + R

        # Effective transmission rate
        beta_eff = params.beta
        if campaign_active:
            beta_eff += params.campaign_boost

        # Derivatives
        dSdt = params.mu * N - beta_eff * S * I / N - params.nu * S
        dIdt = beta_eff * S * I / N - params.gamma * I - params.nu * I
        dRdt = params.gamma * I - params.nu * R

        return [dSdt, dIdt, dRdt]

    def simulate(
        self,
        t_max: float = 52,  # weeks
        dt: float = 0.1,
        initial_conditions: Optional[Tuple[float, float, float]] = None,
        params: Optional[SIRParameters] = None,
        campaign_schedule: Optional[List[Tuple[float, float]]] = None
    ) -> Dict[str, np.ndarray]:
        """
        Simulate the spread of food waste awareness over time.

        Args:
            t_max: Maximum time (weeks)
            dt: Time step for output
            initial_conditions: Optional (S0, I0, R0). Uses defaults if None.
            params: Optional parameters. Uses instance defaults if None.
            campaign_schedule: List of (start_time, end_time) for campaigns

        Returns:
            Dictionary with time and compartment values
        """
        if initial_conditions is None:
            y0 = [self.S0, self.I0, self.R0]
        else:
            y0 = list(initial_conditions)

        if params is None:
            params = self.params

        t = np.arange(0, t_max + dt, dt)

        # If no campaign schedule, run simple simulation
        if campaign_schedule is None or len(campaign_schedule) == 0:
            solution = odeint(
                self._sir_derivatives,
                y0,
                t,
                args=(params, False)
            )
        else:
            # Piecewise simulation with campaigns
            solution = self._simulate_with_campaigns(
                y0, t, params, campaign_schedule
            )

        return {
            'time': t,
            'S': solution[:, 0],
            'I': solution[:, 1],
            'R': solution[:, 2],
            'total_aware': solution[:, 1] + solution[:, 2],  # I + R
            'awareness_pct': (solution[:, 1] + solution[:, 2]) / self.N * 100
        }

    def _simulate_with_campaigns(
        self,
        y0: List[float],
        t: np.ndarray,
        params: SIRParameters,
        campaign_schedule: List[Tuple[float, float]]
    ) -> np.ndarray:
        """Simulate with time-varying campaign effectiveness."""
        solution = np.zeros((len(t), 3))
        solution[0] = y0

        current_y = np.array(y0)

        for i in range(1, len(t)):
            current_t = t[i-1]
            next_t = t[i]

            # Check if campaign is active
            campaign_active = any(
                start <= current_t < end
                for start, end in campaign_schedule
            )

            # Single step integration
            step_solution = odeint(
                self._sir_derivatives,
                current_y,
                [current_t, next_t],
                args=(params, campaign_active)
            )

            current_y = step_solution[-1]
            solution[i] = current_y

        return solution

    def compute_basic_reproduction_number(
        self,
        params: Optional[SIRParameters] = None
    ) -> float:
        """
        Compute R0 (basic reproduction number) for awareness spread.

        R0 = beta / gamma

        If R0 > 1: Awareness will spread through the population
        If R0 < 1: Awareness will die out without sustained intervention

        Args:
            params: Model parameters. Uses instance defaults if None.

        Returns:
            Basic reproduction number R0
        """
        if params is None:
            params = self.params

        return params.beta / params.gamma

    def compute_herd_immunity_threshold(
        self,
        params: Optional[SIRParameters] = None
    ) -> float:
        """
        Compute the "herd immunity" threshold for awareness.

        This is the fraction of the population that needs to be
        aware/reformed for awareness to be self-sustaining.

        Threshold = 1 - 1/R0

        Args:
            params: Model parameters

        Returns:
            Herd immunity threshold (0-1)
        """
        R0 = self.compute_basic_reproduction_number(params)
        if R0 <= 1:
            return 0.0  # Will never be self-sustaining
        return 1 - 1/R0

    def find_optimal_campaign_timing(
        self,
        campaign_duration: float = 4,  # weeks
        n_campaigns: int = 4,
        t_max: float = 52
    ) -> Dict:
        """
        Find optimal timing for awareness campaigns to maximize spread.

        Args:
            campaign_duration: Duration of each campaign in weeks
            n_campaigns: Number of campaigns to schedule
            t_max: Total time horizon

        Returns:
            Dictionary with optimal schedule and results
        """
        # Strategy 1: Evenly spaced campaigns
        interval = t_max / (n_campaigns + 1)
        even_schedule = [
            (interval * (i + 1), interval * (i + 1) + campaign_duration)
            for i in range(n_campaigns)
        ]

        # Strategy 2: Front-loaded campaigns (build momentum early)
        front_schedule = [
            (i * (campaign_duration + 2), i * (campaign_duration + 2) + campaign_duration)
            for i in range(n_campaigns)
        ]

        # Strategy 3: Semester starts (weeks 0, 18 for two semesters)
        semester_schedule = [
            (0, campaign_duration),
            (18, 18 + campaign_duration),
            (36, 36 + campaign_duration),
            (t_max - campaign_duration, t_max)
        ][:n_campaigns]

        # Boost parameters for campaigns
        campaign_params = SIRParameters(
            beta=self.params.beta,
            gamma=self.params.gamma,
            campaign_boost=0.2
        )

        # Simulate each strategy
        results = {}
        strategies = {
            'evenly_spaced': even_schedule,
            'front_loaded': front_schedule,
            'semester_starts': semester_schedule
        }

        for name, schedule in strategies.items():
            sim = self.simulate(
                t_max=t_max,
                params=campaign_params,
                campaign_schedule=schedule
            )
            results[name] = {
                'schedule': schedule,
                'final_awareness_pct': sim['awareness_pct'][-1],
                'final_reformed': sim['R'][-1],
                'peak_active': np.max(sim['I']),
                'simulation': sim
            }

        # Find best strategy
        best_strategy = max(
            results.keys(),
            key=lambda k: results[k]['final_awareness_pct']
        )

        return {
            'strategies': results,
            'best_strategy': best_strategy,
            'best_schedule': results[best_strategy]['schedule'],
            'best_final_awareness': results[best_strategy]['final_awareness_pct']
        }

    def estimate_waste_reduction(
        self,
        simulation_result: Dict[str, np.ndarray],
        waste_by_category: Optional[Dict[str, float]] = None
    ) -> Dict[str, np.ndarray]:
        """
        Estimate food waste reduction based on awareness levels.

        Assumes:
        - Unaware (S): Average 25% waste
        - Active (I): Average 10% waste (actively trying)
        - Reformed (R): Average 5% waste (habitual)

        Args:
            simulation_result: Output from simulate()
            waste_by_category: Optional custom waste percentages

        Returns:
            Dictionary with waste estimates over time
        """
        if waste_by_category is None:
            waste_by_category = {
                'S': 25.0,  # Unaware students waste more
                'I': 10.0,  # Active reducers
                'R': 5.0    # Habitual low-wasters
            }

        S = simulation_result['S']
        I = simulation_result['I']
        R = simulation_result['R']
        N = S + I + R

        # Weighted average waste
        avg_waste = (
            S * waste_by_category['S'] +
            I * waste_by_category['I'] +
            R * waste_by_category['R']
        ) / N

        return {
            'time': simulation_result['time'],
            'average_waste_pct': avg_waste,
            'waste_reduction_pct': avg_waste[0] - avg_waste,
            'relative_reduction': (avg_waste[0] - avg_waste) / avg_waste[0] * 100
        }

    def sensitivity_analysis(
        self,
        parameter: str,
        values: np.ndarray,
        metric: str = 'final_awareness'
    ) -> Dict[str, np.ndarray]:
        """
        Perform sensitivity analysis on a model parameter.

        Args:
            parameter: Parameter to vary ('beta', 'gamma', 'campaign_boost')
            values: Array of values to test
            metric: Metric to track ('final_awareness', 'time_to_50pct', 'peak_active')

        Returns:
            Dictionary with parameter values and corresponding metrics
        """
        results = []

        for value in values:
            # Create modified parameters
            params = SIRParameters(
                beta=self.params.beta if parameter != 'beta' else value,
                gamma=self.params.gamma if parameter != 'gamma' else value,
                campaign_boost=self.params.campaign_boost if parameter != 'campaign_boost' else value
            )

            sim = self.simulate(params=params)

            if metric == 'final_awareness':
                results.append(sim['awareness_pct'][-1])
            elif metric == 'time_to_50pct':
                idx = np.where(sim['awareness_pct'] >= 50)[0]
                results.append(sim['time'][idx[0]] if len(idx) > 0 else np.inf)
            elif metric == 'peak_active':
                results.append(np.max(sim['I']) / self.N * 100)

        return {
            'parameter': parameter,
            'values': values,
            metric: np.array(results)
        }

    def get_model_summary(self) -> str:
        """Get a summary of the current model state."""
        R0 = self.compute_basic_reproduction_number()
        herd_threshold = self.compute_herd_immunity_threshold()

        # Run a baseline simulation
        sim = self.simulate(t_max=52)

        summary = [
            "=" * 60,
            "SIR MODEL SUMMARY - FOOD WASTE AWARENESS SPREAD",
            "=" * 60,
            f"",
            f"Population: {self.N} students",
            f"",
            "Initial Conditions:",
            f"  Unaware (S): {self.S0:.0f} ({self.S0/self.N*100:.1f}%)",
            f"  Active (I):  {self.I0:.0f} ({self.I0/self.N*100:.1f}%)",
            f"  Reformed (R): {self.R0:.0f} ({self.R0/self.N*100:.1f}%)",
            f"",
            "Model Parameters:",
            f"  Transmission rate (beta): {self.params.beta}",
            f"  Habit formation rate (gamma): {self.params.gamma}",
            f"  Campaign boost: {self.params.campaign_boost}",
            f"",
            "Key Metrics:",
            f"  Basic reproduction number (R0): {R0:.2f}",
            f"  Self-sustaining threshold: {herd_threshold*100:.1f}% aware",
            f"",
            "52-Week Projection (No Campaigns):",
            f"  Final awareness: {sim['awareness_pct'][-1]:.1f}%",
            f"  Final reformed (habitual): {sim['R'][-1]/self.N*100:.1f}%",
            f"  Peak actively engaged: {np.max(sim['I'])/self.N*100:.1f}%",
            "=" * 60
        ]

        return "\n".join(summary)


class ExtendedSEIRModel(SIRModel):
    """
    Extended SEIR model with an Exposed/Considering compartment.

    Compartments:
    - S: Unaware students
    - E: Exposed/Considering - heard about waste reduction but not yet acting
    - I: Actively reducing waste and spreading awareness
    - R: Reformed - habitual low-waste behavior

    This model captures the delay between hearing about an issue
    and actually changing behavior.
    """

    def __init__(
        self,
        total_population: int = 820,
        sigma: float = 0.2,  # Rate of moving from E to I (decision rate)
        **kwargs
    ):
        """
        Initialize SEIR model.

        Args:
            total_population: Total students
            sigma: Rate at which exposed students become active
        """
        super().__init__(total_population, **kwargs)
        self.sigma = sigma

        # Adjust initial conditions to include E
        self.S0 = 0.85 * self.N
        self.E0 = 0.07 * self.N  # Some students considering
        self.I0 = 0.06 * self.N
        self.R0 = 0.02 * self.N

    def _seir_derivatives(
        self,
        y: np.ndarray,
        t: float,
        params: SIRParameters,
        sigma: float,
        campaign_active: bool = False
    ) -> List[float]:
        """Compute SEIR derivatives."""
        S, E, I, R = y
        N = S + E + I + R

        beta_eff = params.beta + (params.campaign_boost if campaign_active else 0)

        dSdt = params.mu * N - beta_eff * S * I / N - params.nu * S
        dEdt = beta_eff * S * I / N - sigma * E - params.nu * E
        dIdt = sigma * E - params.gamma * I - params.nu * I
        dRdt = params.gamma * I - params.nu * R

        return [dSdt, dEdt, dIdt, dRdt]

    def simulate(
        self,
        t_max: float = 52,
        dt: float = 0.1,
        params: Optional[SIRParameters] = None
    ) -> Dict[str, np.ndarray]:
        """Simulate the SEIR model."""
        if params is None:
            params = self.params

        y0 = [self.S0, self.E0, self.I0, self.R0]
        t = np.arange(0, t_max + dt, dt)

        solution = odeint(
            self._seir_derivatives,
            y0,
            t,
            args=(params, self.sigma, False)
        )

        return {
            'time': t,
            'S': solution[:, 0],
            'E': solution[:, 1],
            'I': solution[:, 2],
            'R': solution[:, 3],
            'total_aware': solution[:, 1] + solution[:, 2] + solution[:, 3],
            'awareness_pct': (solution[:, 1] + solution[:, 2] + solution[:, 3]) / self.N * 100
        }


if __name__ == "__main__":
    # Demo usage
    model = SIRModel()
    print(model.get_model_summary())

    print("\n" + "=" * 60)
    print("CAMPAIGN TIMING OPTIMIZATION")
    print("=" * 60)

    optimal = model.find_optimal_campaign_timing(
        campaign_duration=4,
        n_campaigns=4
    )

    print(f"\nBest strategy: {optimal['best_strategy']}")
    print(f"Final awareness: {optimal['best_final_awareness']:.1f}%")
    print(f"\nRecommended campaign schedule (weeks):")
    for i, (start, end) in enumerate(optimal['best_schedule']):
        print(f"  Campaign {i+1}: Week {start:.0f} to Week {end:.0f}")
