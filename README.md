# Lawrenceville School Food Waste Optimization

A mathematical modeling project that uses **Markov Chain Models** and **SIR (Susceptible-Informed-Reformed) Models** to analyze, predict, and optimize food waste reduction strategies at the Lawrenceville School.

## Overview

This project applies epidemiological and stochastic modeling techniques to the problem of food waste in institutional dining settings. By treating waste reduction behaviors as states in a Markov chain and awareness as a spreading phenomenon, we can:

- **Predict** how student behaviors will change over time
- **Compare** different intervention strategies quantitatively
- **Optimize** the timing and type of awareness campaigns
- **Project** food and cost savings from waste reduction efforts

## Models

### 1. Markov Chain Model

Models individual student behavior transitions between waste categories:

| State | Description | Waste % |
|-------|-------------|---------|
| High Waste | Students wasting significant food | >30% |
| Medium Waste | Moderate waste levels | 15-30% |
| Low Waste | Below-average waste | 5-15% |
| Minimal Waste | Highly conscious | <5% |

**Key Features:**
- Transition probability matrices for different interventions
- Steady-state distribution computation
- Convergence time analysis
- Multi-intervention comparison

### 2. SIR Model

Adapts the epidemiological SIR model to simulate awareness spread:

| Compartment | Description |
|-------------|-------------|
| S (Susceptible) | Unaware/unconcerned about food waste |
| I (Informed) | Actively reducing waste, spreading awareness |
| R (Reformed) | Habitual low-waste behavior established |

**Key Features:**
- Basic reproduction number (R0) calculation
- Campaign timing optimization
- Herd immunity threshold for self-sustaining change
- SEIR variant with "Exposed/Considering" state

## Installation

```bash
# Clone the repository
git clone https://github.com/vtarumianz/Optimizing-Food-Waste.git
cd Optimizing-Food-Waste

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### Run Full Analysis

```bash
python main.py
```

This will:
1. Run Markov Chain analysis comparing intervention strategies
2. Run SIR model analysis for awareness spread
3. Generate visualizations in `outputs/`
4. Create a comprehensive analysis report

### Generate Sample Data

```bash
python main.py --generate-data
```

### Run Without Plots

```bash
python main.py --no-plots
```

## Usage Examples

### Using the Markov Chain Model

```python
from src.models.markov_chain import MarkovChainModel

# Initialize model
model = MarkovChainModel()

# Print current state
print(model.get_model_summary())

# Compare interventions over 52 weeks
results = model.compare_interventions(n_steps=52)

for intervention, data in results.items():
    print(f"{intervention}: {data['final_waste_pct']:.1f}% final waste")

# Estimate savings
savings = model.estimate_food_savings(
    n_students=820,
    daily_food_per_student_lbs=1.1,
    current_waste_pct=20.0,
    projected_waste_pct=12.0
)
print(f"Annual savings: ${savings['annual_cost_savings']:,.0f}")
```

### Using the SIR Model

```python
from src.models.sir_model import SIRModel, SIRParameters

# Initialize for Lawrenceville population
model = SIRModel(total_population=820)

# Run simulation with awareness campaigns
campaign_schedule = [
    (0, 4),    # Start of year
    (18, 22),  # Second semester
]

results = model.simulate(
    t_max=52,
    campaign_schedule=campaign_schedule
)

print(f"Final awareness: {results['awareness_pct'][-1]:.1f}%")

# Find optimal campaign timing
optimal = model.find_optimal_campaign_timing(
    campaign_duration=4,
    n_campaigns=4
)
print(f"Best strategy: {optimal['best_strategy']}")
```

### Generating Synthetic Data

```python
from src.utils.data_generator import FoodWasteDataGenerator
from datetime import datetime

generator = FoodWasteDataGenerator(seed=42)

# Generate semester data with intervention
data = generator.generate_semester_data(
    start_date=datetime(2024, 9, 5),
    n_weeks=18,
    intervention_start_week=4,
    intervention_type='awareness'
)

# Get statistics
stats = generator.get_statistics_summary(data)
print(f"Overall waste rate: {stats['overall_waste_percentage']:.1f}%")

# Export to CSV
generator.export_to_csv(data, 'data/semester_data')
```

### Creating Visualizations

```python
from src.utils.visualization import FoodWasteVisualizer
from src.models.markov_chain import MarkovChainModel
from src.models.sir_model import SIRModel

viz = FoodWasteVisualizer()

# Create comparison plot
markov = MarkovChainModel()
results = markov.compare_interventions(n_steps=52)
fig = viz.plot_markov_intervention_comparison(results, save_path='comparison.png')

# Create SIR dynamics plot
sir = SIRModel()
sim = sir.simulate(t_max=52)
fig = viz.plot_sir_dynamics(sim, save_path='sir_dynamics.png')

# Create comprehensive dashboard
fig = viz.create_dashboard(results, sim, save_path='dashboard.png')
```

## Project Structure

```
Optimizing-Food-Waste/
├── main.py                     # Main analysis script
├── requirements.txt            # Python dependencies
├── README.md                   # This file
├── src/
│   ├── __init__.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── markov_chain.py     # Markov Chain model
│   │   └── sir_model.py        # SIR model
│   └── utils/
│       ├── __init__.py
│       ├── data_generator.py   # Synthetic data generation
│       └── visualization.py    # Plotting utilities
├── data/                       # Generated/input data
├── notebooks/                  # Jupyter notebooks (optional)
└── outputs/                    # Generated plots and reports
```

## Intervention Strategies

The models support analyzing these intervention types:

| Strategy | Description | Expected Impact |
|----------|-------------|-----------------|
| Awareness Campaign | Educational posters, announcements | Moderate |
| Portion Control | Smaller plates, serving guidance | Good |
| Peer Pressure | Social comparison, public tracking | Strong |
| Incentive Program | Rewards for low-waste behavior | Strong |
| Combined | All strategies together | Maximum |

## Key Findings

Based on model simulations for Lawrenceville School (820 students):

1. **Baseline waste**: ~20% of food taken is wasted
2. **Combined intervention**: Can reduce waste to ~11%
3. **Annual food savings**: ~25,000 lbs
4. **Annual cost savings**: ~$75,000
5. **Critical awareness threshold**: 50% for self-sustaining change

## Mathematical Background

### Markov Chain Theory

The transition matrix P describes weekly behavior changes:
- P[i,j] = probability of moving from state i to state j
- Steady state π satisfies πP = π
- Expected waste = Σ(distribution × waste_per_state)

### SIR Model Equations

```
dS/dt = -β(S·I)/N
dI/dt = β(S·I)/N - γI
dR/dt = γI
```

Where:
- β = transmission rate (peer influence)
- γ = recovery rate (habit formation)
- R0 = β/γ (basic reproduction number)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is developed for educational purposes at the Lawrenceville School.

## Acknowledgments

- Lawrenceville School Dining Services
- Environmental Sustainability Committee
- Mathematics Department

## Contact

For questions about this project, please contact the Lawrenceville School research team.
