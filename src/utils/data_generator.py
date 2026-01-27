"""
Food Waste Data Generator
Lawrenceville School Food Waste Optimization Project

Generates simulated food waste data based on realistic parameters
for the Lawrenceville School dining operations.

This module creates synthetic datasets for testing and validating
the Markov Chain and SIR models.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass


@dataclass
class LawrencevilleConfig:
    """Configuration parameters for Lawrenceville School dining."""
    total_students: int = 820
    boarding_students: int = 640  # ~78% boarding
    day_students: int = 180      # ~22% day students

    # Dining halls
    dining_halls: List[str] = None

    # Meal participation rates (boarding students eat most meals)
    breakfast_participation: float = 0.65
    lunch_participation: float = 0.90
    dinner_participation: float = 0.85

    # Food quantities (lbs per student per meal)
    avg_food_taken_breakfast: float = 0.75
    avg_food_taken_lunch: float = 1.2
    avg_food_taken_dinner: float = 1.4

    # Academic calendar
    school_days_per_year: int = 180
    weeks_per_semester: int = 18

    def __post_init__(self):
        if self.dining_halls is None:
            self.dining_halls = ["Irwin Dining Hall", "Abbott Dining Hall"]


class FoodWasteDataGenerator:
    """
    Generates synthetic food waste data for the Lawrenceville School.

    Creates realistic datasets including:
    - Individual meal records
    - Daily aggregated waste data
    - Weekly summaries
    - Seasonal variations
    - Event-based variations (holidays, exams, etc.)
    """

    def __init__(self, config: Optional[LawrencevilleConfig] = None, seed: int = 42):
        """
        Initialize the data generator.

        Args:
            config: School configuration. Uses defaults if None.
            seed: Random seed for reproducibility
        """
        self.config = config if config else LawrencevilleConfig()
        self.rng = np.random.default_rng(seed)

        # Waste behavior categories
        self.waste_categories = {
            'minimal': (0.0, 0.05),    # 0-5% waste
            'low': (0.05, 0.15),       # 5-15% waste
            'medium': (0.15, 0.30),    # 15-30% waste
            'high': (0.30, 0.50)       # 30-50% waste
        }

        # Initial distribution of student behaviors
        self.behavior_distribution = {
            'minimal': 0.15,
            'low': 0.25,
            'medium': 0.35,
            'high': 0.25
        }

    def generate_student_population(self) -> pd.DataFrame:
        """
        Generate a synthetic student population with attributes.

        Returns:
            DataFrame with student information
        """
        students = []

        for i in range(self.config.total_students):
            # Determine student type
            is_boarding = i < self.config.boarding_students

            # Assign grade (9-12)
            grade = self.rng.choice([9, 10, 11, 12], p=[0.24, 0.26, 0.26, 0.24])

            # Assign initial waste behavior
            behavior = self.rng.choice(
                list(self.behavior_distribution.keys()),
                p=list(self.behavior_distribution.values())
            )

            # Assign dining hall preference
            dining_hall = self.rng.choice(self.config.dining_halls)

            students.append({
                'student_id': f'STU{i+1:04d}',
                'grade': grade,
                'is_boarding': is_boarding,
                'primary_dining_hall': dining_hall,
                'waste_behavior': behavior,
                'awareness_level': self.rng.uniform(0.1, 0.9)
            })

        return pd.DataFrame(students)

    def generate_meal_record(
        self,
        student: pd.Series,
        meal_type: str,
        date: datetime,
        awareness_factor: float = 1.0
    ) -> Dict:
        """
        Generate a single meal waste record for a student.

        Args:
            student: Student record
            meal_type: 'breakfast', 'lunch', or 'dinner'
            date: Date of the meal
            awareness_factor: Multiplier for waste reduction (1.0 = normal)

        Returns:
            Dictionary with meal record
        """
        # Get food quantity based on meal type
        if meal_type == 'breakfast':
            base_food = self.config.avg_food_taken_breakfast
        elif meal_type == 'lunch':
            base_food = self.config.avg_food_taken_lunch
        else:
            base_food = self.config.avg_food_taken_dinner

        # Add individual variation
        food_taken = base_food * self.rng.uniform(0.7, 1.3)

        # Determine waste percentage based on behavior
        behavior = student['waste_behavior']
        waste_range = self.waste_categories[behavior]
        base_waste_pct = self.rng.uniform(waste_range[0], waste_range[1])

        # Apply awareness factor (lower = less waste)
        waste_pct = base_waste_pct * (1 / awareness_factor)
        waste_pct = min(waste_pct, 0.6)  # Cap at 60%

        # Calculate waste
        food_wasted = food_taken * waste_pct
        food_consumed = food_taken - food_wasted

        return {
            'date': date,
            'student_id': student['student_id'],
            'meal_type': meal_type,
            'dining_hall': student['primary_dining_hall'],
            'food_taken_lbs': round(food_taken, 3),
            'food_consumed_lbs': round(food_consumed, 3),
            'food_wasted_lbs': round(food_wasted, 3),
            'waste_percentage': round(waste_pct * 100, 1),
            'grade': student['grade'],
            'is_boarding': student['is_boarding']
        }

    def generate_daily_data(
        self,
        students: pd.DataFrame,
        date: datetime,
        awareness_factor: float = 1.0,
        special_event: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Generate all meal records for a single day.

        Args:
            students: Student population DataFrame
            date: Date to generate data for
            awareness_factor: Awareness multiplier
            special_event: Name of special event (affects participation)

        Returns:
            DataFrame with all meal records for the day
        """
        records = []

        # Adjust participation for special events
        participation_modifier = 1.0
        if special_event == 'exam_week':
            participation_modifier = 0.85  # Students skip meals during exams
        elif special_event == 'holiday':
            participation_modifier = 0.3   # Many students away
        elif special_event == 'sports_event':
            participation_modifier = 1.1   # Higher participation

        meals = [
            ('breakfast', self.config.breakfast_participation),
            ('lunch', self.config.lunch_participation),
            ('dinner', self.config.dinner_participation)
        ]

        for meal_type, base_participation in meals:
            participation = base_participation * participation_modifier

            # Day students less likely to eat dinner on campus
            for _, student in students.iterrows():
                # Determine if student eats this meal
                student_participation = participation
                if not student['is_boarding'] and meal_type == 'dinner':
                    student_participation *= 0.3

                if self.rng.random() < student_participation:
                    record = self.generate_meal_record(
                        student, meal_type, date, awareness_factor
                    )
                    records.append(record)

        return pd.DataFrame(records)

    def generate_semester_data(
        self,
        start_date: datetime,
        n_weeks: int = 18,
        intervention_start_week: Optional[int] = None,
        intervention_type: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Generate a full semester of food waste data.

        Args:
            start_date: First day of semester
            n_weeks: Number of weeks in semester
            intervention_start_week: Week when intervention begins
            intervention_type: Type of intervention ('awareness', 'portion', 'incentive')

        Returns:
            DataFrame with all meal records for the semester
        """
        students = self.generate_student_population()
        all_records = []

        # Define special events (week number: event type)
        special_events = {
            4: 'exam_week',    # Midterm week
            9: 'holiday',      # Fall/Spring break
            17: 'exam_week',   # Finals week
        }

        current_date = start_date
        for week in range(n_weeks):
            # Check for special events
            event = special_events.get(week + 1)

            # Calculate awareness factor based on intervention
            awareness_factor = 1.0
            if intervention_start_week and week >= intervention_start_week:
                weeks_since_intervention = week - intervention_start_week
                if intervention_type == 'awareness':
                    # Gradual improvement
                    awareness_factor = 1.0 + 0.03 * weeks_since_intervention
                elif intervention_type == 'portion':
                    # Stronger initial effect
                    awareness_factor = 1.2 + 0.02 * weeks_since_intervention
                elif intervention_type == 'incentive':
                    # Strong sustained effect
                    awareness_factor = 1.3 + 0.01 * weeks_since_intervention

                awareness_factor = min(awareness_factor, 2.0)  # Cap at 2x

            # Generate data for each day of the week (skip weekends for day students)
            for day in range(7):
                day_records = self.generate_daily_data(
                    students, current_date, awareness_factor, event
                )
                all_records.append(day_records)
                current_date += timedelta(days=1)

        return pd.concat(all_records, ignore_index=True)

    def generate_annual_data(
        self,
        year: int = 2024,
        include_intervention: bool = True
    ) -> pd.DataFrame:
        """
        Generate a full academic year of data.

        Args:
            year: Academic year start
            include_intervention: Whether to simulate an intervention

        Returns:
            DataFrame with annual meal records
        """
        # Fall semester
        fall_start = datetime(year, 9, 5)
        fall_data = self.generate_semester_data(
            fall_start,
            n_weeks=18,
            intervention_start_week=6 if include_intervention else None,
            intervention_type='awareness' if include_intervention else None
        )
        fall_data['semester'] = 'Fall'

        # Spring semester
        spring_start = datetime(year + 1, 1, 15)
        spring_data = self.generate_semester_data(
            spring_start,
            n_weeks=18,
            intervention_start_week=1 if include_intervention else None,
            intervention_type='incentive' if include_intervention else None
        )
        spring_data['semester'] = 'Spring'

        return pd.concat([fall_data, spring_data], ignore_index=True)

    def aggregate_daily_summary(self, meal_data: pd.DataFrame) -> pd.DataFrame:
        """
        Create daily summary statistics from meal records.

        Args:
            meal_data: DataFrame with individual meal records

        Returns:
            DataFrame with daily summaries
        """
        daily = meal_data.groupby('date').agg({
            'student_id': 'count',
            'food_taken_lbs': 'sum',
            'food_consumed_lbs': 'sum',
            'food_wasted_lbs': 'sum',
            'waste_percentage': 'mean'
        }).reset_index()

        daily.columns = [
            'date', 'total_meals', 'total_food_taken_lbs',
            'total_food_consumed_lbs', 'total_food_wasted_lbs',
            'avg_waste_percentage'
        ]

        daily['actual_waste_percentage'] = (
            daily['total_food_wasted_lbs'] / daily['total_food_taken_lbs'] * 100
        )

        return daily

    def aggregate_weekly_summary(self, meal_data: pd.DataFrame) -> pd.DataFrame:
        """
        Create weekly summary statistics from meal records.

        Args:
            meal_data: DataFrame with individual meal records

        Returns:
            DataFrame with weekly summaries
        """
        meal_data = meal_data.copy()
        meal_data['week'] = pd.to_datetime(meal_data['date']).dt.isocalendar().week
        meal_data['year'] = pd.to_datetime(meal_data['date']).dt.year

        weekly = meal_data.groupby(['year', 'week']).agg({
            'student_id': 'count',
            'food_taken_lbs': 'sum',
            'food_consumed_lbs': 'sum',
            'food_wasted_lbs': 'sum',
            'waste_percentage': 'mean'
        }).reset_index()

        weekly.columns = [
            'year', 'week', 'total_meals', 'total_food_taken_lbs',
            'total_food_consumed_lbs', 'total_food_wasted_lbs',
            'avg_waste_percentage'
        ]

        weekly['actual_waste_percentage'] = (
            weekly['total_food_wasted_lbs'] / weekly['total_food_taken_lbs'] * 100
        )

        return weekly

    def export_to_csv(
        self,
        data: pd.DataFrame,
        filepath: str,
        include_summary: bool = True
    ) -> None:
        """
        Export generated data to CSV files.

        Args:
            data: DataFrame to export
            filepath: Base filepath (without extension)
            include_summary: Whether to also export summary files
        """
        # Export raw data
        data.to_csv(f"{filepath}_raw.csv", index=False)

        if include_summary:
            # Export daily summary
            daily = self.aggregate_daily_summary(data)
            daily.to_csv(f"{filepath}_daily_summary.csv", index=False)

            # Export weekly summary
            weekly = self.aggregate_weekly_summary(data)
            weekly.to_csv(f"{filepath}_weekly_summary.csv", index=False)

    def get_statistics_summary(self, data: pd.DataFrame) -> Dict:
        """
        Generate comprehensive statistics from meal data.

        Args:
            data: DataFrame with meal records

        Returns:
            Dictionary with summary statistics
        """
        total_food_taken = data['food_taken_lbs'].sum()
        total_food_wasted = data['food_wasted_lbs'].sum()

        return {
            'total_meals_served': len(data),
            'unique_students': data['student_id'].nunique(),
            'total_food_taken_lbs': round(total_food_taken, 2),
            'total_food_consumed_lbs': round(data['food_consumed_lbs'].sum(), 2),
            'total_food_wasted_lbs': round(total_food_wasted, 2),
            'overall_waste_percentage': round(total_food_wasted / total_food_taken * 100, 2),
            'avg_waste_per_meal_lbs': round(data['food_wasted_lbs'].mean(), 4),
            'avg_waste_percentage_per_meal': round(data['waste_percentage'].mean(), 2),
            'median_waste_percentage': round(data['waste_percentage'].median(), 2),
            'waste_by_meal_type': data.groupby('meal_type')['waste_percentage'].mean().to_dict(),
            'waste_by_grade': data.groupby('grade')['waste_percentage'].mean().to_dict(),
            'waste_by_dining_hall': data.groupby('dining_hall')['waste_percentage'].mean().to_dict()
        }


if __name__ == "__main__":
    # Demo: Generate sample data
    print("Generating Lawrenceville School Food Waste Data...")
    print("=" * 60)

    generator = FoodWasteDataGenerator()

    # Generate student population
    students = generator.generate_student_population()
    print(f"\nGenerated {len(students)} students")
    print(f"  Boarding: {students['is_boarding'].sum()}")
    print(f"  Day: {(~students['is_boarding']).sum()}")
    print(f"\nInitial waste behavior distribution:")
    print(students['waste_behavior'].value_counts(normalize=True))

    # Generate one week of data
    print("\n" + "=" * 60)
    print("Generating one week of meal data...")
    start = datetime(2024, 9, 9)
    weekly_data = generator.generate_semester_data(start, n_weeks=1)

    stats = generator.get_statistics_summary(weekly_data)
    print(f"\nWeek Summary:")
    print(f"  Total meals: {stats['total_meals_served']}")
    print(f"  Total food taken: {stats['total_food_taken_lbs']:.1f} lbs")
    print(f"  Total food wasted: {stats['total_food_wasted_lbs']:.1f} lbs")
    print(f"  Overall waste rate: {stats['overall_waste_percentage']:.1f}%")
    print(f"\nWaste by meal type:")
    for meal, pct in stats['waste_by_meal_type'].items():
        print(f"  {meal}: {pct:.1f}%")
