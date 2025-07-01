#!/usr/bin/env python3
"""
Synthetic Healthcare Data Generator

This script generates realistic synthetic healthcare data with the following features:
- Patient demographics: age, gender, race
- Medical info: diagnosis code, procedure code, length of stay
- Treatment plan: treatment type, insurance type
- Target: total charges

Author: Sneha Dutt
Date: 2025
"""

import pandas as pd
import numpy as np
from faker import Faker
import warnings
warnings.filterwarnings('ignore')

def generate_realistic_healthcare_data(n_samples=1000, seed=42):
    """
    Generate realistic healthcare data with proper distributions and relationships.
    
    Args:
        n_samples (int): Number of samples to generate
        seed (int): Random seed for reproducibility
    
    Returns:
        pd.DataFrame: Generated healthcare data
    """
    # Set random seed for reproducibility
    np.random.seed(seed)
    fake = Faker()
    Faker.seed(seed)
    
    # Patient demographics with realistic distributions
    ages = np.random.normal(55, 18, n_samples)
    ages = np.clip(ages, 18, 95).astype(int)
    
    genders = np.random.choice(['Male', 'Female'], n_samples, p=[0.48, 0.52])
    races = np.random.choice(['White', 'Black', 'Hispanic', 'Asian', 'Other'], 
                            n_samples, p=[0.60, 0.13, 0.18, 0.06, 0.03])
    
    # Medical diagnosis codes (ICD-10) - common healthcare conditions
    diagnosis_codes = [
        'I10', 'E11.9', 'J45.909', 'I50.9', 'E78.5',  # Hypertension, Diabetes, Asthma, Heart failure, Hyperlipidemia
        'K21.9', 'N18.9', 'I25.10', 'E03.9', 'M79.3',  # GERD, CKD, CAD, Hypothyroidism, Back pain
        'F41.9', 'I63.9', 'C50.919', 'E66.9', 'I48.91'  # Anxiety, Stroke, Breast cancer, Obesity, A-fib
    ]
    
    # Procedure codes (ICD-10-PCS) - common medical procedures
    procedure_codes = [
        '0U5B7ZZ', '3E0P3MZ', '0WQF0ZZ', '4A02X4Z', '0D160Z4',  # Various procedures
        '0U5B8ZZ', '3E0P3NZ', '0WQF1ZZ', '4A02X5Z', '0D160Z5'
    ]
    
    # Length of stay with realistic exponential distribution
    length_of_stay = np.random.exponential(3, n_samples)
    length_of_stay = np.clip(length_of_stay, 1, 30).astype(int)
    
    # Treatment types with realistic probabilities
    treatment_types = np.random.choice(['Surgery', 'Medical Therapy', 'Observation', 'Emergency Care', 'Rehabilitation'], 
                                       n_samples, p=[0.25, 0.35, 0.20, 0.15, 0.05])
    
    # Insurance types with realistic distribution
    insurance_types = np.random.choice(['Medicare', 'Private Insurance', 'Medicaid', 'Uninsured'], 
                                       n_samples, p=[0.40, 0.35, 0.20, 0.05])
    
    # Create the dataset
    data = pd.DataFrame({
        'age': ages,
        'gender': genders,
        'race': races,
        'diagnosis_code': np.random.choice(diagnosis_codes, n_samples),
        'procedure_code': np.random.choice(procedure_codes, n_samples),
        'length_of_stay': length_of_stay,
        'treatment_type': treatment_types,
        'insurance_type': insurance_types
    })
    
    # Generate realistic total charges based on other features
    base_charge = 5000
    age_factor = (data['age'] - 40) / 20  # Older patients cost more
    los_factor = data['length_of_stay'] * 800  # Daily rate
    
    # Create treatment factor mapping
    treatment_factors = {
        'Surgery': 1.8, 'Medical Therapy': 1.0, 'Observation': 0.7, 
        'Emergency Care': 1.5, 'Rehabilitation': 1.2
    }
    treatment_factor = data['treatment_type'].map(treatment_factors)
    
    # Create insurance factor mapping
    insurance_factors = {
        'Medicare': 0.9, 'Private Insurance': 1.1, 'Medicaid': 0.8, 'Uninsured': 0.7
    }
    insurance_factor = data['insurance_type'].map(insurance_factors)
    
    # Add some randomness
    noise = np.random.normal(0, 0.2, n_samples)
    
    # Calculate total charges
    total_charges = (base_charge + age_factor * 1000 + los_factor) * treatment_factor * insurance_factor * (1 + noise)
    total_charges = np.clip(total_charges, 1000, 50000)
    
    data['total_charges'] = np.round(total_charges, 2)
    
    return data

def generate_synthetic_data_with_sdv(data, scale=10):
    """
    Generate synthetic data using SDV (Synthetic Data Vault).
    
    Args:
        data (pd.DataFrame): Original data to learn from
        scale (int): Scale factor for synthetic data generation
    
    Returns:
        pd.DataFrame: Synthetic data
    """
    try:
        from sdv.single_table import GaussianCopulaSynthesizer
        from sdv.metadata import SingleTableMetadata
        
        # Create metadata
        metadata = SingleTableMetadata()
        metadata.detect_from_dataframe(data=data)
        
        # Initialize and fit the synthesizer
        synthesizer = GaussianCopulaSynthesizer(metadata)
        synthesizer.fit(data)
        
        # Generate synthetic data
        synthetic_data = synthesizer.sample(num_rows=len(data)*scale)
        
        print("✓ SDV synthetic data generation successful!")
        return synthetic_data
        
    except ImportError:
        return print("Data couldn't be generated :/")

def print_data_summary(data, title="Dataset"):
    """Print comprehensive data summary."""
    print(f"\n{'='*50}")
    print(f"{title.upper()} SUMMARY")
    print(f"{'='*50}")
    print(f"Total records: {len(data):,}")
    print(f"Total columns: {len(data.columns)}")
    
    print(f"\n{'='*30}")
    print("NUMERICAL FEATURES")
    print(f"{'='*30}")
    numerical_cols = ['age', 'length_of_stay', 'total_charges']
    print(data[numerical_cols].describe())
    
    print(f"\n{'='*30}")
    print("CATEGORICAL FEATURES")
    print(f"{'='*30}")
    categorical_cols = ['gender', 'race', 'diagnosis_code', 'procedure_code', 'treatment_type', 'insurance_type']
    for col in categorical_cols:
        print(f"\n{col.upper()}:")
        value_counts = data[col].value_counts()
        for value, count in value_counts.items():
            percentage = (count / len(data)) * 100
            print(f"  {value}: {count:,} ({percentage:.1f}%)")

def main():
    """Main function to generate and save synthetic healthcare data."""
    print(" SYNTHETIC HEALTHCARE DATA GENERATOR")
    print("="*50)
    
    # Generate original dataset
    print("\n Generating original healthcare dataset...")
    original_data = generate_realistic_healthcare_data(n_samples=1000, seed=42)
    
    # Print original data summary
    print_data_summary(original_data, "Original Dataset")
    
    # Save original data
    original_data.to_csv('original_healthcare_data.csv', index=False)
    print(f"\n Original dataset saved as 'original_healthcare_data.csv'")
    
    # Generate synthetic data
    print(f"\n Generating synthetic data (10x scale)...")
    synthetic_data = generate_synthetic_data_with_sdv(original_data, scale=10)
    
    # Print synthetic data summary
    print_data_summary(synthetic_data, "Synthetic Dataset")
    
    # Save synthetic data
    synthetic_data.to_csv('synthetic_healthcare_data.csv', index=False)
    print(f"\n Synthetic dataset saved as 'synthetic_healthcare_data.csv'")
    
    # Comparison
    print(f"\n{'='*50}")
    print("COMPARISON: ORIGINAL vs SYNTHETIC")
    print(f"{'='*50}")
    print(f"Original data records: {len(original_data):,}")
    print(f"Synthetic data records: {len(synthetic_data):,}")
    print(f"Scale factor: {len(synthetic_data) / len(original_data):.1f}x")
    
    print(f"\n{'='*40}")
    print("NUMERICAL FEATURES COMPARISON")
    print(f"{'='*40}")
    comparison_cols = ['age', 'length_of_stay', 'total_charges']
    for col in comparison_cols:
        print(f"\n{col.upper()}:")
        print(f"  Original  - Mean: {original_data[col].mean():.2f}, Std: {original_data[col].std():.2f}")
        print(f"  Synthetic - Mean: {synthetic_data[col].mean():.2f}, Std: {synthetic_data[col].std():.2f}")
    
    # Data quality check
    print(f"\n{'='*40}")
    print("DATA QUALITY CHECK")
    print(f"{'='*40}")
    print(f"Missing values in synthetic data:")
    missing_counts = synthetic_data.isnull().sum()
    if missing_counts.sum() == 0:
        print("  ✓ No missing values found")
    else:
        print(missing_counts[missing_counts > 0])
    
    print(f"\nValue ranges in synthetic data:")
    print(f"  Age: {synthetic_data['age'].min()} - {synthetic_data['age'].max()}")
    print(f"  Length of stay: {synthetic_data['length_of_stay'].min()} - {synthetic_data['length_of_stay'].max()}")
    print(f"  Total charges: ${synthetic_data['total_charges'].min():,.2f} - ${synthetic_data['total_charges'].max():,.2f}")
    
    print(f"\n Synthetic healthcare data generation complete!")
    print(f" Files created:")
    print(f"   - original_healthcare_data.csv ({len(original_data):,} records)")
    print(f"   - synthetic_healthcare_data.csv ({len(synthetic_data):,} records)")

if __name__ == "__main__":
    main()
