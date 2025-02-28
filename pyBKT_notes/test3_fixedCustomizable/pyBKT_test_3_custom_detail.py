"""
Final pyBKT Test Script (Using Fixed Data)
This script loads real student response data, trains a BKT model with improved parameters,
evaluates performance, and visualizes student learning curves.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pyBKT.models import Model

def run_pyBKT_test():
    print("Running pyBKT test...")

    # Step 1: Load fixed data
    print("\nStep 1: Loading fixed data from 'fixed_data.csv'...")
    try:
        df = pd.read_csv(r"C:\Users\footb\Desktop\StudentResearch2025\pyBKT_notes\fixed_data.csv")
        print(f"Loaded dataset with {len(df)} rows")
        print(df.head())
    except Exception as e:
        print(f"Error loading data: {e}")
        return

# Step 2: Train BKT model with modified parameters
    print("\nStep 2: Training BKT model with modified parameters...")

# Directly pass the parameters to the `defaults` argument in the fit method
    model = Model()
    model.fit(
        data=df,
        defaults={
            'prior': 0.3,     # 30% initial knowledge probability
            'learns': 0.1,    # 10% learning probability per question
            'guesses': 0.2,   # 20% chance of guessing correctly
            'slips': 0.1,     # 10% chance of making a mistake
            'forgets': 0.0    # No forgetting
        }
    )





# Step 3: Verifying model parameters
    print("\nStep 3: Verifying model parameters...")
    params = model.params()
    print("Learned parameters:")
    print(params)

# Modify the extraction process to handle different structure of the params
    try:
    # Check the structure of params
        for _, row in params.iterrows():
            if row['param'] == 'prior':
                prior_value = row['value']
            elif row['param'] == 'learns':
                learn_value = row['value']
            elif row['param'] == 'guesses':
                guess_value = row['value']
            elif row['param'] == 'slips':
                slip_value = row['value']
    
        print("\nExtracted Parameters:")
        print(f"Prior: {prior_value:.5f} (should be around 0.3)")
        print(f"Learn: {learn_value:.5f} (should be around 0.1)")
        print(f"Guess: {guess_value:.5f} (should be around 0.2)")
        print(f"Slip: {slip_value:.5f} (should be around 0.1)")
    except Exception as e:
        print(f"Could not extract individual parameters: {e}")


    # Step 4: Make predictions
    print("\nStep 4: Making predictions...")
    predictions = model.predict(data=df)
    print("Prediction examples:")
    print(predictions.head())

    # Step 5: Evaluate model performance
    print("\nStep 5: Evaluating model performance...")

    # Compute accuracy
    binary_predictions = (predictions['correct_predictions'] > 0.5).astype(int)
    accuracy = (binary_predictions == predictions['correct']).mean()
    print(f"Updated Prediction Accuracy: {accuracy:.4f}")

    # Step 6: Visualize learning curves for multiple students
    print("\nStep 6: Visualizing learning curves for multiple students...")

    students_to_plot = df['user_id'].unique()[:5]  # First 5 students

    plt.figure(figsize=(12, 6))

    for student_id in students_to_plot:
        student_data = predictions[predictions['user_id'] == student_id].sort_values('order_id')
        plt.plot(student_data['order_id'], student_data['state_predictions'], marker='o', label=f'Student {student_id}')

    plt.xlabel('Question Order')
    plt.ylabel('Probability of Skill Mastery')
    plt.title('Learning Curves for Multiple Students')
    plt.legend()
    plt.grid(True)

    try:
        plt.savefig('pyBKT_learning_curves.png')
        print("Learning curves plot saved as 'pyBKT_learning_curves.png'")
    except Exception as e:
        print(f"Could not save plot: {e}")

    plt.show()

    print("\nTest completed successfully!")
    return {
        'accuracy': accuracy,
        'parameters': params,
        'model': model
    }

if __name__ == "__main__":
    try:
        results = run_pyBKT_test()
        print("\nSUCCESS: pyBKT is correctly installed and functioning!")
    except Exception as e:
        print("\nERROR: Test failed with the following error:")
        print(e)
        print("\nPlease check your installation and try again.")
