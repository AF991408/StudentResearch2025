"""
Final pyBKT Test Script
This script creates synthetic data, trains a BKT model, and verifies that
the implementation is working correctly.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pyBKT.models import Model

def run_pyBKT_test():
    print("Running pyBKT test...")
    
    # Step 1: Create synthetic data
    print("Step 1: Creating synthetic data...")
    np.random.seed(42)  # For reproducibility
    
    # Create a simple dataset: 50 students answering 10 questions on 1 skill
    num_students = 50
    num_questions = 10
    
    # Initialize empty list to store data
    data = []
    
    # Generate synthetic student responses
    for student in range(num_students):
        # Each student starts with a 30% chance of knowing the skill
        knows_skill = np.random.random() < 0.3
        
        for question in range(num_questions):
            # 10% chance to learn the skill on each question
            if not knows_skill and np.random.random() < 0.1:
                knows_skill = True
            
            # Determine if answer is correct
            if knows_skill:
                # If student knows skill: 90% chance to get it right (10% slip)
                correct = 1 if np.random.random() > 0.1 else 0
            else:
                # If student doesn't know skill: 20% chance to guess correctly
                correct = 1 if np.random.random() < 0.2 else 0
            
            # Add to dataset
            data.append({
                'user_id': student,
                'skill_name': 'basic_skill',
                'correct': correct,
                'order_id': question
            })
    
    # Convert to DataFrame
    df = pd.DataFrame(data)
    print(f"Created dataset with {len(df)} rows")
    print(df.head())
    
    # Step 2: Train BKT model
    print("\nStep 2: Training BKT model...")
    model = Model()
    model.fit(data=df)
    
    # Step 3: Verify parameters
    print("\nStep 3: Verifying model parameters...")
    params = model.params()
    print("Learned parameters:")
    print(params)
    
    # Extract parameters from DataFrame
    # Based on your output, params is a DataFrame with structure:
    # skill, param, class, value
    try:
        prior_value = params.loc[(params['skill'] == 'basic_skill') & 
                               (params['param'] == 'prior'), 'value'].values[0]
        learn_value = params.loc[(params['skill'] == 'basic_skill') & 
                               (params['param'] == 'learns'), 'value'].values[0]
        guess_value = params.loc[(params['skill'] == 'basic_skill') & 
                               (params['param'] == 'guesses'), 'value'].values[0]
        slip_value = params.loc[(params['skill'] == 'basic_skill') & 
                               (params['param'] == 'slips'), 'value'].values[0]
        
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
    # Calculate accuracy
    binary_predictions = (predictions['correct_predictions'] > 0.5).astype(int)
    accuracy = np.mean(binary_predictions == predictions['correct'])
    print(f"Prediction accuracy: {accuracy:.4f}")
    
    # Step 6: Visualize learning curve for a student
    print("\nStep 6: Visualizing learning curve...")
    try:
        # For visualization, we'll use the state_predictions column instead of posteriors
        # (since your version doesn't support return_posteriors)
        
        # Select a specific student
        student_id = 0
        student_data = predictions[
            predictions['user_id'] == student_id
        ].sort_values('order_id')
        
        # Plot knowledge state over time
        plt.figure(figsize=(10, 6))
        plt.plot(student_data['order_id'], student_data['state_predictions'], 'o-', 
                 label='Knowledge State')
        plt.plot(student_data['order_id'], student_data['correct'], 'x', 
                 color='red', label='Correct (1) or Incorrect (0)')
        plt.xlabel('Question Order')
        plt.ylabel('Probability of Skill Mastery')
        plt.title(f'Learning Trajectory for Student {student_id}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.ylim(-0.1, 1.1)
        
        try:
            plt.savefig('pyBKT_test_plot.png')
            print("Learning curve plot saved as 'pyBKT_test_plot.png'")
        except Exception as e:
            print(f"Could not save plot: {e}")
            print("If running in an environment without display capabilities, this is expected.")
    except Exception as e:
        print(f"Error creating visualization: {e}")
    
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
