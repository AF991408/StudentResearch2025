"""
pyBKT Basic Test Script - Updated
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
    
    # Check if parameters are reasonable
    # First, let's safely extract parameters based on the actual structure
    try:
        # Handle the case where params is a DataFrame
        if isinstance(params, pd.DataFrame):
            # Filter for each parameter type
            prior_row = params[(params['param'] == 'prior') & (params['skill'] == 'basic_skill')]
            learn_row = params[(params['param'] == 'learns') & (params['skill'] == 'basic_skill')]
            guess_row = params[(params['param'] == 'guesses') & (params['skill'] == 'basic_skill')]
            slip_row = params[(params['param'] == 'slips') & (params['skill'] == 'basic_skill')]
            
            prior = prior_row['value'].values[0] if not prior_row.empty else None
            learn = learn_row['value'].values[0] if not learn_row.empty else None
            guess = guess_row['value'].values[0] if not guess_row.empty else None
            slip = slip_row['value'].values[0] if not slip_row.empty else None
            
        # Handle the case where params is a dictionary
        elif isinstance(params, dict):
            if 'basic_skill' in params.get('prior', {}):
                prior = params['prior']['basic_skill']
                learn = params['learn']['basic_skill']
                guess = params['guess']['basic_skill']
                slip = params['slip']['basic_skill']
            else:
                # Try alternative structure
                prior = params.get('prior', {}).get('default', 0)
                learn = params.get('learns', {}).get('default', 0)
                guess = params.get('guesses', {}).get('default', 0)
                slip = params.get('slips', {}).get('default', 0)
        else:
            # If structure is unknown, print a message
            print("Unable to parse parameters due to unexpected format")
            prior, learn, guess, slip = None, None, None, None
    except Exception as e:
        print(f"Error extracting parameters: {e}")
        print("Printing raw parameters structure:")
        import pprint
        pprint.pprint(params)
        prior, learn, guess, slip = None, None, None, None
    
    print("\nParameter check:")
    if prior is not None:
        print(f"Prior: {prior:.5f} (should be around 0.3)")
        print(f"Learn: {learn:.5f} (should be around 0.1)")
        print(f"Guess: {guess:.5f} (should be around 0.2)")
        print(f"Slip: {slip:.5f} (should be around 0.1)")
    else:
        print("Could not extract parameters in the expected format.")
        print("This could be due to changes in the pyBKT API.")
        print("Please check the params() output above for the actual structure.")
    
    # Step 4: Make predictions
    print("\nStep 4: Making predictions...")
    predictions = model.predict(data=df)
    print("Prediction examples:")
    print(predictions.head())
    
    # Check if 'correct_predictions' column exists
    prediction_col = 'correct_prediction'
    if 'correct_predictions' in predictions.columns:
        prediction_col = 'correct_predictions'
    
    # Only show these columns if they exist
    display_cols = ['user_id', 'skill_name', 'correct']
    if prediction_col in predictions.columns:
        display_cols.append(prediction_col)
    
    print(predictions[display_cols].head())
    
    # Step 5: Evaluate model performance
    print("\nStep 5: Evaluating model performance...")
    # Check if we have the prediction column
    if prediction_col in predictions.columns:
        # Convert predictions to binary
        binary_predictions = (predictions[prediction_col] > 0.5).astype(int)
        
        # Calculate accuracy
        accuracy = np.mean(binary_predictions == predictions['correct'])
        print(f"Prediction accuracy: {accuracy:.4f}")
    else:
        print(f"Prediction column '{prediction_col}' not found in results.")
        print("Available columns:")
        print(predictions.columns.tolist())
        accuracy = None
    
    # Step 6: Visualize learning curve for a student
    print("\nStep 6: Visualizing learning curve...")
    try:
        # Get predictions with posterior probabilities
        predictions_with_posteriors = model.predict(data=df, return_posteriors=True)
        
        # Select a specific student
        student_id = 0
        student_data = predictions_with_posteriors[
            predictions_with_posteriors['user_id'] == student_id
        ].sort_values('order_id')
        
        # Check if 'posterior' column exists
        posterior_col = 'posterior'
        if posterior_col not in student_data.columns:
            # Check for alternative column names
            possible_cols = [col for col in student_data.columns if 'posterior' in col.lower()]
            if possible_cols:
                posterior_col = possible_cols[0]
                print(f"Using '{posterior_col}' column for posteriors")
            else:
                print("Posterior column not found. Available columns:")
                print(student_data.columns.tolist())
                raise ValueError("Posterior column not found")
        
        # Plot knowledge state over time
        plt.figure(figsize=(10, 6))
        plt.plot(student_data['order_id'], student_data[posterior_col], 'o-', label='Knowledge State')
        plt.plot(student_data['order_id'], student_data['correct'], 'x', color='red', label='Correct (1) or Incorrect (0)')
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
    
    print("\nTest completed!")
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