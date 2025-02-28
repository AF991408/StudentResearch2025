import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pyBKT.models import Model

def run_pyBKT_test():
    print("Running pyBKT test...")

    # Step 1: Load fixed data from CSV
    print("Step 1: Loading fixed data from 'fixed_data.csv'...")
    try:
        df = pd.read_csv(r"C:\Users\footb\Desktop\StudentResearch2025\pyBKT_notes\fixed_data.csv")
        print(f"Loaded dataset with {len(df)} rows")
        print(df.head())
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # Step 2: Train BKT model
    print("\nStep 2: Training BKT model...")
    model = Model()
    model.fit(data=df)

    # Step 3: Verify parameters
    print("\nStep 3: Verifying model parameters...")
    params = model.params()
    print("Learned parameters:")
    print(params)

    # Step 4: Make predictions
    print("\nStep 4: Making predictions...")
    predictions = model.predict(data=df)
    print("Prediction examples:")
    print(predictions.head())

    # Step 5: Evaluate model performance
    print("\nStep 5: Evaluating model performance...")
    binary_predictions = (predictions['correct_predictions'] > 0.5).astype(int)
    accuracy = np.mean(binary_predictions == predictions['correct'])
    print(f"Prediction accuracy: {accuracy:.4f}")

    # Step 6: Visualize learning curve
    print("\nStep 6: Visualizing learning curve...")
    try:
        student_id = df['user_id'].iloc[0]  # Choose the first student in the dataset
        student_data = predictions[predictions['user_id'] == student_id].sort_values('order_id')

        plt.figure(figsize=(10, 6))
        plt.plot(student_data['order_id'], student_data['state_predictions'], 'o-', label='Knowledge State')
        plt.plot(student_data['order_id'], student_data['correct'], 'x', color='red', label='Correct (1) or Incorrect (0)')
        plt.xlabel('Question Order')
        plt.ylabel('Probability of Skill Mastery')
        plt.title(f'Learning Trajectory for Student {student_id}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.ylim(-0.1, 1.1)

        plt.savefig('pyBKT_test_plot.png')
        print("Learning curve plot saved as 'pyBKT_test_plot.png'")

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
        print("\nPlease check your data and try again.")
