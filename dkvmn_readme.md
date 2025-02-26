# Running the DKVMN Implementation

## Prerequisites
To run this DKVMN implementation, follow these steps:

### 1. Save the Code
Save the implementation to a Python file (e.g., `dkvmn.py`).

### 2. Install Dependencies
Ensure you have the required dependencies installed by running:

```bash
pip install torch numpy scikit-learn
```

## Running the Code
For a quick test with the included dummy data, simply execute:

```bash
python dkvmn.py
```
This runs the `main()` function, creating a model with dummy data and training it for 5 epochs.

## Using Your Own Educational Data
To apply this implementation to your dataset, follow these steps:

### 1. Prepare Your Data
Ensure your data is formatted correctly:
- **Questions/skills** represented as integers.
- **Answers** represented as `0` (incorrect) or `1` (correct).
- **Data** organized in sequences (e.g., each student's interaction history).

### 2. Modify the `main()` Function to Load Your Data
Replace the dummy data with your dataset:

```python
from torch.utils.data import TensorDataset, DataLoader

# Load your data
# q_data shape: [num_students, seq_len]
# a_data shape: [num_students, seq_len]
q_data = your_question_data
a_data = your_answer_data

train_dataset = TensorDataset(q_data, a_data)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Split into train/validation/test
from sklearn.model_selection import train_test_split
# (Add code for proper data splitting)
```

### 3. Adjust Hyperparameters
Modify the hyperparameters to match your dataset:

```python
# Set these based on your dataset
num_questions = YOUR_TOTAL_QUESTION_COUNT
dim_key = 64  # Tunable
dim_value = 128  # Tunable
dim_hidden = 64  # Tunable
```

### 4. Enhance Training
For a more complete training script, consider adding:
- **Model checkpointing** to save the best model.
- **Early stopping** based on validation loss.
- **Learning rate scheduling** for better convergence.

By following these steps, you can customize DKVMN for your own educational datasets efficiently.

