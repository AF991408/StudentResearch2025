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

# DKVMN Model Results and Metrics

The results you'll see from running the DKVMN model provide insights into how well the model is predicting student responses. After training, you'll see something like:

```
Epoch: 1, Loss: 0.XXXX  
Epoch: 2, Loss: 0.XXXX  
...  
AUC: 0.XXXX, Accuracy: 0.XXXX  
```

## Metrics Breakdown

### Training Loss
- **What it measures**: How well the model is fitting the training data.  
- **Interpretation**:
  - Lower values indicate a better fit.
  - You should see this decrease over epochs, which means the model is learning.
  - If the loss plateaus (stops decreasing), it suggests the model has learned as much as it can from the data.

---

### AUC (Area Under the ROC Curve)
- **What it measures**: The model's ability to discriminate between correct and incorrect responses.  
- **Range**:
  - **0.5**: The model is no better than random guessing.
  - **0.7-0.8**: Acceptable discrimination.
  - **0.8-0.9**: Excellent discrimination.
  - **>0.9**: Outstanding discrimination.
- **Note**: In educational data mining, AUC values around 0.7-0.8 are often considered good.

---

### Accuracy
- **What it measures**: The proportion of predictions that match the actual student responses.  
- **Range**:
  - **0.0 to 1.0** (or **0% to 100%**).
- **Caution**:  
  - Accuracy can be misleading if there's class imbalance (e.g., if students answer correctly most of the time).  
  - This is why AUC is typically preferred, as it is not affected by class imbalance.

---

## Interpreting the Results

### For Dummy Data (Randomly Generated):
- **AUC**: Near 0.5 (random chance).  
- **Accuracy**: Around 0.5.  
- **Why**: This is expected since the data is randomly generated.

---

### For Real Educational Data:
- **Good results**:
  - Decreasing loss over training epochs.
  - **AUC**: 0.7 or higher.
  - **Accuracy**: Significantly better than the majority class baseline.

---

The DKVMN model specifically learns to track student knowledge states across time. When properly trained on real data, it should be able to predict future student performance based on their interaction history.
