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
