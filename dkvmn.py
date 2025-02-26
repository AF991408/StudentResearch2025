import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class DKVMN(nn.Module):
    def __init__(self, num_questions, dim_key, dim_value, dim_hidden, dropout_rate=0.2):
        """
        Dynamic Key-Value Memory Network
        
        Parameters:
        -----------
        num_questions: int
            Total number of questions/skills in the dataset
        dim_key: int
            Dimension of the key vectors
        dim_value: int
            Dimension of the value vectors
        dim_hidden: int
            Dimension of the hidden layer
        dropout_rate: float
            Dropout rate for regularization
        """
        super(DKVMN, self).__init__()
        
        self.num_questions = num_questions
        self.dim_key = dim_key
        self.dim_value = dim_value
        self.dim_hidden = dim_hidden
        self.dropout_rate = dropout_rate
        
        # Memory size (number of memory slots)
        self.memory_size = 50
        
        # Key Memory - static, stores concept/skill vectors
        self.key_memory = nn.Parameter(torch.zeros(self.memory_size, self.dim_key))
        nn.init.kaiming_normal_(self.key_memory)
        
        # Value Memory - dynamic, stores student knowledge state
        self.value_memory = nn.Parameter(torch.zeros(self.memory_size, self.dim_value))
        nn.init.kaiming_normal_(self.value_memory)
        
        # Question embedding matrix
        self.question_embed = nn.Embedding(self.num_questions + 1, self.dim_key)
        
        # Memory operations
        self.erase = nn.Linear(self.dim_hidden, self.dim_value)
        self.add = nn.Linear(self.dim_hidden, self.dim_value)
        
        # Additional network layers - THE FIX IS HERE
        # Correctly set the input dimension to dim_value + dim_key
        self.f_t = nn.Linear(self.dim_value + self.dim_key, self.dim_hidden)
        self.p_t = nn.Linear(self.dim_hidden, 1)
        
        # Dropout layer
        self.dropout = nn.Dropout(self.dropout_rate)
        
    def attention(self, q_embed):
        """
        Calculate attention weights using key memory and question embedding
        
        Parameters:
        -----------
        q_embed: Tensor [batch_size, dim_key]
            Question embedding
            
        Returns:
        --------
        correlation_weight: Tensor [batch_size, memory_size]
            Attention weights for each memory slot
        """
        # Compute correlation weight using dot product and softmax
        correlation_weight = torch.matmul(q_embed, self.key_memory.T)  # [batch_size, memory_size]
        correlation_weight = F.softmax(correlation_weight, dim=1)
        
        return correlation_weight
    
    def read(self, correlation_weight):
        """
        Read from the value memory using attention weights
        
        Parameters:
        -----------
        correlation_weight: Tensor [batch_size, memory_size]
            Attention weights for each memory slot
            
        Returns:
        --------
        read_content: Tensor [batch_size, dim_value]
            The content read from the memory
        """
        # Read from value memory
        read_content = torch.matmul(correlation_weight, self.value_memory)  # [batch_size, dim_value]
        
        return read_content
    
    def write(self, correlation_weight, student_state, value_memory):
        """
        Write to the value memory using erase and add vectors
        
        Parameters:
        -----------
        correlation_weight: Tensor [batch_size, memory_size]
            Attention weights for each memory slot
        student_state: Tensor [batch_size, dim_hidden]
            Student knowledge state
        value_memory: Tensor [batch_size, memory_size, dim_value]
            Current value memory
            
        Returns:
        --------
        value_memory_updated: Tensor [batch_size, memory_size, dim_value]
            Updated value memory
        """
        # Compute erase and add vectors
        erase_vector = torch.sigmoid(self.erase(student_state))  # [batch_size, dim_value]
        add_vector = torch.tanh(self.add(student_state))  # [batch_size, dim_value]
        
        # Reshape correlation weight for broadcasting
        correlation_weight = correlation_weight.unsqueeze(2)  # [batch_size, memory_size, 1]
        
        # Reshape erase and add vectors for broadcasting
        erase_vector = erase_vector.unsqueeze(1)  # [batch_size, 1, dim_value]
        add_vector = add_vector.unsqueeze(1)  # [batch_size, 1, dim_value]
        
        # Erase operation
        erase_term = correlation_weight * erase_vector  # [batch_size, memory_size, dim_value]
        value_memory_erased = value_memory * (1 - erase_term)
        
        # Add operation
        add_term = correlation_weight * add_vector  # [batch_size, memory_size, dim_value]
        value_memory_updated = value_memory_erased + add_term
        
        return value_memory_updated
    
    def forward(self, q, a):
        """
        Forward pass of the DKVMN model
        
        Parameters:
        -----------
        q: Tensor [batch_size, seq_len]
            Question sequence
        a: Tensor [batch_size, seq_len]
            Answer sequence (0 or 1)
            
        Returns:
        --------
        pred: Tensor [batch_size, seq_len]
            Prediction of the probability of answering correctly
        """
        batch_size, seq_len = q.shape
        
        # Initialize value memory for the batch
        value_memory = self.value_memory.unsqueeze(0).repeat(batch_size, 1, 1)  # [batch_size, memory_size, dim_value]
        
        predictions = []
        
        for t in range(seq_len):
            # Get question and answer at time t
            q_t = q[:, t]  # [batch_size]
            a_t = a[:, t].unsqueeze(1)  # [batch_size, 1]
            
            # Get question embedding
            q_embed = self.question_embed(q_t)  # [batch_size, dim_key]
            
            # Compute attention using the current question
            correlation_weight = self.attention(q_embed)  # [batch_size, memory_size]
            
            # Read from memory using the attention weights
            read_content = self.read(correlation_weight)  # [batch_size, dim_value]
            
            # Concatenate the read content with the question embedding
            concat_feature = torch.cat([read_content, q_embed], dim=1)  # [batch_size, dim_value + dim_key]
            
            # Apply dropout
            concat_feature = self.dropout(concat_feature)
            
            # Get prediction
            student_state = F.relu(self.f_t(concat_feature))  # [batch_size, dim_hidden]
            pred = torch.sigmoid(self.p_t(student_state))  # [batch_size, 1]
            
            predictions.append(pred)
            
            # Update value memory
            value_memory = self.write(correlation_weight, student_state, value_memory)
        
        # Stack predictions
        pred = torch.cat(predictions, dim=1)  # [batch_size, seq_len]
        
        return pred

class DKVMNTrainer:
    def __init__(self, model, optimizer, criterion, device):
        """
        Trainer for DKVMN model
        
        Parameters:
        -----------
        model: DKVMN
            The DKVMN model
        optimizer: torch.optim
            Optimizer for training
        criterion: loss function
            Loss function for training
        device: torch.device
            Device to run training on
        """
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        
    def train(self, train_loader, epochs):
        """
        Train the DKVMN model
        
        Parameters:
        -----------
        train_loader: DataLoader
            DataLoader for training data
        epochs: int
            Number of epochs to train
            
        Returns:
        --------
        losses: list
            List of training losses
        """
        self.model.train()
        losses = []
        
        for epoch in range(epochs):
            epoch_loss = 0
            for batch_idx, (q_data, a_data) in enumerate(train_loader):
                q_data = q_data.to(self.device)
                a_data = a_data.to(self.device)
                
                # Forward pass
                self.optimizer.zero_grad()
                output = self.model(q_data, a_data)
                
                # Get the prediction for non-padding questions
                mask = (q_data != 0).float()
                pred = output * mask
                
                # Get the target
                target = a_data.float() * mask
                
                # Compute loss
                loss = self.criterion(pred, target)
                
                # Backward pass
                loss.backward()
                self.optimizer.step()
                
                epoch_loss += loss.item()
                
            epoch_loss /= (batch_idx + 1)
            losses.append(epoch_loss)
            print(f'Epoch: {epoch+1}, Loss: {epoch_loss:.4f}')
            
        return losses
    
    def evaluate(self, test_loader):
        """
        Evaluate the DKVMN model
        
        Parameters:
        -----------
        test_loader: DataLoader
            DataLoader for test data
            
        Returns:
        --------
        metrics: dict
            Dictionary of evaluation metrics
        """
        self.model.eval()
        total_pred = []
        total_target = []
        
        with torch.no_grad():
            for q_data, a_data in test_loader:
                q_data = q_data.to(self.device)
                a_data = a_data.to(self.device)
                
                # Forward pass
                output = self.model(q_data, a_data)
                
                # Get the prediction for non-padding questions
                mask = (q_data != 0).float()
                pred = output * mask
                
                # Get the target
                target = a_data.float() * mask
                
                total_pred.append(pred.cpu().numpy())
                total_target.append(target.cpu().numpy())
        
        total_pred = np.concatenate(total_pred, axis=0)
        total_target = np.concatenate(total_target, axis=0)
        
        # Calculate evaluation metrics
        auc = self.calculate_auc(total_pred, total_target)
        accuracy = self.calculate_accuracy(total_pred, total_target)
        
        return {
            "auc": auc,
            "accuracy": accuracy
        }
    
    def calculate_auc(self, pred, target):
        """Calculate area under the ROC curve"""
        from sklearn.metrics import roc_auc_score
        
        # Flatten the arrays
        pred_flat = pred.flatten()
        target_flat = target.flatten()
        
        # Remove padding
        mask = (target_flat != 0) & (~np.isnan(pred_flat))
        pred_flat = pred_flat[mask]
        target_flat = target_flat[mask]
        
        if len(np.unique(target_flat)) == 1:  # Only one class in the target
            return 0.5
        
        return roc_auc_score(target_flat, pred_flat)
    
    def calculate_accuracy(self, pred, target):
        """Calculate accuracy"""
        # Flatten the arrays
        pred_flat = pred.flatten()
        target_flat = target.flatten()
        
        # Remove padding
        mask = (target_flat != 0) & (~np.isnan(pred_flat))
        pred_flat = pred_flat[mask]
        target_flat = target_flat[mask]
        
        # Convert predictions to binary
        pred_binary = (pred_flat >= 0.5).astype(int)
        
        # Calculate accuracy
        accuracy = np.mean(pred_binary == target_flat)
        
        return accuracy

# Example usage
def main():
    # Parameters
    num_questions = 100
    dim_key = 64
    dim_value = 128
    dim_hidden = 64
    batch_size = 32
    seq_len = 20
    
    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DKVMN(num_questions, dim_key, dim_value, dim_hidden).to(device)
    
    # Define optimizer and loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCELoss()
    
    # Initialize trainer
    trainer = DKVMNTrainer(model, optimizer, criterion, device)
    
    # Generate dummy data
    q_data = torch.randint(1, num_questions + 1, (batch_size, seq_len))
    a_data = torch.randint(0, 2, (batch_size, seq_len))
    
    # Create data loader
    from torch.utils.data import TensorDataset, DataLoader
    train_dataset = TensorDataset(q_data, a_data)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # Train model
    losses = trainer.train(train_loader, epochs=5)
    
    # Evaluate model
    metrics = trainer.evaluate(train_loader)
    print(f"AUC: {metrics['auc']:.4f}, Accuracy: {metrics['accuracy']:.4f}")
    
if __name__ == "__main__":
    main()