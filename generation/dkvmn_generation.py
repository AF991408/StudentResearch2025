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
    
    def forward(self, q, a=None):
        """Forward pass of the DKVMN model
        
        Parameters:
        -----------
        q: Tensor [batch_size, seq_len]
            Question sequence
        a: Tensor [batch_size, seq_len] or None
            Answer sequence (0 or 1), None during inference
            
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
            # Get question at time t
            q_t = q[:, t]  # [batch_size]
            
            # Skip padding (question_id = 0)
            if (q_t == 0).all():
                # For padding positions, output zero predictions
                pred = torch.zeros(batch_size, 1, device=q.device)
                predictions.append(pred)
                continue
                
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
            
            # If answers are provided (during training), update memory
            if a is not None:
                a_t = a[:, t].unsqueeze(1)  # [batch_size, 1]
                
                # Update value memory based on the correct answer
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
        
    def train(self, train_loader, epochs, val_loader=None):
        """
        Train the DKVMN model
        
        Parameters:
        -----------
        train_loader: DataLoader
            DataLoader for training data
        epochs: int
            Number of epochs to train
        val_loader: DataLoader or None
            DataLoader for validation data, if None, no validation is performed
            
        Returns:
        --------
        history: dict
            Dictionary of training history
        """
        history = {
            'train_loss': [],
            'val_loss': [],
            'val_accuracy': [],
            'val_auc': []
        }

        for epoch in range(epochs):
            self.model.train()
            epoch_loss = 0
            num_batches = 0
            
            for batch_idx, (q_data, a_data) in enumerate(train_loader):
                q_data = q_data.to(self.device)
                a_data = a_data.to(self.device)
                
                # Forward pass
                self.optimizer.zero_grad()
                output = self.model(q_data, a_data)
                
                # Create mask for non-padding positions
                mask = (q_data != 0).float()
                
                # Apply mask to predictions and targets
                masked_output = output * mask
                masked_target = a_data.float() * mask
                
                # Compute loss only on non-padding positions
                valid_elements = mask.sum()
                if valid_elements > 0:
                    loss = self.criterion(masked_output, masked_target)
                    
                    # Backward pass
                    loss.backward()
                    self.optimizer.step()
                    
                    epoch_loss += loss.item()
                    num_batches += 1
            
            # Calculate average loss for the epoch
            avg_epoch_loss = epoch_loss / max(1, num_batches)
            history['train_loss'].append(avg_epoch_loss)
            
            print(f'Epoch: {epoch+1}/{epochs}, Train Loss: {avg_epoch_loss:.4f}', end='')
            
            # Validation if validation loader is provided
            if val_loader is not None:
                metrics = self.evaluate(val_loader)
                val_loss = self.calculate_validation_loss(val_loader)
                
                history['val_loss'].append(val_loss)
                history['val_accuracy'].append(metrics['accuracy'])
                history['val_auc'].append(metrics['auc'])
                
                print(f', Val Loss: {val_loss:.4f}, Val Acc: {metrics["accuracy"]:.4f}, Val AUC: {metrics["auc"]:.4f}')
            else:
                print('')
        return history

    def calculate_validation_loss(self, val_loader):
        """Calculate loss on validation set"""
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for q_data, a_data in val_loader:
                q_data = q_data.to(self.device)
                a_data = a_data.to(self.device)
                
                # Forward pass
                output = self.model(q_data, a_data)
                
                # Create mask for non-padding positions
                mask = (q_data != 0).float()
                
                # Apply mask to predictions and targets
                masked_output = output * mask
                masked_target = a_data.float() * mask
                
                # Compute loss only on non-padding positions
                valid_elements = mask.sum()
                if valid_elements > 0:
                    loss = self.criterion(masked_output, masked_target)
                    total_loss += loss.item()
                    num_batches += 1
        
        return total_loss / max(1, num_batches)
    
    def evaluate(self, test_loader):
        """Evaluate the DKVMN model
        
        Parameters:
        -----------
        test_loader: DataLoader
            DataLoader for test data
            
        Returns:
        --------
        metrics: dict
            Dictionary of evaluation metrics"""
        self.model.eval()
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for q_data, a_data in test_loader:
                q_data = q_data.to(self.device)
                a_data = a_data.to(self.device)
                
                # Forward pass
                output = self.model(q_data, a_data)
                
                # Create a mask for valid questions (non-padding)
                mask = (q_data != 0).float()
                
                # Store predictions and targets for valid positions only
                preds = output.detach().cpu().numpy()
                targets = a_data.detach().cpu().numpy()
                
                # Store only for non-padding positions
                valid_preds = []
                valid_targets = []
                
                for i in range(preds.shape[0]):  # For each student in the batch
                    for j in range(preds.shape[1]):  # For each position in the sequence
                        if mask[i, j] > 0:  # If not padding
                            valid_preds.append(preds[i, j])
                            valid_targets.append(targets[i, j])
                
                if valid_preds:
                    all_preds.extend(valid_preds)
                    all_targets.extend(valid_targets)
        
        # Convert to numpy arrays
        all_preds = np.array(all_preds)
        all_targets = np.array(all_targets)
        
        # Calculate evaluation metrics
        binary_preds = (all_preds >= 0.5).astype(int)
        accuracy = np.mean(binary_preds == all_targets)
        
        # Calculate AUC if possible
        try:
            from sklearn.metrics import roc_auc_score
            if len(np.unique(all_targets)) > 1:  # More than one class
                auc = roc_auc_score(all_targets, all_preds)
            else:
                auc = 0.5  # Default AUC for single class
        except:
            auc = 0.0  # If sklearn is not available
        
        return {
            "auc": auc,
            "accuracy": accuracy,
            "num_samples": len(all_targets)
        }
    
    def calculate_auc(self, pred, target):
        """Calculate area under the ROC curve"""
        from sklearn.metrics import roc_auc_score
        
        # Flatten the arrays
        pred_flat = pred.flatten()
        target_flat = target.flatten()
        
        # Remove padding (where question_id is 0)
        mask = (target_flat != -1)  # Changed from != 0 to != -1 since 0 is a valid answer
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
        
        # Remove padding (where question_id is 0)
        mask = (target_flat != -1)  # Changed from != 0 to != -1 since 0 is a valid answer
        pred_flat = pred_flat[mask]
        target_flat = target_flat[mask]
        
        # Convert predictions to binary (threshold at 0.5)
        pred_binary = (pred_flat >= 0.5).astype(int)
        
        # Calculate accuracy
        accuracy = np.mean(pred_binary == target_flat)
        
        return accuracy

# This function replaces the main() function in your original code
def main():
    # Import the dataset functions
    from dataset_generator import load_dataset
    
    # Load the dataset
    train_loader, test_loader, num_questions = load_dataset(batch_size=32)
    
    # Create a validation set (use 20% of training data)
    from torch.utils.data import random_split, DataLoader
    
    # Extract dataset from train_loader
    train_dataset = train_loader.dataset
    
    # Determine sizes
    val_size = len(train_dataset) // 5  # 20% of training data
    train_size = len(train_dataset) - val_size
    
    # Split the dataset
    train_subset, val_subset = random_split(
        train_dataset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # Create new data loaders
    batch_size = 32
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)
    
    # Parameters
    dim_key = 64
    dim_value = 128
    dim_hidden = 64
    
    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DKVMN(num_questions, dim_key, dim_value, dim_hidden).to(device)
    
    # Define optimizer and loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCELoss()
    
    # Initialize trainer
    trainer = DKVMNTrainer(model, optimizer, criterion, device)
    
    # Train model with validation
    print("Training model...")
    history = trainer.train(train_loader, epochs=10, val_loader=val_loader)
    
    # Evaluate model on test set
    print("\nEvaluating model on test set...")
    test_metrics = trainer.evaluate(test_loader)
    print(f"Test AUC: {test_metrics['auc']:.4f}, Test Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"Number of test samples: {test_metrics['num_samples']}")
    
    # You could also save the model for later use
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'history': history,
        'test_metrics': test_metrics
    }, "dkvmn_model.pth")
    print("Model saved to dkvmn_model.pth")
    
if __name__ == "__main__":
    main()
