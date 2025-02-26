import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import DKVMNTrainer
import dkvmn

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
    model = dkvmn(num_questions, dim_key, dim_value, dim_hidden).to(device)
    
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
