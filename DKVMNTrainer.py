import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math


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
