import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
import pickle
import os

def generate_synthetic_knowledge_tracing_data(
    num_students=100,
    num_questions=100,
    seq_length=50,
    num_skills=5,
    skill_difficulty={0: 0.3, 1: 0.4, 2: 0.5, 3: 0.6, 4: 0.7},
    learning_rate=0.1,
    forgetting_rate=0.05,
    seed=42
):
    """
    Generate synthetic data for knowledge tracing that simulates realistic student learning.
    
    Parameters:
    -----------
    num_students: int
        Number of students
    num_questions: int
        Total number of questions/exercises
    seq_length: int
        Sequence length for each student
    num_skills: int
        Number of different skills/concepts
    skill_difficulty: dict
        Difficulty level for each skill (0 to 1)
    learning_rate: float
        Rate at which students learn skills
    forgetting_rate: float
        Rate at which students forget skills
    seed: int
        Random seed for reproducibility
        
    Returns:
    --------
    q_data: torch.Tensor [num_students, seq_length]
        Question sequences for each student
    a_data: torch.Tensor [num_students, seq_length]
        Answer sequences for each student (0 or 1)
    """
    np.random.seed(seed)
    
    # Map questions to skills (each question tests one skill)
    question_to_skill = np.random.randint(0, num_skills, num_questions)
    
    # Initialize data structures
    q_data = np.zeros((num_students, seq_length), dtype=int)
    a_data = np.zeros((num_students, seq_length), dtype=int)
    
    # For each student
    for student_id in range(num_students):
        # Initialize student knowledge for each skill (probability of answering correctly)
        knowledge = {skill: np.random.uniform(0.1, 0.3) for skill in range(num_skills)}
        
        # Generate sequence of questions for this student
        # Start with more focus on a few skills, gradually expand
        focus_skills = np.random.choice(num_skills, size=min(3, num_skills), replace=False)
        
        for t in range(seq_length):
            # As student progresses, expand to more skills
            if t > seq_length // 2 and np.random.random() < 0.7:
                focus_skills = np.random.choice(num_skills, size=min(num_skills, len(focus_skills) + 1), replace=False)
            
            # Select skill for this timestep (with higher probability for focus skills)
            if np.random.random() < 0.7:  # 70% chance to practice focus skills
                skill = np.random.choice(focus_skills)
            else:
                skill = np.random.randint(0, num_skills)
            
            # Find questions for this skill
            skill_questions = np.where(question_to_skill == skill)[0] + 1  # +1 because 0 is reserved for padding
            
            # Choose a question for this skill
            question_id = np.random.choice(skill_questions)
            q_data[student_id, t] = question_id
            
            # Determine if student answers correctly based on knowledge and question difficulty
            skill_mastery = knowledge[skill]
            difficulty = skill_difficulty[skill]
            
            # Probability of correct answer depends on skill mastery and question difficulty
            prob_correct = skill_mastery * (1 - difficulty)
            
            # Record answer
            is_correct = 1 if np.random.random() < prob_correct else 0
            a_data[student_id, t] = is_correct
            
            # Update knowledge based on answer
            if is_correct:
                # Learning from correct answer
                knowledge[skill] = min(1.0, knowledge[skill] + learning_rate * (1 - knowledge[skill]))
            else:
                # Learning from incorrect answer (smaller increase)
                knowledge[skill] = min(1.0, knowledge[skill] + 0.5 * learning_rate * (1 - knowledge[skill]))
            
            # Apply forgetting to all skills not practiced
            for other_skill in range(num_skills):
                if other_skill != skill:
                    knowledge[other_skill] = max(0.0, knowledge[other_skill] * (1 - forgetting_rate))
    
    return torch.tensor(q_data), torch.tensor(a_data)

def save_dataset(filename="dkvmn_dataset.pkl"):
    """Generate and save a fixed dataset for DKVMN model"""
    # Parameters
    num_students = 200
    num_questions = 100
    seq_length = 50
    num_skills = 5
    
    # Generate data
    q_data, a_data = generate_synthetic_knowledge_tracing_data(
        num_students=num_students,
        num_questions=num_questions,
        seq_length=seq_length,
        num_skills=num_skills,
        seed=42
    )
    
    # Split data into train and test sets (80% train, 20% test)
    num_train = int(0.8 * num_students)
    
    train_q = q_data[:num_train]
    train_a = a_data[:num_train]
    
    test_q = q_data[num_train:]
    test_a = a_data[num_train:]
    
    # Create dataset dictionary
    dataset = {
        'train_q': train_q,
        'train_a': train_a,
        'test_q': test_q,
        'test_a': test_a,
        'num_questions': num_questions
    }
    
    # Save dataset
    with open(filename, 'wb') as f:
        pickle.dump(dataset, f)
    
    print(f"Dataset saved to {filename}")
    return dataset

def load_dataset(filename="dkvmn_dataset.pkl", batch_size=32):
    """Load the fixed dataset and create DataLoaders"""
    if not os.path.exists(filename):
        print(f"Dataset file {filename} not found. Generating new dataset...")
        dataset = save_dataset(filename)
    else:
        # Load dataset
        with open(filename, 'rb') as f:
            dataset = pickle.load(f)
    
    # Create DataLoaders
    train_dataset = TensorDataset(dataset['train_q'], dataset['train_a'])
    test_dataset = TensorDataset(dataset['test_q'], dataset['test_a'])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader, dataset['num_questions']

# Example usage
if __name__ == "__main__":
    # Generate and save the dataset if it doesn't exist
    if not os.path.exists("dkvmn_dataset.pkl"):
        save_dataset()
    
    # Load the dataset
    train_loader, test_loader, num_questions = load_dataset()
    
    # Display some statistics
    print(f"Number of questions: {num_questions}")
    
    # Get a batch
    for q_batch, a_batch in train_loader:
        print(f"Question batch shape: {q_batch.shape}")
        print(f"Answer batch shape: {a_batch.shape}")
        print(f"Sample questions: {q_batch[0][:10]}")
        print(f"Sample answers: {a_batch[0][:10]}")
        break
