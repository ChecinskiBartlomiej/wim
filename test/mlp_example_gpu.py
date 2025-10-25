import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import time

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)


class MLP(nn.Module):
    """Larger Multi-Layer Perceptron with 4 layers"""
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.relu3 = nn.ReLU()
        self.fc4 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        x = self.relu3(x)
        x = self.fc4(x)
        return x


def generate_synthetic_data(n_samples=1000, n_features=10):
    """Generate synthetic binary classification data"""
    # Generate random features
    X = np.random.randn(n_samples, n_features)

    # Create a simple decision boundary: sum of first 3 features
    decision_function = X[:, 0] + 2 * X[:, 1] - X[:, 2]
    y = (decision_function > 0).astype(np.float32)

    return X.astype(np.float32), y


def train_model(model, train_loader, criterion, optimizer, device, epochs=50):
    """Train the model"""
    model.train()

    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        total = 0

        for inputs, labels in train_loader:
            # Move data to device
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), labels)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Calculate accuracy
            predicted = (torch.sigmoid(outputs.squeeze()) > 0.5).float()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            total_loss += loss.item()

        if (epoch + 1) % 10 == 0:
            accuracy = 100 * correct / total
            avg_loss = total_loss / len(train_loader)
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%')


def test_model(model, test_loader, device):
    """Test the model"""
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            # Move data to device
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            predicted = (torch.sigmoid(outputs.squeeze()) > 0.5).float()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f'\nTest Accuracy: {accuracy:.2f}%')
    return accuracy


def main():
    # Start timing
    start_time = time.time()

    # Device configuration - use GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Number of GPUs available: {torch.cuda.device_count()}")
    else:
        print("No GPU available, using CPU")

    # Hyperparameters
    input_size = 10
    hidden_size = 4092
    output_size = 1  # Binary classification
    batch_size = 32
    learning_rate = 0.01
    epochs = 100

    print("\nGenerating synthetic training data...")
    X_train, y_train = generate_synthetic_data(n_samples=10000, n_features=input_size)

    print("Generating synthetic test data...")
    X_test, y_test = generate_synthetic_data(n_samples=2000, n_features=input_size)

    # Convert to PyTorch tensors
    X_train_tensor = torch.from_numpy(X_train)
    y_train_tensor = torch.from_numpy(y_train)
    X_test_tensor = torch.from_numpy(X_test)
    y_test_tensor = torch.from_numpy(y_test)

    # Create data loaders
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Initialize model, loss function, and optimizer
    print(f"\nInitializing MLP with architecture: {input_size} -> {hidden_size} -> {hidden_size} -> {output_size}")
    model = MLP(input_size, hidden_size, output_size).to(device)
    print(f"Model is on device: {next(model.parameters()).device}")
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Train the model
    print(f"\nTraining model for {epochs} epochs...")
    train_model(model, train_loader, criterion, optimizer, device, epochs=epochs)

    # Test the model
    print("\nEvaluating model on test data...")
    test_accuracy = test_model(model, test_loader, device)

    # Calculate elapsed time
    end_time = time.time()
    elapsed_time = end_time - start_time

    # Save test accuracy, execution time, and device info to file
    output_file = "test_results_gpu.txt"
    with open(output_file, 'w') as f:
        f.write(f"Device: {device}\n")
        if device.type == 'cuda':
            f.write(f"GPU Name: {torch.cuda.get_device_name(0)}\n")
            f.write(f"Number of GPUs: {torch.cuda.device_count()}\n")
        f.write(f"Test Accuracy: {test_accuracy:.2f}%\n")
        f.write(f"Execution Time: {elapsed_time:.2f} seconds\n")

    print(f"\nExecution Time: {elapsed_time:.2f} seconds")
    print(f"Results saved to {output_file}")


if __name__ == "__main__":
    main()
