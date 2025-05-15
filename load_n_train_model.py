import os
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

MODEL_PATH = os.path.join("models", "pretrained_mnist.pth")
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)


class SimpleMNISTModel(nn.Module):
    """
    A simple CNN model for MNIST classification.
    """
    def __init__(self):
        """
        Initialize the model layers
        """
        super(SimpleMNISTModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        """
        Forward pass of the model
        This method defines how the input data flows through the model.
        
        Args:
            x: Input tensor
        
        Returns:
            Output tensor
            
        """
        x = self.conv1(x)
        x = nn.functional.relu(x)
        x = self.conv2(x)
        x = nn.functional.relu(x)
        x = nn.functional.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = nn.functional.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        return x


def train_mnist_model(
    model: nn.Module, 
    train_loader: DataLoader, 
    device: torch.device,
    epochs: int = 5
) -> nn.Module:
    """
    Train a model on MNIST dataset
    
    Args:
        model: Model to train
        train_loader: DataLoader for training data
        device: Device to use for training
        epochs: Number of training epochs
        
    Returns:
        Trained model
    """
    model.to(device)
    optimizer = optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()
    
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            if batch_idx % 100 == 99:
                print(f'Epoch: {epoch+1}, Batch: {batch_idx+1}, Loss: {running_loss/100:.4f}')
                running_loss = 0.0
                
    return model


def test_model(
    model: nn.Module, 
    test_loader: DataLoader, 
    device: torch.device
) -> float:
    """
    Test a model on a dataset
    
    Args:
        model: Model to test
        test_loader: DataLoader for test data
        device: Device to use for testing
        
    Returns:
        Accuracy of the model on the test set
    """
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    
    accuracy = correct / total
    print(f'Test Accuracy: {accuracy:.4f}')
    return accuracy


def load_or_train_model(
    train_loader: DataLoader, 
    test_loader: DataLoader, 
    device: torch.device,
    model_path: str = MODEL_PATH
) -> nn.Module:
    """
    Load a pretrained model or train a new one if not available
    
    Args:
        train_loader: DataLoader for training data
        test_loader: DataLoader for test data
        device: Device to use for training/testing
        model_path: Path to save/load the model
        
    Returns:
        The loaded or trained model
    """
    model = SimpleMNISTModel()
    
    # Check if we have a pretrained model
    if os.path.exists(model_path):
        print(f"Loading pretrained model from {model_path}")
        model.load_state_dict(torch.load(model_path, map_location=device))
    else:
        print(f"Training new model and saving to {model_path}")
        model = train_mnist_model(model, train_loader, device)
        # Save the model
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        torch.save(model.state_dict(), model_path)
    
    # Test the model
    test_model(model, test_loader, device)
    
    return model



if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load MNIST dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
    
    model = load_or_train_model(train_loader, test_loader, device)
    model.to(device)

    print("Model loading complete!") 