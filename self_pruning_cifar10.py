import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# =========================
# Device Configuration
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# =========================
# Prunable Linear Layer
# =========================
class PrunableLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features) * 0.01)
        self.bias = nn.Parameter(torch.zeros(out_features))
        self.gate_scores = nn.Parameter(torch.randn(out_features, in_features))

    def forward(self, x):
        gates = torch.sigmoid(self.gate_scores)
        pruned_weights = self.weight * gates
        return F.linear(x, pruned_weights, self.bias)


# =========================
# Model
# =========================
class PrunableMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = PrunableLinear(32 * 32 * 3, 512)
        self.fc2 = PrunableLinear(512, 256)
        self.fc3 = PrunableLinear(256, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# =========================
# Sparsity Loss
# =========================
def sparsity_loss(model):
    loss = 0
    for module in model.modules():
        if isinstance(module, PrunableLinear):
            gates = torch.sigmoid(module.gate_scores)
            loss += torch.sum(gates)
    return loss


# =========================
# Sparsity Calculation
# =========================
def calculate_sparsity(model, threshold=1e-2):
    total = 0
    zero = 0

    for module in model.modules():
        if isinstance(module, PrunableLinear):
            gates = torch.sigmoid(module.gate_scores)
            total += gates.numel()
            zero += torch.sum(gates < threshold).item()

    return (zero / total) * 100


# =========================
# Training Function
# =========================
def train(model, train_loader, optimizer, lambda_sparse, epochs=5):
    model.train()
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        total_loss = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)

            cls_loss = criterion(outputs, labels)
            sp_loss = sparsity_loss(model)

            loss = cls_loss + lambda_sparse * sp_loss

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")


# =========================
# Evaluation Function
# =========================
def test(model, test_loader):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    return accuracy


# =========================
# Get Gate Values
# =========================
def get_all_gates(model):
    gates_list = []

    for module in model.modules():
        if isinstance(module, PrunableLinear):
            gates = torch.sigmoid(module.gate_scores).detach().cpu().numpy()
            gates_list.extend(gates.flatten())

    return gates_list


# =========================
# Data Loader
# =========================
transform = transforms.Compose([transforms.ToTensor()])

train_dataset = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
test_dataset = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)

print(f"Train samples: {len(train_dataset)}, Test samples: {len(test_dataset)}")

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64)


# =========================
# Experiment with Different λ
# =========================
lambdas = [0, 1e-05, 1e-04, 1e-03, 1e-02, 1e-01, 1]
results = []

for lambda_sparse in lambdas:
    print(f"\n===== Training with lambda = {lambda_sparse} =====")

    model = PrunableMLP().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    train(model, train_loader, optimizer, lambda_sparse, epochs=10)

    accuracy = test(model, test_loader)
    sparsity = calculate_sparsity(model)

    print(f"Accuracy: {accuracy:.2f}%")
    print(f"Sparsity: {sparsity:.2f}%")

    results.append((lambda_sparse, accuracy, sparsity))

    # Plot gate distribution for last run
    if lambda_sparse == lambdas[-1]:
        gates = get_all_gates(model)
        plt.hist(gates, bins=50)
        plt.title("Gate Value Distribution")
        plt.xlabel("Gate Value")
        plt.ylabel("Frequency")
        plt.show()


# =========================
# Print Results Table
# =========================
print("\n===== FINAL RESULTS =====")
print("Lambda\t\tAccuracy\tSparsity (%)")
for l, acc, sp in results:
    print(f"{l}\t{acc:.2f}\t\t{sp:.2f}")