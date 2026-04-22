import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import copy
import os
import random
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# =========================
# Device Configuration
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# =========================
# Reproducibility
# =========================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(42)
os.makedirs("graphs", exist_ok=True)

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
class PrunableCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(p=0.5)

        self.fc1 = PrunableLinear(64 * 8 * 8, 512)
        self.fc2 = PrunableLinear(512, 256)
        self.fc3 = PrunableLinear(256, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
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
def train(model, train_loader, val_loader, optimizer, scheduler, lambda_sparse, epochs=5):
    criterion = nn.CrossEntropyLoss()
    best_val_accuracy = 0.0
    best_state_dict = copy.deepcopy(model.state_dict())

    for epoch in range(epochs):
        model.train()
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

        # Validation
        val_accuracy = test(model, val_loader)
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_state_dict = copy.deepcopy(model.state_dict())

        scheduler.step()
        mean_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}, Loss: {mean_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%")

    model.load_state_dict(best_state_dict)
    return best_val_accuracy, best_state_dict


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

train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
])

train_dataset = datasets.CIFAR10(root="./data", train=True, download=True, transform=train_transform)
test_dataset = datasets.CIFAR10(root="./data", train=False, download=True, transform=test_transform)

# Split training data into train and validation (95/5 split)
train_size = int(0.95 * len(train_dataset))
val_size = len(train_dataset) - train_size
split_generator = torch.Generator().manual_seed(42)
train_subset, val_subset = torch.utils.data.random_split(train_dataset, [train_size, val_size], generator=split_generator)

print(f"Train samples: {len(train_subset)}, Val samples: {len(val_subset)}, Test samples: {len(test_dataset)}")

train_loader = DataLoader(train_subset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_subset, batch_size=64)
test_loader = DataLoader(test_dataset, batch_size=64)


# =========================
# Experiment with Different λ
# =========================
lambdas = [0, 1e-06, 1e-05, 1e-04, 5e-04, 9e-04, 1e-03,5e-03, 1e-02, 1e-01]
results = []
best_lambda = None
best_lambda_val_accuracy = 0.0
best_lambda_test_accuracy = 0.0
best_lambda_sparsity = 0.0
best_lambda_model_state = None

# Train each lambda with same initial weights
for lambda_sparse in lambdas:
    print(f"\n===== Training with lambda = {lambda_sparse} =====")

    model = PrunableCNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    val_accuracy, best_state_dict = train(model, train_loader, val_loader, optimizer, scheduler, lambda_sparse, epochs=20)
    model.load_state_dict(best_state_dict)

    accuracy = test(model, test_loader)
    sparsity = calculate_sparsity(model)

    print(f"Accuracy: {accuracy:.2f}%")
    print(f"Sparsity: {sparsity:.2f}%")

    results.append((lambda_sparse, val_accuracy, accuracy, sparsity))

    if val_accuracy > best_lambda_val_accuracy:
        best_lambda = lambda_sparse
        best_lambda_val_accuracy = val_accuracy
        best_lambda_test_accuracy = accuracy
        best_lambda_sparsity = sparsity
        best_lambda_model_state = copy.deepcopy(model.state_dict())
        torch.save(
            {
                "lambda": lambda_sparse,
                "model_state_dict": best_lambda_model_state,
                "val_accuracy": val_accuracy,
                "test_accuracy": accuracy,
                "sparsity": sparsity,
            },
            "best_model_checkpoint.pth",
        )


# =========================
# Print Results Table
# =========================
print("\n===== FINAL RESULTS =====")
print("Lambda\t\tVal Acc\t\tTest Acc\tSparsity (%)")
for l, val_acc, acc, sp in results:
    print(f"{l}\t{val_acc:.2f}\t\t{acc:.2f}\t\t{sp:.2f}")

print(f"\nBest lambda by validation accuracy: {best_lambda}")
print(f"Best validation accuracy: {best_lambda_val_accuracy:.2f}%")
print(f"Best test accuracy: {best_lambda_test_accuracy:.2f}%")
print(f"Best sparsity: {best_lambda_sparsity:.2f}%")

# Keep the sweep ordered from low to high lambda for reporting and plotting
sorted_results = sorted(results, key=lambda item: item[0])
lambdas_only = [item[0] for item in sorted_results]
val_accuracies = [item[1] for item in sorted_results]
sparsities = [item[3] for item in sorted_results]

# Plot lambda vs validation accuracy and highlight the best lambda
plt.figure(figsize=(8, 5))
plt.plot(lambdas_only, val_accuracies, marker="o", linewidth=2)
plt.scatter([best_lambda], [best_lambda_val_accuracy], color="red", s=100, label="Best lambda")
plt.xscale("symlog")
plt.xlabel("Lambda")
plt.ylabel("Validation Accuracy (%)")
plt.title("Validation Accuracy vs Lambda")
plt.legend()
plt.tight_layout()
plt.savefig("graphs/validation_accuracy_vs_lambda.png", dpi=300, bbox_inches="tight")
plt.show()

# Plot sparsity vs validation accuracy to show the trade-off directly
plt.figure(figsize=(8, 5))
tradeoff_scatter = plt.scatter(
    sparsities,
    val_accuracies,
    c=lambdas_only,
    cmap="viridis",
    s=90,
    edgecolors="black",
)

for lambda_value, sparsity, val_accuracy in zip(lambdas_only, sparsities, val_accuracies):
    plt.annotate(
        f"{lambda_value:g}",
        (sparsity, val_accuracy),
        textcoords="offset points",
        xytext=(5, 5),
        fontsize=8,
    )

plt.xlabel("Sparsity (%)")
plt.ylabel("Validation Accuracy (%)")
plt.title("Sparsity vs Validation Accuracy Trade-off")
plt.grid(True, alpha=0.25)
colorbar = plt.colorbar(tradeoff_scatter)
colorbar.set_label("Lambda")
plt.tight_layout()
plt.savefig("graphs/sparsity_vs_validation_accuracy_tradeoff.png", dpi=300, bbox_inches="tight")
plt.show()

# Plot gate distribution for the best model
if best_lambda_model_state is not None:
    best_model = PrunableCNN().to(device)
    best_model.load_state_dict(best_lambda_model_state)
    gates = get_all_gates(best_model)

    plt.figure(figsize=(8, 5))
    plt.hist(gates, bins=50)
    plt.title(f"Best Model Gate Value Distribution (lambda={best_lambda})")
    plt.xlabel("Gate Value")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig("graphs/best_model_gates.png", dpi=300, bbox_inches="tight")
    plt.show()