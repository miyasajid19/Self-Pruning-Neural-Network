import argparse
import csv
import json
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


DEFAULT_LAMBDAS = [0.0, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PrunableLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features) * 0.01)
        self.bias = nn.Parameter(torch.zeros(out_features))
        self.gate_scores = nn.Parameter(torch.randn(out_features, in_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gates = torch.sigmoid(self.gate_scores)
        pruned_weights = self.weight * gates
        return F.linear(x, pruned_weights, self.bias)


class PrunableMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = PrunableLinear(32 * 32 * 3, 512)
        self.fc2 = PrunableLinear(512, 256)
        self.fc3 = PrunableLinear(256, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Self-pruning MLP on CIFAR-10")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--threshold", type=float, default=1e-2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-root", type=str, default="./data")
    parser.add_argument("--outputs-dir", type=str, default="outputs")
    parser.add_argument("--train-subset", type=int, default=None)
    parser.add_argument("--test-subset", type=int, default=None)
    parser.add_argument(
        "--lambdas",
        nargs="+",
        type=float,
        default=DEFAULT_LAMBDAS,
        help="Space-separated sparsity lambdas",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def suppress_known_warnings() -> None:
    if hasattr(np, "exceptions") and hasattr(np.exceptions, "VisibleDeprecationWarning"):
        warnings.filterwarnings(
            "ignore",
            message=r"dtype\(\): align should be passed as Python or NumPy boolean.*",
            category=np.exceptions.VisibleDeprecationWarning,
        )


def build_dataloaders(
    data_root: str,
    batch_size: int,
    train_subset: int | None = None,
    test_subset: int | None = None,
) -> tuple[DataLoader, DataLoader, int, int]:
    transform = transforms.Compose([transforms.ToTensor()])
    root = Path(data_root)

    train_dataset = datasets.CIFAR10(root=str(root), train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10(root=str(root), train=False, download=True, transform=transform)

    if train_subset is not None:
        train_dataset = Subset(train_dataset, range(min(train_subset, len(train_dataset))))
    if test_subset is not None:
        test_dataset = Subset(test_dataset, range(min(test_subset, len(test_dataset))))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    return train_loader, test_loader, len(train_dataset), len(test_dataset)


def sparsity_loss(model: nn.Module) -> torch.Tensor:
    total = torch.tensor(0.0, device=DEVICE)
    for module in model.modules():
        if isinstance(module, PrunableLinear):
            total = total + torch.sigmoid(module.gate_scores).sum()
    return total


def calculate_sparsity(model: nn.Module, threshold: float) -> float:
    total = 0
    zero = 0
    for module in model.modules():
        if isinstance(module, PrunableLinear):
            gates = torch.sigmoid(module.gate_scores)
            total += gates.numel()
            zero += torch.sum(gates < threshold).item()
    return (zero / total) * 100 if total else 0.0


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: optim.Optimizer,
    lambda_sparse: float,
    epochs: int,
) -> None:
    criterion = nn.CrossEntropyLoss()
    model.train()

    for epoch in range(epochs):
        total_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(images)

            cls_loss = criterion(outputs, labels)
            sp_loss = sparsity_loss(model)
            loss = cls_loss + lambda_sparse * sp_loss

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch + 1}, Loss: {total_loss:.4f}")


def evaluate_accuracy(model: nn.Module, test_loader: DataLoader) -> float:
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return (100.0 * correct / total) if total else 0.0


def get_all_gates(model: nn.Module) -> np.ndarray:
    gates_list: list[np.ndarray] = []
    for module in model.modules():
        if isinstance(module, PrunableLinear):
            gates = torch.sigmoid(module.gate_scores).detach().cpu().numpy().ravel()
            gates_list.append(gates)
    return np.concatenate(gates_list) if gates_list else np.array([])


def save_results_files(results: list[dict], outputs_dir: Path) -> None:
    csv_path = outputs_dir / "results_summary.csv"
    json_path = outputs_dir / "results_summary.json"
    md_path = outputs_dir / "results_table.md"

    with csv_path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=["lambda", "accuracy", "sparsity"])
        writer.writeheader()
        writer.writerows(results)

    with json_path.open("w", encoding="utf-8") as fp:
        json.dump(results, fp, indent=2)

    lines = [
        "| Lambda | Test Accuracy | Sparsity Level (%) |",
        "| --- | ---: | ---: |",
    ]
    for item in results:
        lines.append(
            f"| {item['lambda']:g} | {item['accuracy']:.2f}% | {item['sparsity']:.2f}% |"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_gate_distribution(gates: np.ndarray, outputs_dir: Path) -> None:
    if gates.size == 0:
        return

    plt.figure(figsize=(8, 5))
    plt.hist(gates, bins=50)
    plt.title("Gate Value Distribution")
    plt.xlabel("Gate Value")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(outputs_dir / "best_gate_distribution.png", dpi=200)
    plt.close()


def plot_lambda_tradeoff(results: list[dict], outputs_dir: Path) -> None:
    labels = [f"{item['lambda']:g}" for item in results]
    accuracy = [item["accuracy"] for item in results]
    sparsity = [item["sparsity"] for item in results]
    x = np.arange(len(labels))

    fig, ax1 = plt.subplots(figsize=(10, 5.5))
    color1 = "#1f77b4"
    color2 = "#d62728"

    line1 = ax1.plot(x, accuracy, marker="o", color=color1, label="Accuracy (%)")
    ax1.set_ylabel("Accuracy (%)", color=color1)
    ax1.tick_params(axis="y", labelcolor=color1)
    ax1.set_xlabel("Lambda")
    ax1.set_xticks(x, labels)
    ax1.grid(True, axis="y", linestyle="--", alpha=0.4)

    ax2 = ax1.twinx()
    line2 = ax2.plot(x, sparsity, marker="s", color=color2, label="Sparsity (%)")
    ax2.set_ylabel("Sparsity (%)", color=color2)
    ax2.tick_params(axis="y", labelcolor=color2)

    lines = line1 + line2
    labels_legend = [line.get_label() for line in lines]
    ax1.legend(lines, labels_legend, loc="best")
    plt.title("Lambda vs Accuracy/Sparsity Trade-off")
    fig.tight_layout()
    plt.savefig(outputs_dir / "lambda_tradeoff.png", dpi=200)
    plt.close(fig)


def run_experiment(args: argparse.Namespace) -> None:
    suppress_known_warnings()
    set_seed(args.seed)

    outputs_dir = Path(args.outputs_dir)
    outputs_dir.mkdir(parents=True, exist_ok=True)

    print(f"Using device: {DEVICE}")
    train_loader, test_loader, train_count, test_count = build_dataloaders(
        data_root=args.data_root,
        batch_size=args.batch_size,
        train_subset=args.train_subset,
        test_subset=args.test_subset,
    )
    print(f"Train samples: {train_count}, Test samples: {test_count}")

    results: list[dict] = []
    best_model = None
    best_accuracy = -1.0

    for lambda_sparse in args.lambdas:
        print(f"\n===== Training with lambda = {lambda_sparse:g} =====")

        model = PrunableMLP().to(DEVICE)
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

        train_model(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            lambda_sparse=lambda_sparse,
            epochs=args.epochs,
        )

        accuracy = evaluate_accuracy(model, test_loader)
        sparsity = calculate_sparsity(model, threshold=args.threshold)

        print(f"Accuracy: {accuracy:.2f}%")
        print(f"Sparsity: {sparsity:.2f}%")

        item = {
            "lambda": float(lambda_sparse),
            "accuracy": float(accuracy),
            "sparsity": float(sparsity),
        }
        results.append(item)

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = model

    save_results_files(results, outputs_dir)
    plot_lambda_tradeoff(results, outputs_dir)
    if best_model is not None:
        plot_gate_distribution(get_all_gates(best_model), outputs_dir)

    print("\n===== FINAL RESULTS =====")
    print("Lambda\t\tAccuracy\t\tSparsity (%)")
    for item in results:
        print(f"{item['lambda']:g}\t\t{item['accuracy']:.2f}\t\t{item['sparsity']:.2f}")

    print("\nSaved artifacts:")
    print(f"- {outputs_dir / 'results_summary.csv'}")
    print(f"- {outputs_dir / 'results_summary.json'}")
    print(f"- {outputs_dir / 'results_table.md'}")
    print(f"- {outputs_dir / 'lambda_tradeoff.png'}")
    print(f"- {outputs_dir / 'best_gate_distribution.png'}")


if __name__ == "__main__":
    cli_args = parse_args()
    run_experiment(cli_args)