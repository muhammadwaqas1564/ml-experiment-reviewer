"""PyTorch training loop - DEMO WITH HIDDEN BUGS (static review only, torch not required)."""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

X = torch.randn(1000, 20)
y = (X[:, 0] + X[:, 1] > 0).long()
X_train, X_val = X[:800], X[800:]
y_train, y_val = y[:800], y[800:]

model = nn.Sequential(
    nn.Linear(20, 64), nn.ReLU(), nn.Dropout(0.3),
    nn.Linear(64, 2),
)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()
loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True)

for epoch in range(10):
    for xb, yb in loader:
        loss = criterion(model(xb), yb)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        val_acc = (model(X_val).argmax(1) == y_val).float().mean().item()
    print(f"epoch {epoch}: val accuracy {val_acc:.3f}")
