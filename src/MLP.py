import torch
import torch.nn as nn
import torch.optim as optim
import Baselines as bl
from torch.utils.data import TensorDataset, DataLoader
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MLPModule(nn.Module):
    def __init__(
        self, 
        input_dim: int, 
        hidden_dims: list[int] = [32, 16], 
        output_dim: int = 1,
        use_batch_norm: bool = True,
        dropout_rate: float = 0.0,
        activation: type[nn.Module] = nn.ReLU
    ):
        super().__init__()
        
        layers = []
        in_features = input_dim
        
        # Build hidden layers
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_features, h_dim))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(h_dim))
            layers.append(activation())
            if dropout_rate > 0.0:
                layers.append(nn.Dropout(dropout_rate))    
            in_features = h_dim  # Update input size for next layer
            
        layers.append(nn.Linear(in_features, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class MLPModel(bl.Model):
    def __init__(
            self,
            hidden_dims: list[int] = [64, 32],
            lr: float = 0.001,
            epochs: int = 100,
            batch_size: int = 32,
            device: str = "cuda" if torch.cuda.is_available() else "cpu"
        ):
            self.hidden_dims = hidden_dims
            self.lr = lr
            self.epochs = epochs
            self.batch_size = batch_size
            self.device = device

    @property
    def name(self) -> str:
        return f"PyTorchMLP(hidden={self.hidden_dims}, lr={self.lr})"

    def fit(self, dataset: bl.Dataset) -> None:
        X_np = dataset.features
        y_np = dataset.targets.reshape(-1, 1)  # Ensure 2D column vector shape

        input_dim = X_np.shape[1]
        self.module = MLPModule(input_dim=input_dim, hidden_dims=self.hidden_dims).to(self.device)

        # Convert NumPy data to PyTorch Tensors and DataLoader
        X_tensor = torch.tensor(X_np, dtype=torch.float32)
        y_tensor = torch.tensor(y_np, dtype=torch.float32)
        
        torch_dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(torch_dataset, batch_size=self.batch_size, shuffle=True)

        # Setup training parameters
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.module.parameters(), lr=self.lr)

        # Training Loop
        self.module.train()
        for epoch in range(self.epochs):
            for batch_X, batch_y in loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)

                optimizer.zero_grad()
                outputs = self.module(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

    def predict(self, dataset: bl.Dataset) -> np.ndarray:
        if self.module is None:
            raise RuntimeError("PyTorchMLPModel must be fitted bvefore calling predict().")

        self.module.eval()
        X_tensor = torch.tensor(dataset.features, dtype=torch.float32).to(self.device)

        with torch.no_grad():
            predictions_tensor = self.module(X_tensor)

        # Convert output back to 1D NumPy array matching target dimensions
        return predictions_tensor.cpu().numpy().flatten()