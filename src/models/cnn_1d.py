"""
1D Convolutional Neural Network (1D-CNN) for Network Intrusion Detection
Slides 1D kernels across network flow metrics to extract localized spatial-temporal correlations.

Architecture:
Input (Batch, 1, Features) 
    -> Conv1D (16 filters, kernel=3, padding=1, ReLU)
    -> MaxPool1D (pool=2)
    -> Conv1D (32 filters, kernel=3, padding=1, ReLU)
    -> Flatten
    -> Dense (32, ReLU)
    -> Dense (Output)
"""

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.neural_network import MLPClassifier

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


if HAS_TORCH:
    class PyTorch1DCNNModule(nn.Module):
        def __init__(self, in_features, num_classes=2):
            super().__init__()
            self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
            self.bn1 = nn.BatchNorm1d(16)
            self.pool1 = nn.MaxPool1d(kernel_size=2)
            
            self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
            self.bn2 = nn.BatchNorm1d(32)
            self.adaptive_pool = nn.AdaptiveAvgPool1d(4)
            
            self.fc1 = nn.Linear(32 * 4, 32)
            self.dropout = nn.Dropout(0.2)
            self.fc_out = nn.Linear(32, num_classes)

        def forward(self, x):
            if x.dim() == 2:
                x = x.unsqueeze(1)
            x = torch.relu(self.bn1(self.conv1(x)))
            x = self.pool1(x)
            x = torch.relu(self.bn2(self.conv2(x)))
            x = self.adaptive_pool(x)
            x = x.view(x.size(0), -1)
            x = torch.relu(self.fc1(x))
            x = self.dropout(x)
            return self.fc_out(x)


class CNN1DClassifier(BaseEstimator, ClassifierMixin):
    """
    Scikit-learn compatible wrapper for 1D-CNN.
    Uses PyTorch if available; otherwise uses a high-performance 1D-convolutional
    feature synthesis layer + dense classification head.
    """
    def __init__(self, in_features=8, num_classes=2, epochs=30, batch_size=64, lr=0.003, random_state=42):
        self.in_features = in_features
        self.num_classes = num_classes
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.random_state = random_state
        self.classes_ = None
        self.model_ = None

    def _conv1d_numpy_transform(self, X):
        """
        NumPy-based 1D convolution feature extraction:
        Applies a bank of sliding 1D kernels (size 3) across the flow feature vector
        to extract local cross-feature gradients and correlations.
        """
        N, D = X.shape
        # Bank of 8 localized 1D kernels: edge, smoothing, gradient, peak
        kernels = np.array([
            [-1.0,  2.0, -1.0],
            [ 0.5,  1.0,  0.5],
            [-1.0,  0.0,  1.0],
            [ 1.0, -2.0,  1.0],
            [ 0.0,  1.0, -1.0],
            [-1.0,  1.0,  0.0],
            [ 0.3,  0.4,  0.3],
            [-0.5,  1.5, -0.5]
        ])
        
        # Pad X along feature dimension
        X_padded = np.pad(X, ((0, 0), (1, 1)), mode="edge")
        
        feature_maps = []
        for k in kernels:
            # Convolve across feature sequence
            convolved = (
                X_padded[:, :-2] * k[0] +
                X_padded[:, 1:-1] * k[1] +
                X_padded[:, 2:]   * k[2]
            )
            # ReLU activation
            convolved = np.maximum(0, convolved)
            feature_maps.append(convolved)
            
        stacked = np.concatenate(feature_maps, axis=1)
        # Combine original normalized features with convolutional feature maps
        return np.hstack([X, stacked])

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self.num_classes = len(self.classes_)
        
        if HAS_TORCH:
            torch.manual_seed(self.random_state)
            self.model_ = PyTorch1DCNNModule(in_features=X.shape[1], num_classes=self.num_classes)
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(self.model_.parameters(), lr=self.lr)
            
            # Label remapping to 0..K-1
            label_map = {c: i for i, c in enumerate(self.classes_)}
            y_mapped = np.array([label_map[val] for val in y])
            
            tensor_x = torch.tensor(X, dtype=torch.float32)
            tensor_y = torch.tensor(y_mapped, dtype=torch.long)
            dataset = TensorDataset(tensor_x, tensor_y)
            loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
            
            self.model_.train()
            for _ in range(self.epochs):
                for batch_x, batch_y in loader:
                    optimizer.zero_grad()
                    out = self.model_(batch_x)
                    loss = criterion(out, batch_y)
                    loss.backward()
                    optimizer.step()
        else:
            # Seamless fallback: 1D Convolutional Feature Maps + MLP Head
            X_conv = self._conv1d_numpy_transform(X)
            self.model_ = MLPClassifier(
                hidden_layer_sizes=(32, 16),
                activation="relu",
                max_iter=150,
                random_state=self.random_state
            )
            self.model_.fit(X_conv, y)
            
        return self

    def predict(self, X):
        if HAS_TORCH:
            self.model_.eval()
            with torch.no_grad():
                tensor_x = torch.tensor(X, dtype=torch.float32)
                logits = self.model_(tensor_x)
                preds = torch.argmax(logits, dim=1).numpy()
            return self.classes_[preds]
        else:
            X_conv = self._conv1d_numpy_transform(X)
            return self.model_.predict(X_conv)

    def predict_proba(self, X):
        if HAS_TORCH:
            self.model_.eval()
            with torch.no_grad():
                tensor_x = torch.tensor(X, dtype=torch.float32)
                logits = self.model_(tensor_x)
                probs = torch.softmax(logits, dim=1).numpy()
            return probs
        else:
            X_conv = self._conv1d_numpy_transform(X)
            return self.model_.predict_proba(X_conv)

def get_1dcnn_model(in_features=8, num_classes=2, random_state=42):
    return CNN1DClassifier(in_features=in_features, num_classes=num_classes, random_state=random_state)
