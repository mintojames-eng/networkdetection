"""
Artificial Neural Network (ANN) Model
Architecture:
Input Features -> Dense (64, ReLU) -> Dense (32, ReLU) -> Output
"""

from sklearn.neural_network import MLPClassifier

def get_ann_model(random_state=42, max_iter=300):
    """
    Constructs a multi-layer perceptron neural network matching the requested architecture:
    Input -> Dense(64) -> Dense(32) -> Output
    """
    return MLPClassifier(
        hidden_layer_sizes=(64, 32),
        activation="relu",
        solver="adam",
        alpha=0.0001,
        batch_size=64,
        learning_rate="adaptive",
        learning_rate_init=0.001,
        max_iter=max_iter,
        early_stopping=True,
        n_iter_no_change=10,
        random_state=random_state
    )
