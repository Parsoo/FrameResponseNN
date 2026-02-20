import numpy as np
from sklearn.neural_network import MLPRegressor
import joblib

def train():
    data = np.load('data/processed.npz')
    X_train = data['X_train']
    y_train = data['y_train']

    mlp = MLPRegressor(
        hidden_layer_sizes=(64, 64, 32),
        activation='relu',
        solver='adam',
        alpha=0.001,
        batch_size='auto',
        learning_rate='adaptive',
        max_iter=1000,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.1
    )

    mlp.fit(X_train, y_train)

    joblib.dump(mlp, 'models/mlp_model.pkl')
    print("Model trained and saved.")
    return mlp

if __name__ == '__main__':
    train()