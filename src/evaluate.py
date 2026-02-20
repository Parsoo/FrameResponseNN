import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, r2_score
import joblib

def evaluate():
    scaler_y = joblib.load('models/scaler_y.pkl')
    mlp = joblib.load('models/mlp_model.pkl')

    data = np.load('data/processed.npz')
    X_test = data['X_test']
    y_test_scaled = data['y_test']

    y_pred_scaled = mlp.predict(X_test)

    y_test = scaler_y.inverse_transform(y_test_scaled)
    y_pred = scaler_y.inverse_transform(y_pred_scaled)

    target_names = ['Roof Displacement (m)', 'Max Drift Ratio', 'Max Acceleration (m/s²)']
    for i, name in enumerate(target_names):
        mae = mean_absolute_error(y_test[:, i], y_pred[:, i])
        r2 = r2_score(y_test[:, i], y_pred[:, i])
        print(f"{name}: MAE = {mae:.4f}, R² = {r2:.4f}")

    # Scatter plots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for i, ax in enumerate(axes):
        ax.scatter(y_test[:, i], y_pred[:, i], alpha=0.6, edgecolors='k', linewidth=0.5)
        ax.plot([y_test[:, i].min(), y_test[:, i].max()],
                [y_test[:, i].min(), y_test[:, i].max()], 'r--', lw=2)
        ax.set_xlabel('True')
        ax.set_ylabel('Predicted')
        ax.set_title(target_names[i])
        ax.grid(True)
    plt.tight_layout()
    plt.savefig('results/scatter_plots.png', dpi=150)
    print("Scatter plots saved to results/scatter_plots.png")

    # Loss curve
    plt.figure()
    plt.plot(mlp.loss_curve_)
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve')
    plt.grid(True)
    plt.savefig('results/loss_curve.png', dpi=150)
    print("Loss curve saved to results/loss_curve.png")
    plt.show()

if __name__ == '__main__':
    evaluate()