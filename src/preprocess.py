import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

def preprocess(input_file='data/raw_dataset.csv', output_dir='data/'):
    df = pd.read_csv(input_file)
    initial_len = len(df)
    df = df.dropna()
    print(f"Dropped {initial_len - len(df)} rows due to failed simulations.")

    feature_names = ['E', 'I_col', 'mass', 'damping_ratio', 'scale']
    target_names = ['roof_disp', 'drift_max', 'accel_max']

    X = df[feature_names].values
    y = df[target_names].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler_X = StandardScaler()
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)

    scaler_y = StandardScaler()
    y_train_scaled = scaler_y.fit_transform(y_train)
    y_test_scaled = scaler_y.transform(y_test)

    joblib.dump(scaler_X, 'models/scaler_X.pkl')
    joblib.dump(scaler_y, 'models/scaler_y.pkl')

    np.savez('data/processed.npz',
             X_train=X_train_scaled, X_test=X_test_scaled,
             y_train=y_train_scaled, y_test=y_test_scaled,
             feature_names=feature_names, target_names=target_names)

    np.savez('data/test_unscaled.npz',
             X_test_unscaled=X_test, y_test_unscaled=y_test)

    print("Preprocessing completed.")

if __name__ == '__main__':
    preprocess()