import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, PowerTransformer, QuantileTransformer
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import logging
import os
import warnings

warnings.filterwarnings('ignore')

DATA_DIR = "data"
LOG_DIR = "logs"

os.makedirs(LOG_DIR, exist_ok=True)


def create_logger(name, log_file):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Avoid duplicate handlers if the function is run again
    if logger.handlers:
        logger.handlers.clear()

    formatter = logging.Formatter(
        '%(asctime)s | %(levelname)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    file_handler = logging.FileHandler(log_file, mode='w')
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


# Separate logs for Ash Content, Fixed Carbon, and Ignition Temperature
ash_logger = create_logger(
    "ash_content",
    os.path.join(LOG_DIR, "ash_content_training.log")
)

carbon_logger = create_logger(
    "fixed_carbon",
    os.path.join(LOG_DIR, "fixed_carbon_training.log")
)

ignition_logger = create_logger(
    "ignition_temperature",
    os.path.join(LOG_DIR, "ignition_temperature_training.log")
)


def evaluate_and_validate(filename, model, name, logger, scaler=None):
    if scaler is None:
        scaler = StandardScaler()

    # 1. Load and Clean
    logger.info("=" * 60)
    logger.info(f"Starting training: {name}")
    logger.info(f"Input file: {filename}")
    logger.info(f"Model: {model.__class__.__name__}")
    logger.info(f"Scaler: {scaler.__class__.__name__}")

    df = pd.read_csv(filename)

    logger.info(f"Dataset shape: {df.shape}")
    logger.info(f"Number of rows: {len(df)}")
    logger.info(f"Number of columns: {len(df.columns)}")

    df_clean = df.dropna(subset=['target_value']).dropna(axis=1, how='all')

    cols_to_drop = [
        'instance_id',
        'coal_sample',
        'pellet',
        'subsample',
        'scs_class',
        'target_value'
    ]

    X = df_clean.drop(
        columns=[col for col in cols_to_drop if col in df_clean.columns]
    )
    y = df_clean['target_value']
    stratify_col = df_clean['coal_sample'] if 'coal_sample' in df_clean.columns else None

    logger.info(f"Cleaned dataset shape: {df_clean.shape}")
    logger.info(f"Number of features used: {X.shape[1]}")
    logger.info(f"Target mean: {y.mean():.6f}")
    logger.info(f"Target minimum: {y.min():.6f}")
    logger.info(f"Target maximum: {y.max():.6f}")

    # 2. Stratified Train/Test Split (ensures representative coal sample distribution)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=stratify_col
    )

    logger.info(f"Training samples: {len(X_train)}")
    logger.info(f"Testing samples: {len(X_test)}")

    imputer = SimpleImputer(strategy='median')
    X_train_imp = imputer.fit_transform(X_train)
    X_test_imp = imputer.transform(X_test)

    X_train_scaled = scaler.fit_transform(X_train_imp)
    X_test_scaled = scaler.transform(X_test_imp)

    # 3. Train & Evaluate Holdout Metrics
    logger.info("Training model...")
    model.fit(X_train_scaled, y_train)

    train_predictions = model.predict(X_train_scaled)
    test_predictions = model.predict(X_test_scaled)

    train_r2 = r2_score(y_train, train_predictions)
    test_r2 = r2_score(y_test, test_predictions)
    test_mae = mean_absolute_error(y_test, test_predictions)
    test_rmse = np.sqrt(mean_squared_error(y_test, test_predictions))
    test_mape = np.mean(np.abs((y_test - test_predictions) / y_test)) * 100

    # 4. Full Data Preprocessing for 5-Fold Cross Validation
    X_imp = imputer.fit_transform(X)
    X_full_scaled = scaler.fit_transform(X_imp)

    kf = KFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    logger.info("Running 5-Fold Cross Validation...")
    cv_scores = cross_val_score(
        model,
        X_full_scaled,
        y,
        cv=kf,
        scoring='r2'
    )

    cv_mean = np.mean(cv_scores)
    cv_std = np.std(cv_scores)

    # 5. Formatted Output
    output = (
        f"\n{'=' * 45}\n"
        f"{name.upper()} | Model: {model.__class__.__name__}\n"
        f"{'=' * 45}\n"
        f"Scaler                 : {scaler.__class__.__name__}\n"
        f"Train R² Accuracy      : {train_r2:.4f}\n"
        f"Test (Holdout) R²      : {test_r2:.4f}\n"
        f"Test MAE               : {test_mae:.4f}\n"
        f"Test RMSE              : {test_rmse:.4f}\n"
        f"Test MAPE (%)          : {test_mape:.2f}%\n"
        f"---------------------------------------------\n"
        f"CV Mean R² (5 Folds)   : {cv_mean:.4f}  <-- True generalization score\n"
        f"CV Std Dev (±)         : {cv_std:.4f}  <-- Model stability\n"
        f"CV Fold Scores         : {np.array2string(cv_scores, precision=4)}\n"
        f"{'=' * 45}\n"
    )

    print(output)

    logger.info(f"Train R² Accuracy: {train_r2:.4f}")
    logger.info(f"Test (Holdout) R²: {test_r2:.4f}")
    logger.info(f"Test MAE: {test_mae:.4f}")
    logger.info(f"Test RMSE: {test_rmse:.4f}")
    logger.info(f"Test MAPE: {test_mape:.2f}%")
    logger.info(f"CV Mean R² (5 Folds): {cv_mean:.4f}")
    logger.info(f"CV Std Dev: {cv_std:.4f}")
    logger.info(
        f"CV Fold Scores: {np.array2string(cv_scores, precision=4)}"
    )
    logger.info(f"Training completed: {name}")
    logger.info("=" * 60)

    return {
        "name": name,
        "model": model.__class__.__name__,
        "scaler": scaler.__class__.__name__,
        "train_r2": train_r2,
        "test_r2": test_r2,
        "test_mae": test_mae,
        "test_rmse": test_rmse,
        "test_mape": test_mape,
        "cv_mean_r2": cv_mean,
        "cv_std": cv_std,
        "cv_scores": cv_scores
    }


# ============================================================
# OPTIMIZED MODELS & SCALERS PER TARGET PROPERTY
# ============================================================

# 1. Ash Content: Extra Trees + PowerTransformer (Yeo-Johnson)
ash_model = ExtraTreesRegressor(
    n_estimators=200,
    max_depth=8,
    max_features=0.7,
    random_state=42
)
ash_scaler = PowerTransformer()

# 2. Fixed Carbon: Tuned Gradient Boosting + StandardScaler
carbon_model = GradientBoostingRegressor(
    n_estimators=160,
    learning_rate=0.03,
    max_depth=5,
    subsample=0.85,
    random_state=42
)
carbon_scaler = StandardScaler()

# 3. Ignition Temperature: Voting Ensemble (Extra Trees + Gradient Boosting) + QuantileTransformer
ignition_model = VotingRegressor(
    estimators=[
        ('et', ExtraTreesRegressor(
            n_estimators=200,
            max_features=1.0,
            random_state=42
        )),
        ('gb', GradientBoostingRegressor(
            n_estimators=160,
            learning_rate=0.03,
            max_depth=5,
            subsample=0.85,
            random_state=42
        ))
    ],
    weights=[0.65, 0.35]
)
ignition_scaler = QuantileTransformer(n_quantiles=50, random_state=42)


# ============================================================
# EXECUTION
# ============================================================

ash_file = os.path.join(DATA_DIR, "features_ash_content.csv")
carbon_file = os.path.join(DATA_DIR, "features_carbon_content.csv")
ignition_file = os.path.join(DATA_DIR, "features_ignition_temp.csv")

ash_results = evaluate_and_validate(
    ash_file,
    ash_model,
    'Ash Content',
    ash_logger,
    scaler=ash_scaler
)

carbon_results = evaluate_and_validate(
    carbon_file,
    carbon_model,
    'Fixed Carbon',
    carbon_logger,
    scaler=carbon_scaler
)

ignition_results = evaluate_and_validate(
    ignition_file,
    ignition_model,
    'Ignition Temperature',
    ignition_logger,
    scaler=ignition_scaler
)


print("\n\nFINAL MULTI-TARGET COAL PROPERTY MODEL SUMMARY")
print("=" * 86)
print(f"{'Target Property':<22} | {'Model':<22} | {'Train R²':<8} | {'Test R²':<8} | {'CV Mean R²':<10} | {'CV Std':<8}")
print("-" * 86)

for result in [ash_results, carbon_results, ignition_results]:
    print(
        f"{result['name']:<22} | "
        f"{result['model']:<22} | "
        f"{result['train_r2']:<8.4f} | "
        f"{result['test_r2']:<8.4f} | "
        f"{result['cv_mean_r2']:<10.4f} | "
        f"±{result['cv_std']:<7.4f}"
    )

print("=" * 86)
print("\nAdditional Error Metrics (Holdout Test Set):")
for result in [ash_results, carbon_results, ignition_results]:
    print(f"  - {result['name']:<22}: MAE = {result['test_mae']:.4f} | RMSE = {result['test_rmse']:.4f} | MAPE = {result['test_mape']:.2f}%")

print("\nLogs saved in:")
print("  logs/ash_content_training.log")
print("  logs/fixed_carbon_training.log")
print("  logs/ignition_temperature_training.log")
