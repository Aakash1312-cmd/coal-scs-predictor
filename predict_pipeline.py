"""
Coal Properties Multi-Target Prediction Pipeline
=================================================
Predicts:
  1. Ash Content (%)
  2. Fixed Carbon Content (%)
  3. Ignition Temperature (°C)

Uses the trained models from train_coal_model.py:
  - Ash Content: ExtraTreesRegressor + PowerTransformer (Yeo-Johnson)
  - Fixed Carbon: GradientBoostingRegressor + StandardScaler
  - Ignition Temperature: VotingRegressor (ExtraTrees + GradientBoosting) + QuantileTransformer

Usage:
  - From terminal:
      python3 predict_pipeline.py                  # Runs check on holdout test set & representative samples
      python3 predict_pipeline.py --instance C5_P1_01 # Predicts for specific instance
      python3 predict_pipeline.py --sample C10     # Predicts average properties for coal sample C10
  - From Python code:
      from predict_pipeline import CoalPropertyPredictor
      predictor = CoalPropertyPredictor()
      result = predictor.predict_instance("C5_P1_01")
      print(result)
"""
import os
import sys
import argparse
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, PowerTransformer, QuantileTransformer
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, VotingRegressor
import joblib

warnings.filterwarnings('ignore')

MODEL_DIR = "saved_models"
DATA_DIR = "data"


def resolve_file(filename):
    """Checks data/ first, then root workspace."""
    if os.path.exists(os.path.join(DATA_DIR, filename)):
        return os.path.join(DATA_DIR, filename)
    if os.path.exists(filename):
        return filename
    raise FileNotFoundError(f"Cannot find {filename} in data/ or root directory.")


class CoalPropertyPredictor:
    """
    End-to-end multi-target prediction pipeline for Ash Content,
    Fixed Carbon Content, and Ignition Temperature.
    """

    def __init__(self, retrain=False):
        self.model_dir = MODEL_DIR
        os.makedirs(self.model_dir, exist_ok=True)

        self.ash_path = os.path.join(self.model_dir, "ash_pipeline.joblib")
        self.carbon_path = os.path.join(self.model_dir, "carbon_pipeline.joblib")
        self.ignition_path = os.path.join(self.model_dir, "ignition_pipeline.joblib")

        self.ash_file = resolve_file("features_ash_content.csv")
        self.carbon_file = resolve_file("features_carbon_content.csv")
        self.ignition_file = resolve_file("features_ignition_temp.csv")

        # Load datasets to hold feature tables & ground truths
        self.df_ash = pd.read_csv(self.ash_file)
        self.df_carbon = pd.read_csv(self.carbon_file)
        self.df_ignition = pd.read_csv(self.ignition_file)

        self.meta_cols = ['instance_id', 'coal_sample', 'pellet', 'subsample', 'scs_class', 'target_value']

        self.ash_feature_cols = [c for c in self.df_ash.columns if c not in self.meta_cols]
        self.carbon_feature_cols = [c for c in self.df_carbon.columns if c not in self.meta_cols]
        self.ignition_feature_cols = [c for c in self.df_ignition.columns if c not in self.meta_cols]

        if retrain or not self._models_exist():
            print("Training and caching coal prediction models...")
            self._train_and_save_all()
        else:
            self._load_all()

    def _models_exist(self):
        return (
            os.path.exists(self.ash_path) and
            os.path.exists(self.carbon_path) and
            os.path.exists(self.ignition_path)
        )

    def _train_single_target(self, df, feature_cols, model, scaler):
        df_clean = df.dropna(subset=['target_value']).dropna(axis=1, how='all')
        X = df_clean[feature_cols]
        y = df_clean['target_value']
        stratify_col = df_clean['coal_sample'] if 'coal_sample' in df_clean.columns else None

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=stratify_col
        )

        imputer = SimpleImputer(strategy='median')
        X_train_imp = imputer.fit_transform(X_train)
        X_train_scaled = scaler.fit_transform(X_train_imp)

        model.fit(X_train_scaled, y_train)

        pipeline_bundle = {
            "imputer": imputer,
            "scaler": scaler,
            "model": model,
            "feature_names": feature_cols,
            "test_indices": X_test.index.tolist(),
        }
        return pipeline_bundle

    def _train_and_save_all(self):
        # 1. Ash Content
        ash_model = ExtraTreesRegressor(
            n_estimators=200,
            max_depth=8,
            max_features=0.7,
            random_state=42
        )
        ash_scaler = PowerTransformer()
        ash_bundle = self._train_single_target(
            self.df_ash, self.ash_feature_cols, ash_model, ash_scaler
        )
        joblib.dump(ash_bundle, self.ash_path)

        # 2. Fixed Carbon
        carbon_model = GradientBoostingRegressor(
            n_estimators=160,
            learning_rate=0.03,
            max_depth=5,
            subsample=0.85,
            random_state=42
        )
        carbon_scaler = StandardScaler()
        carbon_bundle = self._train_single_target(
            self.df_carbon, self.carbon_feature_cols, carbon_model, carbon_scaler
        )
        joblib.dump(carbon_bundle, self.carbon_path)

        # 3. Ignition Temperature
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
        ignition_bundle = self._train_single_target(
            self.df_ignition, self.ignition_feature_cols, ignition_model, ignition_scaler
        )
        joblib.dump(ignition_bundle, self.ignition_path)

        self._load_all()
        print(f"Models successfully trained and saved to '{self.model_dir}/'.")

    def _load_all(self):
        self.ash_bundle = joblib.load(self.ash_path)
        self.carbon_bundle = joblib.load(self.carbon_path)
        self.ignition_bundle = joblib.load(self.ignition_path)

    def _predict_raw(self, bundle, X_features):
        """Preprocesses feature matrix and runs model prediction."""
        X_arr = np.asarray(X_features)
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(1, -1)
        X_imp = bundle["imputer"].transform(X_arr)
        X_scaled = bundle["scaler"].transform(X_imp)
        return bundle["model"].predict(X_scaled)

    def predict_instance(self, instance_id):
        """
        Predicts Ash, Carbon, and Ignition Temp for a specific instance (e.g. 'C5_P1_01').
        Returns predicted values, ground truth (if available), and percentage errors.
        """
        row_ash = self.df_ash[self.df_ash['instance_id'] == instance_id]
        row_carb = self.df_carbon[self.df_carbon['instance_id'] == instance_id]
        row_ign = self.df_ignition[self.df_ignition['instance_id'] == instance_id]

        if row_ash.empty or row_carb.empty or row_ign.empty:
            raise ValueError(f"Instance '{instance_id}' not found in feature tables.")

        sample_code = row_ash.iloc[0]['coal_sample']
        scs_class = row_ash.iloc[0]['scs_class']

        # Extract feature vectors
        x_ash = row_ash[self.ash_feature_cols].values
        x_carb = row_carb[self.carbon_feature_cols].values
        x_ign = row_ign[self.ignition_feature_cols].values

        pred_ash = float(self._predict_raw(self.ash_bundle, x_ash)[0])
        pred_carb = float(self._predict_raw(self.carbon_bundle, x_carb)[0])
        pred_ign = float(self._predict_raw(self.ignition_bundle, x_ign)[0])

        true_ash = row_ash.iloc[0]['target_value']
        true_carb = row_carb.iloc[0]['target_value']
        true_ign = row_ign.iloc[0]['target_value']

        def calc_err(pred, true_val):
            if pd.isna(true_val):
                return None, None
            abs_err = abs(pred - true_val)
            pct_err = (abs_err / true_val) * 100
            return abs_err, pct_err

        ash_abs, ash_pct = calc_err(pred_ash, true_ash)
        carb_abs, carb_pct = calc_err(pred_carb, true_carb)
        ign_abs, ign_pct = calc_err(pred_ign, true_ign)

        return {
            "instance_id": instance_id,
            "coal_sample": sample_code,
            "scs_class": scs_class,
            "ash_content": {
                "predicted": pred_ash,
                "ground_truth": true_ash,
                "abs_error": ash_abs,
                "pct_error": ash_pct,
                "unit": "%"
            },
            "fixed_carbon": {
                "predicted": pred_carb,
                "ground_truth": true_carb,
                "abs_error": carb_abs,
                "pct_error": carb_pct,
                "unit": "%"
            },
            "ignition_temp": {
                "predicted": pred_ign,
                "ground_truth": true_ign,
                "abs_error": ign_abs,
                "pct_error": ign_pct,
                "unit": "°C"
            }
        }

    def predict_coal_sample(self, sample_code):
        """
        Aggregates all measurement sites for a coal sample (e.g. 'C5')
        and returns mean predicted properties with standard deviations.
        """
        instances = self.df_ash[self.df_ash['coal_sample'] == sample_code]['instance_id'].tolist()
        if not instances:
            raise ValueError(f"Coal sample '{sample_code}' not found.")

        preds_ash = []
        preds_carb = []
        preds_ign = []

        for inst in instances:
            res = self.predict_instance(inst)
            preds_ash.append(res["ash_content"]["predicted"])
            preds_carb.append(res["fixed_carbon"]["predicted"])
            preds_ign.append(res["ignition_temp"]["predicted"])

        first_res = self.predict_instance(instances[0])
        true_ash = first_res["ash_content"]["ground_truth"]
        true_carb = first_res["fixed_carbon"]["ground_truth"]
        true_ign = first_res["ignition_temp"]["ground_truth"]

        mean_ash = float(np.mean(preds_ash))
        std_ash = float(np.std(preds_ash))

        mean_carb = float(np.mean(preds_carb))
        std_carb = float(np.std(preds_carb))

        mean_ign = float(np.mean(preds_ign))
        std_ign = float(np.std(preds_ign))

        return {
            "coal_sample": sample_code,
            "scs_class": first_res["scs_class"],
            "num_shots": len(instances),
            "ash_content": {
                "mean_predicted": mean_ash,
                "std_predicted": std_ash,
                "ground_truth": true_ash,
                "error": abs(mean_ash - true_ash) if pd.notna(true_ash) else None,
                "pct_error": (abs(mean_ash - true_ash) / true_ash * 100) if pd.notna(true_ash) else None,
                "unit": "%"
            },
            "fixed_carbon": {
                "mean_predicted": mean_carb,
                "std_predicted": std_carb,
                "ground_truth": true_carb,
                "error": abs(mean_carb - true_carb) if pd.notna(true_carb) else None,
                "pct_error": (abs(mean_carb - true_carb) / true_carb * 100) if pd.notna(true_carb) else None,
                "unit": "%"
            },
            "ignition_temp": {
                "mean_predicted": mean_ign,
                "std_predicted": std_ign,
                "ground_truth": true_ign,
                "error": abs(mean_ign - true_ign) if pd.notna(true_ign) else None,
                "pct_error": (abs(mean_ign - true_ign) / true_ign * 100) if pd.notna(true_ign) else None,
                "unit": "°C"
            }
        }

    def print_check_table(self):
        """
        Runs check across representative samples from each SCS class
        and displays a formatted multi-target prediction table.
        """
        samples = ["C5", "C9", "C10", "C12", "C4", "C13", "C14", "C6", "C7", "C15", "C16"]

        print("\n" + "=" * 100)
        print("COAL PROPERTY MULTI-TARGET PREDICTION PIPELINE CHECK")
        print("=" * 100)
        header = (
            f"{'Sample':<7} | {'Class':<7} | "
            f"{'Ash True':<8} {'Ash Pred':<9} {'Err %':<6} | "
            f"{'Carb True':<9} {'Carb Pred':<9} {'Err %':<6} | "
            f"{'Ign True':<9} {'Ign Pred':<9} {'Err %':<6}"
        )
        print(header)
        print("-" * 100)

        total_ash_err = []
        total_carb_err = []
        total_ign_err = []

        for code in samples:
            res = self.predict_coal_sample(code)
            a = res["ash_content"]
            c = res["fixed_carbon"]
            i = res["ignition_temp"]

            a_true_str = f"{a['ground_truth']:.2f}%" if pd.notna(a['ground_truth']) else "N/A"
            a_pred_str = f"{a['mean_predicted']:.2f}%"
            a_err_str = f"{a['pct_error']:.1f}%" if a['pct_error'] is not None else "-"
            if a['pct_error'] is not None:
                total_ash_err.append(a['pct_error'])

            c_true_str = f"{c['ground_truth']:.2f}%" if pd.notna(c['ground_truth']) else "N/A"
            c_pred_str = f"{c['mean_predicted']:.2f}%"
            c_err_str = f"{c['pct_error']:.1f}%" if c['pct_error'] is not None else "-"
            if c['pct_error'] is not None:
                total_carb_err.append(c['pct_error'])

            i_true_str = f"{i['ground_truth']:.1f}°C"
            i_pred_str = f"{i['mean_predicted']:.1f}°C"
            i_err_str = f"{i['pct_error']:.1f}%"
            total_ign_err.append(i['pct_error'])

            print(
                f"{code:<7} | {res['scs_class']:<7} | "
                f"{a_true_str:<8} {a_pred_str:<9} {a_err_str:<6} | "
                f"{c_true_str:<9} {c_pred_str:<9} {c_err_str:<6} | "
                f"{i_true_str:<9} {i_pred_str:<9} {i_err_str:<6}"
            )

        print("-" * 100)
        print(
            f"{'MEAN':<7} | {'ALL':<7} | "
            f"{'-':<8} {'-':<9} {np.mean(total_ash_err):.1f}%  | "
            f"{'-':<9} {'-':<9} {np.mean(total_carb_err):.1f}%  | "
            f"{'-':<9} {'-':<9} {np.mean(total_ign_err):.1f}%"
        )
        print("=" * 100)


def main():
    parser = argparse.ArgumentParser(description="Multi-Target Coal Property Prediction Pipeline")
    parser.add_argument("--instance", type=str, help="Instance ID to predict (e.g., C5_P1_01)")
    parser.add_argument("--sample", type=str, help="Coal sample code to predict (e.g., C5, C10, C16)")
    parser.add_argument("--retrain", action="store_true", help="Force retrain models from scratch")
    args = parser.parse_args()

    predictor = CoalPropertyPredictor(retrain=args.retrain)

    if args.instance:
        res = predictor.predict_instance(args.instance)
        print(f"\n=======================================================")
        print(f"PREDICTION REPORT FOR INSTANCE: {args.instance}")
        print(f"=======================================================")
        print(f"Coal Sample : {res['coal_sample']} (SCS Class: {res['scs_class']})")
        print("-" * 55)
        for key, title in [("ash_content", "Ash Content"), ("fixed_carbon", "Fixed Carbon"), ("ignition_temp", "Ignition Temperature")]:
            d = res[key]
            true_str = f"{d['ground_truth']} {d['unit']}" if pd.notna(d['ground_truth']) else "N/A"
            err_str = f"{d['pct_error']:.2f}%" if d['pct_error'] is not None else "N/A"
            print(f"{title:<22}: Predicted = {d['predicted']:.3f} {d['unit']} | True = {true_str} | Error = {err_str}")
        print("=======================================================\n")
    elif args.sample:
        res = predictor.predict_coal_sample(args.sample)
        print(f"\n=======================================================")
        print(f"PREDICTION REPORT FOR SAMPLE: {args.sample} (Aggregated {res['num_shots']} sites)")
        print(f"=======================================================")
        print(f"SCS Class: {res['scs_class']}")
        print("-" * 55)
        for key, title in [("ash_content", "Ash Content"), ("fixed_carbon", "Fixed Carbon"), ("ignition_temp", "Ignition Temperature")]:
            d = res[key]
            true_str = f"{d['ground_truth']} {d['unit']}" if pd.notna(d['ground_truth']) else "N/A"
            err_str = f"{d['pct_error']:.2f}%" if d['pct_error'] is not None else "N/A"
            print(f"{title:<22}: Predicted = {d['mean_predicted']:.3f} ± {d['std_predicted']:.2f} {d['unit']} | True = {true_str} | Error = {err_str}")
        print("=======================================================\n")
    else:
        # Default: run comprehensive check table
        predictor.print_check_table()


if __name__ == "__main__":
    main()
