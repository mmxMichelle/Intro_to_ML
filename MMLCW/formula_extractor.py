import numpy as np
import torch
import torch.nn as nn
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import shap
from gplearn.genetic import SymbolicRegressor
from gplearn.functions import make_function
import warnings
warnings.filterwarnings('ignore')

# Additional imports for improved symbolic regression helper
import os
import sys
import joblib
from typing import Optional, Sequence, Dict, Any, Tuple
from sklearn.metrics import r2_score, mean_squared_error
from scipy.stats import pearsonr

# Try to import PySR; if not available, we'll fallback to gplearn
try:
    from pysr import PySRRegressor
    _HAVE_PYSR = True
except Exception:
    _HAVE_PYSR = False

# gplearn availability flag (module imported earlier in this file)
try:
    _HAVE_GPLEARN = SymbolicRegressor is not None
except Exception:
    _HAVE_GPLEARN = False


# Utility: attempt to convert a PySR expression to a readable string
def _format_pysr_equation(model) -> str:
    try:
        return model.sympy()
    except Exception:
        try:
            return str(model)
        except Exception:
            return "<PySR formula (string conversion failed)>"


def _format_gplearn_program(program) -> str:
    try:
        return str(program)
    except Exception:
        return "<gplearn formula (string conversion failed)>"


def run_improved_symbolic_regression(
    X: np.ndarray,
    y: np.ndarray,
    *,
    scaler=None,
    feature_names: Optional[Sequence[str]] = None,
    use_unscaled: bool = True,
    fit_to_true: bool = True,
    prefer_pysr: bool = True,
    pysr_kwargs: Optional[Dict[str, Any]] = None,
    gplearn_kwargs: Optional[Dict[str, Any]] = None,
    seed: int = 42,
    save_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Improved symbolic regression helper.

    Parameters
    ----------
    X : np.ndarray
        Input features (shape n_samples x n_features). Can be scaled.
    y : np.ndarray
        Target values (true labels) or whichever target you choose.
    scaler : object, optional
        Fitted scaler (e.g., StandardScaler). If provided and use_unscaled=True,
        function will call scaler.inverse_transform(X) to run symbolic regression on original units.
    feature_names : list[str], optional
        Human-readable names for features. If None, uses x0,x1,...
    use_unscaled : bool
        If True and scaler is provided, operate on inverse-transformed X (original units).
    fit_to_true : bool
        If True use `y` as provided (usually true labels). If False, caller can pass in NN predictions instead.
    prefer_pysr : bool
        If True, try PySR first (fast/robust). If PySR not available, fallback to gplearn.
    pysr_kwargs : dict, optional
        Extra kwargs to pass to PySRRegressor.
    gplearn_kwargs : dict, optional
        Extra kwargs to pass to gplearn.SymbolicRegressor.
    seed : int
        Random seed for reproducibility.
    save_path : str, optional
        Directory path to save discovered model/program and metadata (if provided).

    Returns
    -------
    results : dict
        {
            'method': 'pysr' or 'gplearn',
            'model': fitted_model_object,
            'formula': readable_formula_str,
            'metrics': {'r2': ..., 'mse': ..., 'pearson_r': ..., 'pearson_p': ...},
            'X_used': X_used (n x d),
            'y_used': y_used (n,),
            'feature_names': feature_names_used,
            'save_paths': {...} (if saved)
        }
    """
    np.random.seed(seed)

    if feature_names is None:
        feature_names = [f"x{i}" for i in range(X.shape[1])]

    # Decide which X to use (scaled vs unscaled)
    if use_unscaled and scaler is not None:
        try:
            X_used = scaler.inverse_transform(X)
        except Exception as e:
            warnings.warn(f"scaler.inverse_transform failed: {e}. Falling back to input X as-is.")
            X_used = X.copy()
    else:
        X_used = X.copy()

    y_used = y.copy()  # generally true labels as provided

    # Prepare defaults for PySR and gplearn
    pysr_kwargs = {} if pysr_kwargs is None else dict(pysr_kwargs)
    gplearn_kwargs = {} if gplearn_kwargs is None else dict(gplearn_kwargs)

    # sensible defaults for PySR (if used)
    pysr_defaults = dict(
        niterations=1000,
        binary_operators=["+", "-", "*", "/", "^"],
        unary_operators=["sin", "cos", "exp", "log", "sqrt", "abs"],
        # model_selection can be "best" or "accuracy" etc.
        model_selection="best",
        # verbosity: 0 or 1
        verbosity=0,
        random_state=seed,
        # set several threads; PySR uses Julia's multithreading - let user override
        nprocs=1,
    )
    for k, v in pysr_defaults.items():
        pysr_kwargs.setdefault(k, v)

    # sensible defaults for gplearn fallback
    gplearn_defaults = dict(
        population_size=2000,
        generations=200,
        stopping_criteria=0.99,
        # Note: gplearn does not provide a built-in 'pow' operator name. Keep
        # only the standard safe operators here; power is provided via a
        # protected custom function when needed.
        function_set=("add", "sub", "mul", "div", "sqrt", "log", "abs", "sin", "cos"),
        parsimony_coefficient=0.001,
        metric="pearson",  # try to favor correlation
        random_state=seed,
        n_jobs=-1,
    )
    for k, v in gplearn_defaults.items():
        gplearn_kwargs.setdefault(k, v)

    results: Dict[str, Any] = {
        "method": None,
        "model": None,
        "formula": None,
        "metrics": None,
        "X_used": X_used,
        "y_used": y_used,
        "feature_names": feature_names,
        "save_paths": {},
    }

    # Try PySR first if preferred
    if prefer_pysr and _HAVE_PYSR:
        try:
            model = PySRRegressor(**pysr_kwargs)
            model.fit(X_used, y_used)
            formula_readable = _format_pysr_equation(model)
            y_pred = model.predict(X_used)
            results.update({"method": "pysr", "model": model, "formula": formula_readable})
        except Exception as e:
            warnings.warn(f"PySR attempt failed with error: {e}. Will try gplearn fallback.")
            model = None
    else:
        model = None

    # If PySR not used or failed, fallback to gplearn (if available)
    if model is None:
        if not _HAVE_GPLEARN:
            raise RuntimeError(
                "Neither PySR nor gplearn is available. "
                "Install PySR (`pip install pysr`) or gplearn (`pip install gplearn`)."
            )
        # Fit gplearn
        try:
            gp = SymbolicRegressor(**gplearn_kwargs)
            gp.fit(X_used, y_used)
            formula_readable = _format_gplearn_program(gp._program) if hasattr(gp, "_program") else str(gp)
            y_pred = gp.predict(X_used)
            results.update({"method": "gplearn", "model": gp, "formula": formula_readable})
        except Exception as e:
            raise RuntimeError(f"gplearn SymbolicRegressor failed: {e}")

    # Compute metrics
    r2 = float(r2_score(y_used, y_pred))
    mse = float(mean_squared_error(y_used, y_pred))
    try:
        pearson_r, pearson_p = pearsonr(y_used.ravel(), y_pred.ravel())
        pearson_r = float(pearson_r)
        pearson_p = float(pearson_p)
    except Exception:
        pearson_r, pearson_p = float("nan"), float("nan")

    results["metrics"] = {"r2": r2, "mse": mse, "pearson_r": pearson_r, "pearson_p": pearson_p}

    # Optionally save
    if save_path:
        os.makedirs(save_path, exist_ok=True)
        base = os.path.join(save_path, f"symbolic_{results['method']}")
        # Save model (joblib/pickle depending on type)
        try:
            joblib.dump(results["model"], base + ".joblib")
            results["save_paths"]["model"] = base + ".joblib"
        except Exception:
            try:
                with open(base + ".pkl", "wb") as f:
                    pickle.dump(results["model"], f)
                results["save_paths"]["model"] = base + ".pkl"
            except Exception as e:
                warnings.warn(f"Failed to save model object: {e}")

        # Save metadata + formula string
        meta = {
            "method": results["method"],
            "formula": results["formula"],
            "metrics": results["metrics"],
            "feature_names": feature_names,
            "use_unscaled": use_unscaled,
            "fit_to_true": fit_to_true,
            "seed": seed,
        }
        with open(base + "_meta.pkl", "wb") as f:
            pickle.dump(meta, f)
        results["save_paths"]["meta"] = base + "_meta.pkl"

    # Print summary
    print("=== Symbolic Regression Summary ===")
    print("Method:", results["method"])
    print("Formula (readable):", results["formula"])
    print("Metrics: R²={r2:.6f}, MSE={mse:.6f}, Pearson r={r:.6f} (p={p:.2e})".format(
        r2=results["metrics"]["r2"],
        mse=results["metrics"]["mse"],
        r=results["metrics"]["pearson_r"],
        p=results["metrics"]["pearson_p"],
    ))
    if save_path:
        print("Saved: ", results["save_paths"])

    return results

class FormulaExtractor:
    def __init__(self, model_save_path="saved_models"):
        self.model_save_path = model_save_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def load_best_model(self):
        """Load the best trained model and data"""
        # Load model data (weights_only=False needed for sklearn objects in PyTorch 2.6+)
        model_data = torch.load(f"{self.model_save_path}/best_model.pt", 
                               map_location=self.device, 
                               weights_only=False)
        
        # Recreate model
        from mlp_trainer import SimpleMLP
        self.model = SimpleMLP(input_dim=model_data['input_dim'])
        self.model.load_state_dict(model_data['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        # Load scaler and data
        self.scaler = model_data['scaler']
        self.X = np.load(f"{self.model_save_path}/X_original.npy")
        self.y = np.load(f"{self.model_save_path}/y_original.npy")
        
        # Scale the data
        self.X_scaled = self.scaler.transform(self.X)
        
        print(f"Loaded best model with input dimension: {model_data['input_dim']}")
        print(f"Model performance: {model_data['performance']}")
        
        return model_data

    def diagnose_predictions(self):
        """Diagnose the neural network's prediction distribution"""
        print("\n" + "="*60)
        print("DIAGNOSING PREDICTIONS")
        print("="*60)
        
        X_tensor = torch.tensor(self.X_scaled, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            y_pred = self.model(X_tensor).cpu().numpy().flatten()
        
        print(f"\nPrediction Statistics:")
        print(f"  Min:    {y_pred.min():.4f}")
        print(f"  Max:    {y_pred.max():.4f}")
        print(f"  Mean:   {y_pred.mean():.4f}")
        print(f"  Std:    {y_pred.std():.4f}")
        print(f"  Range:  {y_pred.max() - y_pred.min():.4f}")
        
        if y_pred.std() < 0.1:
            print("\n⚠️  WARNING: Low variance! Formula extraction will be difficult.")
        
        return y_pred
    
    def analyze_with_shap(self, n_samples=1000, create_plots=True):
        """Analyze model using SHAP to understand feature importance and interactions"""
        print("\n=== SHAP Analysis ===")
        
        # Sample data for SHAP analysis (SHAP can be slow on large datasets)
        if len(self.X_scaled) > n_samples:
            indices = np.random.choice(len(self.X_scaled), n_samples, replace=False)
            X_sample = self.X_scaled[indices]
            y_sample = self.y[indices]
        else:
            X_sample = self.X_scaled
            y_sample = self.y
        
        # Create a wrapper function for the model
        def model_predict(x):
            x_tensor = torch.tensor(x, dtype=torch.float32).to(self.device)
            with torch.no_grad():
                return self.model(x_tensor).cpu().numpy().flatten()
        
        # Initialize SHAP explainer
        explainer = shap.Explainer(model_predict, X_sample)
        shap_values = explainer(X_sample)
        
        self.shap_values = shap_values
        # Keep both scaled and unscaled copies of the sample ---
        # SHAP plots use scaled inputs, but symbolic regression should
        # operate on unscaled features so multiplicative/power relations
        # remain meaningful.
        self.X_sample = X_sample
        try:
            if hasattr(self, 'scaler') and self.scaler is not None:
                X_sample_unscaled = self.scaler.inverse_transform(X_sample)
            else:
                X_sample_unscaled = X_sample.copy()
        except Exception:
            X_sample_unscaled = X_sample.copy()
        self.X_sample_unscaled = X_sample_unscaled
        self.y_sample = y_sample
        
        if create_plots:
            # Create SHAP plots
            plt.figure(figsize=(15, 10))
            
            # Summary plot
            plt.subplot(2, 2, 1)
            shap.summary_plot(shap_values.values, X_sample, 
                            feature_names=[f'Feature_{i}' for i in range(X_sample.shape[1])],
                            show=False)
            plt.title('SHAP Summary Plot')
            
            # Feature importance
            plt.subplot(2, 2, 2)
            shap.summary_plot(shap_values.values, X_sample,
                            feature_names=[f'Feature_{i}' for i in range(X_sample.shape[1])],
                            plot_type="bar", show=False)
            plt.title('SHAP Feature Importance')
            
            # Dependence plot for top features
            feature_importance = np.mean(np.abs(shap_values.values), axis=0)
            top_features = np.argsort(feature_importance)[-2:]
            
            plt.subplot(2, 2, 3)
            shap.dependence_plot(top_features[-1], shap_values.values, X_sample,
                               feature_names=[f'Feature_{i}' for i in range(X_sample.shape[1])],
                               show=False)
            plt.title(f'SHAP Dependence Plot - Feature {top_features[-1]}')
            
            plt.subplot(2, 2, 4)
            if len(top_features) > 1:
                shap.dependence_plot(top_features[-2], shap_values.values, X_sample,
                                   feature_names=[f'Feature_{i}' for i in range(X_sample.shape[1])],
                                   show=False)
                plt.title(f'SHAP Dependence Plot - Feature {top_features[-2]}')
            
            plt.tight_layout()
            plt.savefig(f"{self.model_save_path}/shap_analysis.png", dpi=300, bbox_inches='tight')
            plt.show()
        
        return shap_values
    
    def extract_formula_symbolic_regression(self, use_shap_weights=True,
                                           population_size=2000, generations=100,
                                           prefer_logits=True):
        """Improved symbolic extraction (fits to true labels or logits; uses unscaled features).

        This implementation prefers using unscaled inputs (if available), fits to
        true labels or logits (pre-sigmoid) when possible, and uses protected
        math functions to make search robust.
        """
        print("\n=== Symbolic Regression (improved) ===")

        if not hasattr(self, 'shap_values'):
            print("Running SHAP analysis first...")
            self.analyze_with_shap()

        # Prefer unscaled inputs for symbolic regression
        if hasattr(self, 'X_sample_unscaled'):
            X_full = self.X_sample_unscaled
        else:
            X_full = self.X_sample.copy()

        # SHAP-driven top features (apply selection on unscaled data)
        feature_importance = np.mean(np.abs(self.shap_values.values), axis=0)
        top_idx = np.argsort(feature_importance)[-5:]
        X_for_symreg = X_full[:, top_idx]
        feature_names = [f'x{i}' for i in top_idx]
        print(f"Using top {len(top_idx)} features (unscaled): {top_idx}")

        # Determine target: true labels (best), else logits (preferred for classification), else probabilities
        if hasattr(self, 'y_sample') and self.y_sample is not None:
            y_target = self.y_sample.astype(float)
            print("Target: TRUE labels (0/1).")
            # If labels are binary and we prefer logits, try to compute logits from model predictions
            if prefer_logits:
                try:
                    X_tensor = torch.tensor(self.X_sample, dtype=torch.float32).to(self.device)
                    with torch.no_grad():
                        y_prob = self.model(X_tensor).cpu().numpy().flatten()
                    eps = 1e-6
                    p = np.clip(y_prob, eps, 1 - eps)
                    logits = np.log(p / (1 - p))
                    y_target = logits
                    print("Converted NN probabilities -> logits and will fit to logits.")
                except Exception:
                    print("Could not compute logits from model; fitting to true labels instead.")
        else:
            # fallback: use model predictions and convert to logits safely
            X_tensor = torch.tensor(self.X_sample, dtype=torch.float32).to(self.device)
            with torch.no_grad():
                y_prob = self.model(X_tensor).cpu().numpy().flatten()
            eps = 1e-6
            p = np.clip(y_prob, eps, 1 - eps)
            y_target = np.log(p / (1 - p))
            print("WARNING: No true labels available - fitting to NN logits (fallback).")

        # Protected functions
        def protected_div(x1, x2):
            return np.where(np.abs(x2) > 1e-6, x1 / x2, 1.0)

        def protected_sqrt(x):
            return np.sqrt(np.abs(x))

        def protected_log(x):
            return np.log(np.abs(x) + 1e-6)

        def protected_pow(x1, x2):
            """Protected power: avoid complex/invalid results by operating on absolute
            value for fractional exponents and clipping large outputs.
            """
            # base and exponent -> use safe absolute base, preserve sign when exponent is integer
            with np.errstate(all='ignore'):
                base = np.abs(x1) + 1e-6
                # compute power; where result is finite use it, else fallback to 1.0
                res = np.power(base, x2)
                res = np.where(np.isfinite(res), res, 1.0)
            # restore sign if exponent is an odd integer (approx)
            try:
                is_odd_int = (np.abs(np.round(x2) - x2) < 1e-9) & (np.round(x2) % 2 != 0)
                res = np.where(is_odd_int & (x1 < 0), -res, res)
            except Exception:
                pass
            return res

        # gplearn custom functions
        pdiv = make_function(function=protected_div, name='pdiv', arity=2)
        psqrt = make_function(function=protected_sqrt, name='psqrt', arity=1)
        plog = make_function(function=protected_log, name='plog', arity=1)
        ppow = make_function(function=protected_pow, name='ppow', arity=2)

        function_set = ['add', 'sub', 'mul', pdiv, 'sin', 'cos', 'abs', ppow, psqrt, plog]

        est_gp = SymbolicRegressor(
            population_size=population_size,
            generations=generations,
            stopping_criteria=0.995,
            function_set=function_set,
            parsimony_coefficient=0.001,
            metric='mean absolute error',
            random_state=42,
            n_jobs=-1,
            verbose=1
        )

        print("Training symbolic regressor (this may take a while)...")
        est_gp.fit(X_for_symreg, y_target)

        # Read formula & evaluate
        try:
            program = est_gp._program
            formula = str(program)
        except Exception:
            formula = str(est_gp)

        # Replace variable names X0.. with readable names
        readable = formula
        for i in range(len(top_idx)):
            readable = readable.replace(f'X{i}', feature_names[i])

        # Predictions on the same samples
        y_formula_pred = est_gp.predict(X_for_symreg)
        # If we fit to logits, convert back to probability for comparison with labels
        corr_logit, corr_label = np.nan, np.nan
        if hasattr(self, 'y_sample') and self.y_sample is not None and np.all(np.isin(self.y_sample, [0, 1])):
            try:
                corr_logit = np.corrcoef(y_target, y_formula_pred)[0, 1]
                p_formula = 1 / (1 + np.exp(-y_formula_pred))
                corr_label = np.corrcoef(self.y_sample, p_formula)[0, 1]
            except Exception:
                corr_logit, corr_label = np.nan, np.nan
        else:
            try:
                corr_logit = np.corrcoef(y_target, y_formula_pred)[0, 1]
            except Exception:
                corr_logit = np.nan

        mse = mean_squared_error(y_target, y_formula_pred)
        r2 = r2_score(y_target, y_formula_pred)

        print("Discovered formula:", readable)
        print(f"R² (on target): {r2:.6f}, MSE: {mse:.6f}, Corr (target vs formula): {corr_logit:.6f}")
        if not np.isnan(corr_label):
            print(f"Corr (labels vs formula->sigmoid): {corr_label:.6f}")

        # Save results
        results = {
            'model': est_gp,
            'formula': formula,
            'readable_formula': readable,
            'r2_score': r2,
            'mse': mse,
            'corr_with_target': float(corr_logit) if not np.isnan(corr_logit) else None,
            'corr_labels_sigmoid': float(corr_label) if not np.isnan(corr_label) else None,
            'top_features_idx': top_idx,
            'feature_names': feature_names
        }
        try:
            with open(f"{self.model_save_path}/extracted_formula_symreg.pkl", "wb") as f:
                pickle.dump(results, f)
        except Exception as e:
            warnings.warn(f"Failed to save symbolic regression results: {e}")

        return est_gp, results
    
    def extract_formula_polynomial(self, degree=2):
        """Fallback: Extract polynomial approximation"""
        print("\n" + "="*60)
        print("FALLBACK: POLYNOMIAL APPROXIMATION")
        print("="*60)
        
        from sklearn.preprocessing import PolynomialFeatures
        from sklearn.linear_model import Ridge
        
        if not hasattr(self, 'shap_values'):
            self.analyze_with_shap()
        
        # Get predictions
        X_tensor = torch.tensor(self.X_sample, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            y_pred = self.model(X_tensor).cpu().numpy().flatten()
        
        # Use top 5 SHAP features
        feature_importance = np.mean(np.abs(self.shap_values.values), axis=0)
        top_features_idx = np.argsort(feature_importance)[-5:]
        X_for_poly = self.X_sample[:, top_features_idx]
        feature_names = [f'x{i}' for i in top_features_idx]
        
        print(f"Using top {len(top_features_idx)} features: {top_features_idx}")
        
        # Create polynomial features
        poly = PolynomialFeatures(degree=degree, include_bias=True)
        X_poly = poly.fit_transform(X_for_poly)
        
        # Fit ridge regression
        ridge = Ridge(alpha=0.1)
        ridge.fit(X_poly, y_pred)
        
        # Build formula
        feature_names_poly = poly.get_feature_names_out(feature_names)
        coefficients = ridge.coef_
        intercept = ridge.intercept_
        
        formula_parts = []
        if abs(intercept) > 0.001:
            formula_parts.append(f"{intercept:.4f}")
        
        for coef, feat_name in zip(coefficients, feature_names_poly):
            if abs(coef) > 0.001:
                sign = "+" if coef > 0 else "-"
                formula_parts.append(f" {sign} {abs(coef):.4f}*{feat_name}")
        
        readable_formula = "".join(formula_parts)
        
        # Evaluate
        y_poly_pred = ridge.predict(X_poly)
        r2 = ridge.score(X_poly, y_pred)
        corr = np.corrcoef(y_pred, y_poly_pred)[0, 1]
        mse = np.mean((y_pred - y_poly_pred) ** 2)
        
        print(f"\nPolynomial Formula:")
        print(readable_formula)
        print(f"\nPerformance:")
        print(f"  R² Score: {r2:.4f}")
        print(f"  Correlation: {corr:.4f}")
        print(f"  MSE: {mse:.6f}")
        
        results = {
            'formula': readable_formula,
            'readable_formula': readable_formula,
            'r2_score': r2,
            'correlation_with_nn': corr,
            'mse': mse,
            'feature_names': feature_names,
            'poly_transformer': poly  # ADD THIS LINE
        }

        return ridge, results

    def extract_formula_decision_tree(self, max_depth=5):
        """Extract interpretable rules using decision tree"""
        print("\n" + "="*60)
        print("DECISION TREE APPROXIMATION")
        print("="*60)

        from sklearn.tree import DecisionTreeRegressor, export_text

        if not hasattr(self, 'shap_values'):
            self.analyze_with_shap()

        # Get predictions
        X_tensor = torch.tensor(self.X_sample, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            y_pred = self.model(X_tensor).cpu().numpy().flatten()

        # Use top 5 SHAP features
        feature_importance = np.mean(np.abs(self.shap_values.values), axis=0)
        top_features_idx = np.argsort(feature_importance)[-5:]
        X_for_tree = self.X_sample[:, top_features_idx]
        feature_names = [f'x{i}' for i in top_features_idx]

        print(f"Using top {len(top_features_idx)} features: {top_features_idx}")

        # Train decision tree
        tree = DecisionTreeRegressor(max_depth=max_depth, random_state=42)
        tree.fit(X_for_tree, y_pred)

        # Get tree rules
        tree_rules = export_text(tree, feature_names=feature_names)

        # Evaluate
        y_tree_pred = tree.predict(X_for_tree)
        r2 = tree.score(X_for_tree, y_pred)
        corr = np.corrcoef(y_pred, y_tree_pred)[0, 1]
        mse = np.mean((y_pred - y_tree_pred) ** 2)

        print(f"\nDecision Tree Rules:")
        print(tree_rules)
        print(f"\nPerformance:")
        print(f"  R² Score: {r2:.4f}")
        print(f"  Correlation: {corr:.4f}")
        print(f"  MSE: {mse:.6f}")

        results = {
            'formula': tree_rules,
            'readable_formula': tree_rules,
            'r2_score': r2,
            'correlation_with_nn': corr,
            'mse': mse,
            'feature_names': feature_names,
            'poly_transformer': None  # No transformer needed for tree
        }

        return tree, results
    
    def compare_predictions(self, symbolic_model=None, poly_transformer=None):
        """Compare predictions between neural network and extracted formula

        If `poly_transformer` is provided, it will be applied to the SHAP-selected
        features before passing them to the model (useful for polynomial models).
        """
        if symbolic_model is None:
            print("No symbolic model provided. Run extract_formula_symbolic_regression first.")
            return

        # Get predictions from both models
        X_tensor = torch.tensor(self.X_sample, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            nn_pred = self.model(X_tensor).cpu().numpy().flatten()

        # Use SHAP-selected features if available
        if hasattr(self, 'shap_values'):
            feature_importance = np.mean(np.abs(self.shap_values.values), axis=0)
            top_features_idx = np.argsort(feature_importance)[-5:]
            X_for_formula = self.X_sample[:, top_features_idx]
        else:
            X_for_formula = self.X_sample

        # Apply polynomial transformation if this is a polynomial model
        if poly_transformer is not None:
            try:
                X_for_formula = poly_transformer.transform(X_for_formula)
            except Exception as e:
                print(f"Warning: poly_transformer failed to transform features: {e}")

        formula_pred = symbolic_model.predict(X_for_formula)
        
        # Create comparison plot
        plt.figure(figsize=(12, 8))
        
        plt.subplot(2, 2, 1)
        plt.scatter(nn_pred, formula_pred, alpha=0.6)
        plt.plot([0, 1], [0, 1], 'r--', lw=2)
        plt.xlabel('Neural Network Prediction')
        plt.ylabel('Formula Prediction')
        plt.title('NN vs Formula Predictions')
        correlation = np.corrcoef(nn_pred, formula_pred)[0, 1]
        plt.text(0.05, 0.95, f'Correlation: {correlation:.3f}', transform=plt.gca().transAxes)
        
        plt.subplot(2, 2, 2)
        residuals = nn_pred - formula_pred
        plt.hist(residuals, bins=30, alpha=0.7)
        plt.xlabel('Residuals (NN - Formula)')
        plt.ylabel('Frequency')
        plt.title('Residual Distribution')
        plt.text(0.05, 0.95, f'Mean: {np.mean(residuals):.3f}\nStd: {np.std(residuals):.3f}', 
                transform=plt.gca().transAxes)
        
        plt.subplot(2, 2, 3)
        plt.scatter(self.y_sample, nn_pred, alpha=0.6, label='Neural Network')
        plt.scatter(self.y_sample, formula_pred, alpha=0.6, label='Formula')
        plt.plot([0, 1], [0, 1], 'k--', lw=1)
        plt.xlabel('True Labels')
        plt.ylabel('Predictions')
        plt.title('Predictions vs True Labels')
        plt.legend()
        
        plt.subplot(2, 2, 4)
        plt.plot(nn_pred, label='Neural Network', alpha=0.7)
        plt.plot(formula_pred, label='Formula', alpha=0.7)
        plt.xlabel('Sample Index')
        plt.ylabel('Prediction')
        plt.title('Prediction Comparison')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(f"{self.model_save_path}/prediction_comparison.png", dpi=300, bbox_inches='tight')
        plt.show()
        
        return correlation, residuals

if __name__ == "__main__":
    extractor = FormulaExtractor()
    extractor.load_best_model()
    shap_values = extractor.analyze_with_shap()
    symbolic_model, formula_results = extractor.extract_formula_symbolic_regression()
    correlation, residuals = extractor.compare_predictions(symbolic_model)