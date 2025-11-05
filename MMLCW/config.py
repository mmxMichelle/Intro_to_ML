"""
Configuration file for MLP training and formula extraction.
Modify these parameters to customize the behavior.
"""

# ============ TRAINING CONFIGURATION ============
TRAINING_CONFIG = {
    # Model architecture (SimpleMLP parameters)
    'hidden_layers': [128, 64],  # Hidden layer sizes
    'dropout_rate': 0.3,
    'activation': 'relu',
    
    # Training parameters
    'n_epochs': 70,
    'batch_size': 64,
    'learning_rate': 0.001,
    'weight_decay': 1e-4,
    'optimizer': 'adam',
    
    # Cross-validation
    'n_folds': 10,
    'cv_random_state': 42,
    'shuffle': True,
    
    # Saving options
    'save_all_folds': True,
    'save_best_only': False
}

# ============ SHAP ANALYSIS CONFIGURATION ============
SHAP_CONFIG = {
    # Sampling for SHAP (SHAP can be slow on large datasets)
    'n_samples': 1000,
    'random_state': 42,
    
    # Plotting options
    'create_plots': True,
    'plot_dpi': 300,
    'figsize': (15, 10),
    
    # Feature selection for symbolic regression
    'top_features_for_symreg': 10  # Number of top SHAP features to use
}

# ============ SYMBOLIC REGRESSION CONFIGURATION ============
SYMBOLIC_CONFIG = {
    # Basic GP parameters
    'population_size': 1000,
    'generations': 20,
    'tournament_size': 20,
    'random_state': 42,
    
    # Stopping criteria
    'stopping_criteria': 0.01,  # Stop if this R² is reached
    'parsimony_coefficient': 0.01,  # Penalize complex formulas
    
    # Genetic operations
    'p_crossover': 0.7,
    'p_subtree_mutation': 0.1,
    'p_hoist_mutation': 0.05,
    'p_point_mutation': 0.1,
    
    # Sampling
    'max_samples': 0.9,  # Fraction of samples to use per generation
    
    # Function set for symbolic regression
    'function_set': ['add', 'sub', 'mul', 'pdiv', 'psqrt', 'plog', 'sin', 'cos'],
    'use_custom_functions': True,
    
    # Multi-run configuration
    'multiple_runs': [
        {"population_size": 500, "generations": 15, "name": "Quick"},
        {"population_size": 1000, "generations": 25, "name": "Standard"},
        {"population_size": 2000, "generations": 30, "name": "Thorough"}
    ]
}

# ============ DATA CONFIGURATION ============
DATA_CONFIG = {
    'data_path': 'data',
    'X_filename': 'kryptonite-10-X.npy',
    'y_filename': 'kryptonite-10-y.npy',
    'model_save_path': 'saved_models',
    
    # Data preprocessing
    'standardize': True,
    'random_state': 42
}

# ============ ANALYSIS OPTIONS ============
ANALYSIS_CONFIG = {
    # Which analyses to run
    'run_shap': True,
    'run_symbolic_regression': True,
    'compare_predictions': True,
    
    # SHAP weighting
    'use_shap_weights': True,  # Use SHAP feature importance for symbolic regression
    
    # Verbose output
    'verbose': True,
    'save_intermediate_results': True
}

# ============ ADVANCED SYMBOLIC REGRESSION ============
# If you want to try different symbolic regression approaches
ADVANCED_SYMBOLIC_CONFIG = {
    # Alternative symbolic regression methods
    'try_pysr': False,  # Set to True if you have PySR installed
    'pysr_config': {
        'niterations': 40,
        'binary_operators': ['+', '-', '*', '/', '^'],
        'unary_operators': ['sin', 'cos', 'exp', 'log'],
        'populations': 30
    },
    
    # Ensemble approaches
    'ensemble_symbolic': False,  # Try multiple symbolic regressors
    'ensemble_size': 5,
    
    # Feature engineering before symbolic regression
    'feature_engineering': {
        'polynomial_features': False,
        'interaction_features': False,
        'max_polynomial_degree': 2
    }
}

# ============ FORMULA INTERPRETATION ============
INTERPRETATION_CONFIG = {
    # Formula simplification
    'simplify_formula': True,
    'max_formula_length': 100,  # Characters
    
    # Validation
    'validate_on_holdout': True,
    'holdout_fraction': 0.2,
    
    # Export options
    'export_formula_latex': True,
    'export_formula_python': True,
    'export_formula_readable': True
}

# Helper function to get configuration
def get_config():
    """Return all configuration as a single dictionary"""
    return {
        'training': TRAINING_CONFIG,
        'shap': SHAP_CONFIG,
        'symbolic': SYMBOLIC_CONFIG,
        'data': DATA_CONFIG,
        'analysis': ANALYSIS_CONFIG,
        'advanced_symbolic': ADVANCED_SYMBOLIC_CONFIG,
        'interpretation': INTERPRETATION_CONFIG
    }

# Helper function to update configuration
def update_config(section, key, value):
    """Update a specific configuration value"""
    configs = {
        'training': TRAINING_CONFIG,
        'shap': SHAP_CONFIG,
        'symbolic': SYMBOLIC_CONFIG,
        'data': DATA_CONFIG,
        'analysis': ANALYSIS_CONFIG,
        'advanced_symbolic': ADVANCED_SYMBOLIC_CONFIG,
        'interpretation': INTERPRETATION_CONFIG
    }
    
    if section in configs and key in configs[section]:
        configs[section][key] = value
        print(f"Updated {section}.{key} = {value}")
    else:
        print(f"Warning: {section}.{key} not found in configuration")
