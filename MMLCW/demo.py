#!/usr/bin/env python3
"""
Demo script showing step-by-step usage of the MLP formula extraction pipeline.
This script demonstrates how to use each component individually.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def create_sample_data():
    """Create sample data with a known formula for testing"""
    print("Creating sample data with known formula: y = sin(x1 * x2) + 0.5 * x3^2")
    
    # Create data directory
    Path("data").mkdir(exist_ok=True)
    
    # Generate synthetic data with known formula
    n_samples = 1000
    n_features = 10
    
    np.random.seed(42)
    X = np.random.uniform(-2, 2, (n_samples, n_features))
    
    # Known formula: y = sin(x1 * x2) + 0.5 * x3^2 + noise
    # Add some noise and make it binary classification
    y_continuous = np.sin(X[:, 0] * X[:, 1]) + 0.5 * X[:, 2]**2 + 0.1 * np.random.normal(0, 1, n_samples)
    y = (y_continuous > np.median(y_continuous)).astype(int)
    
    # Save data
    np.save("data/kryptonite-10-X.npy", X)
    np.save("data/kryptonite-10-y.npy", y)
    
    print(f"Created synthetic data: X.shape = {X.shape}, y.shape = {y.shape}")
    print(f"True formula: y = sin(x0 * x1) + 0.5 * x2^2")
    print(f"Class distribution: {np.bincount(y)}")
    
    return X, y

def demo_training():
    """Demonstrate the training process"""
    print("\n" + "="*60)
    print("DEMO: Training MLP Models")
    print("="*60)
    
    from mlp_trainer import MLPTrainer
    
    trainer = MLPTrainer()
    
    # Load or create data
    try:
        X, y = trainer.load_data()
    except FileNotFoundError:
        print("Data not found, creating sample data...")
        X, y = create_sample_data()
        X, y = trainer.load_data()
    
    # Train with fewer epochs for demo
    print("Training models (reduced epochs for demo)...")
    models, summary = trainer.train_cv_models(n_epochs=20)  # Reduced for demo
    
    print("✅ Training completed!")
    return models, summary

def demo_shap_analysis():
    """Demonstrate SHAP analysis"""
    print("\n" + "="*60)
    print("DEMO: SHAP Analysis")
    print("="*60)
    
    from formula_extractor import FormulaExtractor
    
    extractor = FormulaExtractor()
    
    # Load the trained model
    model_data = extractor.load_best_model()
    
    # Run SHAP analysis with fewer samples for demo
    print("Running SHAP analysis...")
    shap_values = extractor.analyze_with_shap(n_samples=200, create_plots=True)
    
    # Show feature importance
    feature_importance = np.mean(np.abs(shap_values.values), axis=0)
    top_features = np.argsort(feature_importance)[-5:]
    
    print(f"\nTop 5 most important features (by SHAP):")
    for i, feat_idx in enumerate(reversed(top_features)):
        print(f"  {i+1}. Feature {feat_idx}: importance = {feature_importance[feat_idx]:.4f}")
    
    print("✅ SHAP analysis completed!")
    return extractor

def demo_formula_extraction(extractor):
    """Demonstrate formula extraction"""
    print("\n" + "="*60)
    print("DEMO: Formula Extraction")
    print("="*60)
    
    # Run symbolic regression with smaller parameters for demo
    print("Running symbolic regression (reduced parameters for demo)...")
    symbolic_model, formula_results = extractor.extract_formula_symbolic_regression(
        use_shap_weights=True,
        population_size=200,  # Reduced for demo
        generations=10        # Reduced for demo
    )
    
    print(f"\n🎯 EXTRACTED FORMULA:")
    print(f"   Formula: {formula_results['readable_formula']}")
    print(f"   R² Score: {formula_results['r2_score']:.4f}")
    print(f"   Correlation with NN: {formula_results['correlation_with_nn']:.4f}")
    
    # Compare with known formula
    print(f"\n📚 COMPARISON WITH TRUE FORMULA:")
    print(f"   True formula: y = sin(x0 * x1) + 0.5 * x2^2")
    print(f"   Extracted: {formula_results['readable_formula']}")
    
    # Compare predictions
    correlation, residuals = extractor.compare_predictions(symbolic_model)
    
    print(f"\n📊 PREDICTION COMPARISON:")
    print(f"   Correlation: {correlation:.4f}")
    print(f"   Residual mean: {np.mean(residuals):.4f}")
    print(f"   Residual std: {np.std(residuals):.4f}")
    
    print("✅ Formula extraction completed!")
    return symbolic_model, formula_results

def demo_interpretation(formula_results):
    """Demonstrate formula interpretation"""
    print("\n" + "="*60)
    print("DEMO: Formula Interpretation")
    print("="*60)
    
    formula = formula_results['readable_formula']
    
    print(f"📝 FORMULA ANALYSIS:")
    print(f"   Raw formula: {formula}")
    
    # Simple analysis of the formula structure
    components = []
    if 'sin' in formula or 'cos' in formula:
        components.append("trigonometric functions")
    if '*' in formula:
        components.append("multiplicative interactions")
    if '+' in formula or '-' in formula:
        components.append("additive terms")
    if '/' in formula:
        components.append("divisive relationships")
    if 'sqrt' in formula or 'psqrt' in formula:
        components.append("square root transformations")
    if 'log' in formula or 'plog' in formula:
        components.append("logarithmic transformations")
    
    print(f"   Components detected: {', '.join(components)}")
    
    # Feature usage
    features_used = []
    for i in range(10):  # Assuming max 10 features
        if f'x{i}' in formula:
            features_used.append(i)
    
    print(f"   Features used: {features_used}")
    print(f"   Formula complexity: {'Simple' if len(formula) < 50 else 'Complex'}")
    
    # Recommendations
    print(f"\n💡 RECOMMENDATIONS:")
    if formula_results['r2_score'] > 0.8:
        print("   ✅ High R² score - formula captures neural network well")
    else:
        print("   ⚠️  Low R² score - consider more complex function set or longer evolution")
    
    if formula_results['correlation_with_nn'] > 0.9:
        print("   ✅ High correlation - formula predictions align well with neural network")
    else:
        print("   ⚠️  Low correlation - formula may need refinement")
    
    print("✅ Interpretation completed!")

def run_complete_demo():
    """Run the complete demo pipeline"""
    print("🚀 Starting Complete MLP Formula Extraction Demo")
    print("This demo will show you how to extract formulas from neural networks!\n")
    
    # Step 1: Training
    models, summary = demo_training()
    
    # Step 2: SHAP Analysis
    extractor = demo_shap_analysis()
    
    # Step 3: Formula Extraction
    symbolic_model, formula_results = demo_formula_extraction(extractor)
    
    # Step 4: Interpretation
    demo_interpretation(formula_results)
    
    print("\n" + "="*60)
    print("🎉 DEMO COMPLETED SUCCESSFULLY!")
    print("="*60)
    print(f"📁 Check the 'saved_models/' directory for:")
    print(f"   - Trained neural network models")
    print(f"   - SHAP analysis plots")
    print(f"   - Formula extraction results")
    print(f"   - Prediction comparison plots")
    
    print(f"\n🔮 FINAL DISCOVERED FORMULA:")
    print(f"   {formula_results['readable_formula']}")
    print(f"   (R² = {formula_results['r2_score']:.3f}, Correlation = {formula_results['correlation_with_nn']:.3f})")
    
    print(f"\n💻 To use the full system:")
    print(f"   python main.py all        # Complete pipeline")
    print(f"   python main.py train      # Training only")
    print(f"   python main.py analyze    # Analysis only")

if __name__ == "__main__":
    run_complete_demo()
