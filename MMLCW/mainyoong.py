#!/usr/bin/env python3
"""
Main script to orchestrate MLP training and formula extraction.

Usage:
    python main.py train          # Run training only
    python main.py analyze        # Run analysis only (requires trained models)
    python main.py all            # Run both training and analysis
    python main.py formula        # Run only formula extraction (requires SHAP analysis)
"""

import sys
import argparse
import os
from pathlib import Path

def run_training():
    """Run the MLP training pipeline"""
    print("=" * 50)
    print("Starting MLP Training Pipeline")
    print("=" * 50)
    
    from mlp_trainer import MLPTrainer
    
    trainer = MLPTrainer()
    
    # Check if data exists
    if not os.path.exists("data/kryptonite-10-X.npy") or not os.path.exists("data/kryptonite-10-y.npy"):
        print("ERROR: Data files not found!")
        print("Please ensure 'data/kryptonite-10-X.npy' and 'data/kryptonite-10-y.npy' exist.")
        return False
    
    try:
        models, summary = trainer.train_cv_models()
        print("\n✅ Training completed successfully!")
        print(f"Models saved to {trainer.model_save_path}/")
        return True
    except Exception as e:
        print(f"❌ Training failed: {str(e)}")
        return False

def run_analysis():
    """Run the full analysis pipeline (SHAP + Symbolic Regression)"""
    print("=" * 50)
    print("Starting Formula Extraction Pipeline")
    print("=" * 50)
    
    from formula_extractor import FormulaExtractor
    
    # Check if trained models exist
    if not os.path.exists("saved_models/best_model.pt"):
        print("❌ No trained models found!")
        print("Please run training first: python main.py train")
        return False
    
    try:
        extractor = FormulaExtractor()

        # Load model
        print("Loading best trained model...")
        model_data = extractor.load_best_model()

        # ADD THIS NEW LINE:
        y_pred = extractor.diagnose_predictions()

        # SHAP Analysis
        print("\nRunning SHAP analysis...")
        shap_values = extractor.analyze_with_shap(n_samples=1000, create_plots=True)

        # Try multiple methods
        print("\nTrying formula extraction methods...")

        best_model = None
        best_results = None
        best_method = None
        best_score = -999

        # Method 1: Decision Tree
        try:
            print("\n--- Method 1: Decision Tree ---")
            tree_model, tree_results = extractor.extract_formula_decision_tree(max_depth=5)
            if tree_results['r2_score'] > best_score:
                best_model = tree_model
                best_results = tree_results
                best_method = "Decision Tree"
                best_score = tree_results['r2_score']
        except Exception as e:
            print(f"Decision Tree failed: {e}")

        # Method 2: Polynomial
        try:
            print("\n--- Method 2: Polynomial ---")
            poly_model, poly_results = extractor.extract_formula_polynomial(degree=2)
            if poly_results['r2_score'] > best_score:
                best_model = poly_model
                best_results = poly_results
                best_method = "Polynomial"
                best_score = poly_results['r2_score']
        except Exception as e:
            print(f"Polynomial failed: {e}")

        # Method 3: Symbolic Regression (if you have time)
        try:
            print("\n--- Method 3: Symbolic Regression ---")
            symbolic_model, symbolic_results = extractor.extract_formula_symbolic_regression(
                use_shap_weights=True,
                population_size=5000,
                generations=50
            )
            if symbolic_results['r2_score'] > best_score:
                best_model = symbolic_model
                best_results = symbolic_results
                best_method = "Symbolic Regression"
                best_score = symbolic_results['r2_score']
        except Exception as e:
            print(f"Symbolic Regression failed: {e}")

        # Use the best method
        if best_method is None:
            print("\n❌ All extraction methods failed.")
            return False

        print(f"\n🏆 BEST METHOD: {best_method}")
        print(f"   R² Score: {best_results['r2_score']:.4f}")
        print(f"   Correlation: {best_results['correlation_with_nn']:.4f}")

        symbolic_model = best_model
        formula_results = best_results

        # Compare predictions
        print("\nComparing predictions...")
        poly_transformer = formula_results.get('poly_transformer', None)  # Get transformer if it exists
        correlation, residuals = extractor.compare_predictions(symbolic_model, poly_transformer)

        print("\n✅ Analysis completed successfully!")
        print(f"Results saved to saved_models/")
        print(f"\n🔍 DISCOVERED FORMULA:")
        print(f"📊 {formula_results['readable_formula']}")
        print(f"📈 R² Score: {formula_results['r2_score']:.4f}")
        print(f"🔗 Correlation with NN: {formula_results['correlation_with_nn']:.4f}")

        return True
        
    except Exception as e:
        print(f"❌ Analysis failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def run_formula_only():
    """Run only the symbolic regression part (assumes SHAP analysis is done)"""
    print("=" * 50)
    print("Running Symbolic Regression Only")
    print("=" * 50)
    
    from formula_extractor import FormulaExtractor
    
    if not os.path.exists("saved_models/best_model.pt"):
        print("❌ No trained models found!")
        return False
    
    try:
        extractor = FormulaExtractor()
        extractor.load_best_model()
        
        # Quick SHAP analysis
        extractor.analyze_with_shap(n_samples=500, create_plots=False)
        
        # Symbolic regression with different parameters
        print("Trying different symbolic regression configurations...")
        
        configs = [
            {"population_size": 500, "generations": 15, "name": "Quick"},
            {"population_size": 1000, "generations": 25, "name": "Standard"},
            {"population_size": 2000, "generations": 30, "name": "Thorough"}
        ]
        
        best_formula = None
        best_score = -1
        
        for config in configs:
            print(f"\n--- {config['name']} Configuration ---")
            symbolic_model, formula_results = extractor.extract_formula_symbolic_regression(
                population_size=config["population_size"],
                generations=config["generations"]
            )
            
            if formula_results['correlation_with_nn'] > best_score:
                best_score = formula_results['correlation_with_nn']
                best_formula = formula_results
        
        print(f"\n🏆 BEST FORMULA FOUND:")
        print(f"📊 {best_formula['readable_formula']}")
        print(f"📈 R² Score: {best_formula['r2_score']:.4f}")
        print(f"🔗 Correlation with NN: {best_formula['correlation_with_nn']:.4f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Formula extraction failed: {str(e)}")
        return False

def main():
    parser = argparse.ArgumentParser(description='MLP Training and Formula Extraction Pipeline')
    parser.add_argument('mode', choices=['train', 'analyze', 'all', 'formula'], 
                       help='Mode to run: train, analyze, all, or formula')
    parser.add_argument('--data-path', default='data', 
                       help='Path to data directory (default: data)')
    parser.add_argument('--model-path', default='saved_models', 
                       help='Path to save/load models (default: saved_models)')
    
    # If no arguments provided, show help
    if len(sys.argv) == 1:
        parser.print_help()
        print("\nExamples:")
        print("  python main.py train     # Train models")
        print("  python main.py analyze   # Run full analysis")
        print("  python main.py all       # Train + analyze")
        print("  python main.py formula   # Quick formula extraction")
        return
    
    args = parser.parse_args()
    
    # Set global paths (you could modify the classes to accept these)
    # For now, they use the default paths
    
    if args.mode == 'train':
        success = run_training()
    elif args.mode == 'analyze':
        success = run_analysis()
    elif args.mode == 'formula':
        success = run_formula_only()
    elif args.mode == 'all':
        print("Running complete pipeline...")
        train_success = run_training()
        if train_success:
            success = run_analysis()
        else:
            success = False
    
    if success:
        print(f"\n🎉 {args.mode.upper()} completed successfully!")
    else:
        print(f"\n💥 {args.mode.upper()} failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()
