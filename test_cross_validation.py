"""
Test script to verify cross-validation implementation in ModelComparison.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.data_preprocessing import FeatureExtractor, create_train_test_split
from src.news_classifier import ModelComparison
from src.data_generator import create_sample_data
import pandas as pd

def test_cross_validation():
    """Test cross-validation support in ModelComparison."""
    
    print("=" * 80)
    print("Testing Cross-Validation Support in ModelComparison")
    print("=" * 80)
    
    # Generate sample data if needed
    data_path = "data"
    articles_path = os.path.join(data_path, "news_articles.csv")
    
    if not os.path.exists(articles_path):
        print("\nGenerating sample data...")
        create_sample_data(data_path)
    
    # Load data
    print("\nLoading data...")
    articles_df = pd.read_csv(articles_path)
    print(f"Loaded {len(articles_df)} articles")
    
    # Prepare features
    print("\nExtracting features...")
    feature_extractor = FeatureExtractor(max_features=1000)
    X = feature_extractor.fit_transform_text(articles_df['content'].tolist())
    y = feature_extractor.fit_transform_labels(articles_df['category'].tolist())
    
    # Create train-test split
    X_train, X_test, y_train, y_test = create_train_test_split(X, y, test_size=0.2)
    print(f"Train samples: {len(X_train)}, Test samples: {len(X_test)}")
    
    # Test 1: Model comparison WITHOUT cross-validation
    print("\n" + "=" * 80)
    print("TEST 1: Model Comparison WITHOUT Cross-Validation")
    print("=" * 80)
    
    comparison1 = ModelComparison(
        model_types=['logistic_regression', 'naive_bayes'],
        use_cross_validation=False
    )
    
    results1, best_model1 = comparison1.compare_models(X_train, y_train, X_test, y_test)
    
    print("\nResults DataFrame:")
    print(comparison1.get_results_dataframe())
    print(f"\nBest Model: {best_model1}")
    
    # Test 2: Model comparison WITH cross-validation
    print("\n" + "=" * 80)
    print("TEST 2: Model Comparison WITH Cross-Validation (k=5)")
    print("=" * 80)
    
    comparison2 = ModelComparison(
        model_types=['logistic_regression', 'naive_bayes'],
        use_cross_validation=True,
        cv_folds=5
    )
    
    results2, best_model2 = comparison2.compare_models(X_train, y_train, X_test, y_test)
    
    print("\nResults DataFrame:")
    print(comparison2.get_results_dataframe())
    print(f"\nBest Model: {best_model2}")
    
    # Test 3: Visualize cross-validation results
    print("\n" + "=" * 80)
    print("TEST 3: Cross-Validation Visualization")
    print("=" * 80)
    
    print(comparison2.visualize_cv_results())
    
    # Test 4: Statistical comparison
    print("\n" + "=" * 80)
    print("TEST 4: Statistical Comparison")
    print("=" * 80)
    
    stat_comparison = comparison2.get_statistical_comparison()
    if stat_comparison:
        print("\nStatistical significance testing results:")
        for comp_name, comp_data in stat_comparison.items():
            print(f"\n{comp_name}:")
            print(f"  {comp_data['model1']}: {comp_data['model1_mean']:.4f}")
            print(f"  {comp_data['model2']}: {comp_data['model2_mean']:.4f}")
            print(f"  t-statistic: {comp_data['t_statistic']:.4f}")
            print(f"  p-value: {comp_data['p_value']:.4f}")
            print(f"  Statistically significant: {comp_data['significant']}")
            if comp_data['significant']:
                print(f"  Better model: {comp_data['better_model']}")
    
    print("\n" + "=" * 80)
    print("All Tests Completed Successfully! ✓")
    print("=" * 80)

if __name__ == "__main__":
    test_cross_validation()
