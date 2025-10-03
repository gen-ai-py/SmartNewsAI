"""
Demo Script: Data Quality Validation System

This script demonstrates the comprehensive data quality validation
system for news article datasets.

Features demonstrated:
1. Dataset validation with quality scoring
2. Missing value detection
3. Duplicate identification
4. Text quality analysis
5. Label distribution analysis
6. Outlier detection
7. Auto-cleaning functionality
8. Report generation (text, markdown, JSON)
"""

import sys
import os
import pandas as pd
import numpy as np

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.data_quality_validator import DataQualityValidator


def create_sample_dataset():
    """Create a sample dataset with various quality issues for demonstration."""
    print("\n📊 Creating sample dataset with intentional quality issues...")
    
    # Good quality articles
    good_articles = [
        {
            'content': 'Breaking news in technology: AI advances rapidly in healthcare sector with new applications.',
            'category': 'technology'
        },
        {
            'content': 'Sports update: Local team wins championship in thrilling match against rivals yesterday.',
            'category': 'sports'
        },
        {
            'content': 'Business report: Stock markets show strong performance with tech sector leading gains.',
            'category': 'business'
        }
    ] * 30  # 90 good articles
    
    # Articles with quality issues
    problem_articles = [
        {'content': 'Short', 'category': 'technology'},  # Too short
        {'content': None, 'category': 'sports'},  # Missing content
        {'content': 'Good article here', 'category': None},  # Missing label
        {'content': 'A' * 15000, 'category': 'business'},  # Too long
        {'content': 'Breaking news in technology: AI advances rapidly in healthcare sector with new applications.', 
         'category': 'technology'},  # Duplicate
    ]
    
    # Combine all articles
    all_articles = good_articles + problem_articles
    
    df = pd.DataFrame(all_articles)
    
    print(f"✅ Created dataset with {len(df)} articles")
    print(f"   - {len(good_articles)} good quality articles")
    print(f"   - {len(problem_articles)} articles with issues")
    
    return df


def demo_basic_validation():
    """Demonstrate basic validation functionality."""
    print("\n" + "="*70)
    print("  DEMO 1: BASIC DATA QUALITY VALIDATION")
    print("="*70)
    
    # Create sample dataset
    df = create_sample_dataset()
    
    # Initialize validator
    validator = DataQualityValidator(min_text_length=50, max_text_length=10000)
    
    print("\n🔍 Running comprehensive validation...")
    
    # Run validation
    results = validator.validate_dataset(df, text_column='content', label_column='category')
    
    print(f"\n✅ Validation Complete!")
    print(f"   Overall Quality Score: {results['quality_score']:.1f}/100")
    print(f"   Issues Found: {len(validator.issues)}")
    print(f"   Warnings: {len(validator.warnings)}")
    print(f"   Recommendations: {len(validator.recommendations)}")
    
    return validator, df


def demo_detailed_report(validator):
    """Demonstrate detailed report generation."""
    print("\n" + "="*70)
    print("  DEMO 2: DETAILED VALIDATION REPORT")
    print("="*70)
    
    print("\n📋 Generating text report...\n")
    
    # Generate and display text report
    text_report = validator.generate_report(format='text')
    print(text_report)
    
    # Save reports
    print("\n💾 Saving reports to files...")
    
    os.makedirs('validation_reports', exist_ok=True)
    
    validator.save_report('validation_reports/quality_report.txt', format='text')
    print("   ✅ Text report: validation_reports/quality_report.txt")
    
    validator.save_report('validation_reports/quality_report.md', format='markdown')
    print("   ✅ Markdown report: validation_reports/quality_report.md")
    
    validator.save_report('validation_reports/quality_report.json', format='json')
    print("   ✅ JSON report: validation_reports/quality_report.json")


def demo_specific_checks(validator):
    """Demonstrate specific validation checks."""
    print("\n" + "="*70)
    print("  DEMO 3: SPECIFIC VALIDATION CHECKS")
    print("="*70)
    
    results = validator.validation_results
    
    print("\n1️⃣ Missing Values Analysis:")
    mv = results['missing_values']
    print(f"   • Missing text: {mv['missing_text']}")
    print(f"   • Missing labels: {mv['missing_labels']}")
    print(f"   • Empty text: {mv['empty_text']}")
    print(f"   • Total: {mv['total_missing']} ({mv['missing_percentage']:.1f}%)")
    
    print("\n2️⃣ Duplicates Analysis:")
    dup = results['duplicates']
    print(f"   • Duplicate articles: {dup['count']} ({dup['percentage']:.1f}%)")
    
    print("\n3️⃣ Text Quality Metrics:")
    tq = results['text_quality']
    print(f"   • Average length: {tq['avg_length']:.0f} characters")
    print(f"   • Average word count: {tq['avg_word_count']:.0f} words")
    print(f"   • Too short: {tq['too_short_count']} articles")
    print(f"   • Too long: {tq['too_long_count']} articles")
    
    print("\n4️⃣ Label Distribution:")
    ld = results['label_distribution']
    print(f"   • Unique labels: {ld['unique_labels']}")
    print(f"   • Imbalance ratio: {ld['imbalance_ratio']:.2f}:1")
    print(f"   • Distribution:")
    for label, info in ld['distribution'].items():
        print(f"     - {label}: {info['count']} ({info['percentage']:.1f}%)")
    
    print("\n5️⃣ Outliers Detection:")
    out = results['outliers']
    print(f"   • Total outliers: {out['total_count']} ({out['percentage']:.1f}%)")
    print(f"   • Too short: {out['too_short']}")
    print(f"   • Too long: {out['too_long']}")
    print(f"   • IQR bounds: [{out['bounds']['lower']:.0f}, {out['bounds']['upper']:.0f}]")


def demo_auto_cleaning(validator, df):
    """Demonstrate automatic data cleaning."""
    print("\n" + "="*70)
    print("  DEMO 4: AUTOMATIC DATA CLEANING")
    print("="*70)
    
    print(f"\n📊 Original dataset: {len(df)} articles")
    
    # Get clean indices
    clean_indices = validator.get_clean_indices(df)
    print(f"   Clean articles (without issues): {len(clean_indices)}")
    
    print("\n🧹 Running auto-clean...")
    
    # Auto-clean the dataset
    df_clean, cleaning_summary = validator.auto_clean(df)
    
    print(f"\n✅ Cleaning Complete!")
    print(f"   Original size: {cleaning_summary['original_size']}")
    print(f"   Final size: {cleaning_summary['final_size']}")
    print(f"   Retention rate: {cleaning_summary['retention_rate']:.1f}%")
    
    print(f"\n📉 Removed:")
    for category, count in cleaning_summary['removed'].items():
        print(f"   • {category}: {count}")
    
    print(f"\n   Total removed: {cleaning_summary['total_removed']}")
    
    # Validate cleaned dataset
    print("\n🔍 Validating cleaned dataset...")
    validator_clean = DataQualityValidator()
    results_clean = validator_clean.validate_dataset(df_clean)
    
    print(f"\n✅ Cleaned Dataset Quality Score: {results_clean['quality_score']:.1f}/100")
    print(f"   (Improved from {validator.quality_score:.1f}/100)")
    
    return df_clean


def demo_with_real_data():
    """Demonstrate with actual news data if available."""
    print("\n" + "="*70)
    print("  DEMO 5: VALIDATION WITH ACTUAL DATA")
    print("="*70)
    
    data_path = "data/news_articles.csv"
    
    if os.path.exists(data_path):
        print(f"\n📂 Loading data from {data_path}...")
        df = pd.read_csv(data_path)
        
        print(f"   Loaded {len(df)} articles")
        
        # Run validation
        validator = DataQualityValidator()
        results = validator.validate_dataset(df)
        
        print(f"\n📊 Validation Results:")
        print(f"   Quality Score: {results['quality_score']:.1f}/100")
        
        if validator.issues:
            print(f"\n❌ Critical Issues ({len(validator.issues)}):")
            for issue in validator.issues:
                print(f"   • {issue}")
        
        if validator.warnings:
            print(f"\n⚠️  Warnings ({len(validator.warnings)}):")
            for warning in validator.warnings[:5]:  # Show first 5
                print(f"   • {warning}")
            if len(validator.warnings) > 5:
                print(f"   ... and {len(validator.warnings) - 5} more")
        
        if validator.recommendations:
            print(f"\n💡 Recommendations ({len(validator.recommendations)}):")
            for rec in validator.recommendations[:5]:  # Show first 5
                print(f"   • {rec}")
            if len(validator.recommendations) > 5:
                print(f"   ... and {len(validator.recommendations) - 5} more")
    else:
        print(f"\n📁 No data file found at {data_path}")
        print("   Skipping this demo. Run --setup-data first to generate data.")


def demo_comparison():
    """Demonstrate before/after comparison."""
    print("\n" + "="*70)
    print("  DEMO 6: BEFORE/AFTER COMPARISON")
    print("="*70)
    
    # Create dataset with issues
    df = create_sample_dataset()
    
    # Validate original
    validator_before = DataQualityValidator()
    results_before = validator_before.validate_dataset(df)
    
    # Clean dataset
    df_clean, _ = validator_before.auto_clean(df)
    
    # Validate cleaned
    validator_after = DataQualityValidator()
    results_after = validator_after.validate_dataset(df_clean)
    
    print("\n📊 Comparison Summary:")
    print(f"\n{'Metric':<30} {'Before':<15} {'After':<15} {'Improvement':<15}")
    print("-" * 75)
    
    # Quality Score
    improvement = results_after['quality_score'] - results_before['quality_score']
    print(f"{'Quality Score':<30} {results_before['quality_score']:<15.1f} "
          f"{results_after['quality_score']:<15.1f} {improvement:+.1f}")
    
    # Dataset Size
    print(f"{'Dataset Size':<30} {results_before['total_records']:<15} "
          f"{results_after['total_records']:<15} {results_after['total_records'] - results_before['total_records']:+d}")
    
    # Missing Values
    before_missing = results_before['missing_values']['total_missing']
    after_missing = results_after['missing_values']['total_missing']
    print(f"{'Missing Values':<30} {before_missing:<15} "
          f"{after_missing:<15} {after_missing - before_missing:+d}")
    
    # Duplicates
    before_dup = results_before['duplicates']['count']
    after_dup = results_after['duplicates']['count']
    print(f"{'Duplicates':<30} {before_dup:<15} "
          f"{after_dup:<15} {after_dup - before_dup:+d}")
    
    # Issues
    print(f"{'Critical Issues':<30} {len(validator_before.issues):<15} "
          f"{len(validator_after.issues):<15} {len(validator_after.issues) - len(validator_before.issues):+d}")
    
    print("\n✅ Data quality significantly improved after cleaning!")


def main():
    """Run all demos."""
    print("\n" + "="*70)
    print("       DATA QUALITY VALIDATION SYSTEM - DEMO")
    print("="*70)
    print("\nThis demo showcases the comprehensive data quality validation")
    print("system for news article datasets.\n")
    
    try:
        # Demo 1: Basic validation
        validator, df = demo_basic_validation()
        
        # Demo 2: Detailed reports
        demo_detailed_report(validator)
        
        # Demo 3: Specific checks
        demo_specific_checks(validator)
        
        # Demo 4: Auto-cleaning
        df_clean = demo_auto_cleaning(validator, df)
        
        # Demo 5: With real data (if available)
        demo_with_real_data()
        
        # Demo 6: Comparison
        demo_comparison()
        
        print("\n" + "="*70)
        print("                    DEMO COMPLETE!")
        print("="*70)
        print("\n📁 Generated Files:")
        print("   • validation_reports/quality_report.txt")
        print("   • validation_reports/quality_report.md")
        print("   • validation_reports/quality_report.json")
        
        print("\n💡 Usage Tips:")
        print("   1. Run validation before training ML models")
        print("   2. Use auto_clean() to quickly fix common issues")
        print("   3. Review reports to understand data quality problems")
        print("   4. Set min/max text length based on your use case")
        print("   5. Monitor quality score over time for data drift")
        
        print("\n" + "="*70)
        
    except Exception as e:
        print(f"\n❌ Error during demo: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
