"""
Data Quality Validation System for Smart News AI

This module provides comprehensive data quality assessment and validation
for news article datasets before training ML models.

Features:
- Text quality metrics (length, readability, diversity)
- Label distribution analysis
- Outlier detection
- Missing data identification
- Data balance assessment
- Quality score calculation
- Detailed validation reports
- Auto-fix recommendations
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from collections import Counter
import re
import logging
from datetime import datetime
import json

logger = logging.getLogger(__name__)


class DataQualityValidator:
    """
    Comprehensive data quality validator for news article datasets.
    
    Validates:
    - Text content quality
    - Label distribution and balance
    - Missing values and duplicates
    - Outliers and anomalies
    - Overall dataset health
    """
    
    def __init__(self, min_text_length: int = 50, max_text_length: int = 10000):
        """
        Initialize data quality validator.
        
        Args:
            min_text_length: Minimum acceptable article length (characters)
            max_text_length: Maximum acceptable article length (characters)
        """
        self.min_text_length = min_text_length
        self.max_text_length = max_text_length
        self.validation_results = {}
        self.quality_score = 0.0
        self.issues = []
        self.warnings = []
        self.recommendations = []
        
    def validate_dataset(self, df: pd.DataFrame, text_column: str = 'content', 
                        label_column: str = 'category') -> Dict[str, Any]:
        """
        Run complete validation suite on dataset.
        
        Args:
            df: DataFrame containing news articles
            text_column: Name of text content column
            label_column: Name of label column
            
        Returns:
            Dictionary with validation results
        """
        logger.info(f"Starting data quality validation on {len(df)} articles")
        
        self.validation_results = {
            'timestamp': datetime.now().isoformat(),
            'total_records': len(df),
            'columns_validated': [text_column, label_column]
        }
        
        # Run all validation checks
        self._check_missing_values(df, text_column, label_column)
        self._check_duplicates(df, text_column)
        self._check_text_quality(df, text_column)
        self._check_label_distribution(df, label_column)
        self._check_text_length_distribution(df, text_column)
        self._detect_outliers(df, text_column)
        self._check_data_balance(df, label_column)
        self._analyze_vocabulary_diversity(df, text_column)
        
        # Calculate overall quality score
        self.quality_score = self._calculate_quality_score()
        self.validation_results['quality_score'] = self.quality_score
        self.validation_results['issues'] = self.issues
        self.validation_results['warnings'] = self.warnings
        self.validation_results['recommendations'] = self.recommendations
        
        logger.info(f"Validation complete. Quality score: {self.quality_score:.2f}/100")
        
        return self.validation_results
    
    def _check_missing_values(self, df: pd.DataFrame, text_column: str, label_column: str):
        """Check for missing values in critical columns."""
        missing_text = df[text_column].isna().sum()
        missing_labels = df[label_column].isna().sum()
        empty_text = (df[text_column].str.strip() == '').sum()
        
        total_missing = missing_text + missing_labels + empty_text
        missing_percentage = (total_missing / (len(df) * 2)) * 100
        
        self.validation_results['missing_values'] = {
            'missing_text': int(missing_text),
            'missing_labels': int(missing_labels),
            'empty_text': int(empty_text),
            'total_missing': int(total_missing),
            'missing_percentage': round(missing_percentage, 2)
        }
        
        if total_missing > 0:
            severity = 'critical' if missing_percentage > 5 else 'warning'
            message = f"Found {total_missing} missing/empty values ({missing_percentage:.1f}%)"
            
            if severity == 'critical':
                self.issues.append(message)
            else:
                self.warnings.append(message)
                
            self.recommendations.append(
                f"Remove or impute {total_missing} rows with missing values"
            )
    
    def _check_duplicates(self, df: pd.DataFrame, text_column: str):
        """Check for duplicate articles."""
        duplicates = df[text_column].duplicated().sum()
        duplicate_percentage = (duplicates / len(df)) * 100
        
        self.validation_results['duplicates'] = {
            'count': int(duplicates),
            'percentage': round(duplicate_percentage, 2)
        }
        
        if duplicates > 0:
            message = f"Found {duplicates} duplicate articles ({duplicate_percentage:.1f}%)"
            self.warnings.append(message)
            self.recommendations.append(f"Remove {duplicates} duplicate entries")
    
    def _check_text_quality(self, df: pd.DataFrame, text_column: str):
        """Analyze text content quality."""
        texts = df[text_column].dropna()
        
        # Calculate various text metrics
        lengths = texts.str.len()
        word_counts = texts.str.split().str.len()
        
        # Check for very short or very long texts
        too_short = (lengths < self.min_text_length).sum()
        too_long = (lengths > self.max_text_length).sum()
        
        # Check for non-text content (URLs, special chars)
        url_heavy = texts.str.count(r'http[s]?://').sum()
        special_char_heavy = texts.str.count(r'[^a-zA-Z0-9\s]').sum()
        
        quality_issues = too_short + too_long
        
        self.validation_results['text_quality'] = {
            'avg_length': round(lengths.mean(), 2),
            'avg_word_count': round(word_counts.mean(), 2),
            'too_short_count': int(too_short),
            'too_long_count': int(too_long),
            'url_heavy_texts': int(url_heavy),
            'quality_issues': int(quality_issues)
        }
        
        if too_short > 0:
            self.warnings.append(
                f"{too_short} articles are too short (< {self.min_text_length} chars)"
            )
            self.recommendations.append(
                f"Review or remove {too_short} articles below minimum length"
            )
        
        if too_long > 0:
            self.warnings.append(
                f"{too_long} articles are too long (> {self.max_text_length} chars)"
            )
    
    def _check_label_distribution(self, df: pd.DataFrame, label_column: str):
        """Analyze label distribution and balance."""
        label_counts = df[label_column].value_counts()
        total_samples = len(df)
        
        distribution = {}
        for label, count in label_counts.items():
            percentage = (count / total_samples) * 100
            distribution[str(label)] = {
                'count': int(count),
                'percentage': round(percentage, 2)
            }
        
        # Calculate distribution metrics
        max_class = label_counts.max()
        min_class = label_counts.min()
        imbalance_ratio = max_class / min_class if min_class > 0 else float('inf')
        
        self.validation_results['label_distribution'] = {
            'unique_labels': int(len(label_counts)),
            'distribution': distribution,
            'max_class_size': int(max_class),
            'min_class_size': int(min_class),
            'imbalance_ratio': round(imbalance_ratio, 2)
        }
        
        # Check for imbalance
        if imbalance_ratio > 5:
            self.issues.append(
                f"Severe class imbalance detected (ratio: {imbalance_ratio:.1f}:1)"
            )
            self.recommendations.append(
                "Consider using class weights, SMOTE, or collecting more data for minority classes"
            )
        elif imbalance_ratio > 2:
            self.warnings.append(
                f"Moderate class imbalance (ratio: {imbalance_ratio:.1f}:1)"
            )
    
    def _check_text_length_distribution(self, df: pd.DataFrame, text_column: str):
        """Analyze text length distribution."""
        lengths = df[text_column].dropna().str.len()
        
        self.validation_results['length_distribution'] = {
            'min': int(lengths.min()),
            'max': int(lengths.max()),
            'mean': round(lengths.mean(), 2),
            'median': round(lengths.median(), 2),
            'std': round(lengths.std(), 2),
            'quartiles': {
                '25%': round(lengths.quantile(0.25), 2),
                '50%': round(lengths.quantile(0.50), 2),
                '75%': round(lengths.quantile(0.75), 2)
            }
        }
    
    def _detect_outliers(self, df: pd.DataFrame, text_column: str):
        """Detect statistical outliers in text length."""
        lengths = df[text_column].dropna().str.len()
        
        # Use IQR method for outlier detection
        Q1 = lengths.quantile(0.25)
        Q3 = lengths.quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers_low = (lengths < lower_bound).sum()
        outliers_high = (lengths > upper_bound).sum()
        total_outliers = outliers_low + outliers_high
        
        outlier_percentage = (total_outliers / len(df)) * 100
        
        self.validation_results['outliers'] = {
            'total_count': int(total_outliers),
            'percentage': round(outlier_percentage, 2),
            'too_short': int(outliers_low),
            'too_long': int(outliers_high),
            'bounds': {
                'lower': round(lower_bound, 2),
                'upper': round(upper_bound, 2)
            }
        }
        
        if outlier_percentage > 5:
            self.warnings.append(
                f"{total_outliers} outliers detected ({outlier_percentage:.1f}%)"
            )
            self.recommendations.append(
                "Review outlier articles for data quality issues"
            )
    
    def _check_data_balance(self, df: pd.DataFrame, label_column: str):
        """Check if dataset is balanced across categories."""
        label_counts = df[label_column].value_counts()
        total = len(df)
        
        # Calculate balance score (0-100, where 100 is perfectly balanced)
        expected_per_class = total / len(label_counts)
        deviations = [(count - expected_per_class) ** 2 for count in label_counts]
        balance_score = max(0, 100 - (np.sqrt(np.mean(deviations)) / expected_per_class * 100))
        
        self.validation_results['balance_score'] = round(balance_score, 2)
        
        if balance_score < 50:
            self.issues.append(f"Poor data balance (score: {balance_score:.1f}/100)")
        elif balance_score < 70:
            self.warnings.append(f"Moderate data balance (score: {balance_score:.1f}/100)")
    
    def _analyze_vocabulary_diversity(self, df: pd.DataFrame, text_column: str):
        """Analyze vocabulary diversity across the dataset."""
        texts = df[text_column].dropna()
        
        # Sample texts for efficiency (max 1000)
        sample_size = min(1000, len(texts))
        sample_texts = texts.sample(n=sample_size, random_state=42)
        
        # Extract unique words
        all_words = []
        for text in sample_texts:
            words = re.findall(r'\b\w+\b', text.lower())
            all_words.extend(words)
        
        total_words = len(all_words)
        unique_words = len(set(all_words))
        
        # Calculate vocabulary diversity ratio
        diversity_ratio = unique_words / total_words if total_words > 0 else 0
        
        self.validation_results['vocabulary'] = {
            'total_words_sampled': total_words,
            'unique_words': unique_words,
            'diversity_ratio': round(diversity_ratio, 4),
            'avg_word_length': round(np.mean([len(w) for w in all_words]) if all_words else 0, 2)
        }
        
        if diversity_ratio < 0.1:
            self.warnings.append(
                f"Low vocabulary diversity (ratio: {diversity_ratio:.2%})"
            )
    
    def _calculate_quality_score(self) -> float:
        """
        Calculate overall quality score (0-100).
        
        Scoring factors:
        - Missing values: -20 points per 5%
        - Duplicates: -10 points per 5%
        - Class imbalance: -15 points if severe
        - Outliers: -5 points per 5%
        - Text quality issues: -10 points per 5%
        """
        score = 100.0
        
        # Deduct for missing values
        missing_pct = self.validation_results['missing_values']['missing_percentage']
        score -= (missing_pct / 5) * 20
        
        # Deduct for duplicates
        duplicate_pct = self.validation_results['duplicates']['percentage']
        score -= (duplicate_pct / 5) * 10
        
        # Deduct for class imbalance
        imbalance_ratio = self.validation_results['label_distribution']['imbalance_ratio']
        if imbalance_ratio > 5:
            score -= 15
        elif imbalance_ratio > 2:
            score -= 8
        
        # Deduct for outliers
        outlier_pct = self.validation_results['outliers']['percentage']
        score -= (outlier_pct / 5) * 5
        
        # Deduct for text quality issues
        quality_issues = self.validation_results['text_quality']['quality_issues']
        quality_issue_pct = (quality_issues / self.validation_results['total_records']) * 100
        score -= (quality_issue_pct / 5) * 10
        
        return max(0, min(100, score))
    
    def generate_report(self, format: str = 'text') -> str:
        """
        Generate human-readable validation report.
        
        Args:
            format: Report format ('text', 'markdown', 'json')
            
        Returns:
            Formatted report string
        """
        if format == 'text':
            return self._generate_text_report()
        elif format == 'markdown':
            return self._generate_markdown_report()
        elif format == 'json':
            return json.dumps(self.validation_results, indent=2)
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def _generate_text_report(self) -> str:
        """Generate plain text report."""
        lines = []
        lines.append("=" * 70)
        lines.append("      DATA QUALITY VALIDATION REPORT")
        lines.append("=" * 70)
        lines.append(f"Timestamp: {self.validation_results['timestamp']}")
        lines.append(f"Total Records: {self.validation_results['total_records']}")
        lines.append(f"Overall Quality Score: {self.quality_score:.1f}/100")
        lines.append("")
        
        # Missing Values
        lines.append("MISSING VALUES")
        lines.append("-" * 70)
        mv = self.validation_results['missing_values']
        lines.append(f"Missing Text: {mv['missing_text']}")
        lines.append(f"Missing Labels: {mv['missing_labels']}")
        lines.append(f"Empty Text: {mv['empty_text']}")
        lines.append(f"Total Missing: {mv['total_missing']} ({mv['missing_percentage']:.1f}%)")
        lines.append("")
        
        # Duplicates
        lines.append("DUPLICATES")
        lines.append("-" * 70)
        dup = self.validation_results['duplicates']
        lines.append(f"Duplicate Articles: {dup['count']} ({dup['percentage']:.1f}%)")
        lines.append("")
        
        # Text Quality
        lines.append("TEXT QUALITY")
        lines.append("-" * 70)
        tq = self.validation_results['text_quality']
        lines.append(f"Average Length: {tq['avg_length']:.0f} characters")
        lines.append(f"Average Word Count: {tq['avg_word_count']:.0f} words")
        lines.append(f"Too Short: {tq['too_short_count']} articles")
        lines.append(f"Too Long: {tq['too_long_count']} articles")
        lines.append("")
        
        # Label Distribution
        lines.append("LABEL DISTRIBUTION")
        lines.append("-" * 70)
        ld = self.validation_results['label_distribution']
        lines.append(f"Unique Labels: {ld['unique_labels']}")
        lines.append(f"Imbalance Ratio: {ld['imbalance_ratio']:.2f}:1")
        lines.append("")
        for label, info in ld['distribution'].items():
            lines.append(f"  {label}: {info['count']} ({info['percentage']:.1f}%)")
        lines.append("")
        
        # Issues and Recommendations
        if self.issues:
            lines.append("CRITICAL ISSUES")
            lines.append("-" * 70)
            for issue in self.issues:
                lines.append(f"❌ {issue}")
            lines.append("")
        
        if self.warnings:
            lines.append("WARNINGS")
            lines.append("-" * 70)
            for warning in self.warnings:
                lines.append(f"⚠️  {warning}")
            lines.append("")
        
        if self.recommendations:
            lines.append("RECOMMENDATIONS")
            lines.append("-" * 70)
            for rec in self.recommendations:
                lines.append(f"💡 {rec}")
            lines.append("")
        
        lines.append("=" * 70)
        
        return "\n".join(lines)
    
    def _generate_markdown_report(self) -> str:
        """Generate Markdown formatted report."""
        lines = []
        lines.append("# Data Quality Validation Report")
        lines.append(f"\n**Generated:** {self.validation_results['timestamp']}")
        lines.append(f"**Total Records:** {self.validation_results['total_records']}")
        lines.append(f"**Overall Quality Score:** {self.quality_score:.1f}/100\n")
        
        lines.append("## Summary Statistics\n")
        lines.append("| Metric | Value |")
        lines.append("|--------|-------|")
        
        mv = self.validation_results['missing_values']
        lines.append(f"| Missing Values | {mv['total_missing']} ({mv['missing_percentage']:.1f}%) |")
        
        dup = self.validation_results['duplicates']
        lines.append(f"| Duplicates | {dup['count']} ({dup['percentage']:.1f}%) |")
        
        tq = self.validation_results['text_quality']
        lines.append(f"| Avg Article Length | {tq['avg_length']:.0f} chars |")
        lines.append(f"| Avg Word Count | {tq['avg_word_count']:.0f} words |")
        
        ld = self.validation_results['label_distribution']
        lines.append(f"| Unique Categories | {ld['unique_labels']} |")
        lines.append(f"| Class Imbalance Ratio | {ld['imbalance_ratio']:.2f}:1 |")
        
        if self.issues:
            lines.append("\n## ❌ Critical Issues\n")
            for issue in self.issues:
                lines.append(f"- {issue}")
        
        if self.warnings:
            lines.append("\n## ⚠️ Warnings\n")
            for warning in self.warnings:
                lines.append(f"- {warning}")
        
        if self.recommendations:
            lines.append("\n## 💡 Recommendations\n")
            for rec in self.recommendations:
                lines.append(f"- {rec}")
        
        return "\n".join(lines)
    
    def save_report(self, filepath: str, format: str = 'text'):
        """Save validation report to file."""
        report = self.generate_report(format)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(report)
        
        logger.info(f"Validation report saved to {filepath}")
    
    def get_clean_indices(self, df: pd.DataFrame, text_column: str = 'content',
                          label_column: str = 'category') -> np.ndarray:
        """
        Get indices of clean data (no missing values, not duplicates, good quality).
        
        Returns:
            Array of indices for clean data
        """
        clean_mask = (
            df[text_column].notna() &
            df[label_column].notna() &
            (df[text_column].str.len() >= self.min_text_length) &
            (df[text_column].str.len() <= self.max_text_length) &
            ~df[text_column].duplicated()
        )
        
        return np.where(clean_mask)[0]
    
    def auto_clean(self, df: pd.DataFrame, text_column: str = 'content',
                   label_column: str = 'category') -> Tuple[pd.DataFrame, Dict]:
        """
        Automatically clean dataset based on validation results.
        
        Returns:
            Tuple of (cleaned_df, cleaning_summary)
        """
        original_size = len(df)
        cleaning_summary = {'removed': {}, 'original_size': original_size}
        
        # Remove missing values
        df_clean = df.dropna(subset=[text_column, label_column])
        cleaning_summary['removed']['missing'] = original_size - len(df_clean)
        
        # Remove duplicates
        df_clean = df_clean.drop_duplicates(subset=[text_column])
        cleaning_summary['removed']['duplicates'] = len(df) - len(df_clean) - cleaning_summary['removed']['missing']
        
        # Remove too short/long articles
        df_clean = df_clean[
            (df_clean[text_column].str.len() >= self.min_text_length) &
            (df_clean[text_column].str.len() <= self.max_text_length)
        ]
        cleaning_summary['removed']['quality_issues'] = len(df) - len(df_clean) - sum(cleaning_summary['removed'].values())
        
        cleaning_summary['final_size'] = len(df_clean)
        cleaning_summary['total_removed'] = original_size - len(df_clean)
        cleaning_summary['retention_rate'] = (len(df_clean) / original_size * 100) if original_size > 0 else 0
        
        logger.info(f"Auto-cleaning complete. Removed {cleaning_summary['total_removed']} records "
                   f"({100 - cleaning_summary['retention_rate']:.1f}%)")
        
        return df_clean, cleaning_summary
