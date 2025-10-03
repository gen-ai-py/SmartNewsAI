"""
Performance Metrics Export System for Smart News AI

This module provides comprehensive metrics tracking, export, and visualization
for both classification and recommendation systems.

Features:
- Real-time metrics collection
- Export to JSON and CSV formats
- Performance comparison across models
- Historical metrics tracking
- Visualization generation
- Report generation with summary statistics
"""

import json
import csv
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


class MetricsCollector:
    """
    Collects and stores performance metrics for ML models.
    
    Supports:
    - Classification metrics (accuracy, precision, recall, F1)
    - Recommendation metrics (precision@k, recall@k, MAP)
    - Training time and resource usage
    - Model comparison
    """
    
    def __init__(self, metrics_dir="metrics"):
        """
        Initialize metrics collector.
        
        Args:
            metrics_dir: Directory to store metrics files
        """
        self.metrics_dir = metrics_dir
        Path(metrics_dir).mkdir(parents=True, exist_ok=True)
        
        self.classifier_metrics = []
        self.recommender_metrics = []
        self.comparison_metrics = []
        
        logger.info(f"MetricsCollector initialized. Metrics directory: {metrics_dir}")
    
    def add_classifier_metrics(self, model_name: str, metrics: Dict[str, Any],
                              training_time: float = None, dataset_info: Dict = None):
        """
        Add classification model metrics.
        
        Args:
            model_name: Name of the classifier model
            metrics: Dictionary of performance metrics
            training_time: Training time in seconds
            dataset_info: Information about the dataset used
        """
        entry = {
            'timestamp': datetime.now().isoformat(),
            'model_name': model_name,
            'accuracy': metrics.get('accuracy', 0.0),
            'precision': metrics.get('precision', 0.0),
            'recall': metrics.get('recall', 0.0),
            'f1_score': metrics.get('f1_score', 0.0),
            'training_time_seconds': training_time,
            'dataset_size': dataset_info.get('size', 0) if dataset_info else 0,
            'num_classes': dataset_info.get('num_classes', 0) if dataset_info else 0,
        }
        
        self.classifier_metrics.append(entry)
        logger.info(f"Added classifier metrics for {model_name}: Accuracy={entry['accuracy']:.4f}")
    
    def add_recommender_metrics(self, metrics: Dict[str, Any], 
                               num_users: int = None, num_items: int = None):
        """
        Add recommendation system metrics.
        
        Args:
            metrics: Dictionary of recommendation metrics
            num_users: Number of users in the system
            num_items: Number of items/articles
        """
        entry = {
            'timestamp': datetime.now().isoformat(),
            'precision_at_5': metrics.get('precision_at_5', 0.0),
            'recall_at_5': metrics.get('recall_at_5', 0.0),
            'precision_at_10': metrics.get('precision_at_10', 0.0),
            'recall_at_10': metrics.get('recall_at_10', 0.0),
            'map_score': metrics.get('map', 0.0),
            'coverage': metrics.get('coverage', 0.0),
            'diversity': metrics.get('diversity', 0.0),
            'num_users': num_users,
            'num_items': num_items,
        }
        
        self.recommender_metrics.append(entry)
        logger.info(f"Added recommender metrics: P@5={entry['precision_at_5']:.4f}, R@5={entry['recall_at_5']:.4f}")
    
    def add_comparison_entry(self, model_1: str, model_2: str, 
                            metric_name: str, improvement: float):
        """
        Add model comparison entry.
        
        Args:
            model_1: First model name
            model_2: Second model name
            metric_name: Metric being compared
            improvement: Percentage improvement
        """
        entry = {
            'timestamp': datetime.now().isoformat(),
            'model_1': model_1,
            'model_2': model_2,
            'metric_name': metric_name,
            'improvement_percentage': improvement
        }
        
        self.comparison_metrics.append(entry)
        logger.info(f"Added comparison: {model_1} vs {model_2} on {metric_name}: {improvement:+.2f}%")
    
    def export_to_json(self, filename: str = None) -> str:
        """
        Export all metrics to JSON file.
        
        Args:
            filename: Output filename (auto-generated if None)
            
        Returns:
            Path to the exported file
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"metrics_{timestamp}.json"
        
        filepath = os.path.join(self.metrics_dir, filename)
        
        data = {
            'export_date': datetime.now().isoformat(),
            'classifier_metrics': self.classifier_metrics,
            'recommender_metrics': self.recommender_metrics,
            'comparison_metrics': self.comparison_metrics,
            'summary': self._generate_summary()
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Metrics exported to JSON: {filepath}")
        return filepath
    
    def export_to_csv(self, metric_type: str = 'classifier') -> str:
        """
        Export specific metric type to CSV file.
        
        Args:
            metric_type: Type of metrics ('classifier', 'recommender', 'comparison')
            
        Returns:
            Path to the exported CSV file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{metric_type}_metrics_{timestamp}.csv"
        filepath = os.path.join(self.metrics_dir, filename)
        
        if metric_type == 'classifier':
            data = self.classifier_metrics
        elif metric_type == 'recommender':
            data = self.recommender_metrics
        elif metric_type == 'comparison':
            data = self.comparison_metrics
        else:
            raise ValueError(f"Invalid metric_type: {metric_type}")
        
        if not data:
            logger.warning(f"No {metric_type} metrics to export")
            return None
        
        # Convert to DataFrame and export
        df = pd.DataFrame(data)
        df.to_csv(filepath, index=False)
        
        logger.info(f"{metric_type.capitalize()} metrics exported to CSV: {filepath}")
        return filepath
    
    def _generate_summary(self) -> Dict[str, Any]:
        """Generate summary statistics for all metrics."""
        summary = {
            'total_classifier_runs': len(self.classifier_metrics),
            'total_recommender_evaluations': len(self.recommender_metrics),
            'total_comparisons': len(self.comparison_metrics)
        }
        
        # Classifier summary
        if self.classifier_metrics:
            df_clf = pd.DataFrame(self.classifier_metrics)
            summary['classifier_summary'] = {
                'best_accuracy': df_clf['accuracy'].max(),
                'best_model': df_clf.loc[df_clf['accuracy'].idxmax(), 'model_name'],
                'avg_accuracy': df_clf['accuracy'].mean(),
                'avg_training_time': df_clf['training_time_seconds'].mean() if 'training_time_seconds' in df_clf.columns else None
            }
        
        # Recommender summary
        if self.recommender_metrics:
            df_rec = pd.DataFrame(self.recommender_metrics)
            summary['recommender_summary'] = {
                'best_precision_at_5': df_rec['precision_at_5'].max(),
                'avg_map_score': df_rec['map_score'].mean(),
                'avg_coverage': df_rec['coverage'].mean() if 'coverage' in df_rec.columns else None
            }
        
        return summary
    
    def generate_report(self, output_format: str = 'text') -> str:
        """
        Generate comprehensive metrics report.
        
        Args:
            output_format: Format of report ('text', 'markdown', 'html')
            
        Returns:
            Report content as string
        """
        summary = self._generate_summary()
        
        if output_format == 'text':
            return self._generate_text_report(summary)
        elif output_format == 'markdown':
            return self._generate_markdown_report(summary)
        elif output_format == 'html':
            return self._generate_html_report(summary)
        else:
            raise ValueError(f"Unsupported format: {output_format}")
    
    def _generate_text_report(self, summary: Dict) -> str:
        """Generate plain text report."""
        lines = []
        lines.append("=" * 60)
        lines.append("        SMART NEWS AI - PERFORMANCE METRICS REPORT")
        lines.append("=" * 60)
        lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("")
        
        # Classifier metrics
        lines.append("CLASSIFICATION METRICS")
        lines.append("-" * 60)
        if self.classifier_metrics:
            clf_summary = summary.get('classifier_summary', {})
            lines.append(f"Total Model Runs: {summary['total_classifier_runs']}")
            lines.append(f"Best Accuracy: {clf_summary.get('best_accuracy', 0):.4f}")
            lines.append(f"Best Model: {clf_summary.get('best_model', 'N/A')}")
            lines.append(f"Average Accuracy: {clf_summary.get('avg_accuracy', 0):.4f}")
            
            if clf_summary.get('avg_training_time'):
                lines.append(f"Average Training Time: {clf_summary['avg_training_time']:.2f}s")
        else:
            lines.append("No classification metrics recorded")
        lines.append("")
        
        # Recommender metrics
        lines.append("RECOMMENDATION METRICS")
        lines.append("-" * 60)
        if self.recommender_metrics:
            rec_summary = summary.get('recommender_summary', {})
            lines.append(f"Total Evaluations: {summary['total_recommender_evaluations']}")
            lines.append(f"Best Precision@5: {rec_summary.get('best_precision_at_5', 0):.4f}")
            lines.append(f"Average MAP Score: {rec_summary.get('avg_map_score', 0):.4f}")
            
            if rec_summary.get('avg_coverage'):
                lines.append(f"Average Coverage: {rec_summary['avg_coverage']:.2%}")
        else:
            lines.append("No recommendation metrics recorded")
        lines.append("")
        
        # Recent classifier runs
        if self.classifier_metrics:
            lines.append("RECENT CLASSIFIER RUNS (Last 5)")
            lines.append("-" * 60)
            for metric in self.classifier_metrics[-5:]:
                lines.append(f"{metric['timestamp'][:19]} | {metric['model_name']:<20} | Acc: {metric['accuracy']:.4f} | F1: {metric['f1_score']:.4f}")
            lines.append("")
        
        lines.append("=" * 60)
        
        return "\n".join(lines)
    
    def _generate_markdown_report(self, summary: Dict) -> str:
        """Generate Markdown formatted report."""
        lines = []
        lines.append("# Smart News AI - Performance Metrics Report")
        lines.append(f"\n**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        # Classifier metrics
        lines.append("## Classification Metrics\n")
        if self.classifier_metrics:
            clf_summary = summary.get('classifier_summary', {})
            lines.append(f"- **Total Model Runs:** {summary['total_classifier_runs']}")
            lines.append(f"- **Best Accuracy:** {clf_summary.get('best_accuracy', 0):.4f}")
            lines.append(f"- **Best Model:** {clf_summary.get('best_model', 'N/A')}")
            lines.append(f"- **Average Accuracy:** {clf_summary.get('avg_accuracy', 0):.4f}")
            
            lines.append("\n### Recent Runs\n")
            lines.append("| Timestamp | Model | Accuracy | Precision | Recall | F1 Score |")
            lines.append("|-----------|-------|----------|-----------|--------|----------|")
            
            for metric in self.classifier_metrics[-10:]:
                lines.append(f"| {metric['timestamp'][:19]} | {metric['model_name']} | "
                           f"{metric['accuracy']:.4f} | {metric['precision']:.4f} | "
                           f"{metric['recall']:.4f} | {metric['f1_score']:.4f} |")
        else:
            lines.append("No classification metrics recorded\n")
        
        # Recommender metrics
        lines.append("\n## Recommendation Metrics\n")
        if self.recommender_metrics:
            rec_summary = summary.get('recommender_summary', {})
            lines.append(f"- **Total Evaluations:** {summary['total_recommender_evaluations']}")
            lines.append(f"- **Best Precision@5:** {rec_summary.get('best_precision_at_5', 0):.4f}")
            lines.append(f"- **Average MAP Score:** {rec_summary.get('avg_map_score', 0):.4f}")
        else:
            lines.append("No recommendation metrics recorded\n")
        
        return "\n".join(lines)
    
    def _generate_html_report(self, summary: Dict) -> str:
        """Generate HTML formatted report."""
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Smart News AI - Performance Metrics</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        h1 {{ color: #2c3e50; }}
        h2 {{ color: #34495e; border-bottom: 2px solid #3498db; padding-bottom: 5px; }}
        table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #3498db; color: white; }}
        tr:nth-child(even) {{ background-color: #f2f2f2; }}
        .metric-value {{ font-weight: bold; color: #27ae60; }}
    </style>
</head>
<body>
    <h1>Smart News AI - Performance Metrics Report</h1>
    <p><strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    
    <h2>Classification Metrics Summary</h2>
"""
        
        if self.classifier_metrics:
            clf_summary = summary.get('classifier_summary', {})
            html += f"""
    <ul>
        <li>Total Model Runs: <span class="metric-value">{summary['total_classifier_runs']}</span></li>
        <li>Best Accuracy: <span class="metric-value">{clf_summary.get('best_accuracy', 0):.4f}</span></li>
        <li>Best Model: <span class="metric-value">{clf_summary.get('best_model', 'N/A')}</span></li>
        <li>Average Accuracy: <span class="metric-value">{clf_summary.get('avg_accuracy', 0):.4f}</span></li>
    </ul>
    
    <h3>Recent Classifier Runs</h3>
    <table>
        <tr>
            <th>Timestamp</th>
            <th>Model</th>
            <th>Accuracy</th>
            <th>Precision</th>
            <th>Recall</th>
            <th>F1 Score</th>
        </tr>
"""
            for metric in self.classifier_metrics[-10:]:
                html += f"""
        <tr>
            <td>{metric['timestamp'][:19]}</td>
            <td>{metric['model_name']}</td>
            <td>{metric['accuracy']:.4f}</td>
            <td>{metric['precision']:.4f}</td>
            <td>{metric['recall']:.4f}</td>
            <td>{metric['f1_score']:.4f}</td>
        </tr>
"""
            html += "    </table>\n"
        else:
            html += "    <p>No classification metrics recorded</p>\n"
        
        html += """
</body>
</html>
"""
        return html
    
    def save_report(self, format: str = 'text', filename: str = None) -> str:
        """
        Save generated report to file.
        
        Args:
            format: Report format ('text', 'markdown', 'html')
            filename: Output filename (auto-generated if None)
            
        Returns:
            Path to the saved report file
        """
        report = self.generate_report(format)
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            ext = 'txt' if format == 'text' else format
            filename = f"metrics_report_{timestamp}.{ext}"
        
        filepath = os.path.join(self.metrics_dir, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(report)
        
        logger.info(f"Report saved to: {filepath}")
        return filepath
    
    def load_metrics_from_json(self, filepath: str):
        """
        Load metrics from previously exported JSON file.
        
        Args:
            filepath: Path to JSON metrics file
        """
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.classifier_metrics.extend(data.get('classifier_metrics', []))
        self.recommender_metrics.extend(data.get('recommender_metrics', []))
        self.comparison_metrics.extend(data.get('comparison_metrics', []))
        
        logger.info(f"Loaded metrics from {filepath}")
    
    def clear_metrics(self, metric_type: str = 'all'):
        """
        Clear stored metrics.
        
        Args:
            metric_type: Type to clear ('all', 'classifier', 'recommender', 'comparison')
        """
        if metric_type in ('all', 'classifier'):
            self.classifier_metrics.clear()
        if metric_type in ('all', 'recommender'):
            self.recommender_metrics.clear()
        if metric_type in ('all', 'comparison'):
            self.comparison_metrics.clear()
        
        logger.info(f"Cleared {metric_type} metrics")


def calculate_recommendation_metrics(recommendations: List[Dict], 
                                     ground_truth: List[str],
                                     k: int = 5) -> Dict[str, float]:
    """
    Calculate recommendation metrics given recommendations and ground truth.
    
    Args:
        recommendations: List of recommended article dicts with 'article_id'
        ground_truth: List of article IDs that the user actually engaged with
        k: Number of top recommendations to consider
        
    Returns:
        Dictionary of metrics
    """
    if not ground_truth:
        return {
            'precision_at_k': 0.0,
            'recall_at_k': 0.0,
            'f1_at_k': 0.0
        }
    
    # Get top k recommendations
    top_k = [rec['article_id'] for rec in recommendations[:k]]
    ground_truth_set = set(ground_truth)
    
    # Calculate hits
    hits = len(set(top_k) & ground_truth_set)
    
    # Precision@k
    precision = hits / k if k > 0 else 0.0
    
    # Recall@k
    recall = hits / len(ground_truth_set) if ground_truth_set else 0.0
    
    # F1@k
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        'precision_at_k': precision,
        'recall_at_k': recall,
        'f1_at_k': f1,
        'hits': hits
    }


def calculate_map_score(all_recommendations: List[List[Dict]], 
                       all_ground_truth: List[List[str]]) -> float:
    """
    Calculate Mean Average Precision (MAP) across all users.
    
    Args:
        all_recommendations: List of recommendation lists for each user
        all_ground_truth: List of ground truth lists for each user
        
    Returns:
        MAP score
    """
    if not all_recommendations or len(all_recommendations) != len(all_ground_truth):
        return 0.0
    
    average_precisions = []
    
    for recs, truth in zip(all_recommendations, all_ground_truth):
        if not truth:
            continue
        
        truth_set = set(truth)
        hits = 0
        precisions = []
        
        for i, rec in enumerate(recs, 1):
            if rec['article_id'] in truth_set:
                hits += 1
                precisions.append(hits / i)
        
        if precisions:
            average_precisions.append(sum(precisions) / len(truth_set))
    
    return sum(average_precisions) / len(average_precisions) if average_precisions else 0.0
