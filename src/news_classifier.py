"""
News classifier module implementing multiple machine learning algorithms
for news article classification.
"""

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score, GridSearchCV
import logging
import os
from datetime import datetime

class NewsClassifier:
    """Multi-algorithm news classifier with model comparison and selection."""
    
    def __init__(self, model_type='random_forest'):
        """
        Initialize classifier with specified model type.
        
        Args:
            model_type (str): Type of model to use. Options: 
                'random_forest', 'logistic_regression', 'svm', 'naive_bayes', 'gradient_boosting'
        """
        self.model_type = model_type
        self.model = None
        self.is_trained = False
        self.feature_importance = None
        self.classes = None
        self.training_history = []
        
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the specified model with default hyperparameters."""
        model_configs = {
            'random_forest': RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            ),
            'logistic_regression': LogisticRegression(
                max_iter=1000,
                random_state=42,
                multi_class='ovr'
            ),
            'svm': SVC(
                kernel='linear',
                probability=True,
                random_state=42
            ),
            'naive_bayes': MultinomialNB(alpha=1.0),
            'gradient_boosting': GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                random_state=42
            )
        }
        
        if self.model_type not in model_configs:
            raise ValueError(f"Unsupported model type: {self.model_type}")
        
        self.model = model_configs[self.model_type]
        logging.info(f"Initialized {self.model_type} classifier")
    
    def train(self, X_train, y_train, X_val=None, y_val=None):
        """
        Train the classifier.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
        """
        logging.info(f"Training {self.model_type} classifier...")
        
        start_time = datetime.now()
        self.model.fit(X_train, y_train)
        training_time = (datetime.now() - start_time).total_seconds()
        
        self.is_trained = True
        self.classes = self.model.classes_
        
        # Calculate feature importance if available
        if hasattr(self.model, 'feature_importances_'):
            self.feature_importance = self.model.feature_importances_
        elif hasattr(self.model, 'coef_'):
            # For linear models, use coefficient magnitudes
            self.feature_importance = np.abs(self.model.coef_).mean(axis=0)
        
        # Evaluate on training data
        train_pred = self.model.predict(X_train)
        train_accuracy = accuracy_score(y_train, train_pred)
        
        # Evaluate on validation data if provided
        val_accuracy = None
        if X_val is not None and y_val is not None:
            val_pred = self.model.predict(X_val)
            val_accuracy = accuracy_score(y_val, val_pred)
        
        # Record training history
        history_entry = {
            'timestamp': datetime.now(),
            'model_type': self.model_type,
            'training_time': training_time,
            'train_accuracy': train_accuracy,
            'val_accuracy': val_accuracy,
            'train_samples': len(X_train)
        }
        self.training_history.append(history_entry)
        
        logging.info(f"Training completed in {training_time:.2f}s. Train accuracy: {train_accuracy:.4f}")
        if val_accuracy:
            logging.info(f"Validation accuracy: {val_accuracy:.4f}")
    
    def predict(self, X):
        """Make predictions on new data."""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """Get prediction probabilities."""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X)
        else:
            raise ValueError(f"Model {self.model_type} does not support probability predictions")
    
    def evaluate(self, X_test, y_test, detailed=True):
        """
        Evaluate model performance on test data.
        
        Args:
            X_test: Test features
            y_test: Test labels
            detailed (bool): Whether to return detailed metrics
            
        Returns:
            dict: Performance metrics
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before evaluation")
        
        y_pred = self.predict(X_test)
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
            'f1_score': f1_score(y_test, y_pred, average='weighted', zero_division=0)
        }
        
        if detailed:
            metrics['classification_report'] = classification_report(y_test, y_pred, zero_division=0)
            metrics['confusion_matrix'] = confusion_matrix(y_test, y_pred)
        
        return metrics
    
    def cross_validate(self, X, y, cv=5):
        """Perform cross-validation."""
        if not self.is_trained:
            # Use a fresh model for cross-validation
            temp_model = self.__class__(self.model_type)._initialize_model()
        else:
            temp_model = self.model
        
        cv_scores = cross_val_score(temp_model, X, y, cv=cv, scoring='accuracy')
        
        return {
            'cv_scores': cv_scores,
            'mean_score': cv_scores.mean(),
            'std_score': cv_scores.std()
        }
    
    def get_top_features(self, feature_names, n_features=20):
        """Get top important features."""
        if self.feature_importance is None:
            return []
        
        if len(feature_names) != len(self.feature_importance):
            logging.warning("Feature names length doesn't match importance scores")
            return []
        
        # Get indices of top features
        top_indices = np.argsort(self.feature_importance)[-n_features:][::-1]
        
        top_features = [
            (feature_names[idx], self.feature_importance[idx])
            for idx in top_indices
        ]
        
        return top_features
    
    def save_model(self, filepath):
        """Save trained model to disk with compression for large models."""
        if not self.is_trained:
            raise ValueError("No trained model to save")
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        model_data = {
            'model': self.model,
            'model_type': self.model_type,
            'classes': self.classes,
            'feature_importance': self.feature_importance,
            'training_history': self.training_history,
            'is_trained': self.is_trained
        }
        
        # Use compression and protocol 4 for large models
        joblib.dump(model_data, filepath, compress=3, protocol=4)
        logging.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load trained model from disk."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        model_data = joblib.load(filepath)
        
        self.model = model_data['model']
        self.model_type = model_data['model_type']
        self.classes = model_data['classes']
        self.feature_importance = model_data.get('feature_importance')
        self.training_history = model_data.get('training_history', [])
        self.is_trained = model_data['is_trained']
        
        logging.info(f"Model loaded from {filepath}")

class ModelComparison:
    """Compare multiple classifier models."""
    
    def __init__(self, model_types=None, use_cross_validation=False, cv_folds=5):
        """
        Initialize ModelComparison.
        
        Args:
            model_types (list): List of model types to compare
            use_cross_validation (bool): Whether to use cross-validation (default: False)
            cv_folds (int): Number of folds for cross-validation (default: 5)
        """
        if model_types is None:
            model_types = ['random_forest', 'logistic_regression', 'naive_bayes', 'gradient_boosting']
        
        self.model_types = model_types
        self.models = {}
        self.results = {}
        self.use_cross_validation = use_cross_validation
        self.cv_folds = cv_folds
        self.cv_results = {}
    
    def compare_models(self, X_train, y_train, X_test, y_test):
        """
        Compare multiple models and return results.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_test: Test features
            y_test: Test labels
            
        Returns:
            tuple: (results dict, best model type)
        """
        logging.info("Starting model comparison...")
        
        if self.use_cross_validation:
            logging.info(f"Using {self.cv_folds}-fold cross-validation")
        
        for model_type in self.model_types:
            logging.info(f"Training {model_type}...")
            
            try:
                # Initialize and train model
                classifier = NewsClassifier(model_type)
                classifier.train(X_train, y_train)
                
                # Evaluate model on test set
                metrics = classifier.evaluate(X_test, y_test, detailed=False)
                
                # Perform cross-validation if enabled
                if self.use_cross_validation:
                    from sklearn.model_selection import cross_validate
                    from sklearn.model_selection import StratifiedKFold
                    
                    # Combine train and test for cross-validation
                    from scipy.sparse import vstack
                    X_combined = vstack([X_train, X_test])
                    y_combined = np.concatenate([y_train, y_test])
                    
                    # Perform cross-validation
                    cv = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=42)
                    cv_scores = cross_validate(
                        classifier.model,
                        X_combined,
                        y_combined,
                        cv=cv,
                        scoring=['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted'],
                        return_train_score=True
                    )
                    
                    # Store cross-validation results
                    self.cv_results[model_type] = {
                        'cv_accuracy_mean': cv_scores['test_accuracy'].mean(),
                        'cv_accuracy_std': cv_scores['test_accuracy'].std(),
                        'cv_precision_mean': cv_scores['test_precision_weighted'].mean(),
                        'cv_precision_std': cv_scores['test_precision_weighted'].std(),
                        'cv_recall_mean': cv_scores['test_recall_weighted'].mean(),
                        'cv_recall_std': cv_scores['test_recall_weighted'].std(),
                        'cv_f1_mean': cv_scores['test_f1_weighted'].mean(),
                        'cv_f1_std': cv_scores['test_f1_weighted'].std(),
                        'cv_train_accuracy_mean': cv_scores['train_accuracy'].mean(),
                        'cv_scores_detail': cv_scores['test_accuracy']
                    }
                    
                    # Add CV results to metrics
                    metrics.update(self.cv_results[model_type])
                    
                    logging.info(f"{model_type} - CV Accuracy: {metrics['cv_accuracy_mean']:.4f} ± {metrics['cv_accuracy_std']:.4f}")
                    logging.info(f"{model_type} - Test Accuracy: {metrics['accuracy']:.4f}, F1: {metrics['f1_score']:.4f}")
                else:
                    logging.info(f"{model_type} - Accuracy: {metrics['accuracy']:.4f}, F1: {metrics['f1_score']:.4f}")
                
                # Store results
                self.models[model_type] = classifier
                self.results[model_type] = metrics
                
            except Exception as e:
                logging.error(f"Error training {model_type}: {str(e)}")
                continue
        
        # Find best model (use CV accuracy if available, otherwise test accuracy)
        if self.use_cross_validation:
            best_model_type = max(self.results.keys(), key=lambda k: self.results[k].get('cv_accuracy_mean', 0))
            logging.info(f"Best model: {best_model_type} (CV Accuracy: {self.results[best_model_type]['cv_accuracy_mean']:.4f})")
        else:
            best_model_type = max(self.results.keys(), key=lambda k: self.results[k]['accuracy'])
            logging.info(f"Best model: {best_model_type} (Accuracy: {self.results[best_model_type]['accuracy']:.4f})")
        
        return self.results, best_model_type
    
    def get_results_dataframe(self):
        """Get comparison results as pandas DataFrame."""
        if not self.results:
            return pd.DataFrame()
        
        df = pd.DataFrame(self.results).T
        df.index.name = 'Model'
        return df.round(4)
    
    def get_statistical_comparison(self):
        """
        Perform statistical significance testing between models using paired t-test.
        Only available when cross-validation is enabled.
        
        Returns:
            dict: Statistical comparison results
        """
        if not self.use_cross_validation or not self.cv_results:
            logging.warning("Statistical comparison requires cross-validation to be enabled")
            return None
        
        from scipy import stats
        
        model_list = list(self.cv_results.keys())
        comparisons = {}
        
        for i, model1 in enumerate(model_list):
            for model2 in model_list[i+1:]:
                scores1 = self.cv_results[model1]['cv_scores_detail']
                scores2 = self.cv_results[model2]['cv_scores_detail']
                
                # Perform paired t-test
                t_stat, p_value = stats.ttest_rel(scores1, scores2)
                
                comparison_key = f"{model1}_vs_{model2}"
                comparisons[comparison_key] = {
                    'model1': model1,
                    'model2': model2,
                    'model1_mean': scores1.mean(),
                    'model2_mean': scores2.mean(),
                    't_statistic': t_stat,
                    'p_value': p_value,
                    'significant': p_value < 0.05,
                    'better_model': model1 if scores1.mean() > scores2.mean() else model2
                }
        
        return comparisons
    
    def visualize_cv_results(self):
        """
        Visualize cross-validation results.
        Returns a formatted string with visualization data.
        """
        if not self.use_cross_validation or not self.cv_results:
            return "Cross-validation results not available"
        
        output = ["\n=== Cross-Validation Results ==="]
        output.append(f"\nNumber of folds: {self.cv_folds}")
        output.append("\n" + "-" * 80)
        
        for model_type, cv_result in self.cv_results.items():
            output.append(f"\n{model_type.upper()}:")
            output.append(f"  Accuracy:  {cv_result['cv_accuracy_mean']:.4f} ± {cv_result['cv_accuracy_std']:.4f}")
            output.append(f"  Precision: {cv_result['cv_precision_mean']:.4f} ± {cv_result['cv_precision_std']:.4f}")
            output.append(f"  Recall:    {cv_result['cv_recall_mean']:.4f} ± {cv_result['cv_recall_std']:.4f}")
            output.append(f"  F1-Score:  {cv_result['cv_f1_mean']:.4f} ± {cv_result['cv_f1_std']:.4f}")
            output.append(f"  Fold scores: {[f'{score:.4f}' for score in cv_result['cv_scores_detail']]}")
        
        # Statistical comparison
        comparisons = self.get_statistical_comparison()
        if comparisons:
            output.append("\n" + "-" * 80)
            output.append("\n=== Statistical Significance Testing (Paired t-test) ===")
            for comp_name, comp_data in comparisons.items():
                output.append(f"\n{comp_data['model1']} vs {comp_data['model2']}:")
                output.append(f"  Mean accuracy: {comp_data['model1_mean']:.4f} vs {comp_data['model2_mean']:.4f}")
                output.append(f"  t-statistic: {comp_data['t_statistic']:.4f}")
                output.append(f"  p-value: {comp_data['p_value']:.4f}")
                output.append(f"  Significant: {'Yes' if comp_data['significant'] else 'No'} (α=0.05)")
                if comp_data['significant']:
                    output.append(f"  Winner: {comp_data['better_model']}")
        
        output.append("\n" + "=" * 80)
        return "\n".join(output)

def hyperparameter_tuning(X_train, y_train, model_type='random_forest', cv=3):
    """Perform hyperparameter tuning for specified model."""
    
    param_grids = {
        'random_forest': {
            'n_estimators': [50, 100, 200],
            'max_depth': [5, 10, 15],
            'min_samples_split': [2, 5, 10]
        },
        'logistic_regression': {
            'C': [0.1, 1.0, 10.0],
            'penalty': ['l1', 'l2']
        },
        'svm': {
            'C': [0.1, 1.0, 10.0],
            'kernel': ['linear', 'rbf']
        }
    }
    
    if model_type not in param_grids:
        raise ValueError(f"Hyperparameter tuning not available for {model_type}")
    
    classifier = NewsClassifier(model_type)
    
    grid_search = GridSearchCV(
        classifier.model,
        param_grids[model_type],
        cv=cv,
        scoring='accuracy',
        n_jobs=-1
    )
    
    grid_search.fit(X_train, y_train)
    
    return {
        'best_params': grid_search.best_params_,
        'best_score': grid_search.best_score_,
        'best_estimator': grid_search.best_estimator_
    }