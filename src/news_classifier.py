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
        """Save trained model to disk."""
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
        
        joblib.dump(model_data, filepath)
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
    
    def __init__(self, model_types=None):
        if model_types is None:
            model_types = ['random_forest', 'logistic_regression', 'naive_bayes', 'gradient_boosting']
        
        self.model_types = model_types
        self.models = {}
        self.results = {}
    
    def compare_models(self, X_train, y_train, X_test, y_test):
        """Compare multiple models and return results."""
        logging.info("Starting model comparison...")
        
        for model_type in self.model_types:
            logging.info(f"Training {model_type}...")
            
            try:
                # Initialize and train model
                classifier = NewsClassifier(model_type)
                classifier.train(X_train, y_train)
                
                # Evaluate model
                metrics = classifier.evaluate(X_test, y_test, detailed=False)
                
                # Store results
                self.models[model_type] = classifier
                self.results[model_type] = metrics
                
                logging.info(f"{model_type} - Accuracy: {metrics['accuracy']:.4f}, F1: {metrics['f1_score']:.4f}")
                
            except Exception as e:
                logging.error(f"Error training {model_type}: {str(e)}")
                continue
        
        # Find best model
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