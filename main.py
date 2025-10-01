"""
Smart News AI - Main Application Interface
Interactive CLI for news classification and recommendation system.
"""

import os
import sys
import argparse
import logging
from datetime import datetime

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.data_preprocessing import load_and_preprocess_data, create_train_test_split, FeatureExtractor
from src.news_classifier import NewsClassifier, ModelComparison
from src.recommendation_engine import HybridRecommender, generate_sample_interactions
from src.data_generator import create_sample_data
from src.logging_config import setup_logging, get_logger
import pandas as pd
import numpy as np

# Configure logging with rotation and multiple handlers
config_file = os.path.join(os.path.dirname(__file__), 'config', 'logging_config.yaml')
logging_config = setup_logging(
    config_file=config_file if os.path.exists(config_file) else None,
    log_dir='logs',
    log_level='INFO'
)

# Get module logger
logger = get_logger(__name__)

class SmartNewsAI:
    """Main application class for Smart News AI system."""
    
    def __init__(self, data_path="data"):
        self.data_path = data_path
        self.classifier = None
        self.recommender = None
        self.feature_extractor = None
        self.articles_df = None
        self.interactions_df = None
        
        # Ensure data directory exists
        os.makedirs(self.data_path, exist_ok=True)
        os.makedirs("models", exist_ok=True)
    
    def setup_data(self, regenerate=False):
        """Setup or regenerate sample data."""
        articles_path = os.path.join(self.data_path, "news_articles.csv")
        interactions_path = os.path.join(self.data_path, "user_interactions.csv")
        
        if regenerate or not os.path.exists(articles_path):
            print("Generating sample data...")
            create_sample_data(self.data_path)
        
        # Load data
        print("Loading datasets...")
        self.articles_df = pd.read_csv(articles_path)
        
        if os.path.exists(interactions_path):
            self.interactions_df = pd.read_csv(interactions_path)
        else:
            print("Generating user interactions...")
            self.interactions_df = generate_sample_interactions(self.articles_df)
            self.interactions_df.to_csv(interactions_path, index=False)
        
        print(f"Loaded {len(self.articles_df)} articles and {len(self.interactions_df)} interactions")
    
    def train_classifier(self, model_type='random_forest', save_model=True):
        """Train news classifier."""
        print(f"\\nTraining {model_type} classifier...")
        
        # Prepare data for classification
        self.feature_extractor = FeatureExtractor(max_features=3000)
        
        # Extract features from articles
        X = self.feature_extractor.fit_transform_text(self.articles_df['content'].tolist())
        y = self.feature_extractor.fit_transform_labels(self.articles_df['category'].tolist())
        
        # Create train-test split
        X_train, X_test, y_train, y_test = create_train_test_split(X, y, test_size=0.2)
        
        # Initialize and train classifier
        self.classifier = NewsClassifier(model_type=model_type)
        self.classifier.train(X_train, y_train, X_test, y_test)
        
        # Evaluate classifier
        metrics = self.classifier.evaluate(X_test, y_test, detailed=True)
        
        print(f"Classifier Performance:")
        print(f"- Accuracy: {metrics['accuracy']:.4f}")
        print(f"- Precision: {metrics['precision']:.4f}")
        print(f"- Recall: {metrics['recall']:.4f}")
        print(f"- F1 Score: {metrics['f1_score']:.4f}")
        
        # Show top features
        feature_names = self.feature_extractor.get_feature_names()
        if len(feature_names) > 0:
            top_features = self.classifier.get_top_features(feature_names, n_features=10)
            print(f"\\nTop 10 Important Features:")
            for feature, importance in top_features:
                print(f"- {feature}: {importance:.4f}")
        
        # Save model if requested
        if save_model:
            model_path = os.path.join("models", f"classifier_{model_type}.pkl")
            self.classifier.save_model(model_path)
            
            # Save feature extractor with compression
            import joblib
            fe_path = os.path.join("models", "feature_extractor.pkl")
            joblib.dump(self.feature_extractor, fe_path, compress=3, protocol=4)
            print(f"\\nModel saved to {model_path}")
            print(f"Feature extractor saved to {fe_path}")
    
    def compare_models(self, use_cross_validation=True, cv_folds=5):
        """Compare different classification models."""
        print("\\nComparing different models...")
        
        # Prepare data
        X = self.feature_extractor.fit_transform_text(self.articles_df['content'].tolist())
        y = self.feature_extractor.fit_transform_labels(self.articles_df['category'].tolist())
        X_train, X_test, y_train, y_test = create_train_test_split(X, y, test_size=0.2)
        
        # Compare models with optional cross-validation
        comparison = ModelComparison(
            ['random_forest', 'logistic_regression', 'naive_bayes'],
            use_cross_validation=use_cross_validation,
            cv_folds=cv_folds
        )
        results, best_model = comparison.compare_models(X_train, y_train, X_test, y_test)
        
        print("\\nModel Comparison Results:")
        results_df = comparison.get_results_dataframe()
        print(results_df.to_string())
        print(f"\\nBest performing model: {best_model}")
        
        # Display cross-validation visualization if enabled
        if use_cross_validation:
            print(comparison.visualize_cv_results())
    
    def train_recommender(self, save_model=True):
        """Train recommendation system."""
        print("\\nTraining hybrid recommendation system...")
        
        # Initialize hybrid recommender
        self.recommender = HybridRecommender(content_weight=0.6, collaborative_weight=0.4)
        
        # Fit the recommender
        self.recommender.fit(self.articles_df, self.interactions_df)
        
        # Save model if requested
        if save_model:
            rec_path = os.path.join("models", "recommender.pkl")
            self.recommender.save_model(rec_path)
            print(f"Recommender saved to {rec_path}")
        
        print("Recommendation system trained successfully!")
    
    def classify_article(self, text):
        """Classify a single article."""
        if self.classifier is None or self.feature_extractor is None:
            print("Error: Classifier not trained. Please train the classifier first.")
            return None
        
        # Transform text to features
        X = self.feature_extractor.transform_text([text])
        
        # Make prediction
        prediction = self.classifier.predict(X)[0]
        probabilities = self.classifier.predict_proba(X)[0]
        
        # Convert back to category name
        category = self.feature_extractor.inverse_transform_labels([prediction])[0]
        
        # Get top 3 predictions with probabilities
        top_indices = np.argsort(probabilities)[::-1][:3]
        top_predictions = []
        
        for idx in top_indices:
            cat_name = self.feature_extractor.inverse_transform_labels([idx])[0]
            prob = probabilities[idx]
            top_predictions.append((cat_name, prob))
        
        return category, top_predictions
    
    def get_recommendations(self, user_id, n_recommendations=5):
        """Get recommendations for a user."""
        if self.recommender is None:
            print("Error: Recommender not trained. Please train the recommender first.")
            return None
        
        recommendations = self.recommender.get_recommendations(user_id, n_recommendations)
        return recommendations
    
    def batch_recommendations(self, user_ids, n_recommendations=5, log_memory=True):
        """
        Generate recommendations for multiple users in batch mode.
        Optimized for large-scale processing with memory management.
        
        Args:
            user_ids: List of user IDs
            n_recommendations: Number of recommendations per user
            log_memory: Whether to log memory usage
        
        Returns:
            dict: User ID -> recommendations mapping
        """
        if self.recommender is None:
            print("Error: Recommender not trained.")
            return None
        
        import gc
        results = {}
        
        print(f"\nProcessing {len(user_ids)} users in batch mode...")
        
        for i, user_id in enumerate(user_ids):
            # Get recommendations with batch_mode enabled
            recommendations = self.recommender.get_recommendations(
                user_id, 
                n_recommendations, 
                batch_mode=True
            )
            results[user_id] = recommendations
            
            # Periodic memory cleanup
            if (i + 1) % 100 == 0:
                gc.collect()
                
                if log_memory:
                    memory_mb = self.recommender.get_profile_memory_usage()
                    print(f"Processed {i+1}/{len(user_ids)} users | " + 
                          f"Profile memory: {memory_mb:.2f} MB | " +
                          f"Active profiles: {len(self.recommender.user_profiles)}")
        
        print(f"\nBatch processing complete! Total users processed: {len(results)}")
        
        if log_memory:
            final_memory = self.recommender.get_profile_memory_usage()
            print(f"Final profile memory usage: {final_memory:.2f} MB")
        
        return results
    
    def simulate_user_interactions(self, user_id, n_interactions=5):
        """Simulate some user interactions for demo purposes."""
        if self.recommender is None:
            print("Error: Recommender not available.")
            return
        
        print(f"\\nSimulating interactions for {user_id}...")
        
        # Get random articles for interaction
        sample_articles = self.articles_df.sample(n_interactions)
        
        for _, article in sample_articles.iterrows():
            # Simulate rating (higher for tech and science articles)
            base_rating = 3
            if article['category'] in ['technology', 'science']:
                base_rating += np.random.randint(1, 3)
            else:
                base_rating += np.random.randint(-1, 2)
            
            rating = max(1, min(5, base_rating))
            
            # Add interaction
            self.recommender.add_user_interaction(
                user_id, 
                article['id'], 
                article['category'], 
                rating
            )
            
            print(f"- Rated '{article['title'][:50]}...' ({article['category']}): {rating}/5")
    
    def interactive_demo(self):
        """Run interactive demo."""
        print("\\n" + "="*60)
        print("      SMART NEWS AI - INTERACTIVE DEMO")
        print("="*60)
        
        while True:
            print("\\nChoose an option:")
            print("1. Classify article text")
            print("2. Get personalized recommendations")
            print("3. Simulate user interactions")
            print("4. Show article statistics")
            print("5. Exit")
            
            choice = input("\\nEnter your choice (1-5): ").strip()
            
            if choice == '1':
                self.demo_classification()
            elif choice == '2':
                self.demo_recommendations()
            elif choice == '3':
                self.demo_user_interactions()
            elif choice == '4':
                self.show_statistics()
            elif choice == '5':
                print("\\nThanks for using Smart News AI!")
                break
            else:
                print("Invalid choice. Please try again.")
    
    def demo_classification(self):
        """Demo article classification."""
        print("\\n--- Article Classification Demo ---")
        
        # Offer sample articles or custom text
        print("\\nChoose:")
        print("1. Classify a sample article")
        print("2. Enter custom text")
        
        sub_choice = input("Enter choice (1-2): ").strip()
        
        if sub_choice == '1':
            # Show sample article
            sample_article = self.articles_df.sample(1).iloc[0]
            print(f"\\nSample Article:")
            print(f"Title: {sample_article['title']}")
            print(f"Content: {sample_article['content'][:200]}...")
            print(f"Actual Category: {sample_article['category']}")
            
            text = sample_article['content']
        elif sub_choice == '2':
            text = input("\\nEnter article text: ").strip()
        else:
            print("Invalid choice.")
            return
        
        # Classify
        result = self.classify_article(text)
        if result:
            category, top_predictions = result
            print(f"\\nPredicted Category: {category}")
            print("\\nTop 3 Predictions:")
            for cat, prob in top_predictions:
                print(f"- {cat}: {prob:.3f}")
    
    def demo_recommendations(self):
        """Demo recommendation system."""
        print("\\n--- Personalized Recommendations Demo ---")
        
        # Get existing user or create new one
        user_id = input("\\nEnter user ID (or press Enter for 'demo_user'): ").strip()
        if not user_id:
            user_id = "demo_user"
        
        # Get recommendations
        print(f"\\nGetting recommendations for {user_id}...")
        recommendations = self.get_recommendations(user_id, n_recommendations=5)
        
        if recommendations:
            print(f"\\nTop 5 Recommendations for {user_id}:")
            for i, rec in enumerate(recommendations, 1):
                article_info = self.articles_df[self.articles_df['id'] == rec['article_id']].iloc[0]
                print(f"{i}. {rec['title']} ({rec['category']})")
                print(f"   Score: {rec['hybrid_score']:.3f}")
                print(f"   Content: {article_info['content'][:100]}...")
                print()
    
    def demo_user_interactions(self):
        """Demo user interaction simulation."""
        print("\\n--- User Interaction Simulation ---")
        
        user_id = input("\\nEnter user ID: ").strip()
        if not user_id:
            user_id = "demo_user"
        
        n_interactions = input("Number of interactions to simulate (default 3): ").strip()
        try:
            n_interactions = int(n_interactions) if n_interactions else 3
        except ValueError:
            n_interactions = 3
        
        self.simulate_user_interactions(user_id, n_interactions)
    
    def show_statistics(self):
        """Show dataset statistics."""
        print("\\n--- Dataset Statistics ---")
        
        if self.articles_df is not None:
            print(f"\\nArticles Dataset:")
            print(f"- Total articles: {len(self.articles_df)}")
            print(f"- Categories: {', '.join(self.articles_df['category'].unique())}")
            print(f"\\nCategory Distribution:")
            category_counts = self.articles_df['category'].value_counts()
            for category, count in category_counts.items():
                print(f"- {category}: {count}")
        
        if self.interactions_df is not None:
            print(f"\\nUser Interactions Dataset:")
            print(f"- Total interactions: {len(self.interactions_df)}")
            print(f"- Unique users: {self.interactions_df['user_id'].nunique()}")
            print(f"- Average rating: {self.interactions_df['rating'].mean():.2f}")

def main():
    """Main function with command line interface."""
    parser = argparse.ArgumentParser(description="Smart News AI - Intelligent News Classification and Recommendation")
    parser.add_argument('--data-path', default='data', help='Path to data directory')
    parser.add_argument('--setup-data', action='store_true', help='Generate sample data')
    parser.add_argument('--train-classifier', choices=['random_forest', 'logistic_regression', 'naive_bayes', 'svm'], help='Train classifier with specified model')
    parser.add_argument('--compare-models', action='store_true', help='Compare different classification models')
    parser.add_argument('--use-cv', action='store_true', help='Use cross-validation in model comparison')
    parser.add_argument('--cv-folds', type=int, default=5, help='Number of folds for cross-validation (default: 5)')
    parser.add_argument('--train-recommender', action='store_true', help='Train recommendation system')
    parser.add_argument('--demo', action='store_true', help='Run interactive demo')
    parser.add_argument('--classify', type=str, help='Classify given text')
    
    args = parser.parse_args()
    
    # Initialize application
    app = SmartNewsAI(data_path=args.data_path)
    
    try:
        # Setup data
        app.setup_data(regenerate=args.setup_data)
        
        # Train classifier if requested
        if args.train_classifier:
            app.train_classifier(model_type=args.train_classifier)
        
        # Compare models if requested
        if args.compare_models:
            app.compare_models(use_cross_validation=args.use_cv, cv_folds=args.cv_folds)
        
        # Train recommender if requested
        if args.train_recommender:
            app.train_recommender()
        
        # Classify text if provided
        if args.classify:
            if app.classifier is None:
                print("Training classifier first...")
                app.train_classifier()
            
            result = app.classify_article(args.classify)
            if result:
                category, predictions = result
                print(f"Predicted category: {category}")
        
        # Run interactive demo if requested
        if args.demo:
            # Ensure models are trained
            if app.classifier is None:
                print("Training classifier for demo...")
                app.train_classifier()
            
            if app.recommender is None:
                print("Training recommender for demo...")
                app.train_recommender()
            
            app.interactive_demo()
        
        # If no specific action requested, show help
        if not any([args.setup_data, args.train_classifier, args.compare_models, 
                   args.train_recommender, args.demo, args.classify]):
            print("Smart News AI - Intelligent News Classification and Recommendation System")
            print("\\nQuick start:")
            print("  python main.py --setup-data --demo")
            print("\\nFor help: python main.py --help")
    
    except KeyboardInterrupt:
        print("\\n\\nOperation cancelled by user.")
        logger.info("Operation cancelled by user")
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}", exc_info=True)
        print(f"An error occurred: {str(e)}")
        print("Check the log files in 'logs/' directory for details.")

if __name__ == "__main__":
    main()