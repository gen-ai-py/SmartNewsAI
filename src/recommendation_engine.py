"""
Recommendation engine for personalized news recommendations.
Implements collaborative filtering and content-based filtering approaches.
"""

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import logging
from datetime import datetime, timedelta
import os

class UserProfile:
    """User profile management for personalized recommendations."""
    
    def __init__(self, user_id):
        self.user_id = user_id
        self.reading_history = []
        self.preferences = {}
        self.category_weights = {}
        self.last_updated = datetime.now()
    
    def add_interaction(self, article_id, category, rating, timestamp=None):
        """Add user interaction with an article."""
        if timestamp is None:
            timestamp = datetime.now()
        
        interaction = {
            'article_id': article_id,
            'category': category,
            'rating': rating,
            'timestamp': timestamp
        }
        
        self.reading_history.append(interaction)
        self._update_preferences()
        self.last_updated = datetime.now()
        
        # Limit reading history size to prevent memory leak
        max_history_size = 1000
        if len(self.reading_history) > max_history_size:
            self.reading_history = self.reading_history[-max_history_size:]
    
    def _update_preferences(self):
        """Update user preferences based on reading history."""
        if not self.reading_history:
            return
        
        # Calculate category weights based on ratings and recency
        category_ratings = {}
        for interaction in self.reading_history:
            category = interaction['category']
            rating = interaction['rating']
            
            # Apply time decay to older interactions
            days_ago = (datetime.now() - interaction['timestamp']).days
            time_weight = max(0.1, 1.0 - (days_ago / 365))  # Decay over a year
            
            weighted_rating = rating * time_weight
            
            if category not in category_ratings:
                category_ratings[category] = []
            category_ratings[category].append(weighted_rating)
        
        # Calculate average weighted ratings per category
        for category, ratings in category_ratings.items():
            self.category_weights[category] = np.mean(ratings)
    
    def get_preferred_categories(self, top_n=5):
        """Get user's top preferred categories."""
        if not self.category_weights:
            return []
        
        sorted_categories = sorted(
            self.category_weights.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        return [cat for cat, weight in sorted_categories[:top_n]]

class ContentBasedRecommender:
    """Content-based recommendation using article similarity."""
    
    def __init__(self, max_features=5000):
        self.max_features = max_features
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.8
        )
        self.article_features = None
        self.article_ids = None
        self.is_fitted = False
    
    def fit(self, articles_df):
        """
        Fit the content-based recommender.
        
        Args:
            articles_df: DataFrame with columns ['id', 'content', 'category', 'title']
        """
        logging.info("Fitting content-based recommender...")
        
        # Combine title and content for better features
        combined_text = articles_df['title'].fillna('') + ' ' + articles_df['content'].fillna('')
        
        # Extract TF-IDF features
        self.article_features = self.vectorizer.fit_transform(combined_text)
        self.article_ids = articles_df['id'].values
        self.articles_df = articles_df.copy()
        self.is_fitted = True
        
        logging.info(f"Content-based recommender fitted with {len(articles_df)} articles")
    
    def get_similar_articles(self, article_id, n_recommendations=10):
        """Get articles similar to the given article."""
        if not self.is_fitted:
            raise ValueError("Recommender must be fitted first")
        
        # Find article index
        try:
            article_idx = np.where(self.article_ids == article_id)[0][0]
        except IndexError:
            logging.warning(f"Article {article_id} not found")
            return []
        
        # Calculate cosine similarity
        article_vector = self.article_features[article_idx]
        similarities = cosine_similarity(article_vector, self.article_features).flatten()
        
        # Get top similar articles (excluding the article itself)
        similar_indices = np.argsort(similarities)[::-1][1:n_recommendations+1]
        
        recommendations = []
        for idx in similar_indices:
            recommendations.append({
                'article_id': self.article_ids[idx],
                'similarity_score': similarities[idx],
                'title': self.articles_df.iloc[idx]['title'],
                'category': self.articles_df.iloc[idx]['category']
            })
        
        return recommendations
    
    def recommend_by_category_preference(self, user_profile, n_recommendations=10):
        """Recommend articles based on user's category preferences."""
        if not self.is_fitted:
            raise ValueError("Recommender must be fitted first")
        
        preferred_categories = user_profile.get_preferred_categories()
        if not preferred_categories:
            # Return random articles if no preferences
            return self._get_random_articles(n_recommendations)
        
        # Filter articles by preferred categories
        category_mask = self.articles_df['category'].isin(preferred_categories)
        filtered_articles = self.articles_df[category_mask]
        
        if len(filtered_articles) == 0:
            return self._get_random_articles(n_recommendations)
        
        # Score articles based on category preference
        recommendations = []
        for _, article in filtered_articles.iterrows():
            category = article['category']
            preference_score = user_profile.category_weights.get(category, 0.5)
            
            recommendations.append({
                'article_id': article['id'],
                'preference_score': preference_score,
                'title': article['title'],
                'category': article['category']
            })
        
        # Sort by preference score and return top N
        recommendations.sort(key=lambda x: x['preference_score'], reverse=True)
        return recommendations[:n_recommendations]
    
    def _get_random_articles(self, n_recommendations):
        """Get random articles as fallback."""
        if len(self.articles_df) < n_recommendations:
            sample_size = len(self.articles_df)
        else:
            sample_size = n_recommendations
        
        random_articles = self.articles_df.sample(n=sample_size)
        
        recommendations = []
        for _, article in random_articles.iterrows():
            recommendations.append({
                'article_id': article['id'],
                'preference_score': 0.5,
                'title': article['title'],
                'category': article['category']
            })
        
        return recommendations

class CollaborativeFilteringRecommender:
    """Collaborative filtering using matrix factorization."""
    
    def __init__(self, n_components=50):
        self.n_components = n_components
        self.svd_model = TruncatedSVD(n_components=n_components, random_state=42)
        self.user_item_matrix = None
        self.user_ids = None
        self.article_ids = None
        self.is_fitted = False
    
    def fit(self, interactions_df):
        """
        Fit collaborative filtering model.
        
        Args:
            interactions_df: DataFrame with columns ['user_id', 'article_id', 'rating']
        """
        logging.info("Fitting collaborative filtering recommender...")
        
        # Create user-item matrix
        self.user_item_matrix = interactions_df.pivot(
            index='user_id',
            columns='article_id',
            values='rating'
        ).fillna(0)
        
        self.user_ids = self.user_item_matrix.index.values
        self.article_ids = self.user_item_matrix.columns.values
        
        # Fit SVD model
        self.user_factors = self.svd_model.fit_transform(self.user_item_matrix)
        self.item_factors = self.svd_model.components_.T
        
        self.is_fitted = True
        logging.info(f"Collaborative filtering fitted with {len(self.user_ids)} users and {len(self.article_ids)} articles")
    
    def predict_rating(self, user_id, article_id):
        """Predict rating for user-article pair."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        
        try:
            user_idx = np.where(self.user_ids == user_id)[0][0]
            article_idx = np.where(self.article_ids == article_id)[0][0]
            
            # Predict rating using matrix factorization
            prediction = np.dot(self.user_factors[user_idx], self.item_factors[article_idx])
            return max(0, min(5, prediction))  # Clamp between 0 and 5
            
        except IndexError:
            return 2.5  # Return neutral rating for unknown users/articles
    
    def get_user_recommendations(self, user_id, n_recommendations=10):
        """Get recommendations for a specific user."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        
        try:
            user_idx = np.where(self.user_ids == user_id)[0][0]
        except IndexError:
            logging.warning(f"User {user_id} not found")
            return []
        
        # Get user's predicted ratings for all articles
        user_ratings = np.dot(self.user_factors[user_idx], self.item_factors.T)
        
        # Get articles not yet rated by the user
        rated_articles = set(self.user_item_matrix.loc[user_id][self.user_item_matrix.loc[user_id] > 0].index)
        
        recommendations = []
        for i, article_id in enumerate(self.article_ids):
            if article_id not in rated_articles:
                recommendations.append({
                    'article_id': article_id,
                    'predicted_rating': user_ratings[i]
                })
        
        # Sort by predicted rating
        recommendations.sort(key=lambda x: x['predicted_rating'], reverse=True)
        return recommendations[:n_recommendations]

class HybridRecommender:
    """Hybrid recommendation system combining content-based and collaborative filtering."""
    
    def __init__(self, content_weight=0.6, collaborative_weight=0.4, max_user_profiles=10000):
        self.content_recommender = ContentBasedRecommender()
        self.collaborative_recommender = CollaborativeFilteringRecommender()
        self.content_weight = content_weight
        self.collaborative_weight = collaborative_weight
        self.user_profiles = {}
        self.max_user_profiles = max_user_profiles
    
    def fit(self, articles_df, interactions_df=None):
        """Fit both recommendation models."""
        logging.info("Fitting hybrid recommender...")
        
        # Fit content-based recommender
        self.content_recommender.fit(articles_df)
        
        # Fit collaborative filtering if interaction data is available
        if interactions_df is not None and len(interactions_df) > 0:
            self.collaborative_recommender.fit(interactions_df)
            self.has_collaborative = True
        else:
            self.has_collaborative = False
            logging.warning("No interaction data provided. Using content-based only.")
    
    def get_user_profile(self, user_id):
        """Get or create user profile."""
        if user_id not in self.user_profiles:
            # Check if we need to clean up old profiles
            if len(self.user_profiles) >= self.max_user_profiles:
                self._cleanup_old_profiles()
            
            self.user_profiles[user_id] = UserProfile(user_id)
        return self.user_profiles[user_id]
    
    def _cleanup_old_profiles(self):
        """Remove least recently used profiles to prevent memory leak."""
        # Sort profiles by last_updated timestamp
        sorted_profiles = sorted(
            self.user_profiles.items(),
            key=lambda x: x[1].last_updated
        )
        
        # Remove oldest 20% of profiles
        num_to_remove = int(len(sorted_profiles) * 0.2)
        for user_id, _ in sorted_profiles[:num_to_remove]:
            del self.user_profiles[user_id]
        
        logging.info(f"Cleaned up {num_to_remove} old user profiles. Current profiles: {len(self.user_profiles)}")
    
    def add_user_interaction(self, user_id, article_id, category, rating):
        """Add user interaction and update profile."""
        user_profile = self.get_user_profile(user_id)
        user_profile.add_interaction(article_id, category, rating)
    
    def get_recommendations(self, user_id, n_recommendations=10, batch_mode=False):
        """
        Get hybrid recommendations for user.
        
        Args:
            user_id: User ID
            n_recommendations: Number of recommendations to return
            batch_mode: If True, optimizes for batch processing by limiting memory usage
        """
        user_profile = self.get_user_profile(user_id)
        
        # Track already recommended articles to prevent duplicates
        seen_articles = set()
        
        # Get content-based recommendations
        content_recs = self.content_recommender.recommend_by_category_preference(
            user_profile, 
            n_recommendations * 3  # Get more to ensure enough unique articles after deduplication
        )
        
        hybrid_scores = {}
        
        # Score content-based recommendations (avoid duplicates)
        for rec in content_recs:
            article_id = rec['article_id']
            
            # Skip if already seen
            if article_id in seen_articles:
                continue
            
            seen_articles.add(article_id)
            content_score = rec['preference_score']
            hybrid_scores[article_id] = {
                'content_score': content_score,
                'collaborative_score': 0.0,
                'title': rec['title'],
                'category': rec['category']
            }
        
        # Add collaborative filtering scores if available
        if self.has_collaborative:
            collab_recs = self.collaborative_recommender.get_user_recommendations(
                user_id, 
                n_recommendations * 2
            )
            
            for rec in collab_recs:
                article_id = rec['article_id']
                
                # Skip if already seen
                if article_id in seen_articles:
                    continue
                
                collab_score = (rec['predicted_rating'] - 2.5) / 2.5  # Normalize to 0-1
                
                if article_id in hybrid_scores:
                    hybrid_scores[article_id]['collaborative_score'] = collab_score
                else:
                    # Add new recommendation from collaborative filtering
                    try:
                        article_info = self.content_recommender.articles_df[
                            self.content_recommender.articles_df['id'] == article_id
                        ].iloc[0]
                        
                        seen_articles.add(article_id)
                        hybrid_scores[article_id] = {
                            'content_score': 0.0,
                            'collaborative_score': collab_score,
                            'title': article_info['title'],
                            'category': article_info['category']
                        }
                    except:
                        continue
        
        # Calculate hybrid scores (ensure no duplicates in final list)
        final_recommendations = []
        final_article_ids = set()
        
        for article_id, scores in hybrid_scores.items():
            # Double-check for duplicates before adding to final list
            if article_id in final_article_ids:
                continue
            
            final_article_ids.add(article_id)
            hybrid_score = (
                scores['content_score'] * self.content_weight +
                scores['collaborative_score'] * self.collaborative_weight
            )
            
            final_recommendations.append({
                'article_id': article_id,
                'hybrid_score': hybrid_score,
                'content_score': scores['content_score'],
                'collaborative_score': scores['collaborative_score'],
                'title': scores['title'],
                'category': scores['category']
            })
        
        # Sort by hybrid score and return top N
        final_recommendations.sort(key=lambda x: x['hybrid_score'], reverse=True)
        
        # Clean up temporary data in batch mode to prevent memory leak
        if batch_mode:
            del content_recs
            del hybrid_scores
            if self.has_collaborative:
                del collab_recs
        
        return final_recommendations[:n_recommendations]
    
    def clear_user_profiles(self):
        """Clear all user profiles to free memory."""
        self.user_profiles.clear()
        logging.info("All user profiles cleared")
    
    def get_profile_memory_usage(self):
        """Get approximate memory usage of user profiles."""
        import sys
        total_size = sys.getsizeof(self.user_profiles)
        for user_id, profile in self.user_profiles.items():
            total_size += sys.getsizeof(user_id)
            total_size += sys.getsizeof(profile.reading_history)
            total_size += sys.getsizeof(profile.category_weights)
        return total_size / (1024 * 1024)  # Return in MB
    
    def save_model(self, filepath, include_user_profiles=True):
        """Save the hybrid recommender model."""
        model_data = {
            'content_recommender': self.content_recommender,
            'collaborative_recommender': self.collaborative_recommender,
            'user_profiles': self.user_profiles if include_user_profiles else {},
            'content_weight': self.content_weight,
            'collaborative_weight': self.collaborative_weight,
            'has_collaborative': self.has_collaborative,
            'max_user_profiles': self.max_user_profiles
        }
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(model_data, filepath)
        logging.info(f"Hybrid recommender saved to {filepath}")
    
    def load_model(self, filepath):
        """Load the hybrid recommender model."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        model_data = joblib.load(filepath)
        
        self.content_recommender = model_data['content_recommender']
        self.collaborative_recommender = model_data['collaborative_recommender']
        self.user_profiles = model_data.get('user_profiles', {})
        self.content_weight = model_data['content_weight']
        self.collaborative_weight = model_data['collaborative_weight']
        self.has_collaborative = model_data['has_collaborative']
        self.max_user_profiles = model_data.get('max_user_profiles', 10000)
        
        logging.info(f"Hybrid recommender loaded from {filepath}")
        logging.info(f"Loaded {len(self.user_profiles)} user profiles")

def generate_sample_interactions(articles_df, n_users=100, n_interactions=1000):
    """Generate sample user interactions for testing."""
    np.random.seed(42)
    
    interactions = []
    user_ids = [f"user_{i}" for i in range(n_users)]
    
    for _ in range(n_interactions):
        user_id = np.random.choice(user_ids)
        article = articles_df.sample(1).iloc[0]
        
        # Generate rating based on some logic (higher ratings for certain categories)
        base_rating = np.random.normal(3, 1)
        if article['category'] in ['technology', 'science']:
            base_rating += 0.5
        elif article['category'] in ['politics']:
            base_rating -= 0.2
        
        rating = max(1, min(5, int(base_rating)))
        
        interactions.append({
            'user_id': user_id,
            'article_id': article['id'],
            'rating': rating,
            'timestamp': datetime.now() - timedelta(days=np.random.randint(0, 365))
        })
    
    return pd.DataFrame(interactions)