"""
Data preprocessing utilities for news articles.
Handles text cleaning, tokenization, and feature extraction.
Optimized for real-time performance with caching and efficient processing.
"""

import re
import nltk
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, HashingVectorizer
from sklearn.preprocessing import LabelEncoder
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import logging
import functools
from concurrent.futures import ThreadPoolExecutor
import threading
from joblib import Memory
import os

# Setup memory cache location
cache_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'cache')
os.makedirs(cache_dir, exist_ok=True)
memory = Memory(cache_dir, verbose=0)

# Import logging configuration (will use default if not already configured)
try:
    from .logging_config import get_data_logger
    logger = get_data_logger(__name__)
except ImportError:
    logger = logging.getLogger(__name__)

# Download required NLTK data
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
except:
    logger.warning("Failed to download NLTK data. Some features may not work.")

class TextPreprocessor:
    """Text preprocessing pipeline for news articles, optimized for performance."""
    
    def __init__(self, augment_short_text=True):
        self.stemmer = PorterStemmer()
        self.augment_short_text = augment_short_text
        try:
            self.stop_words = set(stopwords.words('english'))
        except:
            self.stop_words = set()
            logger.warning("Stopwords not available. Using empty set.")
            
        # Compile regex patterns for better performance
        self.url_pattern = re.compile(r'http\S+|www\S+|https\S+', flags=re.MULTILINE)
        self.email_pattern = re.compile(r'\S+@\S+')
        self.special_chars_pattern = re.compile(r'[^a-zA-Z\s]')
        self.whitespace_pattern = re.compile(r'\s+')
        
        # Thread-local storage for caching
        self.local = threading.local()
        
        # LRU cache for frequently processed texts
        self.cache_lock = threading.Lock()
        self.MAX_CACHE_SIZE = 1000
    
    @functools.lru_cache(maxsize=1024)
    def clean_text(self, text):
        """Clean and normalize text data with caching."""
        if not isinstance(text, str):
            return ""
        
        # Handle UTF-8 encoding errors
        try:
            text = text.encode('utf-8', errors='ignore').decode('utf-8', errors='ignore')
        except (UnicodeDecodeError, UnicodeEncodeError, AttributeError):
            text = str(text)
        
        # Convert to lowercase
        text = text.lower()
        
        # Apply regex substitutions with compiled patterns
        text = self.url_pattern.sub('', text)
        text = self.email_pattern.sub('', text)
        text = self.special_chars_pattern.sub('', text)
        text = self.whitespace_pattern.sub(' ', text).strip()
        
        return text
    
    @functools.lru_cache(maxsize=2048)
    def tokenize_and_stem(self, text):
        """Tokenize text and apply stemming with caching."""
        try:
            tokens = word_tokenize(text)
        except:
            tokens = text.split()
        
        # Remove stopwords and apply stemming
        filtered_tokens = [
            self.stemmer.stem(token) for token in tokens 
            if token not in self.stop_words and len(token) > 2
        ]
        
        return filtered_tokens
    
    @memory.cache
    def preprocess_pipeline(self, text):
        """Complete preprocessing pipeline with caching."""
        cleaned_text = self.clean_text(text)
        
        # Augment short text by keeping original + cleaned version
        if self.augment_short_text and len(cleaned_text) < 100:
            tokens = self.tokenize_and_stem(cleaned_text)
            # For short text, combine original words with stemmed tokens
            augmented = cleaned_text + ' ' + ' '.join(tokens)
            return augmented
        
        tokens = self.tokenize_and_stem(cleaned_text)
        return ' '.join(tokens)
        
    def batch_preprocess(self, texts, n_jobs=-1):
        """Process multiple texts in parallel."""
        if not texts:
            return []
            
        # Use ThreadPoolExecutor for parallel processing
        with ThreadPoolExecutor(max_workers=n_jobs if n_jobs > 0 else None) as executor:
            processed_texts = list(executor.map(self.preprocess_pipeline, texts))
            
        return processed_texts

class FeatureExtractor:
    """Feature extraction for news articles, optimized for real-time performance."""
    
    def __init__(self, max_features=5000, use_tfidf=True, min_df=1, use_hashing=False, n_jobs=-1):
        self.max_features = max_features
        self.use_tfidf = use_tfidf
        self.use_hashing = use_hashing
        self.min_df = min_df
        self.n_jobs = n_jobs
        self.vectorizer = None
        self.label_encoder = LabelEncoder()
        self.preprocessor = TextPreprocessor()
        self._vocab_size = 0
        self.feature_names = None
        self.feature_selector = None
        self.transform_cache = {}
        self.cache_lock = threading.Lock()
        self.MAX_CACHE_SIZE = 500
        
    @memory.cache
    def fit_transform_text(self, texts):
        """Fit vectorizer and transform texts to feature vectors with caching."""
        logger.info(f"Fitting vectorizer on {len(texts)} texts")
        
        # Preprocess texts in parallel
        processed_texts = self.preprocessor.batch_preprocess(texts, self.n_jobs)
        
        if self.use_hashing:
            # Use HashingVectorizer for memory efficiency
            self.vectorizer = HashingVectorizer(
                n_features=self.max_features,
                analyzer='word',
                ngram_range=(1, 2),
                norm='l2',
                alternate_sign=False  # Ensures non-negative features
            )
            logger.info("Using HashingVectorizer for memory efficiency")
        elif self.use_tfidf:
            self.vectorizer = TfidfVectorizer(
                max_features=self.max_features,
                analyzer='word',  # Changed to word for better performance
                ngram_range=(1, 2),  # Reduced n-gram range for speed
                min_df=self.min_df,
                max_df=0.8,
                sublinear_tf=True,
                use_idf=True,
                norm='l2',
                smooth_idf=True
            )
        else:
            self.vectorizer = CountVectorizer(
                max_features=self.max_features,
                ngram_range=(1, 2),
                min_df=self.min_df,
                max_df=0.8
            )
        
        # Transform texts to feature vectors
        features = self.vectorizer.fit_transform(processed_texts)
        
        # Store vocabulary size if available
        if not self.use_hashing and hasattr(self.vectorizer, 'vocabulary_'):
            self._vocab_size = len(self.vectorizer.vocabulary_)
            self.feature_names = self.vectorizer.get_feature_names_out()
            logger.info(f"Vocabulary size: {self._vocab_size}")
        
        return features

    @functools.lru_cache(maxsize=100)
    def transform_text(self, texts):
        """Transform texts using fitted vectorizer with caching for single texts."""
        if self.vectorizer is None:
            raise ValueError("Vectorizer not fitted. Call fit_transform_text first.")
        
        # For single text, use cache
        if len(texts) == 1:
            text = texts[0]
            text_hash = hash(text)
            
            with self.cache_lock:
                if text_hash in self.transform_cache:
                    return self.transform_cache[text_hash]
        
        # Process texts in parallel for batch processing
        processed_texts = self.preprocessor.batch_preprocess(texts, self.n_jobs)
        features = self.vectorizer.transform(processed_texts)
        
        # Cache result for single text queries
        if len(texts) == 1:
            with self.cache_lock:
                if len(self.transform_cache) >= self.MAX_CACHE_SIZE:
                    # Remove a random item if cache is full
                    self.transform_cache.pop(next(iter(self.transform_cache)))
                self.transform_cache[text_hash] = features
        
        return features
    
    def fit_transform_labels(self, labels):
        """Fit label encoder and transform labels."""
        return self.label_encoder.fit_transform(labels)
    
    def transform_labels(self, labels):
        """Transform labels using fitted encoder."""
        return self.label_encoder.transform(labels)
    
    def inverse_transform_labels(self, encoded_labels):
        """Convert encoded labels back to original labels."""
        return self.label_encoder.inverse_transform(encoded_labels)
    
    def get_feature_names(self):
        """Get feature names from vectorizer."""
        if self.vectorizer is None or self.use_hashing:
            return []
        return self.vectorizer.get_feature_names_out()
    
    def select_features(self, X, y, k=None):
        """
        Select top k features using chi-squared test.
        This reduces dimensionality and improves performance.
        """
        from sklearn.feature_selection import SelectKBest, chi2
        
        if k is None:
            # Default to 80% of features if not specified
            k = min(int(self.max_features * 0.8), X.shape[1])
        
        logger.info(f"Selecting top {k} features from {X.shape[1]} total features")
        
        # Initialize feature selector
        self.feature_selector = SelectKBest(chi2, k=k)
        
        # Fit and transform
        X_new = self.feature_selector.fit_transform(X, y)
        
        # Update feature names if using a regular vectorizer
        if not self.use_hashing and hasattr(self.vectorizer, 'get_feature_names_out'):
            mask = self.feature_selector.get_support()
            self.feature_names = self.vectorizer.get_feature_names_out()[mask]
        
        logger.info(f"Reduced features from {X.shape[1]} to {X_new.shape[1]}")
        return X_new
    
    def prune_vocabulary(self, min_frequency=2):
        """
        Prune vocabulary to reduce memory footprint.
        Only works with CountVectorizer and TfidfVectorizer.
        """
        if self.use_hashing or not hasattr(self.vectorizer, 'vocabulary_'):
            logger.warning("Vocabulary pruning not supported with HashingVectorizer")
            return
        
        # Get document frequency for each term
        df = np.bincount(self.vectorizer.vocabulary_.values())
        
        # Identify terms to keep (those with frequency >= min_frequency)
        terms_to_keep = {term: idx for term, idx in self.vectorizer.vocabulary_.items() 
                         if df[idx] >= min_frequency}
        
        # Update vocabulary
        self.vectorizer.vocabulary_ = terms_to_keep
        self._vocab_size = len(terms_to_keep)
        
        logger.info(f"Pruned vocabulary to {self._vocab_size} terms")
        
    def optimize_for_production(self, X, y):
        """
        Apply multiple optimizations for production deployment:
        1. Feature selection to reduce dimensionality
        2. Vocabulary pruning to reduce memory usage
        """
        # First apply feature selection
        X_optimized = self.select_features(X, y)
        
        # Then prune vocabulary if not using hashing vectorizer
        if not self.use_hashing:
            self.prune_vocabulary()
            
        return X_optimized

def load_and_preprocess_data(data_path, text_column='content', label_column='category'):
    """Load and preprocess news data from CSV file."""
    try:
        # Load with UTF-8 encoding and error handling
        df = pd.read_csv(data_path, encoding='utf-8', encoding_errors='ignore')
        
        # Basic data validation
        if text_column not in df.columns or label_column not in df.columns:
            raise ValueError(f"Required columns {text_column} or {label_column} not found")
        
        # Remove rows with missing values
        df = df.dropna(subset=[text_column, label_column])
        
        # Initialize feature extractor
        feature_extractor = FeatureExtractor()
        
        # Extract features
        X = feature_extractor.fit_transform_text(df[text_column].tolist())
        y = feature_extractor.fit_transform_labels(df[label_column].tolist())
        
        return X, y, feature_extractor, df
    
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        raise

def create_train_test_split(X, y, test_size=0.2, random_state=42):
    """Create train-test split for the data."""
    from sklearn.model_selection import train_test_split
    
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)