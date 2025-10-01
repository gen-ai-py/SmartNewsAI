"""
Data preprocessing utilities for news articles.
Handles text cleaning, tokenization, and feature extraction.
"""

import re
import nltk
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import LabelEncoder
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import logging

# Download required NLTK data
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
except:
    logging.warning("Failed to download NLTK data. Some features may not work.")

class TextPreprocessor:
    """Text preprocessing pipeline for news articles."""
    
    def __init__(self, augment_short_text=True):
        self.stemmer = PorterStemmer()
        self.augment_short_text = augment_short_text
        try:
            self.stop_words = set(stopwords.words('english'))
        except:
            self.stop_words = set()
            logging.warning("Stopwords not available. Using empty set.")
    
    def clean_text(self, text):
        """Clean and normalize text data."""
        if not isinstance(text, str):
            return ""
        
        # Handle UTF-8 encoding errors
        try:
            text = text.encode('utf-8', errors='ignore').decode('utf-8', errors='ignore')
        except (UnicodeDecodeError, UnicodeEncodeError, AttributeError):
            text = str(text)
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def tokenize_and_stem(self, text):
        """Tokenize text and apply stemming."""
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
    
    def preprocess_pipeline(self, text):
        """Complete preprocessing pipeline."""
        cleaned_text = self.clean_text(text)
        
        # Augment short text by keeping original + cleaned version
        if self.augment_short_text and len(cleaned_text) < 100:
            tokens = self.tokenize_and_stem(cleaned_text)
            # For short text, combine original words with stemmed tokens
            augmented = cleaned_text + ' ' + ' '.join(tokens)
            return augmented
        
        tokens = self.tokenize_and_stem(cleaned_text)
        return ' '.join(tokens)

class FeatureExtractor:
    """Feature extraction for news articles."""
    
    def __init__(self, max_features=5000, use_tfidf=True, min_df=1):
        self.max_features = max_features
        self.use_tfidf = use_tfidf
        self.min_df = min_df  # Lower min_df for short text
        self.vectorizer = None
        self.label_encoder = LabelEncoder()
        self.preprocessor = TextPreprocessor()
    
    def fit_transform_text(self, texts):
        """Fit vectorizer and transform texts to feature vectors."""
        # Preprocess texts
        processed_texts = [self.preprocessor.preprocess_pipeline(text) for text in texts]
        
        if self.use_tfidf:
            # Use character n-grams for short text handling
            self.vectorizer = TfidfVectorizer(
                max_features=self.max_features,
                analyzer='char_wb',  # Character word boundary n-grams
                ngram_range=(2, 4),  # Character-level 2-4 grams capture short text features
                min_df=self.min_df,  # Lower threshold for short texts
                max_df=0.8,
                sublinear_tf=True  # Log scaling helps with short text
            )
        else:
            self.vectorizer = CountVectorizer(
                max_features=self.max_features,
                ngram_range=(1, 3),
                min_df=self.min_df,
                max_df=0.8
            )
        
        features = self.vectorizer.fit_transform(processed_texts)
        return features
    
    def transform_text(self, texts):
        """Transform texts using fitted vectorizer."""
        if self.vectorizer is None:
            raise ValueError("Vectorizer not fitted. Call fit_transform_text first.")
        
        processed_texts = [self.preprocessor.preprocess_pipeline(text) for text in texts]
        return self.vectorizer.transform(processed_texts)
    
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
        if self.vectorizer is None:
            return []
        return self.vectorizer.get_feature_names_out()

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
        logging.error(f"Error loading data: {str(e)}")
        raise

def create_train_test_split(X, y, test_size=0.2, random_state=42):
    """Create train-test split for the data."""
    from sklearn.model_selection import train_test_split
    
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)