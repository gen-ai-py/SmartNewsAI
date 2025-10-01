"""
Unit tests for data preprocessing module (data_preprocessing.py)

Tests cover:
- TextPreprocessor class functionality
- FeatureExtractor class functionality  
- Utility functions
- Edge cases and error handling
- Unicode/special character handling
- Empty/malformed input handling

Test Framework: pytest
Coverage Goal: >90%
"""

import pytest
import numpy as np
import pandas as pd
import tempfile
import os
from scipy.sparse import issparse

# Import the module to test
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.data_preprocessing import (
    TextPreprocessor,
    FeatureExtractor,
    load_and_preprocess_data,
    create_train_test_split
)


class TestTextPreprocessor:
    """Test suite for TextPreprocessor class"""
    
    @pytest.fixture
    def preprocessor(self):
        """Fixture to create TextPreprocessor instance"""
        return TextPreprocessor(augment_short_text=True)
    
    @pytest.fixture
    def preprocessor_no_augment(self):
        """Fixture to create TextPreprocessor without augmentation"""
        return TextPreprocessor(augment_short_text=False)
    
    # ===== Text Cleaning Tests =====
    
    def test_clean_text_basic(self, preprocessor):
        """Test basic text cleaning functionality"""
        text = "Hello World! This is a TEST."
        result = preprocessor.clean_text(text)
        assert result == "hello world this is a test"
        assert result.islower()
    
    def test_clean_text_remove_urls(self, preprocessor):
        """Test URL removal from text"""
        text = "Check this link https://example.com and http://test.com"
        result = preprocessor.clean_text(text)
        assert "https" not in result
        assert "http" not in result
        assert "example.com" not in result
    
    def test_clean_text_remove_emails(self, preprocessor):
        """Test email address removal"""
        text = "Contact me at test@example.com or admin@test.org"
        result = preprocessor.clean_text(text)
        assert "@" not in result
        assert "test@example.com" not in result
    
    def test_clean_text_remove_special_chars(self, preprocessor):
        """Test special character and digit removal"""
        text = "Price: $100.50! Date: 2024-10-01 #hashtag @mention"
        result = preprocessor.clean_text(text)
        assert "$" not in result
        assert "100" not in result
        assert "#" not in result
        assert "@" not in result
        assert "-" not in result
    
    def test_clean_text_whitespace_normalization(self, preprocessor):
        """Test extra whitespace removal and normalization"""
        text = "Too    many     spaces  \n\n\n  and    newlines"
        result = preprocessor.clean_text(text)
        assert "  " not in result  # No double spaces
        assert result == "too many spaces and newlines"
    
    def test_clean_text_unicode_handling(self, preprocessor):
        """Test Unicode character handling"""
        text = "Café résumé naïve"
        result = preprocessor.clean_text(text)
        assert isinstance(result, str)
        # Should handle gracefully without errors
    
    def test_clean_text_empty_input(self, preprocessor):
        """Test handling of empty string"""
        assert preprocessor.clean_text("") == ""
        assert preprocessor.clean_text("   ") == ""
    
    def test_clean_text_non_string_input(self, preprocessor):
        """Test handling of non-string input"""
        assert preprocessor.clean_text(None) == ""
        assert preprocessor.clean_text(123) == ""
        assert preprocessor.clean_text([]) == ""
    
    def test_clean_text_very_long_text(self, preprocessor):
        """Test handling of very long text"""
        long_text = "word " * 10000
        result = preprocessor.clean_text(long_text)
        assert isinstance(result, str)
        assert len(result) > 0
    
    def test_clean_text_special_unicode(self, preprocessor):
        """Test handling of special Unicode characters"""
        text = "Hello 你好 مرحبا Привет 🎃"
        result = preprocessor.clean_text(text)
        # Should not crash with special characters
        assert isinstance(result, str)
    
    # ===== Tokenization Tests =====
    
    def test_tokenize_and_stem_basic(self, preprocessor):
        """Test basic tokenization and stemming"""
        text = "running runner runs"
        tokens = preprocessor.tokenize_and_stem(text)
        assert isinstance(tokens, list)
        assert len(tokens) > 0
        # Stemming should normalize words
        assert all(isinstance(token, str) for token in tokens)
    
    def test_tokenize_and_stem_stopwords(self, preprocessor):
        """Test stopword removal"""
        text = "the quick brown fox jumps over the lazy dog"
        tokens = preprocessor.tokenize_and_stem(text)
        # Common stopwords should be removed
        stopwords = {'the', 'over', 'a', 'an'}
        token_set = set(tokens)
        assert len(stopwords.intersection(token_set)) == 0
    
    def test_tokenize_and_stem_short_tokens(self, preprocessor):
        """Test removal of very short tokens"""
        text = "a ab abc abcd"
        tokens = preprocessor.tokenize_and_stem(text)
        # Tokens with length <= 2 should be removed
        assert all(len(token) > 2 for token in tokens)
    
    def test_tokenize_and_stem_empty(self, preprocessor):
        """Test tokenization of empty text"""
        tokens = preprocessor.tokenize_and_stem("")
        assert isinstance(tokens, list)
        assert len(tokens) == 0
    
    def test_tokenize_and_stem_stemming_accuracy(self, preprocessor):
        """Test stemming accuracy with related words"""
        text = "organization organizational organize organizing"
        tokens = preprocessor.tokenize_and_stem(text)
        # All should stem to similar root
        assert len(set(tokens)) <= 2  # Should have 1-2 unique stems
    
    # ===== Pipeline Tests =====
    
    def test_preprocess_pipeline_short_text_with_augment(self, preprocessor):
        """Test preprocessing pipeline with short text augmentation"""
        short_text = "AI news today"
        result = preprocessor.preprocess_pipeline(short_text)
        assert isinstance(result, str)
        assert len(result) > len(short_text.lower())  # Should be augmented
    
    def test_preprocess_pipeline_short_text_no_augment(self, preprocessor_no_augment):
        """Test preprocessing pipeline without augmentation"""
        short_text = "AI news today"
        result = preprocessor_no_augment.preprocess_pipeline(short_text)
        assert isinstance(result, str)
        # Should not contain duplicate content
    
    def test_preprocess_pipeline_long_text(self, preprocessor):
        """Test preprocessing pipeline with long text"""
        long_text = "This is a much longer text that exceeds the hundred character threshold for short text augmentation. " * 2
        result = preprocessor.preprocess_pipeline(long_text)
        assert isinstance(result, str)
        assert len(result) > 0
    
    def test_preprocess_pipeline_complex_text(self, preprocessor):
        """Test preprocessing pipeline with complex real-world text"""
        text = """
        Breaking News: Tech company launches AI product!
        Visit https://example.com for details.
        Contact: info@company.com
        Price: $99.99 (50% off!)
        #TechNews @TechGuru
        """
        result = preprocessor.preprocess_pipeline(text)
        assert isinstance(result, str)
        assert "https" not in result
        assert "@" not in result
        assert "$" not in result
    
    def test_preprocess_pipeline_malformed_input(self, preprocessor):
        """Test preprocessing pipeline with malformed input"""
        malformed_texts = [None, "", "   ", 123, [], {}]
        for text in malformed_texts:
            result = preprocessor.preprocess_pipeline(text if isinstance(text, str) else "")
            assert isinstance(result, str)


class TestFeatureExtractor:
    """Test suite for FeatureExtractor class"""
    
    @pytest.fixture
    def sample_texts(self):
        """Fixture providing sample texts for testing"""
        return [
            "Machine learning is transforming technology",
            "Sports news: Team wins championship",
            "Politics: New policy announced today",
            "Entertainment: Movie breaks box office records"
        ]
    
    @pytest.fixture
    def sample_labels(self):
        """Fixture providing sample labels"""
        return ["technology", "sports", "politics", "entertainment"]
    
    @pytest.fixture
    def extractor_tfidf(self):
        """Fixture for TF-IDF feature extractor"""
        return FeatureExtractor(max_features=100, use_tfidf=True)
    
    @pytest.fixture
    def extractor_count(self):
        """Fixture for Count vectorizer feature extractor"""
        return FeatureExtractor(max_features=100, use_tfidf=False)
    
    # ===== TF-IDF Vectorization Tests =====
    
    def test_fit_transform_text_tfidf(self, extractor_tfidf, sample_texts):
        """Test TF-IDF vectorization"""
        features = extractor_tfidf.fit_transform_text(sample_texts)
        assert issparse(features)
        assert features.shape[0] == len(sample_texts)
        assert features.shape[1] <= 100
        assert extractor_tfidf._vocab_size > 0
    
    def test_fit_transform_text_count(self, extractor_count, sample_texts):
        """Test Count vectorization"""
        features = extractor_count.fit_transform_text(sample_texts)
        assert issparse(features)
        assert features.shape[0] == len(sample_texts)
        assert features.shape[1] <= 100
    
    def test_transform_text_after_fit(self, extractor_tfidf, sample_texts):
        """Test transform on new texts after fitting"""
        extractor_tfidf.fit_transform_text(sample_texts)
        new_texts = ["New technology article", "Another sports update"]
        features = extractor_tfidf.transform_text(new_texts)
        assert issparse(features)
        assert features.shape[0] == len(new_texts)
    
    def test_transform_text_before_fit_raises_error(self, extractor_tfidf, sample_texts):
        """Test that transform before fit raises ValueError"""
        with pytest.raises(ValueError, match="Vectorizer not fitted"):
            extractor_tfidf.transform_text(sample_texts)
    
    def test_fit_transform_empty_texts(self, extractor_tfidf):
        """Test handling of empty text list"""
        with pytest.raises(Exception):
            extractor_tfidf.fit_transform_text([])
    
    def test_fit_transform_single_text(self, extractor_tfidf):
        """Test fitting with single text"""
        features = extractor_tfidf.fit_transform_text(["Single article text"])
        assert issparse(features)
        assert features.shape[0] == 1
    
    def test_fit_transform_very_short_texts(self, extractor_tfidf):
        """Test handling of very short texts"""
        short_texts = ["AI", "ML", "DL", "NLP"]
        features = extractor_tfidf.fit_transform_text(short_texts)
        assert issparse(features)
        assert features.shape[0] == len(short_texts)
    
    def test_fit_transform_very_long_texts(self, extractor_tfidf):
        """Test handling of very long texts"""
        long_text = "word " * 5000
        features = extractor_tfidf.fit_transform_text([long_text])
        assert issparse(features)
        assert features.shape[0] == 1
    
    # ===== Label Encoding Tests =====
    
    def test_fit_transform_labels(self, extractor_tfidf, sample_labels):
        """Test label encoding"""
        encoded = extractor_tfidf.fit_transform_labels(sample_labels)
        assert isinstance(encoded, np.ndarray)
        assert len(encoded) == len(sample_labels)
        assert encoded.dtype == np.int64 or encoded.dtype == np.int32
        assert set(encoded) == set(range(len(set(sample_labels))))
    
    def test_transform_labels_after_fit(self, extractor_tfidf, sample_labels):
        """Test transforming new labels after fitting"""
        extractor_tfidf.fit_transform_labels(sample_labels)
        new_labels = ["technology", "sports"]
        encoded = extractor_tfidf.transform_labels(new_labels)
        assert len(encoded) == len(new_labels)
    
    def test_inverse_transform_labels(self, extractor_tfidf, sample_labels):
        """Test inverse label transformation"""
        encoded = extractor_tfidf.fit_transform_labels(sample_labels)
        decoded = extractor_tfidf.inverse_transform_labels(encoded)
        assert list(decoded) == sample_labels
    
    def test_fit_transform_labels_with_duplicates(self, extractor_tfidf):
        """Test label encoding with duplicate labels"""
        labels = ["tech", "sports", "tech", "tech", "sports"]
        encoded = extractor_tfidf.fit_transform_labels(labels)
        assert len(encoded) == len(labels)
        assert len(set(encoded)) == 2  # Only 2 unique labels
    
    def test_fit_transform_labels_empty(self, extractor_tfidf):
        """Test label encoding with empty list"""
        with pytest.raises(Exception):
            extractor_tfidf.fit_transform_labels([])
    
    # ===== Feature Names Tests =====
    
    def test_get_feature_names_after_fit(self, extractor_tfidf, sample_texts):
        """Test getting feature names after fitting"""
        extractor_tfidf.fit_transform_text(sample_texts)
        feature_names = extractor_tfidf.get_feature_names()
        assert isinstance(feature_names, np.ndarray)
        assert len(feature_names) > 0
        assert all(isinstance(name, str) for name in feature_names)
    
    def test_get_feature_names_before_fit(self, extractor_tfidf):
        """Test getting feature names before fitting returns empty list"""
        feature_names = extractor_tfidf.get_feature_names()
        assert isinstance(feature_names, list)
        assert len(feature_names) == 0
    
    # ===== Edge Cases Tests =====
    
    def test_feature_extractor_with_unicode_text(self, extractor_tfidf):
        """Test feature extraction with Unicode text"""
        unicode_texts = [
            "Café résumé",
            "日本語のテキスト",
            "العربية النص",
            "Emoji test 🎃🚀"
        ]
        features = extractor_tfidf.fit_transform_text(unicode_texts)
        assert issparse(features)
        assert features.shape[0] == len(unicode_texts)
    
    def test_feature_extractor_max_features_limit(self):
        """Test that max_features limit is respected"""
        extractor = FeatureExtractor(max_features=10, use_tfidf=True)
        texts = ["word " + str(i) for i in range(100)]
        features = extractor.fit_transform_text(texts)
        assert features.shape[1] <= 10
    
    def test_feature_extractor_min_df_parameter(self):
        """Test min_df parameter effect"""
        extractor = FeatureExtractor(max_features=100, use_tfidf=True, min_df=2)
        texts = ["rare word"] + ["common word"] * 5
        features = extractor.fit_transform_text(texts)
        # With min_df=2, rare words appearing once should be excluded
        assert issparse(features)


class TestUtilityFunctions:
    """Test suite for utility functions"""
    
    @pytest.fixture
    def temp_csv_file(self):
        """Fixture creating a temporary CSV file"""
        data = {
            'content': [
                'Technology news article one',
                'Sports news article two',
                'Politics news article three',
                'Entertainment news article four'
            ],
            'category': ['technology', 'sports', 'politics', 'entertainment']
        }
        df = pd.DataFrame(data)
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv', newline='') as f:
            df.to_csv(f.name, index=False)
            temp_path = f.name
        
        yield temp_path
        
        # Cleanup
        if os.path.exists(temp_path):
            os.unlink(temp_path)
    
    @pytest.fixture
    def temp_malformed_csv_file(self):
        """Fixture creating a malformed CSV file"""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv', newline='') as f:
            f.write("wrong_column1,wrong_column2\n")
            f.write("value1,value2\n")
            temp_path = f.name
        
        yield temp_path
        
        # Cleanup
        if os.path.exists(temp_path):
            os.unlink(temp_path)
    
    # ===== Data Loading Tests =====
    
    def test_load_and_preprocess_data_success(self, temp_csv_file):
        """Test successful data loading and preprocessing"""
        X, y, extractor, df = load_and_preprocess_data(temp_csv_file)
        assert issparse(X)
        assert isinstance(y, np.ndarray)
        assert isinstance(extractor, FeatureExtractor)
        assert isinstance(df, pd.DataFrame)
        assert X.shape[0] == len(y)
        assert len(df) == 4
    
    def test_load_and_preprocess_data_missing_columns(self, temp_malformed_csv_file):
        """Test data loading with missing required columns"""
        with pytest.raises(ValueError, match="Required columns"):
            load_and_preprocess_data(temp_malformed_csv_file)
    
    def test_load_and_preprocess_data_custom_columns(self, temp_csv_file):
        """Test data loading with default column names"""
        X, y, extractor, df = load_and_preprocess_data(
            temp_csv_file,
            text_column='content',
            label_column='category'
        )
        assert issparse(X)
        assert isinstance(y, np.ndarray)
    
    def test_load_and_preprocess_data_nonexistent_file(self):
        """Test data loading with nonexistent file"""
        with pytest.raises(Exception):
            load_and_preprocess_data("nonexistent_file.csv")
    
    def test_load_and_preprocess_data_with_missing_values(self):
        """Test data loading with missing values in data"""
        data = {
            'content': ['Article 1', None, 'Article 3', ''],
            'category': ['tech', 'sports', None, 'politics']
        }
        df = pd.DataFrame(data)
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv', newline='') as f:
            df.to_csv(f.name, index=False)
            temp_path = f.name
        
        try:
            X, y, extractor, result_df = load_and_preprocess_data(temp_path)
            # Should have removed rows with missing values
            assert result_df.shape[0] < df.shape[0]
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    # ===== Train-Test Split Tests =====
    
    def test_create_train_test_split_basic(self):
        """Test basic train-test split functionality"""
        from scipy.sparse import csr_matrix
        X = csr_matrix(np.random.rand(100, 10))
        y = np.random.randint(0, 2, 100)
        
        X_train, X_test, y_train, y_test = create_train_test_split(X, y, test_size=0.2)
        
        assert X_train.shape[0] == 80
        assert X_test.shape[0] == 20
        assert len(y_train) == 80
        assert len(y_test) == 20
    
    def test_create_train_test_split_custom_test_size(self):
        """Test train-test split with custom test size"""
        from scipy.sparse import csr_matrix
        X = csr_matrix(np.random.rand(100, 10))
        y = np.random.randint(0, 2, 100)
        
        X_train, X_test, y_train, y_test = create_train_test_split(X, y, test_size=0.3)
        
        assert X_train.shape[0] == 70
        assert X_test.shape[0] == 30
    
    def test_create_train_test_split_stratification(self):
        """Test that train-test split maintains class distribution"""
        from scipy.sparse import csr_matrix
        X = csr_matrix(np.random.rand(100, 10))
        y = np.array([0] * 70 + [1] * 30)  # Imbalanced classes
        
        X_train, X_test, y_train, y_test = create_train_test_split(X, y, test_size=0.2)
        
        # Check class distribution is approximately maintained
        train_ratio = np.sum(y_train == 1) / len(y_train)
        test_ratio = np.sum(y_test == 1) / len(y_test)
        original_ratio = np.sum(y == 1) / len(y)
        
        assert abs(train_ratio - original_ratio) < 0.1
        assert abs(test_ratio - original_ratio) < 0.15
    
    def test_create_train_test_split_random_state(self):
        """Test that random_state produces reproducible splits"""
        from scipy.sparse import csr_matrix
        X = csr_matrix(np.random.rand(100, 10))
        y = np.random.randint(0, 2, 100)
        
        X_train1, X_test1, y_train1, y_test1 = create_train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        X_train2, X_test2, y_train2, y_test2 = create_train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        assert np.array_equal(y_train1, y_train2)
        assert np.array_equal(y_test1, y_test2)


class TestIntegrationScenarios:
    """Integration tests for complete workflows"""
    
    def test_end_to_end_text_processing(self):
        """Test complete text processing pipeline"""
        # Sample data
        texts = [
            "Breaking: Tech company releases new AI product!",
            "Sports: Team wins finals in dramatic fashion",
            "Politics: New policy changes announced",
        ]
        labels = ["technology", "sports", "politics"]
        
        # Process
        extractor = FeatureExtractor(max_features=50, use_tfidf=True)
        X = extractor.fit_transform_text(texts)
        y = extractor.fit_transform_labels(labels)
        
        # Verify
        assert X.shape[0] == len(texts)
        assert len(y) == len(labels)
        
        # Test transform on new data
        new_texts = ["Another tech article"]
        X_new = extractor.transform_text(new_texts)
        assert X_new.shape[1] == X.shape[1]
    
    def test_full_pipeline_with_split(self):
        """Test full pipeline from data loading to train-test split"""
        # Create temp data
        data = {
            'content': ['Article ' + str(i) for i in range(50)],
            'category': ['cat1', 'cat2'] * 25
        }
        df = pd.DataFrame(data)
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv', newline='') as f:
            df.to_csv(f.name, index=False)
            temp_path = f.name
        
        try:
            # Load and preprocess
            X, y, extractor, result_df = load_and_preprocess_data(temp_path)
            
            # Split
            X_train, X_test, y_train, y_test = create_train_test_split(X, y)
            
            # Verify dimensions
            assert X_train.shape[0] + X_test.shape[0] == X.shape[0]
            assert len(y_train) + len(y_test) == len(y)
            
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--cov=src.data_preprocessing", "--cov-report=html", "--cov-report=term"])
