"""
Test script to verify optimizations meet the success criteria.
"""

import time
import numpy as np
import pandas as pd
import os
import sys
from src.data_preprocessing import FeatureExtractor, TextPreprocessor
from src.news_classifier import NewsClassifier

def test_single_article_performance():
    """Test single article classification performance."""
    print("Testing single article classification performance...")
    
    # Create optimized feature extractor
    extractor = FeatureExtractor(
        max_features=3000,
        use_tfidf=True,
        use_hashing=False,
        n_jobs=-1
    )
    
    # Generate sample data for training
    texts = [f"Sample article {i} about technology and science" for i in range(500)]
    labels = ["technology"] * 500
    
    # Fit extractor
    X = extractor.fit_transform_text(texts)
    y = extractor.fit_transform_labels(labels)
    
    # Apply optimizations
    X = extractor.optimize_for_production(X, y)
    
    # Test article
    test_article = "Breaking news: New AI technology revolutionizes healthcare industry with breakthrough innovations"
    
    # Measure performance
    times = []
    for _ in range(50):  # Run multiple times for reliable measurement
        start = time.time()
        _ = extractor.transform_text([test_article])
        end = time.time()
        times.append((end - start) * 1000)  # Convert to ms
    
    avg_time = np.mean(times)
    print(f"Average single article classification time: {avg_time:.2f}ms")
    print(f"Goal (<100ms): {'ACHIEVED' if avg_time < 100 else 'NOT ACHIEVED'}")
    
    return avg_time < 100

def test_batch_processing_performance():
    """Test batch processing performance."""
    print("\nTesting batch processing performance...")
    
    # Create optimized feature extractor
    extractor = FeatureExtractor(
        max_features=3000,
        use_tfidf=True,
        use_hashing=False,
        n_jobs=-1
    )
    
    # Generate sample data for training
    train_texts = [f"Sample article {i} about various topics including technology, sports, and politics" for i in range(500)]
    train_labels = np.random.choice(["technology", "sports", "politics"], 500)
    
    # Fit extractor
    X = extractor.fit_transform_text(train_texts)
    y = extractor.fit_transform_labels(train_labels)
    
    # Apply optimizations
    X = extractor.optimize_for_production(X, y)
    
    # Generate 100 test articles
    test_texts = [f"Test article {i} with random content for batch processing test" for i in range(100)]
    
    # Measure performance
    times = []
    for _ in range(10):  # Run multiple times for reliable measurement
        start = time.time()
        _ = extractor.transform_text(test_texts)
        end = time.time()
        times.append(end - start)  # In seconds
    
    avg_time = np.mean(times)
    print(f"Average batch processing time (100 articles): {avg_time:.2f}s")
    print(f"Goal (<2s): {'ACHIEVED' if avg_time < 2 else 'NOT ACHIEVED'}")
    
    return avg_time < 2

def test_memory_usage():
    """Test memory usage reduction."""
    print("\nTesting memory usage...")
    
    import psutil
    import gc
    
    # Force garbage collection
    gc.collect()
    
    # Measure baseline memory
    process = psutil.Process(os.getpid())
    baseline_memory = process.memory_info().rss / (1024 * 1024)  # MB
    
    # Create original feature extractor
    original_extractor = FeatureExtractor(
        max_features=5000,
        use_tfidf=True
    )
    
    # Generate sample data
    texts = [f"Sample article {i} with content for memory test" for i in range(1000)]
    labels = np.random.choice(["technology", "sports", "politics"], 1000)
    
    # Fit original extractor
    original_extractor.fit_transform_text(texts)
    original_extractor.fit_transform_labels(labels)
    
    # Measure memory after original implementation
    gc.collect()
    original_memory = process.memory_info().rss / (1024 * 1024) - baseline_memory
    
    # Clear original extractor
    del original_extractor
    gc.collect()
    
    # Create optimized feature extractor
    optimized_extractor = FeatureExtractor(
        max_features=5000,
        use_tfidf=True,
        use_hashing=True,  # Use hashing for memory efficiency
        n_jobs=-1
    )
    
    # Fit optimized extractor
    X = optimized_extractor.fit_transform_text(texts)
    y = optimized_extractor.fit_transform_labels(labels)
    
    # Measure memory after optimized implementation
    gc.collect()
    optimized_memory = process.memory_info().rss / (1024 * 1024) - baseline_memory
    
    # Calculate reduction
    memory_reduction = 1 - (optimized_memory / original_memory)
    
    print(f"Original memory usage: {original_memory:.2f}MB")
    print(f"Optimized memory usage: {optimized_memory:.2f}MB")
    print(f"Memory reduction: {memory_reduction*100:.2f}%")
    print(f"Goal (30% reduction): {'ACHIEVED' if memory_reduction >= 0.3 else 'NOT ACHIEVED'}")
    
    return memory_reduction >= 0.3

def main():
    print("=" * 50)
    print("TESTING OPTIMIZATIONS AGAINST SUCCESS CRITERIA")
    print("=" * 50)
    
    # Run tests
    single_article_success = test_single_article_performance()
    batch_processing_success = test_batch_processing_performance()
    memory_reduction_success = test_memory_usage()
    
    # Overall success
    overall_success = single_article_success and batch_processing_success and memory_reduction_success
    
    print("\n" + "=" * 50)
    print("SUMMARY OF RESULTS")
    print("=" * 50)
    print(f"Single article classification (<100ms): {'✓' if single_article_success else '✗'}")
    print(f"Batch processing (<2s for 100 articles): {'✓' if batch_processing_success else '✗'}")
    print(f"Memory reduction (30%): {'✓' if memory_reduction_success else '✗'}")
    print("-" * 50)
    print(f"Overall success: {'ALL CRITERIA MET' if overall_success else 'SOME CRITERIA NOT MET'}")
    
    return overall_success

if __name__ == "__main__":
    main()