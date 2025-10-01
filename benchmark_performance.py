"""
Benchmarking suite for measuring feature extraction performance.
Compares original vs. optimized implementations.
"""

import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
import argparse
from joblib import Memory
from src.data_preprocessing import FeatureExtractor, TextPreprocessor
from src.data_generator import create_sample_data
from src.news_classifier import NewsClassifier

# Setup cache directory
cache_dir = os.path.join(os.path.dirname(__file__), 'cache')
os.makedirs(cache_dir, exist_ok=True)
memory = Memory(cache_dir, verbose=0)

def generate_test_data(n_samples=1000, sample_length=500):
    """Generate synthetic test data of varying lengths."""
    # Create a mix of short and long texts
    texts = []
    for i in range(n_samples):
        # Vary text length
        length = np.random.randint(50, sample_length)
        # Generate random text
        text = ' '.join(['word' + str(np.random.randint(1000)) for _ in range(length)])
        texts.append(text)
    
    # Generate random labels
    labels = np.random.choice(['technology', 'sports', 'politics', 'entertainment'], n_samples)
    
    return texts, labels

def benchmark_single_article(extractor, text, n_runs=100):
    """Benchmark single article classification time."""
    times = []
    for _ in range(n_runs):
        start = time.time()
        _ = extractor.transform_text([text])
        end = time.time()
        times.append((end - start) * 1000)  # Convert to ms
    
    return {
        'mean': np.mean(times),
        'median': np.median(times),
        'min': np.min(times),
        'max': np.max(times),
        'std': np.std(times)
    }

def benchmark_batch_processing(extractor, texts, n_runs=10):
    """Benchmark batch processing time."""
    times = []
    for _ in range(n_runs):
        start = time.time()
        _ = extractor.transform_text(texts)
        end = time.time()
        times.append(end - start)  # In seconds
    
    return {
        'mean': np.mean(times),
        'median': np.median(times),
        'min': np.min(times),
        'max': np.max(times),
        'std': np.std(times)
    }

def measure_memory_usage(extractor):
    """Measure memory usage of the feature extractor."""
    import sys
    import gc
    
    # Force garbage collection
    gc.collect()
    
    # Measure memory before
    memory_before = sys.getsizeof(extractor)
    
    # Measure vectorizer size if available
    vectorizer_size = 0
    if hasattr(extractor, 'vectorizer') and extractor.vectorizer is not None:
        vectorizer_size = sys.getsizeof(extractor.vectorizer)
        
        # Add vocabulary size if available
        if hasattr(extractor.vectorizer, 'vocabulary_'):
            vectorizer_size += sys.getsizeof(extractor.vectorizer.vocabulary_)
    
    return {
        'total': memory_before + vectorizer_size,
        'extractor': memory_before,
        'vectorizer': vectorizer_size
    }

def run_benchmarks(use_optimized=True, n_samples=1000):
    """Run all benchmarks and return results."""
    print(f"{'=' * 20} Running benchmarks {'=' * 20}")
    print(f"Configuration: {'Optimized' if use_optimized else 'Original'}, {n_samples} samples")
    
    # Generate test data
    texts, labels = generate_test_data(n_samples)
    
    # Create feature extractor
    if use_optimized:
        extractor = FeatureExtractor(
            max_features=3000,
            use_tfidf=True,
            use_hashing=False,  # Set to True for even more memory efficiency
            n_jobs=-1  # Use all available cores
        )
    else:
        # Original implementation
        extractor = FeatureExtractor(
            max_features=3000,
            use_tfidf=True
        )
    
    # Fit on training data
    print("Fitting vectorizer...")
    start = time.time()
    X = extractor.fit_transform_text(texts)
    y = extractor.fit_transform_labels(labels)
    fit_time = time.time() - start
    print(f"Fitting completed in {fit_time:.2f} seconds")
    
    # Apply optimizations if using optimized version
    if use_optimized:
        print("Applying production optimizations...")
        start = time.time()
        X = extractor.optimize_for_production(X, y)
        optimize_time = time.time() - start
        print(f"Optimization completed in {optimize_time:.2f} seconds")
    
    # Benchmark single article classification
    print("Benchmarking single article classification...")
    single_results = benchmark_single_article(extractor, texts[0])
    print(f"Single article classification: {single_results['mean']:.2f}ms (median: {single_results['median']:.2f}ms)")
    
    # Benchmark batch processing
    print("Benchmarking batch processing (100 articles)...")
    batch_size = min(100, len(texts))
    batch_results = benchmark_batch_processing(extractor, texts[:batch_size])
    print(f"Batch processing (100 articles): {batch_results['mean']:.2f}s (median: {batch_results['median']:.2f}s)")
    
    # Measure memory usage
    print("Measuring memory usage...")
    memory_results = measure_memory_usage(extractor)
    print(f"Memory usage: {memory_results['total'] / (1024*1024):.2f} MB")
    
    return {
        'single_article': single_results,
        'batch_processing': batch_results,
        'memory': memory_results,
        'fit_time': fit_time,
        'optimize_time': optimize_time if use_optimized else 0
    }

def compare_results(original, optimized):
    """Compare original and optimized results."""
    print(f"\n{'=' * 20} Performance Comparison {'=' * 20}")
    
    # Single article performance
    single_speedup = original['single_article']['mean'] / optimized['single_article']['mean']
    print(f"Single article classification:")
    print(f"  Original: {original['single_article']['mean']:.2f}ms")
    print(f"  Optimized: {optimized['single_article']['mean']:.2f}ms")
    print(f"  Speedup: {single_speedup:.2f}x ({(single_speedup-1)*100:.1f}% faster)")
    
    # Batch processing performance
    batch_speedup = original['batch_processing']['mean'] / optimized['batch_processing']['mean']
    print(f"\nBatch processing (100 articles):")
    print(f"  Original: {original['batch_processing']['mean']:.2f}s")
    print(f"  Optimized: {optimized['batch_processing']['mean']:.2f}s")
    print(f"  Speedup: {batch_speedup:.2f}x ({(batch_speedup-1)*100:.1f}% faster)")
    
    # Memory usage
    memory_reduction = 1 - (optimized['memory']['total'] / original['memory']['total'])
    print(f"\nMemory usage:")
    print(f"  Original: {original['memory']['total'] / (1024*1024):.2f} MB")
    print(f"  Optimized: {optimized['memory']['total'] / (1024*1024):.2f} MB")
    print(f"  Reduction: {memory_reduction*100:.1f}%")
    
    # Check if performance goals are met
    print("\nPerformance Goals Assessment:")
    
    # Goal 1: Single article classification < 100ms
    goal1_met = optimized['single_article']['mean'] < 100
    print(f"✓ Single article classification < 100ms: {'ACHIEVED' if goal1_met else 'NOT ACHIEVED'} ({optimized['single_article']['mean']:.2f}ms)")
    
    # Goal 2: Batch processing < 2 seconds for 100 articles
    goal2_met = optimized['batch_processing']['mean'] < 2
    print(f"✓ Batch processing < 2s for 100 articles: {'ACHIEVED' if goal2_met else 'NOT ACHIEVED'} ({optimized['batch_processing']['mean']:.2f}s)")
    
    # Goal 3: Reduced memory footprint
    goal3_met = memory_reduction >= 0.3  # 30% reduction
    print(f"✓ 30% memory reduction: {'ACHIEVED' if goal3_met else 'NOT ACHIEVED'} ({memory_reduction*100:.1f}%)")
    
    # Overall success
    overall_success = goal1_met and goal2_met and goal3_met
    print(f"\nOverall Success: {'ALL GOALS ACHIEVED' if overall_success else 'SOME GOALS NOT MET'}")
    
    return {
        'single_speedup': single_speedup,
        'batch_speedup': batch_speedup,
        'memory_reduction': memory_reduction,
        'goals_met': {
            'single_article': goal1_met,
            'batch_processing': goal2_met,
            'memory_reduction': goal3_met,
            'overall': overall_success
        }
    }

def main():
    parser = argparse.ArgumentParser(description='Benchmark feature extraction performance')
    parser.add_argument('--samples', type=int, default=1000, help='Number of samples to use')
    parser.add_argument('--compare', action='store_true', help='Run both original and optimized versions for comparison')
    parser.add_argument('--original-only', action='store_true', help='Run only the original version')
    args = parser.parse_args()
    
    if args.original_only:
        original_results = run_benchmarks(use_optimized=False, n_samples=args.samples)
    elif args.compare:
        print("Running original implementation benchmarks...")
        original_results = run_benchmarks(use_optimized=False, n_samples=args.samples)
        
        print("\nRunning optimized implementation benchmarks...")
        optimized_results = run_benchmarks(use_optimized=True, n_samples=args.samples)
        
        comparison = compare_results(original_results, optimized_results)
    else:
        # Default: run optimized version only
        optimized_results = run_benchmarks(use_optimized=True, n_samples=args.samples)

if __name__ == "__main__":
    main()