#!/usr/bin/env python3

import argparse
import re
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path
import sys

def parse_benchmark_file(file_path):
    """
    Parse a single benchmark output file and extract benchmark data.
    
    Returns:
        dict: Dictionary with benchmark name as key and list of (input_size, time_ms, cpu_ms, iterations) tuples as value
    """
    benchmarks = {}
    
    try:
        with open(file_path, 'r') as file:
            content = file.read()
    except FileNotFoundError:
        print(f"Error: File {file_path} not found.")
        return benchmarks
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return benchmarks
    
    # Find the benchmark results section
    lines = content.split('\n')
    parsing_results = False
    
    for line in lines:
        # Skip header and metadata lines
        if line.startswith('Run on') or line.startswith('Load Average') or line.startswith('2025-') or line.startswith('Running'):
            continue
        
        # Start parsing when we hit the header line
        if 'Benchmark' in line and 'Time' in line and 'CPU' in line:
            parsing_results = True
            continue
        
        # Skip the dashed line
        if parsing_results and line.startswith('---'):
            continue
        
        # Stop parsing when we hit BigO or RMS lines or empty lines
        if parsing_results and (not line.strip() or '_BigO' in line or '_RMS' in line):
            break
        
        # Parse benchmark lines
        if parsing_results and line.strip():
            # Regex pattern to match benchmark lines
            # Format: BenchmarkName/InputSize    Time    CPU    Iterations
            pattern = r'^(\w+(?:_\w+)*)/(\d+)\s+([0-9.]+)\s*(\w*)\s+([0-9.]+)\s*(\w*)\s+(\d+)'
            match = re.match(pattern, line.strip())
            
            if match:
                benchmark_name = match.group(1)
                input_size = int(match.group(2))
                time_value = float(match.group(3))
                time_unit = match.group(4) if match.group(4) else 'ms'
                cpu_value = float(match.group(5))
                cpu_unit = match.group(6) if match.group(6) else 'ms'
                iterations = int(match.group(7))
                
                # Convert time to milliseconds
                time_ms = convert_to_ms(time_value, time_unit)
                cpu_ms = convert_to_ms(cpu_value, cpu_unit)
                
                if benchmark_name not in benchmarks:
                    benchmarks[benchmark_name] = []
                
                benchmarks[benchmark_name].append((input_size, time_ms, cpu_ms, iterations))
    
    return benchmarks

def convert_to_ms(value, unit):
    """Convert time value to milliseconds based on unit."""
    unit = unit.lower()
    if unit == 'ns':
        return value / 1_000_000
    elif unit == 'us':
        return value / 1_000
    elif unit == 'ms':
        return value
    elif unit == 's':
        return value * 1_000
    else:
        # Default to ms if unit is not specified or unknown
        return value

def plot_benchmarks(benchmark_data, output_file=None):
    """
    Plot benchmark results with input size vs execution time.
    
    Args:
        benchmark_data: Dict with file names as keys and benchmark results as values
        output_file: Optional output file path for saving the plot
    """
    plt.figure(figsize=(12, 8))
    
    colors = plt.cm.Set1(np.linspace(0, 1, len(benchmark_data)))
    
    for (file_name, benchmarks), color in zip(benchmark_data.items(), colors):
        for benchmark_name, results in benchmarks.items():
            if not results:
                continue
                
            # Sort by input size
            results.sort(key=lambda x: x[0])
            
            input_sizes = [r[0] for r in results]
            times = [r[1] for r in results]  # Use wall clock time
            
            # Create label combining file name and benchmark name
            label = f"{Path(file_name).stem} - {benchmark_name}"
            
            plt.loglog(input_sizes, times, marker='o', linewidth=2, markersize=6, 
                      color=color, label=label)
    
    plt.xlabel('Input Size', fontsize=12)
    plt.ylabel('Execution Time (ms)', fontsize=12)
    plt.title('Benchmark Performance Comparison', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Adjust layout to prevent legend cutoff
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {output_file}")
    else:
        plt.show()

def plot_separate_benchmarks(benchmark_data, output_dir=None):
    """
    Create separate plots for each benchmark type.
    
    Args:
        benchmark_data: Dict with file names as keys and benchmark results as values
        output_dir: Optional directory path for saving individual plots
    """
    # Collect all unique benchmark names
    all_benchmark_names = set()
    for benchmarks in benchmark_data.values():
        all_benchmark_names.update(benchmarks.keys())
    
    for benchmark_name in all_benchmark_names:
        plt.figure(figsize=(10, 6))
        
        colors = plt.cm.Set1(np.linspace(0, 1, len(benchmark_data)))
        
        for (file_name, benchmarks), color in zip(benchmark_data.items(), colors):
            if benchmark_name not in benchmarks or not benchmarks[benchmark_name]:
                continue
            
            results = benchmarks[benchmark_name]
            results.sort(key=lambda x: x[0])
            
            input_sizes = [r[0] for r in results]
            times = [r[1] for r in results]
            
            label = Path(file_name).stem
            plt.loglog(input_sizes, times, marker='o', linewidth=2, markersize=6, 
                      color=color, label=label)
        
        plt.xlabel('Input Size', fontsize=12)
        plt.ylabel('Execution Time (ms)', fontsize=12)
        plt.title(f'{benchmark_name} Performance Comparison', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        
        if output_dir:
            output_path = Path(output_dir) / f"{benchmark_name}_comparison.png"
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to: {output_path}")
        else:
            plt.show()

def print_summary(benchmark_data):
    """Print a summary of parsed benchmark data."""
    print("\n" + "="*60)
    print("BENCHMARK SUMMARY")
    print("="*60)
    
    for file_name, benchmarks in benchmark_data.items():
        print(f"\nFile: {file_name}")
        print("-" * 40)
        
        for benchmark_name, results in benchmarks.items():
            if results:
                print(f"  {benchmark_name}:")
                print(f"    Input sizes: {len(results)} data points")
                min_size = min(r[0] for r in results)
                max_size = max(r[0] for r in results)
                print(f"    Size range: {min_size} - {max_size}")
                min_time = min(r[1] for r in results)
                max_time = max(r[1] for r in results)
                print(f"    Time range: {min_time:.6f} - {max_time:.2f} ms")
            else:
                print(f"  {benchmark_name}: No valid data")

def main():
    parser = argparse.ArgumentParser(
        description='Plot Google Benchmark results from output files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python benchmark_plotter.py --file_list=benchmark1.txt,benchmark2.txt
  python benchmark_plotter.py --file_list=bench1.txt,bench2.txt --output=comparison.png
  python benchmark_plotter.py --file_list=bench1.txt,bench2.txt --separate --output_dir=plots/
        """
    )
    
    parser.add_argument('--file_list', required=True,
                       help='Comma-separated list of benchmark output files')
    parser.add_argument('--output', 
                       help='Output file path for saving the combined plot (optional)')
    parser.add_argument('--separate', action='store_true',
                       help='Create separate plots for each benchmark type')
    parser.add_argument('--output_dir',
                       help='Output directory for separate plots (used with --separate)')
    parser.add_argument('--summary', action='store_true',
                       help='Print summary of parsed benchmark data')
    
    args = parser.parse_args()
    
    # Parse file list
    file_list = [f.strip() for f in args.file_list.split(',')]
    
    if not file_list:
        print("Error: No files specified.")
        sys.exit(1)
    
    # Parse all benchmark files
    benchmark_data = {}
    
    for file_path in file_list:
        print(f"Parsing file: {file_path}")
        benchmarks = parse_benchmark_file(file_path)
        
        if benchmarks:
            benchmark_data[file_path] = benchmarks
            print(f"  Found {len(benchmarks)} benchmark(s)")
        else:
            print(f"  Warning: No benchmark data found in {file_path}")
    
    if not benchmark_data:
        print("Error: No valid benchmark data found in any file.")
        sys.exit(1)
    
    # Print summary if requested
    if args.summary:
        print_summary(benchmark_data)
    
    # Create output directory if specified
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Generate plots
    if args.separate:
        print("\nGenerating separate plots for each benchmark type...")
        plot_separate_benchmarks(benchmark_data, args.output_dir)
    else:
        print("\nGenerating combined comparison plot...")
        plot_benchmarks(benchmark_data, args.output)
    
    print("\nPlotting completed!")

if __name__ == "__main__":
    main()