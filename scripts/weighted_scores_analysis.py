#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Weighted Scores Analysis Script
Function: Analyze JSON data and create subplots for different weighting schemes with difficulty levels
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set font for English
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

def get_weighting_scheme_info():
    """
    Get detailed information about weighting schemes
    
    Returns:
        dict: Weighting scheme information dictionary
    """
    return {
        "Method 1": {
            "name": "Answer Correctness Only",
            "description": "Only considers answer correctness, weight 100%",
            "weights": {
                "answer_correctness": 1.0,
                "reasoning_logic": 0.0,
                "step_completeness": 0.0,
                "mathematical_accuracy": 0.0,
                "expression_clarity": 0.0
            }
        },
        "Method 2": {
            "name": "Answer Correctness + Step Completeness",
            "description": "Answer correctness 60%, step completeness 40%",
            "weights": {
                "answer_correctness": 0.6,
                "reasoning_logic": 0.0,
                "step_completeness": 0.4,
                "mathematical_accuracy": 0.0,
                "expression_clarity": 0.0
            }
        },
        "Method 3": {
            "name": "Answer Correctness + Reasoning Logic",
            "description": "Answer correctness 60%, reasoning logic 40%",
            "weights": {
                "answer_correctness": 0.6,
                "reasoning_logic": 0.4,
                "step_completeness": 0.0,
                "mathematical_accuracy": 0.0,
                "expression_clarity": 0.0
            }
        },
        "Method 4": {
            "name": "Comprehensive Evaluation",
            "description": "Answer correctness 30%, reasoning logic 25%, step completeness 25%, mathematical accuracy 20%",
            "weights": {
                "answer_correctness": 0.3,
                "reasoning_logic": 0.25,
                "step_completeness": 0.25,
                "mathematical_accuracy": 0.2,
                "expression_clarity": 0.0
            }
        }
    }

def load_and_analyze_data(file_path, dataset_name):
    """
    Load and analyze JSON data
    
    Parameters:
        file_path (str): JSON file path
        dataset_name (str): Dataset name
        
    Returns:
        dict: Analysis results
    """
    print(f"Loading dataset: {dataset_name}")
    
    # Load data
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Extract score data
    scores_data = []
    for i, result in enumerate(data['results']):
        try:
            scores_data.append({
                'model_name': result['model_name'],
                'model_short_name': result['model_short_name'],
                'model_params': result['model_params'],
                'weighting_scheme': result['weighting_scheme_name'],
                'weighted_score': result['weighted_score'],
                'custom_difficulty': result['custom_difficulty'],
                'answer_correctness': result['evaluation']['answer_correctness'],
                'reasoning_logic': result['evaluation']['reasoning_logic'],
                'step_completeness': result['evaluation']['step_completeness'],
                'mathematical_accuracy': result['evaluation']['mathematical_accuracy'],
                'expression_clarity': result['evaluation']['expression_clarity'],
                'overall_score': result['evaluation']['overall_score']
            })
        except KeyError as e:
            print(f"Warning: Skipping record {i}, missing field: {e}")
            continue
        except Exception as e:
            print(f"Warning: Skipping record {i}, error: {e}")
            continue
    
    # Convert to DataFrame
    df = pd.DataFrame(scores_data)
    
    # Calculate statistics
    analysis = {
        'dataset_name': dataset_name,
        'total_samples': len(df),
        'models': df['model_short_name'].unique(),
        'weighting_schemes': df['weighting_scheme'].unique(),
        'difficulties': df['custom_difficulty'].unique(),
        'avg_scores_by_model_scheme_difficulty': df.groupby(['model_short_name', 'weighting_scheme', 'custom_difficulty'])['weighted_score'].agg(['mean', 'std', 'count']).reset_index(),
        'raw_data': df
    }
    
    return analysis

def create_difficulty_subplots(analysis_results, output_dir):
    """
    Create subplots for each weighting scheme showing difficulty levels
    
    Parameters:
        analysis_results (list): Analysis results list
        output_dir (Path): Output directory
    """
    print("Creating difficulty subplots...")
    
    # Create figure with 2x2 subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    weighting_info = get_weighting_scheme_info()
    
    # Process each weighting scheme
    for scheme_idx, (scheme_name, scheme_info) in enumerate(weighting_info.items()):
        ax = axes[scheme_idx]
        
        # Collect data for this weighting scheme from the current dataset only
        all_data = []
        difficulty_counts = {'Easy': 0, 'Medium': 0, 'Hard': 0}
        
        # Use only the first dataset (assuming single dataset analysis)
        if analysis_results:
            df = analysis_results[0]['raw_data']
            scheme_data = df[df['weighting_scheme'] == scheme_name]
            
            if len(scheme_data) > 0:
                # Count difficulty levels - 每个难度等级的数据量应该是总数据量除以模型数量
                for difficulty in ['Easy', 'Medium', 'Hard']:
                    difficulty_data = scheme_data[scheme_data['custom_difficulty'] == difficulty]
                    # 数据量应该是问题数量，不是总记录数
                    difficulty_counts[difficulty] = len(difficulty_data) // len(df['model_short_name'].unique())
                
                # Calculate average scores by model and difficulty
                avg_scores = scheme_data.groupby(['model_short_name', 'custom_difficulty'])['weighted_score'].mean().reset_index()
                all_data.append(avg_scores)
        
        if all_data:
            # Combine all data
            combined_data = pd.concat(all_data, ignore_index=True)
            
            # Calculate overall average by model and difficulty
            final_scores = combined_data.groupby(['model_short_name', 'custom_difficulty'])['weighted_score'].mean().reset_index()
            
            # Sort by model parameters
            model_params = {'1.5B': 1.5, '7B': 7.0, '14B': 14.0, '32B': 32.0, '70B': 70.0}
            final_scores['model_params'] = final_scores['model_short_name'].map(model_params)
            final_scores = final_scores.sort_values(['custom_difficulty', 'model_params'])
            
            # Create line plot for each difficulty level
            difficulties = ['Easy', 'Medium', 'Hard']
            colors = ['#2ca02c', '#ff7f0e', '#d62728']  # Green, Orange, Red
            markers = ['o', 's', '^']
            
            for i, difficulty in enumerate(difficulties):
                difficulty_data = final_scores[final_scores['custom_difficulty'] == difficulty]
                if len(difficulty_data) > 0:
                    # Sort by model parameters
                    difficulty_data = difficulty_data.sort_values('model_params')
                    
                    # Plot line
                    line = ax.plot(difficulty_data['model_short_name'], 
                                 difficulty_data['weighted_score'], 
                                 marker=markers[i], 
                                 linewidth=2.5, 
                                 markersize=8,
                                 color=colors[i],
                                 label=f'{difficulty} (n={difficulty_counts[difficulty]})')
                    
                    # Add data labels
                    for x, y in zip(difficulty_data['model_short_name'], difficulty_data['weighted_score']):
                        ax.annotate(f'{y:.2f}', 
                                   (x, y), 
                                   textcoords="offset points", 
                                   xytext=(0,10), 
                                   ha='center',
                                   fontsize=9,
                                   fontweight='bold')
        
        # Set subplot properties
        ax.set_title(f'{scheme_name}: {scheme_info["name"]}', fontsize=14, fontweight='bold')
        ax.set_xlabel('Model Name', fontsize=12)
        ax.set_ylabel('Weighted Score', fontsize=12)
        
        # Set y-axis range
        if all_data:
            y_min = final_scores['weighted_score'].min() - 0.5
            y_max = final_scores['weighted_score'].max() + 0.5
            ax.set_ylim(y_min, y_max)
        
        # Add grid
        ax.grid(True, alpha=0.3, linestyle='--')
        
        # Add legend
        ax.legend(title='Difficulty Level', title_fontsize=10, fontsize=9, 
                 loc='lower right', bbox_to_anchor=(1.0, 0.0))
    
    # Adjust layout
    plt.tight_layout()
    
    # Save plot
    plt.savefig(output_dir / 'weighting_schemes_by_difficulty.png', dpi=300, bbox_inches='tight')
    print("Difficulty subplots saved")
    plt.show()

def create_combined_difficulty_plot(analysis_results, output_dir):
    """
    Create a single combined plot showing all weighting schemes and difficulties
    
    Parameters:
        analysis_results (list): Analysis results list
        output_dir (Path): Output directory
    """
    print("Creating combined difficulty plot...")
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 8))
    
    weighting_info = get_weighting_scheme_info()
    
    # Collect all data
    all_combined_data = []
    difficulty_counts = {'Easy': 0, 'Medium': 0, 'Hard': 0}
    
    for scheme_name in weighting_info.keys():
        for analysis in analysis_results:
            df = analysis['raw_data']
            scheme_data = df[df['weighting_scheme'] == scheme_name]
            
            if len(scheme_data) > 0:
                # Count difficulty levels
                for difficulty in ['Easy', 'Medium', 'Hard']:
                    difficulty_counts[difficulty] += len(scheme_data[scheme_data['custom_difficulty'] == difficulty])
                
                # Calculate average scores
                avg_scores = scheme_data.groupby(['model_short_name', 'custom_difficulty'])['weighted_score'].mean().reset_index()
                avg_scores['weighting_scheme'] = scheme_name
                all_combined_data.append(avg_scores)
    
    if all_combined_data:
        # Combine all data
        combined_data = pd.concat(all_combined_data, ignore_index=True)
        
        # Calculate overall average
        final_scores = combined_data.groupby(['model_short_name', 'custom_difficulty'])['weighted_score'].mean().reset_index()
        
        # Sort by model parameters
        model_params = {'1.5B': 1.5, '7B': 7.0, '14B': 14.0, '32B': 32.0, '70B': 70.0}
        final_scores['model_params'] = final_scores['model_short_name'].map(model_params)
        final_scores = final_scores.sort_values(['custom_difficulty', 'model_params'])
        
        # Create line plot for each difficulty level
        difficulties = ['Easy', 'Medium', 'Hard']
        colors = ['#2ca02c', '#ff7f0e', '#d62728']  # Green, Orange, Red
        markers = ['o', 's', '^']
        
        for i, difficulty in enumerate(difficulties):
            difficulty_data = final_scores[final_scores['custom_difficulty'] == difficulty]
            if len(difficulty_data) > 0:
                # Sort by model parameters
                difficulty_data = difficulty_data.sort_values('model_params')
                
                # Plot line
                line = ax.plot(difficulty_data['model_short_name'], 
                             difficulty_data['weighted_score'], 
                             marker=markers[i], 
                             linewidth=2.5, 
                             markersize=8,
                             color=colors[i],
                             label=f'{difficulty} (n={difficulty_counts[difficulty]})')
                
                # Add data labels
                for x, y in zip(difficulty_data['model_short_name'], difficulty_data['weighted_score']):
                    ax.annotate(f'{y:.2f}', 
                               (x, y), 
                               textcoords="offset points", 
                               xytext=(0,10), 
                               ha='center',
                               fontsize=9,
                               fontweight='bold')
    
    # Set plot properties
    ax.set_title('Model Performance by Difficulty Level Across All Weighting Schemes', 
                fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Model Name', fontsize=12, fontweight='bold')
    ax.set_ylabel('Weighted Score', fontsize=12, fontweight='bold')
    
    # Set y-axis range
    if all_combined_data:
        y_min = final_scores['weighted_score'].min() - 0.5
        y_max = final_scores['weighted_score'].max() + 0.5
        ax.set_ylim(y_min, y_max)
    
    # Add grid
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Add legend
    ax.legend(title='Difficulty Level', title_fontsize=12, fontsize=10, 
             loc='lower right', bbox_to_anchor=(1.0, 0.0))
    
    # Adjust layout
    plt.tight_layout()
    
    plt.show()

    # Save plot
    plt.savefig(output_dir / 'combined_difficulty_analysis.png', dpi=300, bbox_inches='tight')
    print("Combined difficulty plot saved")
    plt.show()

def generate_analysis_report(analysis_results, output_dir):
    """
    Generate analysis report
    
    Parameters:
        analysis_results (list): Analysis results list
        output_dir (Path): Output directory
    """
    print("Generating analysis report...")
    
    report_path = output_dir / 'analysis_report.txt'
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("Weighted Scores Analysis Report\n")
        f.write("=" * 80 + "\n\n")
        
        # Weighting scheme description
        f.write("Weighting Scheme Description:\n")
        f.write("-" * 40 + "\n")
        weighting_info = get_weighting_scheme_info()
        for method, info in weighting_info.items():
            f.write(f"{method}: {info['name']}\n")
            f.write(f"Description: {info['description']}\n")
            f.write("Weight Distribution:\n")
            for metric, weight in info['weights'].items():
                f.write(f"  - {metric}: {weight:.2f}\n")
            f.write("\n")
        
        # Dataset analysis
        for analysis in analysis_results:
            dataset_name = analysis['dataset_name']
            f.write(f"\n{dataset_name} Dataset Analysis:\n")
            f.write("-" * 40 + "\n")
            
            f.write(f"Total samples: {analysis['total_samples']}\n")
            f.write(f"Number of models: {len(analysis['models'])}\n")
            f.write(f"Number of weighting schemes: {len(analysis['weighting_schemes'])}\n")
            f.write(f"Difficulty levels: {', '.join(analysis['difficulties'])}\n\n")
            
            # Difficulty level statistics
            df = analysis['raw_data']
            difficulty_stats = df.groupby('custom_difficulty').size()
            f.write("Difficulty level distribution:\n")
            for difficulty, count in difficulty_stats.items():
                f.write(f"  {difficulty}: {count} samples\n")
            f.write("\n")
            
            # Model performance by difficulty
            f.write("Model performance by difficulty level:\n")
            model_difficulty_stats = df.groupby(['model_short_name', 'custom_difficulty'])['weighted_score'].agg(['mean', 'std', 'count']).reset_index()
            for _, row in model_difficulty_stats.iterrows():
                f.write(f"  {row['model_short_name']} - {row['custom_difficulty']}: {row['mean']:.3f} ± {row['std']:.3f} (n={row['count']})\n")
            f.write("\n")
    
    print(f"Analysis report saved to: {report_path}")

def main():
    """
    Main function
    """
    print("Starting weighted scores analysis...")
    
    # Define file paths
    data_dir = Path("plot_data/custom_difficulty/results_with_custom_difficulty")
    output_dir = Path("plot_data/weighted_scores_analysis")
    output_dir.mkdir(exist_ok=True)
    
    # Define datasets
    datasets = {
        "MATH-500": "MATH500_unified_results.json"
        # "MATH-500": "MATH500_unified_results.json",
        # "DeepMath-103K": "DeepMath103K_unified_results.json", 
        # "Hendrycks-Math": "HendrycksMath_unified_results.json"
    }
    
    analysis_results = []
    
    # Analyze each dataset
    for dataset_name, filename in datasets.items():
        file_path = data_dir / filename
        
        if not file_path.exists():
            print(f"Warning: File does not exist: {file_path}")
            continue
        
        # Analyze data
        analysis = load_and_analyze_data(file_path, dataset_name)
        analysis_results.append(analysis)
        
        print(f"Analysis completed for {dataset_name}")
    
    # Generate plots
    if analysis_results:
        create_difficulty_subplots(analysis_results, output_dir)
        
        print(f"\nAnalysis completed!")
        print(f"Output directory: {output_dir}")
        
        # Display weighting scheme information
        print("\nWeighting Scheme Information:")
        weighting_info = get_weighting_scheme_info()
        for method, info in weighting_info.items():
            print(f"{method}: {info['name']}")
            print(f"  {info['description']}")

if __name__ == "__main__":
    main() 