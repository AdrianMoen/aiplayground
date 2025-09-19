"""
Comprehensive benchmark analysis and visualization script.
Compares base model vs fine-tuned model performance across multiple metrics.
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
import re
from scipy import stats
from pathlib import Path
import argparse
import pprint

# Set style for better-looking plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class CypherQualityAnalyzer:
    """Analyze the quality of generated Cypher queries."""
    
    def __init__(self):
        self.cypher_keywords = ['MATCH', 'WHERE', 'RETURN', 'WITH', 'ORDER BY', 'LIMIT', 'CREATE', 'DELETE', 'SET']
        self.graph_elements = ['Movie', 'Person', 'Genre', 'ACTED_IN', 'DIRECTED', 'IN_GENRE']
    
    def is_syntactically_valid(self, query: str) -> bool:
        """Basic syntax validation for Cypher queries."""
        query = query.strip()
        if not query:
            return False
        
        # Must contain at least MATCH and RETURN
        has_match = 'MATCH' in query.upper()
        has_return = 'RETURN' in query.upper()
        
        # Check for balanced parentheses
        if query.count('(') != query.count(')'):
            return False
        
        # Check for balanced brackets
        if query.count('[') != query.count(']'):
            return False
        
        return has_match and has_return
    
    def extract_first_valid_query(self, output: str) -> str:
        """Extract the first valid Cypher query from model output."""
        lines = output.split('\n')
        query_lines = []
        
        for line in lines:
            line = line.strip()
            if not line or line.startswith('#') or line.startswith('//'):
                continue
            
            query_lines.append(line)
            
            # Check if we have a complete query
            current_query = ' '.join(query_lines)
            if 'RETURN' in current_query.upper():
                return current_query
        
        return ' '.join(query_lines)
    
    def calculate_structure_similarity(self, generated: str, expected: str) -> float:
        """Calculate structural similarity between queries."""
        gen_query = self.extract_first_valid_query(generated)
        
        # Extract structural elements
        gen_elements = self.extract_query_elements(gen_query)
        exp_elements = self.extract_query_elements(expected)
        
        # Calculate Jaccard similarity
        intersection = len(gen_elements.intersection(exp_elements))
        union = len(gen_elements.union(exp_elements))
        
        return intersection / union if union > 0 else 0.0
    
    def extract_query_elements(self, query: str) -> set:
        """Extract structural elements from a Cypher query."""
        elements = set()
        query_upper = query.upper()
        
        # Extract keywords
        for keyword in self.cypher_keywords:
            if keyword in query_upper:
                elements.add(keyword)
        
        # Extract graph elements
        for element in self.graph_elements:
            if element in query:
                elements.add(element)
        
        # Extract patterns like (var:Label)
        node_patterns = re.findall(r'\([^)]*:[^)]*\)', query)
        for pattern in node_patterns:
            elements.add('NODE_PATTERN')
        
        # Extract relationship patterns
        rel_patterns = re.findall(r'-\[[^\]]*\]->', query)
        for pattern in rel_patterns:
            elements.add('REL_PATTERN')
        
        return elements

class BenchmarkAnalyzer:
    """Main class for analyzing benchmark results."""
    
    def __init__(self):
        self.quality_analyzer = CypherQualityAnalyzer()
    
    def load_benchmark_data(self, file_path: str) -> Dict:
        """Load benchmark data from JSON file."""
        with open(file_path, 'r') as f:
            return json.load(f)
    
    def analyze_performance_metrics(self, data: Dict) -> Dict:
        """Extract and analyze performance metrics."""
        measurements = data['raw_measurements']
        
        generation_times = [m['generation_time'] for m in measurements]
        tokens_per_sec = [m['tokens_per_second'] for m in measurements]
        output_tokens = [m['output_tokens'] for m in measurements]
        
        return {
            'generation_time': {
                'mean': np.mean(generation_times),
                'median': np.median(generation_times),
                'std': np.std(generation_times),
                'p95': np.percentile(generation_times, 95),
                'p99': np.percentile(generation_times, 99),
                'min': np.min(generation_times),
                'max': np.max(generation_times),
                'data': generation_times
            },
            'tokens_per_second': {
                'mean': np.mean(tokens_per_sec),
                'median': np.median(tokens_per_sec),
                'std': np.std(tokens_per_sec),
                'data': tokens_per_sec
            },
            'output_tokens': {
                'mean': np.mean(output_tokens),
                'median': np.median(output_tokens),
                'std': np.std(output_tokens),
                'data': output_tokens
            }
        }
    
    def analyze_quality_metrics(self, data: Dict) -> Dict:
        """Analyze quality metrics for Cypher queries."""
        measurements = data['raw_measurements']
        
        syntax_scores = []
        similarity_scores = []
        
        for measurement in measurements:
            generated = measurement['output_text']
            expected = measurement['expected_output']
            
            # Syntax validation
            first_query = self.quality_analyzer.extract_first_valid_query(generated)
            is_valid = self.quality_analyzer.is_syntactically_valid(first_query)
            syntax_scores.append(1.0 if is_valid else 0.0)
            
            # Structural similarity
            similarity = self.quality_analyzer.calculate_structure_similarity(generated, expected)
            similarity_scores.append(similarity)
        
        return {
            'syntax_accuracy': {
                'mean': np.mean(syntax_scores),
                'total_valid': sum(syntax_scores),
                'total_queries': len(syntax_scores),
                'data': syntax_scores
            },
            'structural_similarity': {
                'mean': np.mean(similarity_scores),
                'median': np.median(similarity_scores),
                'std': np.std(similarity_scores),
                'data': similarity_scores
            }
        }
    
    def compare_models(self, base_data: Dict, finetuned_data: Dict) -> Dict:
        """Compare two models across all metrics."""
        base_perf = self.analyze_performance_metrics(base_data)
        ft_perf = self.analyze_performance_metrics(finetuned_data)
        
        base_quality = self.analyze_quality_metrics(base_data)
        ft_quality = self.analyze_quality_metrics(finetuned_data)

        # print(pprint.pformat(base_data))
        
        return {
            'base_model': {
                'performance': base_perf,
                'quality': base_quality,
                'model_info': base_data["benchmark_info"]['model_info']
            },
            'finetuned_model': {
                'performance': ft_perf,
                'quality': ft_quality,
                'model_info': finetuned_data["benchmark_info"]['model_info']
            }
        }
    
    def create_visualizations(self, comparison: Dict, output_dir: str = "./benchmark_analysis"):
        """Create comprehensive visualizations."""
        Path(output_dir).mkdir(exist_ok=True)
        
        base = comparison['base_model']
        ft = comparison['finetuned_model']
        
        # 1. Performance Overview
        self._plot_performance_overview(base, ft, output_dir)
        
        # 2. Latency Distribution
        self._plot_latency_distribution(base, ft, output_dir)
        
        # 3. Throughput Comparison
        self._plot_throughput_comparison(base, ft, output_dir)
        
        # 4. Quality Metrics
        self._plot_quality_metrics(base, ft, output_dir)
        
        # 5. Statistical Significance Tests
        self._perform_statistical_tests(base, ft, output_dir)
        
        # 6. Detailed Performance Dashboard
        self._create_dashboard(base, ft, output_dir)
    
    def _plot_performance_overview(self, base: Dict, ft: Dict, output_dir: str):
        """Create performance overview comparison."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Average generation time
        models = ['Base Model', 'Fine-tuned Model']
        gen_times = [base['performance']['generation_time']['mean'], 
                    ft['performance']['generation_time']['mean']]
        
        bars1 = ax1.bar(models, gen_times, color=['skyblue', 'lightcoral'])
        ax1.set_ylabel('Average Generation Time (s)')
        ax1.set_title('Average Generation Time Comparison')
        for i, v in enumerate(gen_times):
            ax1.text(i, v + 0.1, f'{v:.2f}s', ha='center', va='bottom')
        
        # Tokens per second
        throughput = [base['performance']['tokens_per_second']['mean'],
                     ft['performance']['tokens_per_second']['mean']]
        
        bars2 = ax2.bar(models, throughput, color=['skyblue', 'lightcoral'])
        ax2.set_ylabel('Tokens per Second')
        ax2.set_title('Throughput Comparison')
        for i, v in enumerate(throughput):
            ax2.text(i, v + 0.5, f'{v:.1f}', ha='center', va='bottom')
        
        # P95 latency
        p95_times = [base['performance']['generation_time']['p95'],
                    ft['performance']['generation_time']['p95']]
        
        bars3 = ax3.bar(models, p95_times, color=['skyblue', 'lightcoral'])
        ax3.set_ylabel('P95 Generation Time (s)')
        ax3.set_title('P95 Latency Comparison')
        for i, v in enumerate(p95_times):
            ax3.text(i, v + 0.1, f'{v:.2f}s', ha='center', va='bottom')
        
        # Syntax accuracy
        syntax_acc = [base['quality']['syntax_accuracy']['mean'],
                     ft['quality']['syntax_accuracy']['mean']]
        
        bars4 = ax4.bar(models, syntax_acc, color=['skyblue', 'lightcoral'])
        ax4.set_ylabel('Syntax Accuracy')
        ax4.set_title('Query Syntax Accuracy')
        ax4.set_ylim(0, 1.1)
        for i, v in enumerate(syntax_acc):
            ax4.text(i, v + 0.02, f'{v:.1%}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/performance_overview.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_latency_distribution(self, base: Dict, ft: Dict, output_dir: str):
        """Plot latency distribution comparison."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        base_times = base['performance']['generation_time']['data']
        ft_times = ft['performance']['generation_time']['data']
        
        # Histogram comparison
        ax1.hist(base_times, alpha=0.7, label='Base Model', bins=15, color='skyblue')
        ax1.hist(ft_times, alpha=0.7, label='Fine-tuned Model', bins=15, color='lightcoral')
        ax1.set_xlabel('Generation Time (s)')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Generation Time Distribution')
        ax1.legend()
        
        # Box plot comparison
        data_to_plot = [base_times, ft_times]
        box_plot = ax2.boxplot(data_to_plot, labels=['Base Model', 'Fine-tuned Model'], 
                              patch_artist=True)
        
        colors = ['skyblue', 'lightcoral']
        for patch, color in zip(box_plot['boxes'], colors):
            patch.set_facecolor(color)
        
        ax2.set_ylabel('Generation Time (s)')
        ax2.set_title('Generation Time Distribution (Box Plot)')
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/latency_distribution.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_throughput_comparison(self, base: Dict, ft: Dict, output_dir: str):
        """Plot throughput over time."""
        fig, ax = plt.subplots(1, 1, figsize=(12, 6))
        
        base_throughput = base['performance']['tokens_per_second']['data']
        ft_throughput = ft['performance']['tokens_per_second']['data']
        
        x = range(len(base_throughput))
        
        ax.plot(x, base_throughput, 'o-', label='Base Model', alpha=0.7, color='skyblue')
        ax.plot(x, ft_throughput, 's-', label='Fine-tuned Model', alpha=0.7, color='lightcoral')
        
        ax.axhline(y=np.mean(base_throughput), color='blue', linestyle='--', alpha=0.5, 
                  label=f'Base Avg: {np.mean(base_throughput):.1f}')
        ax.axhline(y=np.mean(ft_throughput), color='red', linestyle='--', alpha=0.5,
                  label=f'Fine-tuned Avg: {np.mean(ft_throughput):.1f}')
        
        ax.set_xlabel('Test Number')
        ax.set_ylabel('Tokens per Second')
        ax.set_title('Throughput Over Time')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/throughput_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_quality_metrics(self, base: Dict, ft: Dict, output_dir: str):
        """Plot quality metrics comparison."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Syntax accuracy comparison
        models = ['Base Model', 'Fine-tuned Model']
        syntax_scores = [base['quality']['syntax_accuracy']['mean'],
                        ft['quality']['syntax_accuracy']['mean']]
        
        bars = ax1.bar(models, syntax_scores, color=['skyblue', 'lightcoral'])
        ax1.set_ylabel('Syntax Accuracy')
        ax1.set_title('Cypher Syntax Accuracy')
        ax1.set_ylim(0, 1.1)
        
        for i, v in enumerate(syntax_scores):
            ax1.text(i, v + 0.02, f'{v:.1%}', ha='center', va='bottom')
        
        # Structural similarity
        base_sim = base['quality']['structural_similarity']['data']
        ft_sim = ft['quality']['structural_similarity']['data']
        
        ax2.hist(base_sim, alpha=0.7, label='Base Model', bins=10, color='skyblue')
        ax2.hist(ft_sim, alpha=0.7, label='Fine-tuned Model', bins=10, color='lightcoral')
        ax2.set_xlabel('Structural Similarity Score')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Query Structural Similarity Distribution')
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/quality_metrics.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _perform_statistical_tests(self, base: Dict, ft: Dict, output_dir: str):
        """Perform statistical significance tests."""
        results = {}
        
        # T-test for generation times
        base_times = base['performance']['generation_time']['data']
        ft_times = ft['performance']['generation_time']['data']
        t_stat, p_value = stats.ttest_ind(base_times, ft_times)
        results['generation_time'] = {'t_stat': t_stat, 'p_value': p_value}
        
        # T-test for throughput
        base_throughput = base['performance']['tokens_per_second']['data']
        ft_throughput = ft['performance']['tokens_per_second']['data']
        t_stat, p_value = stats.ttest_ind(base_throughput, ft_throughput)
        results['throughput'] = {'t_stat': t_stat, 'p_value': p_value}
        
        # Chi-square test for syntax accuracy (with safety checks)
        base_syntax = base['quality']['syntax_accuracy']['data']
        ft_syntax = ft['quality']['syntax_accuracy']['data']
        
        base_valid = sum(base_syntax)
        base_invalid = len(base_syntax) - base_valid
        ft_valid = sum(ft_syntax)
        ft_invalid = len(ft_syntax) - ft_valid
        
        contingency_table = [[base_valid, base_invalid], [ft_valid, ft_invalid]]
        
        # FIXED: Check if chi-square test is valid
        # Chi-square requires all expected frequencies >= 5 and no zeros
        min_cell_count = min(base_valid, base_invalid, ft_valid, ft_invalid)
        
        if min_cell_count == 0:
            # Use Fisher's exact test instead for small/zero cell counts
            try:
                from scipy.stats import fisher_exact
                
                # For 2x2 contingency table, use Fisher's exact test
                if len(contingency_table) == 2 and len(contingency_table[0]) == 2:
                    odds_ratio, p_value = fisher_exact(contingency_table)
                    results['syntax_accuracy'] = {
                        'test_type': 'fisher_exact',
                        'odds_ratio': odds_ratio, 
                        'p_value': p_value,
                        'note': 'Used Fisher\'s exact test due to zero cell counts'
                    }
                else:
                    # Fall back to descriptive statistics
                    results['syntax_accuracy'] = {
                        'test_type': 'descriptive_only',
                        'base_accuracy': base_valid / len(base_syntax),
                        'ft_accuracy': ft_valid / len(ft_syntax),
                        'note': 'Statistical test not possible due to zero cell counts'
                    }
            except ImportError:
                # Fall back to descriptive statistics if scipy doesn't have fisher_exact
                results['syntax_accuracy'] = {
                    'test_type': 'descriptive_only',
                    'base_accuracy': base_valid / len(base_syntax),
                    'ft_accuracy': ft_valid / len(ft_syntax),
                    'note': 'Statistical test not possible due to zero cell counts'
                }
        else:
            # Safe to use chi-square test
            try:
                chi2, p_value, dof, expected = stats.chi2_contingency(contingency_table)
                results['syntax_accuracy'] = {
                    'test_type': 'chi_square',
                    'chi2': chi2, 
                    'p_value': p_value,
                    'contingency_table': contingency_table
                }
            except ValueError as e:
                # Fallback if chi-square still fails
                results['syntax_accuracy'] = {
                    'test_type': 'error',
                    'error': str(e),
                    'base_accuracy': base_valid / len(base_syntax),
                    'ft_accuracy': ft_valid / len(ft_syntax),
                    'contingency_table': contingency_table
                }
        
        # Save results
        with open(f"{output_dir}/statistical_tests.json", 'w') as f:
            json.dump(results, f, indent=2)
        
        # Create summary text (UPDATED to handle different test types)
        with open(f"{output_dir}/statistical_summary.txt", 'w') as f:
            f.write("Statistical Significance Tests\n")
            f.write("=" * 40 + "\n\n")
            
            f.write(f"Generation Time Comparison:\n")
            f.write(f"  T-statistic: {results['generation_time']['t_stat']:.4f}\n")
            f.write(f"  P-value: {results['generation_time']['p_value']:.4f}\n")
            f.write(f"  Significant: {'Yes' if results['generation_time']['p_value'] < 0.05 else 'No'}\n\n")
            
            f.write(f"Throughput Comparison:\n")
            f.write(f"  T-statistic: {results['throughput']['t_stat']:.4f}\n")
            f.write(f"  P-value: {results['throughput']['p_value']:.4f}\n")
            f.write(f"  Significant: {'Yes' if results['throughput']['p_value'] < 0.05 else 'No'}\n\n")
            
            # UPDATED: Handle different syntax accuracy test types
            f.write(f"Syntax Accuracy Comparison:\n")
            syntax_result = results['syntax_accuracy']
            test_type = syntax_result.get('test_type', 'unknown')
            
            if test_type == 'chi_square':
                f.write(f"  Test: Chi-square\n")
                f.write(f"  Chi-square: {syntax_result['chi2']:.4f}\n")
                f.write(f"  P-value: {syntax_result['p_value']:.4f}\n")
                f.write(f"  Significant: {'Yes' if syntax_result['p_value'] < 0.05 else 'No'}\n")
            elif test_type == 'fisher_exact':
                f.write(f"  Test: Fisher's Exact Test\n")
                f.write(f"  Odds Ratio: {syntax_result['odds_ratio']:.4f}\n")
                f.write(f"  P-value: {syntax_result['p_value']:.4f}\n")
                f.write(f"  Significant: {'Yes' if syntax_result['p_value'] < 0.05 else 'No'}\n")
                f.write(f"  Note: {syntax_result['note']}\n")
            else:
                f.write(f"  Test: {test_type}\n")
                f.write(f"  Base Accuracy: {syntax_result['base_accuracy']:.1%}\n")
                f.write(f"  Fine-tuned Accuracy: {syntax_result['ft_accuracy']:.1%}\n")
                if 'note' in syntax_result:
                    f.write(f"  Note: {syntax_result['note']}\n")


    def _create_dashboard(self, base: Dict, ft: Dict, output_dir: str):
        """Create a comprehensive dashboard."""
        fig = plt.figure(figsize=(20, 16))
        
        # Create a 4x3 grid
        gs = fig.add_gridspec(4, 3, height_ratios=[1, 1, 1, 1], width_ratios=[1, 1, 1])
        
        # Performance metrics summary table
        ax1 = fig.add_subplot(gs[0, :])
        ax1.axis('tight')
        ax1.axis('off')
        
        # Create performance comparison table
        metrics_data = [
            ['Metric', 'Base Model', 'Fine-tuned Model', 'Improvement'],
            ['Avg Generation Time (s)', 
             f"{base['performance']['generation_time']['mean']:.3f}",
             f"{ft['performance']['generation_time']['mean']:.3f}",
             f"{((base['performance']['generation_time']['mean'] - ft['performance']['generation_time']['mean']) / base['performance']['generation_time']['mean'] * 100):+.1f}%"],
            ['P95 Generation Time (s)',
             f"{base['performance']['generation_time']['p95']:.3f}",
             f"{ft['performance']['generation_time']['p95']:.3f}",
             f"{((base['performance']['generation_time']['p95'] - ft['performance']['generation_time']['p95']) / base['performance']['generation_time']['p95'] * 100):+.1f}%"],
            ['Avg Throughput (tok/s)',
             f"{base['performance']['tokens_per_second']['mean']:.1f}",
             f"{ft['performance']['tokens_per_second']['mean']:.1f}",
             f"{((ft['performance']['tokens_per_second']['mean'] - base['performance']['tokens_per_second']['mean']) / base['performance']['tokens_per_second']['mean'] * 100):+.1f}%"],
            ['Syntax Accuracy',
             f"{base['quality']['syntax_accuracy']['mean']:.1%}",
             f"{ft['quality']['syntax_accuracy']['mean']:.1%}",
             f"{((ft['quality']['syntax_accuracy']['mean'] - base['quality']['syntax_accuracy']['mean']) * 100):+.1f}pp"],
            ['Avg Structural Similarity',
             f"{base['quality']['structural_similarity']['mean']:.3f}",
             f"{ft['quality']['structural_similarity']['mean']:.3f}",
             f"{((ft['quality']['structural_similarity']['mean'] - base['quality']['structural_similarity']['mean']) / base['quality']['structural_similarity']['mean'] * 100):+.1f}%"]
        ]
        
        table = ax1.table(cellText=metrics_data, loc='center', cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1.2, 2)
        
        # Style the header row
        for i in range(len(metrics_data[0])):
            table[(0, i)].set_facecolor('#4CAF50')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        ax1.set_title('Performance Comparison Summary', fontsize=16, fontweight='bold', pad=20)
        
        # Add more subplot visualizations here as needed
        # (You can add the individual plots from the other methods)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/comprehensive_dashboard.png", dpi=300, bbox_inches='tight')
        plt.close()

def main():
    parser = argparse.ArgumentParser(description='Analyze benchmark results')
    parser.add_argument('--base_model_file', type=str, required=True,
                       help='Path to base model benchmark JSON file')
    parser.add_argument('--finetuned_model_file', type=str, required=True,
                       help='Path to fine-tuned model benchmark JSON file')
    parser.add_argument('--output_dir', type=str, default='./benchmark_analysis',
                       help='Output directory for analysis results')
    
    args = parser.parse_args()
    
    analyzer = BenchmarkAnalyzer()
    
    # Load data
    print("Loading benchmark data...")
    base_data = analyzer.load_benchmark_data(args.base_model_file)
    finetuned_data = analyzer.load_benchmark_data(args.finetuned_model_file)
    
    # print(pprint.pformat(base_data))

    # Perform comparison
    print("Analyzing metrics...")
    comparison = analyzer.compare_models(base_data, finetuned_data)
    
    # Create visualizations
    print("Creating visualizations...")
    analyzer.create_visualizations(comparison, args.output_dir)
    
    # Save detailed comparison
    with open(f"{args.output_dir}/detailed_comparison.json", 'w') as f:
        json.dump(comparison, f, indent=2, default=str)
    
    print(f"Analysis complete! Results saved to {args.output_dir}")

if __name__ == "__main__":
    main()