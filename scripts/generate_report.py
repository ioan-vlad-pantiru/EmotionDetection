"""
Generate comprehensive evaluation report with charts and visualizations.
"""
import sys
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import METRICS_DIR, REPORTS_DIR, LANGUAGES, MODEL_TYPES


def load_all_metrics():
    """Load all metrics files."""
    metrics = {}
    for lang in LANGUAGES:
        metrics[lang] = {}
        for model_type in MODEL_TYPES:
            metrics_file = METRICS_DIR / f"{lang}_{model_type}_metrics.json"
            if metrics_file.exists():
                with open(metrics_file, "r") as f:
                    metrics[lang][model_type] = json.load(f)
    return metrics


def create_comparison_charts(metrics, output_dir):
    """Create comparison charts across models."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (12, 6)
    
    # Prepare data for comparison
    comparison_data = []
    for lang in LANGUAGES:
        if lang not in metrics:
            continue
        for model_type in MODEL_TYPES:
            if model_type not in metrics[lang]:
                continue
            m = metrics[lang][model_type]
            comparison_data.append({
                "Language": lang.upper(),
                "Model": model_type,
                "Accuracy": m["accuracy"],
                "Macro F1": m["macro_f1"],
                "Weighted F1": m["weighted_f1"],
            })
    
    if not comparison_data:
        print("No metrics found to plot")
        return
    
    df = pd.DataFrame(comparison_data)
    
    # 1. Overall metrics comparison
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    metrics_to_plot = ["Accuracy", "Macro F1", "Weighted F1"]
    for idx, metric in enumerate(metrics_to_plot):
        ax = axes[idx]
        pivot_df = df.pivot(index="Language", columns="Model", values=metric)
        pivot_df.plot(kind="bar", ax=ax, rot=0)
        ax.set_title(f"{metric} Comparison")
        ax.set_ylabel(metric)
        ax.legend(title="Model Type")
        ax.grid(axis="y", alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / "comparison_overall.png", dpi=300, bbox_inches="tight")
    plt.close()
    
    # 2. Per-language comparison
    for lang in LANGUAGES:
        if lang not in metrics:
            continue
        
        lang_data = [d for d in comparison_data if d["Language"] == lang.upper()]
        if not lang_data:
            continue
        
        lang_df = pd.DataFrame(lang_data)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        x = np.arange(len(MODEL_TYPES))
        width = 0.25
        
        metrics_to_plot = ["Accuracy", "Macro F1", "Weighted F1"]
        for i, metric in enumerate(metrics_to_plot):
            values = [lang_df[lang_df["Model"] == mt][metric].values[0] 
                     if len(lang_df[lang_df["Model"] == mt]) > 0 else 0 
                     for mt in MODEL_TYPES]
            ax.bar(x + i * width, values, width, label=metric)
        
        ax.set_xlabel("Model Type")
        ax.set_ylabel("Score")
        ax.set_title(f"Model Comparison - {lang.upper()}")
        ax.set_xticks(x + width)
        ax.set_xticklabels(MODEL_TYPES)
        ax.legend()
        ax.grid(axis="y", alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / f"comparison_{lang}.png", dpi=300, bbox_inches="tight")
        plt.close()


def create_per_class_charts(metrics, output_dir):
    """Create per-class performance charts."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for lang in LANGUAGES:
        if lang not in metrics:
            continue
        
        # Collect per-class metrics for all models
        fig, axes = plt.subplots(1, len(MODEL_TYPES), figsize=(6 * len(MODEL_TYPES), 6))
        if len(MODEL_TYPES) == 1:
            axes = [axes]
        
        for idx, model_type in enumerate(MODEL_TYPES):
            if model_type not in metrics[lang]:
                continue
            
            ax = axes[idx]
            m = metrics[lang][model_type]
            
            # Extract per-class metrics
            classes = []
            precisions = []
            recalls = []
            f1_scores = []
            
            for label, scores in m["per_class"].items():
                classes.append(label)
                precisions.append(scores["precision"])
                recalls.append(scores["recall"])
                f1_scores.append(scores["f1"])
            
            x = np.arange(len(classes))
            width = 0.25
            
            ax.bar(x - width, precisions, width, label="Precision", alpha=0.8)
            ax.bar(x, recalls, width, label="Recall", alpha=0.8)
            ax.bar(x + width, f1_scores, width, label="F1", alpha=0.8)
            
            ax.set_xlabel("Emotion")
            ax.set_ylabel("Score")
            ax.set_title(f"{model_type.upper()} - {lang.upper()}")
            ax.set_xticks(x)
            ax.set_xticklabels(classes, rotation=45, ha="right")
            ax.legend()
            ax.grid(axis="y", alpha=0.3)
            ax.set_ylim([0, 1])
        
        plt.tight_layout()
        plt.savefig(output_dir / f"per_class_{lang}.png", dpi=300, bbox_inches="tight")
        plt.close()


def create_confusion_matrices(metrics, output_dir):
    """Create confusion matrix heatmaps."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for lang in LANGUAGES:
        if lang not in metrics:
            continue
        
        fig, axes = plt.subplots(1, len(MODEL_TYPES), figsize=(6 * len(MODEL_TYPES), 5))
        if len(MODEL_TYPES) == 1:
            axes = [axes]
        
        for idx, model_type in enumerate(MODEL_TYPES):
            if model_type not in metrics[lang]:
                continue
            
            ax = axes[idx]
            m = metrics[lang][model_type]
            
            cm = np.array(m["confusion_matrix"])
            labels = m["labels"]
            
            # Normalize confusion matrix
            cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            
            sns.heatmap(
                cm_normalized,
                annot=True,
                fmt='.2f',
                cmap='Blues',
                xticklabels=labels,
                yticklabels=labels,
                ax=ax,
                cbar_kws={'label': 'Normalized Count'}
            )
            ax.set_title(f"{model_type.upper()} - {lang.upper()}")
            ax.set_xlabel("Predicted")
            ax.set_ylabel("True")
        
        plt.tight_layout()
        plt.savefig(output_dir / f"confusion_matrix_{lang}.png", dpi=300, bbox_inches="tight")
        plt.close()


def create_summary_report(metrics, output_dir):
    """Create a markdown summary report."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    report_path = output_dir / "evaluation_report.md"
    
    with open(report_path, "w") as f:
        f.write("# Emotion Detection Evaluation Report\n\n")
        f.write("## Overview\n\n")
        f.write("This report summarizes the performance of different emotion detection models.\n\n")
        
        # Overall comparison table
        f.write("## Overall Performance Comparison\n\n")
        
        comparison_data = []
        for lang in LANGUAGES:
            if lang not in metrics:
                continue
            for model_type in MODEL_TYPES:
                if model_type not in metrics[lang]:
                    continue
                m = metrics[lang][model_type]
                comparison_data.append({
                    "Language": lang.upper(),
                    "Model": model_type,
                    "Accuracy": f"{m['accuracy']:.4f}",
                    "Macro F1": f"{m['macro_f1']:.4f}",
                    "Weighted F1": f"{m['weighted_f1']:.4f}",
                })
        
        if comparison_data:
            df = pd.DataFrame(comparison_data)
            f.write(df.to_markdown(index=False))
            f.write("\n\n")
        
        # Per-language detailed results
        for lang in LANGUAGES:
            if lang not in metrics:
                continue
            
            f.write(f"## {lang.upper()} - Detailed Results\n\n")
            
            for model_type in MODEL_TYPES:
                if model_type not in metrics[lang]:
                    continue
                
                m = metrics[lang][model_type]
                f.write(f"### {model_type.upper()} Model\n\n")
                f.write(f"- **Accuracy**: {m['accuracy']:.4f}\n")
                f.write(f"- **Macro F1**: {m['macro_f1']:.4f}\n")
                f.write(f"- **Weighted F1**: {m['weighted_f1']:.4f}\n\n")
                
                # Per-class table
                f.write("#### Per-Class Performance\n\n")
                class_data = []
                for label, scores in m["per_class"].items():
                    class_data.append({
                        "Emotion": label,
                        "Precision": f"{scores['precision']:.3f}",
                        "Recall": f"{scores['recall']:.3f}",
                        "F1": f"{scores['f1']:.3f}",
                        "Support": scores['support']
                    })
                
                class_df = pd.DataFrame(class_data)
                f.write(class_df.to_markdown(index=False))
                f.write("\n\n")
        
        # Charts section
        f.write("## Visualizations\n\n")
        f.write("The following charts are available:\n\n")
        f.write("- `comparison_overall.png`: Overall metrics comparison across all models\n")
        f.write("- `comparison_en.png`: Model comparison for English\n")
        f.write("- `comparison_ro.png`: Model comparison for Romanian\n")
        f.write("- `per_class_en.png`: Per-class performance for English models\n")
        f.write("- `per_class_ro.png`: Per-class performance for Romanian models\n")
        f.write("- `confusion_matrix_en.png`: Confusion matrices for English models\n")
        f.write("- `confusion_matrix_ro.png`: Confusion matrices for Romanian models\n\n")
    
    print(f"Summary report saved to {report_path}")


def main():
    print("=" * 60)
    print("Generating Evaluation Report and Charts")
    print("=" * 60)
    
    # Create output directory
    charts_dir = REPORTS_DIR / "charts"
    charts_dir.mkdir(parents=True, exist_ok=True)
    
    # Load metrics
    print("\nLoading metrics...")
    metrics = load_all_metrics()
    
    if not metrics:
        print("No metrics found. Please run evaluation first:")
        print("  python scripts/evaluate_all.py")
        return
    
    # Generate charts
    print("\nGenerating comparison charts...")
    create_comparison_charts(metrics, charts_dir)
    
    print("Generating per-class performance charts...")
    create_per_class_charts(metrics, charts_dir)
    
    print("Generating confusion matrices...")
    create_confusion_matrices(metrics, charts_dir)
    
    # Generate summary report
    print("Generating summary report...")
    create_summary_report(metrics, REPORTS_DIR)
    
    print("\n" + "=" * 60)
    print("Report generation complete!")
    print("=" * 60)
    print(f"\nCharts saved to: {charts_dir}")
    print(f"Summary report: {REPORTS_DIR / 'evaluation_report.md'}")
    print("\nAvailable visualizations:")
    for chart_file in sorted(charts_dir.glob("*.png")):
        print(f"  - {chart_file.name}")


if __name__ == "__main__":
    main()

