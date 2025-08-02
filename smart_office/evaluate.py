#!/usr/bin/env python3
"""
Smart Office Challenge - Evaluation Script
Evaluates trained model and provides detailed metrics
"""

import os
import json
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from ultralytics import YOLO
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OfficeObjectEvaluator:
    def __init__(self, model_path, data_path, save_plots=True):
        """
        Initialize evaluator
        
        Args:
            model_path (str): Path to trained model
            data_path (str): Path to dataset YAML
            save_plots (bool): Whether to save evaluation plots
        """
        self.model_path = model_path
        self.data_path = data_path
        self.save_plots = save_plots
        self.class_names = ['person', 'chair', 'monitor', 'keyboard', 'laptop', 'phone']
        self.results_dir = Path('evaluation_results')
        self.results_dir.mkdir(exist_ok=True)
        
    def load_model(self):
        """Load the trained model"""
        try:
            self.model = YOLO(self.model_path)
            logger.info(f"Loaded model from: {self.model_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False
    
    def run_evaluation(self):
        """Run comprehensive evaluation"""
        try:
            # Run validation on test set
            logger.info("Running model validation...")
            results = self.model.val(
                data=self.data_path,
                imgsz=640,
                batch=16,
                conf=0.25,
                iou=0.45,
                max_det=300,
                half=False,
                device='cpu',
                dnn=False,
                plots=True,
                save_json=True,
                save_hybrid=False,
                verbose=True,
                split='test',
                save_txt=False,
                save_conf=False,
                save_crop=False,
                show_labels=True,
                show_conf=True,
                visualize=False,
                augment=False,
                agnostic_nms=False,
                retina_masks=False,
                embed=None,
                project='runs/val',
                name='office_eval',
                exist_ok=True
            )
            
            return results
            
        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            return None
    
    def extract_detailed_metrics(self, results):
        """Extract detailed metrics from validation results"""
        try:
            # Overall metrics
            overall_metrics = {
                'mAP50': float(results.box.map50),
                'mAP50-95': float(results.box.map),
                'precision': float(results.box.mp),
                'recall': float(results.box.mr),
                'f1_score': float(2 * results.box.mp * results.box.mr / (results.box.mp + results.box.mr))
            }
            
            # Class-wise metrics
            class_metrics = {}
            if hasattr(results.box, 'maps') and results.box.maps is not None:
                maps = results.box.maps  # mAP50-95 per class
                map50s = results.box.map50s if hasattr(results.box, 'map50s') else None  # mAP50 per class
                
                for i, class_name in enumerate(self.class_names):
                    if i < len(maps):
                        class_metrics[class_name] = {
                            'mAP50-95': float(maps[i]),
                            'mAP50': float(map50s[i]) if map50s is not None and i < len(map50s) else 0.0
                        }
            
            # If no per-class metrics available, create placeholders
            if not class_metrics:
                for class_name in self.class_names:
                    class_metrics[class_name] = {
                        'mAP50-95': 0.0,
                        'mAP50': 0.0,
                        'precision': 0.0,
                        'recall': 0.0,
                        'f1_score': 0.0
                    }
            
            return overall_metrics, class_metrics
            
        except Exception as e:
            logger.error(f"Failed to extract metrics: {e}")
            return None, None
    
    def create_evaluation_report(self, overall_metrics, class_metrics):
        """Create comprehensive evaluation report"""
        try:
            # Create DataFrame for class-wise metrics
            df_metrics = pd.DataFrame(class_metrics).T
            df_metrics = df_metrics.round(4)
            
            # Generate report
            report = {
                'model_path': self.model_path,
                'evaluation_date': pd.Timestamp.now().isoformat(),
                'overall_metrics': overall_metrics,
                'class_wise_metrics': class_metrics,
                'summary': {
                    'total_classes': len(self.class_names),
                    'best_performing_class': df_metrics['mAP50-95'].idxmax() if not df_metrics.empty else 'N/A',
                    'worst_performing_class': df_metrics['mAP50-95'].idxmin() if not df_metrics.empty else 'N/A',
                    'average_class_map50': df_metrics['mAP50'].mean() if not df_metrics.empty else 0.0,
                    'average_class_map50_95': df_metrics['mAP50-95'].mean() if not df_metrics.empty else 0.0
                }
            }
            
            # Save report as JSON
            report_path = self.results_dir / 'evaluation_report.json'
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2)
            
            # Save metrics as CSV
            csv_path = self.results_dir / 'class_metrics.csv'
            df_metrics.to_csv(csv_path)
            
            logger.info(f"Evaluation report saved to: {report_path}")
            logger.info(f"Class metrics saved to: {csv_path}")
            
            return report
            
        except Exception as e:
            logger.error(f"Failed to create report: {e}")
            return None
    
    def create_visualizations(self, class_metrics):
        """Create evaluation visualizations"""
        if not self.save_plots or not class_metrics:
            return
        
        try:
            # Set style (compatible with older seaborn versions)
            try:
                plt.style.use('seaborn-v0_8')
            except OSError:
                try:
                    plt.style.use('seaborn')
                except OSError:
                    plt.style.use('default')
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle('Office Object Detection - Model Evaluation', fontsize=16)
            
            # Prepare data
            classes = list(class_metrics.keys())
            map50_values = [class_metrics[cls].get('mAP50', 0) for cls in classes]
            map50_95_values = [class_metrics[cls].get('mAP50-95', 0) for cls in classes]
            
            # Plot 1: mAP50 by class
            axes[0, 0].bar(classes, map50_values, color='skyblue', alpha=0.7)
            axes[0, 0].set_title('mAP@0.5 by Class')
            axes[0, 0].set_ylabel('mAP@0.5')
            axes[0, 0].tick_params(axis='x', rotation=45)
            axes[0, 0].grid(True, alpha=0.3)
            
            # Plot 2: mAP50-95 by class
            axes[0, 1].bar(classes, map50_95_values, color='lightcoral', alpha=0.7)
            axes[0, 1].set_title('mAP@0.5:0.95 by Class')
            axes[0, 1].set_ylabel('mAP@0.5:0.95')
            axes[0, 1].tick_params(axis='x', rotation=45)
            axes[0, 1].grid(True, alpha=0.3)
            
            # Plot 3: Comparison radar chart (if we have both metrics)
            if map50_values and map50_95_values:
                angles = np.linspace(0, 2 * np.pi, len(classes), endpoint=False)
                angles = np.concatenate((angles, [angles[0]]))
                
                map50_values_plot = map50_values + [map50_values[0]]
                map50_95_values_plot = map50_95_values + [map50_95_values[0]]
                
                ax = axes[1, 0]
                ax.plot(angles, map50_values_plot, 'o-', linewidth=2, label='mAP@0.5', color='blue')
                ax.fill(angles, map50_values_plot, alpha=0.25, color='blue')
                ax.plot(angles, map50_95_values_plot, 'o-', linewidth=2, label='mAP@0.5:0.95', color='red')
                ax.fill(angles, map50_95_values_plot, alpha=0.25, color='red')
                ax.set_xticks(angles[:-1])
                ax.set_xticklabels(classes)
                ax.set_ylim(0, 1)
                ax.set_title('Performance Radar Chart')
                ax.legend()
                ax.grid(True)
            
            # Plot 4: Performance summary
            metrics_summary = {
                'mAP@0.5': np.mean(map50_values) if map50_values else 0,
                'mAP@0.5:0.95': np.mean(map50_95_values) if map50_95_values else 0
            }
            
            axes[1, 1].bar(metrics_summary.keys(), metrics_summary.values(), 
                          color=['green', 'orange'], alpha=0.7)
            axes[1, 1].set_title('Overall Performance Summary')
            axes[1, 1].set_ylabel('Score')
            axes[1, 1].set_ylim(0, 1)
            axes[1, 1].grid(True, alpha=0.3)
            
            # Add values on bars
            for i, (key, value) in enumerate(metrics_summary.items()):
                axes[1, 1].text(i, value + 0.01, f'{value:.3f}', 
                               ha='center', va='bottom', fontweight='bold')
            
            plt.tight_layout()
            
            # Save plot
            plot_path = self.results_dir / 'evaluation_plots.png'
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Evaluation plots saved to: {plot_path}")
            
        except Exception as e:
            logger.error(f"Failed to create visualizations: {e}")
    
    def print_evaluation_summary(self, overall_metrics, class_metrics):
        """Print evaluation summary to console"""
        print("\n" + "="*60)
        print("SMART OFFICE CHALLENGE - EVALUATION SUMMARY")
        print("="*60)
        
        if overall_metrics:
            print("\nOVERALL METRICS:")
            print("-" * 20)
            for metric, value in overall_metrics.items():
                print(f"{metric:12}: {value:.4f}")
        
        if class_metrics:
            print("\nCLASS-WISE METRICS:")
            print("-" * 20)
            print(f"{'Class':10} {'mAP@0.5':>8} {'mAP@0.5:0.95':>12}")
            print("-" * 32)
            for class_name, metrics in class_metrics.items():
                map50 = metrics.get('mAP50', 0)
                map50_95 = metrics.get('mAP50-95', 0)
                print(f"{class_name:10} {map50:8.4f} {map50_95:12.4f}")
        
        print("\n" + "="*60)
    
    def run_full_evaluation(self):
        """Run complete evaluation pipeline"""
        logger.info("Starting comprehensive evaluation...")
        
        # Load model
        if not self.load_model():
            return False
        
        # Run evaluation
        results = self.run_evaluation()
        if results is None:
            return False
        
        # Extract metrics
        overall_metrics, class_metrics = self.extract_detailed_metrics(results)
        if overall_metrics is None:
            return False
        
        # Create report
        report = self.create_evaluation_report(overall_metrics, class_metrics)
        
        # Create visualizations
        self.create_visualizations(class_metrics)
        
        # Print summary
        self.print_evaluation_summary(overall_metrics, class_metrics)
        
        logger.info("Evaluation completed successfully!")
        return True

def main():
    parser = argparse.ArgumentParser(description='Evaluate office object detection model')
    parser.add_argument('--model', type=str, default='best_office_model.pt',
                       help='Path to trained model')
    parser.add_argument('--data', type=str, default='dataset/data.yaml',
                       help='Path to dataset YAML file')
    parser.add_argument('--no-plots', action='store_true',
                       help='Skip creating evaluation plots')
    
    args = parser.parse_args()
    
    # Check if model exists
    if not os.path.exists(args.model):
        logger.error(f"Model file not found: {args.model}")
        logger.info("Please train a model first using train.py")
        return 1
    
    # Initialize evaluator
    evaluator = OfficeObjectEvaluator(
        model_path=args.model,
        data_path=args.data,
        save_plots=not args.no_plots
    )
    
    # Run evaluation
    success = evaluator.run_full_evaluation()
    
    return 0 if success else 1

if __name__ == "__main__":
    exit(main())