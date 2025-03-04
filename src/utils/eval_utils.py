from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F 

def evaluate_anomaly_detection(model, data, gt_labels_dict):
    """Evaluate anomaly detection performance."""
    # For PyTorch models, set evaluation mode
    if hasattr(model, 'eval'):
        model.eval()
    
    with torch.no_grad():
        # Get anomaly scores
        if hasattr(model, 'compute_anomaly_scores'):
            scores = model.compute_anomaly_scores(data)
        else:
            x_hat_member, x_hat_provider, _ = model(data)
            
            # Calculate reconstruction errors
            member_scores = F.mse_loss(x_hat_member, data['member'].x, reduction='none').sum(dim=1)
            provider_scores = F.mse_loss(x_hat_provider, data['provider'].x, reduction='none').sum(dim=1)
            
            scores = {
                'member': member_scores,
                'provider': provider_scores
            }
    
    # Calculate metrics
    results = {}
    for node_type, gt_labels in gt_labels_dict.items():
        if node_type not in scores:
            continue
            
        y_true = gt_labels.cpu().numpy()
        y_score = scores[node_type].cpu().numpy()
        
        auc = roc_auc_score(y_true, y_score)
        ap = average_precision_score(y_true, y_score)
        
        results[node_type] = {
            'auc': auc,
            'ap': ap,
            'scores': y_score,
            'labels': y_true
        }
    
    return results


def plot_precision_recall_curves(results_dict, model_names, node_type='member'):
    """Plot precision-recall curves for multiple models."""
    plt.figure(figsize=(10, 8))
    
    for model_name in model_names:
        if model_name not in results_dict:
            continue
            
        model_results = results_dict[model_name]
        if node_type not in model_results:
            continue
            
        y_true = model_results[node_type]['labels']
        y_score = model_results[node_type]['scores']
        
        precision, recall, _ = precision_recall_curve(y_true, y_score)
        ap = model_results[node_type]['ap']
        
        plt.plot(recall, precision, lw=2, label=f'{model_name} (AP = {ap:.3f})')
    
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curves ({node_type}s)')
    plt.legend(loc="best")
    plt.grid(True)
    plt.show()

def plot_anomaly_distribution(results_dict, model_names, node_type='member'):
    """Plot distribution of anomaly scores for normal and anomalous nodes."""
    plt.figure(figsize=(15, 10))
    
    for i, model_name in enumerate(model_names):
        if model_name not in results_dict:
            continue
            
        model_results = results_dict[model_name]
        if node_type not in model_results:
            continue
            
        plt.subplot(len(model_names), 1, i+1)
        
        scores = model_results[node_type]['scores']
        labels = model_results[node_type]['labels']
        
        normal_scores = scores[labels == 0]
        anomaly_scores = scores[labels == 1]
        
        plt.hist(normal_scores, bins=50, alpha=0.5, label='Normal', density=True)
        plt.hist(anomaly_scores, bins=50, alpha=0.5, label='Anomalous', density=True)
        
        plt.title(f'{model_name}')
        plt.xlabel('Anomaly Score')
        plt.ylabel('Density')
        plt.legend()
    
    plt.tight_layout()
    plt.show()

def compare_models(results_dict, model_names):
    """Compare model performance metrics in a table format."""
    rows = []
    
    # Collect results for each model
    for model_name in model_names:
        if model_name not in results_dict:
            continue
            
        model_results = results_dict[model_name]
        
        # Calculate average metrics across node types
        avg_auc = 0
        avg_ap = 0
        count = 0
        
        for node_type in ['member', 'provider']:
            if node_type in model_results:
                avg_auc += model_results[node_type]['auc']
                avg_ap += model_results[node_type]['ap']
                count += 1
        
        if count > 0:
            avg_auc /= count
            avg_ap /= count
        
        # Add results row
        row = {
            'Model': model_name,
        }
        
        for node_type in ['member', 'provider']:
            if node_type in model_results:
                row[f'{node_type.capitalize()} AUC'] = model_results[node_type]['auc']
                row[f'{node_type.capitalize()} AP'] = model_results[node_type]['ap']
            else:
                row[f'{node_type.capitalize()} AUC'] = 'N/A'
                row[f'{node_type.capitalize()} AP'] = 'N/A'
        
        row['Avg AUC'] = avg_auc
        row['Avg AP'] = avg_ap
        
        rows.append(row)
    
    # Convert to DataFrame for nice display
    results_df = pd.DataFrame(rows)
    
    # Sort by average AUC (descending)
    results_df = results_df.sort_values('Avg AUC', ascending=False)
    
    return results_df


def evaluate_model_with_complex_anomalies(model, data, gt_labels_dict, methods=['structural', 'feature']):
    """Evaluate model performance on different types of anomalies separately."""
    
    # Store original data
    original_data = data.clone()
    results_by_method = {}
    
    for method in methods:
        print(f"\nEvaluating on {method} anomalies...")
        
        # Inject only this type of anomaly
        if method == 'structural':
            modified_data, method_gt_labels = inject_structural_anomalies(original_data, percentage=0.03)
        elif method == 'feature':
            modified_data, method_gt_labels = inject_feature_anomalies(original_data, percentage=0.04)
        elif method == 'healthcare':
            modified_data, method_gt_labels = inject_healthcare_fraud_patterns(
                original_data, df_provider_features, df_member_features, percentage=0.03
            )
        else:
            continue
        
        # Evaluate model on this specific type of anomaly
        results = evaluate_anomaly_detection(model, modified_data, method_gt_labels)
        results_by_method[method] = results
        
        # Print results
        for node_type in results:
            print(f"{node_type} - AUC: {results[node_type]['auc']:.4f}, AP: {results[node_type]['ap']:.4f}")
    
    # Overall evaluation with mixed anomalies
    print("\nEvaluating on mixed anomalies...")
    mixed_results = evaluate_anomaly_detection(model, data, gt_labels_dict)
    results_by_method['mixed'] = mixed_results
    
    for node_type in mixed_results:
        print(f"{node_type} - AUC: {mixed_results[node_type]['auc']:.4f}, AP: {mixed_results[node_type]['ap']:.4f}")
    
    return results_by_method
