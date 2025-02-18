
import os
import numpy as np
import json
import torch
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, roc_auc_score
from scipy.stats import bootstrap
from dataloader.feature_engineering import construct_enhanced_data
from contrastivemodel import ContrastiveModel  
from loss import compute_total_loss  

def run_experiment_mlp_contrastive(filepath, model_type, standardize=False, n_iterations=100, output_folder="outputs"):
    enhanced_data = construct_enhanced_data(filepath, model_type)

    features = enhanced_data.drop(columns=['SEASON', 'ROUND', 'NUMBER', 'TEAM_home', 'RESULT1_home'])
    y = enhanced_data['RESULT1_home']

    results_baseline = {'accuracy': [], 'f1': [], 'recall': [], 'precision': [], 'auroc': []}
    results_enhanced = {'accuracy': [], 'f1': [], 'recall': [], 'precision': [], 'auroc': []}

    for _ in range(n_iterations):
        X_train, X_test, y_train, y_test = train_test_split(features, y, test_size=0.2)
        
        X_train_baseline = X_train.drop(columns=['avg_REB_Percent', 'avg_AS_Percent', 'avg_ST_Percent', 'avg_BS_Percent',
                                                 'avg_FOUL_Percent', 'avg_TO_Percent', 'avg_FG%_Percent', 'avg_eFG%_Percent',
                                                 'avg_TS%_Percent', 'avg_ORTG', 'avg_DRTG', 'HOME_ELO', 'AWAY_ELO', 
                                                 'HOME_RECENT_ELO_CHANGE', 'AWAY_RECENT_ELO_CHANGE'])
        X_test_baseline = X_test.drop(columns=['avg_REB_Percent', 'avg_AS_Percent', 'avg_ST_Percent', 'avg_BS_Percent',
                                                'avg_FOUL_Percent', 'avg_TO_Percent', 'avg_FG%_Percent', 'avg_eFG%_Percent',
                                                'avg_TS%_Percent', 'avg_ORTG', 'avg_DRTG', 'HOME_ELO', 'AWAY_ELO', 
                                                'HOME_RECENT_ELO_CHANGE', 'AWAY_RECENT_ELO_CHANGE'])
        
        X_train_enhanced = X_train
        X_test_enhanced = X_test

        if standardize:
            scaler_baseline = StandardScaler()
            X_train_baseline = scaler_baseline.fit_transform(X_train_baseline)
            X_test_baseline = scaler_baseline.transform(X_test_baseline)
            
            scaler_enhanced = StandardScaler()
            X_train_enhanced = scaler_enhanced.fit_transform(X_train_enhanced)
            X_test_enhanced = scaler_enhanced.transform(X_test_enhanced)
            
        X_train_baseline_tensor = torch.tensor(X_train_baseline, dtype=torch.float32)
        X_test_baseline_tensor = torch.tensor(X_test_baseline, dtype=torch.float32)
        X_train_enhanced_tensor = torch.tensor(X_train_enhanced, dtype=torch.float32)
        X_test_enhanced_tensor = torch.tensor(X_test_enhanced, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32)

        input_dim_baseline = X_train_baseline.shape[1]
        model_baseline = ContrastiveModel(input_dim=input_dim_baseline)
        optimizer = optim.Adam(model_baseline.parameters(), lr=0.001)

        for epoch in range(300):  
            model_baseline.train()
            optimizer.zero_grad()
            output, embeddings = model_baseline(X_train_baseline_tensor)


            total_loss = compute_total_loss(output, embeddings, y_train_tensor, lambda_contra=10.0)
            total_loss.backward()
            optimizer.step()
            
        model_baseline.eval()
        with torch.no_grad():
            output_baseline, _ = model_baseline(X_test_baseline_tensor)
            y_pred_baseline = (torch.sigmoid(output_baseline.squeeze()) > 0.5).numpy()
            y_prob_baseline = torch.sigmoid(output_baseline.squeeze()).numpy()

        results_baseline['accuracy'].append(accuracy_score(y_test, y_pred_baseline))
        results_baseline['f1'].append(f1_score(y_test, y_pred_baseline))
        results_baseline['recall'].append(recall_score(y_test, y_pred_baseline))
        results_baseline['precision'].append(precision_score(y_test, y_pred_baseline))
        results_baseline['auroc'].append(roc_auc_score(y_test, y_prob_baseline))

        input_dim_enhanced = X_train_enhanced.shape[1]
        model_enhanced = ContrastiveModel(input_dim=input_dim_enhanced)
        optimizer = optim.Adam(model_enhanced.parameters(), lr=0.001)

        for epoch in range(300):  
            model_enhanced.train()
            optimizer.zero_grad()
            output, embeddings = model_enhanced(X_train_enhanced_tensor)


            total_loss = compute_total_loss(output, embeddings, y_train_tensor, lambda_contra=10.0)
            total_loss.backward()
            optimizer.step()


        model_enhanced.eval()
        with torch.no_grad():
            output_enhanced, _ = model_enhanced(X_test_enhanced_tensor)
            y_pred_enhanced = (torch.sigmoid(output_enhanced.squeeze()) > 0.5).numpy()
            y_prob_enhanced = torch.sigmoid(output_enhanced.squeeze()).numpy()


        results_enhanced['accuracy'].append(accuracy_score(y_test, y_pred_enhanced))
        results_enhanced['f1'].append(f1_score(y_test, y_pred_enhanced))
        results_enhanced['recall'].append(recall_score(y_test, y_pred_enhanced))
        results_enhanced['precision'].append(precision_score(y_test, y_pred_enhanced))
        results_enhanced['auroc'].append(roc_auc_score(y_test, y_prob_enhanced))


    avg_results_baseline = {metric: np.mean(scores) for metric, scores in results_baseline.items()}
    avg_results_enhanced = {metric: np.mean(scores) for metric, scores in results_enhanced.items()}
    
    results_summary = {
        "baseline": avg_results_baseline,
        "enhanced": avg_results_enhanced
    }

    output_path = os.path.join(output_folder, f"results_{model_type}_ContrastiveMLP.json")
    with open(output_path, "w") as f:
        json.dump(results_summary, f, indent=4)


    print(f"{'MLP Contrastive With Standardization' if standardize else 'MLP Contrastive Without Standardization'}:")
    print("Baseline Model:")
    print(f"Average Accuracy: {avg_results_baseline['accuracy']:.4f}")
    print(f"Average F1 Score: {avg_results_baseline['f1']:.4f}")
    print(f"Average Recall: {avg_results_baseline['recall']:.4f}")
    print(f"Average Precision: {avg_results_baseline['precision']:.4f}")
    print(f"Average AUROC: {avg_results_baseline['auroc']:.4f}")

    print("\nEnhanced Model:")
    print(f"Average Accuracy: {avg_results_enhanced['accuracy']:.4f}")
    print(f"Average F1 Score: {avg_results_enhanced['f1']:.4f}")
    print(f"Average Recall: {avg_results_enhanced['recall']:.4f}")
    print(f"Average Precision: {avg_results_enhanced['precision']:.4f}")
    print(f"Average AUROC: {avg_results_enhanced['auroc']:.4f}")

    print("\nSignificance Tests using Bootstrap:")
    n_bootstrap = 10000  
    for metric in ['accuracy', 'f1', 'recall', 'precision', 'auroc']:
        diff = np.array(results_enhanced[metric]) - np.array(results_baseline[metric])
        
        conf_int = bootstrap((diff,), np.mean, confidence_level=0.95, n_resamples=n_bootstrap, method='basic').confidence_interval
        is_significant = conf_int.low > 0  
        significance = "Significant" if is_significant else "Not Significant"
        print(f"{metric.capitalize()} Significance (95% CI): {conf_int} - {significance}")
