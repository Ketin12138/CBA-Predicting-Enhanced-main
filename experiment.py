import os
import numpy as np
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, roc_auc_score
from scipy.stats import bootstrap
from models import (
    SVMModel, NaiveBayesModel, LogisticRegressionModel,
    KNNModel, XGBoostModel)

from dataloader.feature_engineering import construct_enhanced_data

def run_experiment(filepath, model_type, algorithm, standardize=False, n_iterations=500, output_folder="outputs"):
    os.makedirs(output_folder, exist_ok=True)
    enhanced_data = construct_enhanced_data(filepath, model_type)

    features = enhanced_data.drop(columns=['SEASON', 'ROUND', 'NUMBER', 'TEAM_home', 'RESULT1_home'])
    y = enhanced_data['RESULT1_home']

    results_baseline = {'accuracy': [], 'f1': [], 'recall': [], 'precision': [], 'auroc': []}
    results_enhanced = {'accuracy': [], 'f1': [], 'recall': [], 'precision': [], 'auroc': []}

    for _ in range(n_iterations):
        X_train, X_test, y_train, y_test = train_test_split(features, y, test_size=0.2)
        
        baseline_features = ['avg_REB_Percent', 'avg_AS_Percent', 'avg_ST_Percent', 'avg_BS_Percent',
                             'avg_FOUL_Percent', 'avg_TO_Percent', 'avg_FG%_Percent', 'avg_eFG%_Percent',
                             'avg_TS%_Percent', 'avg_ORTG', 'avg_DRTG', 'HOME_ELO', 'AWAY_ELO', 
                             'HOME_RECENT_ELO_CHANGE', 'AWAY_RECENT_ELO_CHANGE']
        X_train_baseline = X_train.drop(columns=baseline_features)
        X_test_baseline = X_test.drop(columns=baseline_features)

        X_train_enhanced = X_train
        X_test_enhanced = X_test

        if standardize:
            scaler_baseline = StandardScaler()
            X_train_baseline = scaler_baseline.fit_transform(X_train_baseline)
            X_test_baseline = scaler_baseline.transform(X_test_baseline)
            
            scaler_enhanced = StandardScaler()
            X_train_enhanced = scaler_enhanced.fit_transform(X_train_enhanced)
            X_test_enhanced = scaler_enhanced.transform(X_test_enhanced)

        models = {
            "SVM": SVMModel(), "NaiveBayes": NaiveBayesModel(), "LogisticRegression": LogisticRegressionModel(),
            "KNN": KNNModel(), "XGBoost": XGBoostModel()
        }

        model_base = models[algorithm]
        model_base.train(X_train_baseline, y_train)
        y_pred_baseline = model_base.predict(X_test_baseline)
        y_prob_baseline = model_base.predict_proba(X_test_baseline)

        results_baseline['accuracy'].append(accuracy_score(y_test, y_pred_baseline))
        results_baseline['f1'].append(f1_score(y_test, y_pred_baseline))
        results_baseline['recall'].append(recall_score(y_test, y_pred_baseline))
        results_baseline['precision'].append(precision_score(y_test, y_pred_baseline))
        results_baseline['auroc'].append(roc_auc_score(y_test, y_prob_baseline))
        
        model_enhanced = models[algorithm]
        model_enhanced.train(X_train_enhanced, y_train)
        y_pred_enhanced = model_enhanced.predict(X_test_enhanced)
        y_prob_enhanced = model_enhanced.predict_proba(X_test_enhanced)

        results_enhanced['accuracy'].append(accuracy_score(y_test, y_pred_enhanced))
        results_enhanced['f1'].append(f1_score(y_test, y_pred_enhanced))
        results_enhanced['recall'].append(recall_score(y_test, y_pred_enhanced))
        results_enhanced['precision'].append(precision_score(y_test, y_pred_enhanced))
        results_enhanced['auroc'].append(roc_auc_score(y_test, y_prob_enhanced))

    avg_results_baseline = {metric: np.mean(scores) for metric, scores in results_baseline.items()}
    avg_results_enhanced = {metric: np.mean(scores) for metric, scores in results_enhanced.items()}

    results_summary = {
        "model_type": model_type,
        "algorithm": algorithm,
        "baseline": avg_results_baseline,
        "enhanced": avg_results_enhanced
    }

    output_path = os.path.join(output_folder, f"results_{model_type}_{algorithm}.json")
    with open(output_path, "w") as f:
        json.dump(results_summary, f, indent=4)


    print(f"{algorithm} | {model_type} vs Enhanced {model_type}")
    print(f"Baseline Model ({model_type}):")
    for metric, value in avg_results_baseline.items():
        print(f"  {metric.capitalize()}: {value:.4f}")

    print(f"\nEnhanced Model (Enhanced {model_type}):")
    for metric, value in avg_results_enhanced.items():
        print(f"  {metric.capitalize()}: {value:.4f}")
    
    print("\nSignificance Tests using Bootstrap:")
    n_bootstrap = 10000  
    for metric in ['accuracy', 'f1', 'recall', 'precision', 'auroc']:
        diff = np.array(results_enhanced[metric]) - np.array(results_baseline[metric])
        conf_int = bootstrap((diff,), np.mean, confidence_level=0.95, n_resamples=n_bootstrap, method='basic').confidence_interval
        is_significant = conf_int.low > 0  
        significance = "Significant" if is_significant else "Not Significant"
        print(f"{metric.capitalize()} Significance (95% CI): {conf_int} - {significance}")
    
    print("\n" + "="*50 + "\n")