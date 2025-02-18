import argparse
import os
from experiment import run_experiment
from train import run_experiment_mlp_contrastive

# 设置数据集路径
DATA_DIR = "data"
filepath = os.path.join(DATA_DIR, "CBA2020-2024数据集.xlsx")

# 解析命令行参数
def parse_args():
    parser = argparse.ArgumentParser(description="Run basketball game prediction models.")
    parser.add_argument("--model_type", type=str, required=True, choices=[
        "FourFactors", "FourFactors_detailed", "DefenseOfense", "DefenseOfense_detailed"
    ], help="Choose the model type.")
    parser.add_argument("--algorithm", type=str, required=True, choices=[
        "SVM", "NaiveBayes", "LogisticRegression", "KNN", "XGBoost", "MLP_Contrastive"
    ], help="Choose the algorithm.")
    parser.add_argument("--standardize", type=bool, default=True, help="Whether to standardize the data.")
    return parser.parse_args()

def main():
    args = parse_args()

    if args.algorithm == "MLP_Contrastive":
        run_experiment_mlp_contrastive(filepath, model_type=args.model_type, standardize=args.standardize)
    else:
        run_experiment(filepath, model_type=args.model_type, algorithm=args.algorithm, standardize=args.standardize)

if __name__ == "__main__":
    main()


