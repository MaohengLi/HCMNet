import os
import logging
import numpy as np
import torch
from config import set_seed
from trainer import run_train_fold

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    PARAMS = {'bs': 16, 'eps': 30, 'lr': 5e-5, 'enc_h': 1024, 'enc_o': 512, 'gnn_h': 1024, 'heads': 8, 'drop': 0.3}
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    CV_DIR = './final_dataset_cv'
    set_seed(42)
    metrics_summary = {
        "non-sarcastic": {"precision": [], "recall": [], "f1-score": []},
        "sarcastic": {"precision": [], "recall": [], "f1-score": []},
        "macro avg": {"precision": [], "recall": [], "f1-score": []}
    }
    accs = []
    for i in range(1, 6):
        fold_file = os.path.join(CV_DIR, f'mustard_final_dataset_fold_{i}.pkl')
        if not os.path.exists(fold_file):
            logger.warning(f"File not found: {fold_file}")
            continue
        logger.info(f"Training Fold {i}...")
        acc, report = run_train_fold(fold_file, DEVICE, PARAMS)
        accs.append(acc)
        for key in metrics_summary.keys():
            for metric in ["precision", "recall", "f1-score"]:
                metrics_summary[key][metric].append(report[key][metric])
        logger.info(f"Fold {i} Accuracy: {acc:.4f}")
    if accs:
        logger.info("Summary Report:")
        for key, vals in metrics_summary.items():
            res = [f"{np.mean(vals[m]):.4f}±{np.std(vals[m]):.3f}" for m in ["precision", "recall", "f1-score"]]
            logger.info(f"{key:<20} | P: {res[0]} | R: {res[1]} | F1: {res[2]}")
        logger.info(f"{'Overall Accuracy':<20} | {np.mean(accs):.4f} ± {np.std(accs):.4f}")