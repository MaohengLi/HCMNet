import os
import json
import pickle
import logging
from sklearn.model_selection import KFold
import config_align as cfg
from utils_align import set_seed, load_pickle, logger
from data_processor import align_modalities

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    logger.info("Starting modality alignment and Fold 4 extraction")

    try:
        with open(cfg.METADATA_JSON, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        feature_data = {key: load_pickle(path) for key, path in cfg.FEATURE_FILES.items()}
    except Exception as e:
        logger.error(f"Failed to load files: {e}")
        return

    aligned_samples_arr = align_modalities(metadata, feature_data)

    for run_idx in range(cfg.NUM_RUNS):
        logger.info(f"Run {run_idx + 1} with Seed {cfg.SEED}")
        set_seed(cfg.SEED)
        
        kf = KFold(n_splits=cfg.K_FOLDS, shuffle=True, random_state=cfg.SEED)
        
        for fold_idx, (train_idx, test_idx) in enumerate(kf.split(aligned_samples_arr)):
            if fold_idx == 3:
                fold_data = {
                    'train': aligned_samples_arr[train_idx].tolist(),
                    'valid': aligned_samples_arr[test_idx].tolist(), 
                    'test': aligned_samples_arr[test_idx].tolist()
                }

                file_name = f'mustard_final_dataset_fold_{run_idx + 1}.pkl'
                output_path = os.path.join(cfg.OUTPUT_DIR, file_name)
                
                with open(output_path, 'wb') as f:
                    pickle.dump(fold_data, f)
                    
                logger.info(f"Saved: {output_path}")
                break 

    logger.info("All alignment runs completed")

if __name__ == '__main__':
    main()