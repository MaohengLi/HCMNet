import os
import pickle
import random
import numpy as np
import logging

logger = logging.getLogger(__name__)

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    logger.info(f"Seed set to: {seed}")

def load_pickle(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Feature file not found: {path}")
    with open(path, 'rb') as f:
        return pickle.load(f)