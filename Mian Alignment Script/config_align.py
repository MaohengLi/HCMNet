import os

SEED = 42
K_FOLDS = 5
NUM_RUNS = 5
METADATA_JSON = os.path.join('data_preprocessed', 'aligned_sarcasm_data_final.json')
FEATURES_DIR = './features'
OUTPUT_DIR = './final_dataset_cv'

FEATURE_FILES = {
    'text_utterance': os.path.join(FEATURES_DIR, 'mustard_bert_utterance_features_fine_grained.pkl'),
    'text_context': os.path.join(FEATURES_DIR, 'mustard_bert_context_features_fine_grained.pkl'),
    'audio_utterance': os.path.join(FEATURES_DIR, 'mustard_wav2vec2_utterance_features_fine_grained.pkl'),
    'audio_context': os.path.join(FEATURES_DIR, 'mustard_wav2vec2_context_features_fine_grained.pkl'),
    'video_utterance': os.path.join(FEATURES_DIR, 'mustard_clip_utterance_features_fine_grained.pkl'),
    'video_context': os.path.join(FEATURES_DIR, 'mustard_clip_context_features_fine_grained.pkl'),
}