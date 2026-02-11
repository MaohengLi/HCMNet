from tqdm import tqdm
import numpy as np
import logging

logger = logging.getLogger(__name__)

def align_modalities(metadata, feature_data):
    aligned_samples = []
    for video_id, data in tqdm(metadata.items(), desc="Aligning modalities"):
        if not all(video_id in fd for fd in feature_data.values()):
            continue

        aligned_samples.append({
            'video_id': video_id,
            'show': data['show'],
            'label': int(data['sarcasm']),
            'speaker': data['speaker'],
            'context_speakers': data['context_speakers'],
            'text_utterance_features': feature_data['text_utterance'][video_id]['utterance_features'],
            'text_context_features': feature_data['text_context'][video_id]['context_features'],
            'audio_utterance_features': feature_data['audio_utterance'][video_id]['utterance_features'],
            'audio_context_features': feature_data['audio_context'][video_id]['context_features'],
            'video_utterance_features': feature_data['video_utterance'][video_id]['utterance_features'],
            'video_context_features': feature_data['video_context'][video_id]['context_features']
        })
    
    logger.info(f"Alignment complete. Total aligned samples: {len(aligned_samples)}")
    return np.array(aligned_samples)