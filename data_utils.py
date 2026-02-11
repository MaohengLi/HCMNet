import torch
from torch.utils.data import Dataset

class MultimodalDataset(Dataset):
    def __init__(self, data, speaker_map):
        self.data = data
        self.spk_map = speaker_map
        self.unk = speaker_map.get('unknown', len(speaker_map))
    def __len__(self): return len(self.data)
    def __getitem__(self, idx):
        item = self.data[idx]
        sid = self.spk_map.get(item['speaker'], self.unk)
        ctx_sid = [self.spk_map.get(s, self.unk) for s in item['context_speakers']] + [sid]
        return {
            't_u': torch.tensor(item['text_utterance_features'], dtype=torch.float32),
            'a_u': torch.tensor(item['audio_utterance_features'], dtype=torch.float32),
            'v_u': torch.tensor(item['video_utterance_features'], dtype=torch.float32),
            't_c': torch.tensor(item['text_context_features'], dtype=torch.float32),
            'a_c': torch.tensor(item['audio_context_features'], dtype=torch.float32),
            'v_c': torch.tensor(item['video_context_features'], dtype=torch.float32),
            'lbl': torch.tensor(item['label'], dtype=torch.long),
            'sid': torch.tensor(ctx_sid, dtype=torch.long)
        }

def collate_fn(batch):
    return {
        't_u': torch.stack([d['t_u'] for d in batch]),
        'a_u': torch.stack([d['a_u'] for d in batch]),
        'v_u': torch.stack([d['v_u'] for d in batch]),
        't_c': [d['t_c'] for d in batch],
        'a_c': [d['a_c'] for d in batch],
        'v_c': [d['v_c'] for d in batch],
        'lbl': torch.stack([d['lbl'] for d in batch]),
        'sid': [d['sid'] for d in batch]
    }