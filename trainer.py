import torch
import torch.nn as nn
import torch.optim as optim
import pickle
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import classification_report, accuracy_score
from model import SarcasmGNNClassifier
from data_utils import MultimodalDataset, collate_fn

def evaluate(model, loader, device):
    model.eval()
    p, l = [], []
    with torch.no_grad():
        for b in loader:
            out, _ = model(
                b['t_u'].to(device), b['a_u'].to(device), b['v_u'].to(device),
                [x.to(device) for x in b['t_c']], [x.to(device) for x in b['a_c']], [x.to(device) for x in b['v_c']],
                [x.to(device) for x in b['sid']]
            )
            p.extend(torch.max(out, 1)[1].cpu().numpy())
            l.extend(b['lbl'].numpy())
    return l, p

def run_train_fold(fold_path, device, params):
    with open(fold_path, 'rb') as f:
        full_data = pickle.load(f)
    all_spks = {s for k in full_data for i in full_data[k] for s in i.get('context_speakers', [])}
    all_spks.update({i.get('speaker', 'unknown') for k in full_data for i in full_data[k]})
    spk_map = {n: i for i, n in enumerate(sorted(list(all_spks | {'unknown'})))}
    train_loader = DataLoader(MultimodalDataset(full_data['train'], spk_map), batch_size=params['bs'], shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(MultimodalDataset(full_data['valid'], spk_map), batch_size=params['bs'], collate_fn=collate_fn)
    test_loader = DataLoader(MultimodalDataset(full_data['test'], spk_map), batch_size=params['bs'], collate_fn=collate_fn)
    model = SarcasmGNNClassifier(1024, 768, 512, params['enc_h'], params['enc_o'], params['gnn_h'], 2, params['heads'], params['drop']).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=params['lr'], weight_decay=1e-4)
    loss_cls = nn.CrossEntropyLoss()
    total_steps = len(train_loader) * params['eps']
    sched = get_linear_schedule_with_warmup(optimizer, int(total_steps * 0.1), total_steps)
    best_val_acc = 0
    temp_model_path = f"best_model_fold_temp.pt"
    for ep in range(params['eps']):
        model.train()
        for b in train_loader:
            optimizer.zero_grad()
            out, _ = model(
                b['t_u'].to(device), b['a_u'].to(device), b['v_u'].to(device),
                [x.to(device) for x in b['t_c']], [x.to(device) for x in b['a_c']], [x.to(device) for x in b['v_c']],
                [x.to(device) for x in b['sid']]
            )
            loss = loss_cls(out, b['lbl'].to(device))
            loss.backward()
            optimizer.step()
            sched.step()
        val_l, val_p = evaluate(model, val_loader, device)
        val_acc = accuracy_score(val_l, val_p)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), temp_model_path)
    model.load_state_dict(torch.load(temp_model_path))
    test_l, test_p = evaluate(model, test_loader, device)
    report = classification_report(test_l, test_p, target_names=["non-sarcastic", "sarcastic"], output_dict=True, zero_division=0)
    return accuracy_score(test_l, test_p), report