"""Quick NN consistency check with k=1"""
import torch
from transformers import AutoModelForCausalLM, AutoConfig

def load_emb(model_id):
    print(f'Loading: {model_id}')
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32, low_cpu_mem_usage=True)
    state_dict = model.state_dict()
    config = AutoConfig.from_pretrained(model_id)
    weight_tying = config.tie_word_embeddings if hasattr(config, 'tie_word_embeddings') else True
    
    input_emb = None
    output_emb = None
    for key in state_dict.keys():
        if 'embed_tokens.weight' in key: input_emb = state_dict[key]
        if 'lm_head.weight' in key: output_emb = state_dict[key]
    if weight_tying or output_emb is None: output_emb = input_emb
    del model
    return input_emb.clone(), output_emb.clone()

def nn_k1(emb1, emb2, n_samples=2000):
    """Check if nearest neighbor matches between two spaces"""
    n = min(emb1.shape[0], emb2.shape[0])
    emb1, emb2 = emb1[:n].float(), emb2[:n].float()
    emb1_norm = emb1 / (emb1.norm(dim=1, keepdim=True) + 1e-8)
    emb2_norm = emb2 / (emb2.norm(dim=1, keepdim=True) + 1e-8)
    
    torch.manual_seed(42)
    sample_idx = torch.randperm(n)[:n_samples]
    
    matches = 0
    for idx in sample_idx:
        sim1 = emb1_norm[idx] @ emb1_norm.T
        sim2 = emb2_norm[idx] @ emb2_norm.T
        sim1[idx] = -float('inf')  # exclude self
        sim2[idx] = -float('inf')
        nn1 = sim1.argmax().item()
        nn2 = sim2.argmax().item()
        if nn1 == nn2:
            matches += 1
    
    return matches / n_samples

# Load
qwen4b_in, _ = load_emb('Qwen/Qwen3-4B')
qwen8b_in, qwen8b_out = load_emb('Qwen/Qwen3-8B')

print('\n' + '='*60)
print('NEAREST NEIGHBOR CONSISTENCY (k=1)')
print('='*60)

acc = nn_k1(qwen4b_in, qwen8b_in)
print(f'\n4B (tied) vs 8B Input:  {acc:.4f} ({acc*100:.1f}%)')

acc = nn_k1(qwen4b_in, qwen8b_out)
print(f'4B (tied) vs 8B Output: {acc:.4f} ({acc*100:.1f}%)')

acc = nn_k1(qwen8b_in, qwen8b_out)
print(f'8B Input vs 8B Output:  {acc:.4f} ({acc*100:.1f}%)')

print('\n' + '='*60)

