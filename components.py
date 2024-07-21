import torch
import torch.nn as nn
import torch.nn.functional as F

class RoPE(nn.Module):
    def __init__(self, dimension_of_each_individual_head, context_length, rope_theta_base_variation_frequency, original_maximum_sequence_length):
        super().__init__()
        self.dimension_of_each_individual_head = dimension_of_each_individual_head
        self.context_length = context_length
        self.original_maximum_sequence_length = original_maximum_sequence_length or context_length 
        inverse_frequency_bands = 1.0 / (rope_theta_base_variation_frequency ** (torch.arange(0, dimension_of_each_individual_head, 2).float() / dimension_of_each_individual_head))
        self.register_buffer("inverse_frequency_bands", inverse_frequency_bands)
                
    def forward(self, x, sequence_length):
        t = torch.arange(sequence_length, device=x.device).type_as(self.inverse_frequency_bands)
        
        if sequence_length > self.original_maximum_sequence_length:
            scale = sequence_length / self.original_maximum_sequence_length
            t = t / scale
        
        rotary_position_frequencies = torch.einsum("i,j->ij", t, self.inverse_frequency_bands)
        embedding = torch.cat((rotary_position_frequencies, rotary_position_frequencies), dim=-1)
        return embedding[None, None, :, :]

def rotate_half(x):
    x1, x2 = x[..., :x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, rotary_position_frequencies):
    rotated_query = (q * rotary_position_frequencies.cos()) + (rotate_half(q) * rotary_position_frequencies.sin())
    rotated_key = (k * rotary_position_frequencies.cos()) + (rotate_half(k) * rotary_position_frequencies.sin())
    return rotated_query, rotated_key

class LayerNormalization(nn.Module):
    def __init__(self, input_dimension):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(input_dimension))
        self.shift = nn.Parameter(torch.zeros(input_dimension))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift


class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) *
            (x + 0.044715 * torch.pow(x, 3))
        ))


class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["input_dimension"], 4 * cfg["input_dimension"]),
            GELU(),
            nn.Linear(4 * cfg["input_dimension"], cfg["input_dimension"]),
        )

    def forward(self, x):
        return self.layers(x)

def return_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"The total number of parameters in the model is: {total_params:,} | Of which {trainable_params:,} are trainable.")
    return total_params, trainable_params

def generate(model, idx, max_new_tokens, context_size):
    device = next(model.parameters()).device
    model = model.to(device) 
    idx = idx.to(device)

    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]

        with torch.no_grad():
            logits = model(idx_cond)

        logits = logits[:, -1, :]

        idx_next = torch.argmax(logits, dim=-1, keepdim=True)
        idx_next = idx_next.to(device) 

        idx = torch.cat((idx, idx_next), dim=1)

    return idx