import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from components import *

class MultiHeadAttention(nn.Module):
    def __init__(self, input_dimension, output_dimension, context_length, dropout, number_of_attention_heads, sliding_attention_window_size, causality, rotary_embedding_window_chunk_size, rope_theta_base_variation_frequency, original_maximum_sequence_length, qkv_bias=False):
        super().__init__()
       
        assert output_dimension % number_of_attention_heads == 0, "The output dimension must be divisible by the defined number of attention number_of_attention_heads."
        assert output_dimension // number_of_attention_heads % 2 == 0, "Dimension of each attention head must be even for RoPE."
        
        self.dimension_of_each_individual_head = output_dimension // number_of_attention_heads        
        self.output_dimension = output_dimension
        self.number_of_attention_heads = number_of_attention_heads
        self.sliding_attention_window_size = sliding_attention_window_size
        self.causality = causality
        self.original_maximum_sequence_length = original_maximum_sequence_length

        self.Query_Weight = nn.Linear(input_dimension, output_dimension, bias=qkv_bias)
        self.Key_Weight = nn.Linear(input_dimension, output_dimension, bias=qkv_bias)
        self.Value_Weight = nn.Linear(input_dimension, output_dimension, bias=qkv_bias)
     
        self.attention_projection = nn.Linear(output_dimension, output_dimension)
        self.dropout = nn.Dropout(dropout)
        
        if self.causality:
            self.register_buffer("causal_mask", torch.triu(torch.ones(context_length, context_length), diagonal=1).bool())
        
        sliding_window_mask = torch.zeros(context_length, context_length, dtype=torch.bool)
        for i in range(context_length):
            start = max(0, i - sliding_attention_window_size // 2)
            end = min(context_length, i + sliding_attention_window_size // 2 + 1)
            sliding_window_mask[i, start:end] = True
        self.register_buffer("sliding_window_mask", sliding_window_mask)

        self.rope = RoPE(self.dimension_of_each_individual_head, context_length, rope_theta_base_variation_frequency, original_maximum_sequence_length)
        self.rotary_embedding_window_chunk_size = rotary_embedding_window_chunk_size 

    def forward(self, input_sequence):
        batch_size, num_tokens, input_dimension = input_sequence.shape

        keys = self.Key_Weight(input_sequence)
        queries = self.Query_Weight(input_sequence)
        values = self.Value_Weight(input_sequence)

        keys = keys.view(batch_size, num_tokens, self.number_of_attention_heads, self.dimension_of_each_individual_head).transpose(1, 2)
        queries = queries.view(batch_size, num_tokens, self.number_of_attention_heads, self.dimension_of_each_individual_head).transpose(1, 2)
        values = values.view(batch_size, num_tokens, self.number_of_attention_heads, self.dimension_of_each_individual_head).transpose(1, 2)

        mask = self.sliding_window_mask[:num_tokens, :num_tokens]

        for i in range(0, num_tokens, self.rotary_embedding_window_chunk_size):
            rotary_embedding_window_chunk_size = min(self.rotary_embedding_window_chunk_size, num_tokens - i)
            rotary_position_frequencies = self.rope(queries[:, :, i:i+rotary_embedding_window_chunk_size, :], num_tokens)
            queries[:, :, i:i+rotary_embedding_window_chunk_size, :], keys[:, :, i:i+rotary_embedding_window_chunk_size, :] = apply_rotary_pos_emb(
                queries[:, :, i:i+rotary_embedding_window_chunk_size, :], 
                keys[:, :, i:i+rotary_embedding_window_chunk_size, :], 
                rotary_position_frequencies[:, :, i:i+rotary_embedding_window_chunk_size, :]
            )


        attention_scores = queries @ keys.transpose(2, 3)

        if self.causality:
            mask = mask & (~self.causal_mask[:num_tokens, :num_tokens])
        
        attention_scores.masked_fill_(~mask.unsqueeze(0).unsqueeze(1), -torch.inf)

        attention_weights = torch.softmax(attention_scores / keys.shape[-1]**0.5, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        context_vector = (attention_weights @ values).transpose(1, 2)
        context_vector = context_vector.contiguous().view(batch_size, num_tokens, self.output_dimension)
        context_vector = self.attention_projection(context_vector)

        return context_vector

class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.multi_head_attention = MultiHeadAttention(
            input_dimension=cfg["input_dimension"],
            output_dimension=cfg["input_dimension"],
            context_length=cfg["context_length"],
            number_of_attention_heads=cfg["number_of_attention_heads"],
            dropout=cfg["dropout"],
            causality=cfg["causality"],
            sliding_attention_window_size=cfg["sliding_attention_window_size"],
            rotary_embedding_window_chunk_size=cfg["rotary_embedding_window_chunk_size"],
            original_maximum_sequence_length=cfg["original_maximum_sequence_length"],
            rope_theta_base_variation_frequency=cfg["rope_theta_base_variation_frequency"],  # Add rope_theta
            qkv_bias=cfg["qkv_bias"])
        self.FeedForwardNeuralNetwork = FeedForward(cfg)
        self.normalization_1 = LayerNormalization(cfg["input_dimension"]) #normalization_alpha
        self.normalization_2 = LayerNormalization(cfg["input_dimension"]) #normalization_beta
        self.dropout = nn.Dropout(cfg["dropout"])

    def forward(self, x):
        
        first_residual_connection_element = x
        x = self.normalization_1(x)
        x = self.multi_head_attention(x)   
        x = self.dropout(x)
        x = x + first_residual_connection_element #Residual

        second_residual_connection_element = x
        x = self.normalization_2(x)
        x = self.FeedForwardNeuralNetwork(x)
        x = self.dropout(x)
        x = x + second_residual_connection_element #Residual

        return x
    
class Kassandra(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.token_embedding_layer = nn.Embedding(cfg["vocab_size"], cfg["input_dimension"])
        self.dropout = nn.Dropout(cfg["dropout"])
        self.gradient_checkpointing = (cfg["gradient_checkpointing"])
        self.transformer_blocks = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg["layers"])])
        self.final_normalization = LayerNormalization(cfg["input_dimension"])
        self.final_projection_layer = nn.Linear(cfg["input_dimension"], cfg["vocab_size"], bias=False)

    def forward(self, in_idx):
        batch_size, sequence_length = in_idx.shape
        x = self.token_embedding_layer(in_idx)
        x = self.dropout(x)
        for block in self.transformer_blocks:
            if self.gradient_checkpointing:
                x = checkpoint(block, x)
            else:
                x = block(x)
        x = self.final_normalization(x)
        logits = self.final_projection_layer(x)
        return logits
