# Kassandra Architecture

Kassandra is a custom decoder-only Transformer model I implemented in PyTorch tailored for autoregressive language generation. This repository includes the code for building, training, and infering with the Kassandra class of models.

The architecture incorporates a Multi-Head Self-Attention mechanism, a custom implementation of sliding attention, a custom tokenizer with a vocabulary size of 30000 tokens, and a custom implementation of Rotary Positional Embeddings (RoPE).

## Getting Started

### Prerequisites

- Python 3.8 or higher
- PyTorch 2.0 or higher
- Dependencies as listed in `requirements.txt`

### Installation

Clone the repository and install the neccesary dependencies:

```bash
git clone https://github.com/skylersterling/kassandra.git
cd kassandra
pip install -r requirements.txt
```

### Usage

#### Initialization

To initialize an instance of the Kassandra model, you can follow the configuration format as shown in the following example:

```python

config = {
    "vocab_size": 30000,
    "context_length": 1024,
    "original_maximum_sequence_length": 1024,
    "sliding_attention_window_size": 1024,
    "input_dimension": 2048,
    "number_of_attention_heads": 32,
    "layers": 12,
    "dropout": 0.1,
    "causality": True,
    "rotary_embedding_window_chunk_size": 1024,
    "rope_theta_base_variation_frequency": 10000,
    "qkv_bias": False,
    "gradient_checkpointing": False
}

model = Kassandra(config)
```

#### Training

The training process can be managed through various libraries or a custom PyTorch training loop. This repository provides a simple example of fitting Kassandra with PyTorch Lightning on a .txt dataset for your convenience:

```python

tokenizer = ByteLevelBPETokenizer(
    './tokenizer/KassandraTokenizer-vocab.json',
    './tokenizer/KassandraTokenizer-merges.txt'
)

data_module = KassandraDataModule(
    file_path='./UnprocessedWikipediaTextCorpus.txt',
    tokenizer=tokenizer,
    context_length=config['context_length'],
    batch_size=batch_size
)

print("Setting up data module.")
data_module.setup()

model = KassandraLightning(model, config, learning_rate, weight_decay, max_epochs, tokenizer)

learning_rate = 1e-4
weight_decay = 0.1
max_epochs = 10
batch_size = 32
gradient_accumulation = 6

trainer = pl.Trainer(
    max_epochs=max_epochs,
    devices=1,
    precision='bf16-true',
    gradient_clip_val=1.0,
    accumulate_grad_batches=gradient_accumulation,
    enable_progress_bar=True,
)

trainer.fit(model, data_module)

```

#### Inference

To perform inference on a Kassandra model, you can follow the following example:

```python

text = generate_text(
    model,
    tokenizer,
    prompt="Once upon a time",
    max_length=100
)
print(text)
```

## Acknowledgements

- This project is heavily inspired by the Transformer implementation as presented in the paper [Attention is All You Need](https://arxiv.org/abs/1706.03762).

- The RoPE implementation is inspired by the paper [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864).

- The sliding window attention mechanism is inspired by the paper [LongFormer](https://arxiv.org/pdf/2004.05150v2).

## Contact

If you have any questions or issues, or if you need a custom licensing scheme, please open an issue in this repository or feel free to contact me directly at email@.com. You are encouraged to use this architecture to train your own models, provided that proper attribution is given in the model card.

---
