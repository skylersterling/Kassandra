from kassandra_lightning import *

learning_rate = 1e-4
weight_decay = 0.1
max_epochs = 1
batch_size = 6
gradient_accumulation = 1

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
return_parameters(model)

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

#model = torch.compile(model)

print("Tokenizer Vocabulary Size:", tokenizer.get_vocab_size())
model = KassandraLightning(model, config, learning_rate, weight_decay, max_epochs, tokenizer)

trainer = pl.Trainer(
    max_epochs=max_epochs,
    devices=1,
    precision='bf16-true',
    gradient_clip_val=1.0,
    accumulate_grad_batches=gradient_accumulation,
    enable_progress_bar=True,
)

print("Starting training...")
trainer.fit(model, data_module)
