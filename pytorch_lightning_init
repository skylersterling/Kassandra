import os
import json
import mmap
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import ByteLevelBPETokenizer
from tqdm import tqdm
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers.optimization import Adafactor
from kassandra_architecture import *

class KassandraLightning(pl.LightningModule):
    def __init__(self, model, config, learning_rate, weight_decay, max_epochs, tokenizer):
        super().__init__()
        self.model = model
        self.config = config
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.max_epochs = max_epochs
        self.total_steps = 0
        self.tokenizer = tokenizer
        self.prompt = self.tokenizer.encode("The quick brown fox jumped").ids
        self.losses = []

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
        self.log('train_loss', loss, prog_bar=True)
        self.losses.append(loss.item())
        self.total_steps += 1

        if self.total_steps % 10000 == 0:
            avg_loss = sum(self.losses) / len(self.losses)
            self.print(f"Average loss after {self.total_steps} steps: {avg_loss:.4f}")
            self.losses = []

            prompt_tensor = torch.tensor(self.prompt, dtype=torch.long).unsqueeze(0).to(self.device)
            output = generate(self.model, prompt_tensor, max_new_tokens=50, context_size=1024)
            self.print("Model output:", self.tokenizer.decode(output.squeeze().tolist()))

            # Create directory for this step
            save_dir = f'checkpoint_step_{self.total_steps}'
            os.makedirs(save_dir, exist_ok=True)

            # Save model state dictionary
            torch.save(self.model.state_dict(), os.path.join(save_dir, 'model_state_dict.pth'))

            # Save model in safetensors format
            save_file(self.model.state_dict(), os.path.join(save_dir, 'model_safetensors.safetensors'))

            # Save optimizer state
            optimizer = self.trainer.optimizers[0]
            torch.save(optimizer.state_dict(), os.path.join(save_dir, 'optimizer_state_dict.pth'))

            # Save model config
            with open(os.path.join(save_dir, 'model_config.json'), 'w') as f:
                json.dump(self.config, f, indent=4)

            self.print(f"Saved model, optimizer states, and config at step {self.total_steps}")

        return loss

    def configure_optimizers(self):
        optimizer = Adafactor(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay, scale_parameter=False, relative_step=False)
        scheduler = CosineAnnealingLR(optimizer, T_max=self.max_epochs)
        return [optimizer], [scheduler]

    # def configure_optimizers(self):
    #     optimizer = AdamW(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
    #     scheduler = CosineAnnealingLR(optimizer, T_max=self.max_epochs)
    #     return [optimizer], [scheduler]
class TextDataset(Dataset):
    def __init__(self, file_path, tokenizer, context_length):
        self.file_path = file_path
        self.tokenizer = tokenizer
        self.context_length = context_length
        
        file_size = os.path.getsize(file_path)
        with open(file_path, 'rb') as f:
            with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
                print("Tokenizing data...")
                self.ids = []
                chunk_size = 1024 * 1024  # 1MB chunks
                for i in tqdm(range(0, file_size, chunk_size)):
                    chunk = mm[i:min(i+chunk_size, file_size)]
                    try:
                        text = chunk.decode('utf-8', errors='replace')
                    except UnicodeDecodeError:
                        # If there's an error at the end of the chunk, try decoding a slightly smaller chunk
                        text = chunk[:-1].decode('utf-8', errors='replace')
                    self.ids.extend(self.tokenizer.encode(text).ids)
        
    def __len__(self):
        return len(self.ids) - self.context_length
    
    def __getitem__(self, idx):
        chunk = self.ids[idx:idx+self.context_length+1]
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        return x, y

class KassandraDataModule(pl.LightningDataModule):
    def __init__(self, file_path, tokenizer, context_length, batch_size):
        super().__init__()
        self.file_path = file_path
        self.tokenizer = tokenizer
        self.context_length = context_length
        self.batch_size = batch_size

    def setup(self, stage=None):
        self.dataset = TextDataset(self.file_path, self.tokenizer, self.context_length)

    def train_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True, num_workers=4)
