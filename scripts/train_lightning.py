import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
import torch
import argparse
import os

from backend.models.lightning_vae import LightningGOLCVAE
from backend.models.lightning_transformer import LightningTransformer
from backend.datamodules.midi_datamodule import MidiDataModule

def train(args):
    pl.seed_everything(42)
    
    # DataModule
    datamodule = MidiDataModule(
        data_path=args.data_path,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        num_workers=args.num_workers
    )
    
    # Model
    if args.model == 'vae':
        model = LightningGOLCVAE(
            input_dim=128,
            latent_dim=args.latent_dim,
            beta=args.beta,
            beta_orbit=args.beta_orbit,
            learning_rate=args.lr
        )
    elif args.model == 'transformer':
        model = LightningTransformer(
            input_dim=128,
            embed_dim=args.embed_dim,
            num_heads=args.num_heads,
            num_layers=args.num_layers,
            learning_rate=args.lr
        )
    else:
        raise ValueError(f"Unknown model: {args.model}")
        
    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=f"output/checkpoints/{args.model}",
        filename="{epoch}-{val_loss:.2f}",
        monitor="val_loss",
        mode="min",
        save_top_k=3
    )
    
    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=10,
        mode="min"
    )
    
    # Logger
    logger = TensorBoardLogger("output/logs", name=args.model)
    
    # Trainer
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator="auto",
        devices="auto",
        callbacks=[checkpoint_callback, early_stopping],
        logger=logger,
        gradient_clip_val=1.0 if args.model == 'transformer' else 0.0,
        precision=32
    )
    
    # Train
    trainer.fit(model, datamodule)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, choices=['vae', 'transformer'])
    parser.add_argument("--data_path", type=str, default="dataset/processed/full_dataset.pt")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--seq_len", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=4)
    
    # VAE args
    parser.add_argument("--latent_dim", type=int, default=128)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--beta_orbit", type=float, default=0.5)
    
    # Transformer args
    parser.add_argument("--embed_dim", type=int, default=256)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--num_layers", type=int, default=4)
    
    args = parser.parse_args()
    train(args)
