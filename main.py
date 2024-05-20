import os
import yaml
import shutil
import argparse
import math, sys
from tqdm import tqdm
from pathlib import Path
from dotmap import DotMap
from pprint import PrettyPrinter

import torch
from transformers import get_linear_schedule_with_warmup

from dataset.pain import PainDataset
from torch.utils.data import DataLoader
from modules.aggregation import AggregationTransformer
from utils.evaluate import eval, test_model
from utils.saving import save_epoch, save_best
import wandb

def get_config():
    parser = argparse.ArgumentParser(description="Train a model on a dataset")
    parser.add_argument("--config", type=str, default=None, help="Path to the config file")
    parser.add_argument("--log_time", type=str, default=None, help="Current time for logging purposes")
    args = parser.parse_args()

    # Load the config file
    with open(args.config, "r") as f:
        config = yaml.full_load(f)
  
    config['working_dir'] = os.path.join("./exp", config['name'], args.log_time)
    # Log config
    print('-' * 80)
    print(' ' * 20, "working dir: {}".format(config['working_dir']))
    print('-' * 80)

    print('-' * 80)
    print(' ' * 30, "Config")
    pp = PrettyPrinter(indent=4)
    pp.pprint(config)
    print('-' * 80)
    
    config = DotMap(config)

    # Set the working directory
    Path(config.working_dir).mkdir(parents=True, exist_ok=True)
    shutil.copy(args.config, config.working_dir)
    shutil.copy("main.py", config.working_dir)

    # Set up wandb
    wandb.init(project="TFG",
                name="{}_{}".format(config.name, args.log_time),
                config=config)

    return config


def main():
    config = get_config()

    # Set the seed
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed(config.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cpu":
        print("===== WARNING =====")
        print("Running on CPU")
        print("==================")
        sys.exit(1)

    if config.eval:
        loss, acc = test_model(config)
        wandb.log({'test_loss': loss, 'test_accuracy': acc})
        return


    # Create the datasets
    train_ds  = PainDataset(config, "train")
    val_ds = PainDataset(config, "val")

    # Create the dataloaders
    train_loader = DataLoader(train_ds, batch_size=config.data.batch_size, num_workers=config.data.workers, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=config.data.batch_size, num_workers=config.data.workers, shuffle=True)

    # Create the model
    model = AggregationTransformer(config)
    model = model.cuda()

    # Create the optimizer and lr_scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.solver.lr)
    # scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=config.solver.lr_warmup_step * len(train_loader), num_training_steps=len(train_loader) * config.solver.epochs)

    # Create the loss function
    criterion = torch.nn.BCEWithLogitsLoss()
    # if config.data.task == "binary":
    # else:
    #     criterion = torch.nn.CrossEntropyLoss()

    if config.model.freeze_video and config.data.video_path is not None:
        for param in model.video_transformer.parameters():
            param.requires_grad = False


    best = 0.0
    for epoch in range(config.solver.epochs):
        wandb.run.summary["epoch"] = epoch
        model.train()

        for batch, (video, signal, label) in enumerate(tqdm(train_loader)):
            optimizer.zero_grad()
            video = video.cuda()
            signal = signal.cuda()
            label = label.cuda()

            # Forward pass
            output = model((video, signal))

            # Compute loss
            loss = criterion(output, label)
            
            if batch % config.logging.freq == 0:
                print(f'Loss: {loss.item()}')
                wandb.log({'loss': loss.item(), "lr": optimizer.param_groups[0]['lr']})
            
            
            if not math.isfinite(loss):
                print("Loss is infinite")
                sys.exit(1)

            loss.backward()
            optimizer.step()
            # scheduler.step()

        if epoch % config.solver.eval_freq == 0:
            print(f"[{epoch}/{config.solver.epochs}] Saving epoch...")
            save_epoch(epoch, model, optimizer, config.working_dir, "last_epoch.pt")

            accuracy, loss = eval(model, val_loader, criterion)
            wandb.log({'val_loss': loss, 'val_accuracy': accuracy})
            print(f"[{epoch}/{config.solver.epochs}] Validation -- Accuracy: {accuracy} ({best}) -- Loss: {loss}")

            if accuracy > best:
                best = accuracy
                save_best(config.working_dir, "last_epoch.pt", epoch)
            wandb.log({'best_val_accuracy': best})
                
if __name__ == "__main__":
    main()