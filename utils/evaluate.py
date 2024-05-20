import torch
import numpy as np
import os, sys
from dataset.pain import PainDataset
from tqdm import tqdm
from torch.utils.data import DataLoader
from modules.aggregation import AggregationTransformer

@torch.no_grad()
def eval(model, dataloader, criterion):
    model.eval()

    predictions = []
    ground_truth = []
    for batch, (video, signal, label) in enumerate(tqdm(dataloader)):
        video = video.cuda()
        signal = signal.cuda()
        label = label.cuda()

        output = model((video, signal))

        loss = criterion(output, label)

        # Compute accuracy
        _, predicted = torch.max(output, 1) # Returns (values, indices)
        predictions.append(predicted.cpu().numpy())
        ground_truth.append(label.cpu().numpy().argmax(axis=1))

    predictions = np.concatenate(predictions)
    ground_truth = np.concatenate(ground_truth)
    # import pdb; pdb.set_trace()
    correct = (predictions == ground_truth).sum()
    accuracy = correct / ground_truth.shape[0]
    return accuracy, loss.item()


def test_model(config):
    # Create test dataset
    test_ds = PainDataset(config, "test")

    # Create test dataloader from test_ds
    test_loader = DataLoader(test_ds, batch_size=config.data.batch_size, num_workers=config.data.workers, shuffle=False)

    # Create the model
    model = AggregationTransformer(config)
    model = model.cuda()

    # Load the model
    if config.model_path is not None:
        if os.path.isfile(config.model_path):
            print(("=> loading checkpoint '{}'".format(config.model_path)))
            checkpoint = torch.load(config.model_path)
            model.load_state_dict(checkpoint['model_state_dict'])
            del checkpoint
        else:
            print(("=> no checkpoint found at '{}'".format(config.resume)))
    else:
        print("No model path provided")
        sys.exit(1)


    # Define loss function
    criterion = torch.nn.BCEWithLogitsLoss()

    # Evaluate
    return eval(model, test_loader, criterion)    