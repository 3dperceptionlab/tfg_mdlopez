import torch
import sys, os
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import pickle


class PainDataset(Dataset):
    def __init__(self, config, split):
        
        self.anno_path = config.data.anno_path
        self.video = None
        if config.data.video_path is not None:
            self.video = os.path.join(config.data.video_path, f"video_processed-{config.model.video_transformer}")
        self.signal = None
        if config.data.signals is not None:
            # TODO: Load the signal data
            # Cargar el diccionario de características desde el archivo
            with open('/workspace/diccionario01.pickle', 'rb') as f:
                self.signal = pickle.load(f)
            #self.signal= os.path.join(config.data.signals, f"diccionario.pickle")
            #raise not NotImplementedError("Signal data loading not implemented yet")

        self.num_classes = 2 if config.data.task == "binary" else 5
        if self.video is None and self.signal is None:
            raise ValueError("At least one of video or signal data should be provided")

        df = pd.read_csv(os.path.join(self.anno_path, "biovid_ds_" + split + ".csv"), header=None, sep=" ")
            

        self.input = [(r[0],r[1]) for _, r in df.iterrows()]
        # self.input = []
        # bl = 0
        # pa = 0
        # for i in _input:
        #     if "BL1" in i[0]:
        #         self.input.append((i[0], 0))
        #         bl +=1
        #     elif "PA4" in i[0]:
        #         self.input.append((i[0], 1))
        #         pa +=1
        
        # print(f"Split: {split}, BL: {bl}, PA: {pa}")


    def __len__(self):
        # Return the number of samples depending on the available data
        return len(self.input)

    def __getitem__(self, index):
        # Get the sample
        video_name, label = self.input[index]
        participant = video_name.split("-")[0]

        # Initialize the data to be returned (we always return a video and a signal tensor)
        video_data = torch.tensor([])
        signal_data = torch.tensor([])

        # Load the video data, if available
        if self.video is not None:
            video_data = torch.from_numpy(np.load(os.path.join(self.video, participant, video_name + ".npy")))

        if self.signal is not None:
            s = self.signal[video_name]
            signal_data = torch.from_numpy(s).float()

        # Label to one-hot encoded tensor
        if self.num_classes == 2: # Binary classification
            label = 0 if label == 0 else 1
        label = torch.nn.functional.one_hot(torch.tensor(label), num_classes=self.num_classes).float()
        

        return video_data, signal_data, label