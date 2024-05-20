import torch.nn as nn
import torch.nn.functional as F
import torch
from transformers import VivitModel, VideoMAEModel, TimesformerModel


class AggregationTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.input_dim = 0
        self.output_dim = 2 if config.data.task == "binary" else 5 
        self.hidden_dim = config.model.hidden_dim
        self.video = False
        self.signals = False

        if config.data.video_path is not None:
            self.video = True
            self.input_dim += 768
            if config.model.video_transformer == "vivit":
                self.video_fts_layers = lambda x: x['pooler_output']
                self.video_transformer = VivitModel.from_pretrained("google/vivit-b-16x2-kinetics400")
            elif config.model.video_transformer == "videomae":
                self.video_fts_layers = lambda x: x['last_hidden_state'][:, 0, :]
                self.video_transformer = VideoMAEModel.from_pretrained("MCG-NJU/videomae-base")
            elif config.model.video_transformer == "timesformer":
                self.video_fts_layers = lambda x: x['last_hidden_state'][:, 0, :]
                self.video_transformer = TimesformerModel.from_pretrained("facebook/timesformer-base-finetuned-k400")
            else:
                raise ValueError("Invalid video transformer model")
        
        if config.data.signals is not None:
            self.signals = True
            self.input_dim += 20

        if not self.video and not self.signals:
            raise ValueError("At least one of video or signals must be provided")

        self.mlp = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_dim, self.output_dim),
            # nn.Softmax(dim=1)
        )

    def forward(self, x):
        rgb, signals = x
        
        if self.video:
            video_fts = self.video_fts_layers(self.video_transformer(rgb))
            if self.signals:
                fts = torch.cat((video_fts, signals), dim=-1)
            else:
                fts = video_fts
        else:
            fts = signals
        
        return self.mlp(fts)