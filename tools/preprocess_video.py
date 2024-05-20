import av
import numpy as np
import os
import torch
from transformers import VivitImageProcessor, VideoMAEImageProcessor, AutoImageProcessor
np.random.seed(0)


def read_video_pyav(container, indices):
    '''
    Decode the video with PyAV decoder.
    Args:
        container (`av.container.input.InputContainer`): PyAV container.
        indices (`List[int]`): List of frame indices to decode.
    Returns:
        result (np.ndarray): np array of decoded frames of shape (num_frames, height, width, 3).
    '''
    frames = []
    container.seek(0)
    start_index = indices[0]
    end_index = indices[-1]
    for i, frame in enumerate(container.decode(video=0)):
        if i > end_index:
            break
        if i >= start_index and i in indices:
            frames.append(frame)
    return np.stack([x.to_ndarray(format="rgb24") for x in frames])


def sample_frame_indices(clip_len, frame_sample_rate, seg_len):
    '''
    Sample a given number of frame indices from the video.
    Args:
        clip_len (`int`): Total number of frames to sample.
        frame_sample_rate (`int`): Sample every n-th frame.
        seg_len (`int`): Maximum allowed index of sample's last frame.
    Returns:
        indices (`List[int]`): List of sampled frame indices
    '''
    converted_len = int(clip_len * frame_sample_rate)
    end_idx = np.random.randint(converted_len, seg_len)
    start_idx = end_idx - converted_len
    indices = np.linspace(start_idx, end_idx, num=clip_len)
    indices = np.clip(indices, start_idx, end_idx - 1).astype(np.int64)
    return indices

# image_processor = VivitImageProcessor.from_pretrained("google/vivit-b-16x2-kinetics400")
# clip_len = 32
# image_processor = VideoMAEImageProcessor.from_pretrained("MCG-NJU/videomae-base")
# clip_len = 16
image_processor = AutoImageProcessor.from_pretrained("MCG-NJU/videomae-base")
clip_len = 8

videos = []
dir = "/datasets/PainDatasets/BioVid/PartA/video/"
out_dir = "/datasets/PainDatasets/BioVid/PartA/video_processed-timesformer/"
for participant in os.listdir(dir):
    video_files = os.listdir(os.path.join(dir, participant))
    for video_file in video_files:
        if not video_file.endswith('.mp4'):
            print(video_file)
        file_path = os.path.join(dir, participant, video_file)
        container = av.open(file_path)
        indices = sample_frame_indices(clip_len=clip_len, frame_sample_rate=1, seg_len=container.streams.video[0].frames)
        video = read_video_pyav(container=container, indices=indices)
        inputs = image_processor(list(video), return_tensors="np")
        out_path = os.path.join(out_dir, participant)
        if not os.path.exists(out_path):
            os.makedirs(out_path)
        np.save(os.path.join(out_path, os.path.basename(video_file)[:-4] + ".npy"), inputs['pixel_values'].squeeze())
    print(f"Processed {participant}")
