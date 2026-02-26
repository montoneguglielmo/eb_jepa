import os
import numpy as np
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from decord import VideoReader, cpu
import torchvision.transforms as transforms

class VideoRobosuiteDataset(Dataset):
    def __init__(self, folder_path, num_frames=16, resize=(224, 224)):
        self.folder_path = folder_path
        self.video_front_path = self.folder_path + 'videos/chunk-000/observation.images.image'
        self.video_wrist_path = self.folder_path + 'videos/chunk-000/observation.images.wrist'
        self.robot_data_path  = self.folder_path + 'data/chunk-000/'

        self.video_files_names = [
            f for f in os.listdir(self.video_front_path)
            if f.endswith(('.mp4'))
        ]
        self.num_frames = num_frames
        self.resize = resize
        
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(resize),
            transforms.ToTensor(),  # returns C x H x W in [0, 1]
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.video_files_names)

    def __getitem__(self, idx):
        video_front_path = self.video_front_path + '/' + self.video_files_names[idx]
        video_wrist_path = self.video_wrist_path + '/' + self.video_files_names[idx]
        robot_data_path = self.robot_data_path + '/' + self.video_files_names[idx].split('.')[0] + '.parquet'
        
        vr_front = VideoReader(video_front_path, ctx=cpu(0))
        vr_wrist = VideoReader(video_wrist_path, ctx=cpu(0))
        total_frames = len(vr_front)

        # Uniform frame sampling
        indices = torch.linspace(0, total_frames - 1, steps=self.num_frames).long()

        frames_front = vr_front.get_batch(indices).asnumpy()  # T x H x W x C
        frames_wrist = vr_wrist.get_batch(indices).asnumpy()
        
        processed_frames = []
        for fr_front, fr_wrist in zip(frames_front, frames_wrist):
            fr_front = self.transform(fr_front)  # C x H x W
            fr_wrist = self.transform(fr_wrist)
            fr = torch.cat([fr_front, fr_wrist], axis=-1)
            processed_frames.append(fr)

        video_tensor = torch.stack(processed_frames, dim=1)
        # Shape: C x T x H x W

        df = pd.read_parquet(robot_data_path)
        robot_state = torch.tensor(np.stack(df['observation.state'].values), dtype=torch.float32)
        robot_action = torch.tensor(np.stack(df['action'].values), dtype=torch.float32)
        robot_state = robot_state[indices]
        robot_action = robot_action[indices]
        
        return {'video': video_tensor, 
                'robot_state':robot_state, 
                'robot_action': robot_action}
    
if __name__ == "__main__":
    dataset = VideoRobosuiteDataset(
    folder_path="/home/guglielmo/Projects/authonomus_discovery/to_kill/",
    num_frames=16,
    resize=(224, 224))

    loader = DataLoader(
        dataset,
        batch_size=8,
        shuffle=True,
        num_workers=1,
        pin_memory=True
        )

    for batch_video, batch_robot, batch_action in loader:
        print(batch_video.shape, batch_robot.shape, batch_action.shape)
    # Output: torch.Size([B, C, T, H, W])