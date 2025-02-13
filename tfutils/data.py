import os
import h5py
import torch
from torch.utils.data import Dataset
import numpy as np

class SingleHandH5Dataset_MobileLastN(Dataset):
    def __init__(self, root_dir, label_file, n=60):
        self.root_dir = root_dir
        self.files = []
        self.labels = []
        self.n = 60

        for label_folder in os.listdir(root_dir):
            label_path = os.path.join(root_dir, label_folder)
            if os.path.isdir(label_path):
                label = label_folder
                for file_name in os.listdir(label_path):
                    if file_name.endswith('.h5'):
                        self.files.append(os.path.join(label_path, file_name))
                        self.labels.append(label)

        with open(label_file) as f:
            self.label_order = [i.strip() for i in f.readlines()]
        
        self.label_to_idx = {label: idx for idx, label in enumerate(self.label_order)}
        self.labels = [self.label_to_idx[label.lower()] for label in self.labels]

    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        with h5py.File(self.files[idx], 'r') as h5_file:
            data = h5_file['intermediate'][:][:, 0, :, :] #since we only have 1 hand remove that dimension
        
        data_tensor = torch.tensor(data, dtype=torch.float32)

        if data_tensor.shape[0] > self.n:
            data_tensor = data_tensor[-self.n:]
        if data_tensor.shape[0] < self.n:
            middle_idx = data_tensor.shape[0] // 2
            pad_num = self.n - data_tensor.shape[0]

            data_tensor = torch.cat([
                data_tensor[:middle_idx, :],
                data_tensor[middle_idx:middle_idx+1, :].repeat([pad_num, *[1 for i in data_tensor.shape[1:]]]),
                data_tensor[middle_idx:, :]
            ])
        
        data_tensor = data_tensor.view(1, data_tensor.size(0), -1)


        # One Hot Encoding
        label_tensor = torch.zeros(len(self.label_order), dtype=torch.float32)
        label_tensor[self.labels[idx]] = 1.0

        # None One Hot Encoding 
        # label_tensor = torch.zeros(1, dtype=torch.int32)
        # label_tensor[0] = self.labels[idx]

        return data_tensor, label_tensor
    
if __name__ == '__main__':
    print("Running data.py")
    root_dir = '/data'
    dataset = SingleHandH5Dataset_MobileLastN(root_dir, "/meta/563_sign_list.txt")
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
    for i in dataloader:
        print([j.shape for j in i])
        pass

    