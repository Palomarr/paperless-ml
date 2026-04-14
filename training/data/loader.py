import torch
from torch.utils.data import Dataset
import json


class JSONDataset(Dataset):
    def __init__(self, json_path):
        with open(json_path, "r") as f:
            self.data = json.load(f)

        # 如果只有一个 sample → 转成 list
        if isinstance(self.data, dict):
            self.data = [self.data]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]

        # 👉 读取 JSON 字段（关键！）
        crop_url = sample.get("crop_s3_url", None)
        region_id = sample.get("region_id", None)

        # 👉 现在先 fake image
        x = torch.randn(1, 28, 28)

        # 👉 fake label（后面再改）
        y = torch.randint(0, 10, (1,)).item()

        return x, y