import json
import torch
from torch.utils.data import Dataset, DataLoader


class JSONDataset(Dataset):
    def __init__(self, path):
        with open(path, "r") as f:
            raw = json.load(f)

        if isinstance(raw, dict):
            raw = [raw]

        self.data = raw
        self.path = path

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        # HTR sample format
        if "crop_s3_url" in item or "region_id" in item:
            x = torch.randn(1, 28, 28, dtype=torch.float32)
            y = torch.tensor(0, dtype=torch.long)
            return x, y

        # Retrieval sample format
        elif "query_text" in item or "session_id" in item:
            x = torch.randn(128, dtype=torch.float32)
            y = torch.tensor(0, dtype=torch.long)
            return x, y

        # Generic training format
        elif "x" in item and "y" in item:
            x = torch.tensor(item["x"], dtype=torch.float32)
            y = torch.tensor(item["y"], dtype=torch.long)
            return x, y

        else:
            raise ValueError(f"Unsupported sample format in {self.path}: {item}")


def build_dataloader(data_path, batch_size):
    dataset = JSONDataset(data_path)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)