import os
import torch

def get_model_size(model):
    """
    获取模型占用磁盘空间大小（MB）
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        model_path = os.path.join(tmp_dir, "model.pt")
        torch.save(model.state_dict(), model_path)
        size_mb = os.path.getsize(model_path) / (1024 * 1024)
    return round(size_mb, 2)