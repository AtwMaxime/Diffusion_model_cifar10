import torch

class Config:
    batch_size = 64
    learning_rate = 0.001
    num_epochs = 10
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data_dir = './data'
    model_save_path = './outputs/model.pth'