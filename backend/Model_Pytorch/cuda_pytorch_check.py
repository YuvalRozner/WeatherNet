import torch

def check_cuda():
    if torch.cuda.is_available():
        print("CUDA is available. You can use GPU for PyTorch.")
        print(f"Number of GPUs available: {torch.cuda.device_count()}")
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")
    else:
        print("CUDA is not available. You can only use CPU for PyTorch.")
if __name__ == "__main__":
    check_cuda()