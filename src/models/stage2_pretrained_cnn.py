import torch

def load_pretrained_cnn_model(repo: str = "chenyaofo/pytorch-cifar-models", model_name: str = "cifar10_resnet20"):
    print(f"Loading pretrained CNN model: {model_name} from {repo}...")
    model = torch.hub.load(repo, model_name, pretrained=True)
    model.eval()
    print("Model loaded successfully.")
    
    return model