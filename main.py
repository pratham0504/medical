import os
import argparse
import torch
from config import FLConfig
from data_loader import load_medical_partitions
from server import Server

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", type=str, default="fedavg")
    args = parser.parse_args()

    config = FLConfig(algorithm=args.algo)
    # Detect GPU on Windows/Linux or use CPU on Mac
    device = torch.device("cuda" if (torch.cuda.is_available() and config.use_cuda) else "cpu")
    print(f"Running on: {device}")

    train_loaders, test_loader, names = load_medical_partitions(config)
    server = Server(config, train_loaders, test_loader, device, names)
    server.run()

    # Save the model
    save_path = config.save_path.format(algorithm=args.algo)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(server.global_model.state_dict(), save_path)
    print(f"Model saved: {save_path}")

if __name__ == "__main__":
    main()