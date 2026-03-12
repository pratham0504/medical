class FLConfig:
    def __init__(self, algorithm: str = "fedavg", num_clients: int = 5):
        self.algorithm = algorithm
        self.num_clients = num_clients  # Default: one client per dataset modality
        
        # Training Settings
        self.num_rounds = 5            # Total communication rounds
        self.local_epochs = 1          # Training passes per client per round
        self.batch_size = 32           # Small "pieces" to save memory
        self.frac = 1.0                # Fraction of clients used per round
        
        # Hyperparameters
        self.local_lr = 0.001          # Learning rate
        self.lr = self.local_lr        # Alias used by Client optimizer
        self.global_lr = 1.0
        self.mu = 0.01                 # Proximal term for FedProx / FedDANE
        
        # Device setting (Auto-detects CUDA on Windows)
        self.use_cuda = True           

        # Model Save Path
        self.save_path = "models/global_{algorithm}.pth"