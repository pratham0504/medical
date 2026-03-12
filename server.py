import copy
import torch
import torch.nn.functional as F
from model import MNIST_CNN

class Server:
    def __init__(self, config, train_loaders, test_loader, device, dataset_names):
        self.config = config
        self.train_loaders = train_loaders
        self.test_loader = test_loader
        self.device = device
        self.dataset_names = dataset_names
        self.global_model = MNIST_CNN(num_classes=len(dataset_names)).to(self.device)

    def train_clients(self, selected_clients):
        local_models = []
        for idx in selected_clients:
            d_name = self.dataset_names[idx]
            print(f"  [Dataset: {d_name}] Training...")
            local_model = copy.deepcopy(self.global_model)
            local_model.train()
            optimizer = torch.optim.SGD(
                local_model.parameters(),
                lr=self.config.local_lr,
                momentum=getattr(self.config, "momentum", 0.0),
            )

            for epoch in range(self.config.local_epochs):
                for b_idx, (data, target) in enumerate(self.train_loaders[idx]):
                    data, target = data.to(self.device), target.to(self.device)
                    optimizer.zero_grad()
                    output = local_model(data)
                    loss = F.cross_entropy(output, target)
                    loss.backward()
                    optimizer.step()
                    if b_idx % 100 == 0:
                        print(f"    Batch {b_idx}/{len(self.train_loaders[idx])} Loss: {loss.item():.4f}")
            local_models.append(local_model)
        return local_models

    def avg_grads(self, local_models, selected_clients):
        sample_counts = [len(self.train_loaders[idx].dataset) for idx in selected_clients]
        total_samples = sum(sample_counts)
        if total_samples == 0:
            return

        weights = torch.tensor(
            [count / total_samples for count in sample_counts],
            dtype=torch.float32,
            device=self.device,
        )

        global_dict = self.global_model.state_dict()
        for key in global_dict.keys():
            stacked = torch.stack(
                [lm.state_dict()[key].float().to(self.device) for lm in local_models], dim=0
            )
            view_shape = [len(local_models)] + [1] * (stacked.dim() - 1)
            global_dict[key] = (stacked * weights.view(*view_shape)).sum(dim=0).to(global_dict[key].device)
        self.global_model.load_state_dict(global_dict)

    def evaluate(self):
        self.global_model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.global_model(data)
                correct += (output.argmax(dim=1) == target).sum().item()
                total += target.size(0)
        return 100.0 * correct / total if total > 0 else 0

    def run(self):
        for r in range(1, self.config.num_rounds + 1):
            print(f"\n=== ROUND {r} ===")
            selected_clients = list(range(len(self.train_loaders)))
            local_models = self.train_clients(selected_clients)
            self.avg_grads(local_models, selected_clients)
            acc = self.evaluate()
            print(f"ROUND {r} FINISHED | Global Accuracy: {acc:.2f}%")