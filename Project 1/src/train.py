import torch
import optuna
import pytorch_lightning as L
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision.datasets import MNIST
from torchvision import transforms
from pytorch_lightning.loggers import MLFlowLogger
from pytorch_lightning.callbacks import ModelCheckpoint

class MNISTDataSet(Dataset):
    def __init__(self, train=True):
        self.data = MNIST(root='../data', train=train, download=True, transform=transforms.ToTensor())

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image, label = self.data[idx]
        return image, label

class MNISTModel(L.LightningModule):
    def __init__(self, learning_rate=1e-3):
        super().__init__()
        self.learning_rate = learning_rate
        self.save_hyperparameters()

        self.layers = torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Flatten(),
            torch.nn.Linear(64 * 7 * 7, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 10)
        )
        self.loss_fn = torch.nn.CrossEntropyLoss()
    
    def training_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self.layers(images)
        loss = self.loss_fn(outputs, labels)
        self.log('train_loss', loss, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self.layers(images)
        loss = self.loss_fn(outputs, labels)
        self.log('val_loss', loss, on_step=False, on_epoch=True)
        return loss
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
    
def objective(trial):
    lr = trial.suggest_float("lr", 1e-4, 1e-1, log=True)

    model = MNISTModel(learning_rate=lr)
    
    full_dataset = MNISTDataSet(train=True)
    train_set, val_set = random_split(full_dataset, [50000, 10000])
    train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=32, shuffle=False)

    mlflow_logger = MLFlowLogger(
        experiment_name="MNIST_Optuna_Tuning",
        run_name=f"Trial_{trial.number}"
    )

    trainer = L.Trainer(
        max_epochs=2, 
        logger=mlflow_logger,
        enable_progress_bar=True 
    )
    
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    
    return trainer.callback_metrics["val_loss"].item()

def main():
    optimize = True 

    if optimize:
        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=5)

    full_dataset = MNISTDataSet(train=True)
    train_set, val_set = random_split(full_dataset, [50000, 10000])
    
    train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=32, shuffle=False)

    model = MNISTModel(learning_rate=study.best_params["lr"]) if optimize else MNISTModel()
    
    mlflow_logger = MLFlowLogger(
        experiment_name="MNIST_Homework",
        run_name="Train_vs_Val_Run"
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints",
        filename="best-model",
        monitor="val_loss",
        mode="min"
    )

    trainer = L.Trainer(
        max_epochs=5, 
        logger=mlflow_logger,
        callbacks=[checkpoint_callback]
    )
    
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

if __name__ == "__main__":    
    main()