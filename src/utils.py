from torchvision import transforms
from torch.utils.data import DataLoader
import config
from src.nih_dataset import NIHChestXrayDataset


def get_dataloader():
    transform = transforms.Compose([
        transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    dataset = NIHChestXrayDataset(split="train", transform=transform)

    loader = DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=False)
    return loader