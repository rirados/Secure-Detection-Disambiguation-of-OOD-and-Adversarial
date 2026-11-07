import torch
import config

def generate_adversarial_image(model, image, label, epsilon=0.01):
    image = image.clone().detach().to(config.DEVICE).requires_grad_(True)

    outputs = model(image)[:, :config.NUM_CLASSES]
    loss = torch.nn.functional.cross_entropy(outputs, label)

    model.zero_grad()
    loss.backward()

    adv = image + epsilon * image.grad.data.sign()
    adv = torch.clamp(adv, image.min(), image.max())

    return adv.detach()