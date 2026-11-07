import torch

captured_features = []


def hook_fn(module, inputs, output):
    if isinstance(output, tuple):
        output = output[0]

    if output.dim() == 4:
        pooled = torch.nn.functional.adaptive_avg_pool2d(output, (1, 1))
        features = pooled.flatten(1)
        captured_features.append(features)


def register_hooks(model):
    layers = [
        model.layer1,
        model.layer2,
        model.layer3,
        model.layer4
    ]

    handles = []

    for layer in layers:
        handle = layer.register_forward_hook(hook_fn)
        handles.append(handle)

    return handles