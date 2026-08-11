import torch

from .regions import region_token_indices


def capture_block_activation(model, image_tensor, layer_idx):
    captured = {}

    def capture_hook(module, inputs, output):
        captured["activation"] = output.detach().clone()

    block = model.visual.trunk.blocks[layer_idx]
    handle = block.register_forward_hook(capture_hook)

    try:
        with torch.no_grad():
            model.encode_image(image_tensor, normalize=True)
    finally:
        handle.remove()

    if "activation" not in captured:
        raise RuntimeError("The activation hook did not run")

    return captured["activation"]


def run_patched_margin(
    model,
    target_tensor,
    donor_activation,
    text_features,
    layer_idx,
    region,
    alpha,
    random_seed=0,
):
    device = target_tensor.device

    token_idx = torch.tensor(
        region_token_indices(region, random_seed=random_seed),
        dtype=torch.long,
        device=device,
    )
    donor_activation = donor_activation.to(device)

    def patch_hook(module, inputs, output):
        patched = output.clone()
        patched[:, token_idx, :] = (
            (1.0 - alpha) * output[:, token_idx, :]
            + alpha * donor_activation[:, token_idx, :]
        )
        return patched

    block = model.visual.trunk.blocks[layer_idx]
    handle = block.register_forward_hook(patch_hook)

    try:
        with torch.no_grad():
            features = model.encode_image(target_tensor, normalize=True)
            similarities = features @ text_features.T
            margin = similarities[:, 1] - similarities[:, 0]
    finally:
        handle.remove()

    return float(margin.item())