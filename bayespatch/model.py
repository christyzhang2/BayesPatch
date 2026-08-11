import open_clip
import torch


MODEL_NAME = (
    "hf-hub:microsoft/"
    "BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
)

DEFAULT_PROMPTS = [
    "normal retina without diabetic retinopathy",
    "retina with diabetic retinopathy lesions",
]


def load_biomedclip(device=None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model, _, preprocess = open_clip.create_model_and_transforms(MODEL_NAME)
    tokenizer = open_clip.get_tokenizer(MODEL_NAME)

    model = model.to(device).eval()

    for parameter in model.parameters():
        parameter.requires_grad_(False)

    return model, preprocess, tokenizer, device


@torch.no_grad()
def encode_prompts(model, tokenizer, prompts, device):
    tokens = tokenizer(prompts).to(device)
    return model.encode_text(tokens, normalize=True)


@torch.no_grad()
def image_margin(model, image_tensor, text_features):
    features = model.encode_image(image_tensor, normalize=True)
    similarities = features @ text_features.T
    return similarities[:, 1] - similarities[:, 0]