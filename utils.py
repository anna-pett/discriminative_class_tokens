from PIL import Image
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    PNDMScheduler,
    StableDiffusionPipeline,
    UNet2DConditionModel,
    AutoPipelineForText2Image,
    StableDiffusionXLControlNetPipeline,
    ControlNetModel
)
from transformers import CLIPTextModel, CLIPTokenizer
import torchvision.transforms as T
import torch

import kornia

import os
import sys


# From timm.data.constants
IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)


def numpy_to_pil(images):
    """
    Convert a numpy image or a batch of images to a PIL image.
    """
    if images.ndim == 3:
        images = images[None, ...]
    images = (images * 255).round().astype("uint8")
    pil_images = [Image.fromarray(image) for image in images]
    return pil_images


def freeze_params(params):
    for param in params:
        param.requires_grad = False


def transform_img_tensor(image, config):
    """
    Transforms an image based on the specified classifier input configurations.
    """
    if config.classifier == "inet":
        image = kornia.geometry.transform.resize(image, 256, interpolation="bicubic")
        image = kornia.geometry.transform.center_crop(image, (224, 224))
        image = T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])(image)
    elif config.classifier == "ddamfn":
        image = Image.open(image_path).convert('RGB')
        image_tensor = transform(image).unsqueeze(0).to(device)  # Add batch dimension
    else:
        image = kornia.geometry.transform.resize(image, 224, interpolation="bicubic")
        image = kornia.geometry.transform.center_crop(image, (224, 224))
        image = T.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD)(image)
    return image


def prepare_classifier(config):
    if config.classifier == "inet":
        from transformers import ViTForImageClassification

        model = ViTForImageClassification.from_pretrained(
            "google/vit-large-patch16-224"
        ).cuda()
    elif config.classifier == "cub":
        from vitmae import CustomViTForImageClassification

        model = CustomViTForImageClassification.from_pretrained(
            "vesteinn/vit-mae-cub"
        ).cuda()
    elif config.classifier == "inat":
        from vitmae import CustomViTForImageClassification

        model = CustomViTForImageClassification.from_pretrained(
            "vesteinn/vit-mae-inat21"
        ).cuda()
    elif config.classifier == "ddamfn":
        # from networks.DDAM import DDAMNet
        # Add the DDAMFN directory to Python's module search path
        ddamfn_path = os.path.join(os.getcwd(), 'DDAMFN', 'DDAMFN++')
        if ddamfn_path not in sys.path:
            sys.path.append(ddamfn_path)
        
        # Ensure the networks directory is recognized as a Python package
        networks_path = os.path.join(ddamfn_path, 'networks')
        init_file = os.path.join(networks_path, '__init__.py')
        
        if not os.path.exists(init_file):
            os.makedirs(networks_path, exist_ok=True)
            with open(init_file, 'w') as f:
                pass
        
        # Import networks
        try:
            import networks
            from networks.DDAM import DDAMNet
        except ImportError as e:
            raise ImportError(f"Failed to import DDAMNet: {e}")

        # Initialize the model
        model = DDAMNet(num_class=7, pretrained=False)  # Assuming the model has 7 classes
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if not os.path.exists(config.ddamfn_classifier_path):
            raise FileNotFoundError(f"DDAMFN checkpoint not found at {self.ddamfn_classifier_path}")
        checkpoint = torch.load(config.ddamfn_classifier_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()

    return model


def prepare_stable(config):
    # Generative model
    if config.sd_2_1:
        pretrained_model_name_or_path = "stabilityai/stable-diffusion-2-1-base"
    else:
        pretrained_model_name_or_path = "CompVis/stable-diffusion-v1-4"

    unet = UNet2DConditionModel.from_pretrained(
        pretrained_model_name_or_path, subfolder="unet"
    )
    text_encoder = CLIPTextModel.from_pretrained(
        pretrained_model_name_or_path, subfolder="text_encoder"
    )
    if config.is_controlnet:
        # initialize the models and pipeline
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        controlnet = ControlNetModel.from_pretrained(
            config.controlnet_model_path, torch_dtype=torch.float32
        ).to(device)
        # vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float32)
        pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
            config.face_model_path, controlnet=controlnet, torch_dtype=torch.float32, num_inference_steps=config.diffusion_steps
        ).to(device)
        pipe.enable_model_cpu_offload()    # Enable CPU offloading for memory optimization
        pipe.enable_gradient_checkpointing()    # Enable gradient checkpointing to save memory
    else:
        vae = AutoencoderKL.from_pretrained(pretrained_model_name_or_path, subfolder="vae")
        pipe = StableDiffusionPipeline.from_pretrained(pretrained_model_name_or_path).to(
            "cuda"
        )
    scheduler = pipe.scheduler
    del pipe
    tokenizer = CLIPTokenizer.from_pretrained(
        pretrained_model_name_or_path, subfolder="tokenizer"
    )

    return unet, vae, text_encoder, scheduler, tokenizer


def save_progress(text_encoder, placeholder_token_id, accelerator, config, save_path):
    learned_embeds = (
        accelerator.unwrap_model(text_encoder)
        .get_input_embeddings()
        .weight[placeholder_token_id]
    )
    learned_embeds_dict = {config.placeholder_token: learned_embeds.detach().cpu()}
    torch.save(learned_embeds_dict, save_path)


# get canny image for ControlNet
def load_conditioning_image(config):
    # image = np.asarray(PIL.Image.open(f"controlnet/{config.class_index}_face.png"))
    # image_path = "controlnet/conditioning_face.png" 
    image_path = f"controlnet/{config.class_index-1}_face.png"
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Conditioning image file not found: controlnet/conditioning_face.png")
    # from https://faces.mpdl.mpg.de/imeji/
    image = np.asarray(PIL.Image.open(image_path))
    image = cv2.Canny(image, 100, 200)
    image = image[:, :, None]
    image = np.concatenate([image, image, image], axis=2)
    canny_image = Image.fromarray(image)
    return canny_image
    
