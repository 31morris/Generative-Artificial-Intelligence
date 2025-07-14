import json
import os
import yaml
import torch
from torchvision import transforms
from tqdm import tqdm
from diffusers import UNet2DConditionModel
from diffusers import AutoencoderKL, DDPMScheduler,DDIMScheduler
from transformers import CLIPTokenizer, CLIPTextModel

def generate(prompts, vae, text_encoder, tokenizer, unet, device, enable_bar=False, t_steps=50):

    scheduler = DDPMScheduler(
    num_train_timesteps=1000,
    beta_schedule="squaredcos_cap_v2",
    clip_sample=False,
)

    scheduler.set_timesteps(t_steps) # Use fewer steps for faster preview

    batch_size = len(prompts) # Batch size is now determined by the number of prompts
    
    # Tokenize and encode all prompts in the batch
    text_inputs = tokenizer(prompts, return_tensors="pt", padding="max_length", truncation=True, max_length=77).input_ids.to(device)
    cond_emb = text_encoder(text_inputs)[0]
    encoder_input = cond_emb

    # Initialize latents for the entire batch
    latents = torch.randn((batch_size, 4, 32, 32)).to(device)
    # latents = latents * 0.18215 # Apply VAE scaling factor
    
    progress_bar = tqdm(scheduler.timesteps, desc="Diffusion Steps") if enable_bar else scheduler.timesteps

    for t in progress_bar:
        # timestep_input needs to be broadcasted for each item in the batch
        # Create a tensor of shape (batch_size,) with the current timestep
        timestep_input = torch.full((batch_size,), t, dtype=torch.long, device=device)

        # Predict noise and sample x_(t-1)
        noise_pred = unet(
            latents,
            timestep=timestep_input,
            encoder_hidden_states=encoder_input
        ).sample

        latents = scheduler.step(noise_pred, t, latents).prev_sample

    # Decode latents to image for the entire batch
    latents = latents / 0.18215
    
    # vae.decode returns a tensor of images
    images_tensor = vae.decode(latents, return_dict=False)[0]
    images_tensor = (images_tensor.clamp(-1, 1) + 1) / 2 # [-1,1] to [0,1]
    
    to_pil = transforms.ToPILImage()
    
    # Convert each image in the batch tensor to a PIL Image
    generated_images = [to_pil(img.cpu()) for img in images_tensor]
    return generated_images


def test():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint_path = os.path.join("../model/checkpoints", "checkpoint-58500", "pytorch_model", "mp_rank_00_model_states.pt")

    # Load your trained model
    with open("configs/base.yaml", "r") as fp:
        config = yaml.safe_load(fp)

    # UNet
    unet = UNet2DConditionModel(**config["model"]["unet"]).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    unet.load_state_dict(checkpoint["module"] if "module" in checkpoint else checkpoint)
    unet.eval()


    # Check if the checkpoint exists
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        print("Please ensure the checkpoint path is correct and the file exists.")
        return
    

    # Load pre-trained CLIP tokenizer and text encoder
    pretrain_CLIP_path = "openai/clip-vit-base-patch32"
    pretrain_VAE_path = "CompVis/stable-diffusion-v1-4"
    tokenizer = CLIPTokenizer.from_pretrained(pretrain_CLIP_path)
    text_encoder = CLIPTextModel.from_pretrained(pretrain_CLIP_path).eval().to(device)
    text_encoder.requires_grad_(False)

    # Load pre-trained VAE
    vae = AutoencoderKL.from_pretrained(pretrain_VAE_path, subfolder="vae").to(device)
    vae.requires_grad_(False)
    
    with open("test.json", "r") as f:
        test_data_raw = json.load(f)

    save_folder = "results"
    os.makedirs(save_folder, exist_ok=True)
    
    # Prepare data for batching
    all_prompts = []
    all_image_names = []
    for key, value in test_data_raw.items():
        all_prompts.append(value["text_prompt"])
        all_image_names.append(value["image_name"])

    # Define a batch size for processing. You can adjust this based on your GPU memory.
    batch_size = 16 # Example batch size, adjust as needed

    with torch.no_grad():
        # Iterate through the prompts in batches
        for i in tqdm(range(0, len(all_prompts), batch_size), desc="Generating Batches"):
            current_prompts = all_prompts[i:i + batch_size]
            current_image_names = all_image_names[i:i + batch_size]

            # Generate images for the current batch
            generated_images = generate(current_prompts, vae, text_encoder, tokenizer, unet, device, enable_bar=False, t_steps=50) # enable_bar for individual diffusion steps per batch

            # Save each image in the current batch
            for img_idx, img in enumerate(generated_images):
                image_name_to_save = current_image_names[img_idx]
                img.save(os.path.join(save_folder, image_name_to_save))

if __name__ == "__main__":
    test()