import os
import torch
from torchvision import transforms
from tqdm import tqdm
from diffusers import UNet2DModel, DDPMScheduler
import yaml

def generate(batch_size, unet, scheduler, device, t_steps=1000, enable_bar=False):

    # Set the number of timesteps
    scheduler.set_timesteps(t_steps)

    # latent（batch×4×32×32）
    latents = torch.randn(
        (batch_size,
         unet.config.in_channels,
         unet.config.sample_size,
         unet.config.sample_size),
        device=device
    )
    
    progress_bar = tqdm(scheduler.timesteps, desc="Diffusion Steps") if enable_bar else scheduler.timesteps

    for t in progress_bar:
        
        noise_pred = unet(latents, timestep=t).sample
        latents = scheduler.step(noise_pred, t, latents).prev_sample

    
   
    images_tensor =  (latents.clamp(-1, 1) + 1) / 2
    
    to_pil = transforms.ToPILImage()
    
    # Convert each image in the batch tensor to a PIL Image
    generated_images = [to_pil(img.cpu()) for img in images_tensor]
    return generated_images


def test():
    device = "cuda" if torch.cuda.is_available() else "cpu"
  
    # Load your trained model
    with open("configs/base.yaml", "r") as fp:
        config = yaml.safe_load(fp)

    # UNet2DModel
    unet_config = config["model"]["unet"]
    unet = UNet2DModel(
        sample_size=unet_config["sample_size"],
        in_channels=unet_config["in_channels"],
        out_channels=unet_config["out_channels"],
        layers_per_block=unet_config["layers_per_block"],
        block_out_channels=unet_config["block_out_channels"],
        down_block_types=unet_config["down_block_types"],
        up_block_types=unet_config["up_block_types"],
    ).to(device)

    # checkpoint_path
    checkpoint_path = "../model/checkpoint-13500/pytorch_model/mp_rank_00_model_states.pt"
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    # Load checkpoint
    ckpt = torch.load(checkpoint_path, map_location=device)
    unet.load_state_dict(ckpt.get("module", ckpt))
    unet.eval()

       
    ns = config["model"]["noise_scheduler"]
    scheduler = DDPMScheduler(
        num_train_timesteps=ns["num_train_timesteps"],
        beta_schedule=ns["beta_schedule"],
        clip_sample=ns["clip_sample"],
    )

    save_folder = "generated_images"
    os.makedirs(save_folder, exist_ok=True)

    # Generate images
    total = 10000
    batch_size = 64  # 根據 GPU 顯存調整
    t_steps = ns.get("num_inference_timesteps",100)

    with torch.no_grad():
        for i in tqdm(range(0, total, batch_size), desc="Batches"):
            cur_bs = min(batch_size, total - i)
            images = generate(cur_bs, unet, scheduler, device, t_steps)
            for j, img in enumerate(images):
                idx = i + j
                img.save(os.path.join(save_folder, f"{idx:05d}.png"))

    print("Done: generated 10,000 images.")

if __name__ == "__main__":
    test()