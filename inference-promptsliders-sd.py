#!/usr/bin/env python
# coding: utf-8
#Author: Deepak Sridhar


import torch
from PIL import Image
import argparse
import os, json, random, sys
import pandas as pd
import matplotlib.pyplot as plt
import glob, re
import warnings
warnings.filterwarnings("ignore")


from tqdm import tqdm
import numpy as np

from safetensors.torch import load_file
import matplotlib.image as mpimg
import copy
import gc
from transformers import CLIPTextModel, CLIPTokenizer

import diffusers
from diffusers import DiffusionPipeline
from diffusers import AutoencoderKL, DDPMScheduler, DiffusionPipeline, UNet2DConditionModel, LMSDiscreteScheduler
from diffusers.loaders import AttnProcsLayers
from diffusers.models.attention_processor import LoRAAttnProcessor, AttentionProcessor
from typing import Any, Dict, List, Optional, Tuple, Union

import safetensors.torch

def flush():
    torch.cuda.empty_cache()
    gc.collect()
flush()
pretrained_model_name_or_path = "runwayml/stable-diffusion-v1-5"

revision = None
device = 'cuda:0'
concept = sys.argv[1]
mconcept = "iid-1"
weight_dtype = torch.float32

# Load scheduler, tokenizer and models.
noise_scheduler = LMSDiscreteScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000)
tokenizer = CLIPTokenizer.from_pretrained(
    pretrained_model_name_or_path, subfolder="tokenizer", revision=revision
)
text_encoder = CLIPTextModel.from_pretrained(
    pretrained_model_name_or_path, subfolder="text_encoder", revision=revision
)
vae = AutoencoderKL.from_pretrained(pretrained_model_name_or_path, subfolder="vae", revision=revision)
unet = UNet2DConditionModel.from_pretrained(
    pretrained_model_name_or_path, subfolder="unet", revision=revision
)

new_token = f"<{mconcept}>"
learned_embeds_path = f"output/{concept}-slider_prompt/learned_embeds.safetensors"
loaded_embeds = safetensors.torch.load_file(learned_embeds_path)

# Check if the token already exists in the vocabulary
if new_token not in tokenizer.get_vocab():
    # Add the token to the tokenizer
    tokenizer.add_tokens([new_token])
    new_token_id = tokenizer.convert_tokens_to_ids(new_token)
    # Resize the model’s token embeddings to accommodate the new token
    text_encoder.resize_token_embeddings(len(tokenizer))

keyy = list(loaded_embeds.keys())[0]
new_token_embed = loaded_embeds[keyy]

with torch.no_grad():
    text_encoder.get_input_embeddings().weight.data[new_token_id] = new_token_embed.clone()


# freeze parameters of models to save more memory
unet.requires_grad_(False)
unet.to(device, dtype=weight_dtype)
vae.requires_grad_(False)
vae.to(device, dtype=weight_dtype)
text_encoder.requires_grad_(False)
text_encoder.to(device, dtype=weight_dtype)

# prompts to try
prompts = [ 
            f"a photo of a man, <{mconcept}>",
           ]
# scale to test
scales = [0, 0.5, 1.0, 1.25]

# timestep during inference when we switch to scale>0 (this is done to ensure structure in the images)
start_noise = 800


#number of images per prompt
num_images_per_prompt = 1

torch_device = device
negative_prompt = None
batch_size = 1
height = 512
width = 512
ddim_steps = 50
guidance_scale = 7.5
unet = UNet2DConditionModel.from_pretrained(
    pretrained_model_name_or_path, subfolder="unet", revision=revision
)
# freeze parameters of models to save more memory
unet.requires_grad_(False)
unet.to(device, dtype=weight_dtype)

for prompt in prompts:
    # for different seeds on same prompt
    for _ in range(num_images_per_prompt):
        seed = random.randint(0, 5000)
        

        
        images_list = []

        print(prompt, seed)

        for scale in scales:
            generator = torch.manual_seed(seed) 
            text_input = tokenizer(prompt, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt")
            idx = text_input.input_ids.argmax(-1)
            
            

            max_length = text_input.input_ids.shape[-1]
            batch_indices = torch.arange(len(text_input.input_ids))
            if negative_prompt is None:
                uncond_input = tokenizer(
                    [""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
                )
            else:
                uncond_input = tokenizer(
                    [negative_prompt] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
                )
            uncond_embeddings = text_encoder(uncond_input.input_ids.to(torch_device))[0]

            

            latents = torch.randn(
                (batch_size, unet.in_channels, height // 8, width // 8),
                generator=generator,
            )
            latents = latents.to(torch_device)

            noise_scheduler.set_timesteps(ddim_steps)

            latents = latents * noise_scheduler.init_noise_sigma
            latents = latents.to(weight_dtype)
            latent_model_input = torch.cat([latents] * 2)
            
            for t in tqdm(noise_scheduler.timesteps):
                text_embeddings = text_encoder(text_input.input_ids.to(torch_device))[0]
                if t>start_noise and scale > 0.0:
                    text_embeddings[batch_indices, idx, :] = 0.0 * text_embeddings[batch_indices, idx, :]
                else:
                    text_embeddings[batch_indices, idx, :] = scale * text_embeddings[batch_indices, idx, :]
                
                concat_text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
                # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
                latent_model_input = torch.cat([latents] * 2)

                latent_model_input = noise_scheduler.scale_model_input(latent_model_input, timestep=t)
                # predict the noise residual
                
                with torch.no_grad():
                    noise_pred = unet(latent_model_input, t, encoder_hidden_states=concat_text_embeddings).sample
                # perform guidance
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                latents = noise_scheduler.step(noise_pred, t, latents).prev_sample

            # scale and decode the image latents with vae
            latents = 1 / 0.18215 * latents
            with torch.no_grad():
                image = vae.decode(latents).sample
            image = (image / 2 + 0.5).clamp(0, 1)
            image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
            images = (image * 255).round().astype("uint8")
            pil_images = [Image.fromarray(image) for image in images]
            images_list.append(pil_images[0])

        fig, ax = plt.subplots(1, len(images_list), figsize=(20,4))
        for i, a in enumerate(ax):
            a.imshow(images_list[i])
            a.set_title(f"{scales[i]}",fontsize=15)
            a.axis('off')

        plt.show()
        plt.savefig(f'{prompt}.jpg')
plt.close()
