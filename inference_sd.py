
# coding: utf-8

import torch
import os
import sys
import numpy as np
from PIL import Image
from tqdm import tqdm
import requests
from typing import Union
import PIL
from diffusers import StableDiffusionPipeline

model = 'runwayml/stable-diffusion-v1-5'

device = 'cuda'

pipe = StableDiffusionPipeline.from_pretrained(model,safety_checker = None,)

concept = sys.argv[1]
token = sys.argv[2]
textual_inversion_embeds_path = f"{concept}"
pipe.load_textual_inversion(textual_inversion_embeds_path, token=f"{token}")                                                             
pipe.to(device)
image = pipe(f"An image of a male, {token}", num_inference_steps=50).images[0]
token_name = token.replace('<','').replace('>','')
image.save(f"outputs/{token_name}-promptslider/output/ti_{token_name}.png")
