# Prompt Sliders for Fine-Grained Control, Editing and Erasing of Concepts in Diffusion Models
We introduce the Prompt Slider method for precise manipulation, editing, and erasure of concepts in diffusion models. [Project Page](https://deepaksridhar.github.io/promptsliders.github.io/)


### Installing the dependencies

Before running the scripts, make sure to install the library's training dependencies:

You can install diffusers directly from pip or install from the latest version. To do this, execute one of the following steps in a new virtual environment:

Install with pip
```bash
pip install diffusers==0.27
```

Install from source
```bash
git clone https://github.com/huggingface/diffusers
cd diffusers
pip install .
```

Then cd in the promptsliders folder (you can also copy it to the examples folder in diffusers) and run:
```bash
pip install -r requirements.txt
```

And initialize an [🤗 Accelerate](https://github.com/huggingface/accelerate/) environment with:

```bash
accelerate config
```

Now we can launch the training using:

```bash
export MODEL_NAME="runwayml/stable-diffusion-v1-5"
export EMOTION="smiling"

accelerate launch textual_inversion.py \
    --pretrained_model_name_or_path=$MODEL_NAME \
    --learnable_property="object" \
    --placeholder_token="<$EMOTION-lora>" \
    --initializer_token="$EMOTION" \
    --mixed_precision="no" \
    --resolution=512 \
    --train_batch_size=1 \
    --gradient_accumulation_steps=1 \
    --max_train_steps=2000 \
    --learning_rate=5.0e-04 \
    --scale_lr \
    --lr_scheduler="constant" \
    --lr_warmup_steps=0 \
    --save_as_full_pipeline \
    --output_dir=outputs/$EMOTION-promptslider/ \
    --prompts_file="textsliders/data/prompts-$EMOTION.yaml"
```

Alternatively, one could run with default settings

```bash
bash prompt_slider_emotions.sh
```

A full training run takes ~1-2 hours on one A10 GPU.

### Inference

Once you have trained a model using above command, the inference can be done simply using the `StableDiffusionPipeline` wih the following script. Make sure to modify your prompt.

```bash
python inference_sd.py $path_to_the_saved_embedding $token_name
```
## Acknowledgements

Thanks to [diffusers](https://github.com/huggingface/diffusers) and [Concept Sliders](https://github.com/rohitgandikota/sliders)!
