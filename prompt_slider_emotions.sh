#!/bin/bash

# Define the model name and data directory as environment variables
export MODEL_NAME="runwayml/stable-diffusion-v1-5"

# List of emotions
# emotions=("sad" "disgusted" "confused" "fear" "surprised" "angry")
emotions=("smiling" "surprised")

# Loop through each emotion and run the script
for EMOTION in "${emotions[@]}"; do
    echo "Running script for emotion: $EMOTION"
    
    # Run the script with the current emotion
    accelerate launch textual_inversion.py \
        --pretrained_model_name_or_path=$MODEL_NAME \
        --learnable_property="object" \
        --placeholder_token="<$EMOTION-lora>" \
        --initializer_token="$EMOTION" \
        --mixed_precision="no" \
        --resolution=768 \
        --train_batch_size=1 \
        --gradient_accumulation_steps=1 \
        --max_train_steps=2000 \
        --learning_rate=5.0e-04 \
        --scale_lr \
        --lr_scheduler="constant" \
        --lr_warmup_steps=0 \
        --save_as_full_pipeline \
        --output_dir=outputs/v2-$EMOTION-promptslider/ \
        --prompts_file="textsliders/data/prompts-$EMOTION.yaml"
    
    echo "Completed: $EMOTION"
    echo "-----------------------------------"
done
