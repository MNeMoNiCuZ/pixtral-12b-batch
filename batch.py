import os
import time
import torch
from PIL import Image
from transformers import LlavaForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
from pathlib import Path
from tqdm import tqdm
import argparse

# Configuration options
LOW_VRAM_MODE = True  # Default: Use low-VRAM (quantized model) if True, full model if False
PRINT_CAPTIONS = True  # Print captions to the console during inference
PRINT_CAPTIONING_STATUS = False  # Print captioning file status to the console
OVERWRITE = True  # Allow overwriting existing caption files
PREPEND_STRING = ""  # Prefix string to prepend to the generated caption
APPEND_STRING = ""  # Suffix string to append to the generated caption
STRIP_LINEBREAKS = True  # Remove line breaks from generated captions before saving
DEFAULT_SAVE_FORMAT = ".txt"  # Default format for saving captions

# Image resizing options
MAX_WIDTH = 1024  # Set to 0 or less to ignore
MAX_HEIGHT = 1024  # Set to 0 or less to ignore

# Generation parameters
REPETITION_PENALTY = 1.5  # Penalty for repeating phrases, float ~1.5
TEMPERATURE = 0.7  # Sampling temperature to control randomness
TOP_K = 50  # Top-k sampling to limit number of potential next tokens

# Default values for input folder, output folder, prompt, and save format
DEFAULT_INPUT_FOLDER = Path(__file__).parent / "input"
DEFAULT_OUTPUT_FOLDER = DEFAULT_INPUT_FOLDER

# Prompt parameters
USER_PROMPT = "Describe the image in 10 words"
#USER_PROMPT = ""
"""
Example Captions:

Comma Separated Tags:
   Describe the image in 10 words
   
Short Caption:
    Describe the in one sentence

Medium Description:
    Caption with a short description
    
Long Description:
    Caption this image
    
Very Long Description:
    Describe everything about this image in detail without formatting

"""

DEFAULT_PROMPT_PREFIX = "<s>[INST]"
DEFAULT_PROMPT_SUFFIX = "\n[IMG][/INST]"
DEFAULT_PROMPT = DEFAULT_PROMPT_PREFIX + USER_PROMPT + DEFAULT_PROMPT_SUFFIX

# Function to parse command-line arguments
def parse_arguments():
    parser = argparse.ArgumentParser(description="Process images and generate captions using Pixtral-12B model.")
    parser.add_argument("--input_folder", type=str, default=DEFAULT_INPUT_FOLDER, help="Path to the input folder containing images.")
    parser.add_argument("--output_folder", type=str, default=DEFAULT_OUTPUT_FOLDER, help="Path to the output folder for saving captions.")
    parser.add_argument("--prompt", type=str, default=DEFAULT_PROMPT, help="Prompt for generating the caption.")
    parser.add_argument("--save_format", type=str, default=DEFAULT_SAVE_FORMAT, help="Format for saving captions (e.g., .txt, .md, .json).")
    parser.add_argument("--max_width", type=int, default=MAX_WIDTH, help="Maximum width for resizing images (default: no resizing).")
    parser.add_argument("--max_height", type=int, default=MAX_HEIGHT, help="Maximum height for resizing images (default: no resizing).")
    parser.add_argument("--repetition_penalty", type=float, default=REPETITION_PENALTY, help="Penalty for repetition during caption generation (default: 1.10).")
    parser.add_argument("--temperature", type=float, default=TEMPERATURE, help="Sampling temperature for generation (default: 0.7).")
    parser.add_argument("--top_k", type=int, default=TOP_K, help="Top-k sampling during generation (default: 50).")
    parser.add_argument("--low_vram_mode", action="store_true", help="Use the low VRAM model (quantized).")
    return parser.parse_args()

# Function to ignore images that don't have output files yet
def filter_images_without_output(input_folder, save_format):
    images_to_caption = []
    skipped_images = 0
    total_images = 0

    for root, _, files in os.walk(input_folder):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                total_images += 1
                image_path = os.path.join(root, file)
                output_path = os.path.splitext(image_path)[0] + save_format
                if not OVERWRITE and os.path.exists(output_path):
                    skipped_images += 1
                else:
                    images_to_caption.append(image_path)

    return images_to_caption, total_images, skipped_images

# Function to save caption to a file
def save_caption_to_file(image_path, caption, save_format):
    txt_file_path = os.path.splitext(image_path)[0] + save_format  # Same name, but with chosen save format
    caption = PREPEND_STRING + caption + APPEND_STRING  # Apply prepend/append strings

    with open(txt_file_path, "w") as txt_file:
        txt_file.write(caption)

    if PRINT_CAPTIONING_STATUS:
        print(f"\nCaption for {os.path.abspath(image_path)} saved in {save_format} format.")

# Function to process all images recursively in a folder
def process_images_in_folder(images_to_caption, prompt, save_format, max_width=MAX_WIDTH, max_height=MAX_HEIGHT, repetition_penalty=REPETITION_PENALTY, temperature=TEMPERATURE, top_k=TOP_K, model=None, processor=None):
    
    for image_path in tqdm(images_to_caption, desc="Processing Images"):
        try:
            image = Image.open(image_path).convert("RGB")
            
            # Resize the image if necessary
            image = resize_image_proportionally(image, max_width, max_height)
            
            caption = generate_caption(image, prompt, repetition_penalty, temperature, top_k, model, processor)
            save_caption_to_file(image_path, caption, save_format)

            if PRINT_CAPTIONS:
                print(f"\nCaption for {os.path.abspath(image_path)}: {caption}")

        except Exception as e:
            print(f"Error processing {os.path.abspath(image_path)}: {str(e)}")

        torch.cuda.empty_cache()

# Resize the image proportionally based on max width and/or max height.
def resize_image_proportionally(image, max_width=None, max_height=None):
    if (max_width is None or max_width <= 0) and (max_height is None or max_height <= 0):
        return image  # No resizing if both dimensions are not provided or set to 0 or less

    original_width, original_height = image.size

    # Check if resizing is needed
    if ((max_width is None or original_width <= max_width) and
        (max_height is None or original_height <= max_height)):
        return image  # Image is within the specified dimensions, no resizing needed

    # Calculate the scaling ratio, keeping aspect ratio
    if max_width and max_height:
        width_ratio = max_width / original_width
        height_ratio = max_height / original_height
        ratio = min(width_ratio, height_ratio)
    elif max_width:
        ratio = max_width / original_width
    else:  # max_height is specified
        ratio = max_height / original_height

    new_width = int(original_width * ratio)
    new_height = int(original_height * ratio)

    resized_image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    return resized_image

# Generate a caption for the provided image using the Pixtral-12B model
def generate_caption(image, prompt, repetition_penalty, temperature, top_k, model, processor):
    inputs = processor(images=[image], text=prompt, return_tensors="pt").to("cuda")
    prompt_tokens = len(inputs['input_ids'][0])
    print(f"Prompt tokens: {prompt_tokens}")

    t0 = time.time()
    generate_ids = model.generate(**inputs, max_new_tokens=512)
    t1 = time.time()
    total_time = t1 - t0
    generated_tokens = len(generate_ids[0]) - prompt_tokens
    time_per_token = generated_tokens / total_time
    print(f"Generated {generated_tokens} tokens in {total_time:.3f} s ({time_per_token:.3f} tok/s)")

    # Decode the output, but remove the instruction part ("Caption this image:")
    output_text = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

    # Remove the instruction from the output
    if output_text.startswith(USER_PROMPT):
        output_text = output_text[len(USER_PROMPT):].strip()  # Remove the instruction

    if STRIP_LINEBREAKS:
        output_text = output_text.replace('\n', ' ')

    return output_text

# Run the script
if __name__ == "__main__":
    args = parse_arguments()

    # Override LOW_VRAM_MODE based on command-line argument
    low_vram_mode = args.low_vram_mode if args.low_vram_mode else LOW_VRAM_MODE

    input_folder = args.input_folder
    output_folder = args.output_folder
    prompt = args.prompt
    save_format = args.save_format
    max_width = args.max_width
    max_height = args.max_height
    repetition_penalty = args.repetition_penalty
    temperature = args.temperature
    top_k = args.top_k

    # Choose the model based on low_vram_mode flag
    if low_vram_mode:
        model_id = "SeanScripts/pixtral-12b-nf4"
        print(f"Using low VRAM model (quantized): {model_id}")
        # Load model efficiently with safetensors and direct GPU mapping
        model = LlavaForConditionalGeneration.from_pretrained(
            model_id,
            use_safetensors=True,
            device_map="cuda:0"
        )
        processor = AutoProcessor.from_pretrained(model_id)
    else:
        model_id = "mistral-community/pixtral-12b"
        print(f"Using full model (no quantization): {model_id}")
        model = LlavaForConditionalGeneration.from_pretrained(model_id).to("cuda")

    processor = AutoProcessor.from_pretrained(model_id)

    # Filter images before processing
    images_to_caption, total_images, skipped_images = filter_images_without_output(input_folder, save_format)

    # Print summary of found, skipped, and to-be-processed images
    print(f"\nFound {total_images} image{'s' if total_images != 1 else ''}.")
    if not OVERWRITE:
        print(f"{skipped_images} image{'s' if skipped_images != 1 else ''} already have captions with format {save_format}, skipping.")
    print(f"\nCaptioning {len(images_to_caption)} image{'s' if len(images_to_caption) != 1 else ''}.\n\n")

    # Only process if there are images to caption
    if len(images_to_caption) == 0:
        print("No images to process. Exiting.\n\n")
    else:
        # Process the images with optional resizing and caption generation
        process_images_in_folder(
            images_to_caption,
            prompt,
            save_format,
            max_width=max_width,
            max_height=max_height,
            repetition_penalty=repetition_penalty,
            temperature=temperature,
            top_k=top_k,
            model=model,
            processor=processor
        )
