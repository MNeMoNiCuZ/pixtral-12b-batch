# Pixtral 12b - Batch Captioning
This tool utilizes the Pixtral 12-b model to caption image files in a batch.

Place all images you wish to caption in the /input directory and run `py batch.py`.

Support for an `--img_dir` argument added by [CambridgeComputing](https://github.com/CambridgeComputing) It lets you specify a directory other than ./input. If no arguments are provided by the user, the script still defaults to the ./input subdirectory.

Now supports `LOW_VRAM_MODE=true`. This will use a [llama3-8b-bnb-4bit quantized version from unsloth](https://huggingface.co/unsloth/llama-3-8b-bnb-4bit)

# Setup
1. Git clone this repository `git clone https://github.com/MNeMoNiCuZ/pixtral-12b-batch`
2. (Optional) Create a virtual environment for your setup. Use python 3.9 to 3.11. Do not use 3.12. Feel free to use the `venv_create.bat` for a simple windows setup. Activate your venv.
3. Run `pip install -r requirements.txt` (this is done automatically with the `venv_create.bat`).
4. Install [PyTorch with CUDA support](https://pytorch.org/) matching your installed CUDA version. Run `nvcc --version` to find out which CUDA is your default.

You should now be set up and able to run the tool.

# Requirements
- Tested on Python 3.11.
- Tested on Pytorch w. CUDA 12.1.
- 12gb VRAM for the `LOW_VRAM_MODE` ([NF4 (4-bit) quantization](https://huggingface.co/SeanScripts/pixtral-12b-nf4))
- \> 24gb VRAM for [full precision model](https://huggingface.co/mistral-community/pixtral-12b)

> [!CAUTION]
> This model requires a lot of VRAM. Make sure to turn on LOW_VRAM_MODE if you are on or below 24gb VRAM on your GPU.
> This is set to true by default.

# Prompting
Change the `USER_PROMPT` variable to give instructions to the model.

> [!IMPORTANT]
> The longer the resulting prompt, the longer it takes to generate.

> A 10 word caption can take 3 seconds with 24gb VRAM

> A 250 word caption can take 90 seconds with 24gb VRAM

## Example Captions
![example_image](https://github.com/user-attachments/assets/2f086927-6e46-4eb0-89ac-0e2fb7a5ede1)

### Comma Separated Tags
Describe the image in 10 words
> Vibrant city night, neon lights, masks, skateboard, couple, chandelier, crowd, futuristic.
---
### Short Caption
Describe the in one sentence
> This image depicts a vibrant, futuristic scene in a bustling city at night, featuring two individuals in stylish, neon-accented outfits and masks, standing on a skateboard with a crowd of people in the background.
---
### Medium Caption
Caption with a short description
> This image depicts a futuristic, neon-lit cityscape, likely inspired by New York City's Times Square. It features two individuals, a man and a woman, both dressed in stylish, modern outfits with a cyberpunk aesthetic. They are standing on a skateboard, with the woman holding it while the man stands beside her. Both are wearing masks that resemble animal heads, adding an element of mystery and intrigue to their appearance. The background is vibrant with neon lights and bustling with people, creating a lively and dynamic atmosphere. The scene is set at night, with the city's skyscrapers and iconic buildings illuminated by colorful, futuristic lighting.
---
### Long Caption
Caption this image:
> This image depicts a vibrant, futuristic scene set in a bustling city at night. Two individuals, a man and a woman, are the focal point of the image. Both are dressed in stylish, modern outfits with a cyberpunk aesthetic, featuring leather jackets, skinny jeans, and sneakers. They are standing on a skateboard, with the woman balancing on one foot and the man on two. Both are wearing unique, intricate masks that cover their eyes, adding an air of mystery to their personas.  The background is a lively cityscape, reminiscent of Times Square in New York City, with towering skyscrapers and neon lights illuminating the night. The street is crowded with people, suggesting a festive or celebratory atmosphere. The buildings are adorned with various advertisements and signs, contributing to the urban, energetic vibe of the scene.  Above the characters, a large, ornate chandelier hangs, adding a touch of elegance and contrast to the otherwise modern and futuristic setting. The overall color palette is dominated by shades of blue, purple, and pink, creating a visually striking and dynamic composition.
---
### Very Long Caption
Describe everything about this image in detail without formatting
> The image depicts a vibrant, futuristic scene set in a bustling city at night. The focal point is a neon-lit street, reminiscent of Times Square in New York City, with towering skyscrapers and illuminated billboards in the background. The sky is dark, adding to the nighttime ambiance.  In the foreground, two individuals stand out prominently. The person on the left is a woman dressed in a black leather jacket, black crop top, and black leggings, with white sneakers. She has pink hair styled in a high ponytail and wears a mask that resembles a marbled, abstract design. She holds a skateboard with one foot on it, suggesting she might be preparing to ride or has just stopped.  Beside her is a man dressed in a black, sleeveless jumpsuit with long sleeves and black boots. He has a hood that covers his head and wears a similar marbled mask. He stands confidently with one foot on the skateboard, mirroring the woman's stance.  Both individuals have distinctive, stylized hair with sharp, angular points, adding to their futuristic appearance. Their masks and outfits suggest a theme of anonymity or perhaps a futuristic, cyberpunk aesthetic.  The street is crowded with people, all dressed in dark, casual attire, contributing to the lively atmosphere. The neon lights from the billboards and signs cast a colorful glow over the scene, enhancing the futuristic, almost surreal quality of the image. The overall mood is energetic and dynamic, capturing a moment of urban excitement and adventure.
---

# Known issues
## Not enough memory?
You may want to run the model in Low VRAM mode: Set `LOW_VRAM_MODE=true` in `batch.py`
