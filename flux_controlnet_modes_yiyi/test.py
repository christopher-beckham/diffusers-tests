import torch
import gc

###################################################
# fix control mode when num_images_per_prompt > 1 #
###################################################

# num_images_per_prompt = 1
num_images_per_prompt = 2
branch = "test" 
# branch = "main"

def flush():
    gc.collect()
    torch.cuda.empty_cache()


"""
# test1: single controlnet (canny)
import torch
from diffusers.utils import load_image
from diffusers.pipelines.flux.pipeline_flux_controlnet import FluxControlNetPipeline
from diffusers.models.controlnet_flux import FluxControlNetModel

base_model = 'black-forest-labs/FLUX.1-dev'
controlnet_model = 'InstantX/FLUX.1-dev-Controlnet-Canny'
controlnet = FluxControlNetModel.from_pretrained(controlnet_model, torch_dtype=torch.bfloat16)
pipe = FluxControlNetPipeline.from_pretrained(base_model, controlnet=controlnet, torch_dtype=torch.bfloat16)
pipe.to("cuda")

control_image = load_image("https://huggingface.co/InstantX/FLUX.1-dev-Controlnet-Canny/resolve/main/canny.jpg")
prompt = "A girl in city, 25 years old, cool, futuristic"
generator = torch.Generator(device="cuda").manual_seed(42)
images_out = pipe(
    prompt, 
    control_image=control_image,
    controlnet_conditioning_scale=0.6,
    num_inference_steps=4, 
    num_images_per_prompt=num_images_per_prompt,
    guidance_scale=3.5,
    generator=generator,
).images
print(images_out)
for i, image in enumerate(images_out):
    image.save(f"yiyi_test_7_{branch}_num_images_per_prompt_{num_images_per_prompt}_test1_out_{i}.png")
"""

################################
# test single controlnet union #
################################

"""
# test2: single controlnet (union)
import torch
from diffusers.utils import load_image
from diffusers import FluxControlNetPipeline, FluxControlNetModel

base_model = 'black-forest-labs/FLUX.1-dev'
controlnet_model = 'InstantX/FLUX.1-dev-Controlnet-Union'

controlnet = FluxControlNetModel.from_pretrained(controlnet_model, torch_dtype=torch.bfloat16)
pipe = FluxControlNetPipeline.from_pretrained(base_model, controlnet=controlnet, torch_dtype=torch.bfloat16)
pipe.to("cuda")

control_image_canny = load_image("https://huggingface.co/InstantX/FLUX.1-dev-Controlnet-Union-alpha/resolve/main/images/canny.jpg").resize((512,512))
controlnet_conditioning_scale = 0.5
control_mode = 0

width, height = control_image_canny.size

prompt = 'A bohemian-style female travel blogger with sun-kissed skin and messy beach waves.'
generator = torch.Generator(device="cuda").manual_seed(42)

images_out = pipe(
    prompt, 
    control_image=control_image_canny,
    control_mode=control_mode,
    width=width,
    height=height,
    controlnet_conditioning_scale=controlnet_conditioning_scale,
    num_inference_steps=4, 
    num_images_per_prompt=num_images_per_prompt,
    generator=generator,
    guidance_scale=3.5,
).images
for i, image in enumerate(images_out):
    image.save(f"yiyi_test_7_{branch}_num_images_per_prompt_{num_images_per_prompt}_test2_out_{i}.png")

del pipe
flush()
"""

# ------------------------------------------------
# test3: multiple controlnet (regular controlnets)
# ------------------------------------------------

# note that we only have 1 regular controlnet now, so testing with 2 canny (this has no real use case but want to make sure it works regardless)
import torch
from diffusers.utils import load_image
from diffusers.pipelines.flux.pipeline_flux_controlnet import FluxControlNetPipeline
from diffusers.models.controlnet_flux import FluxControlNetModel, FluxMultiControlNetModel

base_model = 'black-forest-labs/FLUX.1-dev'
controlnet_canny = 'InstantX/FLUX.1-dev-Controlnet-Canny'
controlnet = FluxControlNetModel.from_pretrained(controlnet_canny, torch_dtype=torch.bfloat16)
multi_controlnet = FluxMultiControlNetModel([controlnet] * 2)
pipe = FluxControlNetPipeline.from_pretrained(base_model, controlnet=multi_controlnet, torch_dtype=torch.bfloat16)
pipe.to("cuda")

control_image = load_image("https://huggingface.co/InstantX/FLUX.1-dev-Controlnet-Canny/resolve/main/canny.jpg").resize((512,512))
prompt = "A girl in city, 25 years old, cool, futuristic"
generator = torch.Generator(device="cuda").manual_seed(42)
# currently does not work on main branch
try:    
    images_out = pipe(
        prompt,
        control_image=[control_image, control_image],
        controlnet_conditioning_scale=[0.6, 0.6],
        num_inference_steps=28,
        guidance_scale=3.5,
        num_images_per_prompt=num_images_per_prompt,
        generator=generator,
    ).images
    for i, image in enumerate(images_out):
        image.save(f"yiyi_test_7_{branch}_num_images_per_prompt_{num_images_per_prompt}_test3_out_{i}.png")
except Exception as e:
    print(e)

del pipe
flush()


"""
# test4: multi controlnet with union
import torch
from diffusers.utils import load_image
from diffusers import FluxControlNetPipeline, FluxControlNetModel, FluxMultiControlNetModel

base_model = 'black-forest-labs/FLUX.1-dev'
controlnet_model_union = 'InstantX/FLUX.1-dev-Controlnet-Union'

controlnet_union = FluxControlNetModel.from_pretrained(controlnet_model_union, torch_dtype=torch.bfloat16)
controlnet = FluxMultiControlNetModel([controlnet_union]) # we always recommend loading via FluxMultiControlNetModel

pipe = FluxControlNetPipeline.from_pretrained(base_model, controlnet=controlnet, torch_dtype=torch.bfloat16)
pipe.to("cuda")

prompt = 'A bohemian-style female travel blogger with sun-kissed skin and messy beach waves.'
control_image_depth = load_image("https://huggingface.co/InstantX/FLUX.1-dev-Controlnet-Union/resolve/main/images/depth.jpg").resize((512,512))
control_mode_depth = 2

control_image_canny = load_image("https://huggingface.co/InstantX/FLUX.1-dev-Controlnet-Union/resolve/main/images/canny.jpg")
control_mode_canny = 0

width, height = control_image_canny.size
generator = torch.Generator(device="cuda").manual_seed(42)
out_images = pipe(
    prompt,
    control_image=[control_image_depth, control_image_canny],
    control_mode=[control_mode_depth, control_mode_canny],
    width=width,
    height=height,
    controlnet_conditioning_scale=[0.2, 0.4],
    num_inference_steps=24, 
    num_images_per_prompt=num_images_per_prompt,
    guidance_scale=3.5,
    generator=generator,
).images
"""