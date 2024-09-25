import torch
from diffusers.utils import load_image
from diffusers import FluxControlNetPipeline, FluxMultiControlNetModel, FluxControlNetModel
from diffusers import AutoencoderKL

from tabulate import tabulate
import numpy as np

def summarise_pipeline(pipeline):
    # Record statistics for each pipeline component which is a
    # Torch module and pretty-print them.
    pipeline_stats = []
    for comp_name, comp_val in pipeline.components.items():
        if hasattr(comp_val, 'parameters'):
            pipeline_stats.append({
                "name": comp_name,
                "dtypes": get_dtypes_from_parameters(comp_val),
                "devices": get_devices_from_parameters(comp_val),
                "# params": count_parameters(comp_val)[1],
                "# params L": count_parameters(comp_val)[0]
            })
    print(tabulate(pipeline_stats, headers="keys"))

def count_parameters_state_dict(state_dict):
    """Count number of both learnable and total parameters for a module"""
    num_params = sum([np.prod(p.size()) for p in state_dict.values()])
    return num_params

def count_parameters(model):
    """Count number of both learnable and total parameters for a module"""
    learnable_parameters = filter(lambda p: p.requires_grad, model.parameters())
    num_learned_params = sum([np.prod(p.size()) for p in learnable_parameters])
    num_params = sum([np.prod(p.size()) for p in model.parameters()])
    return num_learned_params, num_params

def get_dtypes_from_parameters(module):
    """Tell me all the possible data types used by this module's parameters."""
    types = []
    for param in module.parameters():
        types.append(param.dtype)
    return set(types)

def get_devices_from_parameters(module):
    """Tell me all the possible devices that this module sits on."""
    types = []
    for param in module.parameters():
        types.append(param.device)
    return set(types)

if __name__ == '__main__':

    control_image = load_image(
        "https://huggingface.co/InstantX/FLUX.1-dev-Controlnet-Union-alpha/resolve/main/images/canny.jpg"
    ).resize((512,512))
    controlnet_conditioning_scale = 0.5
    control_mode = 0

    width, height = control_image.size

    base_model = 'black-forest-labs/FLUX.1-dev'
    controlnet_model = 'InstantX/FLUX.1-dev-Controlnet-Union'

    controlnet = FluxControlNetModel.from_pretrained(
        controlnet_model, torch_dtype=torch.bfloat16
    )
    multinet = FluxMultiControlNetModel([controlnet, controlnet])
    vae = AutoencoderKL.from_pretrained(base_model, subfolder="vae", torch_dtype=torch.bfloat16)
    pipe = FluxControlNetPipeline.from_pretrained(
        base_model, vae=vae, controlnet=multinet, torch_dtype=torch.bfloat16
    )
    print("to cuda...")
    pipe.to("cuda")
        
    summarise_pipeline(pipe)

    prompt = 'A bohemian-style female travel blogger with sun-kissed skin and messy beach waves.'

    pipe.enable_model_cpu_offload()

    generator = torch.Generator(device="cuda").manual_seed(0)

    out = pipe(
        prompt=prompt,
        control_image=[control_image, control_image],
        control_mode=[0,None],
        width=width,
        height=height,
        num_inference_steps=1, 
        guidance_scale=3.5,
        controlnet_conditioning_scale=[1., 1.],
        generator=generator
    )

    # print("running pipe 0...")

    ####################
    # multi controlnet #
    ####################


    #one_image_none_control(pipe)
    #one_image_single_control(pipe)
    #two_image_single_control(pipe)
    #one_image_two_control(pipe)
    #one_image_single_control(pipe)
