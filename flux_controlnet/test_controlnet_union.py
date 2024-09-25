import pytest
import torch
from diffusers.utils import load_image
from diffusers import FluxControlNetPipeline, FluxControlNetModel
from diffusers import AutoencoderKL
from functools import partial

from accelerate import infer_auto_device_map, init_empty_weights

@pytest.fixture(scope="session")
def setup_function():

    print("SETTING UP...")

    control_image = load_image(
        "https://huggingface.co/InstantX/FLUX.1-dev-Controlnet-Union-alpha/resolve/main/images/canny.jpg"
    ).resize((256,256))

    width, height = control_image.size

    base_model = 'black-forest-labs/FLUX.1-dev'
    controlnet_model = 'InstantX/FLUX.1-dev-Controlnet-Union'

    with init_empty_weights():

        controlnet = FluxControlNetModel.from_pretrained(
            controlnet_model, torch_dtype=torch.bfloat16
        )
        vae = AutoencoderKL.from_pretrained(base_model, subfolder="vae", torch_dtype=torch.bfloat16)
        pipe = FluxControlNetPipeline.from_pretrained(
            base_model, vae=vae, controlnet=controlnet, torch_dtype=torch.bfloat16
        )
        pipe.to("cuda")

    prompt = 'A bohemian-style female travel blogger with sun-kissed skin and messy beach waves.'

    generator = torch.Generator(device="cuda").manual_seed(0)

    print("Constructing pipe...")

    pipe_partial = partial(
        pipe,
        prompt=prompt,
        width=width,
        height=height,
        num_inference_steps=1, 
        guidance_scale=3.5,
        generator=generator
    )

    yield {"pipe": pipe_partial, "control_image": control_image}


def test_one_control_bs2_single_mode(setup_function):
    """
    Internally, the batch size of control_image will be 2, this is because we use
    num_images_per_prompt==2. This should pass.

    """
    
    obj = setup_function
    pipe = obj["pipe"]
    control_image = obj["control_image"]

    # We cannot do control_image=[control_image, control_image] yet due to some
    # batch size tracking bug not of my doing.
    pipe(control_image=control_image, control_mode=0, num_images_per_prompt=2)

def test_one_control_singleton(setup_function):
    """
    
    """
    
    obj = setup_function
    pipe = obj["pipe"]
    control_image = obj["control_image"]


    pipe(control_image=[control_image], control_mode=0, num_images_per_prompt=1)


#This currently fails: the internal batch size is 2 but this is because we actually
# pass in two control images with [control_image, control_image, and this fails.
def test_two_controls_bs1_single_mode(setup_function):
    """Even though we pass two control images, a single control mode should be 
    interpreted as applying to both of them. This should PASS.
    """
    
    obj = setup_function
    pipe = obj["pipe"]
    control_image = obj["control_image"]

    pipe(control_image=[control_image, control_image], control_mode=0, num_images_per_prompt=1)

# def test_one_image_none_control(setup_function):
#     """one control image, and we specify the control mode with None.

#     should FAIL
#     """
    
#     obj = setup_function
#     pipe = obj["pipe"]
#     control_image = obj["control_image"]

#     with pytest.raises(Exception):
#         # should raise an exception
#         pipe(control_image=control_image, control_mode=None)

# def test_one_image_two_control(setup_function):
#     """One control image, and we specify two control modes (makes no sense).

#     This should fail, since it doesn't make sense to give two controls for the same
#     control image.
#     """
    
#     obj = setup_function
#     pipe = obj["pipe"]
#     control_image = obj["control_image"]

#     with pytest.raises(Exception):
#         # should raise an exception
#         pipe(control_image=control_image, control_mode=[0,1])

if __name__ == '__main__':
    pytest.main()
