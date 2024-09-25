import pytest
import torch
from diffusers.utils import load_image
from diffusers import FluxControlNetPipeline, FluxControlNetModel, FluxMultiControlNetModel
from diffusers import AutoencoderKL
from functools import partial
import numpy as np

def pil_to_numpy(image):
    """to (c,h,w)"""
    return (np.array(image).astype(np.float32)/255.).swapaxes(1,2).swapaxes(0,1)

def pil_to_torch(image):
    return torch.from_numpy(pil_to_numpy(image)).float()

@pytest.fixture(scope="session")
def setup_function():

    print("SETTING UP...")
    
    # NOTE can we meta device this to make it faster???

    control_image = load_image(
        "https://huggingface.co/InstantX/FLUX.1-dev-Controlnet-Union-alpha/resolve/main/images/canny.jpg"
    ).resize((512,512))

    width, height = control_image.size

    base_model = 'black-forest-labs/FLUX.1-dev'
    
    controlnet_union = FluxControlNetModel.from_pretrained(
        'InstantX/FLUX.1-dev-Controlnet-Union', torch_dtype=torch.bfloat16
    )
    controlnet_depth = FluxControlNetModel.from_pretrained(
        "Shakker-Labs/FLUX.1-dev-ControlNet-Depth", torch_dtype=torch.bfloat16
    )
    
    multinet = FluxMultiControlNetModel([controlnet_union, controlnet_depth])
    vae = AutoencoderKL.from_pretrained(base_model, subfolder="vae", torch_dtype=torch.bfloat16)
    pipe = FluxControlNetPipeline.from_pretrained(
        base_model, vae=vae, controlnet=multinet, torch_dtype=torch.bfloat16
    )
    #pipe.to("cuda")
    pipe.enable_model_cpu_offload()

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

@pytest.mark.skip()
def test_torch_tensor_multinet_control_modes(setup_function):
    """
    This should pass.
    """
    obj = setup_function
    pipe = obj["pipe"]
    control_image = obj["control_image"]

    # a batch size of 4
    control_image = pil_to_torch(control_image).unsqueeze(0)

    pipe(
        # the controlnets here are [union, union]
        control_image=[control_image], 
        control_mode=[0], 
        controlnet_conditioning_scale=[1.]
    )


@pytest.mark.skip()
def test_multinet_control_modes(setup_function):
    """
    This should pass. We pass in two control images but we also
    pass in two modes, so no problem.
    """
    obj = setup_function
    pipe = obj["pipe"]
    control_image = obj["control_image"]

    pipe(
        # the controlnets here are [union, union]
        control_image=[control_image, control_image], 
        control_mode=[0, None], 
        controlnet_conditioning_scale=[1., 1.]
    )

def test_multinet_control_modes_torch(setup_function):
    """
    This should pass. We pass in two control images but we also
    pass in two modes, so no problem.
    """
    obj = setup_function
    pipe = obj["pipe"]
    control_image = obj["control_image"]
    control_image = pil_to_torch(control_image)
    # (4,3,512,512)
    control_image = control_image.unsqueeze(0).repeat(4,1,1,1)

    pipe(
        # the controlnets here are [union, union]
        control_image=[control_image, control_image], 
        control_mode=[0, None], 
        controlnet_conditioning_scale=[1., 1.]
    )


@pytest.mark.skip()
def test_multinet_control_modes_3(setup_function):
    """
    This should fail. We pass in an integer for control mode.
    """
    obj = setup_function
    pipe = obj["pipe"]
    control_image = obj["control_image"]

    with pytest.raises(Exception):
        pipe(
            # the controlnets here are [union, union]
            control_image=[control_image], 
            control_mode=0, 
            controlnet_conditioning_scale=[1.]
        )
@pytest.mark.skip()
def test_multinet_control_modes_4(setup_function):
    """
    This should fail. We pass in more control modes than control images.
    """
    obj = setup_function
    pipe = obj["pipe"]
    control_image = obj["control_image"]

    with pytest.raises(Exception):
        pipe(
            # the controlnets here are [union, union]
            control_image=[control_image], 
            control_mode=[0,0], 
            controlnet_conditioning_scale=[1.]
        )

@pytest.mark.skip()
def test_multinet_control_modes_2(setup_function):
    """
    This should pass. We pass in one control image and one
    control mode.
    """
    obj = setup_function
    pipe = obj["pipe"]
    control_image = obj["control_image"]

    pipe(
        # the controlnets here are [union, union]
        control_image=[control_image], 
        control_mode=[0], 
        controlnet_conditioning_scale=[1.]
    )



# def test_multinet_control_modes(setup_function):
#     """The old branch would have converted [None,None] to [-1, -1] and then it would fail
#     due to a device-side assert error. On my code this should pass.
#     """
    
#     obj = setup_function
#     pipe = obj["pipe"]
#     control_image = obj["control_image"]

#     # for the list [1,2,None,None]
#     # [1,2] implies it's for a union controlnet
#     # [None,None] implies it's for a regular controlnet,
#     # these will be converted to -1 but it should not trip
#     # the controlnet because self.union check

#     # I do not want any exceptions to occur here.
#     #with pytest.raises(Exception):
#     pipe(
#         # the controlnets here are [union, union]
#         control_image=[control_image, control_image], 
#         control_mode=[0, None], 
#         controlnet_conditioning_scale=[1., 1.]
#     )

# def test_multi_control_modes_wrong_ordering(setup_function):
#     """Test a mixture of control modes.

#     This *should fail* cos we got the control modes the wrong way around.
#     """
    
#     obj = setup_function
#     pipe = obj["pipe"]
#     control_image = obj["control_image"]

#     with pytest.raises(Exception):
#         pipe(
#             control_image=[control_image, control_image], 
#             control_mode=[None, 0], 
#             controlnet_conditioning_scale=[1., 1.]
#         )


# def test_baseline(setup_function):
#     """Test a mixture of control modes."""
    
#     obj = setup_function
#     pipe = obj["pipe"]
#     control_image = obj["control_image"]

#     # The current code will convert Nones to -1 but this doesn't make sense and will
#     # result in an embedding layer being indexed into with a -ve value. So it should
#     # fail.
#     pipe(
#         control_image=[control_image, control_image], 
#         control_mode=[0,1], 
#         controlnet_conditioning_scale=[1., 1.]
#     )

#     assert True


if __name__ == '__main__':
    pytest.main()
