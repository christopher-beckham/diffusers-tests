from diffusers import ControlNetModel, StableDiffusionXLControlNetPipeline, AutoencoderKL
from diffusers.utils import load_image
from PIL import Image
import torch
import numpy as np
import cv2
import sys

try:
    #from better_exceptions import better_exceptions
    import better_exceptions
    better_exceptions.hook()
except ImportError as e:
    print(e)
    print("Error: was unable to activate better_exceptions...")

def pil_to_numpy(image):
    """to (c,h,w)"""
    return (np.array(image).astype(np.float32)/255.)

def pil_to_torch(image):
    return torch.from_numpy(pil_to_numpy(image)).float()

prompt = "aerial view, a futuristic research complex in a bright foggy jungle, hard lighting"
#negative_prompt = 'low quality, bad quality, sketches'

image = load_image("https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/sd_controlnet/hf-logo.png")

controlnet_conditioning_scale = 0.5  # recommended for good generalization

controlnet = ControlNetModel.from_pretrained(
    "diffusers/controlnet-canny-sdxl-1.0",
    torch_dtype=torch.float16
).to("cuda")
vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)
pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    controlnet=controlnet,
    vae=vae,
    torch_dtype=torch.float16,
).to("cuda")
#pipe.enable_model_cpu_offload()

image = np.array(image)
image = cv2.Canny(image, 100, 200)
image = image[:, :, None]
image = np.concatenate([image, image, image], axis=2)
image = Image.fromarray(image)

#breakpoint()

#prompt: Union[str, List[str]] = None,
#prompt_2: Optional[Union[str, List[str]]] = None,
#image: PipelineImageInput = None,

"""
PipelineImageInput = Union[
    PIL.Image.Image,               <-- tested
    np.ndarray,
    torch.Tensor,
    List[PIL.Image.Image],
    List[np.ndarray],
    List[torch.Tensor],
]
"""

# Testing as PIL images

# images = pipe(
#     prompt, 
#     image=image, 
#     controlnet_conditioning_scale=controlnet_conditioning_scale,
#     num_inference_steps=2
# ).images
# print(f"num images: {len(images)}")


# works
# images = pipe(
#     [prompt, prompt],
#     image=image,
#     controlnet_conditioning_scale=controlnet_conditioning_scale,
#     num_inference_steps=2
# ).images
# print(f"num images: {len(images)}")

# works
# images = pipe(
#     [prompt, prompt],
#     image=[image],
#     controlnet_conditioning_scale=controlnet_conditioning_scale,
#     num_inference_steps=2
# ).images
# print(f"num images: {len(images)}")

if __name__ == '__main__':

    """
    The basic algorithm seems to be the following:
    - We check to make sure the length of control image is the same as length
      of prompt. The only exception allowed is if the control image is a singleton.
      (singleton = either inside a list of len 1, or just on its own.) Otherwise,
      the code will error out.
      - (This is not equivalent backwards: if the prompt is a singleton but the 
         control image is a list then the code will fail.)
    - In `image = prepare_image(image)`, the "batch size" passed in is 
      `len(prompts) * num_images_per_prompt`. If image has batch size 1, then we just
      repeat it on the batch axis by `len(prompts) * num_images_per_prompt`. Otherwise,
      since it already has a batch axis we repeat by `num_images_per_prompt`.

    ```
        print("prepare image got dtype: {}".format(type(image)))
        image = self.control_image_processor.preprocess(image, height=height, width=width).to(dtype=torch.float32)
        image_batch_size = image.shape[0]

        if image_batch_size == 1:
            repeat_by = batch_size
        else:
            # image batch size is the same as prompt batch size
            repeat_by = num_images_per_prompt
    ```
    """

    image_n = pil_to_numpy(image)

    print(image_n.shape)

    # =======================#
    # testing out np.ndarray #
    # =======================#

    # This needs to be in the format (b,h,w,c), NOT (b,c,h,w)
    image_n = pil_to_numpy(image).swapaxes(1,2).swapaxes(1,0) #[None]
    #try:
    #print(f"numpy image shape: {image_n.shape}")
    images = pipe(
        prompt, 
        image=image_n, 
        controlnet_conditioning_scale=controlnet_conditioning_scale,
        num_inference_steps=2
    ).images
    print(f"num images: {len(images)}")
    #except Exception as e:
    #    print(e)
    print("---------------")


    sys.exit(0)

    #image_t = pil_to_torch(image)

    # images = pipe(
    #     prompt, 
    #     image=image_t, 
    #     controlnet_conditioning_scale=controlnet_conditioning_scale,
    #     num_inference_steps=2
    # ).images
    # print(f"num images: {len(images)}")

    # #doesn't work
    # #prompt takes precedence, so bs=4 on control image is not allowed
    # images = pipe(
    #    prompt, 
    #    image=image_t.repeat(4,1,1,1),
    #    controlnet_conditioning_scale=controlnet_conditioning_scale,
    #    num_inference_steps=2
    # ).images
    # print(f"num images: {len(images)}")

    sys.exit(0)

    # This fails because the number of control images must match the
    # number of prompts.
    try:
        print("test 1")
        images = pipe(
            prompt, 
            image=[image, image],
            controlnet_conditioning_scale=controlnet_conditioning_scale,
            num_inference_steps=2
        ).images
    except Exception as e:
        print(e)
    print("---------------")

    # This fails because the number of control images must match the
    # number of prompts.
    try:
        print("test 2")
        images = pipe(
            [prompt], 
            image=[image, image],
            controlnet_conditioning_scale=controlnet_conditioning_scale,
            num_inference_steps=2
        ).images
    except Exception as e:
        print(e)
    print("---------------")

    # This will pass due to broadcasting.
    try:
        print("test 3")
        images = pipe(
            [prompt, prompt], 
            image=image,
            controlnet_conditioning_scale=controlnet_conditioning_scale,
            num_inference_steps=2
        ).images
    except Exception as e:
        print(e)
    print("---------------")

    # This will pass due to broadcasting.
    try:
        print("test 4")
        images = pipe(
            [prompt, prompt], 
            image=[image],
            controlnet_conditioning_scale=controlnet_conditioning_scale,
            num_inference_steps=2
        ).images
    except Exception as e:
        print(e)
    print("---------------")



    # ----------------

    # Testing as Torch images

    #image_t = pil_to_torch(image)

    # images = pipe(
    #     prompt, 
    #     image=image_t, 
    #     controlnet_conditioning_scale=controlnet_conditioning_scale,
    #     num_inference_steps=2
    # ).images
    # print(f"num images: {len(images)}")

    # doesn't work
    # prompt takes precedence, so bs=4 on control image is not allowed
    #images = pipe(
    #    prompt, 
    #    image=image_t.repeat(4,1,1,1),
    #    controlnet_conditioning_scale=controlnet_conditioning_scale,
    #    num_inference_steps=2
    #).images
    #print(f"num images: {len(images)}")

    # the pipeline needs to explicitly check that if image is a torch.tensor then
    # it should be expected to be 4d

    # # should work cos prompt takes precedence
    # images = pipe(
    #     [prompt, prompt, prompt, prompt],
    #     image=image_t,
    #     controlnet_conditioning_scale=controlnet_conditioning_scale,
    #     num_inference_steps=2
    # ).images
    # print(f"num images: {len(images)}")

    # testing as np.array images

    # @staticmethod
    # def numpy_to_pt(images: np.ndarray) -> torch.Tensor:
    #     """
    #     Convert a NumPy image to a PyTorch tensor.
    #     """
    #     if images.ndim == 3:
    #         images = images[..., None]

    #     images = torch.from_numpy(images.transpose(0, 3, 1, 2))
    #     return images


    sys.exit(0)

    # =======================#
    # testing out np.ndarray #
    # =======================#

    # This needs to be in the format (b,h,w,c), NOT (b,c,h,w)
    image_n = pil_to_numpy(image)[None]
    try:
        print(f"numpy image shape: {image_n.shape}")
        images = pipe(
            prompt, 
            image=image_n, 
            controlnet_conditioning_scale=controlnet_conditioning_scale,
            num_inference_steps=2
        ).images
        print(f"num images: {len(images)}")
    except Exception as e:
        print(e)
    print("---------------")




    # # (3,1024,1024)
    # # ValueError: If image batch size is not 1, image batch size must be same as prompt batch size. image batch size: 3, prompt batch size: 1 
    # # Ok, so this must be a 4d tensor
    # image_n = pil_to_numpy(image)
    # print(f"numpy image shape: {image_n.shape}")
    # images = pipe(
    #     prompt, 
    #     image=image_n, 
    #     controlnet_conditioning_scale=controlnet_conditioning_scale,
    #     num_inference_steps=2
    # ).images
    # print(f"num images: {len(images)}")
    #images[0].save(f"hug_lab.png")