import unittest
import torch
import numpy as np
import gc
from diffusers.utils import load_image
from diffusers.pipelines.flux.pipeline_flux_controlnet import FluxControlNetPipeline
from diffusers.models.controlnet_flux import FluxControlNetModel, FluxMultiControlNetModel
from diffusers import AutoencoderKL


def pil_to_numpy(image):
    """returns a tensor of shape (h, w, c)"""
    return (np.array(image).astype(np.float32)/255.)

def pil_to_torch(image):
    """returns a tensor of shape (c, h, w)"""
    return torch.from_numpy(pil_to_numpy(image)).float().swapaxes(2,1).swapaxes(0,1)

#@unittest.skip("done")
class TestFlux(unittest.TestCase):

    @classmethod
    def tearDownClass(cls):
        del cls.pipe
        torch.cuda.empty_cache()
        gc.collect()

    @classmethod
    def setUpClass(cls):
        control_image = load_image(
            "https://huggingface.co/InstantX/FLUX.1-dev-Controlnet-Union-alpha/resolve/main/images/canny.jpg"
        ).resize((512,512))
        base_model = 'black-forest-labs/FLUX.1-dev'
        controlnet = FluxControlNetModel.from_pretrained(
            "Shakker-Labs/FLUX.1-dev-ControlNet-Depth", torch_dtype=torch.bfloat16
        )
        vae = AutoencoderKL.from_pretrained(
            base_model, subfolder="vae", torch_dtype=torch.bfloat16
        )
        pipe = FluxControlNetPipeline.from_pretrained(
            base_model, vae=vae, controlnet=controlnet, torch_dtype=torch.bfloat16
        ).to("cuda")
        cls.pipe = pipe
        cls.image = control_image
        cls.image_np = pil_to_numpy(control_image)
        cls.image_t = pil_to_torch(control_image)

        print(f"image: {cls.image}")
        print(f"image np {cls.image_np.shape}:")
        print(f"image t {cls.image_t.shape}:")

    #@unittest.skip("easy")
    def test_torch_single_ctrl_1ipp(self):
        """ctrl image has bs=1, we use 1 image per prompt"""
        image_t = self.image_t.clone()
        image_t = image_t.unsqueeze(0)
        print(f"test_torch: image_t shape: {image_t.shape}")
        self.pipe(
            prompt="test",
            control_image=image_t, 
            control_mode=0, 
            num_images_per_prompt=1,
            num_inference_steps=2
        )
        return True

    #@unittest.skip("this throws an exception now but its what we wanted, with check_image")
    # Need to post in the PR a before and after and what the old exception was
    def test_torch_batched_ctrl_wrong_1ipp(self):
        """ctrl image has bs=2, we use 1 image per prompt, should fail"""
        image_t = self.image_t.clone()
        image_t = image_t.unsqueeze(0).repeat(2,1,1,1)
        print(f"test_torch: image_t shape: {image_t.shape}")
        with self.assertRaises(ValueError):
            self.pipe(
                prompt=["test"],
                control_image=image_t, 
                control_mode=0, 
                num_images_per_prompt=1,
                num_inference_steps=2
            )

    #@unittest.skip("")
    def test_torch_batched_ctrl_correct_1ipp(self):
        """ctrl image has bs=2, we use 1 image per prompt, should pass"""
        image_t = self.image_t.clone()
        image_t = image_t.unsqueeze(0).repeat(2,1,1,1)
        print(f"test_torch: image_t shape: {image_t.shape}")
        self.pipe(
            prompt=["test", "test"],
            control_image=image_t, 
            control_mode=0, 
            num_images_per_prompt=1,
            num_inference_steps=2
        )
        return True

    #@unittest.skip("")
    def test_torch_batched_ctrl_correct_2ipp(self):
        """ctrl image has bs=2, we use 2 image per prompt, should pass"""
        image_t = self.image_t.clone()
        image_t = image_t.unsqueeze(0).repeat(2,1,1,1)
        print(f"test_torch: image_t shape: {image_t.shape}")
        self.pipe(
            prompt=["test", "test"],
            control_image=image_t, 
            control_mode=0, 
            num_images_per_prompt=2,
            num_inference_steps=2
        )
        return True


#@unittest.skip("")
class TestMultiFlux(unittest.TestCase):

    @classmethod
    def tearDownClass(cls):
        del cls.pipe
        torch.cuda.empty_cache()
        gc.collect()

    @classmethod
    def setUpClass(cls):
        control_image = load_image(
            "https://huggingface.co/InstantX/FLUX.1-dev-Controlnet-Union-alpha/resolve/main/images/canny.jpg"
        ).resize((512,512))
        base_model = 'black-forest-labs/FLUX.1-dev'
        controlnet_canny = 'InstantX/FLUX.1-dev-Controlnet-Canny'
        controlnet = FluxControlNetModel.from_pretrained(
            controlnet_canny, torch_dtype=torch.bfloat16
        )
        multi_controlnet = FluxMultiControlNetModel([controlnet] * 2)
        vae = AutoencoderKL.from_pretrained(
            base_model, subfolder="vae", torch_dtype=torch.bfloat16
        )
        pipe = FluxControlNetPipeline.from_pretrained(
            base_model, vae=vae, controlnet=multi_controlnet, torch_dtype=torch.bfloat16
        ).to("cuda")
        cls.pipe = pipe
        cls.image = control_image
        cls.image_np = pil_to_numpy(control_image)
        cls.image_t = pil_to_torch(control_image)

        print(f"image: {cls.image}")
        print(f"image np {cls.image_np.shape}:")
        print(f"image t {cls.image_t.shape}:")

    #@unittest.skip("easy")
    def test_pil_1ipp(self):
        """we pass in a list of pil images, just 2 since we have 2 controls"""
        image_pil = self.image
        self.pipe(
            prompt="test",
            control_image=[image_pil, image_pil], 
            controlnet_conditioning_scale=[0.6, 0.6],
            control_mode=0,
            # this will default to 1024
            num_images_per_prompt=1,
            num_inference_steps=2
        )
        return True

    # -----------
    # torch tests
    # -----------

    #@unittest.skip("works")
    def test_torch_bs3_1ipp(self):
        """we pass in a bs=3 torch tensor"""        
        image_t = self.image_t.clone()
        image_t = image_t.unsqueeze(0).repeat(3,1,1,1)
        ns = 2
        images = self.pipe(
            prompt=["1","2","3"],           # len must match bs
            control_image=[image_t, image_t], 
            controlnet_conditioning_scale=[0.6, 0.6],
            control_mode=0,
            # no need for w/h since the pipeline just ignores prepare_image for torch dtype
            num_images_per_prompt=1,
            num_inference_steps=ns
        ) 
        assert len(images[0]) == len(image_t)

    #@unittest.skip("works")
    def test_torch_bs3_2ipp(self):
        """we pass in a bs=3 torch tensor but 2 images per prompt"""
        image_t = self.image_t.clone()
        image_t = image_t.unsqueeze(0).repeat(3,1,1,1)
        ns = 2
        images = self.pipe(
            prompt=["1","2","3"],           # len must match bs
            control_image=[image_t, image_t], 
            controlnet_conditioning_scale=[0.6, 0.6],
            # no need for w/h since the pipeline just ignores prepare_image for torch dtype
            control_mode=0,
            num_images_per_prompt=2,
            num_inference_steps=ns
        )
        assert len(images[0]) == len(image_t)*2

    # -----------
    # numpy tests
    # -----------

    #@unittest.skip("")
    def test_numpy_bs3_1ipp(self):
        """we pass in a bs=3 torch tensor"""        
        image_np = np.copy(self.image_np)[None]
        image_np = np.repeat(image_np, 3, axis=0)
        images = self.pipe(
            prompt=["1","2","3"],           # len must match bs
            control_image=[image_np, image_np], 
            controlnet_conditioning_scale=[0.6, 0.6],
            control_mode=0,
            width=512,
            height=512,
            num_images_per_prompt=1,
            num_inference_steps=2
        ) 
        assert len(images[0]) == len(image_np)

    #@unittest.skip("")
    def test_numpy_bs3_2ipp(self):
        """we pass in a bs=3 torch tensor but 2 images per prompt"""
        image_np = np.copy(self.image_np)[None]
        image_np = np.repeat(image_np, 3, axis=0)
        ns = 2
        images = self.pipe(
            prompt=["1","2","3"],           # len must match bs
            control_image=[image_np, image_np], 
            controlnet_conditioning_scale=[0.6, 0.6],
            control_mode=0,
            width=512,
            height=512,
            num_images_per_prompt=2,
            num_inference_steps=ns
        )
        assert len(images[0]) == len(image_np)*2