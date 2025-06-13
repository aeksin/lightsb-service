import inspect
import logging
import os
import sys

from compel import Compel

SCRIPT_PATH = os.path.dirname(os.path.abspath(__file__))

sys.path.append(f"{SCRIPT_PATH}/LightSB/ALAE")
sys.path.append(f"{SCRIPT_PATH}/LightSB")

import lreq
import torchvision
from checkpointer import Checkpointer
from defaults import get_cfg_defaults
from diffusers import DPMSolverMultistepScheduler, StableDiffusionImg2ImgPipeline
from diffusers.utils import load_image
from dlutils.pytorch import count_parameters
from model import Model
from net import *
from PIL import Image
from src.light_sb import LightSB

lreq.use_implicit_lreq.set(True)

indices = [4]

labels = ["young"]


def sample(
    cfg, logger, filenames, result_path, delta, result_num, device, generation_mode
):
    def LightSB_prediction(filename, d_model, training_dataset):

        def decode_lsb(x):
            x = x[:, None, :].repeat(1, model.mapping_f.num_layers, 1)
            layer_count = 9
            decoded = []
            for i in range(x.shape[0]):
                r = model.decoder(x[i][None, ...], layer_count - 1, 1, noise=True)
                decoded.append(r)
            return torch.cat(decoded)

        DIM = 512
        EPSILON = 0.1
        D_LR = 1e-3
        N_POTENTIALS = 10
        SAMPLING_BATCH_SIZE = 128

        IS_DIAGONAL = True
        D = LightSB(
            dim=DIM,
            n_potentials=N_POTENTIALS,
            epsilon=EPSILON,
            sampling_batch_size=SAMPLING_BATCH_SIZE,
            S_diagonal_init=0.1,
            is_diagonal=IS_DIAGONAL,
        ).cpu()
        D.load_state_dict(
            torch.load(
                f"{SCRIPT_PATH}/lightsb_checkpoints/D_{training_dataset}_{d_model}.pt",
                weights_only=True,
            )
        )
        D_opt = torch.optim.Adam(D.parameters(), lr=D_LR)
        D_opt.load_state_dict(
            torch.load(
                f"{SCRIPT_PATH}/lightsb_checkpoints/D_opt_{training_dataset}_{d_model}.pt",
                weights_only=True,
            )
        )
        img = np.asarray(Image.open(filename))

        if img.shape[2] == 4:
            img = img[:, :, :3]
        im = img.transpose((2, 0, 1))
        x = (
            torch.tensor(
                np.asarray(im, dtype=np.float32), device="cpu", requires_grad=True
            ).to(device)
            / 127.5
            - 1.0
        )
        if x.shape[0] == 4:
            x = x[:3]

        needed_resolution = model.decoder.layer_to_resolution[-1]
        while x.shape[2] > needed_resolution:
            x = F.avg_pool2d(x, 2, 2)
        if x.shape[2] != needed_resolution:
            x = F.adaptive_avg_pool2d(x, (needed_resolution, needed_resolution))

        latents = torch.stack(
            [model.encode(x[None, ...].to(device), 8, 1)[0].squeeze()]
        )
        D = D.to(device)
        with torch.no_grad():
            mapped = D(latents.to(device))
            decoded_img = decode_lsb(mapped)
            decoded_img = (
                ((decoded_img * 0.5 + 0.5) * 255)
                .type(torch.long)
                .clamp(0, 255)
                .cpu()
                .type(torch.uint8)
                .permute(0, 2, 3, 1)
                .squeeze(0)
            )
        return decoded_img

    model = Model(
        startf=cfg.MODEL.START_CHANNEL_COUNT,
        layer_count=cfg.MODEL.LAYER_COUNT,
        maxf=cfg.MODEL.MAX_CHANNEL_COUNT,
        latent_size=cfg.MODEL.LATENT_SPACE_SIZE,
        truncation_psi=cfg.MODEL.TRUNCATIOM_PSI,
        truncation_cutoff=cfg.MODEL.TRUNCATIOM_CUTOFF,
        mapping_layers=cfg.MODEL.MAPPING_LAYERS,
        channels=cfg.MODEL.CHANNELS,
        generator=cfg.MODEL.GENERATOR,
        encoder=cfg.MODEL.ENCODER,
    )
    model.to(device)
    model.eval()
    model.requires_grad_(False)

    decoder = model.decoder
    encoder = model.encoder
    mapping_tl = model.mapping_d
    mapping_fl = model.mapping_f
    dlatent_avg = model.dlatent_avg

    logger.info("Trainable parameters generator:")
    count_parameters(decoder)

    logger.info("Trainable parameters discriminator:")
    count_parameters(encoder)

    arguments = dict()
    arguments["iteration"] = 0

    model_dict = {
        "discriminator_s": encoder,
        "generator_s": decoder,
        "mapping_tl_s": mapping_tl,
        "mapping_fl_s": mapping_fl,
        "dlatent_avg": dlatent_avg,
    }

    checkpointer = Checkpointer(cfg, model_dict, {}, logger=logger, save=False)

    checkpointer.load()

    model.eval()

    layer_count = cfg.MODEL.LAYER_COUNT

    def encode(x):
        Z, _ = model.encode(x, layer_count - 1, 1)
        Z = Z.repeat(1, model.mapping_f.num_layers, 1)
        return Z

    def decode(x):
        layer_idx = torch.arange(2 * layer_count)[np.newaxis, :, np.newaxis]
        ones = torch.ones(layer_idx.shape, dtype=torch.float32)
        coefs = torch.where(layer_idx < model.truncation_cutoff, ones, ones)
        x = torch.lerp(model.dlatent_avg.buff.data, x, coefs)
        return model.decoder(x, layer_count - 1, 1, noise=True)

    attribute_values = [0.0 for i in indices]
    if result_path == "":
        print("Error: result_path not set")

    W = [
        torch.tensor(
            np.load(
                f"{SCRIPT_PATH}/LightSB/ALAE/principal_directions/direction_%d.npy" % i
            ),
            dtype=torch.float32,
        )
        for i in indices
    ]

    rnd = np.random.RandomState(5)

    def loadImage(filename):
        img = np.asarray(Image.open(filename))

        if img.shape[2] == 4:
            img = img[:, :, :3]
        im = img.transpose((2, 0, 1))
        x = (
            torch.tensor(
                np.asarray(im, dtype=np.float32), device="cpu", requires_grad=True
            ).to(device)
            / 127.5
            - 1.0
        )
        if x.shape[0] == 4:
            x = x[:3]

        needed_resolution = model.decoder.layer_to_resolution[-1]
        while x.shape[2] > needed_resolution:
            x = F.avg_pool2d(x, 2, 2)
        if x.shape[2] != needed_resolution:
            x = F.adaptive_avg_pool2d(x, (needed_resolution, needed_resolution))

        img_src = (
            ((x * 0.5 + 0.5) * 255)
            .type(torch.long)
            .clamp(0, 255)
            .cpu()
            .type(torch.uint8)
            .transpose(0, 2)
            .transpose(0, 1)
            .numpy()
        )

        latents_original = encode(x[None, ...].to(device))
        latents = latents_original[0, 0].clone()
        latents -= model.dlatent_avg.buff.data[0]
        for v, w in zip(attribute_values, W):
            v = (latents * w).sum()

        for v, w in zip(attribute_values, W):
            latents = latents - v * w

        return latents, latents_original, img_src

    def update_image(w, latents_original):
        with torch.no_grad():
            w = w + model.dlatent_avg.buff.data[0]
            w = w[None, None, ...].repeat(1, model.mapping_f.num_layers, 1)

            layer_idx = torch.arange(model.mapping_f.num_layers)[
                np.newaxis, :, np.newaxis
            ]
            cur_layers = (7 + 1) * 2
            mixing_cutoff = cur_layers
            styles = torch.where(layer_idx < mixing_cutoff, w, latents_original)

            x_rec = decode(styles)
            resultsample = ((x_rec * 0.5 + 0.5) * 255).type(torch.long).clamp(0, 255)
            resultsample = resultsample.cpu()[0, :, :, :]
            return resultsample.type(torch.uint8).transpose(0, 2).transpose(0, 1)

    result = []
    if not os.path.exists(result_path):
        os.mkdir(result_path)
    for filename in filenames:

        if generation_mode == "ALAE":
            latents, latents_original, img_src = loadImage(filename)

            im_size = 2 ** (cfg.MODEL.LAYER_COUNT + 1)
            im = update_image(latents, latents_original)

            for i in range(result_num):
                for x in range(len(attribute_values)):
                    attribute_values[x] += delta

                new_latents = latents + sum(
                    [v * w for v, w in zip(attribute_values, W)]
                )

                im = update_image(new_latents, latents_original)
                new_im = torchvision.transforms.functional.to_pil_image(
                    im.permute(2, 0, 1)
                )
                new_im.save(
                    f"{result_path}/{'.'.join(filename.split('/')[-1].split('.')[:-1])}_var{i}.png"
                )
                result.append(
                    f"{result_path}/{'.'.join(filename.split('/')[-1].split('.')[:-1])}_var{i}.png"
                )

        elif "LightSB" in generation_mode:
            d_models = ["40", "50", "60", "70"]
            training_dataset = generation_mode.split("_")[-1]
            for d_model in d_models:
                im = LightSB_prediction(filename, d_model, training_dataset)

                new_im = torchvision.transforms.functional.to_pil_image(
                    im.permute(2, 0, 1)
                )
                new_im.save(
                    f"{result_path}/{'.'.join(filename.split('/')[-1].split('.')[:-1])}_var{d_model}.png"
                )
                result.append(
                    f"{result_path}/{'.'.join(filename.split('/')[-1].split('.')[:-1])}_var{d_model}.png"
                )
    return result


def generate_ALAE(
    filenames,
    output="",
    generation_model="ALAE",
    result_count=5,
    delta=-4,
):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device == torch.device("cuda"):
        torch.set_default_tensor_type("torch.cuda.FloatTensor")
    cfg = get_cfg_defaults()
    config_file = f"{SCRIPT_PATH}/LightSB/ALAE/configs/ffhq.yaml"
    if len(os.path.splitext(config_file)[1]) == 0:
        config_file += ".yaml"
    if not os.path.exists(config_file) and os.path.exists(
        os.path.join(f"{SCRIPT_PATH}/LightSB/ALAE/configs", config_file)
    ):
        config_file = os.path.join(f"{SCRIPT_PATH}LightSB/ALAE/configs", config_file)
    cfg.merge_from_file(config_file)
    cfg.freeze()

    logger = logging.getLogger("logger")
    logger.setLevel(logging.DEBUG)

    output_dir = cfg.OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)

    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s: %(message)s")
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    logger.info("Loaded configuration file {}".format(config_file))
    with open(config_file, "r") as cf:
        config_str = "\n" + cf.read()
        logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    args_to_pass = dict(
        cfg=cfg,
        logger=logger,
        filenames=filenames,
        result_path=output,
        delta=delta,
        result_num=result_count,
        device=device,
        generation_mode=generation_model,
    )
    signature = inspect.signature(sample)
    matching_args = {}
    for key in args_to_pass.keys():
        if key in signature.parameters.keys():
            matching_args[key] = args_to_pass[key]
    return sample(**matching_args)


def generate_sd(
    filenames,
    result_path,
    strength: float = 0.5,
    guidance_scale: float = 7.5,
    num_inference_steps: int = 35,
):

    pipe = StableDiffusionImg2ImgPipeline.from_single_file(
        f"{SCRIPT_PATH}/sb_checkpoints/stddiff.ckpt",
        safety_checker=None,
        torch_dtype=torch.float16,
    )
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    ages = [50, 60, 70]
    age_prompts = [
        f"realistic aging, age spots, realistic, high details, wrinkles{'--' if age == 50 else '-' if age == 60 else ''}, age of {age}, (age details){'-' if age == 50 else ''}"
        + ("stddiff5060" if age in [50, 60] else "")
        + ("stddiff7080" if age == 70 else "")
        for age in ages
    ]
    negative_prompt = "deformed, blurry, low quality, unrealistic, plastic, young"
    try:

        with torch.no_grad():
            compel = Compel(
                tokenizer=pipe.tokenizer,
                text_encoder=pipe.text_encoder,
                device="cpu",
            )
            prompt_embeds = [compel(prompt) for prompt in age_prompts]
            negative_prompt_embeds = compel(negative_prompt)

        pipe.enable_vae_tiling()
        pipe.enable_vae_slicing()
        pipe.enable_attention_slicing()
        pipe.enable_sequential_cpu_offload()
        results = []

        for filename in filenames:
            for prompt_embed, age in zip(prompt_embeds, ages):

                init_image = load_image(filename).convert("RGB")

                generator = torch.Generator("cpu")

                result = pipe(
                    prompt_embeds=prompt_embed,
                    negative_prompt_embeds=negative_prompt_embeds,
                    image=init_image,
                    strength=strength,
                    guidance_scale=guidance_scale,
                    num_inference_steps=num_inference_steps,
                    generator=generator,
                    output_type="pil",
                ).images[0]
                path_to_generated_image = f"{result_path}/{'.'.join(filename.split('/')[-1].split('.')[:-1])}_{age}.png"
                result.save(path_to_generated_image)
                results.append(path_to_generated_image)
    except Exception as e:
        print(e)
    return results
