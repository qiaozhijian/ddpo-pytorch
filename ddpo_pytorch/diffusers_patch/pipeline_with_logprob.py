# Copied from https://github.com/huggingface/diffusers/blob/fc6acb6b97e93d58cb22b5fee52d884d77ce84d8/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion.py
# with the following modifications:
# - It uses the patched version of `ddim_step_with_logprob` from `ddim_with_logprob.py`. As such, it only supports the
#   `ddim` scheduler.
# - It returns all the intermediate latents of the denoising process as well as the log probs of each denoising step.

from typing import Any, Callable, Dict, List, Optional, Union

import torch

from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import (
    StableDiffusionPipeline,
    rescale_noise_cfg,
)
from .ddim_with_logprob import ddim_step_with_logprob


@torch.no_grad()
def pipeline_with_logprob(
    self: StableDiffusionPipeline,
    prompt: Union[str, List[str]] = None,
    height: Optional[int] = None,
    width: Optional[int] = None,
    num_inference_steps: int = 50,
    guidance_scale: float = 7.5,
    negative_prompt: Optional[Union[str, List[str]]] = None,
    num_images_per_prompt: Optional[int] = 1,
    eta: float = 0.0,
    generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
    latents: Optional[torch.FloatTensor] = None,
    prompt_embeds: Optional[torch.FloatTensor] = None,
    negative_prompt_embeds: Optional[torch.FloatTensor] = None,
    output_type: Optional[str] = "pil",
    return_dict: bool = True,
    callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
    callback_steps: int = 1,
    cross_attention_kwargs: Optional[Dict[str, Any]] = None,
    guidance_rescale: float = 0.0,
):
    """
    调用稳定扩散流水线进行图像生成，并返回每一步去噪过程的中间潜变量和对数概率。
    这是标准StableDiffusionPipeline的修改版本，使用了带有对数概率计算的DDIM步骤函数。

    参数:
        prompt (`str` 或 `List[str]`, 可选):
            用于引导图像生成的提示词或提示词列表。如果未定义，则必须提供`prompt_embeds`作为替代。

        height (`int`, 可选, 默认为self.unet.config.sample_size * self.vae_scale_factor):
            生成图像的高度（像素）。

        width (`int`, 可选, 默认为self.unet.config.sample_size * self.vae_scale_factor):
            生成图像的宽度（像素）。

        num_inference_steps (`int`, 可选, 默认为50):
            去噪步骤的数量。更多的去噪步骤通常会产生更高质量的图像，但推理速度会更慢。

        guidance_scale (`float`, 可选, 默认为7.5):
            分类器自由扩散引导的比例，定义为[无分类器扩散引导](https://arxiv.org/abs/2207.12598)中的`w`。
            设置`guidance_scale > 1`启用引导。较高的引导比例会鼓励生成与文本提示密切相关的图像，
            但可能会降低图像质量。

        negative_prompt (`str` 或 `List[str]`, 可选):
            用于引导图像生成"不要生成什么"的负面提示词。如果未定义，则必须提供`negative_prompt_embeds`作为替代。
            当不使用引导时忽略（即当`guidance_scale < 1`时）。

        num_images_per_prompt (`int`, 可选, 默认为1):
            每个提示词生成的图像数量。

        eta (`float`, 可选, 默认为0.0):
            对应于DDIM论文中的参数eta (η): https://arxiv.org/abs/2010.02502。
            仅适用于[`schedulers.DDIMScheduler`]，对其他调度器将被忽略。
            eta控制了采样过程中注入的噪声量，0.0表示完全确定性，1.0表示等同于DDPM采样器。

        generator (`torch.Generator` 或 `List[torch.Generator]`, 可选):
            一个或多个[torch generator](https://pytorch.org/docs/stable/generated/torch.Generator.html)，
            用于使生成过程确定性。

        latents (`torch.FloatTensor`, 可选):
            预先生成的噪声潜变量，从高斯分布中采样，作为图像生成的输入。
            可用于使用不同的提示词调整相同的生成过程。如果未提供，将使用提供的随机`generator`采样生成潜变量张量。

        prompt_embeds (`torch.FloatTensor`, 可选):
            预先生成的文本嵌入。可用于轻松调整文本输入，例如提示权重。
            如果未提供，将从`prompt`输入参数生成文本嵌入。

        negative_prompt_embeds (`torch.FloatTensor`, 可选):
            预先生成的负面文本嵌入。可用于轻松调整文本输入，例如提示权重。
            如果未提供，将从`negative_prompt`输入参数生成负面文本嵌入。

        output_type (`str`, 可选, 默认为"pil"):
            生成图像的输出格式。在[PIL](https://pillow.readthedocs.io/en/stable/)(`PIL.Image.Image`)
            和NumPy数组(`np.array`)之间选择。

        return_dict (`bool`, 可选, 默认为`True`):
            是否返回[`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`]而不是普通元组。

        callback (`Callable`, 可选):
            一个在推理过程中每隔`callback_steps`步调用一次的函数。
            函数将使用以下参数调用：`callback(step: int, timestep: int, latents: torch.FloatTensor)`。

        callback_steps (`int`, 可选, 默认为1):
            调用`callback`函数的频率。如果未指定，回调将在每一步调用。

        cross_attention_kwargs (`dict`, 可选):
            如果指定，将传递给`AttentionProcessor`的kwargs字典，如
            [diffusers.cross_attention](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/cross_attention.py)
            中`self.processor`下定义的。

        guidance_rescale (`float`, 可选, 默认为0.7):
            引导重新缩放因子，由[通用扩散噪声调度和采样步骤存在缺陷](https://arxiv.org/pdf/2305.08891.pdf)提出。
            `guidance_scale`在该论文方程16中定义为`φ`。引导重新缩放因子可以在使用零终端信噪比时修复过度曝光问题。

    返回:
        `tuple`:
        一个包含四个元素的元组:
        - **images** (`List[PIL.Image.Image]` 或 `np.ndarray`): 生成的图像
        - **has_nsfw_concept** (`List[bool]`): 每个图像是否包含NSFW内容的标志列表
        - **all_latents** (`List[torch.FloatTensor]`): 去噪过程中每一步的潜变量，包括初始噪声和所有中间状态
        - **all_log_probs** (`List[torch.FloatTensor]`): 每一步去噪过程的对数概率
    """
    # 0. 默认高度和宽度设置为unet的配置
    height = height or self.unet.config.sample_size * self.vae_scale_factor
    width = width or self.unet.config.sample_size * self.vae_scale_factor

    # 1. 检查输入参数，如果不正确则引发错误
    self.check_inputs(
        prompt,
        height,
        width,
        callback_steps,
        negative_prompt,
        prompt_embeds,
        negative_prompt_embeds,
    )

    # 2. 定义调用参数
    if prompt is not None and isinstance(prompt, str):
        batch_size = 1
    elif prompt is not None and isinstance(prompt, list):
        batch_size = len(prompt)
    else:
        batch_size = prompt_embeds.shape[0]

    device = self._execution_device
    # 这里的`guidance_scale`定义类似于Imagen论文方程(2)中的引导权重`w`
    # https://arxiv.org/pdf/2205.11487.pdf
    # `guidance_scale = 1`对应于不执行分类器自由引导
    do_classifier_free_guidance = guidance_scale > 1.0

    # 3. 编码输入提示词
    text_encoder_lora_scale = (
        cross_attention_kwargs.get("scale", None)
        if cross_attention_kwargs is not None
        else None
    )
    # 编码提示词为嵌入向量。这个函数的设计很灵活：
    # 1. 如果提供了原始文本提示(prompt)，它会将其编码为嵌入向量
    # 2. 如果已经提供了预计算的嵌入向量(prompt_embeds)，它会直接使用这些嵌入
    # 3. 如果启用了无分类器引导(do_classifier_free_guidance)，它会处理条件和无条件嵌入的连接
    # 4. 它还处理批处理大小、每个提示的图像数量以及可能的LoRA缩放
    # 这种设计使API更加灵活，允许用户选择最适合其使用场景的输入方式
    prompt_embeds = self._encode_prompt(
        prompt,
        device,
        num_images_per_prompt,
        do_classifier_free_guidance,
        negative_prompt,
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_prompt_embeds,
        lora_scale=text_encoder_lora_scale,
    )

    # 4. 准备时间步
    self.scheduler.set_timesteps(num_inference_steps, device=device)
    timesteps = self.scheduler.timesteps

    # 5. 准备潜变量
    num_channels_latents = self.unet.config.in_channels
    latents = self.prepare_latents(
        batch_size * num_images_per_prompt,
        num_channels_latents,
        height,
        width,
        prompt_embeds.dtype,
        device,
        generator,
        latents,
    )

    # 6. 准备额外的步骤参数
    # TODO: 这个逻辑理想情况下应该移出流水线
    extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

    # 7. 去噪循环
    num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
    all_latents = [latents]  # 保存所有中间潜变量，包括初始噪声
    all_log_probs = []  # 保存每一步的对数概率
    with self.progress_bar(total=num_inference_steps) as progress_bar:
        for i, t in enumerate(timesteps):
            # 如果使用分类器自由引导，需要扩展潜变量
            latent_model_input = (
                torch.cat([latents] * 2) if do_classifier_free_guidance else latents
            )
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

            # 预测噪声残差
            noise_pred = self.unet(
                latent_model_input,
                t,
                encoder_hidden_states=prompt_embeds,
                cross_attention_kwargs=cross_attention_kwargs,
                return_dict=False,
            )[0]

            # 执行引导
            if do_classifier_free_guidance:
                # 将预测分为无条件和条件部分
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                # 根据引导比例合并预测
                noise_pred = noise_pred_uncond + guidance_scale * (
                    noise_pred_text - noise_pred_uncond
                )

            # 执行引导重新缩放（如果启用）
            if do_classifier_free_guidance and guidance_rescale > 0.0:
                # 基于 https://arxiv.org/pdf/2305.08891.pdf 的3.4节
                noise_pred = rescale_noise_cfg(
                    noise_pred, noise_pred_text, guidance_rescale=guidance_rescale
                )

            # 计算前一个带噪样本 x_t -> x_t-1，并获取对数概率
            latents, log_prob = ddim_step_with_logprob(
                self.scheduler, noise_pred, t, latents, **extra_step_kwargs
            )

            # 记录潜变量和对数概率
            all_latents.append(latents)
            all_log_probs.append(log_prob)

            # 调用回调函数（如果提供）
            if i == len(timesteps) - 1 or (
                (i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0
            ):
                progress_bar.update()
                if callback is not None and i % callback_steps == 0:
                    callback(i, t, latents)

    # 8. 后处理生成的图像
    if not output_type == "latent":
        # 使用VAE解码潜变量
        image = self.vae.decode(
            latents / self.vae.config.scaling_factor, return_dict=False
        )[0]
        # 运行安全检查器
        image, has_nsfw_concept = self.run_safety_checker(
            image, device, prompt_embeds.dtype
        )
    else:
        # 如果输出类型为latent，直接返回潜变量
        image = latents
        has_nsfw_concept = None

    # 9. 图像后处理
    if has_nsfw_concept is None:
        do_denormalize = [True] * image.shape[0]
    else:
        do_denormalize = [not has_nsfw for has_nsfw in has_nsfw_concept]

    image = self.image_processor.postprocess(
        image, output_type=output_type, do_denormalize=do_denormalize
    )

    # 10. 将最后使用的模型卸载到CPU
    if hasattr(self, "final_offload_hook") and self.final_offload_hook is not None:
        self.final_offload_hook.offload()

    # 11. 返回结果：图像、NSFW标志、所有中间潜变量和对数概率
    return image, has_nsfw_concept, all_latents, all_log_probs
