from collections import defaultdict
import contextlib
import os
import datetime
from concurrent import futures
import time
from absl import app, flags
from ml_collections import config_flags
from accelerate import Accelerator
from accelerate.utils import set_seed, ProjectConfiguration
from accelerate.logging import get_logger
from diffusers import StableDiffusionPipeline, DDIMScheduler, UNet2DConditionModel
from diffusers.loaders import AttnProcsLayers
from diffusers.models.attention_processor import LoRAAttnProcessor
import numpy as np
import ddpo_pytorch.prompts
import ddpo_pytorch.rewards
from ddpo_pytorch.stat_tracking import PerPromptStatTracker
from ddpo_pytorch.diffusers_patch.pipeline_with_logprob import pipeline_with_logprob
from ddpo_pytorch.diffusers_patch.ddim_with_logprob import ddim_step_with_logprob
import torch
import wandb
from functools import partial
import tqdm
import tempfile
from PIL import Image

tqdm = partial(tqdm.tqdm, dynamic_ncols=True)


FLAGS = flags.FLAGS
config_flags.DEFINE_config_file("config", "config/base.py", "Training configuration.")

logger = get_logger(__name__)


class DDPOTrainer:
    """去噪扩散策略优化(DDPO)训练器"""

    def __init__(self, config):
        """初始化训练器

        Args:
            config: 训练配置
        """
        self.config = config
        self.setup_accelerator()
        self.setup_model()
        self.setup_optimizer()
        self.setup_prompt_and_reward()
        self.global_step = 0

        # 设置一个执行器来异步执行回调
        self.executor = futures.ThreadPoolExecutor(max_workers=2)

        # 记录训练配置信息
        self.log_training_config()

    def setup_accelerator(self):
        """设置Accelerator和相关配置

        这个函数配置了Hugging Face Accelerate库的加速器，用于实现分布式训练、混合精度训练等高级功能。
        主要完成的任务包括：

        1. 生成唯一的运行标识符和处理检查点恢复
        2. 配置Accelerator参数（项目目录、混合精度、梯度累积等）
        3. 初始化wandb跟踪器用于实验监控和可视化
        4. 设置随机种子以确保可重复性，同时保持设备间的多样性

        分布式训练配置是DDPO能够高效使用多GPU训练的关键。该函数还处理了以下特殊情况：
        - 如果需要从检查点恢复，会查找并设置正确的检查点路径
        - 计算实际的训练时间步数，用于梯度累积设置
        - 配置自动检查点保存和加载
        """
        # 1. 生成唯一的运行名称，格式为年.月.日_时.分.秒
        unique_id = datetime.datetime.now().strftime("%Y.%m.%d_%H.%M.%S")
        if not self.config.run_name:
            self.config.run_name = unique_id
        else:
            self.config.run_name += "_" + unique_id

        # 2. 如果需要从检查点恢复训练，处理恢复路径
        if self.config.resume_from:
            # 标准化并展开路径（支持~展开等）
            self.config.resume_from = os.path.normpath(
                os.path.expanduser(self.config.resume_from)
            )
            # 如果提供的是目录而非具体检查点
            if "checkpoint_" not in os.path.basename(self.config.resume_from):
                # 获取此目录中最新的检查点
                checkpoints = list(
                    filter(
                        lambda x: "checkpoint_" in x,
                        os.listdir(self.config.resume_from),
                    )
                )
                if len(checkpoints) == 0:
                    raise ValueError(f"在 {self.config.resume_from} 中未找到检查点")
                # 使用最后一个检查点（按编号排序）
                self.config.resume_from = os.path.join(
                    self.config.resume_from,
                    sorted(checkpoints, key=lambda x: int(x.split("_")[-1]))[-1],
                )

        # 3. 计算每个轨迹内要训练的时间步数
        # 可以通过timestep_fraction参数减少训练的时间步数量，加速训练但可能降低质量
        self.num_train_timesteps = int(
            self.config.sample.num_steps * self.config.train.timestep_fraction
        )

        # 4. 设置Accelerator配置
        # ProjectConfiguration处理项目目录和检查点命名
        accelerator_config = ProjectConfiguration(
            project_dir=os.path.join(self.config.logdir, self.config.run_name),
            automatic_checkpoint_naming=True,
            total_limit=self.config.num_checkpoint_limit,  # 限制保存的检查点数量
        )

        # 5. 创建Accelerator实例
        self.accelerator = Accelerator(
            log_with="wandb",  # 使用wandb进行日志记录
            mixed_precision=self.config.mixed_precision,  # 混合精度训练设置
            project_config=accelerator_config,
            # 设置梯度累积步数
            # 注意：我们累积的是时间步和样本的乘积，因为每个样本会对每个时间步计算一次梯度
            gradient_accumulation_steps=self.config.train.gradient_accumulation_steps
            * self.num_train_timesteps,
        )

        # 6. 初始化wandb跟踪器（只在主进程上执行）
        if self.accelerator.is_main_process:
            self.accelerator.init_trackers(
                project_name="ddpo-pytorch",
                config=self.config.to_dict(),
                init_kwargs={"wandb": {"name": self.config.run_name}},
            )

        # 7. 设置随机种子
        # device_specific=True非常重要，确保在不同GPU上生成不同的随机提示
        # 这增加了训练的多样性，同时保持了每个设备上的确定性
        set_seed(self.config.seed, device_specific=True)

    def setup_model(self):
        """设置模型、调度器和相关组件"""
        # 加载调度器、分词器和模型
        self.pipeline = StableDiffusionPipeline.from_pretrained(
            self.config.pretrained.model, revision=self.config.pretrained.revision
        )

        # 冻结模型参数以节省更多内存
        self.pipeline.vae.requires_grad_(False)
        self.pipeline.text_encoder.requires_grad_(False)
        self.pipeline.unet.requires_grad_(not self.config.use_lora)

        # 禁用安全检查器
        self.pipeline.safety_checker = None

        # 设置进度条配置
        self.pipeline.set_progress_bar_config(
            position=1,
            disable=not self.accelerator.is_local_main_process,
            leave=False,
            desc="时间步",
            dynamic_ncols=True,
        )

        # 切换到DDIM调度器
        self.pipeline.scheduler = DDIMScheduler.from_config(
            self.pipeline.scheduler.config
        )

        # 对于混合精度训练，我们将所有不可训练的权重（vae、非lora text_encoder和非lora unet）转换为半精度
        # 因为这些权重仅用于推理，保持权重为全精度不是必需的。
        inference_dtype = torch.float32
        if self.accelerator.mixed_precision == "fp16":
            inference_dtype = torch.float16
        elif self.accelerator.mixed_precision == "bf16":
            inference_dtype = torch.bfloat16

        # 将unet、vae和text_encoder移至设备并转换为inference_dtype
        self.pipeline.vae.to(self.accelerator.device, dtype=inference_dtype)
        self.pipeline.text_encoder.to(self.accelerator.device, dtype=inference_dtype)

        if self.config.use_lora:
            self.pipeline.unet.to(self.accelerator.device, dtype=inference_dtype)
            self.setup_lora()

        # 设置自动转换上下文
        # 由于某种原因，对于非lora训练，autocast是必要的，但对于lora训练，它不是必要的，而且会使用更多内存
        self.autocast = (
            contextlib.nullcontext
            if self.config.use_lora
            else self.accelerator.autocast
        )

    def setup_lora(self):
        """设置LoRA层"""
        # 设置正确的lora层
        lora_attn_procs = {}
        for name in self.pipeline.unet.attn_processors.keys():
            cross_attention_dim = (
                None
                if name.endswith("attn1.processor")
                else self.pipeline.unet.config.cross_attention_dim
            )
            if name.startswith("mid_block"):
                hidden_size = self.pipeline.unet.config.block_out_channels[-1]
            elif name.startswith("up_blocks"):
                block_id = int(name[len("up_blocks.")])
                hidden_size = list(
                    reversed(self.pipeline.unet.config.block_out_channels)
                )[block_id]
            elif name.startswith("down_blocks"):
                block_id = int(name[len("down_blocks.")])
                hidden_size = self.pipeline.unet.config.block_out_channels[block_id]

            lora_attn_procs[name] = LoRAAttnProcessor(
                hidden_size=hidden_size, cross_attention_dim=cross_attention_dim
            )
        self.pipeline.unet.set_attn_processor(lora_attn_procs)

        # 这是一个技巧，用于正确同步梯度。注册我们关心的参数的模块（在这种情况下是
        # AttnProcsLayers）也需要用于前向传递。AttnProcsLayers没有`forward`方法，
        # 所以我们包装它来添加一个，并使用闭包捕获其余的unet参数。
        class _Wrapper(AttnProcsLayers):
            def forward(self, *args, **kwargs):
                return self.pipeline.unet(*args, **kwargs)

        self.unet = _Wrapper(self.pipeline.unet.attn_processors)

        # 注册Accelerate的钩子函数
        self.accelerator.register_save_state_pre_hook(self.save_model_hook)
        self.accelerator.register_load_state_pre_hook(self.load_model_hook)

    def save_model_hook(self, models, weights, output_dir):
        """保存模型的钩子函数"""
        assert len(models) == 1
        if self.config.use_lora and isinstance(models[0], AttnProcsLayers):
            self.pipeline.unet.save_attn_procs(output_dir)
        elif not self.config.use_lora and isinstance(models[0], UNet2DConditionModel):
            models[0].save_pretrained(os.path.join(output_dir, "unet"))
        else:
            raise ValueError(f"未知的模型类型 {type(models[0])}")
        weights.pop()  # 确保accelerate不尝试处理模型的保存

    def load_model_hook(self, models, input_dir):
        """加载模型的钩子函数"""
        assert len(models) == 1
        if self.config.use_lora and isinstance(models[0], AttnProcsLayers):
            # self.pipeline.unet.load_attn_procs(input_dir)
            tmp_unet = UNet2DConditionModel.from_pretrained(
                self.config.pretrained.model,
                revision=self.config.pretrained.revision,
                subfolder="unet",
            )
            tmp_unet.load_attn_procs(input_dir)
            models[0].load_state_dict(
                AttnProcsLayers(tmp_unet.attn_processors).state_dict()
            )
            del tmp_unet
        elif not self.config.use_lora and isinstance(models[0], UNet2DConditionModel):
            load_model = UNet2DConditionModel.from_pretrained(
                input_dir, subfolder="unet"
            )
            models[0].register_to_config(**load_model.config)
            models[0].load_state_dict(load_model.state_dict())
            del load_model
        else:
            raise ValueError(f"未知的模型类型 {type(models[0])}")
        models.pop()  # 确保accelerate不尝试处理模型的加载

    def setup_optimizer(self):
        """设置优化器"""
        # 启用TF32以便在Ampere GPU上更快地训练
        # 参见 https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
        if self.config.allow_tf32:
            torch.backends.cuda.matmul.allow_tf32 = True

        # 初始化优化器
        if self.config.train.use_8bit_adam:
            try:
                import bitsandbytes as bnb
            except ImportError:
                raise ImportError(
                    "请安装bitsandbytes以使用8位Adam。您可以通过运行`pip install bitsandbytes`来完成此操作"
                )
            optimizer_cls = bnb.optim.AdamW8bit
        else:
            optimizer_cls = torch.optim.AdamW

        self.optimizer = optimizer_cls(
            (
                self.unet.parameters()
                if self.config.use_lora
                else self.pipeline.unet.parameters()
            ),
            lr=self.config.train.learning_rate,
            betas=(self.config.train.adam_beta1, self.config.train.adam_beta2),
            weight_decay=self.config.train.adam_weight_decay,
            eps=self.config.train.adam_epsilon,
        )

        # 使用accelerator准备unet和优化器
        self.unet, self.optimizer = self.accelerator.prepare(
            self.unet if self.config.use_lora else self.pipeline.unet, self.optimizer
        )

    def setup_prompt_and_reward(self):
        """设置提示和奖励函数"""
        # 准备提示和奖励函数
        self.prompt_fn = getattr(ddpo_pytorch.prompts, self.config.prompt_fn)
        self.reward_fn = getattr(ddpo_pytorch.rewards, self.config.reward_fn)()

        # 生成负面提示嵌入
        neg_prompt_embed = self.pipeline.text_encoder(
            self.pipeline.tokenizer(
                [""],
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=self.pipeline.tokenizer.model_max_length,
            ).input_ids.to(self.accelerator.device)
        )[0]
        self.sample_neg_prompt_embeds = neg_prompt_embed.repeat(
            self.config.sample.batch_size, 1, 1
        )
        self.train_neg_prompt_embeds = neg_prompt_embed.repeat(
            self.config.train.batch_size, 1, 1
        )

        # 初始化统计跟踪器
        if self.config.per_prompt_stat_tracking:
            self.stat_tracker = PerPromptStatTracker(
                self.config.per_prompt_stat_tracking.buffer_size,
                self.config.per_prompt_stat_tracking.min_count,
            )

    def log_training_config(self):
        """记录训练配置"""
        samples_per_epoch = (
            self.config.sample.batch_size
            * self.accelerator.num_processes
            * self.config.sample.num_batches_per_epoch
        )
        total_train_batch_size = (
            self.config.train.batch_size
            * self.accelerator.num_processes
            * self.config.train.gradient_accumulation_steps
        )

        logger.info("***** 开始训练 *****")
        logger.info(f"  周期数 = {self.config.num_epochs}")
        logger.info(f"  每设备采样批次大小 = {self.config.sample.batch_size}")
        logger.info(f"  每设备训练批次大小 = {self.config.train.batch_size}")
        logger.info(f"  梯度累积步数 = {self.config.train.gradient_accumulation_steps}")
        logger.info("")
        logger.info(f"  每周期样本总数 = {samples_per_epoch}")
        logger.info(
            f"  总训练批次大小 (包括并行、分布式和累积) = {total_train_batch_size}"
        )
        logger.info(
            f"  每内部周期梯度更新次数 = {samples_per_epoch // total_train_batch_size}"
        )
        logger.info(f"  内部周期数 = {self.config.train.num_inner_epochs}")

        # 检查批次大小设置是否合理
        assert self.config.sample.batch_size >= self.config.train.batch_size
        assert self.config.sample.batch_size % self.config.train.batch_size == 0
        assert samples_per_epoch % total_train_batch_size == 0

    def train(self):
        """执行训练流程"""
        # 如果需要恢复训练，加载状态
        if self.config.resume_from:
            logger.info(f"从 {self.config.resume_from} 恢复")
            self.accelerator.load_state(self.config.resume_from)
            first_epoch = int(self.config.resume_from.split("_")[-1]) + 1
        else:
            first_epoch = 0

        # 训练循环
        for epoch in range(first_epoch, self.config.num_epochs):
            # 采样阶段
            samples = self.run_sampling_phase(epoch)

            # 训练阶段
            self.run_training_phase(samples, epoch)

            # 保存检查点
            if (
                epoch != 0
                and epoch % self.config.save_freq == 0
                and self.accelerator.is_main_process
            ):
                self.accelerator.save_state()

    def run_sampling_phase(self, epoch):
        """执行采样阶段

        这个函数实现了DDPO算法的数据采样阶段，是训练循环的第一部分。
        采样阶段生成训练数据，包括图像、潜变量序列和相应的奖励值。

        主要步骤包括：
        1. 设置模型为评估模式
        2. 生成多批次的样本，每批次包括：
           a. 生成文本提示并编码为嵌入向量
           b. 使用扩散模型采样图像，同时记录中间潜变量和对数概率
           c. 异步计算生成图像的奖励
        3. 等待所有奖励计算完成
        4. 记录和可视化生成的图像和奖励
        5. 根据奖励计算优势值，用于后续的策略优化

        异步奖励计算是一个重要优化，它允许在等待奖励计算完成的同时继续生成更多样本，
        大幅提高训练效率，特别是当奖励计算涉及复杂模型（如CLIP或LLaVA）时。

        Args:
            epoch (int): 当前训练周期索引

        Returns:
            dict: 处理后的样本字典，包含潜变量、时间步和优势值等训练所需数据
        """
        # 1. 设置UNet为评估模式（不更新批归一化统计量等）
        self.pipeline.unet.eval()
        # 初始化采样数据存储列表
        samples = []
        all_prompts = []
        all_images = []

        # 2. 采样循环：生成多批次的样本
        for i in tqdm(
            range(self.config.sample.num_batches_per_epoch),
            desc=f"第 {epoch} 周期: 采样中",
            position=0,
            disable=not self.accelerator.is_local_main_process,
        ):
            # 2.1 生成提示词
            # 每次调用prompt_fn生成一个新的随机提示和相关元数据
            prompts_and_metadata = [
                self.prompt_fn(**self.config.prompt_fn_kwargs)
                for _ in range(self.config.sample.batch_size)
            ]
            prompts, prompt_metadata = zip(*prompts_and_metadata)
            all_prompts.extend(prompts)

            # 2.2 编码提示词为嵌入向量
            prompt_ids = self.pipeline.tokenizer(
                prompts,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=self.pipeline.tokenizer.model_max_length,
            ).input_ids.to(self.accelerator.device)
            prompt_embeds = self.pipeline.text_encoder(prompt_ids)[0]

            # 2.3 使用扩散模型采样图像
            # 利用带对数概率的pipeline版本，记录采样过程的所有中间状态
            with self.autocast():
                images, _, latents, log_probs = pipeline_with_logprob(
                    self.pipeline,
                    prompt_embeds=prompt_embeds,
                    negative_prompt_embeds=self.sample_neg_prompt_embeds,
                    num_inference_steps=self.config.sample.num_steps,
                    guidance_scale=self.config.sample.guidance_scale,
                    eta=self.config.sample.eta,
                    output_type="pt",
                )

            # 记录生成的图像
            all_images.extend(images)

            # 2.4 处理采样结果
            # 整理潜变量和对数概率的形状以便于训练
            latents = torch.stack(
                latents, dim=1
            )  # (batch_size, num_steps + 1, 4, 64, 64)
            log_probs = torch.stack(log_probs, dim=1)  # (batch_size, num_steps, 1)
            timesteps = self.pipeline.scheduler.timesteps.repeat(
                self.config.sample.batch_size, 1
            )  # (batch_size, num_steps)

            # 2.5 异步计算奖励
            # 将奖励计算提交到线程池，允许在计算奖励的同时继续生成样本
            rewards = self.executor.submit(
                self.reward_fn, images, prompts, prompt_metadata
            )
            time.sleep(0)  # 让计算开始执行（避免线程调度问题）

            # 2.6 存储采样数据
            samples.append(
                {
                    "prompt_ids": prompt_ids,
                    "prompt_embeds": prompt_embeds,
                    "timesteps": timesteps,
                    "latents": latents[:, :-1],  # 每个条目是时间步 t 之前的潜变量
                    "next_latents": latents[:, 1:],  # 每个条目是时间步 t 之后的潜变量
                    "log_probs": log_probs,
                    "rewards": rewards,  # 这里存储的是Future对象，而非实际奖励值
                }
            )

        # 3. 等待所有奖励计算完成
        for sample in tqdm(
            samples,
            desc="等待奖励计算",
            disable=not self.accelerator.is_local_main_process,
            position=0,
        ):
            # 获取奖励计算的结果（这会阻塞直到计算完成）
            rewards, reward_metadata = sample["rewards"].result()
            # 将结果转换为张量并存储
            sample["rewards"] = torch.as_tensor(rewards, device=self.accelerator.device)

        # 4. 记录生成的图像和奖励值到wandb日志
        self.log_samples_and_rewards(samples, all_prompts, all_images, epoch)

        # 5. 计算优势值
        # 将奖励转换为优势值，用于策略优化
        processed_samples = self.compute_advantages(samples)

        return processed_samples

    def log_samples_and_rewards(self, samples, prompts, images, epoch):
        """记录样本和奖励

        此函数负责将生成的图像样本和对应的奖励值记录到wandb，用于训练过程的可视化和监控。
        主要功能包括：

        1. 将生成的图像保存为JPEG格式
        2. 将图像与提示词和奖励值一起发送到wandb仪表盘
        3. 记录奖励的统计信息（均值、标准差等）

        这种可视化对于理解和监控训练过程至关重要，可以帮助研究人员分析：
        - 模型生成的图像质量随时间的变化
        - 不同提示词的表现差异
        - 奖励分布的变化趋势

        Args:
            samples (list): 包含批次样本数据的列表
            prompts (list): 生成的文本提示词列表
            images (list): 生成的图像张量列表
            epoch (int): 当前训练周期
        """
        # 创建临时目录用于存储图像
        # 这是一个技巧，强制wandb将图像作为JPEG而非PNG记录，减少存储和网络传输开销
        with tempfile.TemporaryDirectory() as tmpdir:
            # 1. 将图像张量转换为JPEG格式并保存
            for i, image in enumerate(images):
                # 将图像张量转换为numpy数组，并调整通道顺序和值范围
                pil = Image.fromarray(
                    (image.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
                )
                # 调整图像大小为标准尺寸（可视化用）
                pil = pil.resize((256, 256))
                # 保存为JPEG文件
                pil.save(os.path.join(tmpdir, f"{i}.jpg"))

            # 2. 获取当前批次的奖励值（用于图像标题）
            rewards = samples[0]["rewards"].cpu().numpy()

            # 3. 创建wandb图像对象列表并记录
            # 每个图像包含提示词和奖励值作为标题
            self.accelerator.log(
                {
                    "images": [
                        wandb.Image(
                            os.path.join(tmpdir, f"{i}.jpg"),
                            caption=f"{prompt:.25} | {reward:.2f}",
                        )
                        for i, (prompt, reward) in enumerate(
                            zip(prompts, rewards)
                        )  # 只记录进程0的奖励
                    ],
                },
                step=self.global_step,
            )

        # 4. 跨进程收集奖励值
        # 在分布式训练环境中，需要收集所有进程的奖励值以计算全局统计信息
        rewards = self.accelerator.gather(samples[0]["rewards"]).cpu().numpy()

        # 5. 记录奖励统计信息到wandb
        self.accelerator.log(
            {
                "reward": rewards,  # 记录原始奖励值分布
                "epoch": epoch,  # 当前周期
                "reward_mean": rewards.mean(),  # 平均奖励
                "reward_std": rewards.std(),  # 奖励标准差
            },
            step=self.global_step,
        )

    def compute_advantages(self, samples):
        """计算优势值（Advantage）用于策略优化

        在强化学习和DDPO算法中，优势函数是关键的概念，用于衡量某个动作相对于平均表现的"好坏程度"。
        优势值 = 实际获得的奖励 - 基线值（通常是奖励的期望）

        这个函数将原始奖励转换为DDPO_IS公式中使用的优势值。在DDPO论文中，优势值直接对应于
        公式中的r(x_0, c)项，即生成图像x_0的奖励。

        具体来说，DDPO_IS的完整公式为：
        ∇_θJ_DDRL = E[ ∑ (p_θ(x_{t-1} | x_t, c) / p_θ_old(x_{t-1} | x_t, c)) ∇_θ log p_θ(x_{t-1} | x_t, c) r(x_0, c)]

        其中，公式中的r(x_0, c)在代码实现中就是这个函数计算的advantages。这个函数通过归一化
        原始奖励（减去均值并除以标准差）来提高训练稳定性，本质上是一种基线减法技术。

        这个函数实现了两种计算优势值的方式：
        1. 使用每个提示词的特定统计信息（如果启用了per_prompt_stat_tracking）
           - 为每个独立的提示词维护奖励均值和标准差
           - 使用提示词特定的统计数据进行归一化，而不是整个批次的统计数据
           - 这有助于减少不同提示词之间奖励分布差异带来的影响

        2. 使用整个批次的全局统计信息（如果未启用per_prompt_stat_tracking）
           - 简单地用整个批次的均值和标准差对奖励进行归一化

        在分布式训练环境中，函数还需要处理跨进程收集和分发优势值的细节。

        Args:
            samples (list): 包含采样数据的列表，每个元素是一个字典，包含rewards、prompt_ids等键

        Returns:
            dict: 处理后的样本字典，包含计算好的优势值，并移除了已不需要的rewards和prompt_ids
        """
        # 跨进程收集奖励值
        # 使用accelerator.gather确保在分布式环境中从所有进程收集奖励
        rewards = self.accelerator.gather(samples[0]["rewards"]).cpu().numpy()

        # 计算优势值：两种方法之一
        if self.config.per_prompt_stat_tracking:
            # 方法1：使用每个提示词特定的统计信息
            # 跨进程收集提示词ID
            prompt_ids = self.accelerator.gather(samples[0]["prompt_ids"]).cpu().numpy()
            # 将ID解码为实际的提示词文本
            prompts = self.pipeline.tokenizer.batch_decode(
                prompt_ids, skip_special_tokens=True
            )
            # 使用统计跟踪器计算每个提示词特定的优势值
            # stat_tracker会为每个唯一的提示词维护一个奖励缓冲区，计算该提示词的均值和标准差
            advantages = self.stat_tracker.update(prompts, rewards)
        else:
            # 方法2：使用全局统计信息
            # 简单地用整个批次的均值和标准差对奖励进行归一化
            # 公式：(reward - mean(rewards)) / (std(rewards) + epsilon)
            # 本质上是将r(x_0, c)转换为更稳定的优势值形式
            # 其中epsilon=1e-8是为了数值稳定性，避免除以零
            advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-8)

        # 整合所有样本为单个字典，便于后续处理
        combined_samples = {
            k: torch.cat([s[k] for s in samples]) for k in samples[0].keys()
        }

        # 处理分布式环境中的优势值分发
        # 1. 将优势值转换为张量
        # 2. 重塑为[num_processes, -1]，每个进程一行
        # 3. 只保留当前进程对应的部分
        # 4. 移至正确的设备
        combined_samples["advantages"] = (
            torch.as_tensor(advantages)
            .reshape(self.accelerator.num_processes, -1)[self.accelerator.process_index]
            .to(self.accelerator.device)
        )

        # 删除不再需要的keys，节省内存
        del combined_samples["rewards"]  # 原始奖励已转换为优势值
        del combined_samples["prompt_ids"]  # 提示词ID已不再需要

        return combined_samples

    def run_training_phase(self, samples, epoch):
        """执行训练阶段

        这个函数实现了DDPO算法的训练循环，它接收采样阶段生成的数据，并执行参数更新。
        训练过程分为以下几个步骤：
        1. 对样本进行多次内部循环训练（由config.train.num_inner_epochs控制）
        2. 在每个内部循环中，对样本在批次维度和时间维度进行打乱，增加训练多样性
        3. 将样本重新批处理为适合训练的格式
        4. 执行实际的训练步骤，更新模型参数

        训练采用PPO（近端策略优化）算法，通过优势函数计算和策略梯度法更新模型权重。
        这种训练方法允许模型渐进式地调整其行为，以最大化期望奖励。

        Args:
            samples (dict): 采样阶段生成的数据字典，包含latents、次时间步、对数概率和优势值等
            epoch (int): 当前外部训练周期

        注意:
            - 函数假设samples的时间步shape正确（与config中定义的一致）
            - 每个内部周期结束时会验证是否进行了梯度同步
        """
        # 验证采样数据的形状是否符合预期
        total_batch_size, num_timesteps = samples["timesteps"].shape
        assert (
            total_batch_size
            == self.config.sample.batch_size * self.config.sample.num_batches_per_epoch
        )
        assert num_timesteps == self.config.sample.num_steps

        # 对每个内部周期进行训练（通过多个内部周期可以充分利用采样的数据）
        for inner_epoch in range(self.config.train.num_inner_epochs):
            # 打乱样本，增加训练的随机性和泛化能力
            # 打乱操作包括批次维度的打乱和独立的时间维度打乱
            shuffled_samples = self.shuffle_samples(
                samples, total_batch_size, num_timesteps
            )

            # 将样本重新批处理为适合训练的格式
            # 这将大批次的样本分割为多个小批次，便于GPU处理
            batched_samples = self.rebatch_for_training(shuffled_samples)

            # 执行实际的训练步骤，更新模型参数
            self.train_inner_epoch(batched_samples, epoch, inner_epoch)

            # 确保在内部周期结束时进行了优化步骤和梯度同步
            # 这是一个断言检查，确保训练过程正确执行
            assert self.accelerator.sync_gradients

    def shuffle_samples(self, samples, total_batch_size, num_timesteps):
        """打乱样本数据

        该函数对训练样本进行两种维度的打乱操作：
        1. 批次维度打乱：重新排列样本的顺序，打破批次间的相关性
        2. 时间维度打乱：对每个样本独立地打乱时间步的顺序

        这种双重打乱机制非常重要，因为：
        - 批次维度打乱确保模型不会对样本的顺序产生依赖
        - 时间维度打乱使模型学习到不同时间步的转换，而不是依赖于特定的时间顺序
        - 这种打乱操作增强了训练的随机性，有助于防止过拟合

        在DDPO算法中，打乱操作尤为重要，因为它帮助模型学习到更通用的去噪过程，
        而不是记住特定的噪声到图像的映射路径。

        Args:
            samples (dict): 包含训练数据的字典
            total_batch_size (int): 总批次大小
            num_timesteps (int): 时间步数量

        Returns:
            dict: 在批次和时间维度都已打乱的样本字典
        """
        # 1. 沿批次维度打乱样本
        # 生成一个随机排列索引，用于重新排列样本
        perm = torch.randperm(total_batch_size, device=self.accelerator.device)
        # 应用随机排列到所有样本数据
        samples = {k: v[perm] for k, v in samples.items()}

        # 2. 为每个样本独立地沿时间维度打乱
        # 为每个样本生成一个独立的随机时间步索引排列
        perms = torch.stack(
            [
                torch.randperm(num_timesteps, device=self.accelerator.device)
                for _ in range(total_batch_size)
            ]
        )

        # 应用时间维度打乱到相关数据
        # 这里只对与时间相关的数据进行打乱：时间步、潜变量、下一个潜变量和对数概率
        for key in ["timesteps", "latents", "next_latents", "log_probs"]:
            # 使用高级索引进行重排列
            # 第一个索引是批次中的样本索引（保持不变）
            # 第二个索引是为每个样本独立打乱的时间步索引
            samples[key] = samples[key][
                torch.arange(total_batch_size, device=self.accelerator.device)[:, None],
                perms,
            ]

        return samples

    def rebatch_for_training(self, samples):
        """重新批处理用于训练

        这个函数将采样阶段生成的大批次样本重新组织为更小的批次，以便于训练。
        在DDPO训练流程中，采样阶段通常使用较大的批次大小以提高采样效率，
        而训练阶段则使用较小的批次大小以减少显存占用并提高训练稳定性。

        具体处理过程包括：
        1. 将每个大张量重塑为多个小批次的形式，保持内部维度不变
        2. 将字典结构转换为列表结构，便于迭代处理

        这种重组方式能够保持数据的内部结构和关联性，同时使每个训练步骤只处理一小部分数据，
        更适合GPU内存管理和优化器操作。

        Args:
            samples (dict): 包含已打乱数据的字典，各个键对应不同类型的数据（潜变量、时间步等）

        Returns:
            list: 包含多个小批次样本的列表，每个元素是一个包含完整训练数据的字典
        """
        # 将每个样本大张量重塑为多个小批次的形式
        # 例如，将[total_batch_size, ...] 重塑为 [num_batches, batch_size, ...]
        # *v.shape[1:] 保持了除了第一维外的所有维度不变
        samples_batched = {
            k: v.reshape(-1, self.config.train.batch_size, *v.shape[1:])
            for k, v in samples.items()
        }

        # 将字典形式转换为列表形式，便于后续迭代处理
        # 这里使用了一个技巧，将字典的键和多个值列表配对，然后重构为多个字典
        # 结果是一个列表，每个元素是一个完整的小批次样本字典
        samples_batched = [
            dict(zip(samples_batched, x)) for x in zip(*samples_batched.values())
        ]

        return samples_batched

    def train_inner_epoch(self, batched_samples, epoch, inner_epoch):
        """训练单个内部周期

        这个函数实现了DDPO算法中的一个内部训练周期，对每个小批次样本执行梯度更新。
        内部训练循环是DDPO的核心部分，它使用PPO（近端策略优化）算法来更新模型参数。

        训练过程的主要步骤包括：
        1. 将模型设置为训练模式
        2. 对每个小批次样本：
           a. 准备条件嵌入（包括负面提示嵌入和提示嵌入）
           b. 对每个时间步执行训练（计算梯度并更新参数）
        3. 记录和收集训练信息

        这个函数处理了训练过程中的梯度累积、同步以及日志记录，确保在分布式环境中
        正确地执行参数更新。

        Args:
            batched_samples (list): 经过重新批处理的样本列表，每个元素是一个包含训练数据的字典
            epoch (int): 当前外部训练周期索引
            inner_epoch (int): 当前内部训练周期索引
        """
        # 将UNet设置为训练模式，启用梯度计算和批归一化更新
        self.pipeline.unet.train()
        # 创建一个默认字典用于收集训练信息，值默认为空列表
        info = defaultdict(list)

        # 遍历每个小批次样本
        for i, sample in tqdm(
            list(enumerate(batched_samples)),
            desc=f"第 {epoch}.{inner_epoch} 周期: 训练中",
            position=0,
            disable=not self.accelerator.is_local_main_process,
        ):
            # 准备嵌入向量
            # 如果使用分类器自由引导(CFG)，将负面提示嵌入和提示嵌入连接起来
            if self.config.train.cfg:
                embeds = torch.cat(
                    [self.train_neg_prompt_embeds, sample["prompt_embeds"]]
                )
            else:
                embeds = sample["prompt_embeds"]

            # 对每个时间步执行训练
            # num_train_timesteps是基于时间步分数截断的实际训练时间步数量
            for j in tqdm(
                range(self.num_train_timesteps),
                desc="时间步",
                position=1,
                leave=False,
                disable=not self.accelerator.is_local_main_process,
            ):
                # 执行单个时间步的训练，获取该步的训练信息
                batch_info = self.train_timestep(sample, j, embeds)

                # 收集训练信息
                for k, v in batch_info.items():
                    info[k].append(v)

                # 检查是否已执行优化步骤并同步梯度
                if self.accelerator.sync_gradients:
                    # 验证我们是否处于预期的同步点
                    # 同步应该发生在：1) 最后一个时间步，以及 2) 累积了预定数量的梯度后
                    assert (j == self.num_train_timesteps - 1) and (
                        i + 1
                    ) % self.config.train.gradient_accumulation_steps == 0

                    # 处理收集的训练信息，计算平均值
                    info = {k: torch.mean(torch.stack(v)) for k, v in info.items()}
                    # 在分布式环境中减少信息，计算所有进程的平均值
                    info = self.accelerator.reduce(info, reduction="mean")
                    # 添加额外的元信息
                    info.update({"epoch": epoch, "inner_epoch": inner_epoch})
                    # 记录训练信息到日志系统
                    self.accelerator.log(info, step=self.global_step)
                    # 更新全局步数
                    self.global_step += 1
                    # 重置信息收集字典
                    info = defaultdict(list)

    def train_timestep(self, sample, timestep_idx, embeds):
        """训练单个时间步

        这个函数实现了DDPO训练过程中最核心的部分：单个时间步的训练更新。
        它基于PPO（近端策略优化）算法，执行以下关键步骤：

        1. 前向传播：使用UNet预测噪声残差
        2. 计算对数概率：评估在当前模型下观察到的潜变量转换的概率
        3. 计算PPO目标函数：基于优势函数和新旧策略比率
        4. 反向传播：计算梯度并更新模型参数

        DDPO算法将扩散模型去噪过程建模为马尔可夫决策过程(MDP)，并使用重要性采样(IS)
        变体实现多步优化。论文中的DDPO_IS公式为：

        ∇_θJ_DDRL = E[ ∑ (p_θ(x_{t-1} | x_t, c) / p_θ_old(x_{t-1} | x_t, c)) ∇_θ log p_θ(x_{t-1} | x_t, c) r(x_0, c)]

        其中：
        - p_θ/p_θ_old 的比值就是代码中计算的ratio
        - r(x_0, c)对应于优势函数advantages
        - 最终的梯度通过反向传播自动计算

        为防止优化不稳定，DDPO采用PPO的截断技巧限制每次更新的幅度。

        PPO算法的核心思想是在策略更新时加入"近端"约束，防止单步更新过大导致训练不稳定。
        具体实现中使用了以下几个关键机制：
        - 重要性采样：通过计算新旧策略的概率比（ratio）评估更新方向
        - 比率裁剪：将策略更新的比率限制在[1-ε, 1+ε]范围内，防止过大更新
        - 优势函数：衡量特定行为相对于平均表现的"好坏"，指导策略更新方向

        Args:
            sample (dict): 包含批次训练数据的字典
            timestep_idx (int): 当前处理的时间步索引
            embeds (torch.Tensor): 文本提示的嵌入向量，如果使用CFG则包含条件和无条件嵌入

        Returns:
            dict: 包含当前步骤训练信息的字典，如损失值、KL散度估计和裁剪比率
        """
        # 初始化信息收集字典
        info = {}

        # 使用accelerator.accumulate上下文管理器处理梯度累积
        # 这允许在多个小批次上累积梯度，然后一次性更新模型参数
        with self.accelerator.accumulate(self.unet):
            # 使用自动混合精度训练（如果启用）
            with self.autocast():
                # 1. 前向传播阶段
                if self.config.train.cfg:
                    # 1.1 如果使用分类器自由引导（CFG），需要同时进行条件和无条件推理
                    # 复制潜变量和时间步，构建包含两份数据的批次（无条件+条件）
                    noise_pred = self.unet(
                        torch.cat([sample["latents"][:, timestep_idx]] * 2),
                        torch.cat([sample["timesteps"][:, timestep_idx]] * 2),
                        embeds,
                    ).sample
                    # 将预测分割为无条件和条件部分
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    # 应用CFG公式：噪声预测 = 无条件预测 + guidance_scale * (条件预测 - 无条件预测)
                    noise_pred = (
                        noise_pred_uncond
                        + self.config.sample.guidance_scale
                        * (noise_pred_text - noise_pred_uncond)
                    )
                else:
                    # 1.2 如果不使用CFG，直接进行单次推理
                    noise_pred = self.unet(
                        sample["latents"][:, timestep_idx],
                        sample["timesteps"][:, timestep_idx],
                        embeds,
                    ).sample

                # 2. 计算当前模型下next_latents相对于latents的对数概率
                # 使用修改版的DDIM步骤函数，它能返回转换概率的对数值
                # 注意：这里是关键步骤 - 使用当前模型(θ)计算对数概率，而不是使用采样时的旧模型(θ_old)
                _, log_prob = ddim_step_with_logprob(
                    self.pipeline.scheduler,
                    noise_pred,
                    sample["timesteps"][:, timestep_idx],
                    sample["latents"][:, timestep_idx],
                    eta=self.config.sample.eta,
                    prev_sample=sample["next_latents"][:, timestep_idx],
                )

            # 3. PPO策略优化逻辑 - 实现DDPO_IS公式
            # 3.1 对优势值进行裁剪，防止极端值影响训练稳定性
            # 优势值advantages对应于DDPO_IS公式中的reward项r(x_0, c)
            # 重要提示：advantages是从rewards计算得来的常数，与当前网络参数θ无关，
            # 所以在反向传播时，advantages和网络参数之间没有梯度
            advantages = torch.clamp(
                sample["advantages"],
                -self.config.train.adv_clip_max,
                self.config.train.adv_clip_max,
            )

            # 3.2 计算重要性采样比率：新策略概率 / 旧策略概率
            # 这个比率对应于DDPO_IS公式中的 p_θ(x_{t-1} | x_t, c) / p_θ_old(x_{t-1} | x_t, c)
            # 注意：新模型是当前正在训练的模型，旧模型是生成样本时使用的模型
            ratio = torch.exp(log_prob - sample["log_probs"][:, timestep_idx])

            # 3.3 计算未裁剪的PPO损失：-优势值 * 比率
            # 这里实现了DDPO_IS公式中的关键部分：
            # 重要性权重(ratio) * reward(advantages)
            # 负号是因为我们想要最大化目标函数，而优化器执行的是最小化
            # 通过反向传播，我们隐式地计算 ∇_θ log p_θ(x_{t-1} | x_t, c) 部分
            #
            # 详细梯度计算过程如下：
            # ∇_θ(-advantages * ratio)
            # = -advantages * ∇_θ(ratio)
            # = -advantages * ∇_θ(exp(log_prob - old_log_prob))
            # = -advantages * ratio * ∇_θ(log_prob)
            #
            # 注意，sample["log_probs"]是常数（从旧模型生成），不依赖于当前参数θ
            # ratio中只有log_prob依赖于当前参数θ
            # 因此，当PyTorch计算梯度时，自动实现了DDPO_IS公式中的三项乘积：
            # 重要性权重ratio * 对数似然梯度∇_θ(log_prob) * 奖励advantages
            unclipped_loss = -advantages * ratio

            # 3.4 计算裁剪后的PPO损失
            # 通过限制ratio在[1-ε, 1+ε]范围内，防止策略更新过大
            # 这是PPO算法的信任区域约束，防止模型偏离过远
            clipped_loss = -advantages * torch.clamp(
                ratio,
                1.0 - self.config.train.clip_range,
                1.0 + self.config.train.clip_range,
            )

            # 3.5 取两个损失中的最大值作为最终损失
            # 这确保了我们采用更保守的更新策略
            loss = torch.mean(torch.maximum(unclipped_loss, clipped_loss))

            # 4. 收集调试和监控指标
            # 4.1 近似KL散度估计（评估策略更新幅度）
            # 注：John Schulman建议使用(ratio - 1) - log(ratio)作为更好的估计器
            # 参考：http://joschu.net/blog/kl-approx.html
            info["approx_kl"] = 0.5 * torch.mean(
                (log_prob - sample["log_probs"][:, timestep_idx]) ** 2
            )

            # 4.2 计算裁剪比例（有多少样本的ratio超出了裁剪范围）
            info["clipfrac"] = torch.mean(
                (torch.abs(ratio - 1.0) > self.config.train.clip_range).float()
            )

            # 4.3 记录损失值
            info["loss"] = loss

            # 5. 反向传播和优化
            # 5.1 计算梯度
            # 这一步隐式计算了DDPO_IS公式中的 ∇_θ log p_θ(x_{t-1} | x_t, c) 部分
            # 完整实现了公式：∇_θJ_DDRL = E[ratio * ∇_θ log p_θ * reward]
            self.accelerator.backward(loss)

            # 5.2 如果启用了梯度同步（在分布式训练中），执行梯度裁剪
            if self.accelerator.sync_gradients:
                self.accelerator.clip_grad_norm_(
                    self.unet.parameters(), self.config.train.max_grad_norm
                )

            # 5.3 更新模型参数
            self.optimizer.step()

            # 5.4 清零梯度，准备下一次更新
            self.optimizer.zero_grad()

        # 返回本次训练步骤的信息
        return info


def main(_):
    """主函数

    这是DDPO训练脚本的入口点，负责初始化训练环境并启动训练流程。
    主要步骤包括：

    1. 获取命令行参数中指定的配置
    2. 创建DDPO训练器实例
    3. 启动训练循环

    该函数设计为通过absl.app框架调用，这使得它可以方便地与命令行参数集成。
    实际的训练逻辑都封装在DDPOTrainer类中，main函数只负责高层次的流程控制。

    Args:
        _ : 未使用的位置参数，由absl.app框架提供
    """
    # 1. 获取配置
    # FLAGS是由absl.flags模块定义的全局变量，包含所有命令行参数
    # config包含了从配置文件加载的所有训练参数
    config = FLAGS.config

    # 2. 创建训练器
    # DDPOTrainer类封装了所有训练逻辑，包括模型初始化、数据生成和优化过程
    trainer = DDPOTrainer(config)

    # 3. 开始训练
    # 训练过程包括多个周期的采样-训练循环
    trainer.train()


if __name__ == "__main__":
    app.run(main)
