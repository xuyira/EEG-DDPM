"""
主入口文件
用于训练 DDPM 模型生成 EEG 数据的 CWT 表示
"""
import os
import torch
import matplotlib.pyplot as plt
import matplotlib as mpl
from pathlib import Path
from torch.utils.data import DataLoader
from diffusers import DDPMScheduler, DDPMPipeline
import torch.nn.functional as F
import gc
from diffusers.optimization import get_cosine_schedule_with_warmup

from config import TrainingConfig
from model import create_unet_model
from utils.data_preprocessing import bci2a_preprcessing_method, DelayEmbedder

from accelerate import Accelerator
from huggingface_hub import create_repo, upload_folder
from tqdm.auto import tqdm

def main():
    """主函数"""
    # 创建配置（目录会在 __post_init__ 中自动创建）
    config = TrainingConfig()
    
    print("图片输出文件夹：", config.pic_dir)
    print("模型输出文件夹：", config.output_dir)
    
    # 选择运行设备
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")

    # 数据预处理（使用 DelayEmbedder 将 EEG 时序信号转成图片）
    print("\n开始数据预处理（DelayEmbedder）...")
    X_train_img_np, X_train_label, X_train_sub = bci2a_preprcessing_method(
        root=config.data_root,
        leave_sub=config.leave_sub,
        subjects=config.subjects,
        delay=config.delay,
        embedding=config.embedding,
        device=device,
    )

    # 转换为 tensor，(B, C, H, W)
    print("转换为 tensor...")
    X_train_img = torch.from_numpy(X_train_img_np).float()
    # 删除 numpy 数组以释放内存
    del X_train_img_np
    gc.collect()  # 强制垃圾回收
    print("数据维度：", X_train_img.shape)
    
    # 可视化一个样本
    show_trial = 0
    show_channel = 0
    plt.figure()
    plt.imshow(X_train_img[show_trial, show_channel, :, :].numpy(), aspect='auto')
    plt.colorbar()
    plt.title("Normalized EEG Signal after transform trial{} channel{}".format(show_trial, show_channel))
    plt.savefig(f"{config.pic_dir}/sample_transpic_trial{show_trial}_channel{show_channel}.png", dpi=300)
    plt.close()
    
    # 创建模型
    print("\n创建模型...")
    model = create_unet_model(config)
    
    # 测试模型输入输出
    sample_image = X_train_img[0].unsqueeze(0)
    print("输入图像维度：", sample_image.shape)
    print("输出维度：", model(sample_image, timestep=0).sample.shape)
    
    # 创建数据加载器
    train_dataloader = DataLoader(
        X_train_img, 
        batch_size=config.train_batch_size, 
        shuffle=True
    )
    
    # 创建噪声调度器
    noise_scheduler = DDPMScheduler(num_train_timesteps=1000)
    
    # 创建优化器和学习率调度器
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=config.lr_warmup_steps,
        num_training_steps=(len(train_dataloader) * config.num_epochs),
    )
    
    # 开始训练
    print("\n开始训练...")
    train_loop(
        config=config,
        model=model,
        noise_scheduler=noise_scheduler,
        optimizer=optimizer,
        train_dataloader=train_dataloader,
        lr_scheduler=lr_scheduler,
        pic_dir=config.pic_dir
    )
    
    print("\n训练完成！")


if __name__ == "__main__":
    main()

def train_loop(config, model, noise_scheduler, optimizer, train_dataloader, lr_scheduler, pic_dir):
    # Initialize accelerator and tensorboard logging
    accelerator = Accelerator(
        mixed_precision=config.mixed_precision,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        log_with="tensorboard",
        project_dir=os.path.join(config.output_dir, "logs"),
    )
    if accelerator.is_main_process:
        if config.output_dir is not None:
            os.makedirs(config.output_dir, exist_ok=True)
        if config.push_to_hub:
            repo_id = create_repo(
                repo_id=config.hub_model_id or Path(config.output_dir).name, exist_ok=True
            ).repo_id
        accelerator.init_trackers("train_example")

    # Prepare everything
    # There is no specific order to remember, you just need to unpack the
    # objects in the same order you gave them to the prepare method.
    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )

    global_step = 0

    # Now you train the model
    for epoch in range(config.num_epochs):
        progress_bar = tqdm(total=len(train_dataloader), disable=not accelerator.is_local_main_process)
        progress_bar.set_description(f"Epoch {epoch}")

        for step, batch in enumerate(train_dataloader):
            clean_images = batch
            # Sample noise to add to the images
            noise = torch.randn(clean_images.shape, device=clean_images.device)
            bs = clean_images.shape[0]

            # Sample a random timestep for each image
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps, (bs,), device=clean_images.device,
                dtype=torch.int64
            )

            # Add noise to the clean images according to the noise magnitude at each timestep
            # (this is the forward diffusion process)
            noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)

            with accelerator.accumulate(model):
                # Predict the noise residual
                noise_pred = model(noisy_images, timesteps, return_dict=False)[0]
                loss = F.mse_loss(noise_pred, noise)
                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            progress_bar.update(1)
            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0], "step": global_step}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)
            global_step += 1

        # After each epoch you optionally sample some demo images with evaluate() and save the model
        if accelerator.is_main_process:
            from diffusers import DDPMPipeline
            pipeline = DDPMPipeline(unet=accelerator.unwrap_model(model), scheduler=noise_scheduler)

            if (epoch + 1) % config.save_image_epochs == 0 or epoch == config.num_epochs - 1:
                evaluate(config, epoch, pipeline, pic_dir)

            if (epoch + 1) % config.save_model_epochs == 0 or epoch == config.num_epochs - 1:
                if config.push_to_hub:
                    upload_folder(
                        repo_id=repo_id,
                        folder_path=config.output_dir,
                        commit_message=f"Epoch {epoch}",
                        ignore_patterns=["step_*", "epoch_*"],
                    )
                else:
                    pipeline.save_pretrained(config.output_dir)

def evaluate(config, epoch, pipeline, pic_dir, sample_channel=0):
    # 从随机噪声中生成一些图像（这是反向扩散过程）。
    # 默认的管道输出类型是 `List[torch.Tensor]`
    # 取sample_channel来展示

    images = pipeline(
        batch_size=config.eval_batch_size,
        generator=torch.Generator(device='cpu').manual_seed(config.seed),  # 使用单独的 torch 生成器来避免回绕主训练循环的随机状态
        output_type="np.array",
    ).images

    print("生成图像维度：", images.shape)
    # (batch, height, width, channel)

    # 将生成的eval_batch_size个图像拼接成一张大图（图片域）
    fig, ax = plt.subplots(2, 8, figsize=(20, 4))
    for i in range(2):
        for j in range(8):
            if i * 8 + j < images.shape[0]:
                ax[i, j].imshow(images[i * 8 + j, :, :, sample_channel], aspect='auto')
                ax[i, j].axis("off")
                ax[i, j].set_title(f"Image {i * 8 + j}")

    plt.savefig(f"{pic_dir}/{epoch:04d}.png", dpi=400)
    plt.close()

    # ================== 基于 img_to_ts 还原 EEG 信号并画图 ==================
    try:
        import numpy as np

        # images: (B, H, W, C) -> (B, C, H, W) 并转为 tensor
        images_bchw = np.transpose(images, (0, 3, 1, 2))
        img_tensor = torch.from_numpy(images_bchw).float()

        # 使用与预处理相同的 DelayEmbedder 参数
        # BCI2a 每 trial 的时间点数为 1000，这里直接使用 1000
        seq_len = 1000
        delay = config.delay
        embedding = config.embedding

        # 按照 DelayEmbedder.ts_to_img 的逻辑计算原始列数 original_rows（未 pad 前的宽度）
        i = 0
        while (i * delay + embedding) <= seq_len:
            i += 1
        if i * delay != seq_len and i * delay + embedding > seq_len:
            i += 1

        original_cols = embedding
        original_rows = i

        embedder = DelayEmbedder(device="cpu", seq_len=seq_len, delay=delay, embedding=embedding)
        # 手动设置 img_shape，供 unpad 使用
        embedder.img_shape = (img_tensor.shape[0], img_tensor.shape[1], original_cols, original_rows)

        # 利用 img_to_ts 将图像还原为时间序列: (B, T, C)
        ts_tensor = embedder.img_to_ts(img_tensor)
        ts_np = ts_tensor.detach().cpu().numpy()  # (B, T, C)

        # 画第二种图：时间域 EEG，大图按 epoch 存一张
        fig_ts, ax_ts = plt.subplots(2, 8, figsize=(20, 6))
        for i in range(2):
            for j in range(8):
                idx = i * 8 + j
                if idx < ts_np.shape[0]:
                    sample = ts_np[idx]  # (T, C) = (1000, 22)
                    t = np.arange(sample.shape[0])
                    offset = 0.0
                    # 仿照 visualize_eeg 的堆叠效果，简单做通道平移
                    for ch in range(sample.shape[1]):
                        ax_ts[i, j].plot(t, sample[:, ch] + offset, linewidth=0.4)
                        offset += 5.0
                    ax_ts[i, j].axis("off")
                    ax_ts[i, j].set_title(f"EEG {idx}")

        plt.tight_layout()
        plt.savefig(f"{pic_dir}/{epoch:04d}_ts.png", dpi=400)
        plt.close(fig_ts)
    except Exception as e:
        print(f"基于 img_to_ts 的 EEG 信号可视化失败: {e}")