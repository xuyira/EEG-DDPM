import torch
import matplotlib.pyplot as plt
import matplotlib as mpl
from diffusers import DDPMPipeline

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

    # 将生成的eval_batch_size个图像拼接成一张大图
    fig, ax = plt.subplots(2, 10, figsize=(20, 4))
    for i in range(2):
        for j in range(10):
            if i * 10 + j < images.shape[0]:
                ax[i, j].imshow(images[i * 10 + j, :, :, sample_channel], aspect='auto')
                ax[i, j].axis("off")
                ax[i, j].set_title(f"Image {i * 10 + j}")

    plt.savefig(f"{pic_dir}/{epoch:04d}.png", dpi=400)
    plt.close()

