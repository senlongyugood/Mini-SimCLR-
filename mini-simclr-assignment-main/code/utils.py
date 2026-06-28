import torch
import os


def save_checkpoint(model, optimizer, epoch, loss, config, path):
    """保存训练检查点"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'config': config,
    }, path)


def load_checkpoint(model, path, optimizer=None, device='cpu'):
    """加载检查点，可选择恢复 optimizer 状态"""
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint.get('epoch', 0), checkpoint.get('loss', None)


def load_pretrained_encoder(model, path, device='cpu'):
    """只加载预训练的 encoder 权重（用于 linear probe）

    训练好的 SimCLR checkpoint 里 key 有 'encoder.xxx' 前缀，
    直接 load_state_dict 整体加载即可。
    """
    checkpoint = torch.load(path, map_location=device)
    state = checkpoint['model_state_dict']
    # 过滤出 encoder 相关权重
    encoder_state = {k.replace('encoder.', ''): v
                     for k, v in state.items()
                     if k.startswith('encoder.')}
    model.load_state_dict(encoder_state, strict=False)
    return model
