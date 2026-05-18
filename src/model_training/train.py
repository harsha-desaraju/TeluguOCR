
import time
import torch
from tqdm import tqdm
from src.model_training.model import mae_loss



def build_optimizer(model, base_lr, weight_decay):
    param_groups = [
        {
            "params": [p for n, p in model.named_parameters() if p.requires_grad],
            "weight_decay": weight_decay,
        }
    ]
    return torch.optim.AdamW(
        param_groups,
        lr=base_lr,
        betas=(0.9, 0.95)
    )


def build_scheduler(optimizer, epochs, steps_per_epoch, warm_up_ratio):
    total_steps = epochs * steps_per_epoch
    warmup_steps = int(warm_up_ratio * steps_per_epoch)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1. + torch.cos(torch.tensor(progress * 3.1415926535)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train_one_step(model, images, optimizer, cfg):
    optimizer.zero_grad(set_to_none=True)

    pred, mask = model(images)
    loss = mae_loss(images, pred, mask, cfg)

    loss.backward()
    optimizer.step()

    return loss.detach()


def test_one_step(model, images, cfg):
    with torch.no_grad():
        pred, mask = model(images)
        loss = mae_loss(images, pred, mask, cfg)

    return loss.detach()


def train_one_epoch(model, dataloader, optimizer, scheduler, device, cfg, epoch):
    model.train()
    total_loss = 0.0

    pbar = tqdm(dataloader, desc=f"Training Epoch {epoch}")

    num_batches = 0
    for images in pbar:
        images = images.to(device, non_blocking=True)

        loss = train_one_step(
            model=model,
            images=images,
            optimizer=optimizer,
            cfg=cfg
        )

        scheduler.step()

        total_loss += loss.item()
        pbar.set_postfix(loss=f"{loss.item():.4f}")

        num_batches += 1

    return total_loss / num_batches



def test_one_epoch(model, dataloader, device, cfg, epoch):
    model.eval()
    total_loss = 0.0

    pbar = tqdm(dataloader, desc=f"Testing Epoch {epoch}")

    num_batches = 0
    for images in pbar:
        images = images.to(device, non_blocking=True)

        loss = test_one_step(
            model=model,
            images=images,
            cfg=cfg
        )

        total_loss += loss.item()
        pbar.set_postfix(loss=f"{loss.item():.4f}")

        num_batches += 1

    return total_loss / num_batches




def train_mae(model, train_dataloader, test_dataloader, cfg, epochs=100, base_lr=1.5e-4, weight_decay=0.05, steps_per_epoch=1000, warm_up_ratio=0.1, device="mps"):
    model.to(device)

    optimizer = build_optimizer(
        model,
        base_lr=base_lr,
        weight_decay=weight_decay
    )

    scheduler = build_scheduler(
        optimizer,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        warm_up_ratio=warm_up_ratio
    )

    for epoch in range(epochs):
        # Train the model
        # st = time.perf_counter()
        avg_train_loss = train_one_epoch(
            model=model,
            dataloader=train_dataloader,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            cfg=cfg,
            epoch=epoch
        )
        # print(f"\nTime taken for training epoch {epoch}: {(time.perf_counter() - st):.2f}s")

        # Test the model
        # st = time.perf_counter()
        avg_test_loss = test_one_epoch(
            model=model,
            dataloader=test_dataloader,
            device=device,
            cfg=cfg,
            epoch=epoch
        )
        # print(f"\nTime taken for testing epoch {epoch}: {(time.perf_counter() - st):.2f}s")
        print(f"\nEpoch {epoch} |  Avg Train Loss: {avg_train_loss:.4f} | Avg Test Loss: {avg_test_loss:.4f}")

        check_point = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "train_loss": avg_train_loss,
            "test_loss": avg_test_loss
        }

        # Save model after an epoch
        torch.save(check_point,
                   f"/Users/xai/Personal/Projects/TeluguOCR/src/model_training/model_2/model_checkpoint_{epoch}.pt")




if __name__ == '__main__':
    from torch.utils.data import DataLoader
    from src.model_training.model import MAEConfig, MaskedAutoencoderViT
    from src.model_training.utils import ImageDataset, create_batch

    num_words_per_page_est = 150
    num_train_epochs = 2
    num_images_per_epoch_train = 30
    num_images_per_epoch_test = 10
    batch_size = 64
    num_steps_per_epoch = (num_images_per_epoch_train * num_words_per_page_est) // batch_size
    warmup_ratio = 0.1


    train_dataset = ImageDataset(split="train", num_images_per_epoch=num_images_per_epoch_train)
    test_dataset = ImageDataset(split="test", num_images_per_epoch=num_images_per_epoch_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=create_batch, num_workers=4, prefetch_factor=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=create_batch, num_workers=4, prefetch_factor=2)

    model_config = MAEConfig(img_height=32, img_width=128, enc_depth=8, enc_num_heads=4,
                             in_channels=3, enc_embed_dim=256, dec_embed_dim=128, dec_num_heads=4, mask_ratio=0.7)

    model = MaskedAutoencoderViT(model_config)
    model.load_state_dict(
        torch.load("/Users/xai/Personal/Projects/TeluguOCR/src/model_training/model_1/model_final.pt"))

    train_mae(model, train_loader, test_loader, model_config, epochs=num_train_epochs, device="mps", steps_per_epoch=num_steps_per_epoch, warm_up_ratio=warmup_ratio)


    # Save the model
    torch.save(model.state_dict(), "/Users/xai/Personal/Projects/TeluguOCR/src/model_training/model_2/model_final.pt")


