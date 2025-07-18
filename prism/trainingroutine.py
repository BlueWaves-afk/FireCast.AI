# Optimized training loop for RTX 4060 (8GB)
import torch
from tqdm import tqdm
from torch.amp import GradScaler, autocast

# Enable cuDNN optimizations
torch.backends.cudnn.benchmark = True

def train_one_epoch(model, loader, optimizer, loss_fn, device, scaler, accumulation_steps=1):
    model.train()
    running_loss = 0.0
    device_type = device.type

    for i, (x, y) in enumerate(tqdm(loader)):
        x = x.to(device, memory_format=torch.channels_last, non_blocking=True)
        y = y.to(device, non_blocking=True)

        if y.ndim == 4 and y.shape[1] == 1:
            y = y.squeeze(1)
        y = y.long()

        if (y != 255).sum() == 0:
            continue

        with autocast(device_type=device_type):
            outputs = model(x)
            loss = loss_fn(outputs, y) / accumulation_steps

        if torch.isnan(loss):
            raise ValueError("Loss is NaN")

        scaler.scale(loss).backward()

        if (i + 1) % accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        running_loss += loss.item() * accumulation_steps

    torch.cuda.empty_cache()
    return running_loss / len(loader)



def validate(model, loader, loss_fn, device):
    model.eval()
    val_loss = 0.0
    device_type = device.type

    with torch.no_grad():
        for i, (x, y) in enumerate(loader):
            x = x.to(device, memory_format=torch.channels_last, non_blocking=True)
            y = y.to(device, non_blocking=True)

            if y.ndim == 4 and y.shape[1] == 1:
                y = y.squeeze(1)
            y = y.long()

            # Skip batch if all pixels are ignore_index (255)
            if (y != 255).sum() == 0:
                continue

            with autocast(device_type=device_type):
                outputs = model(x)
                loss = loss_fn(outputs, y)

            if torch.isnan(loss):
                raise ValueError("NaN loss in validation")

            val_loss += loss.item()

    torch.cuda.empty_cache()
    return val_loss / len(loader)


# Full training routine with RTX 4060 optimizations
def FireCastTrainer(model, train_loader, val_loader, loss_fn, optimizer, scheduler):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device, memory_format=torch.channels_last)

    num_epochs = 10
    val_loss_threshold = 0.20  # 🔹 Set your desired threshold here
    scaler = GradScaler()
    accumulation_steps = 2  # Simulate larger batch sizes

    for epoch in range(num_epochs):
        print(f"🌟 Epoch {epoch+1}/{num_epochs}")
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device, scaler, accumulation_steps)
        val_loss = validate(model, val_loader, loss_fn, device)

        print(f"📉 Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        scheduler.step(val_loss)

        torch.cuda.empty_cache()  # Keep GPU clean each epoch
    torch.save(model.state_dict(), "best_model.pth")
