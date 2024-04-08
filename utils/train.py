import torch
import torch.nn as nn
import matplotlib.pyplot as plt

def train_model(model,
                dataloaders,
                criterion = nn.MSELoss(),
                prev_checkpoint=None,
                learning_rate=0.001,
                optimizer = None, 
                schedular=None, 
                num_epoch=10,
                save_checkpoint=True):
    
    import copy
    import time
    from tqdm import tqdm
    from IPython.display import clear_output

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # define default parameters
    if optimizer == None:
        optimizer = torch.optim.Adam(
            params=model.parameters(),
            lr=learning_rate
        )

    # get dataset and dataloader sizes
    dataset_sizes = {
        "train": len(dataloaders["train"].dataset),
        "val": len(dataloaders["val"].dataset)
    }
    dataloader_sizes = {
        "train": len(dataloaders["train"]),
        "val": len(dataloaders["val"])
    }

    # defining variables for storing loss and accuracy
    train_time_list = []

    if prev_checkpoint != None:
        print(model.load_state_dict(prev_checkpoint["model_state"]))
        optimizer.load_state_dict(prev_checkpoint["optim_state"])
        epochs_completed = prev_checkpoint["epoch"]
        epoch_loss_list = prev_checkpoint["epoch_losses"]
        train_time = prev_checkpoint["time_taken"]
        print('Loaded checkpoint')
    else:
        epochs_completed = 0
        epoch_loss_list = {
            "train": [],
            "val": []
        }
        train_time = 0
    
    print(f"Training Started on {device}")

    for epoch in range(epochs_completed, num_epoch+epochs_completed):

        # Each epoch has a training and a validation phase
        for phase in ["train", "val"]:
            time_start = time.time()

            model.train() if phase == "train" else model.eval()

            epoch_losses = []
            running_loss = 0.0

            # Use tqdm for progress bar during training
            data_loader = tqdm(dataloaders[phase], desc=f'{phase.capitalize()} Epoch {epoch+1}/{num_epoch+epochs_completed}')

            for i, (images, labels) in enumerate(data_loader):
                images = images.to(device)
                labels = labels.to(device)
                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(images)
                    loss = criterion(outputs, labels)

                    # Backward + optimize only if in training loop
                    if phase == "train":
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                
                if i % 100 == 0:
                    clear_output()
                    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15,5))
                    axes[0].imshow(images[0].detach().cpu().permute(1,2,0))
                    axes[1].imshow(labels[0].detach().cpu().permute(1,2,0))
                    axes[2].imshow(outputs[0].detach().cpu().permute(1,2,0))
                    axes[0].axis('off')
                    axes[1].axis('off')
                    axes[2].axis('off')
                    fig.tight_layout()
                    plt.show()

                current_loss = loss.item() * images.size(0)
                epoch_losses.append(current_loss)
                running_loss += current_loss

                if (i+1) % 10 == 0:
                    data_loader.set_postfix({'loss': loss.item()})

            if schedular != None and phase == "train":
                    schedular.step()

            epoch_time = time.time() - time_start

            if phase == "train":
                train_time_list.append(epoch_time) 

            epoch_loss_list[phase].append(epoch_losses)

        print("\n")

    train_time += sum(train_time_list)

    checkpoint = {
        "epoch": num_epoch,
        "criterion": criterion,
        "model_state": model.state_dict(),
        "optim_state": optimizer.state_dict(),
        "epoch_losses": epoch_loss_list,
        "time_taken": train_time
    }
    
    if save_checkpoint == True:
        file_name = f"checkpoint_{time.time()}.pth"
        torch.save(checkpoint, file_name)
    
    return model, checkpoint
