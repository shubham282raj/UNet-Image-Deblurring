import torch
import torch.nn as nn
import matplotlib.pyplot as plt

def train_model(model,
                dataloaders,
                criterion = nn.MSELoss(),
                learning_rate=0.001,
                optimizer = None, 
                schedular=None, 
                num_epoch=10,
                save_checkpoint=False,
                time_start_from=0):
    
    import copy
    import time
    from tqdm import tqdm
    from IPython.display import clear_output
    
    if optimizer == None:
        optimizer = torch.optim.Adam(
            params=model.parameters(),
            lr=learning_rate
        )

    dataset_sizes = {
        "train": len(dataloaders["train"].dataset),
        "val": len(dataloaders["val"].dataset)
    }

    dataloader_sizes = {
        "train": len(dataloaders["train"]),
        "val": len(dataloaders["val"])
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)

    print(f"Training Started on {device}")
    train_time = time_start_from
    train_time_list = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc_epoch = 0

    epoch_loss_list = {
        "train": [],
        "val": []
    }

    epoch_acc_list = {
        "train": [],
        "val": []
    }

    for epoch in range(num_epoch):
        # print(f"Epoch {epoch+1}/{num_epoch}", end="")

        # Each epoch has a training and a validation phase
        for phase in ["train", "val"]:
            time_start = time.time()

            model.train() if phase == "train" else model.eval()

            running_loss = 0.0

            # Use tqdm for progress bar during training
            data_loader = tqdm(dataloaders[phase], desc=f'{phase.capitalize()} Epoch {epoch+1}/{num_epoch}')

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

                if phase == "train" and (i+1) % int(dataloader_sizes[phase]/10) == 0:
                    data_loader.set_postfix({'loss': loss.item()})
                
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

                running_loss += loss.item() * images.size(0)

            if schedular != None and phase == "train":
                    schedular.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            
            epoch_time = time.time() - time_start
            if phase == "train":
                train_time_list.append(epoch_time) 

            print(f"\n{phase} Loss: {epoch_loss:.2f}", f"Time_Taken: {epoch_time//60:.0f}m {epoch_time%60:.0f}s", end="")

            epoch_loss_list[phase].append(float(epoch_loss))

            # deep copy the best model
            if phase == "val" and epoch_loss < epoch_loss_list["val"][best_acc_epoch]:
                best_acc_epoch = epoch
                best_model_wts = copy.deepcopy(model.state_dict())

        print("\n")

    train_time += sum(train_time_list[:best_acc_epoch+1])
    print(f"Training Finished (till best accuracy) in {train_time//60:.0f}m {train_time%60:.0f}s")
    print(model.__class__.__name__ + "_epoch_" + str(num_epoch) + "_optim_" + optimizer.__class__.__name__ + "_criterion_" + criterion.__class__.__name__)

    model.load_state_dict(best_model_wts)

    checkpoint = {
        "epoch": num_epoch,
        "criterion": criterion,
        "model_state": model.state_dict(),
        "optim_state": optimizer.state_dict(),
        "epoch_losses": epoch_loss_list,
        "epoch_accuracies": epoch_acc_list,
        "best_acc_epoch": best_acc_epoch,
        "time_taken": train_time
    }
    
    if save_checkpoint == True:
        file_name = f"checkpoint_{time.time()}.pth"
        torch.save(checkpoint, file_name)
    
    return model, checkpoint
