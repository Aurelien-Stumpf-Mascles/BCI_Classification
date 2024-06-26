import torch
from tqdm import tqdm

#Training the model
def train_model(model,trainloader,testloader,device,criterion,epochs=1,optimizer=None,scheduler=None,print_epoch=10):
    if optimizer is None:
        optimizer = torch.optim.Adam(model.parameters())

    model.train()

    # compute class sizes
    class_size = torch.zeros(2).to(device)

    for epoch in range(epochs):

        total_loss = 0
        correct = 0
        size = 0

        with tqdm(trainloader, total=len(trainloader), unit="batch", desc=f'Epoch {epoch}') as tepoch:
            for X,y in tepoch:
                tepoch.set_description(f"Epoch {epoch}")
                X = X.to(device)
                y = y.to(device)
                outputs = model(X)
                loss = criterion(outputs,y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()/len(trainloader.dataset)
                _, predicted = torch.max(outputs, 1)

                # compute accuracy
                correct += (predicted == y.flatten()).sum().item()
                size += len(y)
                accuracy = correct/size

                tepoch.set_postfix(loss=total_loss, accuracy=accuracy)
            
            scheduler.step()

            if epoch % print_epoch == 0:
                print(f'Epoch {epoch}')
                print("lr: ", optimizer.param_groups[0]['lr'])
                print("-------------------------")
                evaluate_classification_model(model,trainloader,device,criterion,dataset="Train")
                evaluate_classification_model(model,testloader,device,criterion,dataset="Test")
                print("-------------------------")

#Evaluating the model with balanced accuracy
def evaluate_classification_model(model,dataloader,device,criterion,dataset="Train"):
    model.eval()
    total_loss = 0
    correct = torch.zeros(2).to(device)
    class_size = torch.zeros(2).to(device)
    class_size[0] = dataloader.dataset.labels.eq(0).sum().item()
    class_size[1] = dataloader.dataset.labels.eq(1).sum().item()

    with torch.no_grad():
        for X,y in dataloader:
            X = X.to(device)
            y = y.to(device)
            outputs = model(X)
            loss = criterion(outputs,y)
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            # compute balanced accuracy
            correct[0] += ((predicted == y.flatten())*(y.flatten() == 0)).sum()
            correct[1] += ((predicted == y.flatten())*(y.flatten() == 1)).sum()
    
    balanced_accuracy = (0.5*correct[0]/class_size[0] + 0.5*correct[1]/class_size[1]).item()
    total_loss /= len(dataloader.dataset)

    print('{} set: Avg. loss: {:.4f}, Balanced Accuracy: {} ({:.0f}%)'.format(
    dataset,
    total_loss, balanced_accuracy ,
    100. * balanced_accuracy))
