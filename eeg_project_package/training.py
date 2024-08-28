import torch
from tqdm import tqdm
import copy

#Training the model
def train_model(model,trainloader,testloader,device,criterion,feature_type,epochs=1,optimizer=None,schedulers_dict={},print_epoch=10):
    if optimizer is None:
        optimizer = torch.optim.Adam(model.parameters())

    logs = { 
            "train_balanced_accs": [], 
            "train_losses": [],
            "test_balanced_accs": [],
            "test_losses": [],
            "best_model_weifhts": None,
            }

    for epoch in range(epochs):

        total_loss = 0
        correct = 0
        correct0 = 0
        correct1 = 0
        size = 0
        class_size = torch.zeros(2).to(device)
        class_size[0] = trainloader.dataset.labels.eq(0).sum().item()
        class_size[1] = trainloader.dataset.labels.eq(1).sum().item()

        with tqdm(trainloader, total=len(trainloader), unit="batch", desc=f'Epoch {epoch}') as tepoch:

            if epoch % print_epoch == 0:
                print(f'Epoch {epoch}')
                print("lr: ", optimizer.param_groups[0]['lr'])
                print("-------------------------")
                train_balanced_accuracy, train_loss = evaluate_classification_model(model,trainloader,device,criterion,feature_type,dataset="Train")
                logs["train_balanced_accs"].append(train_balanced_accuracy)
                logs["train_losses"].append(train_loss)
                test_balanced_accuracy, test_loss = evaluate_classification_model(model,testloader,device,criterion,feature_type,dataset="Test")
                logs["test_balanced_accs"].append(test_balanced_accuracy)
                logs["test_losses"].append(test_loss)
                if len(logs["test_balanced_accs"]) == 1 or test_balanced_accuracy > max(logs["test_balanced_accs"][:-1]):
                    logs["best_model_weights"] = copy.deepcopy(model.state_dict())
                    print("Saving best model weights with balanced accuracy: ", test_balanced_accuracy)
                print("-------------------------")

                for key in schedulers_dict:
                    if key == "ReduceLROnPlateau":
                        schedulers_dict[key].step(test_loss)

            model.train()
            
            for features,label in tepoch:
                tepoch.set_description(f"Epoch {epoch}")
                X = features[feature_type]
                X = X.to(device)
                y = label.to(device)
                outputs = model(X)
                
                if criterion.__class__.__name__ == "BCEWithLogitsLoss":
                    loss = criterion(outputs.flatten(),y.float())
                    total_loss += loss.item()
                    predicted = (outputs > 0.5).float().flatten()
                else:
                    loss = criterion(outputs,y)
                    total_loss += loss.item()
                    _, predicted = torch.max(outputs, 1)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # compute accuracy
                correct += (predicted == y.flatten()).sum().item()
                correct0 += ((predicted == y.flatten())*(y.flatten() == 0)).sum().item()
                correct1 += ((predicted == y.flatten())*(y.flatten() == 1)).sum().item()
                size += len(y)
                accuracy = correct/size
                loss_elem = total_loss/size

                tepoch.set_postfix(loss=loss_elem, accuracy=accuracy)
        
        for key in schedulers_dict:
            if key == "MultiplicativeLR":
                schedulers_dict[key].step()

    return logs


#Evaluating the model with balanced accuracy
def evaluate_classification_model(model,dataloader,device,criterion,feature_type,dataset="Train"):
    model.eval()
    total_loss = 0
    correct = torch.zeros(2).to(device)
    class_size = torch.zeros(2).to(device)
    class_size[0] = dataloader.dataset.labels.eq(0).sum().item()
    class_size[1] = dataloader.dataset.labels.eq(1).sum().item()

    with torch.no_grad():
        for features,label in dataloader:
            X = features[feature_type]
            X = X.to(device)
            y = label.to(device)
            outputs = model(X)
            # check if loss is BCEWithLogitsLoss or CrossEntropyLoss
            if criterion.__class__.__name__ == "BCEWithLogitsLoss":
                loss = criterion(outputs.flatten(),y.float())
                total_loss += loss.item()
                predicted = (outputs > 0.5).float().flatten()
            else:
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

    return balanced_accuracy, total_loss
