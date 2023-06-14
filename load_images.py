import model as m
import torch as t
import torchvision as tv
import matplotlib.pyplot as plt

def imshow(sample_element, shape = (64, 64)):
    plt.imshow(sample_element[0].numpy().reshape(shape), cmap='gray')
    plt.title('Label = ' + str(sample_element[1]))
    plt.show()
    
transform = tv.transforms.Compose(
    [tv.transforms.ToTensor(),
        tv.transforms.Normalize((0.5,), (0.5,)),
        tv.transforms.Resize(size=(64,64)),
        tv.transforms.Grayscale(1)])
dataset_training = tv.datasets.ImageFolder(root='.\Pictures', transform=transform)
dataloader_training = t.utils.data.DataLoader(dataset_training, batch_size=32, shuffle = True)

dataset_validation = tv.datasets.ImageFolder(root='.\Validation', transform=transform)
dataloader_validation = t.utils.data.DataLoader(dataset_validation, batch_size=32, shuffle = False)

model = m.MPL()
criterion = t.nn.CrossEntropyLoss()
optimizer = t.optim.Adam(model.parameters(), lr=0.01)
num_epochs = 20
best_vloss = 1_000_000.

train_loss_history = []
train_acc_history = []
val_loss_history = []
val_acc_history = []

for epoch in range(num_epochs):
    train_loss = 0.0
    train_acc = 0.0
    val_loss = 0.0
    val_acc = 0.0

    model.train()
    
    for inputs, labels in dataloader_training:
        optimizer.zero_grad()
        output = model(inputs)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        train_acc += (output.argmax(1) == labels).sum().item()
        train_loss += loss.item()

        # calculate the average training loss and accuracy
    train_loss /= len(dataloader_training)
    train_loss_history.append(train_loss)
    train_acc /= len(dataloader_training.dataset)
    train_acc_history.append(train_acc)
 
    #set the model to evaluation mode
    model.eval()
    with t.no_grad():
        for inputs, labels in dataloader_validation:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            val_acc += (outputs.argmax(1) == labels).sum().item()
 
    # calculate the average validation loss and accuracy
    val_loss /= len(dataloader_validation)
    val_loss_history.append(val_loss)
    val_acc /= len(dataloader_validation.dataset)
    val_acc_history.append(val_acc)
 
    print(f'Epoch {epoch+1}/{num_epochs}, train loss: {train_loss:.4f}, train acc: {train_acc:.4f}')
    print(f'val loss: {val_loss:.4f}, val acc: {val_acc:.4f}')

    # Track best performance, and save the model's state
    if val_loss < best_vloss:
        best_vloss = val_loss
        model_path = 'model_{}_{}'.format('1', epoch)
        t.save(model.state_dict(), model_path)

# Plot the training and validation loss
plt.plot(train_loss_history, label='train loss')
plt.plot(val_loss_history, label='val loss')
plt.legend()
plt.show()
 
# Plot the training and validation accuracy
plt.plot(train_acc_history, label='train acc')
plt.plot(val_acc_history, label='val acc')
plt.legend()
plt.show()