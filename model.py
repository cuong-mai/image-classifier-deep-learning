import torch
from torch import nn, optim
from torchvision import models
import numpy as np
from utils import process_image, get_class_from_index
from workspace_utils import active_session


# Build a model
def build_model(arch='vgg16', hidden_units=1024, epochs_done=0, state_dict=None, class_to_idx={}, device='cpu'):
    print('Building model ' + arch + '... ')
    
    if arch == 'vgg16':
        model = models.vgg16(pretrained=True)
        input_units = 25088
    elif arch == 'resnet18':
        model = models.resnet18(pretrained=True)
        input_units = 512

    for param in model.parameters():
        param.requires_grad = False

    classifier = nn.Sequential(
        nn.Linear(input_units, hidden_units),
        nn.ReLU(),
        nn.Dropout(p=0.2),
        nn.Linear(hidden_units, 102),
        nn.LogSoftmax(dim=1)
    )

    if arch == 'vgg16':
        model.classifier = classifier
    elif arch == 'resnet18':
        model.fc = classifier
    
    model.arch = arch
    model.hidden_units = hidden_units
    model.epochs_done = epochs_done
    
    if state_dict != None:
        model.load_state_dict(state_dict)
    
    model.class_to_idx = class_to_idx
    
    model.eval()
    model.to(device)
    
    print('Done!\n')
    
    return model

    
# Build optimizer
def build_adam_optimizer(model, learning_rate=0.001, state_dict=None):
    print('Building Adam optimizer... ')
    
    arch = model.arch
    if arch == 'vgg16':
        classifier = model.classifier
    elif arch == 'resnet18':
        classifier = model.fc

    optimizer = optim.Adam(classifier.parameters(), lr=learning_rate)
    if state_dict != None:
        optimizer.load_state_dict(state_dict)
    print('Done!\n')
    return optimizer


# Build criterion
def build_nllloss_criterion():
    print('Building NLLLoss criterion...')
    criterion = nn.NLLLoss()
    print('Done!\n')
    return criterion
    
    
# Evaluate a model: get loss and accuracy
def evaluate(model, criterion, data_loader, device):
    model.eval()

    loss = 0
    accuracy = 0
    
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            logps = model.forward(inputs)
            batch_loss = criterion(logps, labels)

            loss += batch_loss.item()

            ps = torch.exp(logps)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
            
        return loss, accuracy


# Train model, evaluate and print result
def train_model(model, criterion, optimizer, epochs, train_data_loader, validation_data_loader, device, print_every):
    with active_session():    
        print('Training model...\n')       
        model.train()
        
        model.class_to_idx = train_data_loader.dataset.class_to_idx
        
        steps = 0
        for epoch in range(epochs):
            running_loss = 0
            for inputs, labels in train_data_loader:
                steps += 1

                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()

                logps = model.forward(inputs)
                loss = criterion(logps, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

                if steps % print_every == 0:
                    validation_loss, validation_accuracy = evaluate(model, criterion, validation_data_loader, device)
                    
                    print(f"Epoch {epoch+1}/{epochs}.. "
                          f"Train loss: {running_loss/print_every:.3f}.. "
                          f"Validation loss: {validation_loss/len(validation_data_loader):.3f}.. "
                          f"Validation accuracy: {validation_accuracy/len(validation_data_loader):.3f}")

                    running_loss = 0
                    
        model.epochs_done += epochs
        print('Done!\n')       

        
# Save checkpoint of the model
def save_check_point(model, optimizer, check_point_file):
    print('Saving model check point to ' + check_point_file + ' ... ')       
    check_point = {
        'model': {
            'arch': model.arch,
            'hidden_units': model.hidden_units,
            'epochs_done': model.epochs_done,
            'state_dict': model.state_dict(),
            'class_to_idx': model.class_to_idx
         },
        'optimizer': {
            'state_dict': optimizer.state_dict()
        }
    }
    torch.save(check_point, check_point_file)
    print('Done!\n')       

    
# Load model and optimizer from check point
def load_check_point(check_point_file, device):
    
    check_point = torch.load(check_point_file)   
   
    arch = check_point['model']['arch']    
    hidden_units = check_point['model']['hidden_units']    
    epochs_done = check_point['model']['epochs_done']    
    model_state_dict = check_point['model']['state_dict']
    class_to_idx = check_point['model']['class_to_idx']
    
    model = build_model(arch=arch, hidden_units=hidden_units, epochs_done=epochs_done, 
                        state_dict=model_state_dict, class_to_idx=class_to_idx, device=device)
        
    optimizer_state_dict = check_point['optimizer']['state_dict']     
    optimizer = build_adam_optimizer(model, state_dict=optimizer_state_dict)
    
    return model, optimizer
    

# Predict Top K classes and their probabilities
def predict(image_file, model, device, topk=5):
    
    print('Predicting image ' + image_file + ' ...')

    model.eval()
    
    image = process_image(image_file)
    image = image.unsqueeze_(0)
    image = image.float()

    model.to(device)
    image = image.to(device)

    with torch.no_grad():
        logps = model.forward(image)
        
        ps = torch.exp(logps)
        top_ps, top_idx = ps.topk(topk)
        
        class_to_idx = model.class_to_idx
        
        top_classes = []
        for i in top_idx.cpu().numpy()[0]:
            top_classes.append(get_class_from_index(i, class_to_idx))
        
        print('Done!\n')

        return top_ps[0].cpu().numpy(), np.array(top_classes)


 
   