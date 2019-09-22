import argparse
import torch
from torchvision import datasets, transforms
import json
from PIL import Image

# Parsing command-line arguments for train.py
def parse_train_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(action='store', dest='data_dir', help='Root directory for data')
    parser.add_argument('-s', '--save_dir', action='store', dest='save_dir', default='.', help='Directory for saving the model checkpoint')
    parser.add_argument('-a', '--arch', action='store', dest='arch', default='vgg16', help='Architecture of the model: vgg16 or resnet18')
    parser.add_argument('-l', '--learning_rate', action='store', type=float, dest='learning_rate', default=0.001, help='Learning rate of the optimizer')
    parser.add_argument('-u', '--hidden_units', action='store', type=int, dest='hidden_units', default=1024, help='Hidden units of the classifier')
    parser.add_argument('-e', '--epochs', action='store', type=int, dest='epochs', default=3, help='Number of epochs to train')
    parser.add_argument('-g', '--gpu', action='store_true', dest='gpu', help='Use GPU')

    return parser.parse_args()


# Parsing command-line arguments for predict.py
def parse_predict_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(action='store', dest='image_file', help='Path and file name for the image')
    parser.add_argument(action='store', dest='check_point_file', default='./check_point.pth', help='Path and file name for the model check point. E.g: check_point_vgg16.pth or check_point_resnet18.pth')
    parser.add_argument('-k', '--top_k', type=int, action='store', dest='topk', default=1, help='Top K most likely classes')
    parser.add_argument('-c', '--category_names', action='store', dest='cat_to_name_file', default='./cat_to_name.json', help='Path and file name for JSON file for category to name')
    parser.add_argument('-g', '--gpu', action='store_true', dest='gpu', help='Use GPU')

    return parser.parse_args()


# Get data directories
def data_dir(root_data_dir, sub_dir):
    return root_data_dir + '/' + sub_dir


# Load data
def load_data(root_data_dir, sub_dir):
    train_data_transforms = transforms.Compose([
        transforms.RandomRotation(30),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    validation_test_data_transforms = transforms.Compose([
        transforms.Resize(255),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    full_dir = data_dir(root_data_dir, sub_dir)
    
    print('Loading ' + sub_dir + ' data from ' + full_dir + ' ... ')
    
    if sub_dir == 'train':
        dataset = datasets.ImageFolder(full_dir, transform=train_data_transforms)
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)
    else:
        dataset = datasets.ImageFolder(full_dir, transform=validation_test_data_transforms)
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=32)
    
    print('Done!\n')
    return data_loader
        
        
# Device
def device(gpu):
    d = 'cuda' if gpu else 'cpu'
    print('Device used: ' + d + '\n')
    return torch.device(d)


# Return class from index
def get_class_from_index(idx, class_to_idx):
    for c, i in class_to_idx.items():
        if i == idx:
            return c


# Load and process the image
def process_image(image_file):
    processed_image = Image.open(image_file)
   
    transform = transforms.Compose([
        transforms.Resize(255),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    processed_image = transform(processed_image)
    
    return processed_image

                
# Print Top K result
def print_predict_result(top_ps, top_classes, cat_to_name_file):
    with open(cat_to_name_file, 'r') as f:
        cat_to_name = json.load(f)
        print(f"{'Name':{20}} | Probability")
        for i in range(top_classes.size):
            print(f"{cat_to_name[top_classes[i]]:{20}} | "
                  f"{top_ps[i]:.3f}")

    

    
    
    
    
    
