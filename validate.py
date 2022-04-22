import numpy
from torchvision.models import vgg16
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import argparse
import time
from utils import settings_parser
from pathlib import Path
import numpy as np
from scipy.special import softmax
from sklearn.metrics import accuracy_score


def onehot_encoding(idx, nume_elements):
    vec = numpy.zeros(nume_elements)
    vec[idx] = 1
    return vec


def cross_entropy(y, y_pre):
    y_pre = np.array(y_pre)
    y_pre = softmax(y_pre, axis=1)
    loss = -np.sum(y*np.log(y_pre))
    return loss/float(y_pre.shape[0])


def validate(val_loader, model, args):
    batch_time = []
    losses = []
    top1 = []

    # Set to evaluate mode
    model.eval()
    inv_map = {v: k for k, v in val_loader.dataset.class_to_idx.items()}

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):

            ground_truth = []
            for elem in np.array(target):
                value = int(inv_map[elem])
                encoded_value = onehot_encoding(value, 1000)
                ground_truth.append(encoded_value)
            ground_truth = np.array(ground_truth)

            # compute output
            output = model(images)

            # measure loss
            loss = cross_entropy(ground_truth, output)
            # measure accuracy
            acc = accuracy_score(np.argmax(ground_truth, axis=1), np.argmax(output, axis=1))

            # todo controllare quando l'accuracy è zero
            losses.append(loss)
            top1.append(acc)

            # measure elapsed time
            batch_time.append(time.time() - end)
            end = time.time()

            if i % args['print_freq'] == 0:
                print("Accuracy: ", acc, " ,loss: ", loss)

    return np.average(batch_time), np.average(losses), np.average(top1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    # Get settings
    settings_system = settings_parser.get_settings('System')
    settings_dataset = settings_parser.get_settings('Dataset')
    settings_model = settings_parser.get_settings('Model')

    print_freq = int(settings_system['print_freq'])
    batch_size = int(settings_model['batch_size'])
    weights_save_path = settings_model['weights_save_path']
    val_images_path = settings_dataset['val_images_path']

    # load ImageNet VGG16 Weights
    if Path(weights_save_path).is_file():
        VGG16 = vgg16()
        VGG16.load_state_dict(torch.load(weights_save_path))
    # otherwise download ImageNet VGG16 Weights and save them
    else:
        VGG16 = vgg16(pretrained=True)
        torch.save(VGG16.state_dict(), weights_save_path)
    # pre-process the dataset
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transf = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])
    # create DataLoader for the validation set
    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(val_images_path, transf),
        batch_size=batch_size
    )

    model = VGG16
    print(VGG16)

    args = {'print_freq': print_freq}

    batch_time_avg, losses_avg, top1_avg = validate(val_loader, model, args)
    print("Average batch time: ", batch_time_avg)
    print("Average loss: ", losses_avg)
    print("Average top1 acc: ", top1_avg)