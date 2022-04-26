import argparse
from torchvision.models import vgg16
import torch
from torchvision.transforms import transforms
import os
from PIL import Image
import numpy as np
from utils import settings_parser


def enable_dropout(m):
    for module in m.modules():
        if isinstance(module, torch.nn.Dropout):
            module.train()


if __name__ == '__main__':
    # Get settings
    parser = argparse.ArgumentParser()
    settings_system = settings_parser.get_settings('System')
    settings_dataset = settings_parser.get_settings('Dataset')
    test_img_dir = settings_dataset['test_images_path']

    # Get model and set dropout layers to active
    model = vgg16(pretrained=True)
    model.eval()
    enable_dropout(model)
    n_classes = 1000
    num_images = 5
    num_exp = 100

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transf = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])
    softmax = torch.nn.Softmax(dim=1)

    list_of_tensors = []
    for img in os.listdir(test_img_dir):

        if img.endswith(".png") or img.endswith(".JPEG"):
            d = os.path.join(test_img_dir, img)

            pil_image = Image.open(d)
            tensor_image = transf(pil_image)
            list_of_tensors.append(tensor_image)

    tensor_images = torch.stack(list_of_tensors)

    dropout_outputs = np.empty((0, num_images, n_classes))
    # Make n experiments and calculate variance
    for it in range(num_exp):
        output = model(tensor_images)
        norm_output = softmax(output)
        dropout_outputs = np.vstack((dropout_outputs, norm_output.detach().numpy()[np.newaxis, :, :]))  # shape (num_exp, n_samples, n_classes)

    # Calculating mean across multiple MCD forward passes
    mean = np.mean(dropout_outputs, axis=0)  # shape (n_samples, n_classes)
    variance = np.var(dropout_outputs, axis=0)  # shape (n_samples, n_classes)

    predicted_class = np.argmax(mean, axis=1)
    dropout_predictios = np.argmax(dropout_outputs, axis=2)

    # Calculating variance across multiple MCD forward passes
    variance_mean = np.mean(variance, axis=1)

    variance_predicted_class = []
    num_images = mean.shape[0]
    for i in range(0, num_images):
        pred_class = predicted_class[i]
        pred_class_var = variance[i][pred_class]
        variance_predicted_class.append(pred_class_var)

    print("Classes most probable for each image: ")
    print(predicted_class)
    print("Variance of each image of the testset for the most probable class: ")
    print(variance_predicted_class)
    print("Mean variance of each image of the testset for all the classes: ")
    print(variance_mean)