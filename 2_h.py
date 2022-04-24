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


def change_drop_out_ratio(model, ratio=0.5):
    for m in model.modules():
        if isinstance(m, torch.nn.Dropout):
            m.p = ratio


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    settings_system = settings_parser.get_settings('System')
    settings_dataset = settings_parser.get_settings('Dataset')
    test_img_dir = settings_dataset['test_images_path']
    true_labels = [795, 23, 270, 858, 756]

    iterations = 100
    model = vgg16(pretrained=True)
    enable_dropout(model)
    n_classes = 1000
    num_images = 5

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
    for it in range(iterations):
        output = model(tensor_images)
        norm_output = softmax(output)
        dropout_outputs = np.vstack((dropout_outputs, norm_output.detach().numpy()[np.newaxis, :, :]))
        # dropout predictions - shape (forward_passes, n_samples, n_classes)

    # Calculating mean across multiple MCD forward passes
    mean = np.mean(dropout_outputs, axis=0)  # shape (n_samples, n_classes)

    predicted_class = np.argmax(mean, axis=1)
    dropout_predictios = np.argmax(dropout_outputs, axis=2)

    num_class_pred = dropout_predictios.transpose().tolist() # shape (num_image, num_exp)
    counts = []
    for i in range(num_images):
        count = 0
        for j in range(iterations):
            if num_class_pred[i][j] == true_labels[i]:
                count += 1
        counts.append(count / iterations)
    counts = np.array(counts)
    counts = counts

    # Calculating variance across multiple MCD forward passes
    variance_drop_outputs = np.var(dropout_outputs, axis=0)  # shape (n_samples, n_classes)

    variance_predicted_class = []
    num_images = mean.shape[0]
    for i in range(0, num_images):
        pred_class = predicted_class[i]
        pred_class_var = variance_drop_outputs[i][pred_class]
        variance_predicted_class.append(pred_class_var)

    print("(Human-assigned) correct classes: ")
    print(true_labels)
    print("Classes most probable for each image: ")
    print(predicted_class)
    print("Perc correct classification: ")
    print(counts)
    print("Variance of each image of the testset: ")
    print(variance_predicted_class)
