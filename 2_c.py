import argparse
import os
import numpy as np
import random
import torch
from torchvision import models
from torch.optim import SGD
import torchvision.transforms as transforms
from Utils import auto_grad, squeeze, unsqueeze, rescale, gauss_filter
from utils import settings_parser


# todo do i need this class?
class ModelClassVisualization():
    def __init__(self, model, target_class):
        self.model = model
        self.model.eval()
        self.target_class = target_class

        # Process: from a Pil image to a normalized tensor that can be given as input to the model
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            normalize,
            transforms.Lambda(unsqueeze),
            transforms.Lambda(auto_grad)
        ])
        # Process: variant with a gauss filter
        self.transform_gauss = transforms.Compose([
            transforms.Lambda(gauss_filter),
            transforms.ToTensor(),
            normalize,
            transforms.Lambda(unsqueeze),
            transforms.Lambda(auto_grad)
        ])
        # Inverse process: from tensor to a Pil Image
        inv_normalize = transforms.Normalize(
            mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
            std=[1 / 0.229, 1 / 0.224, 1 / 0.225]
        )
        self.inv_transform = transforms.Compose([
            transforms.Lambda(squeeze),
            inv_normalize,
            transforms.Lambda(rescale),
            transforms.ToPILImage()
        ])

        self.gauss_freq = 5
        if not os.path.exists('./model_class_visualization/class_' + str(self.target_class)):
            os.makedirs('./model_class_visualization/class_' + str(self.target_class))

    def run(self, iterations=150, gauss=False):
        # Start with a random image
        pil_image = np.uint8(np.random.uniform(0, 255, (224, 224, 3)))

        initial_learning_rate = 10  # todo chiedere a prof

        for i in range(1, iterations):  # todo chiedere a prof

            # Process image and return tensor variable
            if gauss and i % self.gauss_freq == 0:
                tensor_img = self.transform_gauss(pil_image)
            else:
                tensor_img = self.transform(pil_image)

            optimizer = SGD([tensor_img], lr=initial_learning_rate)
            # Forward pass to get the image classification
            output = self.model(tensor_img)
            class_loss = - output[0, self.target_class]

            if i % 10 == 0:
                print("Iteration: ", i, " loss: ", class_loss)

            self.model.zero_grad()
            # Backward pass
            class_loss.backward()
            # Update the tensor image
            optimizer.step()

            # Recreate the Pil Image
            pil_image = self.inv_transform(tensor_img)

        # Save image
        im_path = './model_class_visualization/class_' + str(self.target_class) + '/' + str(self.target_class)
        if gauss is True:
            im_path += '_gauss'
        im_path += '.png'
        pil_image.save(im_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Get settings
    settings_system = settings_parser.get_settings('System')
    settings_dataset = settings_parser.get_settings('Dataset')

    rootdir = settings_dataset['val_images_path']
    classes = []
    for file in os.listdir(rootdir):
        d = os.path.join(rootdir, file)
        if os.path.isdir(d):
            classes.append(int(file))

    id_class = random.choices(classes, k=1)[0]
    print("Id random class: ", id_class)
    vgg16 = models.vgg16(pretrained=True)

    mcv = ModelClassVisualization(vgg16, id_class)
    mcv.run(gauss=False)

    mcv = ModelClassVisualization(vgg16, id_class)
    mcv.run(gauss=True)