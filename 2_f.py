import argparse
import os
from torchvision.models import vgg16
import torch
from PIL import Image
from Utils import unsqueeze, auto_grad, squeeze, rescale
from torchvision.transforms import transforms
import matplotlib.pyplot as plt
from utils import settings_parser
import numpy as np


if __name__ == '__main__':
    # Get settings
    parser = argparse.ArgumentParser()
    settings_dataset = settings_parser.get_settings('Dataset')

    test_img_dir = settings_dataset['test_images_path']

    # Process image function
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transf = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
        transforms.Lambda(unsqueeze),
        transforms.Lambda(auto_grad)
    ])
    # Get the model
    model = vgg16(pretrained=True)
    model.eval()

    list_relu_forward = []
    # Define new forward and backward pass for the ReLu layers
    def guided_relu_forward(module, ten_in, ten_out):
        # save the outputs
        list_relu_forward.append(ten_out)

    def guided_relu_backward(module, grad_in, grad_out):
        # Get last forward output
        forward_output = list_relu_forward.pop()
        # Set to zero the elements of the backwards pass < 0
        guided_grad_out = torch.clamp(grad_in[0], min=0.0)
        # Remember the forward stored pass and set to zero the elements that were negative
        forward_output[forward_output > 0] = 1
        guided_grad_out = forward_output * guided_grad_out
        return (guided_grad_out,)

    # Modify all Relu to apply guided backpropagation
    for i, module in enumerate(model.modules()):
        if isinstance(module, torch.nn.ReLU):
            module.register_forward_hook(guided_relu_forward)
            module.register_backward_hook(guided_relu_backward)

    for img in os.listdir(test_img_dir):

        if img.endswith(".png") or img.endswith(".JPEG"):
            d = os.path.join(test_img_dir, img)

            pil_image = Image.open(d)
            tensor_image = transf(pil_image)

            output = model(tensor_image)
            class_loss, indices = torch.max(output, 1)
            # Get the derivative of the score with respect to the image
            class_loss.backward()
            grad = tensor_image.grad[0]

            # M is equal to max of the abs values in the channel axis for every pixel
            M, _ = torch.max(torch.abs(grad), dim=0)
            # Normalize to [0,1]
            M = (M - M.min()) / (M.max() - M.min())

            # Get the color map
            cm = plt.get_cmap('inferno')
            # Apply the colormap like a function to any array:
            colored_image = cm(M)
            path = ".\guid_back_class_saliency_maps/" + img

            # But we want to convert to RGB in uint8 and save it:
            Image.fromarray((colored_image[:, :, :3] * 255).astype(np.uint8)).save(path)


