import os
from torchvision.models import vgg16
import torch
from PIL import Image
from Utils import unsqueeze, auto_grad, squeeze, rescale
from torchvision.transforms import transforms
import matplotlib.pyplot as plt


if __name__ == '__main__':

    test_img_dir = './dataset/DL_data/test_images'

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transf = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
        transforms.Lambda(unsqueeze),
        transforms.Lambda(auto_grad)
    ])

    model = vgg16(pretrained=True)
    model.eval()

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

            plt.imshow(M, cmap="inferno")
            path = ".\class_saliency_maps/" + img
            plt.savefig(path)
            plt.clf()