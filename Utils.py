from torch.autograd import Variable
import torch
from PIL import Image, ImageFilter


def squeeze(img_tensor):
    ten = img_tensor.squeeze(0)
    return ten


def unsqueeze(img_tensor):
    """
        Add a channel at the beginning of the tensor
        Args:
            img_tensor: The image tensor (C, W, H)
        returns:
            ten: The new tensor (1, C, W, H)
    """
    ten = img_tensor.unsqueeze_(0)
    return ten


def auto_grad(img_tensor):
    """
        To let PyTorch automatically track and calculate gradients for the tensor.
    """
    ten = Variable(img_tensor, requires_grad=True)
    return ten


def rescale(img_tensor):
    """
            To let PyTorch automatically track and calculate gradients for the tensor.
        """
    ten = torch.clamp(img_tensor, min=0, max=1)
    ten = torch.round(ten * 255)
    ten = torch.tensor(ten, dtype=torch.uint8)
    return ten


def gauss_filter(pil_image):
    gauss_img = pil_image.filter(ImageFilter.GaussianBlur())  # todo chiedere a prof il radius di gausss blur = 2
    return gauss_img

