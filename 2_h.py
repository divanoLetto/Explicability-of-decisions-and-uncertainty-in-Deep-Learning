from torchvision.models import vgg16
import torch
from torchvision.transforms import transforms
from Utils import unsqueeze, auto_grad, squeeze, rescale
import os
from PIL import Image
import numpy as np


def enable_dropout(m):
    for module in m.modules():
        if isinstance(module, torch.nn.Dropout):
            module.train()


def change_drop_out_ratio(model, ratio=0.5):
    for m in model.modules():
        if isinstance(m, torch.nn.Dropout):
            m.p = ratio


if __name__ == '__main__':
    test_img_dir = './dataset/DL_data/test_images'
    iterations = 10  # todo magic number
    models = []
    n_classes = 1000

    ratio = 0.0
    # Create 6 model with dropout enable on test and dropout ratio = 0, 0.2, 0.4, 0.6, 0.8, 1
    for i in range(0,6):
        model_i = vgg16(pretrained=True)
        model_i.eval()
        change_drop_out_ratio(model_i, ratio=ratio)
        ratio += 0.2
        enable_dropout(model_i)
        models.append(model_i)

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

    # For each model with different dropout rate
    for mod in models:

        dropout_predictions = np.empty((0, 5, n_classes))  # todo fix 5=len images
        # Make n experiments and calculate variance
        for it in range(iterations):
            output = mod(tensor_images)
            predictions = softmax(output)
            dropout_predictions = np.vstack((dropout_predictions, predictions.detach().numpy()[np.newaxis, :, :]))
            # dropout predictions - shape (forward_passes, n_samples, n_classes)

        # Calculating mean across multiple MCD forward passes
        mean = np.mean(dropout_predictions, axis=0)  # shape (n_samples, n_classes)

        # Calculating variance across multiple MCD forward passes
        variance = np.var(dropout_predictions, axis=0)  # shape (n_samples, n_classes)

    # todo experiment with conv dropout
    # model_conv_linear_drop = vgg16(pretrained=True)
    # feats_list = list(model_no_drop.classifier)  #  model.features
    # new_feats_list = []
    # for feat in feats_list:
    #     new_feats_list.append(feat)
    #     if isinstance(feat, torch.nn.Linear):
    #         new_feats_list.append(torch.nn.Dropout(p=0.0, inplace=True))
    #
    # feats_list = list(model_no_drop.classifier)  # model.features
    # new_feats_list = []
    # for feat in feats_list:
    #     new_feats_list.append(feat)
    #     if isinstance(feat, torch.nn.Linear):
    #         new_feats_list.append(torch.nn.Dropout(p=0.0, inplace=True))
    #
    # # modify convolution layers
    # model_no_drop.features = torch.nn.Sequential(*new_feats_list)
    # print(model_no_drop)