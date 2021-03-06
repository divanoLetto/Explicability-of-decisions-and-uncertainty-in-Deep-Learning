import argparse
import os
from torchvision.models import vgg16
from torchvision.transforms import transforms
from PIL import Image
from Utils import unsqueeze
import torch
from utils import settings_parser


if __name__ == '__main__':
    # Get settings
    parser = argparse.ArgumentParser()
    settings_system = settings_parser.get_settings('System')
    settings_dataset = settings_parser.get_settings('Dataset')

    test_img_dir = settings_dataset['test_images_path']
    name_class_file = open(settings_dataset['synset_words_path'], 'r')

    # Get a dictionary that maps class idx to their class names
    Lines = name_class_file.readlines()
    dict_class_id_name = {}
    count = 0
    for line in Lines:
        dict_class_id_name[count] = line.replace("\n", "")
        count += 1

    vgg16 = vgg16(pretrained=True)
    vgg16.eval()

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transf = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])
    softmax = torch.nn.Softmax(dim=1)

    for img in os.listdir(test_img_dir):

        if img.endswith(".png") or img.endswith(".JPEG"):
            d = os.path.join(test_img_dir, img)

            # Get pil image and tensor image
            pil_image = Image.open(d)
            tensor_image = transf(pil_image)
            tensor_image = unsqueeze(tensor_image)

            # compute output
            output = vgg16(tensor_image)
            # Softmax on output and store in a list
            norm_output = softmax(output).tolist()[0]

            # Sort the classes prediction based on the output probability
            indices = [i for i in range(0,1000)]
            sorted_value_class = sorted(zip(norm_output, indices), key=lambda x: x[0], reverse=True)

            # print the 5 most probable class for each image
            print("Image: ", img," :")
            for i in range(0,5):
                id_class = sorted_value_class[i][1]
                value_class = sorted_value_class[i][0]
                print("   class: ", id_class, "-", dict_class_id_name[id_class],", score: ", value_class)
