import os
from PIL import Image
from torchvision.transforms import transforms


if __name__ == '__main__':
    # Just resize and crop the test images, useful just for the report

    test_img_dir = "DL_data/test_images/"

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transf = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
    ])

    for img in os.listdir(test_img_dir):

        if img.endswith(".png") or img.endswith(".JPEG"):
            d = os.path.join(test_img_dir, img)

            path = "cropped_test_images/" + img

            pil_image = Image.open(d)
            cropped_pil_image = transf(pil_image)
            cropped_pil_image.save(path)

