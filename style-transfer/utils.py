import torch
import matplotlib.pyplot as plt
from torchvision import transforms


def transfer_style(transnet, content, content_image, style_image):
    transnet.eval()
    with torch.no_grad():
        output = draw(transnet(content), content_image, style_image)
    return output


# Pre-processing
def prep(image, device, size=256, normalize=True):
    # ImageNet statistics
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    resize = transforms.Compose([transforms.Resize(size),
                                 transforms.CenterCrop(size)])
    image = resize(image.convert('RGB'))
    if normalize:
        norm = transforms.Compose([transforms.ToTensor(),
                                   transforms.Normalize(mean, std)])
        return norm(image).unsqueeze(0).to(device)
    else:
        return image


# Post-processing
def post(tensor):
    # ImageNet statistics
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    mean, std = torch.tensor(mean).view(3, 1, 1), torch.tensor(std).view(3, 1, 1)
    tensor = transforms.Lambda(lambda x: x * std + mean)(tensor.cpu().clone().squeeze(0))
    return transforms.ToPILImage()(tensor.clamp_(0, 1))




# Draw content, style and output images
def draw(input, content_image, style_image):
    output = post(input)

    plt.figure(figsize=(18, 6))

    ax1 = plt.subplot(1, 3, 1)
    ax1.imshow(content_image)
    ax1.axis('off')
    ax1.set_title('Content Image')

    ax2 = plt.subplot(1, 3, 2)
    ax2.imshow(style_image)
    ax2.axis('off')
    ax2.set_title('Style Image')

    ax3 = plt.subplot(1, 3, 3)
    ax3.imshow(output)
    ax3.axis('off')
    ax3.set_title('Output Image')

    plt.savefig('outputs/style_transfer.jpg')
    return output