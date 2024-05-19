import torch
from PIL import Image
from torchvision import transforms
import numpy as np

from model import efficientnetv2_s as create_model


def get_deep_feature():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    img_size = {"s": [300, 384],  # train_size, val_size
                "m": [384, 480],
                "l": [384, 480]}
    num_model = "s"

    data_transform = transforms.Compose(
        [transforms.Resize(img_size[num_model][1]),
         transforms.CenterCrop(img_size[num_model][1]),
         transforms.ToTensor(),
         transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
    # create model
    model = create_model(num_classes=20).to(device)
    # load model weights
    model_weight_path = "model/model-99.pth"
    model.load_state_dict(torch.load(model_weight_path, map_location=device))
    model.eval()

    # 加载图片
    img_path = "temp/logmel.png"
    img = Image.open(img_path).convert('RGB')
    # plt.imshow(img)
    # [N, C, H, W]
    img = data_transform(img)
    # expand batch dimension
    img = torch.unsqueeze(img, dim=0)
    
    with torch.no_grad():

        output,feature = model(img.to(device))
        feature1 =  np.squeeze(feature).cpu().numpy()
        feature1 = feature1.reshape(-1,1)
        feature1 = feature1.T

    return feature1


