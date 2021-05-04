from PIL import Image

import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.use('tkagg')

from torch.autograd import Variable
import torch
from ImageTransformationNN import ImageTransformationNN

transformer = ImageTransformationNN()
import argparse
from utils import preprocess
import numpy as np



def load_rgb_img_tensor(filename:str, dim:int=None,scale:int=None) -> torch.Tensor:
    img = Image.open(filename).convert('RGB')
    # accept dim or scale but but not both, priortize dim
    if dim:
        img = img.resize((dim, dim), Image.ANTIALIAS)
    elif scale:
        img = img.resize((int(img.size[0] / scale), int(img.size[1] / scale)), Image.ANTIALIAS)
    
    img = np.array(img).transpose(2, 0, 1)
    img = torch.from_numpy(img).float()
    return img


def main():

    parser = argparse.ArgumentParser(description='Arguments for testing the network.')

    parser.add_argument('--image', type=str,
                    help='Path to content image')  
                    
    parser.add_argument('--model', type=str,
                    help="Path to model's weights")  

    args = parser.parse_args()


    transformer.load_state_dict(torch.load(args.model, map_location=torch.device('cpu')))

    content = load_rgb_img_tensor(args.image).unsqueeze(0)
    content = content
    content = Variable(preprocess(content), requires_grad=True)
    output = transformer(content)
    img = output.clone().clamp(0, 255).detach().numpy()
    img = img[0].transpose(1, 2, 0).astype('uint8')
    img = Image.fromarray(img)

    f = plt.figure()
    f.set_figwidth(10)
    f.set_figheight(15)
    plt.imshow(img, vmin=0, vmax=256)
    plt.show()



if __name__ == "__main__":
    main()