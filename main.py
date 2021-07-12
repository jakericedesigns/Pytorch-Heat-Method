import torch
from torchvision.transforms import functional as TF
import os
from PIL import Image
import numpy as np
import kornia
import heatmethod as heat


def fitrange(x):
    c_max = torch.max(x)
    c_min = torch.min(x)

    return (x - c_min) / (c_max - c_min)

if __name__ == "__main__":
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)

    #change me :)
    init_image = "assets/city_depth.png"


    pil_image = Image.open(init_image).convert('RGB')
    image = TF.to_tensor(pil_image).to(device).unsqueeze(0)

    #extract edges from our input images
    print("Extracting edges from input picture")
    edges = kornia.filters.canny(image)[1]


    #Genereate the depth map to our input edges
    print("Generating distance transform to the extracted edges")
    depth_map = heat.heat_method(edges, timestep=0.1, mass=.01, iters_diffusion=500, iters_poisson=1000)


    #Save shit out
    print("Saving images")
    TF.to_pil_image(edges[0].cpu()).save('assets/edges_out.png')
    TF.to_pil_image(fitrange(depth_map)[0].cpu()).save('assets/depth_out.png')
