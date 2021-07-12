import torch
from torch import nn, optim

# A crude pytorch implementation of The Heat Method for Distance Computation (Geodesics in Heat), by Crane Et Al.
# https://www.cs.cmu.edu/~kmcrane/Projects/HeatMethod/

# I find the easiest reference for finite differences related stuff is this:
# https://developer.download.nvidia.com/books/HTML/gpugems/gpugems_ch38.html
# Its easy to find since u just have to google gpu gems fluids :) 

def solve_poisson(img, iters=1000):
    # Solve Δx=b, a regular poisson eq.
    # A=Δ
    # A = D + L + U (D is the diagonal of the laplace matrix, L and U are the lower and upper parts)

    # solving for x (jacobi method): x_1 = D^-1 * (b - (L + U) * x_0)

    # which ultimately takes the form of this, when plugging our stuff in:
    #(solution - convolve(x, LU)) / -4 

    LU = torch.tensor([[[[0, 1., 0],
                         [1, 0,  1],
                         [0, 1,  0]]]], dtype=img.dtype).to(img.device)


    BC = nn.ReplicationPad2d(1) #boundary condition
    solution = img
    for i in range(iters):
            solution = (img - nn.functional.conv2d(BC(solution), LU)) / -4.0
    return solution

def screened_poisson(img, timestep=.1, mass=.01, iters=1000):
    # Solve (M - t*Δ)x=b, a screened poisson eq.
    # Set A = (M - t*Δ)
    # A = D + L + U (D is the diagonal of the laplace matrix, L and U are the lower and upper parts)

    # solving for x (jacobi method): x_1 = D^-1 * (b - (L + U) * x_0)

    # M is a mass matrix, which has only diagonal entries.

    # Since M - t*A  is our right hand side
    # M - D*t = M - (D * t),
    # M - (L + U)*t =   0 - (L + U) * t 

    
    # which ultimately takes the form of this, when plugging our stuff in:
    # (solution - convolve(x, -LU * t)) / (M - (-4 * t))

    LU = torch.tensor([[[[0, -1., 0],
                         [-1, 0, -1],
                         [0, -1,  0]]]], dtype=img.dtype).to(img.device)


    BC = nn.ReplicationPad2d(1)  #boundary condition
    solution = img
    for i in range(iters):
            solution = (img - nn.functional.conv2d(BC(solution), LU * timestep)) / (mass + 4.0 * timestep)   
    return solution    


def finite_diff_grad(img):
    #expects a 1d input: B,1,H,W
    kernel_x = torch.tensor([[[[0, 0., 0],
                               [1, 0, -1],
                               [0, 0,  0]]]], dtype=img.dtype).to(img.device)

    kernel_y= torch.tensor([[[[0, 1,  0],
                              [0, 0,  0],
                              [0, -1, 0]]]], dtype=img.dtype).to(img.device)
    div_x = nn.functional.conv2d(img, kernel_x)
    div_y = nn.functional.conv2d(img, kernel_y)
    return torch.cat((div_x, div_y), 1) / 2.0

def finite_diff_div(grad):
    #expects a 2d input B,2,H,W
    kernel_x = torch.tensor([[[[0, 0., 0],
                               [1, 0, -1],
                               [0, 0,  0]]]], dtype=grad.dtype).to(grad.device)
    kernel_y= torch.tensor([[[[0, 1,  0],
                              [0, 0,  0],
                              [0, -1, 0]]]], dtype=grad.dtype).to(grad.device)

    #there's for sure a better way to do this indexing stuff, but i'm bad at numpy style indexing
    div_x = nn.functional.conv2d(grad[:,0:1,...], kernel_x)
    div_y = nn.functional.conv2d(grad[:,1:,...], kernel_y)
    return (div_x + div_y) / 4.0

def heat_method(image, timestep=1.0, mass=.01, iters_diffusion=500, iters_poisson=1000):
    heat = screened_poisson(image, timestep, mass, iters_diffusion) #diffusion
    grad = finite_diff_grad(heat) * -1 #inverted gradient
    grad = grad / grad.norm(dim=1) #normalize gradient
    div = finite_diff_div(grad)
    distance = solve_poisson(div, iters_poisson)
    return distance
