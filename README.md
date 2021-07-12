# Pytorch-Heat-Method
A (crude) differentiable implementation of [The Heat Method for Distance Computation (Geodesics in Heat), by Crane Et Al.](https://www.cs.cmu.edu/~kmcrane/Projects/HeatMethod/)

I didn't wrap it up into a proper pytorch method, because I'm lazy and also don't fully understand all of that junk. But this works, and seems to generate valid gradients during backprop. 

## Dependencies
  - Pytorch
  - Kornia (optional, used in the demo for edge extraction)

The `heat_method` function performs best when fed a single channel image tensor, where the edges to find the distance to, are a constant value, and everything else is set to 0. 

Example Input:

![Input Edges](https://github.com/jakericedesigns/Pytorch-Heat-Method/blob/main/assets/edges_out.png)

Example Output:

![Output Distance](https://github.com/jakericedesigns/Pytorch-Heat-Method/blob/main/assets/depth_out.png)
