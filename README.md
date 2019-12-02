## CVAE_Pose
Convolutional Variational Autoencoder for object pose estimation (Under construction)

**FC layer is added on latent vectors (z) to estimate rotation angle.**

* Input image
  * Rectangles with various rotation.
* Output image
  * Reconstructed rectangle images
* Output (FC)
  * Rotation angle (pose)

# Reconstructed image result
![Reconstructed image](https://github.com/peytonhong/CVAE_Pose/blob/master/cvae.gif)

# Latent space walking
![Latent space walking](https://github.com/peytonhong/CVAE_Pose/blob/master/results/PMLR_epoch_300.jpg)

# Latent space to angle
![latent space to angle](https://github.com/peytonhong/CVAE_Pose/blob/master/results/scattered_z.png)

![latent space to angle by angle](https://github.com/peytonhong/CVAE_Pose/blob/master/results/scattered_z_by_angle.png)
