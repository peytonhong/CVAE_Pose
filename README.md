# CVAE_Pose
Convolutional (Variational) Autoencoder for object pose estimation (Pytorch version, Under construction)

**FC layer is added on latent vectors (z) to estimate rotation matrix.**

**Pointcloud 3d model is rendered along with the estimated rotation angle. The rendered (and rotated) images are then evaluated with the output of decoder using MSE loss.**

### Network Architecture
![Architecture](https://github.com/peytonhong/CVAE_Pose/blob/master/docs/structure(silhouette_matching).png)

* Input image
  * LineMod dataset (duck) with various augmentation. (Random bubble, Random background, Random scale, Gamma correction)
* Output image
  * Estimated mask (Silhouette)
* Output (FC)
  * Rotation matrix R (3x3)
* Pose refinement
  * Pytorch3D Silhouette Matching

### Result images
![Reconstructed image_ae](https://github.com/peytonhong/CVAE_Pose/blob/master/docs/image_at_epoch_9999(silhouette).png)

##### Reconstructed Image Description
* Left: Training results
* Right: Test results.
* 1st row: Input images
* 2nd row: Ground truth
* 3rd row: Reconstructed images

### Pose Refinement using Pytorch3D
![pose refinement](https://github.com/peytonhong/CVAE_Pose/blob/master/docs/obj_optimization_demo.gif)

### Rotation loss comparison
* Before pose refinement: 0.170
* After pose refinement: TODO

