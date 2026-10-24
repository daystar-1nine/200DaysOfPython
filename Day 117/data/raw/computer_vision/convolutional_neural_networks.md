# Convolutional Neural Networks (CNNs) and Spatial Feature Hierarchies

Convolutional Neural Networks process visual grid data by applying localized 2D filter kernels across spatial dimensions. Convolutions preserve 2D spatial topology and leverage weight sharing, drastically reducing parameter counts compared to fully connected networks. Deeper layers capture increasingly abstract hierarchical features, progressing from low-level edges and textures to complex object parts.

Pooling layers (such as Max Pooling and Average Pooling) downsample spatial feature maps, conferring translation invariance and expanding the effective receptive field of subsequent convolutional layers. Strided convolutions can also perform spatial downsampling while learning parameterized decimation kernels.
