# Source attribution

This repository includes a modified copy of the Ultralytics Python package.
The original license text and existing source notices are preserved.
The included source reports version 8.4.24. The original working directory
was an unpacked source tree; an upstream Git commit ID was not available.

Project source: https://github.com/ultralytics/ultralytics

ORMSFD-specific changes include MSFEM, the shared image-predicted overlap prior,
sequential feature fusion, parser registration, custom bounding-box and repulsion
losses, and auxiliary box-overlap supervision.
The modified package uses the `ultralytics` import path.

The released model configuration follows the current source implementation.
The compatible pretrained neck layers and Detect head are mapped explicitly
to the revised model indices. Ground-truth boxes supervise the auxiliary prior
loss during training. The feature module uses image-predicted priors during
training and inference.
