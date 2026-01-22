# Models Directory

This directory stores the pre-trained model weights used for seismic event detection.

## Structure
- `EQCCT/`: Contains TensorFlow model checkpoints (`.h5` files) for the EQCCT architecture.
    - `test_trainer_021.h5`
    - `test_trainer_024.h5`

SeisBench models are typically downloaded automatically by the SeisBench library into its own cache, but custom weights can also be integrated here.
