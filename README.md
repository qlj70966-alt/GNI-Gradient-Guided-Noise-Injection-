# GNI-Gradient-Guided-Noise-Injection-

GNI: Gradient-Guided Noise Injection – a lightweight adaptive regularization for fine-tuning PLMs. Noise intensity is dynamically scaled per layer based on gradient norms: β = α / (1 + ||∇W||_F). Improves generalization and stability. Includes PyTorch implementation for GLUE tasks. Paper results show consistent gains over baselines. 

For more details, please refer to our paper: Gradient-Guided Layerwise Adaptive Noise Injection for Pre-trained Language Model Fine-tuning. DOI: https://doi.org/10.21203/rs.3.rs-9574488/v1
