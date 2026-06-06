# GNI-Gradient-Guided-Noise-Injection-

GNI: Gradient-Guided Noise Injection – a lightweight adaptive regularization for fine-tuning PLMs. Noise intensity is dynamically scaled per layer based on gradient norms: β = α / (1 + ||∇W||_F). Improves generalization and stability. Includes PyTorch implementation for GLUE tasks. Paper results show consistent gains over baselines.
