# Point-Based Object Detection with Deep Learning
This project implements a point-based object detector from scratch using deep learning. The goal is to detect and localize objects such as karts, bombs/projectiles, and pickup items from images. Instead of traditional bounding box detection, the model predicts heatmaps of object centers, enabling efficient and precise object localization.

# Key Achievements
- Designed a CNN model to predict heatmaps representing object centers.
- Implemented peak extraction using max pooling and torch.topk for local maxima detection.
- Optimized training using BCEWithLogitsLoss, focal loss, and data augmentation.
- Integrated PyTorch and GPU acceleration for efficient processing.
- Evaluated performance using Average Precision (AP) based on object center proximity and bounding box overlap.
- Developed a full object detector predicting both object centers and sizes.

# Applications
- Efficient real-time object detection for autonomous navigation, robotics, and gaming.
- Adaptable methodology for medical imaging, surveillance, and anomaly detection.
