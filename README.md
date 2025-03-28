# SafeSleep AI - Infant Sleep Detection Model

Overview

SafeSleep AI is an advanced AI-powered infant sleep detection model designed to improve sleep safety for babies. Using computer vision and deep learning, the model detects infant sleep positions and identifies potential hazards such as unsafe objects (e.g., toys, blankets) in the crib. The goal is to help parents and baby monitor companies enhance sleep safety.

Features

Sleep Position Classification: Detects various infant sleep positions, including back, stomach, and side sleeping.

Object Detection: Identifies unsafe objects such as loose bedding, toys, and other items that may pose a risk.

Real-Time Processing: Optimized for real-time detection to provide immediate feedback.

Cloud Storage: Supports integration with AWS S3 for storing and retrieving sleep images.

API Integration: Designed as a SaaS product for seamless integration with baby monitor companies.

Model Details

Architecture: YOLOv8 for object detection and classification.

Training Data: Utilizes a dataset of over 10,000 images, including augmented images for improved model robustness.

Training Framework: PyTorch-based implementation.

Annotation Format: YOLO format used for labeled images.

Installation

To run SafeSleep AI locally, follow these steps:

Prerequisites

Python 3.8+

PyTorch

OpenCV

Ultralytics YOLOv8
