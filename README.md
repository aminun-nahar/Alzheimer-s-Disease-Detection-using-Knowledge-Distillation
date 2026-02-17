# Alzheimer-s-Disease-Detection-using-Knowledge-Distillation
Ongoing Project
📌 Overview

Alzheimer’s Disease (AD) is a progressive neurodegenerative disorder that affects memory and cognitive function. Early and accurate detection is crucial for timely treatment and clinical decision-making.

This project implements Knowledge Distillation (KD) in TensorFlow/Keras to transfer knowledge from a high-capacity teacher model to a lightweight student model, enabling efficient and accurate Alzheimer’s disease classification.

# 🎯 Objective

Improve student model performance using teacher supervision

Reduce computational complexity for deployment

Maintain high classification accuracy

Enable efficient medical AI systems

# 🧠 Knowledge Distillation Framework

A custom Distiller class is implemented by extending keras.Model.

The teacher model is pre-trained and frozen.

The student model learns from:

Ground-truth labels (hard targets)

Soft probability outputs from the teacher (soft targets)

# 🔬 Loss Function

The total training loss is defined as:

$\mathcal{L}_{total}=\alpha \, \mathcal{L}_{CE}(y, y_s)+(1 - \alpha)\,\mathcal{L}_{KL}\left(\text{Softmax}\left(\frac{z_t}{T}\right),\text{Softmax}\left(\frac{z_s}{T}\right)\right)$



Where:

$\mathcal{L}_{CE}$ → Categorical Cross-Entropy loss

$\mathcal{L}_{KL}$ → KL Divergence loss

$z_t$ → Teacher logits

$z_s$ → Student logits

$T = 5.0$ → Temperature parameter

$\alpha = 0.5$ → Loss balancing factor

# ⚙️ Training Process

Forward pass through teacher (no gradient updates).

Forward pass through student.

Compute:

Hard label loss (Cross-Entropy)

Soft label loss (KL Divergence with temperature scaling)

Combine losses using weighted sum.

Backpropagate and update student weights only.

# 📊 Evaluation Metrics

The following metrics are tracked:

Accuracy

Precision

Recall

AUC

Student Loss

Distillation Loss

Total Loss
