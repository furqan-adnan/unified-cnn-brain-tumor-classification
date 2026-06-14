#  Unified CNN Approach for Multi-Class MRI Brain Tumor Classification

An advanced, end-to-end Deep Learning repository utilizing optimized **Convolutional Neural Networks (CNNs)** to automate the diagnostic classification of human brain Magnetic Resonance Imaging (MRI) scans. This project addresses a high-stakes clinical challenge by accurately identifying and categorizing brain abnormalities into four distinct structural classes. By leveraging custom-engineered spatial pre-processing, data augmentation matrices, and deep convolutional pipelines with strict regularizations, the flagship model achieves an outstanding **99.14% Classification Accuracy on Unseen Test Cohorts**.


##  Executive Project Abstract

In neuro-oncology, manual inspection of high-resolution MRI slices is time-intensive, subject to intra-observer variability, and reliant on expert availability. This repository presents an automated, highly scalable computer vision solution. Bypassing fragile feature-engineering systems, the architecture utilizes stacked 2D convolutional filters to dynamically capture hierarchy patterns—ranging from micro-level tissue edge boundaries to macro-level tumor structural geometries. 

The primary contribution of this work is an optimized, multi-tier CNN framework engineered from scratch using TensorFlow and Keras. It demonstrates superior convergence and generalization stability by integrating Batch Normalization layers and Dropout drop rates, neutralizing the classic high-variance overfitting patterns commonly encountered in medical imaging datasets.



##  Clinical Domain & Dataset Architecture

The project processes a heavily validated, consolidated medical imaging cohort consisting of **7,023 structural human brain MRI scans**. The target space is partitioned into four explicit diagnostic classes, covering the most prevalent forms of intracranial growths along with healthy control baselines:

* **Glioma**: Primary brain tumors originating within the glial cells of the central nervous system. Characterized by ill-defined margins, infiltrative growth behaviors, and highly variable interior tissue densities on MRI sequences.
* **Meningioma**: Extracranial, typically slow-growing tumors arising from the meningeal layers enveloping the brain and spinal cord. They present distinct, well-circumscribed structural boundaries pushing against normal brain tissue.
* **Pituitary Tumors**: Abnormal growths developing inside the pituitary gland pocket at the skull base. These trigger severe mass effects on neighboring optic structures and cause massive systemic endocrine/hormonal imbalances.
* **No Tumor**: Clean, benchmark healthy control scans displaying non-pathological neuroanatomy. Essential for training the system to minimize false positives and prevent healthy patients from undergoing unwarranted clinical interventions.

###  Comprehensive Cohort Distribution
To prevent data leakage and guarantee experimental integrity, the collection is systematically divided into isolated training and evaluation testing frameworks:

| Target Diagnostic Class | Training Subset | Evaluation Test Subset | Total Volumetric Samples |
| :--- | :--- | :--- | :--- |
| **Pituitary Tumor** | 1,457 slices | 300 slices | 1,757 samples |
| **No Tumor (Healthy Control)** | 1,595 slices | 405 slices | 2,000 samples |
| **Meningioma Tumor** | 1,339 slices | 306 slices | 1,645 samples |
| **Glioma Tumor** | 1,321 slices | 300 slices | 1,621 samples |
| **Total Aggregated Portfolio** | **5,712 Slices** | **1,311 Slices** | **7,023 Slices** |

###  Source Dataset Provenance
The model tracks performance against the unified multi-source repository curated and hosted on Kaggle. It aggregates key multi-institutional historical medical assets (including Figshare, SARTAJ, and Br35H reference standards) into a clean, benchmark-ready format.
* **Dataset Link:** [Kaggle - Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)


##  Core Data Engineering & Input Pipeline

Raw medical MRI images arrive from clinical systems with diverse spatial proportions, non-standardized brightness distributions, and substantial background scanning noise. The data pipeline implements an automated execution chain to format incoming image tensors:

### 1. Spatial Uniformity & Interpolation
All images are dynamically decoded from their raw formats and rescaled to a uniform isotropic tensor shape of **`150 x 150 pixels`** across 3 color channels (RGB). This spatial resolution balances structural feature retention with compute-efficiency, preventing memory allocation overflows during training loops.

### 2. Global Pixel Intensity Normalization
Pixel intensity values are mapped down from their native integer ranges `[0, 255]` to floating-point vectors bounded strictly between `[0.0, 1.0]`. This is done via deterministic multiplication scaling ($1./255$). Normalizing the inputs prevents gradient explosion risks and ensures that initial weight matrices converge evenly across all hidden dimensions.

### 3. Real-Time Stochastic Data Augmentation
Medical image repositories frequently suffer from structural variances caused by patient head positioning inside the scanner bore. To build strict rotational and spatial invariance directly into the neural layers, a robust train-time augmentation pipeline is introduced using an append-only `ImageDataGenerator` setup with the following operational constraints:
* **Rotation Range**: Up to $\pm 10$ degrees to handle minor head tilting.
* **Width & Height Shifts**: Stochastically translated by fractions up to $0.1$ of total dimensions to simulate variable field-of-view centering.
* **Shear Transform Intensity**: Fixed at a maximum mapping of $0.1$ radians.
* **Zoom Scale Factors**: Constrained within a random range of $[0.9, 1.1]$ to mimic natural variations in scanner magnification levels.
* **Horizontal / Vertical Reflections**: Enabled to account for mirror orientations.
* **Fill Mode Allocation**: Set to `nearest` to cleanly handle empty pixel spaces generated by spatial transformations without injecting artificial edge boundaries.


##  Neural Network Architecture Details

This repository tracks architectural evolution by developing and contrasting two separate Deep Learning configurations:

### Architectural Approach 1: The Baseline Prototype
A lighter convolutional setup designed to establish an operational baseline. It utilizes sequential convolutional groups followed by standard spatial pool downsamplings, ending in a dense classification head. While compute-efficient, it serves as the foundation for the high-capacity, regularized architecture that follows.

### Architectural Approach 2: The Optimized Multi-Tier Deep Network (Flagship)
This model features an enterprise-grade, deeply stacked network utilizing alternating blocks of feature extraction, internal standardization, and aggressive regularization parameters.



[Input Tensor: 150x150x3]
│
▼
[Conv2D (32 Filters, 3x3, ReLU)] ──► [Batch Normalization] ──► [MaxPooling2D (2x2)] ──► [Dropout (0.10)]
│
▼
[Conv2D (64 Filters, 3x3, ReLU)] ──► [Batch Normalization] ──► [MaxPooling2D (2x2)] ──► [Dropout (0.20)]
│
▼
[Conv2D (128 Filters, 3x3, ReLU)] ──► [Batch Normalization] ──► [MaxPooling2D (2x2)] ──► [Dropout (0.20)]
│
▼
[Conv2D (128 Filters, 3x3, ReLU)] ──► [Batch Normalization] ──► [MaxPooling2D (2x2)] ──► [Dropout (0.20)]
│
▼
[Conv2D (256 Filters, 3x3, ReLU)] ──► [Batch Normalization] ──► [MaxPooling2D (2x2)] ──► [Dropout (0.20)]
│
▼
[Flatten] ──► [Dense (512 Units, ReLU)] ──► [Dropout (0.30)] ──► [Dense (4 Units, Softmax Output)]



#### Detailed Layer Breakdown of the Flagship Model:
1.  **Input Layer**: Accepts pre-processed image matrices formatted as standard `(150, 150, 3)` feature tensors.
2.  **Tier-1 Extraction Block**: Instantiates a 2D Convolutional layer driving 32 unique filters using small, highly focused $3\times3$ sliding spatial kernels backed by Rectified Linear Unit (`ReLU`) activations.
3.  **Batch Normalization Layer**: Inserted directly post-activation to continuously rescale internal feature map outputs. This structural element minimizes internal covariate shift, allowing for higher learning rates and significantly faster convergence.
4.  **Spatial Downsampling**: A Maximum Pooling layer (`MaxPooling2D`) utilizing a $2\times2$ stride window steps across the activation planes, shrinking spatial dimensions by 50% to preserve dominant structural activations while dropping non-essential parameters.
5.  **Dropout Safety Valve**: Triggers a localized dropout coefficient ($10\%$), temporarily deactivating arbitrary nodes on each training tick to break complex co-dependencies between neighboring weights.
6.  **Tiers-2 through 5 Advanced Extraction Blocks**: The channel depth scales sequentially through **64 $\rightarrow$ 128 $\rightarrow$ 128 $\rightarrow$ 256** distinct convolutional filters. This progressive depth scaling allows early layers to capture primitive edge features, mid-layers to resolve regional tissue contours, and deep layers to map complex holistic tumor structures. Dropout probabilities scale up to $20\%$ across these deep layers to maintain generalization robustness.
7.  **The Flattening Transition**: A flat multi-dimensional feature compressor transforms the high-dimensional feature map volumes down into a continuous, singular **1D vector pipeline**.
8.  **Dense Classifier Head**: Feeds the feature vector into a large, fully connected hidden layer housing **512 functional neural units** backed by `ReLU` activation. This layer is regularized with an intense **$30\%$ Dropout penalty** to insulate final predictions from noise overfitting.
9.  **Softmax Classification Output Layer**: Ends in a final output layer with **4 dedicated units** controlled by a multi-class `Softmax` activation function. This generates a clean probability vector summing to exactly $1.0$, mapping the input scan's likelihood across the target diagnostic domains.



##  Optimization Engine & Strategic Compilation

To ensure highly stable gradient updates and protect the network from falling into flat local minima, the training loop is coupled with a highly defensive optimization ecosystem:

* **Core Optimizer**: Powered by the **Adam (Adaptive Moment Estimation)** algorithm. Adam computes adaptive learning rates for individual parameters by tracking both the first moment (gradient mean) and second moment (uncentered variance), providing smooth trajectories through complex optimization surfaces.
* **Loss Formulation Tracker**: Regulated through **Categorical Cross-Entropy Loss**. This evaluates the information-theoretic distance separating the ground-truth one-hot encoded diagnosis vector from the network's predictive Softmax output distribution.
* **Dynamic Learning Rate Reduction Callback (`ReduceLROnPlateau`)**: Monitors validation loss tracking metrics. If the validation score exhibits stagnation for a flat plateau window of **2 consecutive epochs**, the controller step-down reduces the current learning rate by a multiplicative factor of **0.5 (50% reduction)**. This decay allows the model to carefully settle into deep, narrow global minima.
* **Automated Weight Preservation Callback (`ModelCheckpoint`)**: Continually audits network states throughout the training lifecycle, automatically exporting and preserving the absolute lowest historical validation loss weight configuration to disk. This ensures that the finalized model remains unaffected by late-stage epoch overfitting.
* **Proactive System Shutdown Callback (`EarlyStopping`)**: Tracks global validation convergence vectors. If validation loss scores fail to show progressive reductions over a designated patience threshold of **5 consecutive epochs**, the loop terminates training to preserve generalization performance.


##  Comprehensive Performance & Metric Audit

Upon conclusion of the training and callback loops, the flagship network architecture demonstrates near-flawless convergence stability. The model avoids the standard accuracy separations between training and validation trends that indicate high variance.

###  Testing Sub-Set Execution Readout
When subjected to rigorous evaluation across the fully isolated, unseen test subset consisting of **1,311 novel medical scans**, the model delivered the following metrics:


###  Multi-Class Performance Matrix Evaluation
The predictive performance across all four classes highlights balanced diagnostic precision:

* **Precision (Positive Predictive Value)**: Achieves exceptional high-tier performance across all four metrics. This minimizes dangerous false-positive classifications, ensuring that scans flagged with a specific tumor classification are highly accurate.
* **Recall / Sensitivity (True Positive Rate)**: Achieves high scores across the Glioma, Meningioma, and Pituitary classes. This confirms that the model consistently catches true positive pathologies, minimizing false negatives where a tumor might be missed.
* **F1-Score Metrics**: The harmonic mean metrics across all four diagnostic domains remain closely balanced, indicating consistent, reliable classification performance without any class-bias degradation.


cnn-brain-tumor-classification/
│
├── cnn-brain-tumor-classification.ipynb  # Primary Jupyter Engineering Notebook
├── README.md                             # Production Portfolio Documentation
│
├── dataset/                              # Root Directory for Medical Slices
│   ├── Training/                         # 5,712 Ingestion Scans
│   │   ├── glioma/
│   │   ├── meningioma/
│   │   ├── no_tumor/
│   │   └── pituitary/
│   │
│   └── Testing/                          # 1,311 Validation Evaluation Scans
│       ├── glioma/
│       ├── meningioma/
│       ├── no_tumor/
│       └── pituitary/
│
└── saved_models/                         # Best Historical Performance Output Backups
└── best_brain_tumor_model.h5         # Exported Low-Loss Checkpoint Weights






