# Lab 3: CNN for CIFAR-10 Image Classification - Report

## Task 3: Data Augmentation

### 1. Objective

Train a CNN model with real-time data augmentation to reduce overfitting and improve generalization. Compare the results with the baseline CNN (Task 2).

### 2. Data Augmentation Techniques

The augmentation pipeline includes:

- **Random Horizontal Flip**: `RandomFlip("horizontal")` - flips images horizontally with 50% probability
- **Random Rotation**: `RandomRotation(0.08)` - rotates images by ±8% of π radians (±14.4 degrees)
- **Random Translation**: `RandomTranslation(0.1, 0.1)` - translates images by ±10% horizontally and vertically

These augmentations are applied **in real-time** during training using a Keras `Sequential` preprocessing layer.

### 3. Results Comparison

| Metric | CNN (Task 2) | CNN + Aug (Task 3) | Difference |
|--------|-------------|-------------------|------------|
| **Test Accuracy** | 83.22% | 72.46% | -10.76% |
| **Precision** | 0.8350 | 0.7523 | -0.0827 |
| **Recall** | 0.8322 | 0.7246 | -0.1076 |
| **F1-score** | 0.8310 | 0.7194 | -0.1116 |
| **Weights** | 816,938 | 1,117,354 | +300,416 |

### 4. Training Curves Analysis

#### CNN (Task 2)
- Training accuracy: 91.56% (final epoch)
- Validation accuracy: 83.76% (final epoch)
- **Gap**: ~7.8% - indicates moderate overfitting

#### CNN + Aug (Task 3)
- Training accuracy: 70.35% (final epoch)
- Validation accuracy: 74.52% (final epoch)
- **Gap**: ~-4.2% - validation higher than training, **no overfitting**

### 5. Key Observations

**1. Overfitting Reduction**
- The augmented model shows **no overfitting** (validation accuracy > training accuracy)
- The baseline CNN shows moderate overfitting with ~7.8% gap
- Data augmentation successfully regularizes the model

**2. Performance Trade-off**
- Despite reduced overfitting, overall accuracy dropped by ~10.76%
- The augmentation architecture (`CIFARCNNAugClassifier`) is **simpler** than the baseline CNN (`CIFARCNNClassifier`):
  - Fewer convolutional blocks (2 blocks vs 3 blocks)
  - No BatchNormalization layers
  - Less aggressive dropout

**3. Confusion Analysis**

CNN + Aug Top-3 confused pairs:
| True Class | Predicted As | Count |
|------------|--------------|-------|
| deer | frog | 312 |
| cat | frog | 273 |
| bird | frog | 266 |

The model confuses animals (deer, cat, bird) with frogs - likely due to similar textures and colors in training images.

### 6. Conclusion

Data augmentation **successfully reduced overfitting** as evidenced by the smaller train-validation gap. However, the overall performance is lower because:

1. The augmentation architecture is simpler than the baseline CNN
2. 30 epochs may not be sufficient for the augmented model to fully converge
3. The random augmentations make the learning task more challenging

**Recommendation**: Combine data augmentation with the full CNN architecture from Task 2 to achieve both reduced overfitting and high accuracy.

### 7. Recommendations for Future Work

1. **Apply augmentation to the full CNN architecture** (3 blocks with BatchNorm)
2. **Train for more epochs** (50-100) with early stopping
3. **Add more augmentation techniques**: CutOut, MixUp, or CutMix
4. **Use learning rate scheduling** with cosine decay
