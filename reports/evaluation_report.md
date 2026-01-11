# Emotion Detection Evaluation Report

## Overview

This report summarizes the performance of different emotion detection models.

## Overall Performance Comparison

| Language   | Model   |   Accuracy |   Macro F1 |   Weighted F1 |
|:-----------|:--------|-----------:|-----------:|--------------:|
| EN         | lexicon |     0.3041 |     0.2383 |        0.3043 |
| EN         | ml      |     0.6721 |     0.6324 |        0.6834 |
| EN         | hybrid  |     0.6804 |     0.6737 |        0.6818 |
| RO         | lexicon |     0.3786 |     0.3624 |        0.3789 |
| RO         | ml      |     0.8484 |     0.8468 |        0.8482 |
| RO         | hybrid  |     0.8055 |     0.806  |        0.805  |

## EN - Detailed Results

### LEXICON Model

- **Accuracy**: 0.3041
- **Macro F1**: 0.2383
- **Weighted F1**: 0.3043

#### Per-Class Performance

| Emotion      |   Precision |   Recall |    F1 |   Support |
|:-------------|------------:|---------:|------:|----------:|
| anger        |       0.265 |    0.107 | 0.153 |      6724 |
| anticipation |       0.298 |    0.39  | 0.338 |      4118 |
| disgust      |       0.08  |    0.537 | 0.139 |       738 |
| fear         |       0.066 |    0.23  | 0.102 |      1096 |
| joy          |       0.375 |    0.393 | 0.384 |      6973 |
| neutral      |       0.395 |    0.45  | 0.421 |     16021 |
| sadness      |       0.175 |    0.224 | 0.197 |      2929 |
| surprise     |       0.172 |    0.106 | 0.132 |      3374 |
| trust        |       0.428 |    0.209 | 0.281 |     12290 |

### ML Model

- **Accuracy**: 0.6721
- **Macro F1**: 0.6324
- **Weighted F1**: 0.6834

#### Per-Class Performance

| Emotion      |   Precision |   Recall |    F1 |   Support |
|:-------------|------------:|---------:|------:|----------:|
| anger        |       0.658 |    0.606 | 0.631 |      6724 |
| anticipation |       0.633 |    0.749 | 0.687 |      4118 |
| disgust      |       0.177 |    0.854 | 0.294 |       738 |
| fear         |       0.454 |    0.827 | 0.586 |      1096 |
| joy          |       0.755 |    0.81  | 0.782 |      6973 |
| neutral      |       0.766 |    0.594 | 0.669 |     16021 |
| sadness      |       0.6   |    0.811 | 0.69  |      2929 |
| surprise     |       0.541 |    0.775 | 0.637 |      3374 |
| trust        |       0.851 |    0.619 | 0.717 |     12290 |

### HYBRID Model

- **Accuracy**: 0.6804
- **Macro F1**: 0.6737
- **Weighted F1**: 0.6818

#### Per-Class Performance

| Emotion      |   Precision |   Recall |    F1 |   Support |
|:-------------|------------:|---------:|------:|----------:|
| anger        |       0.604 |    0.662 | 0.632 |      6724 |
| anticipation |       0.577 |    0.755 | 0.654 |      4118 |
| disgust      |       0.509 |    0.851 | 0.637 |       738 |
| fear         |       0.585 |    0.874 | 0.701 |      1096 |
| joy          |       0.74  |    0.787 | 0.762 |      6973 |
| neutral      |       0.747 |    0.599 | 0.665 |     16021 |
| sadness      |       0.585 |    0.796 | 0.674 |      2929 |
| surprise     |       0.539 |    0.747 | 0.626 |      3374 |
| trust        |       0.807 |    0.638 | 0.713 |     12290 |

## RO - Detailed Results

### LEXICON Model

- **Accuracy**: 0.3786
- **Macro F1**: 0.3624
- **Weighted F1**: 0.3789

#### Per-Class Performance

| Emotion   |   Precision |   Recall |    F1 |   Support |
|:----------|------------:|---------:|------:|----------:|
| anger     |       0.433 |    0.311 | 0.362 |       823 |
| fear      |       0.277 |    0.364 | 0.315 |       535 |
| joy       |       0.403 |    0.53  | 0.458 |       642 |
| neutral   |       0.358 |    0.412 | 0.383 |      1284 |
| sadness   |       0.544 |    0.426 | 0.478 |      1093 |
| surprise  |       0.293 |    0.27  | 0.281 |       523 |
| trust     |       0.273 |    0.248 | 0.26  |       549 |

### ML Model

- **Accuracy**: 0.8484
- **Macro F1**: 0.8468
- **Weighted F1**: 0.8482

#### Per-Class Performance

| Emotion   |   Precision |   Recall |    F1 |   Support |
|:----------|------------:|---------:|------:|----------:|
| anger     |       0.837 |    0.905 | 0.87  |       823 |
| fear      |       0.818 |    0.849 | 0.833 |       535 |
| joy       |       0.877 |    0.891 | 0.884 |       642 |
| neutral   |       0.853 |    0.8   | 0.826 |      1284 |
| sadness   |       0.886 |    0.856 | 0.871 |      1093 |
| surprise  |       0.823 |    0.851 | 0.836 |       523 |
| trust     |       0.806 |    0.809 | 0.807 |       549 |

### HYBRID Model

- **Accuracy**: 0.8055
- **Macro F1**: 0.8060
- **Weighted F1**: 0.8050

#### Per-Class Performance

| Emotion   |   Precision |   Recall |    F1 |   Support |
|:----------|------------:|---------:|------:|----------:|
| anger     |       0.797 |    0.852 | 0.824 |       823 |
| fear      |       0.761 |    0.852 | 0.804 |       535 |
| joy       |       0.825 |    0.869 | 0.847 |       642 |
| neutral   |       0.837 |    0.726 | 0.777 |      1284 |
| sadness   |       0.84  |    0.798 | 0.818 |      1093 |
| surprise  |       0.762 |    0.818 | 0.789 |       523 |
| trust     |       0.761 |    0.805 | 0.782 |       549 |

## Visualizations

The following charts are available:

- `comparison_overall.png`: Overall metrics comparison across all models
- `comparison_en.png`: Model comparison for English
- `comparison_ro.png`: Model comparison for Romanian
- `per_class_en.png`: Per-class performance for English models
- `per_class_ro.png`: Per-class performance for Romanian models
- `confusion_matrix_en.png`: Confusion matrices for English models
- `confusion_matrix_ro.png`: Confusion matrices for Romanian models

