# Pneumonia Detection using Machine Learning

**Author**: Akash Kumar Yadav  
**Course**: CISC 5800 - Machine Learning  
**Institution**: Fordham University  
**Date**: December 2024

## Problem Statement

Pneumonia is an acute respiratory infection that significantly impacts global health. Early and accurate diagnosis is crucial for effective treatment. Chest X-ray imaging is the most common diagnostic tool, but its interpretation can be challenging.

## Project Goal

The objective of this project is to develop and evaluate a machine learning pipeline to automatically classify chest X-ray images as either depicting Normal lungs or lungs with Pneumonia.

## Proposed Dataset:
    - Source: Chest X-Ray Images (Pneumonia) – Kaggle
        Link: https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia
    - Contains 5,863 pediatric chest X-ray images divided int training, validation, and test sets.
    - Binary classification: Normal vs Pneumonia. 


## Dual-Approach Implementation

### Notebook 1: Comprehensive Analysis
- **Models**: Custom CNN, VGG16, ResNet50
- **Validation**: 15% split + K-fold cross-validation
- **Advanced Analysis**: Grad-CAM, confidence scores, error analysis
- **Best Result**: ResNet50 - 99.74% recall (1 missed case / 390)

### Notebook 2: Independent Validation with Focused Optimization
- **Models**: VGG16, ResNet50
- **Validation**: 20% split (larger validation set)
- **Optimization**: Lower learning rate (1e-5 for VGG16)
- **Best Result**: VGG16 - 98.97% recall (4 missed cases / 390)

## Key Achievements

✅ 99.74% recall - only 1 pneumonia case missed  
✅ Both approaches exceed 95% clinical threshold  
✅ Transfer learning validated across methods  
✅ Grad-CAM interpretability implemented  
✅ Publication-quality IEEE paper  

## Files

- `pneumonia_detection_1.ipynb` - Comprehensive approach
- `pneumonia_detection_2.ipynb` - Independent Validation with Optimized approach
- `pneumonia_detection_paper.pdf` - IEEE conference paper
- `project_proposal.docx` - Original proposal

## Results

All models achieve >95% recall, with ResNet50 (Notebook 1) achieving 
the best overall performance at 99.74% sensitivity.