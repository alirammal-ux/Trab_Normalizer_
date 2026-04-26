# 🔬 TRAb Cross-Instrument Normalization System

# Status

🚧 **Work in progress** 🚧

## Overview
My thesis work compared an ELISA method for TRAb measurement against two chemiluminescence immunoassays and one TRACE-based platform. The goal was to identify the best candidate for replacing the existing method. Once the thesis was complete, a problem that had been present throughout the work remained unresolved — and I started working on a solution.

Even when methods are calibrated against the same international standard (NIBSC 08/204), they frequently produce numerically non-comparable results. This creates a cascade of practical consequences: discontinuity in patient clinical history, difficulty interpreting results across laboratories, and instrument-dependent diagnostic thresholds that cannot be straightforwardly transferred.

This meant that instrument selection could not rely solely on analytical performance metrics. Any replacement method would also need to preserve continuity with historical data, to protect patient clinical history and avoid misinterpretation by clinicians.

![Blant-Altman plot Alinity-DSX](Immagine2.png)


To address this problem, I am developing a latent-variable normalization framework designed to harmonize outputs across all platforms onto a unified, comparable scale — enabling instrument selection to be driven purely by empirical performance.

---

## Objective
- Harmonize TRAb measurements across multiple instruments  
- Provide a **common latent score** independent of the platform  
- Improve clinical interpretability and longitudinal consistency  

---

## Methodology

### 1. Latent Variable Extraction
- Apply **Principal Component Analysis (PCA)** on multi-instrument data  
- Extract the first principal component (PC1) as a **shared latent variable (gold standard)**  

### 2. Instrument-Specific Mapping
- Train a **Random Forest Regressor** for each instrument  
- Learn a non-linear mapping from raw measurements to the latent space

### 3. Inference
- Any new measurement, regardless of the instrument, is projected into the **latent space**  
- Results become directly comparable across platforms

---
## Licence
The project is under MIT LICENCE

