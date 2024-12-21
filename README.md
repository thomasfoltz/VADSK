# Video Anomaly Detection with Structured Rules (VADSR)

## Authors

Thomas Foltz, Huijuan Xu


![image](https://github.com/user-attachments/assets/77c8f19e-bff2-4171-a5a9-694f24f3f4ec)




## Overview

This project introduces VADSR, a novel approach for video anomaly detection that leverages structured rules. Inspired by AnomalyRuler and TF-IDF, VADSR combines video captioning, text feature extracting, and deep neural networks to effectively identify anomalous events in video data.

## Key Features

*   **Detection Precision**
*   **Interpretable Input**
*   **Efficient Processing**

## Methodology

VADSR builds upon the foundations of AnomalyRuler and the SKLearn TF-IDF library, adapting and extending their core principles:

*   **AnomalyRuler:**  VADSR adopts the rule-based anomaly detection framework from [AnomalyRuler](https://github.com/Yuchen413/AnomalyRuler), utilizing a similar induction and deduction pipeline for extracting meaningful text features. However, VADSR simplifies these pipelines by deriving keywords with text processing rather than an LLM and employing a classification network for detection instead of employing an extra LLM for reasoning.
*   **TF-IDF:** For more details on TF-IDF, please refer to their [Documentation](https://scikit-learn.org/1.5/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html).

## Dependencies

This project requires Python 3.10 and the packages listed in `requirements.txt`.

### Creating a Virtual Environment (Recommended)

```bash
conda create -n vadsr python=3.10
conda activate vadsr
pip install -r requirements.txt
```

### Running VADSR

```bash
python induct.py --data {SHTech, avenue, ped2 (default)}
python deduct.py --data {SHTech, avenue, ped2 (default)}
```
