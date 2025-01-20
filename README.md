# Video Anomaly Detection with Structured Keywords (VADSK)

## Authors

Thomas Foltz, Huijuan Xu


![image](https://github.com/user-attachments/assets/77c8f19e-bff2-4171-a5a9-694f24f3f4ec)

## Overview

This project introduces VADSK, a novel approach for video anomaly detection that leverages video captioning, keyword extraction, and a classification network to efficiently identify anomalous events in video data. See the [source code](src/README.md) for instructions on how to run the codebase.

## Key Features

*   **Simple Framework**
*   **Interpretable Input**
*   **Efficient Processing**

## Methodology

VADSK draws inspiration from AnomalyRuler and utilizes the TF-IDF library for its core text-processing capabilities:

*   **[AnomalyRuler](https://github.com/Yuchen413/AnomalyRuler):**  Adopts a similar induction and deduction approach for detecting anomalies. However, VADSK simplifies this process by extracting keywords with text processing techniques and a classification network for detection rather than LLM reasoning capabilities.
*   **[TF-IDF](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html):** Short for term frequency-inverse document frequency, is a measure of the importance of a word to a document in a corpus, considering that some words appear more frequently than others.

## Results
| Benchmark | Accuracy | Precision | Recall | ROC AUC | F1 Score | Avg Inference (s) |
|---|---|---|---|---|---|---|
| UCSD-Ped2 | 0.8367 | 0.9786 | 0.8228 | 0.8653 | 0.8940 | 5.7692 |
| CUHK Avenue | 0.7596 | 0.519 | 0.7049 | 0.7415 | 0.5978| TBD |
| SHTech | 0.7651 | 0.7559 | 0.6682 | 0.7530 | 0.7093 | 5.1735 |

## CUDA Memory Usage
| Component | Total Memory (MB) | Peak Memory (MB) |
|---|---|---|
| Llama-3.2-11B-Vision-Instruct (induction/deduction) | 7293.34 | 8094.52 |
| VADSK (training) | 17.22 | 18.74 |
| VADSK (inference) | 0.22 | 0.45 |