# Video Anomaly Detection with Structured Keywords (VADSK)

## Authors

Thomas Foltz, Huijuan Xu


![image](https://github.com/user-attachments/assets/77c8f19e-bff2-4171-a5a9-694f24f3f4ec)




## Overview

This project introduces VADSK, a novel approach for video anomaly detection that leverages video captioning, keyword extraction, and a classification network to efficiently identify anomalous events in video data.

## Key Features

*   **Detection Precision**
*   **Interpretable Input**
*   **Efficient Processing**

## Methodology

VADSK draws inspiration from AnomalyRuler and utilizes the TF-IDF library for its core text-processing capabilities:

*   **[AnomalyRuler](https://github.com/Yuchen413/AnomalyRuler):**  Adopts a similar induction and deduction approach for detecting anomalies. However, VADSK simplifies this process by extracting keywords with text processing techniques and a classification network for detection rather than LLM reasoning capabilities.
*   **[TF-IDF](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html):** Short for term frequency-inverse document frequency, is a measure of the importance of a word to a document in a corpus, considering that some words appear more frequently than others.

## Results
| Benchmark | Accuracy | Precision | Recall | ROC AUC |
|---|---|---|---|---|
| UCSD-Ped2 | 0.8442 | 0.9928 | 0.8214 | 0.8946 |
| CUHK Avenue | 0.7540 | 0.5635 | 0.6404 | 0.7164 |
| Shanghai Tech | 0.7712 | 0.7742 | 0.6565 | 0.7567 |

## Dependencies

This project requires Python>=3.12 and the packages listed in `requirements.txt`.

### Creating a Virtual Environment (Recommended)

```bash
conda create -n vadsk python=3.12
conda activate vadsk
pip install -r requirements.txt
```

### Running VADSK

```bash
# Generate keywords for the dataset (defaults to ped2)
python induct.py --root <path_to_datasets> --dataset <ped2|avenue|SHTech>

# Generate frame descriptions for the selected test dataset (defaults to ped2)
python deduct.py --root <path_to_datasets> --dataset <ped2|avenue|SHTech>

# Add train and test flags for model training and final results (can be ran independently from frame description generation)
python deduct.py --train --test

# Add interpret flag during test time to view the feature input heatmap
python deduct.py --test --interpret
```
