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

VADSK draws inspiration from AnomalyRuler and utilizes the TF-IDF and NLTK libraries for its core text-processing capabilities:

*   **[AnomalyRuler](https://github.com/Yuchen413/AnomalyRuler):**  Adopts a similar induction and deduction approach for detecting anomalies. However, VADSK simplifies this process by extracting keywords with text processing techniques and a classification network for detection rather than LLM reasoning capabilities.
*   **[TF-IDF](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html):** Short for term frequency-inverse document frequency, is a measure of the importance of a word to a document in a corpus, considering that some words appear more frequently than others.
*   **[NLTK](https://www.nltk.org/howto/wordnet.html):** A platform for building Python programs to work with human language data.

## Results
| Benchmark | Accuracy | Precision | Recall | ROC AUC |
|---|---|---|---|---|
| UCSD-Ped2 | 0.8116 | 0.9850 | 0.7874 | 0.8625 |
| CUHK Avenue | 0.7828 | 0.5635 | 0.6052 | 0.7238 |
| Shanghai Tech | 0.7583 | 0.7402 | 0.6695 | 0.7471 |

## Dependencies

This project requires Python 3.10 and the packages listed in `requirements.txt`.

### Creating a Virtual Environment (Recommended)

```bash
conda create -n vadsk python=3.10
conda activate vadsk
pip install -r requirements.txt
```

### Running VADSK

```bash
python induct.py --data {ped2 (default), avenue, SHTech} # generates keywords
python deduct.py --data {ped2 (default), avenue, SHTech} # generates frame descriptions for selected test dataset
python deduct.py --train --test # add train and test flags for training the classification network and metrics
```
