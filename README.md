# Video Anomaly Detection with Structured Rules (VADSR)

## Authors

Thomas Foltz, Huijuan Xu

## Overview

This project introduces VADSR, a novel approach for video anomaly detection that leverages structured rules. Inspired by AnomalyRuler and PiShield, VADSR combines rule-based anomaly detection with a propositional requirement layer to effectively identify and explain anomalous events in video data.

## Key Features

*   **Accurate Anomaly Detection:**
*   **Explainable Anomalies**
*   **Efficient Processing**

## Methodology

VADSR builds upon the foundations of AnomalyRuler and PiShield, adapting and extending their core principles:

*   **AnomalyRuler:**  VADSR adopts the rule-based anomaly detection framework from [AnomalyRuler](https://github.com/Yuchen413/AnomalyRuler), utilizing a similar induction and deduction pipeline for rule and frame description generation. However, VADSR extends this framework by employing a classification network for final anomaly identification rather than utilizing a LLM for reasoning.
*   **PiShield:** For more details on PiShield, please refer to the [PiShield README](./pishield/README.md).

## Dependencies

This project requires Python 3.10 and the packages listed in `requirements.txt`.

### Creating a Virtual Environment (Recommended)

```bash
conda create -n vadsr python=3.10
conda activate vadsr
pip install -r requirements.txt
```
