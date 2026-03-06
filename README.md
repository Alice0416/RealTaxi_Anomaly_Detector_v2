# RealTaxi Anomaly Detector

## Overview

This project builds a real-time anomaly detection system for NYC taxi activity data.
The goal is to detect sudden drops, spikes, or unusual behavior in city-level taxi demand.

## Dataset

We use the NYC Taxi dataset from the Numenta Anomaly Benchmark 
The data is a 30 minute aggregated time series of passenger counts with labeled anomalies.

## Objective
Our model generates an anomaly score at each time step and flags abnormal events.
Performance is evaluated using AUC-ROC and detection delay.

## Approach
We compare traditional methods (LOF, Isolation Forest, KNN), incremental models, ensemble methods, and deep temporal models (CNN, GRU, LSTM, Transformer) with our custom built anomaly detector.

---

## Pipeline
Run from the project root:

python -m scripts.01_make_splits

python -m scripts.02_train_models

python -m scripts.03_generate_scores

python -m scripts.04_select_thresholds

python -m scripts.05_eval_and_plots

---

## Models
LOF      density-based anomaly detection baseline  
RNN      supervised temporal classifier on sliding windows  
VAE      reconstruction-based anomaly detection  
ENS      ensemble of normalized model scores  

---

## Evaluation
Metrics reported:

AUROC    ranking ability of anomaly scores  
AUPRC    precision-recall performance under class imbalance  

Results are saved in:

outputs/scores.csv
outputs/preds.csv
outputs/figures/
outputs/tables/table1.csv

---

## Setup
pip install -r requirements.txt
