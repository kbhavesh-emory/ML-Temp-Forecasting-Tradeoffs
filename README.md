# ML-Temp-Forecasting-Tradeoffs

This repository contains a comprehensive study evaluating the accuracy-complexity trade-offs of various Recurrent Neural Network (RNN) architectures applied to ambient temperature forecasting. 

## Overview
The project compares five neural architectures on an 8-year dataset of daily temperature readings (3,103 samples):
* SimpleRNN
* GRU
* LSTM
* BiLSTM
* Stacked LSTM

It also investigates the sensitivity of the models to different historical time-window sizes (Look-Back analysis).

## Repository Contents
* `Project.ipynb`: Jupyter notebook containing all experiments and visualizations.
* `run_comprehensive_study.py`: Standalone Python script for running the comprehensive benchmark.
* `training.csv`, `testing.csv`: Datasets used for model training and testing.
* `comprehensive_results.png`: Summary visualization of all experimental findings.
* `results_model_comparison.csv`, `results_lookback_sensitivity.csv`: Exported result metrics.

## Key Findings
* **Simpler models generalize better:** The SimpleRNN (37 parameters) achieved lower test RMSE compared to more complex architectures like BiLSTM or Stacked LSTM, which exhibited signs of overfitting on this specific time-series.
* **Optimal Look-back window:** Historical windows of $L=1$ or $L=3$ are optimal; larger windows introduce noise and degrade predictive accuracy.

## Installation and Execution
1. Install dependencies:
```bash
pip install -r requirements.txt
```
2. Run the study script:
```bash
python run_comprehensive_study.py
```
Or open and execute `Project.ipynb` in Jupyter Notebook.
