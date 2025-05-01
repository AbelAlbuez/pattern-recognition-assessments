# Traffic Congestion Prediction using Time Series Modeling

This project aims to analyze and predict urban traffic congestion using time series modeling techniques. We preprocess a mobility dataset with PCA, and then apply ARIMA and LSTM models to forecast traffic behavior.

---

## 📁 Project Structure

traffic_congestion_project/
│
├── 01_datasets/
│   ├── smart_mobility_dataset.csv           # Original raw dataset
│   └── traffic_data_preprocessed.csv        # Cleaned + PCA-transformed dataset
│
├── 02_preprocessing/
│   └── preprocess.py                        # Script to clean data, apply PCA, and export ready dataset
│
├── 03_time_series_modeling/
│   ├── arima_model/                         # ARIMA implementation (Omar's part)
│   │   ├── arima_model.ipynb                # Jupyter Notebook for ARIMA
│   │   ├── arima_predictions.png            # Plot of ARIMA forecast vs real
│   │   └── mae_arima.txt                    # MAE error value
│   │
│   └── lstm_model/                          # LSTM implementation (Your part)
│       ├── lstm_model.ipynb                 # Jupyter Notebook for LSTM
│       ├── lstm_predictions.png             # Plot of LSTM forecast vs real
│       └── mae_lstm.txt                     # MAE error value
│
├── 04_final_outputs/
│   ├── comparison_plot.png                  # Combined LSTM vs ARIMA prediction plot
│   ├── model_summary.pdf                    # Table comparing metrics and performance
│   └── deliverable_phase2.docx              # Final report for Phase 2 (second deliverable)
│
└── README.md                                # Project overview and instructions
