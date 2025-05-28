

# Taxi Driver Identification using Deep Learning  

## Introduction  
This project addresses a **sequence classification problem** using deep learning models to predict **taxi driver identities** based on their **daily driving trajectories**. The dataset consists of **six months of GPS trajectory data** for five drivers, capturing both **spatial and temporal** driving patterns. The goal is to classify an input trajectory sequence to its corresponding driver using **deep learning models**.  

## Features  
- **Two Deep Learning Models**:  
  - **Fully Connected (FC) Model**: A simple feedforward neural network as a baseline.  
  - **LSTM Model**: A recurrent neural network that captures sequential dependencies in driving trajectories.  
- **Data Preprocessing**: Merging, deduplication, normalization, and feature engineering (cyclical time encoding).  
- **Model Training & Evaluation**: Accuracy tracking, loss monitoring, and performance comparison between FC and LSTM models.  
- **Hyperparameter Optimization**: Learning rate tuning, dropout regularization, and activation function selection.  
- **Visualization**: Training loss, validation accuracy, and trajectory-based analysis.  

## Dataset  
The dataset contains **GPS records** for **five taxi drivers** over **six months**, including:  
- **Spatial Features**: Longitude & Latitude (normalized).  
- **Temporal Features**: Time-based features (month, day, hour, minute) encoded cyclically (sine/cosine).  
- **Labels**: Driver IDs (multi-class classification with 5 classes).  

## Model Architectures  

### **1. Fully Connected (FC) Model**  
- **Input**: Processed GPS trajectory features.  
- **Hidden Layers**:  
  - 64 neurons (ReLU)  
  - 32 neurons (ReLU)  
  - 16 neurons (ReLU)  
- **Regularization**: Dropout (0.2).  
- **Output**: Softmax layer for 5-class classification.  

### **2. LSTM Model**  
- **Input**: Sequential GPS trajectory features.  
- **LSTM Layers**:  
  - 2 LSTM layers with **hidden size = 16**.  
- **Fully Connected Layers**: Classifier on top of LSTM output.  
- **Regularization**: Dropout (0.1).  
- **Output**: Softmax layer for 5-class classification.  

## Training Details  
- **Epochs**: 10  
- **Batch Size**: 20  
- **Optimizer**: Adam (learning rate = 0.001)  
- **Loss Function**: Cross-Entropy Loss  

## Experimental Results  

| Model  | Final Training Accuracy | Final Validation Accuracy | Final Validation Loss |  
|--------|------------------------|--------------------------|----------------------|  
| **FC Model** | 61.10% | **64.91%** | 0.6141 |  
| **LSTM Model** | 61.28% | 63.64% | **0.6217** |  

### **Key Observations**  
- The **FC model** achieved slightly higher validation accuracy but showed **training instability**.  
- The **LSTM model** demonstrated **better generalization** and **stable loss behavior**, making it more suitable for sequential trajectory classification.  

## Installation & Setup  
1. Clone the repository:  
   ```sh
   git clone https://github.com/your-repo/Taxi-Driver-Classification.git  
   cd Taxi-Driver-Classification  
   ```  
2. Install dependencies:  
   ```sh
   pip install torch pandas numpy matplotlib  
   ```  
3. Run the training script:  
   ```sh
   python train.py  
   ```  

## Future Improvements  
- Experiment with **GRU or Transformer models** for better sequence modeling.  
- Use **larger datasets** with more drivers to improve classification robustness.  
- Implement **GPS trajectory visualization tools** to analyze driving behaviors.  

