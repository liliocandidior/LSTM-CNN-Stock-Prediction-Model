# LSTM-CNN-Stock-Prediction-Model

To run the scripts, please start up a virtual environment using `venv`:
-  run `python -m venv ./venv` to create the virtual environment named "venv" in the root directory
- run `source ./venv/bin/activate` to activate the virtual environment to install all dependencies

Required packages (dependencies) are in the file `requirements.txt`

To install all required packages, please run `pip install -r requirements.txt` command line from the root folder, where the `requirements.txt` is located

## Initiative
### Construct and Compare Model Accuracy and Efficiency between 3 Stock Prediction Models

- Basic LSTM
- LSTM Seq2seq
- CNN-Seq2seq

## Project Layout
1. Data Acquisition 
    - Retrieve live data giving stock ticket label
2. Model Construction 
    - Build and tune the model for the above three model methodologies
3. Output Agent 
    - Apply the built model and plot output stock predictions as well as Buy & Sell signaling point
4. Result Analysis 
    - Compare and analyze the performance metrics between the above three models

