# **HCMNet (Hierarchical Cross-modal Network)**

HCMNet is a humor and sarcasm recognition system based on Graph Neural Networks (GNN) and cross-modal fusion technology. This project integrates three modalities—text, audio, and video—and utilizes a hierarchical structure for feature extraction and contextual modeling.

## **Project File Structure**

The project consists of the following core Python files:

| File Name | Description |
| :---- | :---- |
| config.py | **Environment Configuration**: Handles global random seed settings to ensure experimental reproducibility and configures necessary library references. |
| modules.py | **Network Components**: Defines base units including Unimodal Encoders (CNN+BiLSTM), Cross-modal Attention Fusion modules, and the Intra-sentence GNN. |
| model.py | **Main Architecture**: Integrates components from modules.py to build the full HCMNet class (SarcasmGNNClassifier), implementing end-to-end feature fusion and sarcasm prediction logic. |
| data\_utils.py | **Data Processing**: Contains the MultimodalDataset class and collate\_fn for data loading, format conversion, and multi-modal alignment. |
| trainer.py | **Training Engine**: Encapsulates the training, validation, and testing loops, supporting single-fold training and performance evaluation (Accuracy, F1-score) within cross-validation. |
| main.py | **Program Entry**: The project's startup script. It defines hyperparameters, loads 5-fold cross-validation data, and outputs the final summarized experimental results. |

## **Data Preparation**

This project uses pre-extracted multi-modal features for training. You can download the processed feature data package via the following link:

* **Feature Data Download (Google Drive)**: [Click to download feature data](https://drive.google.com/file/d/1DZo54Tgaehkw8_A6iyJtwhdN87ir0cMn/view?usp=drive_link)

After downloading, please place the .pkl files in the final\_dataset\_cv/ directory so that main.py can correctly read the data for 5-fold cross-validation.

## **Environment Requirements**

* Python 3.8+  
* PyTorch  
* PyTorch Geometric (PyG)  
* Transformers  
* Scikit-learn  
* Tqdm  
* Numpy

## **Quick Start**

1. Install the environment dependencies.  
2. Download the feature data and place it in the directory structure mentioned above.  
3. Execute the training:  
   python main.py  
