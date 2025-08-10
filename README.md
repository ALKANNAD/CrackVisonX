# **CrackVisionX: A Fine-Tuned Framework for Efficient Binary Concrete Crack Detection**

This repository contains the official implementation for the paper **"CrackVisionX: A Fine-Tuned Framework for Efficient Binary Concrete Crack Detection"**. The project presents an advanced deep learning framework for the binary classification of concrete cracks, leveraging state-of-the-art CNN architectures and extensive hyper-parameter tuning.

## **Overview**

Automated crack detection is critical for maintaining the safety and integrity of concrete infrastructure. CrackVisionX is a state-of-the-art framework that integrates advanced CNN architectures—**ResNet50**, **MobileNet\_v3\_large**, **DenseNet121**, and **EfficientNetB0**—to optimize crack detection accuracy while maintaining low model complexity. The framework introduces a robust data augmentation strategy to address dataset imbalances and employs comprehensive preprocessing on the METU and SDNET2018 datasets. The models were evaluated across six distinct domains, with the EfficientNetB0 architecture demonstrating superior performance, achieving a peak accuracy of **99.98%**.

## **Publication**

The work in this repository is detailed in the following paper:

* **Title:** CrackVisionX: A Fine-Tuned Framework for Efficient Binary Concrete Crack Detection  
* **Authors:** Abdulrahman A. ALKannad, Ahmad AL Smadi, Moeen AL-Makhlafi, Shuyuan Yang, and Zhixi Feng.
* **Link :**  https://ieeexplore.ieee.org/document/10959054
* **GitHub Repository:** [https://github.com/ALKANNAD/CrackVisonX](https://www.google.com/search?q=https://github.com/ALKANNAD/CrackVisonX)

## **Models and Results**

The framework's performance was evaluated across six domains, created from the METU and SDNET2018 datasets. The table below summarizes the top test accuracies achieved by the best-performing model, **EfficientNetB0**, for each domain.

| Classification Domain | Dataset Source(s) | Model Architecture | Top Accuracy (Test Set) |
| :---- | :---- | :---- | :---- |
| **Bridge Deck** (Crack vs. No-Crack) | SDNET2018 | EfficientNetB0 | 99.71% |
| **Wall** (Crack vs. No-Crack) | SDNET2018 | EfficientNetB0 | 99.78% |
| **Pavement** (Crack vs. No-Crack) | SDNET2018 | EfficientNetB0 | 99.55% |
| **SDNET2018** (Crack vs. No-Crack) | SDNET2018 | EfficientNetB0 | 99.89% |
| **METU** (Crack vs. No-Crack) | METU | EfficientNetB0 | **99.98%** |
| **METU \+ SDNET2018** (Crack vs. No-Crack) | Combined | EfficientNetB0 | 99.92% |

## **Setup and Installation**

To reproduce the results, follow these steps to set up the environment.

### **1\. Prerequisites**

* Python 3.8+  
* Pip (Python package installer)  
* Git version control  
* (Recommended) An NVIDIA GPU with CUDA and cuDNN for accelerated training.

### **2\. Clone the Repository**

Open your terminal or command prompt and clone this repository:

git clone https://github.com/ALKANNAD/CrackVisonX.git  
cd CrackVisonX

### **3\. Set Up a Virtual Environment**

It is highly recommended to use a virtual environment to manage project dependencies.

\# Create the virtual environment  
python \-m venv venv

\# Activate the environment  
\# On Windows:  
venv\\Scripts\\activate  
\# On macOS/Linux:  
source venv/bin/activate

### **4\. Install Dependencies**

Install all required Python libraries using the requirements.txt file.

pip install \-r requirements.txt

### **5\. Download the Datasets**

You will need to download the datasets and organize them as described below.

* **METU Dataset:**  
  1. Download from [Kaggle](https://www.google.com/search?q=https://www.kaggle.com/datasets/kozistr/meta-koz-crack-dataset).  
  2. Place the Positive and Negative image folders inside a METU directory at the project root.  
* **SDNET2018 Dataset:**  
  1. Download from [Mendeley Data](https://data.mendeley.com/datasets/5y9wdsg2zt/2).  
  2. Place the unzipped image folders (e.g., CD, CW, CP, UD, etc.) inside a Dataset/SDNET2018 directory at the project root.

## **Usage**

To run an experiment, execute the desired Python script from your terminal. Make sure your virtual environment is activated.

**Example:**

\# To run the binary classification experiment on the METU dataset  
python train\METU\main.py

\# To run the binary classification experiment on the SDNET2018 dataset  
python train\SDNET2018\main.py

\# To run the binary classification experiment on the METU+SDNET2018 dataset  
python train\SDNET2018\main.py

The scripts will preprocess the data, build the model, train it, and save the evaluation results and plots in the Models/ directory.

## **Citation**

If you use this code or the findings from our paper in your research, please cite our work.

@article{CrackVisionX2025,  
    title={CrackVisionX: A Fine-Tuned Framework for Efficient Binary Concrete Crack Detection},  
    author={ALKannad, Abdulrahman A. and AL Smadi, Ahmad and AL-Makhlafi, Moeen and Yang, Shuyuan and Feng, Zhixi},  
    year={2025},  
    journal={IEEE Transactions on Intelligent Transportation Systems},  
}

