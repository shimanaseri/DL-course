# Fifth Homework

The homeworks in this repository cover a variety of deep learning models and techniques, including:

### **Homework 1: Speech Emotion Recognition (SER)**
This homework focuses on applying the HuBERT model for speech emotion recognition. The tasks include:

- **Model Introduction:** Getting familiar with the HuBERT model for processing speech data and extracting features.
- **Data Preprocessing:** Preparing and preprocessing audio data for training, including handling challenges with speech data.
- **Data Loader Construction:** Implementing a data loader to handle varying sequence lengths in the dataset.
- **Problem Definition:** Defining the problem of emotion detection in speech, where the goal is to classify speech signals based on emotion (such as happiness, sadness, anger, etc.).
- **Training the Model:** Using HuBERT for emotion recognition and fine-tuning it for this specific task.

### **Homework 2: Fine-Tuning the BERT Model**
This homework deals with fine-tuning the BERT model for text classification tasks. The tasks include:

- **Model Introduction and Preprocessing:** Understanding BERT and applying it to text data, including preprocessing steps such as tokenization and padding.
- **Fine-Tuning the Model:** Fine-tuning BERT on the specified dataset and evaluating the results.
- **Layer Freezing:** Experimenting with freezing different layers of BERT to improve training efficiency and understanding the effect on model performance.
- **Attention Head Pruning:** Implementing and testing the pruning of attention heads to speed up training without significantly reducing performance.
- **Model Evaluation:** Evaluating the fine-tuned BERT model and comparing results based on accuracy, F1 score, and confusion matrix.

## How to Run the Code

1. Clone this repository or download the notebook files.
2. Install the required libraries. You can install the dependencies using:

   ```bash
   pip install -r requirements.txt
