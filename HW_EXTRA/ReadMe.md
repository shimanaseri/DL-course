# Extra Homework

The homeworks in this repository cover a variety of deep learning models and techniques, including:

### **Homework 1: Bidirectional Recurrent Convolutional Neural Network (BRCNN)**
This homework focuses on using BRCNN for relation classification. The tasks include:

- **Model Overview:** Understanding the Bidirectional Recurrent Convolutional Neural Network (BRCNN) and its application in NLP tasks like relation classification.
- **Data Preprocessing:** Preparing and processing text data, tokenizing sentences, and transforming them into a suitable format for input into the model.
- **Model Training:** Training the BRCNN model with the processed data and evaluating the results.
- **Model Evaluation:** Reporting results based on accuracy, precision, recall, and F1 score, and analyzing whether the model's performance reflects real-world usage.

### **Homework 2: LoRA (Low-Rank Adaptation)**
This homework explores the Low-Rank Adaptation method in transfer learning:

- **LoRA Overview:** Understanding how LoRA enables efficient fine-tuning of large pre-trained models like RoBERTa by adapting specific layers of the model instead of fine-tuning all the parameters.
- **Model Implementation:** Implementing LoRA on the RoBERTa model and fine-tuning it on a large language model (LLM) task.
- **Comparative Analysis:** Comparing the performance of LoRA with traditional fine-tuning methods.
- **Data Handling and Preprocessing:** Selecting and preprocessing appropriate datasets, including the multiNLI and QQP datasets, for sentiment analysis and inference tasks.
- **Model Evaluation:** Evaluating the fine-tuned model using various metrics, such as accuracy, F1 score, and comparison of training times.

### **Homework 3: Fraud Detection**
This homework focuses on using machine learning models to detect fraudulent transactions:

- **Dataset Understanding:** Using a dataset of credit card transactions to train a fraud detection model.
- **Data Preprocessing:** Handling imbalanced classes, performing data normalization, and using techniques like Adaptive Synthetic Sampling (ADASYN) to balance the dataset.
- **Model Training:** Implementing a fraud detection model and training it on the preprocessed dataset.
- **Model Evaluation:** Evaluating the model's performance using confusion matrices, precision, recall, F1 score, and accuracy.
- **Model Comparison:** Comparing the performance of different models and analyzing the results.


## How to Run the Code

1. Clone this repository or download the notebook files.
2. Install the required libraries. You can install the dependencies using:

   ```bash
   pip install -r requirements.txt
