
# **DDoS Detection using Machine Learning Models**

This project implements multiple machine learning models for detecting Distributed Denial of Service (DDoS) attacks. The models used include Random Forest, Logistic Regression, Multilayer Perceptron (Neural Network), and Naive Bayes. The dataset includes network traffic features that help identify whether a given traffic pattern is benign or part of a DDoS attack.



## **Project Overview**

The goal of this project is to build a machine learning pipeline for detecting DDoS attacks based on network traffic data. The dataset includes various network-related features such as packet size, duration, and source/destination port information. By training multiple machine learning models on this dataset, the system learns to classify whether a given traffic pattern is benign or malicious (DDoS attack).

### **Key Features of the Project**:
- Data preprocessing, including feature scaling and splitting into training and test sets.
- Training multiple models for classification:
  - Random Forest
  - Logistic Regression
  - Multilayer Perceptron (Neural Network)
  - Gaussian Naive Bayes
- Evaluation of models based on accuracy, F1 score, precision, recall, and ROC-AUC curve.
- Visualizations to aid model comparison and performance analysis.

---


### **Dataset**:
- The dataset used for training the models is assumed to be a CSV file containing network traffic data with various features.
- **Example Data Columns**:
  - `Destination Port`
  - `Flow Duration`
  - `Total Fwd Packets`
  - `Total Backward Packets`
  - `Label` (BENIGN or DDoS attack)

You can load the dataset as follows:


## **Machine Learning Models Used**

The following machine learning models are implemented in the project:

1. **Random Forest Classifier**:
   - An ensemble learning method that builds multiple decision trees and aggregates their results for classification.
   - Trained using bagging to ensure diversity in the decision trees.

2. **Logistic Regression**:
   - A linear model used for binary classification tasks. It predicts the probability that an instance belongs to a certain class.
   
3. **Multilayer Perceptron (MLP)**:
   - A neural network model with one or more hidden layers.
   - Each neuron applies an activation function (e.g., ReLU) to its weighted inputs to produce an output.

4. **Gaussian Naive Bayes**:
   - A probabilistic classifier based on Bayes' Theorem. It assumes that features follow a Gaussian distribution and are conditionally independent given the class label.

---

## **Model Evaluation**

The models are evaluated using various performance metrics:
- **Accuracy**: The proportion of correctly classified instances.
- **F1 Score**: The harmonic mean of precision and recall, giving an idea of the model's performance in terms of both false positives and false negatives.
- **Precision**: The number of true positive predictions out of all positive predictions.
- **Recall**: The number of true positive predictions out of all actual positives.
- **ROC-AUC**: A performance measure that summarizes the trade-off between true positive rate and false positive rate.

Here’s an example of how the model evaluation is performed:

```python
# Random Forest Model Evaluation
y_pred_rf = rf_model.predict(X_test)
accuracy_rf = accuracy_score(y_test, y_pred_rf)
print(f"Random Forest Accuracy: {accuracy_rf:.4f}")

# ROC Curve for Random Forest
rf_proba = rf_model.predict_proba(X_test)
fpr, tpr, thresholds = roc_curve(y_test, rf_proba[:, 1])
plt.plot(fpr, tpr, label='Random Forest')
plt.legend()
plt.show()
```

---

## **Results**

After training the models, their performance is compared based on the metrics mentioned earlier. The ROC curve is used to visualize and compare the true positive and false positive rates of the models.
![image](https://github.com/user-attachments/assets/527fa0ca-1912-43ab-a353-65c06245ad93)





---

