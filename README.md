# Weather Prediction Project

This project focuses on predicting weather conditions using various machine learning algorithms. It utilizes data preprocessing, model training, evaluation, and prediction techniques to forecast whether it will rain tomorrow based on weather attributes.

## Dataset
The project uses a dataset named "Weather_Data.csv" containing various weather attributes. The dataset includes features like temperature, humidity, wind speed, and rainfall, along with the target variable indicating whether it will rain tomorrow.

## Data Preprocessing
- The target variable "RainTomorrow" is encoded as 0 for "No" and 1 for "Yes" to facilitate model training.
- Irrelevant columns like "WindGustDir", "WindDir9am", "WindDir3pm", and "Date" are dropped.
- The dataset is split into training and testing sets using the train_test_split function.

## Models Implemented
1. **Linear Regression:**
   - Utilizes the LinearRegression model to predict the target variable.
   - Calculates Mean Absolute Error (MAE), Mean Squared Error (MSE), and R-squared (R2) score for evaluation.

2. **K-Nearest Neighbors (KNN):**
   - Employs the KNeighborsClassifier with k=4 for classification.
   - Evaluates accuracy, Jaccard index, and F1 score.

3. **Decision Tree:**
   - Uses the DecisionTreeClassifier for classification.
   - Measures accuracy, Jaccard index, and F1 score.

4. **Logistic Regression:**
   - Implements LogisticRegression with the liblinear solver.
   - Computes accuracy, Jaccard index, F1 score, and log loss.

5. **Support Vector Machine (SVM):**
   - Utilizes svm.SVC for classification.
   - Calculates accuracy, Jaccard index, and F1 score.

## Evaluation Metrics
- **Accuracy:** Percentage of correctly predicted outcomes.
- **Jaccard Index:** Measure of similarity between predicted and actual values.
- **F1 Score:** Harmonic mean of precision and recall.
- **Log Loss:** Loss function representing the performance of a classification model.

## Files
- **falseprediction.csv:** CSV file containing weather data where it did not rain tomorrow.
- **trueprediction.csv:** CSV file containing weather data where it rained tomorrow.
- **report.json:** JSON file containing the evaluation metrics of different models.
- **Weather_Data.csv:** Dataset used for training and testing the models.

## Usage
1. Clone the project repository.
2. Ensure the required libraries are installed (`numpy`, `pandas`, `scikit-learn`).
3. Run the Python script to train and evaluate the models.
4. Use the trained models for weather prediction tasks.

This README provides an overview of the Weather Prediction Project, detailing its objectives, dataset, preprocessing steps, models implemented, evaluation metrics, and associated files.

