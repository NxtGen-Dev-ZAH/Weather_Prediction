import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from sklearn.metrics import (
    log_loss,
    confusion_matrix,
    accuracy_score,
    f1_score,
    jaccard_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
import sklearn.metrics as metrics

df = pd.read_csv("Weather_Data.csv")

# # d = {"Yes": 1, "No": 0}
# # df_weather["RainTomorrow"] = df_weather["RainTomorrow"].map(d)
# # print(df_weather["RainTomorrow"])
df.replace(["No", "Yes"], [0, 1], inplace=True)
# df_weather.replace(["No", "Yes"], [0, 1], inplace=True)
# # Assuming df_weather is your DataFrame with the target column "RainTomorrow"
df.drop(
    columns=["WindGustDir", "WindDir9am", "WindDir3pm", "Date"], axis=1, inplace=True
)
rows_with_true_value = df[df["RainTomorrow"] == 1]
rows_with_true_value.to_csv("trueprediction.csv", index=False)
rows_with_true_value2 = df[df["RainTomorrow"] == 0]
rows_with_true_value2.to_csv("falseprediction.csv", index=False)


df = pd.read_csv("Weather_Data.csv")
df_weather = pd.get_dummies(data=df, columns=["RainToday"])
df_weather.drop(
    columns=["WindGustDir", "WindDir9am", "WindDir3pm"], axis=1, inplace=True
)
print(df_weather.columns)
df_weather.replace(["No", "Yes"], [0, 1], inplace=True)
df_weather.drop("Date", axis=1, inplace=True)

df_weather = df_weather.astype(float)
print(df_weather.columns)

features = df_weather.drop(columns="RainTomorrow", axis=1).values
Y = df_weather["RainTomorrow"].values
print(features)
x_train, x_test, y_train, y_test = train_test_split(
    features, Y, test_size=0.2, random_state=10
)
print(df_weather.info())
LinearReg = LinearRegression()
LinearReg.fit(x_train, y_train)
predictions = LinearReg.predict(x_test)

LinearRegression_MAE = mean_absolute_error(y_test, predictions)
LinearRegression_MSE = mean_squared_error(y_test, predictions)
MSE = LinearRegression_MSE / len(y_test)
LinearRegression_r2 = r2_score(y_test, predictions)
# Metric Implementations
# confusionMatrix = confusion_matrix(y_test, predictions)
# print("\nConfusion Matrix:\n", confusionMatrix)
REPORT = pd.DataFrame(
    {
        "Metric": ["MAE", "MSE", "R2"],
        "Value": [LinearRegression_MAE, LinearRegression_MSE, LinearRegression_r2],
    }
)
print(REPORT)

KNN = KNeighborsClassifier(
    n_neighbors=4,
)

# Fit the model to the training data
KNN.fit(x_train, y_train)
# Make predictions on the test set
predictions = KNN.predict(x_test)
predic = [
    [
        19.5,
        22.4,
        15.6,
        6.2,
        0.0,
        41.3,
        17.6,
        20.6,
        92.8,
        84.0,
        1017.6,
        1017.4,
        8.0,
        8.0,
        20.7,
        20.9,
        1.0,
        1.0,
    ]
]
predic = KNN.predict(predic)
print("this is the prediction ", predic)
predi = [
    [
        12.2,
        22.0,
        0.0,
        3.2,
        8.2,
        30,
        13,
        17,
        82,
        55,
        1031.7,
        1029.3,
        6,
        3,
        16.0,
        21.0,
        0,
        0,
    ]
]
predi = KNN.predict(predi)
print("This is the predicted values of the given input", predi)
# Calculate accuracy on the test set
KNN_Accuracy_Score = accuracy_score(y_test, predictions)
KNN_JaccardIndex = jaccard_score(y_test, predictions)
KNN_F1_Score = f1_score(y_test, predictions)

Tree = DecisionTreeClassifier(random_state=10)
# Train the model with the training data
Tree.fit(x_train, y_train)
predictions = Tree.predict(x_test)
Tree_Accuracy_Score = accuracy_score(y_test, predictions)
Tree_JaccardIndex = jaccard_score(y_test, predictions)
Tree_F1_Score = f1_score(y_test, predictions)

x_train, x_test, y_train, y_test = train_test_split(
    features, Y, test_size=0.2, random_state=1
)
LR = LogisticRegression(solver="liblinear").fit(x_train, y_train)
predictions = LR.predict(x_test)
predict_proba = LR.predict_proba(x_test)
LR_Accuracy_Score = (y_test, predictions)
LR_JaccardIndex = (y_test, predictions)
LR_F1_Score = (y_test, predictions)
LR_Log_Loss = (y_test, predict_proba)


SVM = svm.SVC().fit(x_train, y_train)
predictions = SVM.predict(x_test)

predic = [
    [
        19.5,
        22.4,
        15.6,
        6.2,
        0.0,
        41.3,
        17.6,
        20.6,
        92.8,
        84.0,
        1017.6,
        1017.4,
        8.0,
        8.0,
        20.7,
        20.9,
        1.0,
        1.0,
    ]
]


print("Prediction: ", SVM.predict(predi))
SVM_Accuracy_Score = accuracy_score(y_test, predictions)
SVM_JaccardIndex = jaccard_score(y_test, predictions)
SVM_F1_Score = f1_score(y_test, predictions)

report = pd.DataFrame(
    {
        "Model": [
            "Linear Regression",
            "KNN",
            "Decision Tree",
            "Logistic Regression",
            "SVM",
        ],
        "Accuracy": [
            LinearRegression_MAE,
            KNN_Accuracy_Score,
            Tree_Accuracy_Score,
            LR_Accuracy_Score[1],
            SVM_Accuracy_Score,
        ],
        "Jaccard Index": [
            None,
            KNN_JaccardIndex,
            Tree_JaccardIndex,
            LR_JaccardIndex[1],
            SVM_JaccardIndex,
        ],
        "F1 Score": [None, KNN_F1_Score, Tree_F1_Score, LR_F1_Score[1], SVM_F1_Score],
        "Log Loss": [None, None, None, log_loss(y_test, predict_proba), None],
    }
)
report.to_json("report.json")
print(report)
