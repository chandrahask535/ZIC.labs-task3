# ZIC.labs-TASK3

## IRIS FLOWER CLASSIFICATION

## Author: CHANDRAHAS K

## Batch: NOVEMBER

## Domain: Data Science

## Aim:
To develop a model that can classify Iris flowers into different species based on their sepal and petal
measurements. 

## Datasets

The following datasets were used for this project:
iris.csv: The iris dataset contains information about iris flowers, including sepal length, sepal width, petal length, petal width, and species.

## Libraries Used

The following important libraries were used for this project:

- numpy
- pandas
- matplotlib.pyplot
- seaborn
- train_test_split from sklearn.model_selection
- StandardScaler from sklearn.preprocessing
- KNeighborsClassifier from sklearn.neighbors
- accuracy_score, classification_report and confusion_matrix from from sklearn.metrics
- datasets from sklearn
- Axes3D from mpl_toolkits.mplot3d
- load_iris from sklearn.datasets

  ## DATA ANALYSIS

  1. To read the first 5 lines of the datas set `iris.head()` was used.
  2. Descriptive statistics for the dataset were displayed using `iris.describe()`.
  3. Missing values in the dataset were checked using `iris.isna().sum()`.
  4. To check number of columns and rows in the dataframes `iris.shape`.
  5. information about the `datairis.info()`
 
  ## DATA VISUALISATION

  1. 3D scatter plots were created to visualize the relationship between species, petal length, and petal width, as well as between species, sepal length, and sepal width using `matplotlib.pyplot`, `sklearn.datasets.load_iris` and `mpl_toolkits.mplot3d.Axes3D`.
  2. 2D scatter plots were created to visualize the relationship between species and sepal length
  as well as between species and sepal width using `sklearn.datasets.load_iris`,`matplotlib.pyplot`.

## Model Training and Evaluation

This code uses the K-Nearest Neighbors (KNN) algorithm, which is a simple and effective classification algorithm. The dataset is split into training and testing sets, features are standardized, and the KNN classifier is trained on the training set. Finally, predictions are made on the test set, and the model's performance is evaluated.

 # Accuracy Measure

1. The confusion matrix was calculated to evaluate the accuracy of the KMeans clustering.
2. The confusion matrix was plotted using `matplotlib.pyplot.imshow` and `plt.text` to visualize the true and predicted labels.

# Display the confusion matrix using seaborn

A confusion matrix is used to describe the performance of a classification model on a set of test data for which the true values are known. It is a way to visualize the performance of a classification algorithm. The confusion matrix is particularly useful in evaluating the performance of a model in terms of its ability to correctly and incorrectly classify instances.
  
