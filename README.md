
# Pincipal Component Analysis in scikit-learn - Lab

## Introduction

PCA algorithm is generally applied in dimension reduction contexts with an option to visualize a complex high dimensional dataset in 2D or 3D. PCA can also do an amazing job towards removing the computational cost of other machine learning algorithms by allowing them to train on a reduced set of features (principal components)
In this lesson, we shall look into implementing PCA with `scikit-learn` to the popular iris dataset, in an attempt to reduce the number of dimensions from 4 to 2 and see if the reduced set of dimensions would still preserve the variance of complete dataset. 

## Objectives

You will be able to:

- Perform PCA in Python and scikit-learn using Iris dataset
- Measure the impact of PCA on the accuracy of classification algorithms
- Plot the decision boundary of different classification experiments to visually inspect their performance. 

## Iris Dataset

In this post we'll see how to use Principal Component Analysis to perform linear data reduction for the purpose of data visualization. Let's load the necessary libraries and iris dataset to get us started. 

Perform following steps:

- Load Iris dataset into a pandas data frame  from the source "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data". (You can use `read_scv()` to load it directly from the server. 
- Give appropriate column names to dataset
- View the contents of the dataset


```python
# Load necessary libraries


# Your code here 


```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>sepal length</th>
      <th>sepal width</th>
      <th>petal length</th>
      <th>petal width</th>
      <th>target</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>5.1</td>
      <td>3.5</td>
      <td>1.4</td>
      <td>0.2</td>
      <td>Iris-setosa</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4.9</td>
      <td>3.0</td>
      <td>1.4</td>
      <td>0.2</td>
      <td>Iris-setosa</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4.7</td>
      <td>3.2</td>
      <td>1.3</td>
      <td>0.2</td>
      <td>Iris-setosa</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4.6</td>
      <td>3.1</td>
      <td>1.5</td>
      <td>0.2</td>
      <td>Iris-setosa</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5.0</td>
      <td>3.6</td>
      <td>1.4</td>
      <td>0.2</td>
      <td>Iris-setosa</td>
    </tr>
  </tbody>
</table>
</div>



So here we see a set of four input features i.e. four dimensions. Our goal for this simple analysis is to reduce this number to 2 (or 3) so that we can visualize the resulting principal components using the standard plotting techniques that we have learned so far in the course. 

## Standardize the Data

We have seen that PCA creates a feature __subspace__ that maximizes the variance along the axes. As features could belong to different scales of measurement, our first step in PCA is __always__ to standardize the feature set. Although, all features in the Iris dataset were measured on a same scale (i.e. cm), we shall still perform this step to get a mean=0 and variance=1 as a "standard practice". This helps PCA and a number of other machine learning algorithms to perform optimally. Visit [Importance of feature scaling](https://scikit-learn.org/stable/auto_examples/preprocessing/plot_scaling_importance.html#sphx-glr-auto-examples-preprocessing-plot-scaling-importance-py) at sk-learn documentation to read more on this. 

Let's create our feature and target datasets first.
- Create a set of features with 'sepal length', 'sepal width', 'petal length', 'petal width'. 
- Create X and y datasets based on features and target variables


```python
# Create features and Target dataset


# Your code here 


```

Now we can take our feature set `X`  and standardize it using `StandardScalar` method from sk-learn. 
- Standardize the feature set X


```python
# Standardize the features


# Your code here 


```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>sepal length</th>
      <th>sepal width</th>
      <th>petal length</th>
      <th>petal width</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-0.900681</td>
      <td>1.032057</td>
      <td>-1.341272</td>
      <td>-1.312977</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-1.143017</td>
      <td>-0.124958</td>
      <td>-1.341272</td>
      <td>-1.312977</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-1.385353</td>
      <td>0.337848</td>
      <td>-1.398138</td>
      <td>-1.312977</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-1.506521</td>
      <td>0.106445</td>
      <td>-1.284407</td>
      <td>-1.312977</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-1.021849</td>
      <td>1.263460</td>
      <td>-1.341272</td>
      <td>-1.312977</td>
    </tr>
  </tbody>
</table>
</div>



## PCA Projection to 2D Space

We shall now project the original data which is 4 dimensional into 2 dimensions. Remember,  there usually isn’t a particular meaning assigned to each principal component. The new components are just the two main dimensions of variance present in the data. To perform `PCA` with sk-learn, we need to import it first and create an instance of PCA while defining the number of principal components. 

- Initialize an instance of PCA from scikit-learn with 2 components
- Fit the data to the model
- Extract the first 2 principal components from the trained model


```python
# Run the PCA algorithm


# Your code here 


```

We can now save the results in a new dataframe and name the columns according the first/second component. 

- Append the target (flower name) to the principal components in a pandas dataframe 


```python
# Create a new dataset fro principal components 


# Your code here 


```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PC1</th>
      <th>PC2</th>
      <th>target</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-2.264542</td>
      <td>0.505704</td>
      <td>Iris-setosa</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-2.086426</td>
      <td>-0.655405</td>
      <td>Iris-setosa</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-2.367950</td>
      <td>-0.318477</td>
      <td>Iris-setosa</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-2.304197</td>
      <td>-0.575368</td>
      <td>Iris-setosa</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-2.388777</td>
      <td>0.674767</td>
      <td>Iris-setosa</td>
    </tr>
  </tbody>
</table>
</div>



Great, we now have a set of two dimensions, reduced from four against our target variable, the flower name. Let's now try to visualize this dataset and see if the different flower species remain separable. 

## Visualize Principal Components 

Using the target data, we can visualize the principal components according to the class distribution. 
- Create a scatter plot from principal components while color coding the examples


```python
# Principal Componets scatter plot


# Your code here 


```

## Explained Variance

> __The explained variance tells us how much information (variance) can be attributed to each of the principal components__

We can see above that the three classes in the dataset remain well separable. iris-virginica and iris-versicolor could be better separated, but we have to remember that we just reduced the size of dimensions to half. the cost-performance trade-off is something that data scientists often have to come across. In order to get a better idea around how much variance of the original dataset is explained in principal components, we can use the attribute `explained_variance_ratio_`.

- Check the explained variance of the two principal components using `explained_variance_ratio_`


```python
# Calculate the variance explained by pricipal components


# Your code here 


```

    Variance of each component: [0.72770452 0.23030523]
    
     Total Variance Explained: 95.8


First two PCs contain 95.80% of the information. The first PC contains 72.77% of the variance and the second PC contains 23.03% of the variance. The third and fourth principal component contained the rest of the variance of the dataset. 

## Compare Performance of an Classifier with PCA

So our principal components above explained 95% of variance in the data. How much would it effect the accuracy of a classifier? The best way to answer this is with a simple classifier like `KNeighborsClassifier`. We can try to classify this dataset in its original form vs. principal components computed above. 

- Run a `KNeighborsClassifier` to classify the Iris dataset 
- Use a trai/test split of 80/20
- For reproducability of results, set random state =9 for the split
- Time the process for splitting, training and making prediction


```python
# classification complete Iris dataset

# Your code here 


```

    Accuracy: 1.0
    Time Taken: 0.0017656260024523363


Great , so we see that we are able to classify the data with 100% accuracy in the given time. Remember the time taken may different randomly based on the load on your cpu and number of processes running on your PC. 

Now let's repeat the above process for dataset made from principal components 
- Run a `KNeighborsClassifier` to classify the Iris dataset with principal components
- Use a trai/test split of 80/20
- For reproducability of results, set random state =9 for the split
- Time the process for splitting, training and making prediction


```python
# Run the classifer on PCA'd data


# Your code here 


```

    Accuracy: 0.9666666666666667
    Time Taken: 0.00035927799763157964


So we see that going from 4 actual dimensions to two derived dimensions. We manage to get an accuracy of 96%. There is some loss but considering big data domain with data possibly having thousands of features, this trade-off is often accepted in order to simplify and speed up computation. The time taken to run the classifer is much less than what we saw with complete dataset. 

## Bonus : Visualize Decision Boundary 

visualizing decision boundary is good way to develop the intuition around a classifier's performance with 2/3 dimensional data. We can do this often to point out the examples that may not get classified correctly. It also helps us get an insight into how a certain algorithm draws these boundaries i.e. the learning process of an algorithm. 

- Draw the decision boundary for the classification with principal components (Optional - with complete dataset)


```python
# Plot decision boundary using principal components 


# Your code here 


```




    Text(0.5,1,'decision boundary')




![png](index_files/index_26_1.png)


## Level Up - Optional 

- Use following classifier instead of KNN shown above to see how much PCA effects the accuracy, coming from 4 to 2 dimensions. 

```python
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.grid_search import GridSearchCV
```

- Use 3 principal components instead of two and re-run your experiment to see the impact on the accuracy. 

## Summary 

In this lab we applied PCA to the popular Iris dataset. We looked at performance of a simple classifier and impact of PCA on it. NExt we shall take PCA to a more specialized domain i.e. Computer Vision and Image Processing and see how this technique can be used to image classification and data compression tasks. 
