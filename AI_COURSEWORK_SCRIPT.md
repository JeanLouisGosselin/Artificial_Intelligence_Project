# PART 1 - PRELIMINARIES

## (a) Importing all necessary packages


```python
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
```

## (b) Loading the dataset


```python
originalDataSet = pd.read_csv('houses_data.csv')
```

## (c) Exploring the dataset

Let's explore the dataset a little:


```python
originalDataSet.shape
```


```python
originalDataSet.head()
```


```python
originalDataSet.tail()
```


```python
originalDataSet.info()
```

# PART 2 - CLEANING THE DATASET

## (a) Getting rid of all NaN values

Now that we have loaded and inspected our dataset, let's check for any columns that have NaN values (we will replace these NaN values shortly):


```python
originalDataSet.columns[originalDataSet.isnull().any()]
```

We will replace all NaN values with 0s:


```python
originalDataSet = originalDataSet.replace(np.nan, 0)
```

Now checking that no NaN values are left:


```python
originalDataSet.columns[originalDataSet.isnull().any()]
```

## (b) Special case

Despite our best efforts later on to discard 'object'-typed data and keep nothing but numeric-type data (see below), we have noticed that the string value 'Grvl', present in certain cells in both 'Street' and 'Alley' columns of the data set (53 cells, to be exact, found by doing Ctrl-F inside the excel sheet itself ---> attach screenshot!!) never disappears.

We have therefore taken the decision to replace this 'Grvl' with the integer value "1", since all columns containg string values will soon be discarded.

Link:

https://stackoverflow.com/questions/17114904/python-pandas-replacing-strings-in-dataframe-with-numbers


```python
mymap = {'Grvl':"1", 'Pave':"1"}

originalDataSet.applymap(lambda s: mymap.get(s) if s in mymap else s)
```

## (c) Discarding all non-numeric values

Now let’s do some filtering to extract only the columns containing numeric-based values from the dataset:


```python
str_list = []

for colname, colvalue in originalDataSet.iteritems():
    if type(colvalue[1]) == str:
        str_list.append(colname)
        
num_list = originalDataSet.columns.difference(str_list)
```

We can now replace our initial dataset ("originalDataSet") with  new one ("cleansedDataSet") which will now only contain numeric-type values:


```python
cleansedDataSet = originalDataSet[num_list]
```

Investigating this new dataset a little:


```python
cleansedDataSet.head()
```


```python
cleansedDataSet.tail()
```


```python
cleansedDataSet.info()
```

## (d) Dropping all remaining and unnecessary string-based columns

As we can see in the results above, there are still a few columns that are not integer-typed (so far, we cannot find the reason why they should persist).

Having opened the Excel spreadsheet, we have come to realise that the following columns:

Alley
Fence
PoolQC

...are string-typed (and for some reason, haven't been dropped in the previous process...).

These will need to be dropped:


```python
cleansedDataSet = cleansedDataSet.drop(['Alley','PoolQC','Fence', 'MiscFeature'], axis=1)
```


```python
cleansedDataSet.info()
```

## (e) Convert all INT values to FLOAT

As a final process, we shall convert all INT to FLOAT (needed?).

Link: https://stackoverflow.com/questions/21291259/convert-floats-to-ints-in-pandas

Why do we do so? Simply because PCA is desinged for continuous variables. It tries to minimize variance (=squared deviations). The concept of squared deviations breaks down when you have binary variables.

https://stackoverflow.com/questions/40795141/pca-for-categorical-features


```python
# technique 1:
arrConvertValues = ['1stFlrSF',
                    '2ndFlrSF',
                    '3SsnPorch',
                    'BedroomAbvGr',
                    'BsmtFinSF1',
                    'BsmtFinSF2',
                    'BsmtFullBath',
                    'BsmtHalfBath',
                    'BsmtUnfSF',
                    'EnclosedPorch',
                    'Fireplaces',
                    'FullBath',
                    'GarageArea',
                    'GarageCars',
                    'GrLivArea',
                    'HalfBath',
                    'Id',
                    'KitchenAbvGr',
                    'LotArea',
                    'LowQualFinSF',
                    'MSSubClass',
                    'MiscVal',
                    'MoSold',
                    'OpenPorchSF',
                    'OverallCond',
                    'OverallQual',
                    'PoolArea',
                    'SalePrice',
                    'ScreenPorch',
                    'TotRmsAbvGrd',
                    'TotalBsmtSF',
                    'WoodDeckSF',
                    'YearBuilt',
                    'YearRemodAdd',
                    'YrSold']

for x in arrConvertValues:
    cleansedDataSet[x] = cleansedDataSet[x].astype(float)

```

Checking the data from our latest dataset one last time:


```python
cleansedDataSet.info()
```

# PART 3 - THE "SALEPRICE" COLUMN

## (a) Viewing the column's values

From exploring the data, we notice an interesting column: SALEPRICE


```python
print(cleansedDataSet['SalePrice'])
```

Let's sort the house prices, from lowest to highest:


```python
cleansedDataSet.sort_values(by='SalePrice')

# We see that house prices range from 34,900$ to 755,000$
```

## (b) Viewing distribution of SALEPRICE values on a chart

Let's visualize how this SALEPRICE data is distributed:


```python
fig= plt.figure(figsize=(40,10))
plt.subplot(1, 2, 1)
sns.set(style = 'whitegrid')
sns.distplot(cleansedDataSet['SalePrice'], color = 'red')
plt.title('Distribution of Sale Prices', fontsize = 20)
plt.xlabel('Range of House Prices')
plt.ylabel('Count')
plt.show()
```

## (c) Getting the median & average & max of SALEPRICE values

Let's find the median of all the house sale prices:


```python
import statistics
theMedian = statistics.median(cleansedDataSet['SalePrice'])

print(theMedian)
```

NOTE: the result given here (163,000) is different from the one I obtained in the Excel spreadsheet itself (by
using the following formula in a blank cell: =MEDIAN(CC1, CC1461), which gave me 147,500!)

Now, let's also calculate the average price of all these houses:


```python
def Average(lst): 
    return sum(lst) / len(lst) 

lst=cleansedDataSet['SalePrice']
theAverage = Average(lst) 
  
print("Average of the list =", round(theAverage, 2))
```

Finally, getting the max value of all these house sale prices:


```python
maxPrice = cleansedDataSet['SalePrice'].max()

print(maxPrice)
```

## (d) Viewing percentage split of SALEPRICE values based on the median

Code for pie chart found here:

https://stackoverflow.com/questions/54379345/i-want-to-create-a-pie-chart-using-a-dataframe-column-in-python


```python
salesSplitFromMedian = pd.DataFrame(cleansedDataSet['SalePrice'])
# print(salesSplitFromMedian)
```


```python
salesSplitFromMedian['SALE PRICES SPLIT FROM MEDIAN'] = pd.cut(salesSplitFromMedian['SalePrice'], 
                                                        bins=[0, theMedian, maxPrice],
                                                        labels=['0-163000.0(median)','163000.0(median)-755000.0(max price)'], 
                                                        right=True)
```


```python
bin_percent_median = pd.DataFrame(salesSplitFromMedian['SALE PRICES SPLIT FROM MEDIAN'].value_counts(normalize=True) * 100)
plot = bin_percent_median.plot.pie(y='SALE PRICES SPLIT FROM MEDIAN', figsize=(10, 10), autopct='%1.1f%%')
```

## (e) Viewing percentage split of SALEPRICE values based on the average


```python
salesSplitFromAverage = pd.DataFrame(cleansedDataSet['SalePrice'])
# print(salesSplitFromMedian)
```


```python
salesSplitFromAverage['SALE PRICES SPLIT FROM AVERAGE'] = pd.cut(salesSplitFromAverage['SalePrice'], 
                                                        bins=[0, theAverage, maxPrice],
                                                        labels=['0-180921.2(average)','180921.2(average)-755000.0(max price)'], 
                                                        right=True)
```


```python
bin_percent_average = pd.DataFrame(salesSplitFromAverage['SALE PRICES SPLIT FROM AVERAGE'].value_counts(normalize=True) * 100)
plot = bin_percent_average.plot.pie(y='SALE PRICES SPLIT FROM AVERAGE', figsize=(10, 10), autopct='%1.1f%%')
```

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

# PART 3b - Attempt to standardize / MinMax data

Normally at this stage we would proceed to standardize (or "minmax") our data.

This is an important step, since failing to do so could result in having unequal weights in our distance computation (more on this in the report).

However, as we shall demonstrate below, proceeding to standardize or "minmax" the data will in either case result in still having a very high data dimensionality (going from the initial 38 dimensions down to only 35...).

It is why we have sadly skipped this step.

------------------------------------------------------------

---> TALK ABOUT THIS MORE AT LENGTH IN THE REPORT ITSELF (use notes below)

(Reasons for data normalization are explained here:https://www.import.io/post/what-is-data-normalization-and-why-is-it-important

Also here: https://towardsdatascience.com/scale-standardize-or-normalize-with-scikit-learn-6ccc7d176a02

## (a) Standardizing data


```python
X = cleansedDataSet.values
from sklearn.preprocessing import StandardScaler
standardizedDataSet = StandardScaler().fit_transform(X)
```

Below: a quick comparison between the initial cleansed dataset structure & contents and the standardized dataset structure & contents:


```python
cleansedDataSet.shape
```


```python
standardizedDataSet.shape
```


```python
cleansedDataSet
```


```python
standardizedDataSet
```


```python
from sklearn.decomposition import PCA
std_pca = PCA().fit(standardizedDataSet)
plt.plot(np.cumsum(std_pca.explained_variance_ratio_))
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance');
```

AS we can observe from the graph above, the tipping point in the "elbow curve" above is at 35.

We will therefore transform our initial dataset to a 35-component PCA-transformed dataset (which is absolutely not ideal):


```python
pca_1 = PCA(n_components=35)
transformedSet_1 = pca_1.fit_transform(standardizedDataSet)

print("standardized dataset contents (rows/columns):   ", standardizedDataSet.shape)
print("transformed dataset contents (rows/columns):  ", transformedSet_1.shape)
```

From these inappropriate figures we've obtained above, juxtaposing the plotting of the initial and PCA-transformed datasets will naturally produce a flawed diagram, as that shown below:


```python
fig= plt.figure(figsize=(20,20))
transData_1 = pca_1.inverse_transform(transformedSet_1)
plt.scatter(standardizedDataSet[:, 0], standardizedDataSet[:, 1], alpha=0.2)
plt.scatter(transData_1[:, 0], transData_1[:, 1], alpha=0.8)
plt.axis('equal');
```

## (b) MinMax-ing data


```python
from sklearn import preprocessing
mm_scaler = preprocessing.MinMaxScaler()
X = cleansedDataSet.values
minMaxedDataSet = mm_scaler.fit_transform(X)
mm_scaler.transform(X)
```


```python
minMaxedDataSet.shape
```


```python
minMaxedDataSet
```


```python
from sklearn.decomposition import PCA
mm_pca = PCA().fit(minMaxedDataSet)
plt.plot(np.cumsum(mm_pca.explained_variance_ratio_))
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance');
```


```python
pca_2 = PCA(n_components=10)
transformedSet_2 = pca_2.fit_transform(minMaxedDataSet)

print("minmax-ed dataset contents (rows/columns):   ", minMaxedDataSet.shape)
print("transformed dataset contents (rows/columns):  ", transformedSet_2.shape)
```

Again, from these inappropriate figures we've obtained above, juxtaposing the plotting of the initial and PCA-transformed datasets will naturally produce a flawed diagram, as that shown below:


```python
fig= plt.figure(figsize=(20,20))
transData_2 = pca_2.inverse_transform(transformedSet_2)
plt.scatter(minMaxedDataSet[:, 0], minMaxedDataSet[:, 1], alpha=0.2)
plt.scatter(transformedSet_2[:, 0], transformedSet_2[:, 1], alpha=0.8)
plt.axis('equal');
```

It is for this reason, sadly, that we are omitting the standardization of our data in this coursework.

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

# PART 4 - plotting the initial cleansed dataset


```python
fig= plt.figure(figsize=(20,20))
plt.scatter(cleansedDataSet.to_numpy()[:, 0], cleansedDataSet.to_numpy()[:, 1])
plt.axis('equal');
```

# PART 5 - implementing PCA

## (a) Scree plot ("elbow curve")

In order to find the number of components for our PCA, we need to implement an "elbow curve" (this is further explained in the report):


```python
from sklearn.decomposition import PCA
pca = PCA().fit(cleansedDataSet)
fig= plt.figure(figsize=(7,7))
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance');
```

From this diagram (which we have expanded slightly in order to view the break in the elbow curve more clearly), we can deduce fairly confidently that the dataset can now be reduced to 2 components.

## (b) Setting our initial dataset to PCA-transformed dataset:


```python
pca = PCA(n_components=2)
PCAtransformedDataSet = pca.fit_transform(cleansedDataSet)
```

## (c) Comparing contents between initial & PCA-transformed datasets:


```python
print("cleansed dataset contents (rows/columns):   ", cleansedDataSet.shape)
print("transformed dataset contents (rows/columns):", PCAtransformedDataSet.shape)
```

## (d) Printing results from PCA:


```python
PCAtransformedDataSet
```

## (e) Juxtaposing the plotting of both cleansed and PCA-transformed datasets:

The PCA-transformed dataset is in orange, while the cleansed dataset is in blue.


```python
# NOTE (having to change our DataFrame to a numpy array by applying .TO_NUMPY())

fig= plt.figure(figsize=(20,20))
X_new = pca.inverse_transform(PCAtransformedDataSet)
plt.scatter(cleansedDataSet.to_numpy()[:, 0], cleansedDataSet.to_numpy()[:, 1], alpha=0.2)
plt.scatter(X_new[:, 0], X_new[:, 1], alpha=0.8)
plt.axis('equal');
```

The diagram below re-plots the PCA-transformed dataset onto a normalized graph:


```python
color_array = ['b']*730 + ['r']*730
fig= plt.figure(figsize=(20,20))
plt.scatter(PCAtransformedDataSet[:, 0], PCAtransformedDataSet[:, 1],
            c=color_array, edgecolor='none', alpha=0.5,
            cmap=plt.cm.get_cmap('nipy_spectral', 100))
plt.xlabel('component 1')
plt.ylabel('component 2')
plt.title('DATA LAID OUT IN 2D (a clear case of overplotting)', loc='left')
plt.colorbar();

# colours available:
# b: blue
# g: green
# r: red
# c: cyan
# m: magenta
# y: yellow
# k: black
# w: white
```

## (f) Comparing accuracy scores, from 38 components down to 2 components:

In order for us to verify how accuracy scores do change according to the number of principle components applied to the initial dataset, we will apply an early classification algorithm to the PCA-transformed dataset.

For this, we will work with SVM classifiers, as they do a good job of classifying the data depending on the number of dimensions we serve them.

We will start with the full 38 dimensions we obtained from the cleaning the original dataset, then slowly decrease the number of dimensions down to 2 as so: [38, 30, 20, 10, 8, 6, 4, 3, 2]

WE will therefore realise that accuracy in fact ***increases*** as the number of components ***decreases***. 


```python
pca = PCA(38)

X = pca.fit_transform(cleansedDataSet)
y = cleansedDataSet['SalePrice']

from sklearn.model_selection import train_test_split
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, random_state=7)

from sklearn.svm import SVC
model = SVC(kernel='linear', C=10)
model.fit(Xtrain, ytrain)
y_model = model.predict(Xtest)

from sklearn.metrics import accuracy_score
accuracy_score(ytest, y_model)
```


```python
pca = PCA(30)

X = pca.fit_transform(cleansedDataSet)
y = cleansedDataSet['SalePrice']

from sklearn.model_selection import train_test_split
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, random_state=7)

from sklearn.svm import SVC
model = SVC(kernel='linear', C=10)
model.fit(Xtrain, ytrain)
y_model = model.predict(Xtest)

from sklearn.metrics import accuracy_score
accuracy_score(ytest, y_model)
```


```python
pca = PCA(20)

X = pca.fit_transform(cleansedDataSet)
y = cleansedDataSet['SalePrice']

from sklearn.model_selection import train_test_split
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, random_state=7)

from sklearn.svm import SVC
model = SVC(kernel='linear', C=10)
model.fit(Xtrain, ytrain)
y_model = model.predict(Xtest)

from sklearn.metrics import accuracy_score
accuracy_score(ytest, y_model)
```


```python
pca = PCA(10)

X = pca.fit_transform(cleansedDataSet)
y = cleansedDataSet['SalePrice']

from sklearn.model_selection import train_test_split
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, random_state=7)

from sklearn.svm import SVC
model = SVC(kernel='linear', C=10)
model.fit(Xtrain, ytrain)
y_model = model.predict(Xtest)

from sklearn.metrics import accuracy_score
accuracy_score(ytest, y_model)
```


```python
pca = PCA(8)

X = pca.fit_transform(cleansedDataSet)
y = cleansedDataSet['SalePrice']

from sklearn.model_selection import train_test_split
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, random_state=7)

from sklearn.svm import SVC
model = SVC(kernel='linear', C=10)
model.fit(Xtrain, ytrain)
y_model = model.predict(Xtest)

from sklearn.metrics import accuracy_score
accuracy_score(ytest, y_model)
```


```python
pca = PCA(6)

X = pca.fit_transform(cleansedDataSet)
y = cleansedDataSet['SalePrice']

from sklearn.model_selection import train_test_split
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, random_state=7)

from sklearn.svm import SVC
model = SVC(kernel='linear', C=10)
model.fit(Xtrain, ytrain)
y_model = model.predict(Xtest)

from sklearn.metrics import accuracy_score
accuracy_score(ytest, y_model)
```


```python
pca = PCA(4)

X = pca.fit_transform(cleansedDataSet)
y = cleansedDataSet['SalePrice']

from sklearn.model_selection import train_test_split
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, random_state=7)

from sklearn.svm import SVC
model = SVC(kernel='linear', C=10)
model.fit(Xtrain, ytrain)
y_model = model.predict(Xtest)

from sklearn.metrics import accuracy_score
accuracy_score(ytest, y_model)
```


```python
pca = PCA(3)

X = pca.fit_transform(cleansedDataSet)
y = cleansedDataSet['SalePrice']

from sklearn.model_selection import train_test_split
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, random_state=7)

from sklearn.svm import SVC
model = SVC(kernel='linear', C=10)
model.fit(Xtrain, ytrain)
y_model = model.predict(Xtest)

from sklearn.metrics import accuracy_score
accuracy_score(ytest, y_model)
```


```python
pca = PCA(2)

X = pca.fit_transform(cleansedDataSet)
y = cleansedDataSet['SalePrice']

from sklearn.model_selection import train_test_split
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, random_state=7)

from sklearn.svm import SVC
model = SVC(kernel='linear', C=10)
model.fit(Xtrain, ytrain)
y_model = model.predict(Xtest)

from sklearn.metrics import accuracy_score
accuracy_score(ytest, y_model)
```

# PART 6 - clustering the PCA-transformed dataset

In this step, we will use k-means clustering to view both PCA components. 

In order to do this, we will first fit these principal components to the k-means algorithm and determine the best number of clusters. 

Determining the ideal number of clusters for our k-means model can be done by measuring the sum of the squared distances to the nearest cluster center aka inertia (more of this in the report).

Much like the scree plot for PCA seen previously, the k-means scree plot below indicates the percentage of variance explained, but in slightly different terms, as a function of the number of clusters.

## (a) Determining the number of clusters:


```python
ks = range(1, 10)
inertias = []
for k in ks:
    # Create a KMeans instance with k clusters: model
    model = KMeans(n_clusters=k)
    
    # Fit model to samples
    model.fit(PCAtransformedDataSet)
    
    # Append the inertia to the list of inertias
    inertias.append(model.inertia_)
    
plt.plot(ks, inertias, '-o', color='black')
plt.xlabel('number of clusters, k')
plt.ylabel('inertia')
plt.xticks(ks)
plt.show()
```
From the diagram above, we now know that the optimal number of clusters according to the elbow curve has been identified as 2. 

Therefore, we set n_clusters equal to 2, and upon generating the k-means output use the data originally transformed using pca in order to plot the clusters.
## (b) Plotting the clusters (with centroids):

Plotting:


```python
kmeans = KMeans(n_clusters=2)
X_clustered = kmeans.fit_predict(PCAtransformedDataSet)

LABEL_COLOR_MAP = {0 : 'r', 1 : 'g'}
label_color = [LABEL_COLOR_MAP[l] for l in X_clustered]

plt.figure(figsize=(7,7))
plt.scatter(PCAtransformedDataSet[:,0],PCAtransformedDataSet[:,1],c=label_color,alpha=0.5)

centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5);
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('2-cluster K-Means')

plt.show()
```

The scree plot above shows some clearly defined clusters in the data. 

Now that we know how many clusters there are in our data, we have a better sense of how many groups we can label the PCA-transformed dataset with. 

It is certainly our intentions to implement with a model that grades house sale prices in the dataset based on two classes: EXPENSIVE and AFFORDABLE. 

Introducing these labels back into the reduced dataset on the unique id of each sample would allow us to visualize them by cluster.

The ability to notice otherwise unseen patterns and to come up with a model to generalize those patterns onto observations is precisely why tools like PCA and k-means are essential in any data scientist’s toolbox. They allow us to see the big picture while we pay attention to the details.

# PART 7 - attempting to classify the clustered, PCA-transformed dataset

We can already pre-test our PCA data for classification purposes (this would be fully implemented *after* having
done clustering...)

from: https://stats.stackexchange.com/questions/144439/applying-pca-to-test-data-for-classification-purposes

PCA is a dimension reduction tool, not a classifier. In Scikit-Learn, all classifiers and estimators have a 
predict method which PCA does not. 
You need to fit a classifier on the PCA-transformed data. Scikit-Learn has many classifiers. 
Here is an example of using a decision tree on PCA-transformed data. I chose the decision tree classifier as 
it works well for data with more than two classes which is the case with the iris dataset.

This can be achieved by using Pipeline.


```python
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

# The following fits PCA, transforms the data and fits the decision tree classifier onto the transformed data:

pipe = Pipeline([('pca', PCA()), ('tree', DecisionTreeClassifier())])

pipe.fit(PCAtransformedDataSet, cleansedDataSet['SalePrice'])
pipe.predict(PCAtransformedDataSet)

# pipe.fit(cleansedDataSet, cleansedDataSet['SalePrice'])
# pipe.predict(cleansedDataSet)
```

We need code to transform numpy array (PCAtransformedDataSet) into a dataframe.

Info on how to do so:

https://stackoverflow.com/questions/20763012/creating-a-pandas-dataframe-from-a-numpy-array-how-do-i-specify-the-index-colum


```python
finalDataSet = pd.DataFrame({'expensive': PCAtransformedDataSet[:, 0], 'affordable': PCAtransformedDataSet[:, 1]})
```

The following requires uploading PYDOTPLUS on Anaconda prompt, as so:

      conda install -c conda-forge pydotplus

(info found here: https://anaconda.org/conda-forge/pydotplus)


```python
# code for visualizing a decision tree found here:
# https://chrisalbon.com/machine_learning/trees_and_forests/visualize_a_decision_tree/
# https://www.datacamp.com/community/tutorials/decision-tree-classification-python

from sklearn.tree import DecisionTreeClassifier
from sklearn import datasets
from IPython.display import Image  
from sklearn import tree
import pydotplus

X = finalDataSet
y = cleansedDataSet['SalePrice']

# Create decision tree classifer object
clf = DecisionTreeClassifier(random_state=0)

# Train model
model = clf.fit(X, y)

# Create DOT data
dot_data = tree.export_graphviz(clf, out_file=None, 
                                feature_names=finalDataSet.columns.values, 
                                class_names=[X, y]) # NO LABELS AVAILABLE!

# Draw graph
graph = pydotplus.graph_from_dot_data(dot_data)  

# Show graph
Image(graph.create_png())

```
