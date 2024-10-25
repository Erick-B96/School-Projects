#Erick Barron
#Introduction to the Dataset
#added cell 1
import pandas as pd
import numpy as np
#added cell 2
from sklearn.preprocessing import MinMaxScaler
#added cell 3
import matplotlib.pyplot as plt
import seaborn as sns
#added cell 4
from sklearn.cluster import KMeans
from sklearn import metrics
#added cell 5
from sklearn.metrics import silhouette_score
#added in cell 6
from sklearn.decomposition import PCA

#Load the dataset
colleges = pd.read_csv('College.csv', index_col=0)

#display first few rows using head()
print(colleges.head())

#Display a summary of dataset
print(colleges.info())
colleges.describe()

#initial observations are that it does what it says it will
#this step was pretty straightforward
#It takes the data and prints the first 5 colleges
#It also breaks up the info into a more readable form that displays relevent info together
#however maybe that is specific to this assignment
#Data Preprocessing
#check if there are any missing values can see which method you should use
missing_values = colleges.isnull().sum()
print("num of missing values:\n", missing_values)

#using imputation which I chose since there was no missing values
#I wanted to keep as much data as possible so decided on this method
#This also encoded the data I changed the yes/no for private since it was causing me trouble and not running
colleges_encoded = pd.get_dummies(colleges, columns=['Private'])

#Normalize the data using minmax since I read that if we use columns it is better due to bounding it
scaler = MinMaxScaler()
colleges_normalized = pd.DataFrame(scaler.fit_transform(colleges_encoded), columns=colleges_encoded.columns)

#Here we show the statistics after preprocessing
print("Before Imputation and Encoding:\n", colleges.describe())
print("\nAfter Imputation, Encoding, and Normalization:\n", colleges_normalized.describe())
#The output shows that the data has no missing values
#also we can see the data as it was originally and the data after our processes 
#it is more readable and easier to work with
#Exploratory Data Analysis
#Histogram
#We get the random data to analyze
data1 = np.random.randn(1000)
data2 = np.random.normal(loc=3, scale=1, size=1000)
#Create the histogram 
plt.hist([data1, data2], bins=30, stacked=True, color=['Red', 'Blue'], edgecolor='black')
#Add labels to histogram and display
plt.xlabel('Values')
plt.ylabel('Frequency')
plt.title('Histogram')
plt.legend(['Datatset 1', 'Dataset 2'])
plt.show()

#Boxplots
#Create dataset essentially find random numbers to use
np.random.seed(10)
data_1 = np.random.normal(100, 10, 200)
data_2 = np.random.normal(90, 20, 200)
data_3 = np.random.normal(80, 30, 200)
data_4 = np.random.normal(70, 40, 200)
data = [data_1, data_2, data_3, data_4]
 
fig = plt.figure(figsize =(10, 6))
 
#Create axes instance and plot then print
ax = fig.add_axes([0, 0, 1, 1])
bp = ax.boxplot(data)
plt.show()

#ScatterPlots
#Create random data
x = np.random.rand(50)
y = np.random.rand(50)
colors = np.random.rand(50)
sizes = 100 * np.random.rand(50)
 
#Create a scatter plot
plt.scatter(x, y, c=colors, s=sizes, alpha=0.7, cmap='Greens')
 
#Add labels to scatter plot
plt.title("Scatter Plot")
plt.xlabel("X-axis")
plt.ylabel("Y-axis")

plt.colorbar(label='Color Intensity')
#Print
plt.show()

#Summary Statistics compute and display important statistics
summary_stats = colleges.describe()
print("\nSummary Statistics:\n", summary_stats)

#Correlation matrix and heatmap
correlation_matrix = colleges_normalized.corr()
plt.figure(figsize=(10, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix Heatmap')
plt.show()

#This section was challenging but straightforward just create the graphs its cool to play around with
#The statistics seems to be the same as the first section so maybe I did something wrong but not sure
#The heatmap I should study to see exaclty what it is saying
#Implementation of K-Means clustering
#Choose the number of clusters using the elbow method
inertia = []
for k in range(1, 10):
    kmeans = KMeans(n_clusters=k, random_state=20)
    kmeans.fit(colleges_normalized)
    inertia.append(kmeans.inertia_)

#Plot using the the elbow method
plt.figure(figsize=(10, 6))
plt.plot(range(1, 10), inertia, marker='o')
plt.title('Elbow Method for Optimal Clusters')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.show()

#Our optimal number of clusters
optimal_clusters = 3 

#Apply K-means clustering with the optimal clusters
kmeans = KMeans(n_clusters=optimal_clusters, random_state=20)
kmeans.fit(colleges_normalized)

#Display cluster centers and labels
cluster_centers = pd.DataFrame(scaler.inverse_transform(kmeans.cluster_centers_), columns=colleges_encoded.columns)
print("\nCluster Centers:\n", cluster_centers)

#Add cluster labels
#Display the final clustering results
colleges['Cluster_Labels'] = kmeans.labels_
print("\nFinal Results:\n", colleges[['Cluster_Labels']])

#Visualization of clustered data
#x and y are columns you want to analyze in the dataset 
#for this example I am analyzing acceptance and enrollement
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Accept', y='Enroll', hue='Cluster_Labels', data=colleges, palette='viridis', legend='full')
plt.title('K-means Clustering Results')
plt.show()

#The elbow shows a reduction in inertia signaling lower cluster qualities
#3 clusters is optimal as it seems to be a good sweet spot before lower quality and oversimplifying
#Evaluation of Clusters
#Silhouette Score
#measure how similar an object is to won cluster as opposed to others
silhouette_avg = silhouette_score(colleges_normalized, kmeans.labels_)
print(f"\nSilhouette Score: {silhouette_avg}")

#Within-cluster sum of squares aka Inertia
#check how coherent clusters are. Balance low Inertia and good number of clusters
#Lower Inertia is better defined, but higher K means lower Inertia
inertia_value = kmeans.inertia_
print(f"\nWithin-cluster Sum of Squares or Inertia: {inertia_value}")
#Visualization
#Apply PCA to reduce dimension to 2D
pca = PCA(n_components=2)
colleges_pca = pd.DataFrame(pca.fit_transform(colleges_normalized), columns=['PC1', 'PC2'])

#Concatenate cluster labels to the reduced-dimensional dataset
colleges_pca['Cluster_Labels'] = kmeans.labels_

#Plot clusters in reduced-dimensional
#good for seeing patterns taht would be hard to see otherwise
plt.figure(figsize=(10, 6))
sns.scatterplot(x='PC1', y='PC2', hue='Cluster_Labels', data=colleges_pca, palette='viridis', legend='full')
plt.title('K-means Clustering Results in Reduced-dimensional Space (PCA)')
plt.show()

#We use PCA since SVD is for sparse data
#Each point is a college
#The more seperated the clusters the better the algorithm is working
#Interpretation and Insights
#One thing I would like to point out is the heatmap shows correlation closer 1 being good and -1 bad with 0 being neutral.
#Darker colors are stronger correlated and lighter weaker.
#This insight may help us understand colleges better in the way acceptance is used, graduation rates, needs, etc.
#Using this data is important for understanding colleges and seeing trends we normally wouldn't.
#Reflection
#I thought this was a fun assignment as I have never used any of the software used.
#Getting the results gave me a sense of accomplishment and wonder. The capabilities of programs is amazing.
#I found learning and using python was not difficult although I did have to research and learn on the go.
#I think playing around with different datasets will improve my understanding more.
#Upon completeing the assignment I think I will study clustering and how exactly it should be read and interpreted.