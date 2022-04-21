# C8074 Commodities in India - Where can I get a cheap bag of Rice? 


## Introduction: 

Github: GitHub - SConstant/C7804-Big-Data-and-Decision-Making [click me](https://github.com/SConstant/C7804-Big-Data-and-Decision-Making)
Kaggle: Food Prices In India | Kaggle [click me] (https://www.kaggle.com/datasets/arshadali12/food-prices-in-india)

The Food prices in India dataset is taken as a subset of the data provided by the World Food Programme Price Database. 
The World Food Programme Price Database contains information around food as a commodity for 98 countries and 3000 markets. Though data collection dates back from 1992, much of the data is from 2003 or thereafter. This is due to countries choosing to submit data in later years. It is updated weekly however displays mostly monthly data.
This particular dataset is concerned specifically with the food commodities in India. A table of the commodities is listed below, and these are a price ranging from 0.056 USD to 88.7 USD with a mean of 1.5 USD. Of the commodities Rice was the most frequent sale. 
The focus of this assignment is to see how the prices of rice differed in the Markets of different regions, and volumes of sales. There was also interest in what the application of a deep learning technique would look like on the entire dataset. 

---


## Methods

Primarily Saturn Galaxy was used for the access to the NVIDIA GPU cluster via Jupyter server, with this decision was taken due to the user friendliness of access to a reasonably universal Jupyter notebook. The environment grants access to a tesla T4 GPU (Graphics Processing Unit), containing 320 Turing tensor cores (NVIDIA 2019). This lending itself well to Big data problems and exploration as the increased performance allows much less the time taken in the computation of data and execution of programming commands and functions with an natural expectation and subsequent organic growth towards uses in machine and deep learning training and modelling.  

Due to the limitation of 30 hours GPU, and 3 hours Dask Cluster access, much of the data wrangling and some of the EDA was carried out in Google Colab. This notebook is in the Github here [click me](https://github.com/SConstant/C7804-Big-Data-and-Decision-Making/blob/main/Data_EDA.ipynb).The techniques used for the data analysis include DBSCAN and K-Nearest Neighbours. 

Within google colab, with the CPU Pandas was used to explore the data as part of EDA (Exploratory Data Analysis) with a look at the counts of various columns, and the means, standard deviations, min and max values as well as the quartiles for USD. Histograms were also used to explore the shape of the data, with USD_Price (value in dollars), Date and Market being of particular interest. 

The decision was taken to use the price of commodities in dollars as this acts as a unifying and so universal currency. The data was skewed heavily, as expected for all three columns. As mentioned in the introduction there was more data collection in later years and this would have an impact on the frequency of values in all other columns. As USD_Price was heavily skewed, a box plot was chosen to further explore, which showed early signs of clustering towards a higher frequency of low prices with many outliers in terms of market fluctuation. 

Following the EDA and Data Wrangling, the Jupyter server was used within the Saturn Cloud environment. The indian_commodities_geo_date.csv was called via github and parsed as both a pandas dataframe and CuDF dataframe. The reason for this being that some simple dataframe interrogations were simpler or possible within Pandas and it was assumed that the data would remain the same between the two types of dataframe in terms of indexing. CuDF is a Python GPU DataFrame library (built on the Apache Arrow columnar memory format) for loading, joining, aggregating, filtering, and otherwise manipulating data (NVIDA 2019). 

A small amount of exploration was carried out in the CuDF dataframe expediated by the previous EDA. The data types were changed for float 64 bit to float 32 in order to be compatible with the functionality of the CuML (Compute Unified Machine Learning) package. This was called into the notebook to facilitate the deep learning techniques described below. 

DBSCAN (Density-based spatial clustering of applications with noise) was chosen as a non-parametric technique and unsupervised learning. The algorithm is used to find associations and structures in data or patterns which can then be used to predict trends. CuML’s DBSCAN expects an array-like object or cuDF DataFrame, and constructs an adjacency graph to compute the distances between close neighbours (docs.rapids.ai, n.d.). The particular application here was to try and ascertain clusters where the most rice was sold. One of the requirements of DBscan within CuML is parsing float32 numerical or integer data and in the case of Market location this was given as a categorical object. As such a the geocoding of the Market data was required and was carried out using the GPU available via google. Initially the Google Geocoding API was explored though superior there was concern around the potential fiscal burden so the Open streetmap API was selected instead. The code is within the notebook Open.ipynb saved in the Github. It is possible to use CudF alongside geopandas however while much of the functionality is synonymous there are some gaps between CuDF and Pandas. Compatibility was a concern and it was felt more efficient to geocode through pandas. Following the addition of Latitude and Longitude, the dataframe was saved as a CSV, downloaded and reuploaded to Github. 
After some research into maps, it was decided to convert the Latitude and Longitude data into Easting and Northing for a more accurate and globally recognised Transverse Mercator projection. The code to create easting and Northing coordinates was taken verbatim from the Jupyter notebook 2-02_population_viz.ipynb. 

Following a DBSCAN instance was created, and a new CuDF dataframe object by way of subsetting to include rows with rice only. The rice sales were then clustered using the DBSCAN instance on the Transverse Mercator values Northing and Easting. These values were then put into a new column, which allowed the number of clustered to be identified. KNN (K-Nearest neighbours) was selected to try and identify which Markets and dates were likely to have lower prices specifically in the staple rice. Rice was selected as this was the commodities with the most sales, subsequently most data as well as interest.

A KNN instance was created, with the K value, 5 was chosen as a default, Following this, the Markets column was fitted with using the knn.fit method. Following this knn.neighbours was used to cluster the markets and dates with USD_Price.  As the K value selected was 5, 5 locations would be returned. 
It was then possible to explore individual datapoints, a price in dollars was selected via index (initially index 10 to try out the code) and assigned a new object, which using the iloc function was use to cross reference the markets. 

A decision was taken to look at the clustering around the lower prices, the lower quartile had a USD value of 0.33 or lower and yielded several thousand results. The lowest 0.01% of the Prices was chosen and returned with the index data. The lowest USD value 0.1405 was explored.


## Results

DBSCAN, the number of clusters which were identified for rice only was 150, though not included for clarity when DBSCAN was run for the entire dataset, 160 clustered were identified. There is also a data visualisation which displays the scatter across the Northing and Easting of the clusters of which the code is below. This data visualisation would be improved by the addition of a country outline to better contextualise the areas of cluster. 
Please see the link and run the code snippet below for the graph


my_url ='http://j-knitk-c7084-data-lab-666823ad3bda4c55bef6749866ebe22d.community.saturnenterprise.io'

dash.show(my_url, port=8789)

For KNN the clusters returned the data rows at indices 5279, 5278, 5277, 5276 and 5275, and these rows were then called via iloc.  

Please see the table below for the Indices location, and values for date, Market location and Price in USD

##### Cluster

|Index      |Market       |Date            |USD Price      |
|-----------|-------------|----------------|---------------|
|5277       |Aizawl       |15/11/2020      |0.4032         |
|5279       |Aizawl       |16/03/2021      |0.5832         |
|5278       |Aizawl       |16/02/2021      |0.5832         |
|5276       |Aizawl       |16/10/2020      |0.4083         |
|5275       |Aizawl       |16/12/2019      |0.4761         |

##### Selected Row

|Index      |Market       |Date            |USD Price      |
|-----------|-------------|----------------|---------------|
|5053       |Aizawl       |16/11/1996      |0.1405         |


---

## Discussion: 

##### DBSCAN 

Previously, DBSCAN had been used on the entire dataset and this was removed for clarity. It’s interesting but unsurprising that rice only yielded 10 more clusters that the entire dataset (15 compare to 160 in the entire dataset) This could be due to the commodity having the greatest number of sales. It would be interesting to identify clusters on commodities with fewer instances such as chickpeas for comparison however if there is only a small amount of data it’s possible that the technique DBSCAN may not be suitable. K-means clustering may prove better for the subsets of smaller datasets however there would need to be a literature review on the fidelity and compatibility in terms of any inference should both of these techniques be used. 

##### KNN 

While the K value 5 was chosen by default, as computationally safe, relatively inexpensive and a purpose in identifying 5 locations however this could have been improved through exploring the error rate or rate of accuracy and choosing a K value with the minimum error rate or maximal accuracy rate. Deep learning can be used here again to train and fit the model in order to identify an optimal K-value. Upon exploring the cluster itself it was interesting see that the dates vary from the original selected row 5053 being in the year 1996, with the years of the cluster ranging from 2019 to 2021. It would be interesting to explore more parameters for KNN, to see if this is consistent or potentially an error in the method. The price in USD also varied further with the range being from 0.40 to 0.58 dollars, which is further away than expected but interesting that these were defined as the nearest neighbours. The one consistency in the data is that of the Market value being Aizawl, which given its presence is inferred from the above as the market that experiences the lowest prices in the dataset. However to be able to infer this with more confidence there would need to be more exploration of the parameters for KNN, and possibly an attempt at a different analysis clustering technique such as K-Means performed on each of the indices included in the cluster to see how different the results yielded would be. The Dask K-means functionality would also be interesting to use to see how different speeds could be achieved in this data analysis but also to do further analysis and comparison in terms of the different locations of markets and Prices. Finally it would be interesting to look at a more predictive approach to this data using XGboost as part of the cuda toolkit to explore the probabilities of different Prices of rice appearing in different locations. 


##### References

1.	Google Developers. 2022. Overview  |  Geocoding API  |  Google Developers. [ONLINE] Available at: https://developers.google.com/maps/documentation/geocoding/overview. [Accessed 20 April 2022].

2.	A Guide to Coordinate Systems in Great Britain An introduction to mapping coordinate systems and the use of GNSS datasets with Ordnance Survey mapping. (n.d.). [online] Available at: https://www.ordnancesurvey.co.uk/documents/resources/guide-coordinate-systems-great-britain.pdf.

3.	API v0.6 - OpenStreetMap Wiki. 2022. API v0.6 - OpenStreetMap Wiki. [ONLINE] Available at: https://wiki.openstreetmap.org/wiki/API_v0.6. [Accessed 20 April 2022].


4.	Natassha Selvaraj. 2022. Geocoding in Python: A Complete Guide. [ONLINE] Available at: https://www.natasshaselvaraj.com/a-step-by-step-guide-on-geocoding-in-python/. [Accessed 20 April 2022].

5.	Abdishakur (2020). How To Handle Map Projections Properly In Python. [online] Medium. Available at: https://towardsdatascience.com/how-to-handle-map-projections-properly-in-python-bcbff78895c4 [Accessed 21 Apr. 2022].


6.	GitHub. 2022. GitHub - rapidsai/cudf: cuDF - GPU DataFrame Library. [ONLINE] Available at: https://github.com/rapidsai/cudf. [Accessed 20 April 2022].

7.	docs.rapids.ai. (n.d.). cuML API Reference — cuml 22.04.00 documentation. [online] Available at: https://docs.rapids.ai/api/cuml/stable/api.html#dbscan [Accessed 20 Apr. 2022].


8.	kururu002 (2022). kururu002/Fundamentals-of-Accelerated-Data-Science-with-RAPIDS. [online] GitHub. Available at: https://github.com/kururu002/Fundamentals-of-Accelerated-Data-Science-with-RAPIDS [Accessed 21 Apr. 2022].

9.	kururu002 (2022). kururu002/Fundamentals-of-Accelerated-Data-Science-with-RAPIDS. [online] GitHub. Available at: https://github.com/kururu002/Fundamentals-of-Accelerated-Data-Science-with-RAPIDS/blob/main/2-02_population_viz.ipynb [Accessed 20 Apr. 2022].


10.	Kim, A.Y., Escobedo-Land, A., 2015. OkCupid Data for Introductory Statistics and Data Science Courses. null 23, null-null. https://doi.org/10.1080/10691898.2015.11889737

11.	Shi, Q., Abdel-Aty, M., 2015. Big Data applications in real-time traffic operation and safety monitoring and improvement on urban expressways. Transportation Research Part C: Emerging Technologies, Big Data in Transportation and Traffic Engineering 58, 380–394. https://doi.org/10.1016/j.trc.2015.02.022


12.	Günther, W.A., Rezazade Mehrizi, M.H., Huysman, M., Feldberg, F., 2017. Debating big data: A literature review on realizing value from big data. The Journal of Strategic Information Systems 26, 191–209. https://doi.org/10.1016/j.jsis.2017.07.003

13.	Guo, H., 2017. Big Earth data: A new frontier in Earth and information sciences. Big Earth Data 1, 4–20. https://doi.org/10.1080/20964471.2017.1403062

14.	Pham, X., Stack, M., 2018. How data analytics is transforming agriculture. Business Horizons 61, 125–133. https://doi.org/10.1016/j.bushor.2017.09.011

15.	docs.rapids.ai. (n.d.). Training and Evaluating Machine Learning Models in cuML — cuml 22.02.00 documentation. [online] Available at: https://docs.rapids.ai/api/cuml/legacy/estimator_intro.html [Accessed 21 Apr. 2022].


16.	NVIDIA Developer. (2021). Run NVIDIA Jupyter Notebooks. [online] Available at: https://developer.nvidia.com/run-jupyter-notebooks [Accessed 21 Apr. 2022].

17.	Band, A. (2020). How to find the optimal value of K in KNN? [online] Medium. Available at: https://towardsdatascience.com/how-to-find-the-optimal-value-of-k-in-knn-35d936e554eb.


18.	Acevedo, M.F. (2013). Data analysis and statistics for geography, environmental science, and engineering. Boca Raton: Crc Press.

19.	Maklin, C. (2019). K Nearest Neighbor Algorithm In Python. [online] Medium. Available at: https://towardsdatascience.com/k-nearest-neighbor-python-2fccc47d2a55.


20.	Schubert, E., Sander, J., Ester, M., Kriegel, H.P. and Xu, X., 2017. DBSCAN revisited, revisited: why and how you should (still) use DBSCAN. ACM Transactions on Database Systems (TODS), 42(3), pp.1-21.

