workflow:
(1):preprocess.py   ---> preprocess data
(2):CF.py           ---> calculate CF using Lowess
(3):main_lastest.py ---> models(L2 Norm; One-Class SVM; iForest; RandNet; WBiGan-GP)

others:
utils.py: geotools using rasterio and geopandas
test.ipynb: code test
Lowess.py: calculate lowess
...
The data and results will be uploaded later.
