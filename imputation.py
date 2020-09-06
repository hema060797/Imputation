import numpy as np 

# Importing the SimpleImputer class 
from sklearn.impute import SimpleImputer 

# Imputer object using the mean strategy and 
# missing_data type for imputation 
imputer = SimpleImputer(missing_values = np.nan, 
						strategy ='mean') 

data = [[12, np.nan, 34], [10, 32, np.nan], 
		[np.nan, 11, 20]] 

print("Original Data : \n", data) 
# Fitting the data to the imputer object 
imputer = imputer.fit(data) 

# Imputing the data	 
data = imputer.transform(data) 

print("Imputed Data : \n", data) 
