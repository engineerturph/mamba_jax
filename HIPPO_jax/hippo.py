import pandas as pd 

# Load the data
data = pd.read_csv("data.csv")

# Print the first 5 rows of the data
print(data.head())

# Print the shape of the data
print(data.shape)

# Print the columns of the data
print(data.columns)

# Print the data types of the columns
print(data.dtypes)
