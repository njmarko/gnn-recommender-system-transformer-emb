import csv
import os
import pandas as pd

# this is how ratings.csv looks like in data/ml-latest-small folder
# userId,movieId,rating,timestamp
# create a new CSV file with just userId's from ratings.csv

input_file = os.path.join(os.getcwd(), 'data', 'ml-latest-small', 'ratings.csv')
output_file = os.path.join(os.getcwd(), 'data', 'users.csv')

# Read the ratings.csv file using pandas
df = pd.read_csv(input_file)

# Extract the unique user IDs from the 'userId' column
user_ids = df['userId'].unique()

# Create a new DataFrame with the user IDs
users_df = pd.DataFrame({'userId': user_ids})

# Save the DataFrame to the output file
users_df.to_csv(output_file, index=False)