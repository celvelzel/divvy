import pandas as pd
import os

# Load the Excel file into a DataFrame
# file_path = 'C:\\Users\\celcelcel\\Desktop\\test\\2019.10+11+12.xlsx'
file_path = 'C:\\Users\\celcelcel\\Desktop\\19Q12\\Divvy_Trips_2019_Q2.xlsx'
df = pd.read_excel(file_path)

# Extract the month from the 'start_time' column and create new columns for October, November, and December
df['start_time_month'] = df['start_time'].dt.month

# Split the data into separate DataFrames for October, November, and December
df_1 = df[df['start_time_month'] == 4]
df_2 = df[df['start_time_month'] == 5]
df_3 = df[df['start_time_month'] == 6]
# df_1 = df[df['start_time_month'] == 10]
# df_2 = df[df['start_time_month'] == 11]
# df_3 = df[df['start_time_month'] == 12]

# Save the DataFrames to separate Excel files
output_folder = 'C:\\Users\\celcelcel\\Desktop\\19Q12'
os.makedirs(output_folder, exist_ok=True)

# df_1.to_excel(os.path.join(output_folder, '2019.10.xlsx'), index=False)
# df_2.to_excel(os.path.join(output_folder, '2019.11.xlsx'), index=False)
# df_3.to_excel(os.path.join(output_folder, '2019.12.xlsx'), index=False)
df_1.to_excel(os.path.join(output_folder, '2019.4.xlsx'), index=False)
df_2.to_excel(os.path.join(output_folder, '2019.5.xlsx'), index=False)
df_3.to_excel(os.path.join(output_folder, '2019.6.xlsx'), index=False)

'Files have been created: {output_folder}/October_data.xlsx, {output_folder}/November_data.xlsx, {output_folder}/December_data.xlsx'
