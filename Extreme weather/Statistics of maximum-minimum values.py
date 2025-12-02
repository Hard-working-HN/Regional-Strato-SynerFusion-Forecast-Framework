import pandas as pd

'''
# ===================================================================================Max===================================================================================
input_path = r'F:\Base_Year_Data\Max_Tem.xlsx'
output_path = r'F:\Base_Year_Data\Base_Max_Tem.xlsx'
 
df = pd.read_excel(input_path)

df.iloc[:, 0] = pd.to_datetime(df.iloc[:, 0])
df.set_index(df.columns[0], inplace=True)

years = df.index.year.unique()

max_val_dict = {}
second_val_dict = {}
max_date_dict = {}
second_date_dict = {}

for county in df.columns:
    max_vals = []
    second_vals = []
    max_dates = []
    second_dates = []

    for year in years:
        data = df.loc[df.index.year == year, county].dropna()

        if data.empty:
            max_vals.append(float('nan'))
            second_vals.append(float('nan'))
            max_dates.append(pd.NaT)
            second_dates.append(pd.NaT)
        else:
            sorted_data = data.sort_values(ascending=False)
            max_vals.append(sorted_data.iloc[0])
            max_dates.append(sorted_data.index[0])

            if len(sorted_data) > 1:
                second_vals.append(sorted_data.iloc[1])
                second_dates.append(sorted_data.index[1])
            else:
                second_vals.append(float('nan'))
                second_dates.append(pd.NaT)

    max_val_dict[county] = max_vals
    second_val_dict[county] = second_vals
    max_date_dict[county] = max_dates
    second_date_dict[county] = second_dates

max_val_df = pd.DataFrame(max_val_dict, index=years)
second_val_df = pd.DataFrame(second_val_dict, index=years)
max_date_df = pd.DataFrame(max_date_dict, index=years)
second_date_df = pd.DataFrame(second_date_dict, index=years)

with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
    max_val_df.to_excel(writer, sheet_name='max value')
    max_date_df.to_excel(writer, sheet_name='max date')
    second_val_df.to_excel(writer, sheet_name='second value')
    second_date_df.to_excel(writer, sheet_name='second date')

'''

# ===================================================================================Min===================================================================================

input_path = r'F:\Base_Year_Data\Min.xlsx'
output_path = r'F:\Base_Year_Data\Base_Min.xlsx'

df = pd.read_excel(input_path)

df.iloc[:, 0] = pd.to_datetime(df.iloc[:, 0])
df.set_index(df.columns[0], inplace=True)

years = df.index.year.unique()

min_val_dict = {}
second_val_dict = {}
min_date_dict = {}
second_date_dict = {}

for county in df.columns:
    min_vals = []
    second_vals = []
    min_dates = []
    second_dates = []

    for year in years:
        data = df.loc[df.index.year == year, county].dropna()

        if data.empty:
            min_vals.append(float('nan'))
            second_vals.append(float('nan'))
            min_dates.append(pd.NaT)
            second_dates.append(pd.NaT)
        else:
            sorted_data = data.sort_values(ascending=True)  
            min_vals.append(sorted_data.iloc[0])
            min_dates.append(sorted_data.index[0])

            if len(sorted_data) > 1:
                second_vals.append(sorted_data.iloc[1])
                second_dates.append(sorted_data.index[1])
            else:
                second_vals.append(float('nan'))
                second_dates.append(pd.NaT)

    min_val_dict[county] = min_vals
    second_val_dict[county] = second_vals
    min_date_dict[county] = min_dates
    second_date_dict[county] = second_dates

min_val_df = pd.DataFrame(min_val_dict, index=years)
second_val_df = pd.DataFrame(second_val_dict, index=years)
min_date_df = pd.DataFrame(min_date_dict, index=years)
second_date_df = pd.DataFrame(second_date_dict, index=years)

with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
    min_val_df.to_excel(writer, sheet_name='min value')
    min_date_df.to_excel(writer, sheet_name='min date')
    second_val_df.to_excel(writer, sheet_name='second value')
    second_date_df.to_excel(writer, sheet_name='second date')

