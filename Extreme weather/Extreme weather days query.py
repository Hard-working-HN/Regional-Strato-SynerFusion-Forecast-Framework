import pandas as pd
import os

threshold_file = r"F:\Base_Year_Data\SPEI.xlsx"
rain_file = r"F:\Com_Year_Data\SPEI_2013_2022.xlsx"
output_file = rain_file                         

threshold_df = pd.read_excel(threshold_file, sheet_name="threshold")
rain_df = pd.read_excel(rain_file)

counties = threshold_df.columns
date_series = rain_df.iloc[:, 0]
rain_data = rain_df.iloc[:, 1:] 

rain_exceed_dict = {}
date_exceed_dict = {}

for county in counties:
    if county not in rain_data.columns:
        continue

    threshold = threshold_df[county].iloc[0]
    values = rain_data[county]
    exceed_mask = values < threshold

    rain_exceed_dict[county] = values[exceed_mask].reset_index(drop=True)
    date_exceed_dict[county] = date_series[exceed_mask].reset_index(drop=True)

rain_exceed_df = pd.concat(rain_exceed_dict, axis=1)
date_exceed_df = pd.concat(date_exceed_dict, axis=1)


with pd.ExcelWriter(output_file, engine="openpyxl", mode="a", if_sheet_exists="replace") as writer:
    rain_exceed_df.to_excel(writer, sheet_name="Value", index=False)
    date_exceed_df.to_excel(writer, sheet_name="Date", index=False)

