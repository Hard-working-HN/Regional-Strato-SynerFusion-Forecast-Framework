import pandas as pd
import numpy as np
from prophet import Prophet

file_path = 'Origin_Data.xlsx' 
raw_df = pd.read_excel(file_path)

year_columns = [col for col in raw_df.columns if isinstance(col, int)]
year_int_columns = year_columns

final_results = []

for city, group in raw_df.groupby('Prefecture-Level'):
    available_years = []
    year_to_sum_col = {2020: '2020_sum', 2021: '2021_sum', 2022: '2022_sum'}

    for year, sum_col in year_to_sum_col.items():
        if sum_col in group.columns and not group[sum_col].dropna().empty:
            available_years.append(year)

    if not available_years:
        print(f"{city} pass")
        continue

    city_totals = [group[year_to_sum_col[year]].dropna().iloc[0] for year in available_years]

    county_data = group.set_index('County_Level')[year_columns]
    county_data = county_data.T  

    predictions = pd.DataFrame(index=available_years)

    for county in county_data.columns:
        series = county_data[county].dropna()
        if len(series) < 2:
            continue  

        df_prophet = pd.DataFrame({
            'ds': pd.to_datetime(series.index.astype(str), format='%Y'),
            'y': series.values
        })

        model = Prophet(
            yearly_seasonality=False,
            weekly_seasonality=False,
            daily_seasonality=False,
            seasonality_mode='additive'
        )
        model.fit(df_prophet)

        future = pd.DataFrame({
            'ds': pd.to_datetime(available_years, format='%Y')
        })

        forecast = model.predict(future)
        predictions[county] = forecast['yhat'].values

    predictions['Sum'] = predictions.sum(axis=1)
    predictions['Adjust_factor'] = np.array(city_totals) / predictions['Sum']

    for county in predictions.columns[:-2]:
        predictions[county] = predictions[county] * predictions['Adjust_factor']

    predictions = predictions.drop(columns=['Sum', 'Adjust_factor'])
    predictions['Prefecture-Level'] = city
    predictions = predictions.set_index('Prefecture-Level', append=True)
    final_results.append(predictions)

final_allocated = pd.concat(final_results)
final_allocated = final_allocated.round(0)

output_path = 'Prophet_result.xlsx'
final_allocated.to_excel(output_path, merge_cells=False)
