import pandas as pd
import numpy as np
from prophet import Prophet

# 读取数据
file_path = '死亡人数_原始数据.xlsx'  # 修改为你的路径
raw_df = pd.read_excel(file_path)

# 识别年份列
year_columns = [col for col in raw_df.columns if isinstance(col, int)]
year_int_columns = year_columns

# 结果列表
final_results = []

# 按地级市分组
for city, group in raw_df.groupby('Prefecture-Level'):
    # 动态判断该市存在的总数年份
    available_years = []
    year_to_sum_col = {2020: '2020_sum', 2021: '2021_sum', 2022: '2022_sum'}

    for year, sum_col in year_to_sum_col.items():
        if sum_col in group.columns and not group[sum_col].dropna().empty:
            available_years.append(year)

    if not available_years:
        print(f"⚠️ 市 {city} 缺少所有总死亡数，跳过。")
        continue

    # 获取对应年份的市总数
    city_totals = [group[year_to_sum_col[year]].dropna().iloc[0] for year in available_years]

    # 准备县级数据
    county_data = group.set_index('County_Level')[year_columns]
    county_data = county_data.T  # 转置

    predictions = pd.DataFrame(index=available_years)

    for county in county_data.columns:
        series = county_data[county].dropna()
        if len(series) < 2:
            continue  # 跳过数据不足的县

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

    # 总数归一化
    predictions['Sum'] = predictions.sum(axis=1)
    predictions['Adjust_factor'] = np.array(city_totals) / predictions['Sum']

    for county in predictions.columns[:-2]:
        predictions[county] = predictions[county] * predictions['Adjust_factor']

    # 整理
    predictions = predictions.drop(columns=['Sum', 'Adjust_factor'])
    predictions['Prefecture-Level'] = city
    predictions = predictions.set_index('Prefecture-Level', append=True)
    final_results.append(predictions)

# 合并所有市结果
final_allocated = pd.concat(final_results)
final_allocated = final_allocated.round(0)

# 导出
output_path = '县级分配_Prophet_动态分配.xlsx'
final_allocated.to_excel(output_path, merge_cells=False)
print(f"✅ 结果已导出：{output_path}")
