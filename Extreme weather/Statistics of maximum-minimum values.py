import pandas as pd

'''
# ===================================================================================最大值===================================================================================
# 文件路径
input_path = r'F:\Base_Year_Data\总结数据（excel）\基准年最大温度.xlsx'
output_path = r'F:\Base_Year_Data\总结数据（excel）\百分比阈值\基准年最大温度.xlsx'

# 读取 Excel
df = pd.read_excel(input_path)

# 日期列转为 datetime 并设置为索引
df.iloc[:, 0] = pd.to_datetime(df.iloc[:, 0])
df.set_index(df.columns[0], inplace=True)

# 年份列表
years = df.index.year.unique()

# 用字典暂存每个县的数据，避免性能警告
max_val_dict = {}
second_val_dict = {}
max_date_dict = {}
second_date_dict = {}

# 遍历每个县（列）
for county in df.columns:
    max_vals = []
    second_vals = []
    max_dates = []
    second_dates = []

    for year in years:
        # 取出该县该年的所有数据，去除 NaN
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

    # 存入字典
    max_val_dict[county] = max_vals
    second_val_dict[county] = second_vals
    max_date_dict[county] = max_dates
    second_date_dict[county] = second_dates

# 最后统一构建 DataFrame，避免碎片化
max_val_df = pd.DataFrame(max_val_dict, index=years)
second_val_df = pd.DataFrame(second_val_dict, index=years)
max_date_df = pd.DataFrame(max_date_dict, index=years)
second_date_df = pd.DataFrame(second_date_dict, index=years)

# 保存为 Excel（4 个工作表）
with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
    max_val_df.to_excel(writer, sheet_name='最大值')
    max_date_df.to_excel(writer, sheet_name='最大值日期')
    second_val_df.to_excel(writer, sheet_name='次大值')
    second_date_df.to_excel(writer, sheet_name='次大值日期')

print("处理完成，已保存到：", output_path)
'''

# ===================================================================================最小值===================================================================================
# 文件路径
input_path = r'F:\Base_Year_Data\总结数据（excel）\基准年最小温度.xlsx'
output_path = r'F:\Base_Year_Data\总结数据（excel）\百分比阈值\基准年最小温度.xlsx'

# 读取 Excel
df = pd.read_excel(input_path)

# 日期列转为 datetime 并设置为索引
df.iloc[:, 0] = pd.to_datetime(df.iloc[:, 0])
df.set_index(df.columns[0], inplace=True)

# 年份列表
years = df.index.year.unique()

# 用字典暂存每个县的数据，避免性能警告
min_val_dict = {}
second_val_dict = {}
min_date_dict = {}
second_date_dict = {}

# 遍历每个县（列）
for county in df.columns:
    min_vals = []
    second_vals = []
    min_dates = []
    second_dates = []

    for year in years:
        # 取出该县该年的所有数据，去除 NaN
        data = df.loc[df.index.year == year, county].dropna()

        if data.empty:
            min_vals.append(float('nan'))
            second_vals.append(float('nan'))
            min_dates.append(pd.NaT)
            second_dates.append(pd.NaT)
        else:
            sorted_data = data.sort_values(ascending=True)  # 升序：最小在前
            min_vals.append(sorted_data.iloc[0])
            min_dates.append(sorted_data.index[0])

            if len(sorted_data) > 1:
                second_vals.append(sorted_data.iloc[1])
                second_dates.append(sorted_data.index[1])
            else:
                second_vals.append(float('nan'))
                second_dates.append(pd.NaT)

    # 存入字典
    min_val_dict[county] = min_vals
    second_val_dict[county] = second_vals
    min_date_dict[county] = min_dates
    second_date_dict[county] = second_dates

# 构建 DataFrame
min_val_df = pd.DataFrame(min_val_dict, index=years)
second_val_df = pd.DataFrame(second_val_dict, index=years)
min_date_df = pd.DataFrame(min_date_dict, index=years)
second_date_df = pd.DataFrame(second_date_dict, index=years)

# 保存为 Excel（4 个工作表）
with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
    min_val_df.to_excel(writer, sheet_name='最小值')
    min_date_df.to_excel(writer, sheet_name='最小值日期')
    second_val_df.to_excel(writer, sheet_name='次小值')
    second_date_df.to_excel(writer, sheet_name='次小值日期')

print("处理完成，已保存到：", output_path)