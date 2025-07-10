import pandas as pd
import os

# === 路径设置 ===
threshold_file = r"F:\Base_Year_Data\总结数据（excel）\百分比阈值\基准年最小SPEI阈值.xlsx"# 阈值所在文件
rain_file = r"F:\Com_Year_Data\总结数据（excel）\SPEI_2013_2022.xlsx"# 降水数据文件
output_file = rain_file                         # 保存到同一个文件中

# === 读取数据 ===
threshold_df = pd.read_excel(threshold_file, sheet_name="阈值")
rain_df = pd.read_excel(rain_file)

# 获取县名列表
counties = threshold_df.columns
date_series = rain_df.iloc[:, 0]
rain_data = rain_df.iloc[:, 1:]  # 除去日期列

# 用于收集结果的字典
rain_exceed_dict = {}
date_exceed_dict = {}

# === 筛选每个县超过阈值的数据 ===
for county in counties:
    if county not in rain_data.columns:
        print(f"⚠️ 警告：{county} 在降水数据中找不到，跳过。")
        continue

    threshold = threshold_df[county].iloc[0]
    values = rain_data[county]
    exceed_mask = values < threshold

    # 只保留超过阈值的行
    rain_exceed_dict[county] = values[exceed_mask].reset_index(drop=True)
    date_exceed_dict[county] = date_series[exceed_mask].reset_index(drop=True)

# === 一次性拼接所有列，避免碎片化警告 ===
rain_exceed_df = pd.concat(rain_exceed_dict, axis=1)
date_exceed_df = pd.concat(date_exceed_dict, axis=1)

# === 保存结果到原文件的两个新工作表 ===
with pd.ExcelWriter(output_file, engine="openpyxl", mode="a", if_sheet_exists="replace") as writer:
    rain_exceed_df.to_excel(writer, sheet_name="超过阈值的数值", index=False)
    date_exceed_df.to_excel(writer, sheet_name="超过阈值的日期", index=False)

print(f"[✔] 已将结果写入到: {output_file}")
