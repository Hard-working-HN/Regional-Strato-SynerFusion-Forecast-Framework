import cdsapi
import calendar
import os
import subprocess
import time

c = cdsapi.Client(timeout=10)

# 定义需要下载的年份范围和变量
years = list(range(1983, 1984))  # 1983-2012
variable = 'potential_evaporation'
area = [55, 73, 3, 137]
output_file = 'potential_evaporation2.txt'

# 需要下载的月份列表
months_to_download = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12']

# 打开文件以写入URL
with open(output_file, 'w') as f:
    for year in years:
        year_str = str(year)
        for month_str in months_to_download:
            # 获取该月的天数
            num_days = calendar.monthrange(year, int(month_str))[1]
            try:
                result = c.retrieve(
                    'reanalysis-era5-land',
                    {
                        'variable': variable,
                        'year': year_str,
                        'month': month_str,
                        'day': [f'{day:02d}' for day in range(1, num_days + 1)],
                        'time': [f'{hour:02d}:00' for hour in range(24)],
                        'area': area,
                        'data_format': 'netcdf',
                        "download_format": "unarchived"
                    },
                )
                url = result.location
                f.write(f"{url} # {year_str}-{month_str}\n")
            except Exception as e:
                print(f'Error retrieving data for {year_str}-{month_str}: {e}')

print(f"Download links for 1983-2012 saved to {output_file}. You can now use IDM to batch download the files.")

time.sleep(30)

# 下载链接文件路径
input_file = 'potential_evaporation2.txt'

# IDM可执行文件路径
idm_path = 'D:\\Internet Download Manager\\IDMan.exe'

# 下载文件保存路径
save_path = "F:\\Base_Year_Data\\potential_evaporation"

# 创建保存路径文件夹（如果不存在）
if not os.path.exists(save_path):
    os.makedirs(save_path)

# 读取下载链接文件
with open(input_file, 'r') as file:
    lines = file.readlines()

# 遍历每一行
for line in lines:
    # 去掉行末的换行符并分割出URL和日期
    parts = line.strip().split('#')
    if len(parts) == 2:
        url = parts[0].strip()
        date = parts[1].strip()

        # 文件名为日期名
        filename = f"{date}.nc"

        # 构建完整文件路径
        file_path = os.path.join(save_path, filename)

        # 使用IDM进行下载
        idm_command = f'"{idm_path}" /d "{url}" /p "{save_path}" /f "{filename}" /n /a'
        subprocess.run(idm_command, shell=True)

# 启动IDM进行下载任务
subprocess.run(f'"{idm_path}" /s', shell=True)

print("下载任务已添加到IDM中。")
