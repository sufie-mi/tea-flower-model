import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st
import matplotlib.dates as mdates  
import pandas as pd

plt.rc('font', family='Times New Roman')
file_path = r"F:\results\variety1280\amount of flowers\929var_addtime.csv"  # 替换为你的 CSV 文件路径

data = pd.read_csv(file_path)

data.columns = data.columns.str.strip()
print(data.head())

variety_dict = {
    'HGY': 1, 'HJG': 2, 'YMX': 3,
    'HQ': 4, 'JBH': 5, 'MK1': 6,
    'MX': 7, 'NZ2': 8, 'NZZ': 8, 'SCZ': 9,
    'SMZ': 10, 'TGY': 11, 'WN95': 12,
    'ZHDB': 13, 'AH1': 14, 'AH7': 15,
    'BHZ': 16, 'DMB': 17,
    'FADB': 18, 'FDDH': 19, 'FZ2': 20, 'FZZ': 20,
    'EC1': 21, 'CYQ': 22, 'ZYQ': 22, 'XC5': 23, 'XC11': 24,
    'FY6': 25, 'GYQ': 26, 'FDDB': 27, 'LJ43': 28, 'AH3': 29, 'JX': 30
}
selected_var_nums1 = [22, 24, 5, 23, 21, 26, 19, 20]  
selected_var_nums2 = [25, 9, 12, 13, 1, 6, 8, 4, 2]  
selected_var_nums3 = [17, 16, 10, 7, 18, 14, 3, 15, 11] 
# select one variety for plotting
variety = 18  

# filtered data
filtered_data = data[data['var'] == variety]
print(filtered_data)

filtered_data.loc[:, 'time'] = pd.to_datetime(filtered_data['time'], format='%d-%b')
filtered_data = filtered_data.sort_values('time')
# caculate mean value
# groupby caculate mean value of buds、flowers 和 old flower
grouped_data = filtered_data.groupby('time').agg({
    'buds': 'mean',
    'flowers': 'mean',
    'old flower': 'mean'
}).reset_index()

x_values = grouped_data['time']
print(x_values)
buds_values = grouped_data['buds']
flowers_values = grouped_data['flowers']
old_flower_values = grouped_data['old flower']


# plotting
plt.figure(figsize=(15, 10))

plt.plot(x_values, buds_values, marker='o', markerfacecolor=(217/255, 238/255, 238/255), markersize=8,
         color="#1B9E77", linewidth=2, linestyle='--', dashes=[4, 3], label='bud')
plt.plot(x_values, flowers_values, marker='o', markerfacecolor=(248/255, 216/255, 185/255), markersize=8,
         color=(239/255, 153/255, 75/255), linewidth=2, linestyle='--', dashes=[4, 3], label='B flower')
plt.plot(x_values, old_flower_values, marker='o', markerfacecolor=(220/255, 228/255, 250/255), markersize=8,
         color=(124/255, 135/255, 181/255), linewidth=2, linestyle='--', dashes=[4, 3], label='W flower')

plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))

# add title and label
plt.title('FADB'.format(variety), x = 0.5, 
    y = 0.920,
    ha="center", fontsize=36, weight="bold",)
plt.xlabel('time', fontsize=28)
plt.ylabel('predicted flower quantity', fontsize=28)
plt.xticks(rotation=45, size=15)


HLINES = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140]  # 刻度位置
HLINE = [20, 60, 100]
# HLINES = [0, 20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220, 240, 260]  # 刻度位置
# HLINE = [50, 100, 150, 200]
# HLINES = [0, 10, 20, 30, 40, 50, 60]
# HLINES = [0, 5, 10, 15, 20]
# HLINE = [5, 15]
plt.tick_params(length=5)  
ax = plt.gca()  
ax.set_yticks(HLINES) 
ax.set_yticklabels(HLINES, size=15) 


y_ticks = ax.get_yticks()  
new_labels = [str(y) if y % 5 == 0 else '' for y in y_ticks] 
ax.set_yticklabels(new_labels) # size=15
# plt.ylabel("Flower Quantity", size=18, weight="bold") 

# add line
for h in HLINE:
    plt.axhline(h, color="#7F7F7F", ls=(0, (5, 6)), alpha=0.3, zorder=0)

plt.grid(visible=True, linestyle='--', dashes=[3, 1])
plt.legend()
plt.grid()

# show figure

folder_path = r"save_path" 
file_name = "FADB.svg"  
file_path = f"{folder_path}\\{file_name}"

# save figure
plt.savefig(file_path, format='svg')
plt.show()