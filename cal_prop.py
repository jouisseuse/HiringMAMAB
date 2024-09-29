import pandas as pd
import matplotlib.pyplot as plt
import sys
import os

# 加载数据
data = pd.read_csv(sys.argv[1])
dir_path = os.path.dirname(sys.argv[1])

# 过滤出 agent 为 C1 的数据
c1_data = data[data['agent'] == 'C1']

# 设置图形大小
plt.figure(figsize=(10, 6))

# 绘制每一列随时间轮数的变化曲线
for column in ['1', '2', '3', '4']:
    plt.plot(c1_data['round'], c1_data[column], label=f'C {column}')
    # 添加图表标题和标签
plt.title('C1 Data Across Rounds')
plt.xlabel('Round')
plt.ylabel('Values')
plt.legend(title='Columns')

# 显示图表
plt.grid(True)
plt.savefig(f'{dir_path}/C1_data_across_rounds.png', dpi=300)
plt.clf()

for column in ['1', '2', '3', '4']:
    plt.plot(c1_data['round'], c1_data[column], label=f'F {column}')
    # 添加图表标题和标签
    plt.title('C1 Data Across Rounds')
    plt.xlabel('Round')
    plt.ylabel('Values')
    plt.legend()

    # 显示图表
    plt.grid(True)
    plt.savefig(f"{dir_path}/C1_F{column}_data_across_rounds.png", dpi=300)
    plt.clf()


# plt.show()