import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = 'Microsoft YaHei'
plt.rcParams['axes.unicode_minus'] = False

original = pd.read_csv("ow2.csv")
# print(original.columns)

# 合并命中率:Weapon Accuracy,Scoped Accuracy, Secondary Accuracy
original['Accuracy'] = original['Weapon Accuracy, %'].fillna(0) + original['Scoped Accuracy, %'].fillna(0) + original[
    'Secondary Accuracy, %'].fillna(0)

# 合并技能击杀
original['Skill Kills'] = 0
for x in original.columns[11:]:
    if 'Kills' in x:
        original['Skill Kills'] = original['Skill Kills'] + original[x].fillna(0)

# 处理段位保留ALL
mean_ori = original[original['Skill Tier'] == 'All']

# 绘制各个定位的KDA小提琴图
sns.set_style("darkgrid")
plt.title('KDA')
sns.violinplot(x='Role', y='KDA Ratio', data=original, color='lavender')
plt.savefig("./img/AllKDA")

# 绘制各个定位的Damage箱型图
sns.set_style("darkgrid")
plt.title('Damage')
sns.boxplot(x='Role', y='Damage / 10min', data=original, color='plum')
plt.savefig("./img/ALLDmg")

# 绘制各个段位的抓取率折线图
sns.set_style("darkgrid")
sns.relplot(kind='line', x='Skill Tier', y='Pick Rate, %', hue='Hero', data=original, row="Role", legend="auto", height=7)
plt.savefig("./img/AllPick")

# 绘制各个段位的英雄的胜率折线图
sns.set_style("darkgrid")
sns.relplot(kind='line', x='Skill Tier', y='Win Rate, %', hue='Hero', data=original, row="Role", legend="auto", height=7)
plt.savefig("./img/WinRate")

# 按位置处理抓取率
pick = original['Pick Rate, %'].groupby(original['Role']).sum()
original['PickRate'] = original['Pick Rate, %']
for x in original['Role'].unique():
    original.loc[original['Role'] == x, 'PickRate'] = original.loc[original['Role'] == x, 'Pick Rate, %'].div(float(pick[x]))

# 绘制抓取率和胜率的散点图
sns.set_style("darkgrid")
plt.figure(figsize=(10, 10))
plt.title("PickRate and WinRate")
sns.scatterplot(x='PickRate', y='Win Rate, %', hue='Role', data=original)
plt.savefig("./img/PickWin")

# 命中率和定位的散点图
sns.set_style("darkgrid")
sns.catplot(x="Role", y="Accuracy", data=mean_ori, hue="Accuracy", palette='Blues', jitter=0.07, dodge=False,
            legend=False)
plt.savefig("./img/RoleAcc")

# 分离C
C_ori = mean_ori[mean_ori['Role'] == 'Damage'].copy()

# 绘制C的抓取率饼图
C_ori['PickPerc'] = C_ori['Pick Rate, %'] / C_ori['Pick Rate, %'].sum()
exp = [0] * len(C_ori.index)
exp[C_ori['PickPerc'].argmax()] = 0.1
plt.figure(figsize=(10, 10))
plt.title("PickRate of C")
plt.pie(C_ori['PickPerc'], labels=C_ori['Hero'], explode=exp, shadow=True)
plt.savefig("./img/C_PR")

# C位的参与击杀和最终击杀的百分比
C_ori['FofE'] = C_ori['Final Blows / 10min'] / C_ori['Eliminations / 10min']
C_ori['Undirectly Kill / 10min'] = C_ori['Eliminations / 10min'] - C_ori['Final Blows / 10min']
sns.set_style("darkgrid")
plt.figure(figsize=(10, 10))
plt.xticks(rotation=90)
plt.title("FinalBlows of Eliminations")
sns.barplot(x="Hero", y="Eliminations / 10min", data=C_ori, label="Original", color="cornflowerblue")
sns.barplot(x="Hero", y="Final Blows / 10min", data=C_ori, label="FinalBlows", color="salmon")
plt.savefig("./img/C_FinEli")

# 排序并绘制C位最终击杀占比柱状图
C_ori.sort_values(by="FofE", inplace=True, ascending=False)
plt.figure(figsize=(10, 10))
plt.xticks(rotation=90)
plt.title("FinalBlows Percent")
sns.barplot(x="Hero", y="FofE", data=C_ori, hue="Hero", palette="PuBuGn_r")
plt.savefig("./img/C_FinPer")

# 排序并绘制C位技能击杀占比柱状图
C_ori['SK'] = C_ori['Skill Kills'] / C_ori['Final Blows / 10min']
C_ori.sort_values(by="SK", inplace=True, ascending=False)
plt.figure(figsize=(10, 10))
plt.xticks(rotation=90)
plt.title("Skill Kills Percent")
sns.barplot(x="Hero", y="SK", data=C_ori, hue="Hero", palette="Reds_r")
plt.savefig("./img/C_SK")

# 分离S
S_ori = mean_ori[mean_ori['Role'] == 'Support'].copy()

# 绘制奶量排名柱状图
S_ori.sort_values(by="Healing / 10min", inplace=True, ascending=False)
plt.figure(figsize=(10, 10))
plt.xticks(rotation=90)
plt.title("Healing")
sns.barplot(x="Hero", y="Healing / 10min", data=S_ori, hue="Hero", palette="YlOrBr_r")
plt.savefig("./img/S_Heal")

# 绘制雷达图，分析奶量，参与击杀，死亡，站场时间
S_labels = np.array(['Heal', 'Elims', 'Death', 'Time'])
S_num = 4
angles = np.linspace(0, 2 * np.pi, S_num, endpoint=False)
angles = np.append(angles, [angles[0]])
fig, ax = plt.subplots(2, 4, figsize=(21, 10), subplot_kw=dict(polar=True))
ax = ax.flatten()
hero_ls = [i for i in S_ori['Hero']]
data_ls = ['Healing / 10min', 'Eliminations / 10min', 'Deaths / 10min', 'Objective Time / 10min']
for i in data_ls:
    S_ori[i] = S_ori[i].div(S_ori[i].sum())

for i in hero_ls:
    idx = hero_ls.index(i)
    data = np.array(S_ori.loc[S_ori['Hero'] == i, data_ls].values.tolist()).astype(float).flatten()
    data = np.append(data, [data[0]])

    ax[idx].fill(angles, data, alpha=0.7)
    ax[idx].plot(angles, data)
    ax[idx].set_rlim(0, 0.25)
    ax[idx].set_thetagrids(angles[:-1] * 180 / np.pi, labels=S_labels)
    ax[idx].set_xticks(angles[:-1])
    ax[idx].set_xticklabels(S_labels)
    ax[idx].set_yticks([0.1, 0.15, 0.2], ['B', 'A', 'S'])
    ax[idx].set_title(i)
plt.savefig("./img/H_Abt")

# 分离T
T_ori = mean_ori[mean_ori['Role'] == 'Tank'].copy()

# 对T的战场击杀绘制柱状图和散点图
T_ori.sort_values(by="Objective Kills / 10min", inplace=True, ascending=False)
plt.figure(figsize=(10, 12))
plt.xticks(rotation=90)
plt.title("OBJ Kills")
sns.barplot(x="Hero", y="Objective Kills / 10min", data=T_ori, hue="Hero", palette="PuBu_r")
plt.savefig("./img/T_ObjK")

# 对T的战场时间绘制柱状图
T_ori.sort_values(by="Objective Time / 10min", inplace=True, ascending=False)
plt.figure(figsize=(10, 12))
plt.xticks(rotation=90)
plt.title("OBJ Time")
sns.barplot(x="Hero", y="Objective Time / 10min", data=T_ori, hue="Hero", palette="BuPu_r")
plt.savefig("./img/T_ObjT")

