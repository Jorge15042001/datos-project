# %%!
import pandas as pd
import numpy as np
#  import cv2
import seaborn as sbn
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize

from IPython.display import clear_output


# %%!
df = pd.read_csv("./datosGuayasMothly.csv")
df["time"] = df["time"].astype("datetime64[ms]")
df["day"] = df["time"].dt.day
df["month"] = df["time"].dt.month
df["year"] = df["time"].dt.year
df = df.loc[df["avg_rad"] > 0]


# %%!
df
# %% [md]

# preugnta 1:

# como varia la intensidad de la luz por mes
* en febrero se registra el mes mas oscuro durante la noche
* julio, agosto y septiempre son los meses mas claros
# %%!
df_barrio_month = df.groupby(["Name", "month"])[
    "avg_rad"].mean().unstack(level=1)
_, axs = plt.subplots(2, 1, figsize=(30, 100))
sbn.heatmap(df_barrio_month, norm=LogNorm(), ax=axs[0])
df_total_light = df.groupby(["month"])["avg_rad"].sum()
sbn.lineplot(df_total_light, ax=axs[1])
#  sbn.heatmap(df_parroquia_nightlight,linewidths=0.5, annot_kws={'size': 8},annot=True, norm=LogNorm())
# %% [md]

# preugnta 2:

variacion realtiva de las luces nocturnas

# %%!
#  df_total_light = df.groupby(["time"])["avg_rad"].sum()
#  sbn.lineplot(df_total_light)
brillo_total = df.groupby(["Name", "time"])["avg_rad"].sum()
brillo_rel = []
growth = []
growth_absolute = []
for index, row in df.iterrows():
    print(index/421796)
    name = row["Name"]
    time = row["time"]
    time_before = row["time"]-pd.DateOffset(month=2)
    brillo_rel.append(row["avg_rad"]/brillo_total[name][time])
    try:
        growth.append((brillo_total[name][time]-brillo_total[name]
                      [time_before])/brillo_total[name][time_before])
    except:
        growth.append(0)
    try:
        growth_absolute.append(
            brillo_total[name][time]-brillo_total[name][time_before])
    except:
        growth_absolute.append(0)


# %%!
df["rad_rel"] = brillo_rel
df["growth"] = growth
df["growth_absolute"] = growth_absolute
# %%!
df
# %%!

rad_rel = df.groupby(["Name", "time"])["growth_absolute"].mean().unstack(level=1)
plt.figure(figsize=(30, 100))
sbn.heatmap(rad_rel,  fmt='f', linewidths=0.5, annot_kws={'size': 8},cmap = "RdBu_r")
# %%!
#  temp = df.groupby(["time", "Name"])["rad_rel"].mean().unstack(level=1)
rad_rel_grouped = df.groupby(["Name", "time"])["rad_rel"].mean()
rad_rel_grouped.fillna(0)


names = ["Altos del Rio", "Arcadia",
         "Cooperativa Portete de Tarqui", "Metropolis 1", "Porto Acqua", "Rio Lindo", "San Sebastian"]
dates = np.unique(df["time"])


def get_series(i):
    response = rad_rel_grouped[names[i]]
    for d in dates:
        try:
            response[d]
        except:
            response[d] = 0
    return response


#  print(np.array(rad_rel_grouped[names[0]]))
arrays = list(map(get_series, range(len(names))))
#  indexs =list(rad_rel.index)
#  indexs = str(read_rel.index[0])
print(rad_rel_grouped[names[0]])
#  plt.stackplot(dates, *arrays, labels=names)

fig, axs = plt.subplots(1,1,figsize=(10,10))
for a in arrays:
    sbn.lineplot(a,ax=axs)

#  plt.legend(loc='upper left')
#  plt.show()
#  sbn.heatmap(rad_rel, )
#  sbn.heatmap(rad_rel, cmap = "RdBu_r")



# %%!
por_anio = df.groupby(["year"])["avg_rad"].mean()
primero_enero = df[(df["day"]==1) & (df["month"]==1)].groupby(["year"])["avg_rad"].mean()

fin_de_anio_data= pd.DataFrame()
fin_de_anio_data["Fin de anio"]= primero_enero
fin_de_anio_data["Promedio Anual"]= por_anio

fig, axs = plt.subplots(1,1,figsize=(10,10))
sbn.lineplot(data=fin_de_anio_data).set_label("1ro de enero")

