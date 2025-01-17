#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dieses Skript dient als Beispiel, wie man die in der Aufgabenstellung gegebenen
Daten laden, analysieren und erste Modelle für pm10 und no2 erstellen kann.
Es erfüllt die in der Aufgabenstellung genannten Punkte (1) bis (4) in einer
möglichen Umsetzung. Anpassungen nach Bedarf sind natürlich möglich.

Voraussetzung:
    - Datei "feinstaubdataexercise.pickle" liegt im selben Ordner (oder Pfad anpassen)
    - Installation der benötigten Bibliotheken (pandas, numpy, matplotlib, seaborn,
      statsmodels, scikit-learn etc.)

Autor: (Dein Name)
"""

import pickle as pkl
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Für die Modellierung
import statsmodels.api as sm
from statsmodels.formula.api import ols
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

###############################################################################
# 0) Daten laden
###############################################################################

# Pfad zu deiner Pickle-Datei anpassen
DATA_PATH = "feinstaubdataexercise.pickle"

with open(DATA_PATH, 'rb') as file:
    dailymeansdata = pkl.load(file)

# Dictionary Keys: 'Graz-DB' und 'Kalkleiten'
df_graz = dailymeansdata["Graz-DB"].copy()   # Das DataFrame, auf das wir uns konzentrieren
df_kalk = dailymeansdata["Kalkleiten"].copy()  # Für spätere Inversions-Variable (temp-Diff)

# Index ist DateTime mit TimeZone, optional kann man die Zeitzone droppen oder konvertieren
df_graz.index = df_graz.index.tz_localize(None)
df_kalk.index = df_kalk.index.tz_localize(None)

# Ein kurzer Blick auf die Daten
print("---- Info Graz-DB ----")
df_graz.info()
print(df_graz.head())

print("\n---- Info Kalkleiten ----")
df_kalk.info()
print(df_kalk.head())

###############################################################################
# 1) Erste Exploration: Zusammenhänge, Verteilung, Missing Data
###############################################################################

# a) In welchen Bereichen bewegen sich die Variablen?
desc = df_graz.describe()
print("\n--- DESCRIBE (Graz-DB) ---")
print(desc)

# b) Zeitverlauf anschauen (z.B. pm10, no2)
#   Wir betrachten nur ein paar Beispiele der Zeitreihenplots

plt.figure(figsize=(10, 5))
plt.plot(df_graz.index, df_graz["pm10"], label="pm10")
plt.plot(df_graz.index, df_graz["no2"], label="no2")
plt.title("Zeitverlauf pm10 vs. no2")
plt.ylabel("Konzentration [µg/m³]")
plt.legend()
plt.tight_layout()
plt.show()

# c) Missing Data Visualisierung
#   Wir sehen bereits in df_graz.info(), dass no2 und pm10 Lücken haben.
#   Hier ein schnelles Heatmap-Plot (Seaborn) oder .isnull().sum()
print("\n--- Missing Values ---")
print(df_graz.isnull().sum())

sns.heatmap(df_graz.isnull(), cbar=False)
plt.title("Heatmap der fehlenden Werte (Graz-DB)")
plt.show()

# d) Paarweise Scatterplots, um einen Eindruck der Zusammenhänge zu gewinnen
#   (ohne day_type, da kategorisch)
cols_numeric = ["humidity", "temp", "no2", "pm10", "prec", "windspeed", "peak_velocity"]
sns.pairplot(df_graz[cols_numeric], diag_kind='kde')
plt.suptitle("Pairplot Graz-DB (numeric)", y=1.02)
plt.savefig("pairplot_graz_db.png")

###############################################################################
# 2) Erstes Modell für pm10 und no2 (2015-2019) - Lineare Regression
###############################################################################

# a) Train/Test Split: 2015-2019 Trainingsdaten, 2020 Testdaten
#   Wir filtern nach Jahr
train_df = df_graz.loc[(df_graz.index.year >= 2015) & (df_graz.index.year <= 2019)].copy()
test_df = df_graz.loc[(df_graz.index.year == 2020)].copy()

# b) Wir lassen no2 NICHT als Prädiktor in pm10-Modell zu und umgekehrt
#   => Prädiktoren: day_type, humidity, temp, prec, windspeed, peak_velocity
#   (pm10 oder no2 jeweils ausgeschlossen)

# Umkoden der kategorialen Variable day_type (Sunday/Holiday, Weekday, Saturday).
#   Beispiel: Dummy-Codierung für day_type
train_df = pd.get_dummies(train_df, columns=["day_type"], drop_first=True)
test_df = pd.get_dummies(test_df, columns=["day_type"], drop_first=True)

train_df["day_type"] = df_graz.loc[train_df.index, "day_type"]
test_df["day_type"] = df_graz.loc[test_df.index, "day_type"]

train_df["day_type"] = df_graz["day_type"]
test_df["day_type"] = df_graz["day_type"]

print("Spalten im DataFrame:", train_df.columns)
df = pd.get_dummies(train_df, columns=["day_type"], drop_first=True)
print(df.columns)

# c) Modell für pm10
#   Simple OLS in statsmodels, wir schließen no2 aus dem Modell aus.
pm10_train = train_df.dropna(subset=["pm10"])  # Zeilen ohne pm10 weg
predictors_pm10 = ["humidity", "temp", "prec", "windspeed", "peak_velocity",
                   "day_type_Saturday", "day_type_Weekday"]

formula_pm10 = "pm10 ~ " + " + ".join(predictors_pm10)
model_pm10 = ols(formula_pm10, data=pm10_train).fit()
print("\n--- Modell PM10: OLS Summary ---")
print(model_pm10.summary())

# ANOVA
anova_pm10 = sm.stats.anova_lm(model_pm10, typ=2)
print("\n--- ANOVA PM10 ---")
print(anova_pm10)

# d) Modell für no2
#   Analog, wir schließen pm10 aus.
no2_train = train_df.dropna(subset=["no2"])
predictors_no2 = ["humidity", "temp", "prec", "windspeed", "peak_velocity",
                  "day_type_Saturday", "day_type_Weekday"]

formula_no2 = "no2 ~ " + " + ".join(predictors_no2)
model_no2 = ols(formula_no2, data=no2_train).fit()
print("\n--- Modell NO2: OLS Summary ---")
print(model_no2.summary())

anova_no2 = sm.stats.anova_lm(model_no2, typ=2)
print("\n--- ANOVA NO2 ---")
print(anova_no2)

# Erste Bewertung der Modelle:
#   - R^2, p-Werte, Residuenplots etc.
res_pm10 = model_pm10.resid
fitted_pm10 = model_pm10.fittedvalues

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.scatter(fitted_pm10, res_pm10, alpha=0.5)
plt.axhline(0, color='red')
plt.title("Residuen vs. Fitted (pm10)")
plt.xlabel("Fitted Values (pm10)")
plt.ylabel("Residuen")

plt.subplot(1, 2, 2)
sns.histplot(res_pm10, kde=True)
plt.title("Histogramm der Residuen (pm10)")
plt.tight_layout()
plt.show()

res_no2 = model_no2.resid
fitted_no2 = model_no2.fittedvalues

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.scatter(fitted_no2, res_no2, alpha=0.5)
plt.axhline(0, color='red')
plt.title("Residuen vs. Fitted (no2)")
plt.xlabel("Fitted Values (no2)")
plt.ylabel("Residuen")

plt.subplot(1, 2, 2)
sns.histplot(res_no2, kde=True)
plt.title("Histogramm der Residuen (no2)")
plt.tight_layout()
plt.show()

###############################################################################
# 3) Inversion und Auswirkung auf Luftverschmutzung
###############################################################################

"""
Was ist eine Inversion?
-----------------------
Eine (Temperatur-)Inversion ist ein meteorologisches Phänomen, bei dem die 
Temperatur in oberen Luftschichten höher ist als in tiefer liegenden Schichten. 
Dadurch wird der Luftaustausch behindert, was zu einer erhöhten Konzentration 
von Luftschadstoffen (z.B. Feinstaub) in Bodennähe führen kann.

Untersuchung:
-------------
Wir haben Daten einer höher gelegenen Wetterstation in Kalkleiten. Wir definieren
eine neue Variable: temp_diff = temp(Graz) - temp(Kalkleiten). 
Wenn temp_diff negativ oder sehr klein ist, könnte das ein Indiz für Inversion sein 
(bzw. für Schichtung, in der es unten kälter als oben ist).
"""

# 1) temp_diff berechnen
#   Achtung, Zeitstempel beider DataFrames müssen identisch sein (oder gejoint)
df_temp = pd.merge(
    df_graz[["temp"]].rename(columns={"temp": "temp_graz"}),
    df_kalk[["temp"]].rename(columns={"temp": "temp_kalk"}),
    left_index=True, right_index=True, how="inner"
)

df_temp["temp_diff"] = df_temp["temp_graz"] - df_temp["temp_kalk"]

# 2) In unser df_graz zurückspielen (evtl. join/merge)
df_graz = df_graz.join(df_temp["temp_diff"], how="left")
train_df = df_graz.loc[(df_graz.index.year >= 2015) & (df_graz.index.year <= 2019)]
test_df = df_graz.loc[(df_graz.index.year == 2020)]

# 3) Neue Regression mit temp_diff
train_df = pd.get_dummies(train_df, columns=["day_type"], drop_first=True)

#   pm10-Modell
pm10_train = train_df.dropna(subset=["pm10", "temp_diff"])
predictors_pm10_inversion = ["humidity", "temp", "prec",
                             "windspeed", "peak_velocity",
                             "temp_diff",
                             "day_type_Saturday", "day_type_Weekday"]
formula_pm10_inversion = "pm10 ~ " + " + ".join(predictors_pm10_inversion)
model_pm10_inversion = ols(formula_pm10_inversion, data=pm10_train).fit()

print("\n--- Modell PM10 mit temp_diff: OLS Summary ---")
print(model_pm10_inversion.summary())

#   no2-Modell
no2_train = train_df.dropna(subset=["no2", "temp_diff"])
predictors_no2_inversion = ["humidity", "temp", "prec",
                            "windspeed", "peak_velocity",
                            "temp_diff",
                            "day_type_Saturday", "day_type_Weekday"]
formula_no2_inversion = "no2 ~ " + " + ".join(predictors_no2_inversion)
model_no2_inversion = ols(formula_no2_inversion, data=no2_train).fit()

print("\n--- Modell NO2 mit temp_diff: OLS Summary ---")
print(model_no2_inversion.summary())

# Auswertung: Hat die Variable temp_diff einen signifikanten Effekt?
# (p-Wert, R^2-Veränderungen etc.)

###############################################################################
# 4) Erstellung neuer Variablen (z.B. frost) und Test
###############################################################################

# Frost-Tage (binary feature)
df_graz["frost"] = (df_graz["temp"] < 0).astype(int)

# Beispiel: wir könnten auch saisonale Features erstellen (Monat, Jahreszeit)
df_graz["month"] = df_graz.index.month
df_graz["season"] = df_graz["month"] % 12 // 3 + 1  # 1=Winter, 2=Frühling, 3=Sommer, 4=Herbst

# Nochmals ins Training 2015-2019
train_df = df_graz.loc[(df_graz.index.year >= 2015) & (df_graz.index.year <= 2019)].copy()
train_df = pd.get_dummies(train_df, columns=["day_type", "season"], drop_first=True)

# pm10-Modell mit frost
pm10_train = train_df.dropna(subset=["pm10", "temp_diff"])
predictors_pm10_frost = [
    "humidity", "temp", "prec", "windspeed", "peak_velocity",
    "temp_diff", "frost",
    "day_type_Saturday", "day_type_Weekday",
    # "season_2", "season_3", "season_4",  # falls Saisonvariablen getestet werden sollen
]
formula_pm10_frost = "pm10 ~ " + " + ".join(predictors_pm10_frost)
model_pm10_frost = ols(formula_pm10_frost, data=pm10_train).fit()

print("\n--- Modell PM10 mit temp_diff + frost: OLS Summary ---")
print(model_pm10_frost.summary())

# no2-Modell mit frost
no2_train = train_df.dropna(subset=["no2", "temp_diff"])
predictors_no2_frost = [
    "humidity", "temp", "prec", "windspeed", "peak_velocity",
    "temp_diff", "frost",
    "day_type_Saturday", "day_type_Weekday",
]
formula_no2_frost = "no2 ~ " + " + ".join(predictors_no2_frost)
model_no2_frost = ols(formula_no2_frost, data=no2_train).fit()

print("\n--- Modell NO2 mit temp_diff + frost: OLS Summary ---")
print(model_no2_frost.summary())

# Ende der Beispiel-Auswertung.
# Weiterführende Schritte wären:
#   - Genauere Modell-Diagnostik
#   - Nichtlineare Zusammenhänge
#   - Kreuzvalidierung
#   - Prognosegüte bei den Testdaten (2020)
#   - Interaktionseffekte
#   - etc.

print("\nFertig! Dies war ein Beispiel-Skript für die bearbeiteten Aufgaben.")
