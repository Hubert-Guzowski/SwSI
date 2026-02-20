import marimo

__generated_with = "0.19.9"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Lab 3: Zadania - Regresja liniowa

    Zadania dotyczą zbioru UCI Air Quality
    (https://archive.ics.uci.edu/dataset/360/air+quality).
    """)
    return


@app.cell
def _():
    import pandas as pd
    import numpy as np
    import statsmodels.formula.api as smf
    import statsmodels.api as sm
    import plotly.express as px
    return np, pd, px, sm, smf


@app.cell
def _(pd):
    air_quality_df = pd.read_csv("AirQualityUCI.csv", sep=";", decimal=",")

    air_quality_df = air_quality_df.iloc[:, :-2]
    air_quality_df['Date'] = pd.to_datetime(air_quality_df['Date'], format='%d/%m/%Y')
    air_quality_df['Time'] = pd.to_datetime(air_quality_df['Time'], format='%H.%M.%S')

    columns_rename = {
        'CO(GT)': 'CO',
        'PT08.S1(CO)': 'PT08_S1_CO',
        'NMHC(GT)': 'NMHC',
        'C6H6(GT)': 'Benzene',
        'PT08.S2(NMHC)': 'PT08_S2_NMHC',
        'NOx(GT)': 'NOx',
        'PT08.S3(NOx)': 'PT08_S3_NOx',
        'NO2(GT)': 'NO2',
        'PT08.S4(NO2)': 'PT08_S4_NO2',
        'PT08.S5(O3)': 'PT08_S5_O3',
        'T': 'Temperature',
        'RH': 'RelativeHumidity',
        'AH': 'AbsoluteHumidity'
    }

    air_quality_df = air_quality_df.rename(columns=columns_rename)

    air_quality_df.info()
    return air_quality_df, columns_rename


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Zadanie 1

    Dopasuj model regresji liniowej przewidujący wartość **CO** wykorzystując
    5 wybranych zmiennych i zinterpretuj otrzymane wyniki.
    """)
    return


@app.cell
def _():
    # Uzupełnij kod poniżej
    ...
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Zadanie 2

    Porównaj modele regresji wielomianowej stężenia **CO** względem ozonu
    **PT08_S5_O3** dla różnych stopni wykorzystanego wielomianu.
    """)
    return


@app.cell
def _():
    # Uzupełnij kod poniżej
    ...
    return


if __name__ == "__main__":
    app.run()
