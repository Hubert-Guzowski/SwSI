import marimo

__generated_with = "0.19.9"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Lab 4: Zadania - GLM

    Pracujemy z wykorzystaniem zbioru **winequality** z repozytorium UCI Irvine
    (https://archive.ics.uci.edu/dataset/186/wine+quality).
    """)
    return


@app.cell
def _():
    import pandas as pd
    import numpy as np
    import statsmodels.formula.api as smf
    import statsmodels.api as sm
    import plotly.express as px
    from sklearn.model_selection import train_test_split
    return np, pd, px, sm, smf, train_test_split


@app.cell
def _(pd):
    winequality_white = pd.read_csv(
        "http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv",
        sep=";"
    )
    winequality_red = pd.read_csv(
        "http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv",
        sep=";"
    )
    winequality_white.head()
    return winequality_red, winequality_white


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Zadanie 1

    Do obu tabel `winequality_white` i `winequality_red` należy dodać kolumnę
    `type` zawierającą zmienną kategoryczną o wartości odpowiednio `'white'` i
    `'red'`. Następnie połącz tabele w jedną o nazwie `winequality`.
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

    Dopasuj i przeanalizuj **regresję logistyczną** przewidującą gatunek wina.
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
    ## Zadanie 3a

    Dopasuj i przeanalizuj **regresję porządkową** przewidującą jakość wina.

    Wskazówka: użyj `statsmodels.miscmodels.ordinal_model.OrderedModel`.
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
    ## Zadanie 3b

    Porównaj wyniki z wybranym innym modelem spośród:
    - **KNN** (`sklearn.neighbors.KNeighborsClassifier`)
    - **LDA** (`sklearn.discriminant_analysis.LinearDiscriminantAnalysis`)
    - **QDA** (`sklearn.discriminant_analysis.QuadraticDiscriminantAnalysis`)
    """)
    return


@app.cell
def _():
    # Uzupełnij kod poniżej
    ...
    return


if __name__ == "__main__":
    app.run()
