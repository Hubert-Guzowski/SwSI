import marimo

__generated_with = "0.19.9"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Lab 6: Zadania - Modele nieliniowe
    """)
    return


@app.cell
def _():
    import numpy as np
    import pandas as pd
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    from statsmodels.datasets import get_rdataset
    from patsy import dmatrix
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error
    import plotly.express as px
    import plotly.graph_objects as go
    return (
        PolynomialFeatures,
        dmatrix,
        get_rdataset,
        go,
        mean_squared_error,
        np,
        pd,
        px,
        sm,
        smf,
        train_test_split,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Zadanie 1 — zbiór Boston

    Ustal zbiór walidacyjny (testowy) zawierający 20% losowo wybranych danych
    (ziarno generatora ustaw na swój numer albumu). Licząc błąd średniokwadratowy
    na tym zbiorze, ustal optymalny stopień wielomianu (między 1 a 10) w regresji
    wielomianowej `medv` względem `lstat`. Modele mają być uczone na danych
    nienależących do zbioru walidacyjnego.
    """)
    return


@app.cell
def _(get_rdataset):
    boston = get_rdataset("Boston", package="MASS").data
    boston.head()
    return (boston,)


@app.cell
def _():
    # Uzupełnij kod poniżej
    ...
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Zadanie 2 — zbiór Abalone (UCI ML)

    Dane z pomiarów cech fizycznych słuchotek (uchowców). Interesuje nas
    zależność wieku osobnika (liczba pierścieni `Rings`) od pozostałych parametrów.

    Dane z UCI ML repo (ucimlrepo lub bezpośredni URL):
    https://archive.ics.uci.edu/dataset/1/abalone
    """)
    return


@app.cell
def _(pd):
    # Pobieranie danych Abalone
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/abalone/abalone.data"
    col_names = ['Sex', 'Length', 'Diameter', 'Height', 'Whole_weight',
                 'Shucked_weight', 'Viscera_weight', 'Shell_weight', 'Rings']
    abalone = pd.read_csv(url, header=None, names=col_names)
    abalone['Sex'] = abalone['Sex'].astype('category')
    abalone.head()
    return abalone, col_names, url


@app.cell
def _(abalone, sm, smf):
    # Zmienna Whole_weight jest praktycznie liniowo zależna od pozostałych wag
    lm_weight = smf.ols('Whole_weight ~ Shucked_weight + Viscera_weight + Shell_weight',
                        data=abalone).fit()
    print(lm_weight.summary())
    return (lm_weight,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Zmienna `Whole_weight` jest redundantna — należy ją usunąć z dalszej analizy.

    **Krok 1**: Dopasuj model regresji Poissonowskiej (liczba pierścieni jest całkowita).
    """)
    return


@app.cell
def _():
    # Regresja Poissonowska
    # Uzupełnij kod poniżej
    ...
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Krok 2**: Usuń nieistotne predyktory.
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
    **Krok 3**: Sprawdź, czy lepszego modelu nie da się uzyskać przez nieliniowe
    transformacje predyktorów (wygładzające funkcje sklejane lub regresja lokalna).

    Użyj `statsmodels.gam.api.GLMGam` z `BSplines`.
    """)
    return


@app.cell
def _():
    # GAM z nieparametrycznymi transformacjami
    # Uzupełnij kod poniżej
    ...
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Krok 4**: Porównaj oba finalne modele przy pomocy ANOVA.

    **Krok 5**: Wykonaj PDP dla obu modeli.
    """)
    return


@app.cell
def _():
    # ANOVA + PDP
    # Uzupełnij kod poniżej
    ...
    return


if __name__ == "__main__":
    app.run()
