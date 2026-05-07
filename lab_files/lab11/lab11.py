import marimo

__generated_with = "0.19.9"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Lab 9: MARS i AutoML

    ## Wprowadzenie

    Laboratorium podzielone jest na dwie części:
    - **MARS** (Multivariate Adaptive Regression Splines)
    - **AutoML** (FLAML)

    Przed wykonaniem poniższych komórek upewnij się, że masz zainstalowane:
    ```
    pip install pyearth flaml
    ```
    """)
    return


@app.cell
def _():
    import numpy as np
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error, r2_score
    from sklearn.linear_model import LinearRegression
    import plotly.express as px
    import plotly.graph_objects as go
    return (
        LinearRegression,
        go,
        mean_squared_error,
        np,
        pd,
        px,
        r2_score,
        train_test_split,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## MARS (Multivariate Adaptive Regression Splines)

    MARS to adaptacyjna metoda regresji dobrze dostosowana do problemów
    wielowymiarowych. Działa podobnie do drzew regresyjnych, ale jako funkcje
    bazowe używa spline'ów (zamiast stałych).

    MARS wykonuje:
    1. **Forward pass** — sekwencyjnie dobiera punkty przełamania minimalizując RSS
    2. **Pruning** — usuwa najmniej wnoszące spline'y

    W Pythonie dostępny jest pakiet `pyearth` (implementacja `earth` z R):
    ```
    pip install sklearn-contrib-py-earth
    ```

    Alternatywnie można użyć `pygam` lub implementacji w `flaml`.
    """)
    return


@app.cell
def _():
    from statsmodels.datasets import get_rdataset

    boston = get_rdataset("Boston", package="MASS").data
    print(boston.describe())
    boston.head()
    return boston, get_rdataset


@app.cell
def _(boston, train_test_split):
    X_boston = boston.drop('medv', axis=1)
    y_boston = boston['medv']

    X_train_b, X_test_b, y_train_b, y_test_b = train_test_split(
        X_boston, y_boston, test_size=0.3, random_state=123
    )
    return X_boston, X_test_b, X_train_b, y_boston, y_test_b, y_train_b


@app.cell
def _(X_test_b, X_train_b, mean_squared_error, r2_score, y_test_b, y_train_b):
    try:
        from pyearth import Earth

        # Podstawowy model MARS (degree=1: brak interakcji)
        mars_basic = Earth(max_degree=1)
        mars_basic.fit(X_train_b, y_train_b)

        print("=== MARS (degree=1) ===")
        print(mars_basic.summary())

        pred_mars = mars_basic.predict(X_test_b)
        print(f"\nRMSE testowe: {mean_squared_error(y_test_b, pred_mars)**0.5:.3f}")
        print(f"R²: {r2_score(y_test_b, pred_mars):.3f}")

        # Model z interakcjami (degree=2)
        mars_tuned = Earth(max_degree=2, max_terms=25, min_samples_leaf=10)
        mars_tuned.fit(X_train_b, y_train_b)

        pred_mars_tuned = mars_tuned.predict(X_test_b)
        print("\n=== MARS (degree=2, tuned) ===")
        print(f"RMSE testowe: {mean_squared_error(y_test_b, pred_mars_tuned)**0.5:.3f}")
        print(f"R²: {r2_score(y_test_b, pred_mars_tuned):.3f}")

    except ImportError:
        print("Pakiet pyearth nie jest zainstalowany.")
        print("Instalacja: pip install sklearn-contrib-py-earth")
        print("\nAlternatywnie używamy GradientBoostingRegressor jako porównanie:")

        from sklearn.ensemble import GradientBoostingRegressor
        gb_model = GradientBoostingRegressor(n_estimators=200, max_depth=3, random_state=123)
        gb_model.fit(X_train_b, y_train_b)
        pred_gb = gb_model.predict(X_test_b)
        print(f"GBR RMSE: {mean_squared_error(y_test_b, pred_gb)**0.5:.3f}")
        print(f"GBR R²: {r2_score(y_test_b, pred_gb):.3f}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Porównanie MARS z regresją liniową
    """)
    return


@app.cell
def _(LinearRegression, X_boston, X_test_b, X_train_b, go, mean_squared_error, r2_score, y_boston, y_test_b, y_train_b):
    lm_model = LinearRegression()
    lm_model.fit(X_train_b, y_train_b)
    lm_pred = lm_model.predict(X_test_b)
    lm_rmse = mean_squared_error(y_test_b, lm_pred) ** 0.5
    lm_r2 = r2_score(y_test_b, lm_pred)

    print("=== Regresja liniowa ===")
    print(f"RMSE testowe: {lm_rmse:.3f}")
    print(f"R²: {lm_r2:.3f}")

    # Wykres Actual vs Predicted
    fig_comp = go.Figure()
    fig_comp.add_scatter(x=y_test_b, y=lm_pred, mode='markers', opacity=0.5,
                         name='Regresja liniowa')
    fig_comp.add_scatter(x=[y_boston.min(), y_boston.max()],
                         y=[y_boston.min(), y_boston.max()],
                         mode='lines', name='Idealne', line=dict(color='red', dash='dash'))
    fig_comp.update_layout(title='Regresja liniowa: Actual vs Predicted',
                           xaxis_title='Rzeczywiste', yaxis_title='Predykowane')
    fig_comp.show()
    return fig_comp, lm_model, lm_pred, lm_r2, lm_rmse


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## AutoML

    AutoML automatyzuje proces uczenia maszynowego:
    - podstawowa obróbka danych (brakujące wartości, zmienne kategoryczne, skalowanie)
    - trening wielu modeli z różnymi parametrami
    - podstawowa interpretacja modeli

    Przykłady narzędzi AutoML:
    - **H2O.ai**, **PyCaret**, **FLAML** (Microsoft), **AutoGluon** (Amazon)

    Poniżej przykład z **FLAML** na zbiorze wine quality:
    """)
    return


@app.cell
def _(pd):
    url_white = "http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv"
    url_red = "http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"

    wine_white = pd.read_csv(url_white, sep=';')
    wine_red = pd.read_csv(url_red, sep=';')

    wine_white['type'] = 'white'
    wine_red['type'] = 'red'

    wine_df = pd.concat([wine_white, wine_red], ignore_index=True)
    wine_df.info()
    return url_red, url_white, wine_df, wine_red, wine_white


@app.cell
def _(train_test_split, wine_df):
    train_wine, test_wine = train_test_split(wine_df, test_size=0.7, random_state=42)
    return test_wine, train_wine


@app.cell
def _(test_wine, train_wine):
    try:
        from flaml import AutoML

        X_train_wine = train_wine.drop(columns=['quality'])
        y_train_wine = train_wine['quality']

        settings = {
            "time_budget": 60,       # czas treningu w sekundach
            "metric": "r2",          # metryka oceny
            "task": "regression",    # typ zadania
            "log_file_name": "wine_experiment.log",
            "seed": 7654321,
        }

        automl = AutoML()
        automl.fit(X_train_wine, y_train_wine, **settings)

        # Predykcja
        from sklearn.metrics import r2_score as r2

        X_test_wine = test_wine.drop(columns=['quality'])
        y_test_wine = test_wine['quality']

        y_pred_wine = automl.predict(X_test_wine)
        print(f"AutoML R²: {r2(y_test_wine, y_pred_wine):.4f}")
        print(f"Najlepszy model: {automl.model.estimator}")
        print(f"Najlepsze parametry: {automl.best_config}")
        print(f"Czas treningu najlepszego modelu: {automl.best_config_train_time:.2f}s")

    except ImportError:
        print("Pakiet flaml nie jest zainstalowany.")
        print("Instalacja: pip install flaml")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **3. Pobierz dane z konkursu i wyślij swoje pierwsze wyniki w oparciu o AutoML.**
    """)
    return


if __name__ == "__main__":
    app.run()
