import marimo

__generated_with = "0.19.9"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Lab 13: SHAP — SHapley Additive exPlanations

    ## Wprowadzenie

    Wartości SHAP to metoda oparta o teorię gier stosowana w uczeniu maszynowym
    do wyjaśniania predykcji modeli. Pozwalają zrozumieć, jak każda zmienna wpływa
    na konkretną predykcję.

    **Podstawy teoretyczne:**
    W teorii gier mamy:
    - **Graczy** (w ML: cechy/zmienne)
    - **Koalicje** (w ML: podzbiory cech)
    - **Funkcję wypłaty** (w ML: predykcję modelu)

    Wartość SHAP dla cechy $i$:
    $$
    \phi_i = \sum_{S \subseteq F \setminus \{i\}} \frac{|S|! \, (M - |S| - 1)!}{M!}
    \left[ f(S \cup \{i\}) - f(S) \right]
    $$

    SHAP jest matematycznie uzasadnione, niezależne od modelu i sprawiedliwie
    rozdziela wartość predykcji między cechy. Główne ograniczenia: wymagające
    obliczeniowo i nie wyjaśnia automatycznie korelacji między cechami.

    Przed wykonaniem upewnij się, że masz zainstalowane:
    ```
    pip install shap catboost
    ```
    """)
    return


@app.cell
def _():
    import numpy as np
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error, r2_score
    import plotly.express as px
    import plotly.graph_objects as go
    import matplotlib.pyplot as plt
    return go, mean_squared_error, np, pd, plt, px, r2_score, train_test_split


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Dane: Wine Quality

    Używamy zbioru wine-quality z UCI. Zadaniem jest predykcja jakości wina.
    """)
    return


@app.cell
def _(pd):
    url_white = "http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv"
    url_red = "http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"

    wine_white = pd.read_csv(url_white, sep=';')
    wine_red = pd.read_csv(url_red, sep=';')

    wine_white['white'] = 1
    wine_red['white'] = 0

    wine_data = pd.concat([wine_red, wine_white], ignore_index=True)

    print("Rozkład zmiennej docelowej:")
    print(wine_data['quality'].value_counts().sort_index())
    return url_red, url_white, wine_data, wine_red, wine_white


@app.cell
def _(px, wine_data):
    fig_hist = px.histogram(wine_data, x='quality', nbins=7,
                            title='Rozkład jakości wina',
                            labels={'quality': 'Jakość', 'count': 'Liczba obserwacji'})
    fig_hist.show()
    return (fig_hist,)


@app.cell
def _(px, wine_data):
    fig_corr = px.imshow(wine_data.corr().round(2),
                         text_auto=True, color_continuous_scale='RdBu_r',
                         title='Macierz korelacji - Wine Quality', zmin=-1, zmax=1)
    fig_corr.show()
    return (fig_corr,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## SHAP dla jakości wina

    Używamy CatBoost do predykcji jakości wina, a następnie SHAP do wyjaśnienia
    predykcji.
    """)
    return


@app.cell
def _(train_test_split, wine_data):
    y_wine = wine_data["quality"]
    X_wine = wine_data.drop(columns=["quality"])

    X_wine_train, X_wine_test, y_wine_train, y_wine_test = train_test_split(
        X_wine, y_wine, test_size=0.2, random_state=42
    )
    print(f"Train: {X_wine_train.shape}, Test: {X_wine_test.shape}")
    return X_wine, X_wine_test, X_wine_train, y_wine, y_wine_test, y_wine_train


@app.cell
def _(X_wine_test, X_wine_train, mean_squared_error, np, r2_score, y_wine_test, y_wine_train):
    try:
        from catboost import CatBoostRegressor

        cb_model = CatBoostRegressor(verbose=0, random_state=42)
        cb_model.fit(X_wine_train, y_wine_train)

        y_wine_pred = cb_model.predict(X_wine_test)
        print(f"RMSE: {np.sqrt(mean_squared_error(y_wine_test, y_wine_pred)):.3f}")
        print(f"R²: {r2_score(y_wine_test, y_wine_pred):.3f}")

    except ImportError:
        print("CatBoost nie jest zainstalowany.")
        print("Używamy GradientBoostingRegressor jako zastępnika:")
        from sklearn.ensemble import GradientBoostingRegressor

        cb_model = GradientBoostingRegressor(n_estimators=200, max_depth=3, random_state=42)
        cb_model.fit(X_wine_train, y_wine_train)
        y_wine_pred = cb_model.predict(X_wine_test)
        print(f"RMSE: {np.sqrt(mean_squared_error(y_wine_test, y_wine_pred)):.3f}")
        print(f"R²: {r2_score(y_wine_test, y_wine_pred):.3f}")
    return CatBoostRegressor, cb_model, y_wine_pred


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Wykres roju SHAP (Beeswarm)

    Agregat wartości SHAP dla wszystkich predykcji. Każda kropka = jedna obserwacja.
    Kolor = wartość cechy (czerwony = wysoka, niebieski = niska).
    """)
    return


@app.cell
def _(X_wine_test, cb_model, plt):
    try:
        import shap

        shap_explainer = shap.Explainer(cb_model)
        shap_values = shap_explainer(X_wine_test)

        plt.figure(figsize=(10, 8))
        shap.plots.beeswarm(shap_values, show=False)
        plt.tight_layout()
        plt.subplots_adjust(left=0.25)
        plt.title("SHAP Beeswarm - Wine Quality")
        plt.show()

    except ImportError:
        print("Pakiet shap nie jest zainstalowany.")
        print("Instalacja: pip install shap")
        shap_values = None
        shap_explainer = None
    return shap, shap_explainer, shap_values


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Wykres Waterfall dla pojedynczej predykcji

    Wykres waterfall przypomina formą wyjaśnienia LIME — pokazuje wkład
    poszczególnych cech w predykcję dla konkretnej obserwacji.
    """)
    return


@app.cell
def _(X_wine_test, cb_model, plt, shap_values, y_wine_pred, y_wine_test):
    if shap_values is not None:
        i_instance = 0  # Zmień, aby zobaczyć inne obserwacje

        plt.figure(figsize=(10, 5))
        import shap as shap_local
        shap_local.plots.waterfall(shap_values[i_instance], show=False)
        plt.tight_layout()
        plt.subplots_adjust(left=0.25)
        plt.show()

        print(f"Obserwacja {i_instance}:")
        print(f"  Rzeczywista jakość: {y_wine_test.iloc[i_instance]}")
        print(f"  Predykcja modelu: {y_wine_pred[i_instance]:.2f}")
        print(f"  Wartość bazowa: {shap_values[i_instance].base_values:.2f}")
    return i_instance, shap_local


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **2. Jakie dwie informacje przekazuje nam wykres scatter SHAP alkohol vs gęstość?**

    ### Wykres scatter: alkohol vs gęstość
    """)
    return


@app.cell
def _(plt, shap_values):
    if shap_values is not None:
        import shap as shap2
        plt.figure(figsize=(10, 5))
        shap2.plots.scatter(
            shap_values[:, "alcohol"],
            color=shap_values[:, "density"],
            show=False
        )
        plt.title("Wartości SHAP dla alkoholu pokolorowane według gęstości")
        plt.tight_layout()
        plt.show()
    return (shap2,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Analiza SHAP według rodzaju wina

    Dzielimy zbiór na wina czerwone i białe, i porównujemy wyjaśnienia SHAP.
    """)
    return


@app.cell
def _(X_wine_test, plt, shap_values):
    if shap_values is not None:
        import shap as shap3

        mask_red_wine = (X_wine_test['white'] == 0).values
        mask_white_wine = (X_wine_test['white'] == 1).values

        # Beeswarm dla win czerwonych
        shap_red = shap_values[mask_red_wine]
        plt.figure(figsize=(10, 7))
        shap3.plots.beeswarm(shap_red, show=False)
        plt.title(f"SHAP beeswarm - Wina czerwone (n={mask_red_wine.sum()})")
        plt.tight_layout()
        plt.subplots_adjust(left=0.25)
        plt.show()

        # Beeswarm dla win białych
        shap_white = shap_values[mask_white_wine]
        plt.figure(figsize=(10, 7))
        shap3.plots.beeswarm(shap_white, show=False)
        plt.title(f"SHAP beeswarm - Wina białe (n={mask_white_wine.sum()})")
        plt.tight_layout()
        plt.subplots_adjust(left=0.25)
        plt.show()
    return (
        mask_red_wine,
        mask_white_wine,
        shap3,
        shap_red,
        shap_white,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **3a)** Jakie największe różnice dla istotności cech można zaobserwować
    porównując wina czerwone i białe?

    **3b)** Jak zinterpretować cechę `white` widoczną na wykresie dla win czerwonych?

    ---

    ## SHAP dla rodzaju wina (zadanie demonstracyjne)

    Regresja logistyczna do predykcji rodzaju wina (biały = 1, czerwony = 0).
    """)
    return


@app.cell
def _(X_wine, plt, wine_data):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    X_type = wine_data.drop(columns=['white', 'quality'])
    y_type = wine_data['white']

    scaler_type = StandardScaler()
    X_type_scaled = scaler_type.fit_transform(X_type)

    lr_type = LogisticRegression(max_iter=1000, random_state=42)
    lr_type.fit(X_type_scaled, y_type)

    pred_type = lr_type.predict(X_type_scaled)
    accuracy_type = (pred_type == y_type).mean()
    print(f"Dokładność predykcji rodzaju wina: {accuracy_type:.3f}")

    # SHAP dla modelu liniowego
    try:
        import shap as shap4
        shap_lr_explainer = shap4.LinearExplainer(lr_type, X_type_scaled)
        shap_lr_values = shap_lr_explainer(X_type_scaled[:100])

        plt.figure(figsize=(10, 6))
        shap4.plots.beeswarm(shap_lr_values, show=False)
        plt.title("SHAP - Predykcja rodzaju wina (regresja logistyczna)")
        plt.tight_layout()
        plt.subplots_adjust(left=0.25)
        plt.show()

    except Exception as e:
        print(f"Błąd SHAP dla modelu liniowego: {e}")
    return (
        LogisticRegression,
        StandardScaler,
        X_type,
        X_type_scaled,
        accuracy_type,
        lr_type,
        pred_type,
        scaler_type,
        shap4,
        shap_lr_explainer,
        shap_lr_values,
        y_type,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **1. Zastanów się, jaki problem może pojawić się przy obliczaniu wartości SHAP
    dla modelu predykującego rodzaj wina, gdy cecha `white` jest w zbiorze predyktorów.
    Opisz swoje spostrzeżenia.**
    """)
    return


if __name__ == "__main__":
    app.run()
