import marimo

__generated_with = "0.20.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _():
    from pathlib import Path

    import numpy as np
    import pandas as pd
    import plotly.express as px
    import plotly.graph_objects as go
    from sklearn.preprocessing import StandardScaler
    import statsmodels.api as sm
    import statsmodels.formula.api as smf

    return Path, StandardScaler, go, np, pd, px, sm, smf


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Lab 2: Regresja liniowa

    Lab jest podzielony na dwie części:

    1. **Implementacja** regresji liniowej od podstaw.
    2. **Analiza** gotowej regresji liniowej z biblioteki `statsmodels` z rozbudowanymi raportami
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Część 1: Regresja liniowa od podstaw

    W tym laboratorium opieramy się na materiałach z kursu CS229 (Stanford): https://cs229.stanford.edu/main_notes.pdf. Poniżej skrót najważniejszych funkcji, które nas interesują (notacja lekko odbiega od materiału źródłowego. Pozostawiłem natomiast wektor parametrów jako $\theta$, choć chyba częściej spotykany jest zapis $\beta$).

    Model liniowy (zakłada obecność szumu gaussowskiego):

    $$y^{(i)} = \theta^T x^{(i)} + \epsilon^{(i)}, \qquad \epsilon^{(i)} \sim \mathcal{N}(0, \sigma^2)$$

    Ponieważ $\mathbb{E}[\epsilon] = 0$, jako predykcję przyjmujemy $h_\theta(x) = \mathbb{E}[y \mid x] = \theta^T x$. Minimalizujemy sumę kwadratów reszt:

    Funkcja kosztu:

    $$J(\theta) = \frac{1}{2} \|X\theta - y\|^2$$

    Gradient:

    $$\nabla_\theta J(\theta) = X^T(X\theta - y)$$

    Batchowa aktualizacja metodą najmniejszych kwadratów:

    $$\theta := \theta - \alpha \, X^T(X\theta - y)$$

    Inkrementalna aktualizacja metodą najmniejszych kwadratów. Dla każdego $i$:

    $$\theta := \theta - \alpha \left(h_\theta(x^{(i)}) - y^{(i)}\right) x^{(i)}$$
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Zadanie 1: Implementacja

    Uzupełnij implementację poniższych funkcji, żeby otrzymać funkcjonalną regresję liniową w dwóch wariantach treningu (batch i iteracyjny).
    """)
    return


@app.cell
def _(np):
    MAX_ITERATION = 1000
    LEARNING_RATE = 1e-5

    def initialize_theta(size):
        # TODO: zwróć wektor wartości początkowych o rozmiarze `size`
        ...

    def loss_function(theta, x, y):
        # TODO: oblicz i zwróć wartość funkcji kosztu J(theta)
        ...

    def loss_gradient(theta, x, y):
        # TODO: oblicz i zwróć gradient J względem theta
        ...

    def batch_least_mean_squares(learning_rate, x, y):
        intercept = np.ones(y.size)
        x_with_intercept = np.column_stack((intercept, x))
        theta = initialize_theta(x_with_intercept.shape[1])
        iteration = 0

        while iteration < MAX_ITERATION:
            # TODO: zaktualizuj theta
            ...
            iteration += 1

        return theta, loss_function(theta, x_with_intercept, y)

    def incremental_least_mean_squares(learning_rate, x, y):
        intercept = np.ones(y.size)
        x_with_intercept = np.column_stack((intercept, x))
        theta = initialize_theta(x_with_intercept.shape[1])
        iteration = 0

        while iteration < MAX_ITERATION:
            for i in range(y.size):
                # TODO: zaktualizuj theta
                ...
            iteration += 1

        return theta, loss_function(theta, x_with_intercept, y)

    return (
        LEARNING_RATE,
        batch_least_mean_squares,
        incremental_least_mean_squares,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Weryfikacja

    Dla przypadku 1d oczekujemy $\theta_1 \approx 1$, $\theta_0 \approx 0$.
    """)
    return


@app.cell
def _(
    LEARNING_RATE,
    batch_least_mean_squares,
    incremental_least_mean_squares,
    np,
):
    x_1d = np.linspace(0, 20).reshape(-1, 1)
    y_1d = np.linspace(0, 20)

    theta_batch_1d, loss_batch_1d = batch_least_mean_squares(LEARNING_RATE, x_1d, y_1d)
    theta_inc_1d, loss_inc_1d = incremental_least_mean_squares(LEARNING_RATE, x_1d, y_1d)

    print(f"Batch LMS:       theta={theta_batch_1d}, J={loss_batch_1d:.4f}")
    print(f"Incremental LMS: theta={theta_inc_1d}, J={loss_inc_1d:.4f}")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Zadanie 2: Analiza funkcji kosztu

    Poniżej mamy dwa zestawy danych do wytrenowania modelu. Jest między nimi bardzo istotna różnica. Aby ją wyłapać, zwizualizuj najpierw powierzchnie funkcji kosztu w zależności od współczynników $\theta$ przy obu wymiarach $x$. $\theta$ przy wyrazie wolnym traktujemy jako ustalone (równe 0). Następnie skomentuj różnicę między dwoma wykresami (bez zastosowania skali logarytmicznej będą trudne do rozróżnienia). Z czego ona wynika?
    """)
    return


@app.cell
def _(
    LEARNING_RATE,
    batch_least_mean_squares,
    incremental_least_mean_squares,
    np,
):
    x_2d_1 = np.linspace([0, 0], [20, 40], 50)
    y_2d_1 = np.linspace(0, 20, 50)

    theta_batch_2d_1, loss_batch_2d_1 = batch_least_mean_squares(LEARNING_RATE, x_2d_1, y_2d_1)
    theta_inc_2d_1, loss_inc_2d_1 = incremental_least_mean_squares(LEARNING_RATE, x_2d_1, y_2d_1)

    print(f"Batch LMS:       theta={theta_batch_2d_1}, J={loss_batch_2d_1:.4f}")
    print(f"Incremental LMS: theta={theta_inc_2d_1}, J={loss_inc_2d_1:.4f}")
    return


@app.cell
def _(
    LEARNING_RATE,
    batch_least_mean_squares,
    incremental_least_mean_squares,
    np,
):
    x_2d_2 = np.column_stack((np.linspace(0, 20, 50), np.random.uniform(0, 10, 50)))
    y_2d_2 = np.linspace(0, 20, 50)

    theta_batch_2d_2, loss_batch_2d_2 = batch_least_mean_squares(LEARNING_RATE, x_2d_2, y_2d_2)
    theta_inc_2d_2, loss_inc_2d_2 = incremental_least_mean_squares(LEARNING_RATE, x_2d_2, y_2d_2)

    print(f"Batch LMS:       theta={theta_batch_2d_2}, J={loss_batch_2d_2:.4f}")
    print(f"Incremental LMS: theta={theta_inc_2d_2}, J={loss_inc_2d_2:.4f}")
    return


@app.cell
def _():
    # Tutaj umieść kod wizualizacji
    return


@app.cell
def _(mo):
    mo.md(r"""
    A tutaj komentarz
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Dane mieszkaniowe – Kraków

    W danych znajduje się wstępnie obrobiony zbiór ofert sprzedaży mieszkań z Krakowa z kolumnami: `cena_brutto_pln`, `powierzchnia_uzytkowa`, `liczba_izb`, `kondygnacja`, `dzielnica`, który posłuży do kolejnych ćwiczeń.
    """)
    return


@app.cell
def _(Path, pd):
    housing_df = pd.read_csv(Path(__file__).parents[2] / "data" / "krakow_housing.csv", index_col=0)
    housing_df = housing_df[housing_df['cena_brutto_pln'] < 20_000_000]
    housing_df.info()
    return (housing_df,)


@app.cell
def _(mo):
    mo.md(r"""
    ## Zadanie 3: Zależności nieliniowe
    Regresję można bardzo łatwo zaadoptować do zmiennych nieliniowych. Dopasuj model regresji przewidujacy cenę brutto za pomocą zmiennych `powierzchnia_uzytkowa`, `liczba_izb`, `kondygnacja` oraz kwadratu `powierzchnia_uzytkowa`.
    """)
    return


@app.cell
def _():
    # Tutaj umieść kod
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## Część 2: Regresja liniowa z biblioteką `statsmodels`

    Biblioteka `statsmodels` dostarcza implementacje OLS z pełną diagnostyką statystyczną – odpowiednik `lm()` z R.

    ### Dane krakowskie – OLS
    """)
    return


@app.cell
def _(housing_df, sm):
    X_krk_simple = sm.add_constant(housing_df[['powierzchnia_uzytkowa']])
    y_krk_ols = housing_df['cena_brutto_pln']

    fit_krk_simple = sm.OLS(y_krk_ols, X_krk_simple).fit()
    print(fit_krk_simple.summary())
    return (fit_krk_simple,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Wizualizacja z przedziałami ufności
    """)
    return


@app.cell
def _(fit_krk_simple, go, housing_df, np, px, sm):
    x_grid_krk = np.linspace(
        housing_df['powierzchnia_uzytkowa'].min(),
        housing_df['powierzchnia_uzytkowa'].max(),
        200
    )
    X_grid_krk = sm.add_constant(x_grid_krk)
    pred_krk = fit_krk_simple.get_prediction(X_grid_krk).summary_frame()

    fig_krk_ols = px.scatter(
        housing_df.sample(min(2000, len(housing_df)), random_state=42),
        x='powierzchnia_uzytkowa', y='cena_brutto_pln',
        opacity=0.3,
        title='OLS: cena ~ powierzchnia (Kraków)',
        labels={'powierzchnia_uzytkowa': 'Powierzchnia [m²]', 'cena_brutto_pln': 'Cena brutto [PLN]'}
    )
    fig_krk_ols.add_trace(go.Scatter(x=x_grid_krk, y=pred_krk['mean'],
        mode='lines', name='OLS', line=dict(color='red', width=2)))
    fig_krk_ols.add_trace(go.Scatter(x=x_grid_krk, y=pred_krk['mean_ci_upper'],
        mode='lines', name='CI górny', line=dict(color='red', dash='dash', width=1)))
    fig_krk_ols.add_trace(go.Scatter(x=x_grid_krk, y=pred_krk['mean_ci_lower'],
        mode='lines', name='CI dolny', line=dict(color='red', dash='dash', width=1)))
    fig_krk_ols.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Regresja wielokrotna (Kraków)
    """)
    return


@app.cell
def _(housing_df, smf):
    fit_krk_multi = smf.ols(
        'cena_brutto_pln ~ powierzchnia_uzytkowa + liczba_izb + kondygnacja',
        data=housing_df
    ).fit()
    print(fit_krk_multi.summary())
    return (fit_krk_multi,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Dzielnica jako zmienna kategoryczna

    `dzielnica` reprezentuje kategorię – używamy `C(dzielnica)` aby automatycznie wygenerować zmienne dummy. Model regresji przyjmuje wtedy następujacą formę:

    $$\text{cena} = \theta_0 + \theta_1 \cdot \text{powierzchnia} + \theta_2 \cdot \text{liczba\_izb} + \theta_3 \cdot \text{kondygnacja} + \sum_{j=2}^{18} \gamma_j D_j + \epsilon$$

    Pomijamy jedną kategorię, bo przy jej zawarciu dla parametrów zachodziłoby:

    $$D_1 + D_2 + \cdots + D_{18} = 1 \quad \Rightarrow \quad D_1 = 1 - D_2 - \cdots - D_{18}$$

    Czyli mielibyśmy powtórkę z przykładu ze współliniowością i nieskończenie wiele poprawnych zestawów parametrów.
    """)
    return


@app.cell
def _(housing_df, smf):
    fit_krk_cat = smf.ols(
        'cena_brutto_pln ~ powierzchnia_uzytkowa + liczba_izb + kondygnacja + C(dzielnica)',
        data=housing_df
    ).fit()
    print(fit_krk_cat.summary())
    return (fit_krk_cat,)


@app.cell
def _(mo):
    mo.md(r"""
    ## Ćwiczenie 1.
    Zinterpretuj model. Co oznaczają kolejne elementy raportu? Które zmienne są najważniejsze w predykcji?
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    Tutaj zapisz wnioski
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Porównanie modeli (Kraków)
    """)
    return


@app.cell
def _(fit_krk_cat, fit_krk_multi, fit_krk_simple):
    print(f"{'Model':<45} {'R²':>8}  {'AIC':>12}")
    print("-" * 68)
    for _name, _fit in [
        ("prosta: powierzchnia", fit_krk_simple),
        ("wielokrotna: pow + izby + piętro", fit_krk_multi),
        ("wielokrotna + C(dzielnica)", fit_krk_cat),
    ]:
        print(f"{_name:<45} {_fit.rsquared:>8.4f}  {_fit.aic:>12.1f}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Wykres rezyduów
    """)
    return


@app.cell
def _(fit_krk_cat, px):
    _fitted = fit_krk_cat.fittedvalues
    _resid = fit_krk_cat.resid

    fig_resid_krk = px.scatter(
        x=_fitted, y=_resid,
        title='Residua vs. Wartości dopasowane (Kraków)',
        labels={'x': 'Wartości dopasowane', 'y': 'Residua'}
    )
    fig_resid_krk.add_hline(y=0, line_dash="dash", line_color="red")
    fig_resid_krk.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Standaryzacja cech a gradient descent

    Przy dużej różnicy skali między cechami krok uczenia musi być ekstremalnie mały.
    Poniżej widać, że na surowych danych krakowskich potrzebujemy `LR = 1e-12`, żeby gradient descent w ogóle zbiegał (do wywołania komórek trzeba zaimplementować regresję):
    """)
    return


@app.cell
def _(LEARNING_RATE, batch_least_mean_squares, housing_df, np):
    x_krk_raw = housing_df[['powierzchnia_uzytkowa']].values
    y_krk_raw = housing_df['cena_brutto_pln'].values

    _, loss_div = batch_least_mean_squares(LEARNING_RATE, x_krk_raw, y_krk_raw)
    print(f"LR={LEARNING_RATE} (surowe dane): J={loss_div:.2e}  ← dywergencja (nan/inf)")

    LEARNING_RATE_KRK = 1e-12
    theta_krk_raw, loss_krk_raw = batch_least_mean_squares(LEARNING_RATE_KRK, x_krk_raw, y_krk_raw)
    print(f"LR={LEARNING_RATE_KRK}:           J={loss_krk_raw:.2e}, theta={np.round(theta_krk_raw, 0)}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Standaryzacja $x \leftarrow \frac{x - \mu}{\sigma}$ wyrównuje skale i pozwala używać standardowego `LEARNING_RATE`.
    Dodatkowo uspójnia nam skalę współczynników między cechami, co mówi bardziej bezpośrednio o ich istotności.
    """)
    return


@app.cell
def _(
    LEARNING_RATE,
    StandardScaler,
    batch_least_mean_squares,
    housing_df,
    incremental_least_mean_squares,
    np,
):
    scaler = StandardScaler()
    x_krk_scaled = scaler.fit_transform(housing_df[['powierzchnia_uzytkowa']])
    y_krk_scaled = housing_df['cena_brutto_pln'].values

    theta_krk_scaled_batch, loss_krk_scaled_batch = batch_least_mean_squares(LEARNING_RATE, x_krk_scaled, y_krk_scaled)
    theta_krk_scaled_inc, loss_krk_scaled_inc = incremental_least_mean_squares(LEARNING_RATE, x_krk_scaled, y_krk_scaled)

    print(f"Batch LMS (skalowane):       theta={np.round(theta_krk_scaled_batch, 2)}, J={loss_krk_scaled_batch:.2e}")
    print(f"Incremental LMS (skalowane): theta={np.round(theta_krk_scaled_inc, 2)}, J={loss_krk_scaled_inc:.2e}")
    return


@app.cell
def _(StandardScaler, housing_df, smf):
    _housing_std = housing_df.copy()
    _housing_std[['powierzchnia_uzytkowa', 'liczba_izb', 'kondygnacja']] = StandardScaler().fit_transform(
        housing_df[['powierzchnia_uzytkowa', 'liczba_izb', 'kondygnacja']]
    )
    fit_krk_std = smf.ols(
        'cena_brutto_pln ~ powierzchnia_uzytkowa + liczba_izb + kondygnacja + C(dzielnica)',
        data=_housing_std
    ).fit()
    print(fit_krk_std.summary())
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Ćwiczenie 2.
    Jak zmieniły się współczynniki przy zmiennych? Czy zmienia to poprzednia interpretację ich istotności z ćwiczenia 1?
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    Tutaj zapisz wnioski
    """)
    return


if __name__ == "__main__":
    app.run()
