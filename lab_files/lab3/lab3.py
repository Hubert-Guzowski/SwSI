import marimo

__generated_with = "0.20.4"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Lab 4: Uogólnione modele liniowe (GLM)

    W statystyce dopasowujemy model do zaobserwowanych danych. W regresji liniowej
    zakładamy liniową kombinację predyktorów. Nie wszystkie zmienne objaśniane
    pasują do takiego modelu — GLM pozwalają modelować relacje przez **funkcję
    wiążącą** (link function).
    """)
    return


@app.cell
def _():
    import numpy as np
    import marimo as mo
    import pandas as pd
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import (classification_report, confusion_matrix,
                                  accuracy_score)
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
    from sklearn.neighbors import KNeighborsClassifier
    from scipy.stats import poisson
    import plotly.express as px

    return (
        KNeighborsClassifier,
        LinearDiscriminantAnalysis,
        QuadraticDiscriminantAnalysis,
        accuracy_score,
        classification_report,
        confusion_matrix,
        mo,
        np,
        pd,
        poisson,
        px,
        sm,
        train_test_split,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Regresja logistyczna

    Gdy zmienna objaśniana $Y$ przyjmuje wartości binarne (0/1), regresja liniowa
    nie sprawdza się — może przewidywać wartości poza przedziałem $[0, 1]$ i zakłada
    normalność reszt, co dla danych zero-jedynkowych nie zachodzi. Regresja
    logistyczna modeluje bezpośrednio prawdopodobieństwo $\Pr(Y = 1 \mid X)$ przez
    funkcję logistyczną, która bardziej naturalnie mieści wynik w $(0, 1)$.

    Funkcja wiążąca — **logit**:
    $$
    Y \sim \frac{1}{1 + e^{-(\beta_0 + \beta_1 X_1 + \cdots + \beta_p X_p)}}
    $$

    ### Przykład demonstracyjny
    """)
    return


@app.cell
def _(mo):
    beta0_slider = mo.ui.slider(-5, 5, value=-2, step=0.5, label="β₀ (wyraz wolny)")
    beta1_slider = mo.ui.slider(0.1, 10.1, value=3, step=0.5, label="β₁ (nachylenie)")
    n_slider = mo.ui.slider(50, 500, value=100, step=50, label="n (liczba obserwacji)")
    mo.vstack([
        mo.md("**Parametry modelu**"),
        beta0_slider,
        beta1_slider,
        n_slider,
    ])
    return beta0_slider, beta1_slider, n_slider


@app.cell
def _(beta0_slider, beta1_slider, n_slider, np, pd, px, sm):
    rng = np.random.default_rng(123)
    X_demo = rng.normal(size=n_slider.value)
    log_odds = beta0_slider.value + beta1_slider.value * X_demo
    p_demo = 1 / (1 + np.exp(-log_odds))
    Y_demo = rng.binomial(1, p_demo)

    demo_model = sm.GLM(Y_demo, sm.add_constant(X_demo), family=sm.families.Binomial()).fit()
    decision_boundary = -demo_model.params[0] / demo_model.params[1]
    pred_probs_demo = demo_model.predict()

    df_demo = pd.DataFrame({'X': X_demo, 'Y': Y_demo, 'pred_probs': pred_probs_demo})

    fig_demo = px.scatter(df_demo, x='X', y='Y', opacity=0.5,
                          title='Regresja logistyczna - przykład')
    fig_demo.add_scatter(x=X_demo[np.argsort(X_demo)], y=pred_probs_demo[np.argsort(X_demo)],
                         mode='lines', name='P(Y=1|X)', line=dict(color='blue'))
    fig_demo.add_vline(x=decision_boundary, line_dash='dash', line_color='red',
                       annotation_text=f'Granica: X={decision_boundary:.2f}')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Dane Titanic

    Zbiór danych Titanic zawiera dane pasażerów statku i informację o tym,
    czy przeżyli katastrofę.
    """)
    return


@app.cell
def _(pd):
    titanic_data = pd.read_csv(
        "https://web.stanford.edu/class/archive/cs/cs109/cs109.1166/stuff/titanic.csv"
    )
    titanic_data.describe()
    return (titanic_data,)


@app.cell
def _(titanic_data, train_test_split):
    # Usuwamy imiona (kolumna Name) - nie wnoszą istotnej informacji
    titanic_clean = titanic_data.drop(columns=['Name'])
    # Kodujemy płeć jako 0/1 (sm.GLM nie nam domyślnie zmiennych kategorycznych)
    titanic_clean = titanic_clean.copy()
    titanic_clean['Sex'] = (titanic_clean['Sex'] == 'male').astype(int)

    # Żeby zweryfikować, jak model uogólnia się na niewidziane wcześniej przypadki, dzielimy zbiór na treningowy i testowy
    train_data, test_data = train_test_split(titanic_clean, test_size=0.3, random_state=123)
    print(f"Zbiór treningowy: {len(train_data)} wierszy")
    print(f"Zbiór testowy: {len(test_data)} wierszy")
    return test_data, train_data


@app.cell
def _(sm, train_data):
    # Dopasowanie modelu logistycznego
    X_titanic = sm.add_constant(train_data.drop(columns=['Survived']))
    y_titanic = train_data['Survived']

    titanic_model = sm.GLM(y_titanic, X_titanic, family=sm.families.Binomial()).fit()
    titanic_model.summary()
    return (titanic_model,)


@app.cell
def _(confusion_matrix, pd, sm, test_data, titanic_model):
    X_test_titanic = sm.add_constant(test_data.drop(columns=['Survived']))
    titanic_pred_probs = titanic_model.predict(X_test_titanic)
    titanic_pred_classes = (titanic_pred_probs > 0.5).astype(int)

    cm = confusion_matrix(test_data['Survived'], titanic_pred_classes)
    cm_df = pd.DataFrame(cm,
                         index=['Actual 0', 'Actual 1'],
                         columns=['Predicted 0', 'Predicted 1'])
    print("Macierz pomyłek:")
    print(cm_df)
    return (titanic_pred_classes,)


@app.cell
def _(accuracy_score, classification_report, test_data, titanic_pred_classes):
    print("Metryki jakości modelu:")
    print(f"Dokładność: {accuracy_score(test_data['Survived'], titanic_pred_classes):.3f}")
    print(classification_report(test_data['Survived'], titanic_pred_classes))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Regresja Poissonowska

    Gdy $Y$ zlicza rzadkie zdarzenia (wypadki, awarie, zgony), przyjmuje
    nieujemne wartości całkowite o rozkładzie dalekim od normalnego. Regresja
    liniowa mogłaby przewidywać wartości ujemne, a wariancja danych zliczeniowych
    rośnie wraz ze średnią — czego model gaussowski nie uwzględnia. Regresja
    Poissonowska modeluje warunkową oczekiwaną liczbę zdarzeń $\lambda = E[Y \mid X]$
    przez logarytm, który gwarantuje dodatniość predykcji.

    Funkcja wiążąca — **logarytm**:
    $$
    Y \sim \text{Pois}\left(e^{\beta_0 + \beta_1 X_1 + \cdots + \beta_p X_p}\right)
    $$

    ### Przykład: kopnięcia konia
    """)
    return


@app.cell
def _(pd):
    from pathlib import Path
    DATA_DIR = Path(__file__).parents[2] / "data"
    kicks_df = pd.read_csv(DATA_DIR / 'kicks.csv', index_col=0)
    kicks_df.head()
    return (kicks_df,)


@app.cell
def _(kicks_df, pd):
    kicks_long = kicks_df.stack().reset_index()
    kicks_long.columns = ['battalion', 'year', 'deaths']
    kicks_long['year'] = pd.to_numeric(kicks_long['year'])
    kicks_long.head()
    return (kicks_long,)


@app.cell
def _(kicks_long, np, pd, poisson):
    observed_counts = kicks_long.groupby('deaths').size()
    total_corps_years = sum(observed_counts)
    observed_props = observed_counts / total_corps_years

    lambda_hat = sum(np.array([0, 1, 2, 3, 4]) * observed_counts) / total_corps_years
    print(f"Estymata lambda: {lambda_hat:.3f}")

    poisson_probs = poisson.pmf(k=[0, 1, 2, 3, 4], mu=lambda_hat)
    expected_counts = poisson_probs * total_corps_years

    results_kicks = pd.DataFrame({
        'Deaths': [0, 1, 2, 3, 4],
        'Observed': observed_counts.values,
        'Expected': expected_counts,
        'Observed_%': observed_props.values * 100,
        'Expected_%': poisson_probs * 100
    })
    results_kicks
    return


@app.cell
def _(kicks_long, pd, sm):
    kicks_model = kicks_long.copy()
    kicks_model['battalion'] = pd.factorize(kicks_model['battalion'])[0]
    poisson_model = sm.GLM(
        kicks_model['deaths'],
        sm.add_constant(kicks_model[['battalion', 'year']]),
        family=sm.families.Poisson()
    ).fit()
    poisson_model.summary()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Regresja porządkowa

    Gdy $Y$ ma kategoryczny charakter z **naturalnym porządkiem** (np. niski/średni/wysoki,
    ocena 1–5), zwykła regresja logistyczna ignoruje tę kolejność traktując klasy jako
    równorzędne, a regresja liniowa zakłada równe odległości między stopniami — co rzadko
    jest uzasadnione. Regresja porządkowa modeluje **skumulowane prawdopodobieństwa**
    $P(Y \leq k)$ dla każdego progu $k$, zachowując informację o kolejności kategorii.

    Funkcja wiążąca — **logit skumulowany**:
    $$
    P(Y \leq k) = \frac{e^{\theta_k - (\beta_0 + \beta_1 X_1 + \cdots + \beta_p X_p)}}
                       {1 + e^{\theta_k - (\beta_0 + \beta_1 X_1 + \cdots + \beta_p X_p)}}
    $$

    Progi $\theta_k$ są wspólnymi parametrami modelu wyznaczającymi granice między
    klasami; efekty predyktorów $\beta$ są natomiast wspólne dla wszystkich progów
    (założenie proporcjonalnych szans).

    W Pythonie używamy `statsmodels.miscmodels.ordinal_model.OrderedModel`.
    """)
    return


@app.cell
def _(np, pd):
    from scipy.stats import logistic as logistic_dist

    rng2 = np.random.default_rng(123)
    X_ord = rng2.normal(size=200)
    latent_score = -2 + 3 * X_ord + logistic_dist.rvs(size=200, random_state=123)
    Y_ord = pd.cut(latent_score,
                   bins=[-np.inf, -1, 1, np.inf],
                   labels=['Low', 'Medium', 'High'])

    df_ord = pd.DataFrame({'X': X_ord, 'Y': Y_ord})
    df_ord.head()
    return (df_ord,)


@app.cell
def _(df_ord, np, pd, px):
    from statsmodels.miscmodels.ordinal_model import OrderedModel

    ord_model = OrderedModel(df_ord['Y'], df_ord[['X']], distr='logit')
    ord_result = ord_model.fit(method='bfgs', disp=False)
    print(ord_result.summary())

    # Wizualizacja predykowanych prawdopodobieństw
    X_grid_ord = np.linspace(df_ord['X'].min(), df_ord['X'].max(), 300)
    pred_probs_ord = ord_result.predict(exog=X_grid_ord.reshape(-1, 1))

    pred_df = pd.DataFrame(pred_probs_ord, columns=['Low', 'Medium', 'High'])
    pred_df['X'] = X_grid_ord

    fig_ord = px.line(pred_df, x='X', y=['Low', 'Medium', 'High'],
                      title='Regresja porządkowa - prawdopodobieństwa klas',
                      labels={'value': 'Prawdopodobieństwo', 'variable': 'Klasa'})
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Modele do porównania na Titanicu

    Regresja logistyczna modeluje $\Pr(Y = k \mid X = x)$ bezpośrednio — przez
    funkcję logistyczną opisującą warunkowy rozkład odpowiedzi $Y$ przy danych
    predyktorach $X$. Istnieje jednak alternatywne podejście: modelujemy rozkład
    predyktorów $X$ **osobno w każdej klasie** (tj. dla każdej wartości $Y$),
    a następnie korzystamy z **twierdzenia Bayesa**, aby odwrócić to w estymaty
    $\Pr(Y = k \mid X = x)$. Gdy rozkład $X$ w każdej klasie jest normalny,
    model przyjmuje postać bardzo zbliżoną do regresji logistycznej.

    Po co kolejna metoda, skoro mamy regresję logistyczną? Istnieje kilka powodów:

    - Gdy klasy są wyraźnie rozdzielone, estymaty parametrów regresji logistycznej
      bywają **niestabilne**. Metody opisane poniżej nie mają tego problemu.
    - Jeśli rozkład predyktorów $X$ jest w każdej klasie w przybliżeniu normalny,
      a próba jest mała, metody te mogą dawać **dokładniejsze wyniki** niż regresja
      logistyczna.
    - Metody te naturalnie rozszerzają się na przypadek **więcej niż dwóch klas**
      (choć wtedy można też stosować wielomianową regresję logistyczną).

    **Dalszy opis i bardzo dobre wytłumaczenie metod LDA i QDA wraz zwizualizacjami znajduje się w książce ISLP (link na stronie kursu).**

    ### LDA (Liniowa Analiza Dyskryminacyjna)

    Metoda statystyczna do klasyfikacji zakładająca wspólną macierz kowariancji
    dla wszystkich klas (rozkłady normalne). Optymalizuje liniowe granice decyzyjne.
    """)
    return


@app.cell
def _(LinearDiscriminantAnalysis, confusion_matrix, pd, test_data, train_data):
    titanic_lda = LinearDiscriminantAnalysis()
    X_train_lda = train_data.drop(columns=['Survived'])
    X_test_lda = test_data.drop(columns=['Survived'])

    titanic_lda.fit(X_train_lda, train_data['Survived'])
    lda_pred = titanic_lda.predict(X_test_lda)

    cm_lda = confusion_matrix(test_data['Survived'], lda_pred)
    print("LDA - macierz pomyłek:")
    print(pd.DataFrame(cm_lda, index=['Actual 0', 'Actual 1'],
                       columns=['Predicted 0', 'Predicted 1']))
    return X_test_lda, X_train_lda, lda_pred


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### QDA (Kwadratowa Analiza Dyskryminacyjna)

    Rozszerzenie LDA dopuszczające różne macierze kowariancji dla każdej klasy
    — prowadzi do nieliniowych (kwadratowych) granic decyzyjnych.
    """)
    return


@app.cell
def _(
    QuadraticDiscriminantAnalysis,
    X_test_lda,
    X_train_lda,
    confusion_matrix,
    pd,
    test_data,
    train_data,
):
    titanic_qda = QuadraticDiscriminantAnalysis()
    titanic_qda.fit(X_train_lda, train_data['Survived'])
    qda_pred = titanic_qda.predict(X_test_lda)

    cm_qda = confusion_matrix(test_data['Survived'], qda_pred)
    print("QDA - macierz pomyłek:")
    print(pd.DataFrame(cm_qda, index=['Actual 0', 'Actual 1'],
                       columns=['Predicted 0', 'Predicted 1']))
    return (qda_pred,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### KNN (K Nearest Neighbors)

    Metoda nieparametryczna. W scikit-learn predykcja jest częścią klasy
    `KNeighborsClassifier` (fit + predict). Ze względu na losowość w rozstrzyganiu
    remisów, ustawiamy `random_state`.
    """)
    return


@app.cell
def _(
    KNeighborsClassifier,
    X_test_lda,
    X_train_lda,
    confusion_matrix,
    pd,
    test_data,
    train_data,
):
    titanic_knn = KNeighborsClassifier(n_neighbors=5)
    titanic_knn.fit(X_train_lda, train_data['Survived'])
    knn_pred = titanic_knn.predict(X_test_lda)

    cm_knn = confusion_matrix(test_data['Survived'], knn_pred)
    print("KNN (k=5) - macierz pomyłek:")
    print(pd.DataFrame(cm_knn, index=['Actual 0', 'Actual 1'],
                       columns=['Predicted 0', 'Predicted 1']))
    return (knn_pred,)


@app.cell
def _(
    accuracy_score,
    knn_pred,
    lda_pred,
    qda_pred,
    test_data,
    titanic_pred_classes,
):
    print("Porównanie dokładności modeli:")
    print(f"  Regresja logistyczna: {accuracy_score(test_data['Survived'], titanic_pred_classes):.3f}")
    print(f"  LDA:                  {accuracy_score(test_data['Survived'], lda_pred):.3f}")
    print(f"  QDA:                  {accuracy_score(test_data['Survived'], qda_pred):.3f}")
    print(f"  KNN (k=5):            {accuracy_score(test_data['Survived'], knn_pred):.3f}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    # Zadania - GLM

    Pracujemy z wykorzystaniem zbioru **winequality** z repozytorium UCI Irvine
    (Opis zbioru: https://archive.ics.uci.edu/dataset/186/wine+quality).
    Został on zebrany w ramach badania mającego za cel sprawdzić, czy jesteśmy w stanie przewidywać preferencję osób dla konkretnego wina jedynie w oparciu o ich właściwości chemiczne i jest jednym z bardziej popularnych zbiorów w repozytorium UCI.
    """)
    return


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
    return


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
