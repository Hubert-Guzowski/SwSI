import marimo

__generated_with = "0.19.9"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Lab 4: Uogólnione modele liniowe (GLM)

    W statystyce dopasowujemy model do zaobserwowanych danych. W regresji liniowej
    zakładamy liniową kombinację predyktorów. Nie wszystkie zmienne objaśniane
    pasują do takiego modelu — GLM pozwalają modelować relacje przez **funkcję
    łączącą** (link function).
    """)
    return


@app.cell
def _():
    import numpy as np
    import pandas as pd
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import (classification_report, confusion_matrix,
                                  accuracy_score, ConfusionMatrixDisplay)
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
    from sklearn.neighbors import KNeighborsClassifier
    from scipy.stats import poisson
    import plotly.express as px
    import plotly.graph_objects as go
    return (
        ConfusionMatrixDisplay,
        KNeighborsClassifier,
        LinearDiscriminantAnalysis,
        QuadraticDiscriminantAnalysis,
        accuracy_score,
        classification_report,
        confusion_matrix,
        go,
        np,
        pd,
        poisson,
        px,
        sm,
        smf,
        train_test_split,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Regresja logistyczna

    Funkcja mapująca — **logit**:
    $$
    Y \sim \frac{1}{1 + e^{-(\beta_0 + \beta_1 X_1 + \cdots + \beta_p X_p)}}
    $$

    ### Przykład demonstracyjny
    """)
    return


@app.cell
def _(np, pd, px, sm):
    rng = np.random.default_rng(123)
    X_demo = rng.normal(size=100)
    log_odds = -2 + 3 * X_demo
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
    fig_demo.show()
    return (
        Y_demo,
        X_demo,
        decision_boundary,
        demo_model,
        df_demo,
        fig_demo,
        log_odds,
        p_demo,
        pred_probs_demo,
        rng,
    )


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
    # Usuwamy imiona (kolumna Name)
    titanic_clean = titanic_data.drop(columns=['Name'])

    train_data, test_data = train_test_split(titanic_clean, test_size=0.3, random_state=123)
    print(f"Zbiór treningowy: {len(train_data)} wierszy")
    print(f"Zbiór testowy: {len(test_data)} wierszy")
    return test_data, titanic_clean, train_data


@app.cell
def _(sm, train_data):
    # Dopasowanie modelu logistycznego
    X_titanic = sm.add_constant(train_data.drop(columns=['Survived']))
    y_titanic = train_data['Survived']

    titanic_model = sm.GLM(y_titanic, X_titanic, family=sm.families.Binomial()).fit()
    titanic_model.summary()
    return X_titanic, titanic_model, y_titanic


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
    return X_test_titanic, cm, cm_df, titanic_pred_classes, titanic_pred_probs


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

    Funkcja łącząca — **logarytm**:
    $$
    Y \sim \text{Pois}\left(e^{\beta_0 + \beta_1 X_1 + \cdots + \beta_p X_p}\right)
    $$

    ### Przykład: kopnięcia konia

    Dane Bortkiewicza z 1898 r. — liczba żołnierzy zabitych przez kopnięcia
    konia w pruskiej kawalerii (14 korpusów, 20 lat).
    """)
    return


@app.cell
def _(pd):
    kicks_df = pd.read_csv('kicks.csv', index_col=0)
    kicks_df.head()
    return (kicks_df,)


@app.cell
def _(kicks_df):
    kicks_long = kicks_df.stack().reset_index()
    kicks_long.columns = ['battalion', 'year', 'deaths']
    kicks_long['year'] = pd.to_numeric(kicks_long['year'])
    kicks_long.head()
    return (kicks_long,)


@app.cell
def _(kicks_long, np, poisson):
    import pandas as pd_kicks
    observed_counts = kicks_long.groupby('deaths').size()
    total_corps_years = sum(observed_counts)
    observed_props = observed_counts / total_corps_years

    lambda_hat = sum(np.array([0, 1, 2, 3, 4]) * observed_counts) / total_corps_years
    print(f"Estymata lambda: {lambda_hat:.3f}")

    poisson_probs = poisson.pmf(k=[0, 1, 2, 3, 4], mu=lambda_hat)
    expected_counts = poisson_probs * total_corps_years

    results_kicks = pd_kicks.DataFrame({
        'Deaths': [0, 1, 2, 3, 4],
        'Observed': observed_counts.values,
        'Expected': expected_counts,
        'Observed_%': observed_props.values * 100,
        'Expected_%': poisson_probs * 100
    })
    results_kicks
    return (
        expected_counts,
        lambda_hat,
        observed_counts,
        observed_props,
        pd_kicks,
        poisson_probs,
        results_kicks,
        total_corps_years,
    )


@app.cell
def _(kicks_long, sm):
    poisson_model = sm.GLM(
        kicks_long['deaths'],
        kicks_long[['battalion', 'year']],
        family=sm.families.Poisson()
    ).fit()
    poisson_model.summary()
    return (poisson_model,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Regresja porządkowa

    Regresja porządkowa jest rozwinięciem regresji logistycznej dla **uporządkowanej**
    zmiennej $Y$ o $K$ kategoriach:
    $$
    P(Y \leq k) = \frac{e^{\theta_k - (\beta_0 + \beta_1 X_1 + \cdots + \beta_p X_p)}}
                       {1 + e^{\theta_k - (\beta_0 + \beta_1 X_1 + \cdots + \beta_p X_p)}}
    $$

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
    return X_ord, Y_ord, df_ord, latent_score, logistic_dist, rng2


@app.cell
def _(df_ord, np, px):
    from statsmodels.miscmodels.ordinal_model import OrderedModel

    ord_model = OrderedModel(df_ord['Y'], df_ord[['X']], distr='logit')
    ord_result = ord_model.fit(method='bfgs', disp=False)
    print(ord_result.summary())

    # Wizualizacja predykowanych prawdopodobieństw
    X_grid_ord = np.linspace(df_ord['X'].min(), df_ord['X'].max(), 300)
    pred_probs_ord = ord_result.predict(exog=X_grid_ord.reshape(-1, 1))

    import pandas as pd_ord
    pred_df = pd_ord.DataFrame(pred_probs_ord, columns=['Low', 'Medium', 'High'])
    pred_df['X'] = X_grid_ord

    fig_ord = px.line(pred_df, x='X', y=['Low', 'Medium', 'High'],
                      title='Regresja porządkowa - prawdopodobieństwa klas',
                      labels={'value': 'Prawdopodobieństwo', 'variable': 'Klasa'})
    fig_ord.show()
    return (
        OrderedModel,
        X_grid_ord,
        fig_ord,
        ord_model,
        ord_result,
        pd_ord,
        pred_df,
        pred_probs_ord,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Modele do porównania na Titanicu

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
    return X_test_lda, X_train_lda, cm_lda, lda_pred, titanic_lda


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### QDA (Kwadratowa Analiza Dyskryminacyjna)

    Rozszerzenie LDA dopuszczające różne macierze kowariancji dla każdej klasy
    — prowadzi do nieliniowych (kwadratowych) granic decyzyjnych.
    """)
    return


@app.cell
def _(QuadraticDiscriminantAnalysis, X_test_lda, X_train_lda, confusion_matrix, pd, test_data, train_data):
    titanic_qda = QuadraticDiscriminantAnalysis()
    titanic_qda.fit(X_train_lda, train_data['Survived'])
    qda_pred = titanic_qda.predict(X_test_lda)

    cm_qda = confusion_matrix(test_data['Survived'], qda_pred)
    print("QDA - macierz pomyłek:")
    print(pd.DataFrame(cm_qda, index=['Actual 0', 'Actual 1'],
                       columns=['Predicted 0', 'Predicted 1']))
    return cm_qda, qda_pred, titanic_qda


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
def _(KNeighborsClassifier, X_test_lda, X_train_lda, confusion_matrix, pd, test_data, train_data):
    titanic_knn = KNeighborsClassifier(n_neighbors=5)
    titanic_knn.fit(X_train_lda, train_data['Survived'])
    knn_pred = titanic_knn.predict(X_test_lda)

    cm_knn = confusion_matrix(test_data['Survived'], knn_pred)
    print("KNN (k=5) - macierz pomyłek:")
    print(pd.DataFrame(cm_knn, index=['Actual 0', 'Actual 1'],
                       columns=['Predicted 0', 'Predicted 1']))
    return cm_knn, knn_pred, titanic_knn


@app.cell
def _(accuracy_score, knn_pred, lda_pred, qda_pred, test_data, titanic_pred_classes):
    print("Porównanie dokładności modeli:")
    print(f"  Regresja logistyczna: {accuracy_score(test_data['Survived'], titanic_pred_classes):.3f}")
    print(f"  LDA:                  {accuracy_score(test_data['Survived'], lda_pred):.3f}")
    print(f"  QDA:                  {accuracy_score(test_data['Survived'], qda_pred):.3f}")
    print(f"  KNN (k=5):            {accuracy_score(test_data['Survived'], knn_pred):.3f}")
    return


if __name__ == "__main__":
    app.run()
