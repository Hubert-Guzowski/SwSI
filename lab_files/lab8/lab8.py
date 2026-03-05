import marimo

__generated_with = "0.19.9"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Lab 8: Metody boostingowe

    Boosting to sekwencyjne trenowanie modeli, gdzie każdy kolejny stara się
    skorygować błędy poprzednich. Fundament teoretyczny: "Greedy Function
    Approximation: A Gradient Boosting Machine" (Friedman, 2001).

    W tym laboratorium zapoznajemy się z dwoma popularnymi implementacjami:
    - **XGBoost** — https://xgboost.readthedocs.io
    - **CatBoost** — https://catboost.ai

    (Warto też zapoznać się z **LightGBM** — https://lightgbm.readthedocs.io)

    Przed wykonaniem poniższych komórek upewnij się, że masz zainstalowane:
    ```
    pip install xgboost catboost
    ```
    """)
    return


@app.cell
def _():
    import numpy as np
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, r2_score
    import plotly.express as px
    import plotly.graph_objects as go
    return (
        accuracy_score,
        classification_report,
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
    ## XGBoost

    XGBoost to najbardziej elastyczna biblioteka boostingowa pod kątem parametryzacji.
    Posiada wbudowaną regularyzację L2 (w przeciwieństwie do LightGBM i CatBoost).

    ### Przykład: dane ToothGrowth

    Dane o wzroście zębów u świnek morskich przy różnych dawkach witaminy C.
    Przewidujemy typ suplementu (sok pomarańczowy vs kwas askorbinowy).
    """)
    return


@app.cell
def _(pd, train_test_split):
    import xgboost as xgb
    from statsmodels.datasets import get_rdataset

    tooth_data = get_rdataset("ToothGrowth").data
    tooth_data['supp_enc'] = (tooth_data['supp'] == 'VC').astype(int)

    X_tooth = tooth_data[['len', 'dose']]
    y_tooth = tooth_data['supp_enc']

    X_tooth_train, X_tooth_test, y_tooth_train, y_tooth_test = train_test_split(
        X_tooth, y_tooth, test_size=0.3, random_state=42
    )
    tooth_data.head()
    return (
        X_tooth,
        X_tooth_test,
        X_tooth_train,
        get_rdataset,
        tooth_data,
        xgb,
        y_tooth,
        y_tooth_test,
        y_tooth_train,
    )


@app.cell
def _(X_tooth_test, X_tooth_train, accuracy_score, xgb, y_tooth_test, y_tooth_train):
    # Podstawowy model XGBoost
    xgb_model = xgb.XGBClassifier(n_estimators=2, random_state=42, eval_metric='logloss')
    xgb_model.fit(X_tooth_train, y_tooth_train)

    # Predykcja
    y_pred_xgb = xgb_model.predict(X_tooth_test)
    y_proba_xgb = xgb_model.predict_proba(X_tooth_test)

    print(f"Dokładność XGBoost (podstawowy): {accuracy_score(y_tooth_test, y_pred_xgb):.3f}")
    print(f"\nSześć pierwszych predykcji (klasa): {y_pred_xgb[:6]}")
    print(f"Prawdopodobieństwa (VC): {y_proba_xgb[:6, 1]}")
    return xgb_model, y_pred_xgb, y_proba_xgb


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Parametryzacja XGBoost

    XGBoost oferuje rozbudowany zestaw parametrów kontrolujących trening:
    - `n_estimators` — liczba modeli składowych (drzew)
    - `max_depth` — maksymalna głębokość pojedynczego drzewa
    - `reg_lambda` — regularyzacja L2
    - `learning_rate` — krok gradientu (shrinkage)

    Pełna lista: https://xgboost.readthedocs.io/en/release_3.0.0/parameter.html
    """)
    return


@app.cell
def _(X_tooth_test, X_tooth_train, accuracy_score, xgb, y_tooth_test, y_tooth_train):
    xgb_conservative = xgb.XGBClassifier(
        n_estimators=5,
        max_depth=2,
        reg_lambda=0.5,
        learning_rate=0.15,
        random_state=42,
        eval_metric='logloss'
    )
    xgb_conservative.fit(X_tooth_train, y_tooth_train)

    y_pred_cons = xgb_conservative.predict(X_tooth_test)
    print(f"Dokładność XGBoost (konserw.): {accuracy_score(y_tooth_test, y_pred_cons):.3f}")
    print(f"\nPierwsze 6 predykcji: {y_pred_cons[:6]}")
    return xgb_conservative, y_pred_cons


@app.cell
def _(go, xgb_model):
    # Ważność cech
    feat_imp_xgb = xgb_model.feature_importances_

    fig_xgb_imp = go.Figure()
    fig_xgb_imp.add_bar(x=['len', 'dose'], y=feat_imp_xgb)
    fig_xgb_imp.update_layout(title='XGBoost - ważność predyktorów',
                              xaxis_title='Predyktor', yaxis_title='Ważność')
    fig_xgb_imp.show()
    return feat_imp_xgb, fig_xgb_imp


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **1. Przetestuj kilka zestawów parametrów w oparciu o dokumentację. Kod i
    wnioski zawrzyj poniżej.**

    **2. Który z predyktorów okazał się istotniejszy? Co o tym świadczy?**
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## CatBoost

    CatBoost jest szczególnie przydatny gdy dane zawierają zmienne kategoryczne
    — obsługuje je natywnie, bez potrzeby ręcznego kodowania.

    ### Przykład: dane dochodowe Adult (UCI)
    """)
    return


@app.cell
def _(pd, train_test_split):
    url_train = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
    url_test = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test"

    columns = [
        "age", "workclass", "fnlwgt", "education", "education-num",
        "marital-status", "occupation", "relationship", "race", "sex",
        "capital-gain", "capital-loss", "hours-per-week", "native-country", "income"
    ]

    df_train = pd.read_csv(url_train, header=None, names=columns,
                           na_values=" ?", skipinitialspace=True)
    df_test = pd.read_csv(url_test, header=0, names=columns,
                          na_values=" ?", skipinitialspace=True, comment='|')

    income_df = pd.concat([df_train, df_test], ignore_index=True)
    income_df.info()
    return columns, df_test, df_train, income_df, url_test, url_train


@app.cell
def _(income_df, np, train_test_split):
    from catboost import CatBoostClassifier, Pool

    X_income = income_df.drop('income', axis=1)
    y_income = income_df['income'].astype(str).str.rstrip('.')

    X_income_train, X_income_test, y_income_train, y_income_test = train_test_split(
        X_income, y_income, test_size=0.2, random_state=42
    )

    # CatBoost automatycznie wykrywa kolumny kategoryczne
    cat_features = np.where(X_income_train.dtypes == 'object')[0]
    print(f"Zmienne kategoryczne: {cat_features}")
    return (
        CatBoostClassifier,
        Pool,
        X_income,
        X_income_test,
        X_income_train,
        cat_features,
        y_income,
        y_income_test,
        y_income_train,
    )


@app.cell
def _(CatBoostClassifier, Pool, X_income_test, X_income_train, cat_features, y_income_test, y_income_train):
    # Pool to wygodna klasa reprezentująca zbiór danych w CatBoost
    train_pool = Pool(X_income_train, y_income_train, cat_features=cat_features)
    test_pool = Pool(X_income_test, y_income_test, cat_features=cat_features)

    cat_model = CatBoostClassifier(
        iterations=100,
        depth=4,
        learning_rate=0.1,
        loss_function='MultiClass',
        verbose=10,
        random_seed=42
    )
    cat_model.fit(train_pool)
    return cat_model, test_pool, train_pool


@app.cell
def _(cat_model, test_pool):
    preds_class = cat_model.predict(test_pool)
    preds_proba = cat_model.predict_proba(test_pool)

    print("Pierwsze predykcje (klasa):", preds_class[:5].flatten())
    print("Pierwsze prawdopodobieństwa:", preds_proba[:5])
    return preds_class, preds_proba


@app.cell
def _(X_income_train, cat_model, go):
    # Ważność cech
    feature_importance = cat_model.get_feature_importance()
    feature_names_income = X_income_train.columns.tolist()

    fig_cat_imp = go.Figure()
    fig_cat_imp.add_bar(x=feature_names_income, y=feature_importance,
                        orientation='v')
    fig_cat_imp.update_layout(title='CatBoost - ważność predyktorów',
                              xaxis_title='Predyktor', yaxis_title='Ważność',
                              xaxis_tickangle=-45)
    fig_cat_imp.show()
    return feature_importance, feature_names_income, fig_cat_imp


@app.cell
def _(cat_model, test_pool):
    from catboost.utils import get_confusion_matrix

    cm_cat = get_confusion_matrix(cat_model, test_pool)
    print("Macierz pomyłek (CatBoost):")
    print(cm_cat)
    return cm_cat, get_confusion_matrix


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **3. Wykorzystaj model boostingowy do predykcji jakości wina na podstawie
    zbioru UCI winequality (https://archive.ics.uci.edu/dataset/186/wine+quality).
    Wybierz i porównaj dwa rodzaje modeli (regresji, klasyfikacji, CatBoostRanker).**

    ## Tabela porównawcza bibliotek

    | Cecha | **XGBoost** | **LightGBM** | **CatBoost** |
    |---|---|---|---|
    | **Model bazowy** | Drzewa CART | Drzewa leaf-wise | Drzewa symetryczne |
    | **Wydajność** | Wysoka | Bardzo wysoka | Wysoka |
    | **Dane kategoryczne** | Wymaga kodowania | Wymaga kodowania | Wbudowana obsługa |
    | **Odporność na overfitting** | Dobra (L1/L2) | Może przeuczać | Wysoka (ordered boosting) |
    | **Zalety** | Stabilność, elastyczność | Szybkość, mała pamięć | Brak konieczności kodowania |
    | **Wady** | Ręczne przygotowanie danych | Wrażliwość na parametry | Dłuższy trening dla numerycznych |
    """)
    return


if __name__ == "__main__":
    app.run()
