import marimo

__generated_with = "0.23.3"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Lab 8: Metody boostingowe

    Boosting to ogólne podejście do poprawy predykcji wielu metod uczenia statystycznego. Skupimy się tu na zastosowaniu w drzewach decyzyjnych. Tak jak w baggingu, łączymy dużą liczbę drzew $\hat f^1, \dots, \hat f^B$, ale **rosną one sekwencyjnie**: każde kolejne korzysta z informacji od poprzednich, zamiast być trenowane niezależnie na bootstrapowych próbkach. W regresji oznacza to dopasowywanie kolejnych małych drzew do **rezyduów** aktualnego modelu. Zamiast jednego dużego drzewa, które łatwo przeuczyć, otrzymujemy sekwencję słabych modeli, z których każdy poprawia fragment błędu poprzednika.

    Trzy parametry, które trzeba świadomie ustawić:

    - **Liczba drzew $B$** — w odróżnieniu od baggingu i lasów losowych boosting może się przeuczyć przy zbyt dużym $B$ (choć powoli). Zwykle dobiera się go walidacją krzyżową albo wczesnym zatrzymaniem na zbiorze walidacyjnym.
    - **Współczynnik $\lambda$** (`learning_rate`) — skala, przez którą mnożony jest wkład każdego drzewa do ogólnej predykcji. Typowe wartości to 0.01 lub 0.001. Mniejsze $\lambda$ wymaga większego $B$, ale daje wolniejsze, "ostrożniejsze" uczenie i zwykle lepszą generalizację.
    - **Głębokość $d$** — liczba podziałów w pojedynczym drzewie, kontrolująca rząd interakcji w modelu. Często wystarczają pniaki ($d = 1$, ang. _stumps_), dające model addytywny.

    Główna zasada: **wolne uczenie się zwykle prowadzi do lepszych wyników**. Fundament teoretyczny: "Greedy Function Approximation: A Gradient Boosting Machine" (Friedman, 2001, https://www.jstor.org/stable/pdf/2699986.pdf), gdzie pokazano, że ten proces można interpretować jako spadek gradientu w przestrzeni funkcji.

    W tym laboratorium zapoznajemy się z trzema popularnymi implementacjami boostingu drzew:

    - **XGBoost** — https://xgboost.readthedocs.io
    - **LightGBM** — https://lightgbm.readthedocs.io
    - **CatBoost** — https://catboost.ai

    Każda z nich realizuje tę samą ideę nieco inaczej i optymalizuje inny aspekt. XGBoost kładzie nacisk na elastyczność i regularyzację, LightGBM na szybkość, a CatBoost na natywną obsługę zmiennych kategorycznych. Wykorzystamy dwa zbiory danych: **Boston** z ISLP, znany z poprzedniego laboratorium, gdzie zadaniem regresji jest mediana cen domów w tys. USD, oraz **Adult** z UCI, na którym zaprezentujemy obsługę zmiennych kategorycznych w klasyfikacji dochodu.
    """)
    return


@app.cell
def _():
    import time
    import marimo as mo
    import numpy as np
    import pandas as pd
    import plotly.graph_objects as go
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
    from ISLP import load_data
    import xgboost as xgb
    import lightgbm as lgb
    from catboost import CatBoostClassifier, Pool
    from catboost.utils import get_confusion_matrix

    return (
        CatBoostClassifier,
        Pool,
        get_confusion_matrix,
        go,
        lgb,
        load_data,
        mean_squared_error,
        mo,
        pd,
        train_test_split,
        xgb,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## XGBoost

    XGBoost to obecnie najbardziej elastyczna biblioteka boostingowa pod kątem parametryzacji. Wyróżnia ją wbudowana regularyzacja **L1 i L2** wpływająca na wagi liści (czego brakuje w klasycznej maszynie wzmocnienia gradientowego, ang. Gradient Boosting Machine), histogramowe szukanie podziałów (kandydaci na próg są dyskretyzowani do kubełków, co znacząco przyspiesza trenowanie) oraz rozrost drzew **poziomami** (level-wise) — wszystkie liście danego poziomu są dzielone, zanim model przejdzie głębiej. Powstają dzięki temu bardziej zrównoważone drzewa, co wymusza jednak rozpatrywanie również podziałów o niskim zysku.

    Artykuł źródłowy: "XGBoost: A Scalable Tree Boosting System" (2016) https://dl.acm.org/doi/pdf/10.1145/2939672.2939785

    Pierwsze podejście — Boston z ISLP, ten sam podział train/test co w lab7, najprostszy `XGBRegressor` z domyślnymi parametrami.
    """)
    return


@app.cell
def _(load_data, train_test_split):
    Boston = load_data('Boston')
    X_bos = Boston.drop(columns=['medv'])
    y_bos = Boston['medv']
    X_bos_tr, X_bos_te, y_bos_tr, y_bos_te = train_test_split(
        X_bos, y_bos, test_size=0.5, random_state=1
    )
    print(f"Boston: {X_bos_tr.shape[0]} trening / {X_bos_te.shape[0]} test  ({X_bos.shape[1]} cech)")
    return X_bos_te, X_bos_tr, y_bos_te, y_bos_tr


@app.cell
def _(X_bos_te, X_bos_tr, mean_squared_error, xgb, y_bos_te, y_bos_tr):
    xgb_basic = xgb.XGBRegressor(n_estimators=100, random_state=42)
    xgb_basic.fit(X_bos_tr, y_bos_tr)
    mse_xgb_basic = mean_squared_error(y_bos_te, xgb_basic.predict(X_bos_te))
    print(f"XGBoost (domyślne, 100 drzew) — MSE testowe: {mse_xgb_basic:.2f}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Parametryzacja XGBoost

    Rozbudowany zestaw parametrów daje dużą kontrolę nad procesem treningu. Najczęściej dostrajane:

    - `n_estimators` — liczba drzew $B$ z opisu boostingu wyżej.
    - `max_depth` — maksymalna głębokość pojedynczego drzewa, czyli parametr $d$.
    - `learning_rate` — $\lambda$ z opisu powyżej. Nie mylić ze współczynnikiem regularyzacji.
    - `reg_lambda`, `reg_alpha` — siła regularyzacji L2 i L1 na wagach liści.
    - `subsample`, `colsample_bytree` — odsetek wierszy/kolumn losowanych dla każdego drzewa, czyli element stochastyczny zapożyczony z lasu losowego.

    Pełna lista: https://xgboost.readthedocs.io/en/stable/parameter.html

    Przykładowy zestaw parametrów, bardziej zachowawczy od ustawień domyślnych (więcej drzew, mniejszy krok, płytsze drzewa, regularyzacja):
    """)
    return


@app.cell
def _(X_bos_te, X_bos_tr, mean_squared_error, xgb, y_bos_te, y_bos_tr):
    xgb_tuned = xgb.XGBRegressor(
        n_estimators=500,
        max_depth=3,
        learning_rate=0.05,
        reg_lambda=1.0,
        subsample=0.8,
        random_state=42,
    )
    xgb_tuned.fit(X_bos_tr, y_bos_tr)
    mse_xgb_tuned = mean_squared_error(y_bos_te, xgb_tuned.predict(X_bos_te))
    print(f"XGBoost (dostrojony) — MSE testowe: {mse_xgb_tuned:.2f}")
    return (xgb_tuned,)


@app.cell
def _(X_bos_tr, go, pd, xgb_tuned):
    imp_xgb = pd.Series(
        xgb_tuned.feature_importances_, index=X_bos_tr.columns
    ).sort_values(ascending=False)

    fig_xgb_imp = go.Figure(go.Bar(x=imp_xgb.index, y=imp_xgb.values))
    fig_xgb_imp.update_layout(
        title='XGBoost — ważność cech (Boston)',
        xaxis_title='Cecha', yaxis_title='Ważność',
    )
    fig_xgb_imp.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Ćwiczenie 1

    **a)** Przetestuj kilka zestawów parametrów `n_estimators`, `learning_rate`. Czy widać znaną z teorii zależność, że małe $\lambda$ wymaga większego $B$?

    **b)** Który z predyktorów okazał się najistotniejszy? Czy ranking pokrywa się z analizą lasu losowego z poprzedniego laboratorium?
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
    ## LightGBM

    LightGBM (gradient boosting Microsoftu) jest zaprojektowany pod kątem **szybkości** i **niskiego zużycia pamięci**, szczególnie na dużych zbiorach. Dwie kluczowe optymalizacje odróżniające go od XGBoost:

    - **Rozrost liści** (leaf-wise) — zamiast dzielić wszystkie liście danego poziomu, model w każdej iteracji wybiera ten, którego podział da największy zysk. Powstające drzewa są bardziej asymetryczne i lepiej wychwytują rzeczywiste zależności, ale łatwiej ulegają przeuczeniu — dlatego `num_leaves` (a nie `max_depth`) jest tu głównym parametrem kontrolującym złożoność.
    - **GOSS** (Gradient-based One-Side Sampling) i **EFB** (Exclusive Feature Bundling) — pierwsza optymalizacja próbkuje obserwacje preferując te o dużym gradiencie (czyli "trudne" dla aktualnego modelu), druga łączy wzajemnie wykluczające się rzadkie cechy w jedną cechę. Razem dają duże przyspieszenie bez znaczącej utraty jakości.

    LightGBM oferuje też mechanizm **wczesnego zatrzymania** — przerywa trening, gdy metryka na zbiorze walidacyjnym przestaje się poprawiać przez ustaloną liczbę iteracji.

    Artykuł źródłowy: “LightGBM: A Highly Efficient Gradient Boosting Decision Tree” (2017) https://proceedings.neurips.cc/paper/2017/file/6449f44a102fde848669bdd9eb6b76fa-Paper.pdf
    """)
    return


@app.cell
def _(X_bos_tr, lgb, train_test_split, y_bos_tr):
    X_bos_tr2, X_bos_val, y_bos_tr2, y_bos_val = train_test_split(
        X_bos_tr, y_bos_tr, test_size=0.2, random_state=0
    )

    lgb_model = lgb.LGBMRegressor(
        n_estimators=2000,
        num_leaves=15,
        learning_rate=0.05,
        random_state=42,
        verbose=-1,
    )
    lgb_model.fit(
        X_bos_tr2, y_bos_tr2,
        eval_set=[(X_bos_val, y_bos_val)],
        callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)],
    )
    print(f"Najlepsza iteracja (early stopping): {lgb_model.best_iteration_} z 2000")
    return (lgb_model,)


@app.cell
def _(X_bos_te, lgb_model, mean_squared_error, y_bos_te):
    mse_lgb = mean_squared_error(y_bos_te, lgb_model.predict(X_bos_te))
    print(f"LightGBM — MSE testowe: {mse_lgb:.2f}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Wczesne zatrzymanie zakończyło trening przy znacznie mniejszej liczbie drzew niż 2000 — model nie zyskiwał już na zbiorze walidacyjnym, więc kolejne iteracje wiązałyby się jedynie z ryzykiem przeuczenia.

    Różnicę między rozrostem level-wise a leaf-wise dobrze widać w rozkładzie głębokości liści we wszystkich wytrenowanych drzewach. XGBoost z `max_depth=3` produkuje płytkie, wyrównane drzewa — większość liści powinna ulokować się dokładnie na głębokości 3. LightGBM z `num_leaves=15` bez ograniczenia głębokości rośnie asymetrycznie, dzieląc tylko ten liść, który najbardziej zmniejsza stratę, więc liście rozkładają się na wielu poziomach, w tym znacznie głębszych:
    """)
    return


@app.cell
def _(go, lgb_model, xgb_tuned):
    depths_xgb = [
        line.count('\t')
        for tree in xgb_tuned.get_booster().get_dump()
        for line in tree.split('\n')
        if 'leaf=' in line
    ]
    df_lgb_trees = lgb_model.booster_.trees_to_dataframe()
    depths_lgb = (
        df_lgb_trees.loc[df_lgb_trees['split_feature'].isna(), 'node_depth'] - 1
    ).tolist()

    fig_depth = go.Figure()
    fig_depth.add_histogram(x=depths_xgb, name='XGBoost (level-wise, max_depth=3)', opacity=0.7)
    fig_depth.add_histogram(x=depths_lgb, name='LightGBM (leaf-wise, num_leaves=15)', opacity=0.7)
    fig_depth.update_layout(
        barmode='overlay',
        title='Rozkład głębokości liści we wszystkich drzewach',
        xaxis_title='Głębokość liścia', yaxis_title='Liczba liści',
    )
    fig_depth.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Porównanie ważności cech pokazuje natomiast, jak te dwa sposoby rozrostu oceniają znaczenie cech tego samego zbioru:
    """)
    return


@app.cell
def _(X_bos_tr, go, lgb_model, pd, xgb_tuned):
    imp_xgb_norm = pd.Series(xgb_tuned.feature_importances_, index=X_bos_tr.columns)
    imp_xgb_norm = imp_xgb_norm / imp_xgb_norm.sum()

    imp_lgb = pd.Series(lgb_model.feature_importances_, index=X_bos_tr.columns)
    imp_lgb_norm = imp_lgb / imp_lgb.sum()

    order_imp = imp_xgb_norm.sort_values(ascending=False).index

    fig_imp_cmp = go.Figure()
    fig_imp_cmp.add_bar(x=order_imp, y=imp_xgb_norm[order_imp].values, name='XGBoost', offsetgroup=0)
    fig_imp_cmp.add_bar(x=order_imp, y=imp_lgb_norm[order_imp].values, name='LightGBM', offsetgroup=1)
    fig_imp_cmp.update_layout(
        barmode='group',
        title='Porównanie ważności cech: XGBoost vs LightGBM (Boston)',
        xaxis_title='Cecha', yaxis_title='Ważność (znormalizowana)',
    )
    fig_imp_cmp.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Ćwiczenie 2

    **a)** Przetestuj różne `num_leaves` (np. 7, 31, 127) trzymając pozostałe parametry stałe. Jak zmienia się MSE testowe i przy której wartości model zaczyna się przeuczać?

    **b)** Wyłącz wczesne zatrzymanie (zostaw `n_estimators` jako jedyne kryterium) i wytrenuj model na pełnym zbiorze treningowym. Porównaj precyzję na zbiorze testowym i czas treningu z wersją z `early_stopping`. Czy zysk z wczesnego zatrzymania jest istotny?
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
    ## CatBoost

    CatBoost (Yandex) to biblioteka boostingowa zaprojektowana wokół jednej głównej funkcjonalności — **zmiennych kategorycznych**. W większości implementacji (XGBoost, LightGBM) trzeba je zakodować ręcznie (one-hot, target encoding), co przy zmiennych o dużej kardynalności jest kłopotliwe i ryzykowne. CatBoost obsługuje je natywnie, używając wariantu kodowania na podstawie zmiennej celu (target encoding) z istotnym usprawnieniem.

    Kluczowe pomysły:

    - **Uporządkowany boosting** (ordered boosting) — zwykły target encoding może prowadzić do wycieku informacji o zmiennej celu: kodując kategorię $i$ średnią wartością $y$ w tej kategorii, wykorzystujemy wartość $y_i$ do utworzenia jej własnej cechy. CatBoost rozwiązuje ten problem, ustalając losowy porządek obserwacji i kodując każde $y_i$ wyłącznie na podstawie obserwacji *wcześniejszych* w tym porządku. Analogicznie konstruowane są kolejne drzewa — predykcja dla obserwacji $i$ powstaje z modelu wytrenowanego bez tej obserwacji, co eliminuje przesunięcie predykcji (prediction shift).
    - **Drzewa symetryczne** (oblivious trees) — w każdym poziomie drzewa wszystkie węzły używają tego samego warunku podziału. Są one mniej elastyczne niż w XGBoost czy LightGBM, ale za to mniej podatne na przeuczenie, a predykcja jest błyskawiczna i sprowadza się do bitowego wyboru indeksu liścia.

    Artykuł źródłowy: "CatBoost: unbiased boosting with categorical features" (2018) https://proceedings.neurips.cc/paper/2018/file/14491b756b3a51daac41c24863285549-Paper.pdf

    ### Przykład: dane dochodowe Adult (UCI)

    Zadaniem jest predykcja, czy osoba zarabia powyżej 50 tys. USD rocznie, na podstawie cech demograficznych i zawodowych. Zbiór zawiera zmienne numeryczne i kategoryczne.
    """)
    return


@app.cell
def _(pd):
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
    return (income_df,)


@app.cell
def _(income_df, train_test_split):
    X_income = income_df.drop('income', axis=1)
    y_income = income_df['income'].astype(str).str.rstrip('.')

    cat_features = X_income.select_dtypes(exclude='number').columns.tolist()
    X_income[cat_features] = X_income[cat_features].fillna('missing').astype(str)

    X_income_train, X_income_test, y_income_train, y_income_test = train_test_split(
        X_income, y_income, test_size=0.2, random_state=42
    )

    print(f"Zmienne kategoryczne ({len(cat_features)}): {cat_features}")
    return (
        X_income_test,
        X_income_train,
        cat_features,
        y_income_test,
        y_income_train,
    )


@app.cell
def _(
    CatBoostClassifier,
    Pool,
    X_income_test,
    X_income_train,
    cat_features,
    y_income_test,
    y_income_train,
):
    train_pool = Pool(X_income_train, y_income_train, cat_features=cat_features)
    test_pool = Pool(X_income_test, y_income_test, cat_features=cat_features)

    cat_model = CatBoostClassifier(
        iterations=100,
        depth=4,
        learning_rate=0.1,
        loss_function='MultiClass',
        verbose=10,
        random_seed=42,
    )
    cat_model.fit(train_pool)
    return cat_model, test_pool


@app.cell
def _(cat_model, test_pool):
    preds_class = cat_model.predict(test_pool)
    preds_proba = cat_model.predict_proba(test_pool)

    print("Pierwsze predykcje (klasa):", preds_class[:5].flatten())
    print("Pierwsze prawdopodobieństwa:", preds_proba[:5])
    return


@app.cell
def _(X_income_train, cat_model, go):
    feature_importance = cat_model.get_feature_importance()
    feature_names_income = X_income_train.columns.tolist()

    fig_cat_imp = go.Figure()
    fig_cat_imp.add_bar(x=feature_names_income, y=feature_importance, orientation='v')
    fig_cat_imp.update_layout(
        title='CatBoost — ważność cech (Adult)',
        xaxis_title='Cecha', yaxis_title='Ważność',
        xaxis_tickangle=-45,
    )
    fig_cat_imp.show()
    return


@app.cell
def _(cat_model, get_confusion_matrix, test_pool):
    cm_cat = get_confusion_matrix(cat_model, test_pool)
    print("Macierz pomyłek (CatBoost):")
    print(cm_cat)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Zadania

    ## Zadanie 1 — porównanie trzech bibliotek na wine quality

    Wczytaj zbiory `winequality-red.csv` i `winequality-white.csv` z UCI, połącz je dodając kolumnę `type` ('red'/'white'). Celem jest predykcja `quality` (ocena 3-9) na podstawie pozostałych cech fizykochemicznych. Wytrenuj trzy modele — XGBoost, LightGBM, CatBoost — z porównywalnymi parametrami i wczesnym zatrzymaniem. Zmierz dla każdego: testowe MSE, czas treningu (`time.perf_counter()`) i liczbę faktycznie wytrenowanych drzew. Skomentuj różnice między modelami.

    Pomocnicze ładowanie danych:
    """)
    return


@app.cell
def _(pd):
    winequality_white = pd.read_csv(
        "http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv",
        sep=";",
    )
    winequality_red = pd.read_csv(
        "http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv",
        sep=";",
    )
    winequality_white['type'] = 'white'
    winequality_red['type'] = 'red'
    wine = pd.concat([winequality_white, winequality_red], ignore_index=True)
    print(f"Wine quality: {wine.shape[0]} obserwacji, {wine.shape[1]} kolumn")
    wine.head()
    return


@app.cell
def _():
    # Uzupełnij kod poniżej
    ...
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Zadanie 2 — strojenie i analiza ważności cech na wine quality

    Korzystając z danych z Zadania 1:

    **a)** Dla każdej z trzech bibliotek dobierz hiperparametry (`max_depth` lub `num_leaves` lub `depth`, `learning_rate`, `n_estimators` z wczesnym zatrzymaniem, ewentualnie regularyzacją). Możesz użyć walidacji krzyżowej (np. `sklearn.model_selection.GridSearchCV`) lub osobnego zbioru walidacyjnego. Zmierz jakość najlepszego modelu na zbiorze testowym.

    **b)** Dla każdej z trzech wytrenowanych instancji policz znormalizowaną ważność cech (`feature_importances_` dla XGBoost/LightGBM, `get_feature_importance()` dla CatBoost). Narysuj wszystkie trzy rankingi na jednym pogrupowanym wykresie słupkowym. Przy których cechach modele się zgadzają, a przy których nie? Czy zmienna `type` (kategoryczna) ma podobną ważność we wszystkich trzech, mimo że tylko CatBoost obsługuje ją natywnie? Pamiętaj, że dla XGBoost i LightGBM musisz zakodować ją liczbowo (np. `pd.get_dummies` lub mapowanie 0/1 — co da inny wynik analizy ważności).
    """)
    return


@app.cell
def _():
    # Uzupełnij kod poniżej
    ...
    return


if __name__ == "__main__":
    app.run()
