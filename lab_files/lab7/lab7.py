import marimo

__generated_with = "0.19.9"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Lab 7: Drzewa decyzyjne i modele pochodne

    Drzewa decyzyjne są dostępne w bibliotece **scikit-learn**.
    Używamy zbiorów danych **Carseats** (ISLR) i **Boston** (MASS).
    """)
    return


@app.cell
def _():
    import numpy as np
    import pandas as pd
    import statsmodels.api as sm
    from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, plot_tree
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, BaggingRegressor
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import (classification_report, accuracy_score,
                                  mean_squared_error, r2_score)
    import plotly.express as px
    import plotly.graph_objects as go
    import matplotlib.pyplot as plt
    return (
        BaggingRegressor,
        DecisionTreeClassifier,
        DecisionTreeRegressor,
        RandomForestClassifier,
        RandomForestRegressor,
        accuracy_score,
        classification_report,
        cross_val_score,
        go,
        mean_squared_error,
        np,
        pd,
        plot_tree,
        plt,
        px,
        r2_score,
        sm,
        train_test_split,
    )


@app.cell
def _(sm):
    carseats = sm.datasets.get_rdataset("Carseats", "ISLR").data
    boston = sm.datasets.get_rdataset("Boston", "MASS").data
    print("Carseats:", carseats.shape)
    print("Boston:", boston.shape)
    return boston, carseats


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Drzewa klasyfikacyjne

    Klasyfikujemy sprzedaż foteli samochodowych (`Sales`) na:
    - **No** — niska sprzedaż (Sales ≤ 8)
    - **Yes** — wysoka sprzedaż (Sales > 8)
    """)
    return


@app.cell
def _(DecisionTreeClassifier, carseats, pd, train_test_split):
    carseats_h = carseats.copy()
    carseats_h['High'] = (carseats_h['Sales'] > 8).map({True: 'Yes', False: 'No'})

    X_car = carseats_h.drop(['Sales', 'High'], axis=1)
    X_car = pd.get_dummies(X_car, drop_first=True)
    y_car = (carseats_h['High'] == 'Yes').astype(int)

    X_car_train, X_car_test, y_car_train, y_car_test = train_test_split(
        X_car, y_car, test_size=0.3, random_state=42
    )

    tree_model = DecisionTreeClassifier(random_state=42)
    tree_model.fit(X_car_train, y_car_train)
    y_car_pred = tree_model.predict(X_car_test)

    print("Drzewo klasyfikacyjne (Carseats):")
    print(f"  Liczba liści: {tree_model.get_n_leaves()}")
    print(f"  Głębokość: {tree_model.get_depth()}")
    print(f"  Dokładność: {(y_car_pred == y_car_test).mean():.3f}")
    return (
        X_car,
        X_car_test,
        X_car_train,
        carseats_h,
        tree_model,
        y_car,
        y_car_pred,
        y_car_test,
        y_car_train,
    )


@app.cell
def _(classification_report, tree_model, y_car_pred, y_car_test):
    print("Raport klasyfikacji:")
    print(classification_report(y_car_test, y_car_pred, target_names=['No', 'Yes']))

    # Ważność cech
    import pandas as pd_imp
    feat_imp = pd_imp.DataFrame({
        'Feature': tree_model.feature_names_in_,
        'Importance': tree_model.feature_importances_
    }).sort_values('Importance', ascending=False)
    print("\nNajważniejsze predyktory:")
    print(feat_imp.head(5))
    return feat_imp, pd_imp


@app.cell
def _(plt, plot_tree, tree_model):
    # Wizualizacja drzewa (pierwsze 3 poziomy)
    fig_tree, ax_tree = plt.subplots(figsize=(16, 6))
    plot_tree(tree_model, max_depth=3, filled=True,
              feature_names=tree_model.feature_names_in_,
              class_names=['No', 'Yes'],
              fontsize=8, ax=ax_tree)
    plt.title("Drzewo klasyfikacyjne (Carseats) - pierwsze 3 poziomy")
    plt.tight_layout()
    plt.show()
    return ax_tree, fig_tree


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **1. Które predyktory są najbardziej istotne?**

    ## Przycinanie drzewa (Pruning)

    scikit-learn używa **cost-complexity pruning** (parametr `ccp_alpha`).
    Rosnące alfa eliminuje kolejne najmniej efektywne liście — tworząc ścieżkę
    pruningową.
    """)
    return


@app.cell
def _(DecisionTreeClassifier, X_car_train, go, y_car_train):
    clf_prune = DecisionTreeClassifier(random_state=41)
    path = clf_prune.cost_complexity_pruning_path(X_car_train, y_car_train)
    ccp_alphas, impurities = path.ccp_alphas, path.impurities

    fig_prune = go.Figure()
    fig_prune.add_scatter(x=ccp_alphas[:-1], y=impurities[:-1],
                          mode='lines+markers', name='Zanieczyszczenie liści')
    fig_prune.update_layout(title='Zanieczyszczenie vs alpha (ścieżka pruningowa)',
                            xaxis_title='Effective alpha', yaxis_title='Total impurity')
    fig_prune.show()
    return ccp_alphas, clf_prune, fig_prune, impurities, path


@app.cell
def _(DecisionTreeClassifier, X_car_test, X_car_train, classification_report, y_car_test, y_car_train):
    # Wybieramy alpha = 0.009 (można eksperymentować)
    pruned_tree = DecisionTreeClassifier(random_state=0, ccp_alpha=0.009)
    pruned_tree.fit(X_car_train, y_car_train)
    y_pruned_pred = pruned_tree.predict(X_car_test)

    print(f"Drzewo po pruning (alpha=0.009):")
    print(f"  Liczba liści: {pruned_tree.get_n_leaves()}")
    print(f"  Głębokość: {pruned_tree.get_depth()}")
    print(classification_report(y_car_test, y_pruned_pred, target_names=['No', 'Yes']))
    return pruned_tree, y_pruned_pred


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **2. Narysuj wykres błędu testowego w zależności od rozmiaru poddrzewa.**

    ## Drzewa regresyjne

    Zbiór Boston — przewidywanie `medv` (mediana wartości domów).
    """)
    return


@app.cell
def _(DecisionTreeRegressor, boston, mean_squared_error, train_test_split):
    X_boston = boston.drop('medv', axis=1)
    y_boston = boston['medv']

    X_b_train, X_b_test, y_b_train, y_b_test = train_test_split(
        X_boston, y_boston, test_size=0.5, random_state=1
    )

    medv_tree = DecisionTreeRegressor(random_state=43)
    medv_tree.fit(X_b_train, y_b_train)
    medv_pred = medv_tree.predict(X_b_test)

    mse_tree = mean_squared_error(y_b_test, medv_pred)
    print(f"Drzewo regresyjne (Boston):")
    print(f"  Liczba liści: {medv_tree.get_n_leaves()}")
    print(f"  Głębokość: {medv_tree.get_depth()}")
    print(f"  MSE testowe: {mse_tree:.2f}")
    return (
        X_b_test,
        X_b_train,
        X_boston,
        medv_pred,
        medv_tree,
        mse_tree,
        y_b_test,
        y_b_train,
        y_boston,
    )


@app.cell
def _(DecisionTreeRegressor, X_b_test, X_b_train, go, mean_squared_error, np, y_b_test, y_b_train):
    # Porównanie drzew o różnej głębokości (ścieżka CV)
    from sklearn.model_selection import cross_val_score as cvs

    max_depths = range(1, 15)
    cv_scores_tree = []
    test_mses = []

    for depth in max_depths:
        dt = DecisionTreeRegressor(max_depth=depth, random_state=1)
        cv_sc = -cvs(dt, X_b_train, y_b_train, cv=5, scoring='neg_mean_squared_error')
        cv_scores_tree.append(cv_sc.mean())
        dt.fit(X_b_train, y_b_train)
        test_mses.append(mean_squared_error(y_b_test, dt.predict(X_b_test)))

    opt_depth = max_depths[np.argmin(cv_scores_tree)]
    print(f"Optymalna głębokość (CV): {opt_depth}")

    fig_cv_tree = go.Figure()
    fig_cv_tree.add_scatter(x=list(max_depths), y=cv_scores_tree, name='CV MSE', mode='lines+markers')
    fig_cv_tree.add_scatter(x=list(max_depths), y=test_mses, name='Test MSE', mode='lines+markers')
    fig_cv_tree.add_vline(x=opt_depth, line_dash='dash', line_color='red')
    fig_cv_tree.update_layout(title='MSE vs głębokość drzewa', xaxis_title='Max depth', yaxis_title='MSE')
    fig_cv_tree.show()
    return (
        cvs,
        cv_scores_tree,
        depth,
        dt,
        fig_cv_tree,
        max_depths,
        opt_depth,
        test_mses,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Bagging i lasy losowe

    Bagging to szczególny przypadek lasu losowego (mtry = p).

    ### Bagging
    """)
    return


@app.cell
def _(RandomForestRegressor, X_b_test, X_b_train, mean_squared_error, y_b_test, y_b_train):
    # Bagging: mtry = liczba wszystkich cech = p
    n_features = X_b_train.shape[1]

    medv_bag = RandomForestRegressor(
        n_estimators=500,
        max_features=n_features,  # Bagging: wszystkie cechy
        random_state=2
    )
    medv_bag.fit(X_b_train, y_b_train)
    bag_pred = medv_bag.predict(X_b_test)
    print(f"Bagging MSE: {mean_squared_error(y_b_test, bag_pred):.2f}")
    return bag_pred, medv_bag, n_features


@app.cell
def _(X_boston, go, medv_bag, pd):
    # Ważność predyktorów
    feat_imp_bag = pd.DataFrame({
        'Feature': X_boston.columns,
        'Importance': medv_bag.feature_importances_
    }).sort_values('Importance', ascending=False)

    fig_imp_bag = go.Figure()
    fig_imp_bag.add_bar(x=feat_imp_bag['Feature'], y=feat_imp_bag['Importance'])
    fig_imp_bag.update_layout(title='Ważność predyktorów (Bagging)',
                              xaxis_title='Predyktor', yaxis_title='Ważność')
    fig_imp_bag.show()
    return feat_imp_bag, fig_imp_bag


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Lasy losowe

    Domyślna wartość parametru `max_features` to $\sqrt{p}$ dla klasyfikacji i
    $p/3$ dla regresji (`'sqrt'` i `'auto'` / `None`).
    """)
    return


@app.cell
def _(RandomForestRegressor, X_b_test, X_b_train, go, mean_squared_error, pd, y_b_test, y_b_train):
    # Domyślny las losowy (max_features = 1/3 cech dla regresji)
    medv_rf = RandomForestRegressor(
        n_estimators=500,
        max_features='sqrt',
        random_state=2
    )
    medv_rf.fit(X_b_train, y_b_train)
    rf_pred = medv_rf.predict(X_b_test)
    print(f"Las losowy MSE (sqrt features): {mean_squared_error(y_b_test, rf_pred):.2f}")

    # Ważność predyktorów
    feat_imp_rf = pd.DataFrame({
        'Feature': X_b_train.columns,
        'Importance': medv_rf.feature_importances_
    }).sort_values('Importance', ascending=False)

    fig_imp_rf = go.Figure()
    fig_imp_rf.add_bar(x=feat_imp_rf['Feature'], y=feat_imp_rf['Importance'])
    fig_imp_rf.update_layout(title='Ważność predyktorów (Las losowy)',
                             xaxis_title='Predyktor', yaxis_title='Ważność')
    fig_imp_rf.show()
    return feat_imp_rf, fig_imp_rf, medv_rf, rf_pred


@app.cell
def _(RandomForestRegressor, X_b_test, X_b_train, go, mean_squared_error, np, y_b_test, y_b_train):
    # Porównanie OOB errors dla różnych wartości mtry
    mtry_values = [2, 4, 6, 8, 13]
    mse_mtry = []

    for mtry in mtry_values:
        rf_mtry = RandomForestRegressor(n_estimators=500, max_features=mtry,
                                        random_state=2, oob_score=True)
        rf_mtry.fit(X_b_train, y_b_train)
        pred_mtry = rf_mtry.predict(X_b_test)
        mse_mtry.append(mean_squared_error(y_b_test, pred_mtry))

    fig_mtry = go.Figure()
    fig_mtry.add_scatter(x=mtry_values, y=mse_mtry, mode='lines+markers', name='Test MSE')
    fig_mtry.update_layout(title='MSE testowe vs mtry', xaxis_title='mtry', yaxis_title='MSE')
    fig_mtry.show()
    return fig_mtry, mse_mtry, mtry, mtry_values, pred_mtry, rf_mtry


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **6. Co można powiedzieć o istotności predyktorów w lasach losowych?**

    **7. Porównaj na wykresie błędy OOB dla baggingu i domyślnie skonfigurowanego
    lasu losowego.**
    """)
    return


if __name__ == "__main__":
    app.run()
