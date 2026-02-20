import marimo

__generated_with = "0.19.9"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Lab 11: LIME — Local Interpretable Model-agnostic Explanations

    LIME to technika tworzenia **lokalnych** wyjaśnień dla pojedynczych predykcji.
    Metoda działa poprzez:

    1. Generowanie permutacji wokół badanej obserwacji
    2. Trenowanie prostego interpretowalnego modelu na tych permutacjach
    3. Użycie tego modelu do wyjaśnienia lokalnego zachowania złożonego modelu

    **Ważne ograniczenia:**
    - Niestabilność wyjaśnień przy małej liczbie próbek
    - Brak gwarancji globalnej spójności
    - Wrażliwość na parametry

    **Literatura:**
    - Artykuł: https://arxiv.org/pdf/1602.04938
    - Krytyka LIME i SHAP: https://arxiv.org/pdf/1806.08049

    Przed wykonaniem upewnij się, że masz zainstalowane:
    ```
    pip install lime
    ```
    """)
    return


@app.cell
def _():
    import numpy as np
    import pandas as pd
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, classification_report
    import plotly.express as px
    import plotly.graph_objects as go
    return (
        RandomForestClassifier,
        accuracy_score,
        classification_report,
        go,
        np,
        pd,
        px,
        train_test_split,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Przygotowanie danych

    Używamy zbioru **UCI Air Quality** (dostępnego w katalogu `../lab3/`).
    """)
    return


@app.cell
def _(pd):
    air_quality_df = pd.read_csv("../lab3/AirQualityUCI.csv", sep=";", decimal=",")

    air_quality_df = air_quality_df.iloc[:, :-2]
    air_quality_df['Date'] = pd.to_datetime(air_quality_df['Date'], format='%d/%m/%Y')
    air_quality_df['Time'] = pd.to_datetime(air_quality_df['Time'], format='%H.%M.%S')

    columns_rename = {
        'CO(GT)': 'CO', 'PT08.S1(CO)': 'PT08_S1_CO', 'NMHC(GT)': 'NMHC',
        'C6H6(GT)': 'Benzene', 'PT08.S2(NMHC)': 'PT08_S2_NMHC',
        'NOx(GT)': 'NOx', 'PT08.S3(NOx)': 'PT08_S3_NOx',
        'NO2(GT)': 'NO2', 'PT08.S4(NO2)': 'PT08_S4_NO2',
        'PT08.S5(O3)': 'PT08_S5_O3', 'T': 'Temperature',
        'RH': 'RelativeHumidity', 'AH': 'AbsoluteHumidity'
    }
    air_quality_df = air_quality_df.rename(columns=columns_rename)

    print(air_quality_df.info())
    return air_quality_df, columns_rename


@app.cell
def _(air_quality_df, pd, train_test_split):
    # Czyszczenie danych
    air_data = air_quality_df.drop(columns=['Date', 'Time'])
    air_data = air_data.loc[:, ~air_data.isna().all()]
    air_data[air_data == -200] = __import__('numpy').nan

    # Usunięcie kolumn z >50% braków
    missing_pct = air_data.isna().mean()
    air_data = air_data.loc[:, missing_pct <= 0.5]

    # Imputacja medianą
    for col in air_data.columns:
        if air_data[col].dtype in ['float64', 'int64']:
            air_data[col].fillna(air_data[col].median(), inplace=True)

    # Zmienna docelowa i usunięcie NOx (zbyt skorelowany z NO2)
    air_data['high_pollution'] = (air_data['NO2'] > 140).astype(int)
    air_clean = air_data.drop(columns=['NO2', 'NOx'], errors='ignore')

    print(f"Rozkład zmiennej docelowej:\n{air_clean['high_pollution'].value_counts()}")

    X = air_clean.drop('high_pollution', axis=1)
    y = air_clean['high_pollution']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=123, stratify=y
    )
    print(f"\nTrain: {X_train.shape}, Test: {X_test.shape}")
    return (
        X,
        X_test,
        X_train,
        air_clean,
        air_data,
        col,
        missing_pct,
        y,
        y_test,
        y_train,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Budowa modelu bazowego (Random Forest)
    """)
    return


@app.cell
def _(RandomForestClassifier, X_test, X_train, accuracy_score, classification_report, y_test, y_train):
    rf_model = RandomForestClassifier(n_estimators=500, random_state=123)
    rf_model.fit(X_train.values, y_train.values)

    y_pred_rf = rf_model.predict(X_test.values)
    print(f"Dokładność modelu: {accuracy_score(y_test, y_pred_rf):.3f}")
    print("\nRaport klasyfikacji:")
    print(classification_report(y_test, y_pred_rf))
    return rf_model, y_pred_rf


@app.cell
def _(X_train, go, pd, rf_model):
    # Ważność cech globalnie
    feat_imp = pd.DataFrame({
        'Feature': X_train.columns,
        'Importance': rf_model.feature_importances_
    }).sort_values('Importance', ascending=False)

    fig_imp = go.Figure()
    fig_imp.add_bar(x=feat_imp['Feature'], y=feat_imp['Importance'])
    fig_imp.update_layout(title='Random Forest - globalna ważność cech',
                          xaxis_title='Cecha', yaxis_title='Ważność')
    fig_imp.show()
    return feat_imp, fig_imp


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## LIME — wyjaśnienia lokalne
    """)
    return


@app.cell
def _(X_test, X_train, rf_model, y_test):
    try:
        import lime
        import lime.lime_tabular

        # Tworzenie explainera LIME
        explainer = lime.lime_tabular.LimeTabularExplainer(
            X_train.values,
            feature_names=list(X_train.columns),
            class_names=['Low', 'High'],
            mode='classification',
            discretize_continuous=True
        )

        # Wyjaśnienie pierwszej obserwacji testowej
        instance_idx = 0
        instance = X_test.iloc[instance_idx].values
        actual = y_test.iloc[instance_idx]
        predicted = rf_model.predict([instance])[0]
        prob = rf_model.predict_proba([instance])[0]

        print(f"Instancja {instance_idx}:")
        print(f"  Rzeczywista etykieta: {actual}")
        print(f"  Predykcja modelu: {predicted}")
        print(f"  Prawdopodobieństwo [Low, High]: {prob}")

        explanation = explainer.explain_instance(
            instance,
            rf_model.predict_proba,
            num_features=6,
            num_samples=1000
        )

        print("\nWyjaśnienie LIME:")
        for feature, weight in explanation.as_list():
            print(f"  {feature}: {weight:.4f}")

    except ImportError:
        print("Pakiet lime nie jest zainstalowany.")
        print("Instalacja: pip install lime")
        explainer = None
        explanation = None
    return (
        actual,
        explainer,
        explanation,
        instance,
        instance_idx,
        lime,
        predicted,
        prob,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **1. Przeanalizuj wyjaśnienia LIME dla pierwszej obserwacji. Które czynniki
    środowiskowe najbardziej wpływają na predykcję? Zinterpretuj kierunki wpływu.**
    """)
    return


@app.cell
def _(X_test, explainer, go, pd, rf_model, y_test):
    if explainer is not None:
        # Analiza wielu obserwacji
        def analyze_multiple(explainer_obj, X_data, y_data, model, n=3):
            results = []
            for i in range(min(n, len(X_data))):
                inst = X_data.iloc[i].values
                actual_label = y_data.iloc[i]
                pred = model.predict([inst])[0]
                prob_val = model.predict_proba([inst])[0]

                exp = explainer_obj.explain_instance(
                    inst, model.predict_proba, num_features=5, num_samples=1000
                )

                results.append({
                    'instance': i,
                    'actual': actual_label,
                    'predicted': pred,
                    'prob_high': prob_val[1],
                    'top_features': exp.as_list()[:3]
                })

                print(f"\n--- Instancja {i} ---")
                print(f"Rzeczywista: {actual_label}, Predykcja: {pred}, Prob(High): {prob_val[1]:.3f}")
                for feat, w in exp.as_list()[:3]:
                    print(f"  {feat}: {w:.4f}")

            return results

        analysis_results = analyze_multiple(explainer, X_test, y_test, rf_model, 3)
    return (analyze_multiple, analysis_results)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Analiza stabilności wyjaśnień

    Stabilność sprawdzamy przez wielokrotne generowanie wyjaśnień dla tej samej
    obserwacji z różną liczbą próbek permutacyjnych.
    """)
    return


@app.cell
def _(X_test, explainer, go, rf_model):
    if explainer is not None:
        sample_sizes = [200, 500, 1000, 2000]
        test_instance = X_test.iloc[0].values

        stability_results = {}
        for n_samples in sample_sizes:
            exp = explainer.explain_instance(
                test_instance, rf_model.predict_proba,
                num_features=5, num_samples=n_samples
            )
            stability_results[n_samples] = dict(exp.as_list())

        # Porównanie wyjaśnień dla różnych liczb próbek
        import pandas as pd_stab
        stab_df = pd_stab.DataFrame(stability_results).T
        print("Stabilność wyjaśnień (wiersze = liczba próbek, kolumny = cechy):")
        print(stab_df.round(4))
    return (
        exp,
        n_samples,
        pd_stab,
        sample_sizes,
        stab_df,
        stability_results,
        test_instance,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Porównanie ważności globalnej vs lokalnej
    """)
    return


@app.cell
def _(X_test, feat_imp, explainer, go, pd, rf_model, y_test):
    if explainer is not None:
        # Obliczamy lokalną ważność dla N=10 obserwacji
        local_weights = {}
        n_explain = min(10, len(X_test))

        for idx in range(n_explain):
            exp_local = explainer.explain_instance(
                X_test.iloc[idx].values, rf_model.predict_proba,
                num_features=len(X_test.columns), num_samples=500
            )
            for feat, w in exp_local.as_list():
                feat_clean = feat.split(' ')[0] if ' ' in feat else feat
                local_weights.setdefault(feat_clean, []).append(abs(w))

        local_avg = {k: __import__('numpy').mean(v) for k, v in local_weights.items()}
        local_df = pd.DataFrame({'Feature': list(local_avg.keys()),
                                  'Local_avg': list(local_avg.values())}).sort_values('Local_avg', ascending=False)

        print("Globalna ważność (Random Forest):")
        print(feat_imp.head(5).to_string(index=False))

        print("\nŚrednia lokalna ważność (LIME):")
        print(local_df.head(5).to_string(index=False))
    return (
        exp_local,
        feat_clean,
        idx,
        local_avg,
        local_df,
        local_weights,
        n_explain,
        w,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **2. Porównaj wyjaśnienia LIME dla podobnych obserwacji. Czy są różnice
    w interpretacji? Co może być ich przyczyną?**

    **3. Wybierz kilka przypadków o różnych poziomach pewności predykcji i
    porównaj ich wyjaśnienia LIME. Jakie wzorce można zaobserwować?**
    """)
    return


if __name__ == "__main__":
    app.run()
