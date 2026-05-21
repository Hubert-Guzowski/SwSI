import marimo

__generated_with = "0.19.9"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Lab 10: Interpretowalność modeli — LIME i SHAP

    ## LIME — Local Interpretable Model-Agnostic Explanations

    LIME to technika tworzenia **lokalnych** wyjaśnień dla pojedynczych predykcji. Metoda działa poprzez:

    1. Generowanie perturbacji wokół badanej obserwacji.
    2. Ocenianie modelu na tych perturbacjach z wagami proporcjonalnymi do bliskości punktu wyjaśnianego.
    3. Trenowanie prostego modelu interpretowalnego (regresji liniowej) na ważonych ocenach.
    4. Użycie wag tego modelu jako wyjaśnienia lokalnego zachowania złożonego modelu.

    **Ważne ograniczenia:**
    - Niestabilność wyjaśnień przy małej liczbie próbek.
    - Brak gwarancji globalnej spójności.
    - Wrażliwość na parametry perturbacji.

    Artykuł: https://arxiv.org/pdf/1602.04938

    ## SHAP — SHapley Additive exPlanations

    Wartości SHAP to metoda oparta o teorię gier stosowana do wyjaśniania predykcji modeli uczenia maszynowego. Pozwala zrozumieć, jak każda zmienna wpływa na konkretną predykcję.

    W teorii gier mamy:
    - **Graczy** (w ML: cechy/zmienne)
    - **Koalicje** (w ML: podzbiory cech)
    - **Funkcję wypłaty** (w ML: predykcję modelu)

    Wartość SHAP dla cechy $i$:

    $$
    \phi_i = \sum_{S \subseteq F \setminus \{i\}} \frac{|S|!\,(M-|S|-1)!}{M!}\,\bigl[f(S \cup \{i\}) - f(S)\bigr]
    $$

    SHAP jest matematycznie uzasadnione i sprawiedliwie rozdziela wartość predykcji między cechy. Główne ograniczenia: wymagające obliczeniowo (dla drzew TreeExplainer daje dokładne wartości w czasie wielomianowym) i nie wyjaśnia automatycznie korelacji między cechami.
          
    Artykuł: https://arxiv.org/pdf/1705.07874
    Krytyka LIME i SHAP: https://arxiv.org/pdf/1806.08049
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

    Używamy połączonego zbioru czerwonych i białych win z repozytorium UCI. Zadaniem jest predykcja jakości wina (skala 3–9) na podstawie 12 cech fizykochemicznych (kwasowość, cukier resztkowy, alkohol itp.). Kolumna `white` (1 = białe, 0 = czerwone) koduje kolor wina.
    """)
    return


@app.cell
def _(pd):
    url_white = "http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv"
    url_red = "http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"

    _white = pd.read_csv(url_white, sep=";")
    _red = pd.read_csv(url_red, sep=";")
    _white["white"] = 1
    _red["white"] = 0
    wine_data = pd.concat([_red, _white], ignore_index=True)

    print(f"Rozmiar zbioru: {wine_data.shape}")
    print("\nRozkład zmiennej docelowej (jakość):")
    print(wine_data["quality"].value_counts().sort_index())
    return (wine_data,)


@app.cell
def _(px, wine_data):
    fig_hist = px.histogram(
        wine_data, x="quality", nbins=7,
        title="Rozkład jakości wina",
        labels={"quality": "Jakość", "count": "Liczba obserwacji"},
        color_discrete_sequence=["steelblue"],
    )
    fig_hist.show()
    return


@app.cell
def _(px, wine_data):
    fig_corr = px.imshow(
        wine_data.corr().round(2), text_auto=True,
        color_continuous_scale="RdBu_r",
        title="Macierz korelacji — Wine Quality",
        zmin=-1, zmax=1,
    )
    fig_corr.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Model bazowy: CatBoost

    Trenujemy `CatBoostRegressor` do predykcji jakości wina. Ten sam model posłuży do demonstracji obu technik wyjaśniania (LIME i SHAP), co pozwoli bezpośrednio porównać wyniki.
    """)
    return


@app.cell
def _(train_test_split, wine_data):
    X_wine = wine_data.drop(columns=["quality"])
    y_wine = wine_data["quality"]

    X_train, X_test, y_train, y_test = train_test_split(
        X_wine, y_wine, test_size=0.2, random_state=42
    )
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")
    return X_test, X_train, X_wine, y_test, y_train, y_wine


@app.cell
def _(X_test, X_train, mean_squared_error, np, r2_score, y_test, y_train):
    from catboost import CatBoostRegressor

    cb_model = CatBoostRegressor(iterations=300, verbose=0, random_state=42)
    cb_model.fit(X_train, y_train)

    y_pred = cb_model.predict(X_test)
    print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.3f}")
    print(f"R²:   {r2_score(y_test, y_pred):.3f}")
    return CatBoostRegressor, cb_model, y_pred


@app.cell
def _(X_train, cb_model, go, pd):
    feat_imp = pd.DataFrame({
        "Feature": X_train.columns,
        "Importance": cb_model.get_feature_importance(),
    }).sort_values("Importance", ascending=False)

    fig_imp = go.Figure()
    fig_imp.add_bar(x=feat_imp["Feature"], y=feat_imp["Importance"])
    fig_imp.update_layout(
        title="CatBoost — globalna ważność cech",
        xaxis_title="Cecha", yaxis_title="Ważność",
    )
    fig_imp.show()
    return (feat_imp,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## LIME — wyjaśnienia lokalne

    Tworzymy explainer LIME i wyjaśniamy predykcję dla pierwszej obserwacji testowej. Każda wartość na wykresie to wkład w **lokalną** predykcję, w tych samych jednostkach co zmienna docelowa (skala jakości 3–9).

    Cechy mają formę zdyskretyzowanych warunków (np. `alcohol > 10.5`), bo LIME domyślnie dyskretyzuje zmienne ciągłe przed budową lokalnego modelu liniowego.
    """)
    return


@app.cell
def _(X_test, X_train, cb_model, y_pred, y_test):
    import lime.lime_tabular

    lime_explainer = lime.lime_tabular.LimeTabularExplainer(
        X_train.values,
        feature_names=list(X_train.columns),
        mode="regression",
        discretize_continuous=True,
        random_state=42,
    )

    instance_idx = 0
    instance = X_test.iloc[instance_idx].values

    lime_exp = lime_explainer.explain_instance(
        instance,
        cb_model.predict,
        num_features=len(X_train.columns),
        num_samples=1000,
    )

    print(f"Obserwacja {instance_idx}:")
    print(f"  Rzeczywista jakość:  {y_test.iloc[instance_idx]}")
    print(f"  Predykcja CatBoost: {y_pred[instance_idx]:.2f}")
    print(f"\nWyjaśnienie LIME (posortowane wg |wagi|):")
    for _feat, _w in sorted(lime_exp.as_list(), key=lambda x: abs(x[1]), reverse=True):
        print(f"  {_feat}: {_w:+.4f}")
    return instance, instance_idx, lime_exp, lime_explainer


@app.cell
def _(go, lime_exp):
    _sorted = sorted(lime_exp.as_list(), key=lambda x: x[1])
    _features = [f for f, _ in _sorted]
    _weights = [w for _, w in _sorted]
    _colors = ["crimson" if w < 0 else "steelblue" for w in _weights]

    fig_lime = go.Figure()
    fig_lime.add_bar(
        x=_weights, y=_features, orientation="h",
        marker_color=_colors,
    )
    fig_lime.add_vline(x=0, line_width=1, line_color="black")
    fig_lime.update_layout(
        title="LIME — wkład cech w predykcję (obserwacja 0)",
        xaxis_title="Wkład w predykcję jakości",
        height=500,
    )
    fig_lime.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Stabilność wyjaśnień LIME

    LIME losuje perturbacje, więc wyjaśnienia mogą się różnić między wywołaniami.
    Sprawdzamy, jak wagi zmieniają się przy różnej liczbie próbek perturbacyjnych.
    """)
    return


@app.cell
def _(cb_model, instance, lime_explainer, pd):
    _sample_sizes = [100, 500, 1000, 3000]
    _stability = {}

    for _n in _sample_sizes:
        _exp = lime_explainer.explain_instance(
            instance, cb_model.predict,
            num_features=6, num_samples=_n,
        )
        _stability[_n] = {feat: w for feat, w in _exp.as_list()}

    stab_df = pd.DataFrame(_stability).T
    stab_df.index.name = "num_samples"
    print("Wagi LIME dla 6 cech przy różnych num_samples:")
    print(stab_df.round(3).to_string())
    return (stab_df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Ćwiczenie 1

    **a)** Wygeneruj wyjaśnienia LIME dla obserwacji z najwyższą i najniższą predykcją jakości w zbiorze testowym. Które cechy podnoszą, a które obniżają przewidywaną ocenę?

    **b)** Dla jednej wybranej obserwacji uruchom LIME 5 razy z `num_samples=200` (bez `random_state` w `explain_instance`). Czy wyjaśnienia są powtarzalne? Od jakiej wartości `num_samples` wagi przestają istotnie fluktuować?
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
    ## SHAP — wyjaśnienia lokalne i globalne

    Własność addytywności $\sum_i \phi_i = f(x) - E[f(X)]$ sprawia, że SHAP nadaje się zarówno do wyjaśniania pojedynczych predykcji (wykresy kaskadowe), jak i do opisu globalnego zachowania modelu (wykresy roju). Używamy `shap.Explainer`, który dla modeli drzewiastych automatycznie wybiera dokładny `TreeExplainer`.
    """)
    return


@app.cell
def _(X_test, cb_model, plt):
    import shap

    shap_explainer = shap.Explainer(cb_model)
    shap_values = shap_explainer(X_test)

    plt.figure(figsize=(10, 7))
    shap.plots.beeswarm(shap_values, show=False)
    plt.title("SHAP – wykres roju – Wine Quality (CatBoost)")
    plt.tight_layout()
    plt.subplots_adjust(left=0.25)
    plt.show()
    return shap, shap_explainer, shap_values


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Każda kropka to jedna obserwacja testowa. Oś pozioma: wartość SHAP (wkład w predykcję). Kolor: wartość cechy (czerwony = wysoka, niebieski = niska). Cechy posortowane malejąco według $\overline{|\phi_i|}$ na zbiorze testowym.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Wykres kaskadowy (Waterfall) — dekompozycja pojedynczej predykcji

    Wykres kaskadowy pokazuje, jak model dochodzi od wartości bazowej $E[f(X)]$ do predykcji dla konkretnej obserwacji: każda cecha przesuwa wartość w górę (czerwony) lub w dół (niebieski). Porównaj z wyjaśnieniem LIME powyżej — oba dotyczą tej samej obserwacji (indeks 0).
    """)
    return


@app.cell
def _(instance_idx, plt, shap, shap_values, y_pred, y_test):
    plt.figure(figsize=(10, 5))
    shap.plots.waterfall(shap_values[instance_idx], show=False)
    plt.tight_layout()
    plt.subplots_adjust(left=0.25)
    plt.show()

    print(f"Obserwacja {instance_idx}:")
    print(f"  Rzeczywista jakość:  {y_test.iloc[instance_idx]}")
    print(f"  Predykcja CatBoost: {y_pred[instance_idx]:.2f}")
    print(f"  Wartość bazowa SHAP: {shap_values[instance_idx].base_values:.2f}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Wykres rozrzutu — zależność wartości SHAP od wartości cechy

    Wykres rozrzutu dla cechy `alcohol` pokazuje jednocześnie:
    - jak wartość alkoholu wpływa na predykcję (kształt krzywej),
    - jak ta zależność zmienia się w połączeniu z inną cechą (kolor).
    """)
    return


@app.cell
def _(plt, shap, shap_values):
    plt.figure(figsize=(10, 4))
    shap.plots.scatter(
        shap_values[:, "alcohol"],
        color=shap_values[:, "volatile acidity"],
        show=False,
    )
    plt.title("SHAP – alkohol (kolor: kwasowość lotna)")
    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Ćwiczenie 2

    **a)** Co mówi nam kolor na powyższym wykresie o związku między alkoholem a kwasowością lotną w kontekście predykcji jakości?

    **b)** Wygeneruj analogiczny wykres rozrzutu dla cechy `sulphates` z kolorem według `pH`. Zinterpretuj wykres.

    **c)** Znajdź obserwację testową z predykcją powyżej 7 i narysuj jej wykres kaskadowy. Które cechy odpowiadają za tak wysoką ocenę?
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
    ## Porównanie LIME i SHAP

    Obie metody wyjaśniają tę samą predykcję, ale bardzo różnią się sposobem działania:

    | | LIME | SHAP |
    |---|---|---|
    | **Podejście** | Lokalna aproksymacja przez model liniowy | Rozkład predykcji wg teorii gier |
    | **Cechy** | Zdyskretyzowane warunki (np. `alcohol > 10.5`) | Ciągłe wartości cech |
    | **Gwarancje** | Brak globalnej spójności | Addytywność, sprawiedliwość Shapleya |
    | **Szybkość** | Szybki (próbkowanie) | Wolniejszy; dla drzew: dokładny i szybki |
    | **Kiedy używać** | Szybka inspekcja, niezależność od modelu | Rzetelna analiza, modele drzewiaste |

    Wróć do wykresu LIME i wykresu kaskadowego SHAP dla obserwacji 0 i porównaj je. Czy te same cechy dominują w obu wyjaśnieniach? Gdzie pojawiają się różnice i co może je powodować?

    ---

    # Zadania

    ## Zadanie 1 — LIME dla przypadków skrajnych

    Wydziel ze zbioru testowego 10 obserwacji z **najwyższą** i 10 z **najniższą** predykcją jakości. Dla każdej grupy wygeneruj wyjaśnienia LIME i oblicz **średni wkład** każdej cechy (uśredniony absolutnie po 10 obserwacjach w grupie, ze znakiem dla kierunku). Narysuj oba rankingi na jednym pogrupowanym wykresie słupkowym.

    Które cechy konsekwentnie separują wina o wysokiej i niskiej predykcji?

    *Wskazówka:* `np.argsort(y_pred)` daje indeksy posortowane rosnąco; użyj `[-10:]` i `[:10]`. Przy uśrednianiu wag z LIME pamiętaj, że nazwy cech to warunki (np. `alcohol > 10.5`) – możesz uprościć je do nazwy cechy przez `feat.split()[0]`.
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
    ## Zadanie 2 — SHAP i regresja logistyczna: predykcja koloru wina

    Wytrenuj `LogisticRegression` do predykcji koloru wina (biały/czerwony) na podstawie cech fizykochemicznych. Użyj `StandardScaler` i `shap.LinearExplainer` do obliczenia wartości SHAP, a następnie narysuj wykres roju.

    Pomocnicze przygotowanie danych i modelu:
    """)
    return


@app.cell
def _(wine_data):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    X_type = wine_data.drop(columns=["white", "quality"])
    y_type = wine_data["white"]

    scaler_type = StandardScaler()
    X_type_scaled = scaler_type.fit_transform(X_type)

    lr_type = LogisticRegression(max_iter=1000, random_state=42)
    lr_type.fit(X_type_scaled, y_type)

    accuracy_type = (lr_type.predict(X_type_scaled) == y_type).mean()
    print(f"Dokładność predykcji koloru wina: {accuracy_type:.3f}")
    return LogisticRegression, StandardScaler, X_type, X_type_scaled, lr_type, scaler_type, y_type


@app.cell
def _():
    # Uzupełnij kod poniżej:
    # 1. Oblicz wartości SHAP za pomocą shap.LinearExplainer dla lr_type
    # 2. Narysuj wykres roju (shap.plots.beeswarm)
    # 3. Odpowiedz: jakie cechy najlepiej odróżniają wina czerwone od białych?
    # 4. Co się stanie, jeśli do X_type dodasz kolumnę `white`?
    #    Jaki problem pojawia się przy obliczaniu SHAP dla takiego modelu?
    ...
    return


if __name__ == "__main__":
    app.run()
