import marimo

__generated_with = "0.23.6"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Lab 11: Optymalizacja hiperparametrów i AutoML — Optuna i FLAML

    Dotychczas hiperparametry tunowaliście przeszukiwaniem siatki (`GridSearchCV`). Działa to dla 2-3 parametrów i kilku wartości, ale rośnie wykładniczo wraz z wymiarem siatki i nie wykorzystuje informacji o tym, które rejony przestrzeni są obiecujące, żeby tam próbkować gęściej.

    W tym laboratorium poznamy dwa narzędzia:

    - **Optuna** to framework do inteligentnej optymalizacji hiperparametrów. Sugeruje wartości na podstawie wcześniejszych prób (np. wykorzystując TPE, CMA-ES), potrafi przerywać nieobiecujące próby i wizualizować przeszukiwaną przestrzeń.
    - **FLAML** (od Microsoftu) sam wybiera model spośród kilku rodzin (LightGBM, XGBoost, Random Forest itd.) i równocześnie dobiera mu hiperparametry w ramach zadanego budżetu czasowego.

    Strona Optuny: https://optuna.org
    Strona FLAML: https://microsoft.github.io/FLAML/
    """)
    return


@app.cell
def _():
    import time
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)

    import numpy as np
    import pandas as pd
    import plotly.express as px
    import plotly.graph_objects as go
    import matplotlib.pyplot as plt

    from sklearn.datasets import fetch_openml
    from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
    from sklearn.metrics import mean_squared_error, r2_score

    import lightgbm as lgb
    import optuna
    from optuna.samplers import TPESampler, RandomSampler, CmaEsSampler
    from optuna.pruners import MedianPruner
    from optuna.integration import LightGBMPruningCallback

    from flaml import AutoML

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    return (
        AutoML,
        CmaEsSampler,
        GridSearchCV,
        LightGBMPruningCallback,
        MedianPruner,
        RandomSampler,
        TPESampler,
        cross_val_score,
        fetch_openml,
        go,
        lgb,
        mean_squared_error,
        np,
        optuna,
        pd,
        px,
        r2_score,
        time,
        train_test_split,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Dane: Bike Sharing Demand (UCI)

    Godzinowy zbiór wypożyczeń rowerów miejskich z lat 2011-2012. Zadanie regresyjne: przewidzieć liczbę wypożyczeń w danej godzinie na podstawie cech kalendarzowych/czasowych (pora roku, godzina, dzień tygodnia, czy święto) i pogodowych (temperatura, wilgotność, wiatr).

    Zbiór jest większy niż wine z poprzedniego laboratorium (~17 tys. obserwacji), zawiera mieszankę cech numerycznych i kategorycznych, więc powinien być ciekawszym celem dla bardziej zaawansowanych narzędzi.
    """)
    return


@app.cell
def _(fetch_openml):
    bike = fetch_openml("Bike_Sharing_Demand", version=2, as_frame=True)
    X_bike = bike.data.copy()
    y_bike = bike.target

    print(f"Rozmiar zbioru: {X_bike.shape}")
    print(f"\nTypy zmiennych:")
    print(X_bike.dtypes)
    print(f"\nRozkład celu (count):")
    print(y_bike.describe().round(1))
    return X_bike, y_bike


@app.cell
def _(px, y_bike):
    fig_target = px.histogram(
        y_bike, nbins=50,
        title="Rozkład liczby wypożyczeń w godzinie",
        labels={"value": "count", "count": "Liczba obserwacji"},
        color_discrete_sequence=["steelblue"],
    )
    fig_target.show()
    return


@app.cell
def _(X_bike, px, y_bike):
    fig_hour = px.box(
        x=X_bike["hour"], y=y_bike,
        title="Wypożyczenia względem godziny doby",
        labels={"x": "Godzina", "y": "count"},
    )
    fig_hour.show()
    return


@app.cell
def _(X_bike, train_test_split, y_bike):
    X_train, X_test, y_train, y_test = train_test_split(
        X_bike, y_bike, test_size=0.2, random_state=42
    )
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")
    return X_test, X_train, y_test, y_train


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Baseline: LightGBM bez tuningu i z grid search

    Najpierw model bazowy z domyślnymi parametrami. Potem klasyczny grid search po małej siatce 3³ = 27 kombinacji.
    """)
    return


@app.cell
def _(X_test, X_train, lgb, mean_squared_error, np, r2_score, y_test, y_train):
    base_model = lgb.LGBMRegressor(random_state=42, verbose=-1)
    base_model.fit(X_train, y_train)
    base_pred = base_model.predict(X_test)
    base_rmse = np.sqrt(mean_squared_error(y_test, base_pred))
    base_r2 = r2_score(y_test, base_pred)

    print(f"LightGBM (domyślne): RMSE = {base_rmse:.2f}, R² = {base_r2:.3f}")
    return


@app.cell
def _(GridSearchCV, X_train, lgb, time, y_train):
    grid_params = {
        "num_leaves": [15, 31, 63],
        "learning_rate": [0.01, 0.05, 0.1],
        "n_estimators": [100, 300, 500],
    }

    _t0 = time.perf_counter()
    grid = GridSearchCV(
        lgb.LGBMRegressor(random_state=42, verbose=-1),
        grid_params, cv=3, scoring="neg_root_mean_squared_error", n_jobs=-1,
    )
    grid.fit(X_train, y_train)
    t_grid = time.perf_counter() - _t0

    print(f"Grid search: {len(grid.cv_results_['params'])} kombinacji, {t_grid:.1f}s")
    print(f"Najlepsze parametry: {grid.best_params_}")
    print(f"Najlepszy RMSE (CV): {-grid.best_score_:.2f}")
    return (grid,)


@app.cell
def _(X_test, grid, mean_squared_error, np, r2_score, y_test):
    grid_pred = grid.best_estimator_.predict(X_test)
    grid_rmse = np.sqrt(mean_squared_error(y_test, grid_pred))
    grid_r2 = r2_score(y_test, grid_pred)
    print(f"Grid search na teście: RMSE = {grid_rmse:.2f}, R² = {grid_r2:.3f}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Optuna

    Trzy podstawowe pojęcia:

    - **study** to obiekt reprezentujący całe doświadczenie optymalizacyjne (sekwencję prób, najlepszy wynik, historię).
    - **trial** to pojedyncza próba z konkretnymi wartościami hiperparametrów.
    - **objective(trial)** to funkcja, którą definiujemy. Dostaje obiekt `trial`, używa go do "wylosowania" hiperparametrów (`trial.suggest_int`, `trial.suggest_float`) i zwraca wartość, którą Optuna ma minimalizować lub maksymalizować.

    Kluczowa różnica względem grid search: Optuna nie odwiedza punktów ze sztywnej siatki, tylko sama proponuje kolejne wartości na podstawie wyników poprzednich prób. Domyślnym algorytmem jest **TPE** (Tree-structured Parzen Estimator), który modeluje rozkład "dobrych" i "złych" konfiguracji i preferuje obiecujące rejony.
    """)
    return


@app.cell
def _(TPESampler, X_train, cross_val_score, lgb, optuna, time, y_train):
    def objective(trial):
        params = {
            "num_leaves": trial.suggest_int("num_leaves", 8, 128),
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.2, log=True),
            "n_estimators": trial.suggest_int("n_estimators", 100, 800),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        }
        model = lgb.LGBMRegressor(random_state=42, verbose=-1, **params)
        scores = cross_val_score(
            model, X_train, y_train, cv=3,
            scoring="neg_root_mean_squared_error", n_jobs=-1,
        )
        return -scores.mean()

    study = optuna.create_study(
        direction="minimize",
        sampler=TPESampler(seed=42),
        study_name="lgbm_bike_tpe",
    )

    _t0 = time.perf_counter()
    study.optimize(objective, n_trials=30, show_progress_bar=False)
    t_optuna = time.perf_counter() - _t0

    print(f"Optuna (TPE, 30 prób): {t_optuna:.1f}s")
    print(f"Najlepszy RMSE (CV): {study.best_value:.2f}")
    print(f"Najlepsze parametry:")
    for _k, _v in study.best_params.items():
        print(f"  {_k}: {_v}")
    return (study,)


@app.cell
def _(
    X_test,
    X_train,
    lgb,
    mean_squared_error,
    np,
    r2_score,
    study,
    y_test,
    y_train,
):
    best_optuna_model = lgb.LGBMRegressor(
        random_state=42, verbose=-1, **study.best_params
    )
    best_optuna_model.fit(X_train, y_train)
    optuna_pred = best_optuna_model.predict(X_test)
    optuna_rmse = np.sqrt(mean_squared_error(y_test, optuna_pred))
    optuna_r2 = r2_score(y_test, optuna_pred)

    print(f"Optuna na teście: RMSE = {optuna_rmse:.2f}, R² = {optuna_r2:.3f}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Wizualizacje przeszukiwania

    Optuna ma wbudowany moduł `optuna.visualization` (wykresy plotly), który pozwala zrozumieć przebieg optymalizacji bez pisania dodatkowego kodu. Trzy najprzydatniejsze wykresy:

    - **Historia optymalizacji** pokazuje wartość celu w kolejnych próbach oraz najlepszy dotychczasowy wynik.
    - **Ważność hiperparametrów** szacuje, które z nich najbardziej wpływały na wartość celu (analogicznie do feature importance w drzewach).
    - **Współrzędne równoległe** to interaktywny wykres ukazujący, w jakich rejonach przestrzeni leżały dobre próby.
    """)
    return


@app.cell
def _(optuna, study):
    fig_hist = optuna.visualization.plot_optimization_history(study)
    fig_hist.update_layout(title="Historia optymalizacji (TPE)")
    fig_hist.show()
    return


@app.cell
def _(optuna, study):
    fig_imp = optuna.visualization.plot_param_importances(study)
    fig_imp.update_layout(title="Ważność hiperparametrów")
    fig_imp.show()
    return


@app.cell
def _(optuna, study):
    fig_par = optuna.visualization.plot_parallel_coordinate(study)
    fig_par.update_layout(title="Współrzędne równoległe")
    fig_par.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Pruning, czyli przerywanie nieobiecujących prób

    Dla modeli iteracyjnych (boosting, sieci neuronowe) można obserwować metrykę walidacyjną w trakcie trenowania i przerwać próbę wcześnie, jeśli wyniki są wyraźnie gorsze od dotychczasowej mediany. Optuna realizuje to przez **pruner** oraz integrację z konkretną biblioteką (tu `LightGBMPruningCallback`).

    Efekt: przy tym samym budżecie prób część z nich zostanie zakończona po kilkudziesięciu iteracjach zamiast po pełnym treningu, co zostawia więcej czasu na obiecujące konfiguracje.
    """)
    return


@app.cell
def _(
    LightGBMPruningCallback,
    MedianPruner,
    TPESampler,
    X_train,
    lgb,
    optuna,
    time,
    train_test_split,
    y_train,
):
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42
    )

    def objective_pruned(trial):
        params = {
            "num_leaves": trial.suggest_int("num_leaves", 8, 128),
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.2, log=True),
            "n_estimators": 800,
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "random_state": 42,
            "verbose": -1,
        }
        model = lgb.LGBMRegressor(**params)
        callback = LightGBMPruningCallback(trial, metric="rmse")
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            eval_metric="rmse",
            callbacks=[callback, lgb.early_stopping(50, verbose=False)],
        )
        return model.best_score_["valid_0"]["rmse"]

    study_pruned = optuna.create_study(
        direction="minimize",
        sampler=TPESampler(seed=42),
        pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=30),
    )

    _t0 = time.perf_counter()
    study_pruned.optimize(objective_pruned, n_trials=30, show_progress_bar=False)
    t_pruned = time.perf_counter() - _t0

    n_pruned = sum(t.state.name == "PRUNED" for t in study_pruned.trials)
    n_complete = sum(t.state.name == "COMPLETE" for t in study_pruned.trials)

    print(f"Optuna z pruningiem (30 prób): {t_pruned:.1f}s")
    print(f"  ukończone: {n_complete}, przerwane: {n_pruned}")
    print(f"  najlepszy RMSE (walidacja): {study_pruned.best_value:.2f}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Porównanie algorytmów próbkowania

    Domyślnie Optuna używa TPE, ale dostępne są też inne algorytmy. Np.:

    - **RandomSampler** to losowe próbkowanie z przestrzeni. Niby baseline, ale przy bardzo małej ilości prób ciężej wymyślić coś mądrzejszego.
    - **TPESampler** to modelowanie probabilistyczne (Tree-structured Parzen Estimator). Domyślny wybór, dobrze radzi sobie z mieszanką typów hiperparametrów.
    - **CmaEsSampler** (Covariance Matrix Adaptation Evolution Strategy) to algorytm ewolucyjny dla ciągłych zmiennych. Świetnie sprawdza się przy dużych budżetach, słabiej w pierwszych próbach.

    Uruchamiamy je na tej samej, lekkiej funkcji celu (mniejsza siatka, krótszy CV), żeby porównanie zmieściło się w rozsądnym czasie.
    """)
    return


@app.cell
def _(
    CmaEsSampler,
    RandomSampler,
    TPESampler,
    X_train,
    cross_val_score,
    go,
    lgb,
    optuna,
    pd,
    y_train,
):
    def light_objective(trial):
        params = {
            "num_leaves": trial.suggest_int("num_leaves", 8, 128),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "n_estimators": 200,
            "random_state": 42,
            "verbose": -1,
        }
        scores = cross_val_score(
            lgb.LGBMRegressor(**params), X_train, y_train, cv=3,
            scoring="neg_root_mean_squared_error", n_jobs=-1,
        )
        return -scores.mean()

    samplers = {
        "Random": RandomSampler(seed=42),
        "TPE": TPESampler(seed=42),
        "CMA-ES": CmaEsSampler(seed=42),
    }

    histories = {}
    for name, sampler in samplers.items():
        s = optuna.create_study(direction="minimize", sampler=sampler)
        s.optimize(light_objective, n_trials=25, show_progress_bar=False)
        histories[name] = [t.value for t in s.trials]

    hist_df = pd.DataFrame(histories)
    hist_df["trial"] = hist_df.index + 1
    best_so_far = hist_df[list(samplers.keys())].cummin()

    fig_samp = go.Figure()
    for name in samplers:
        fig_samp.add_scatter(
            x=hist_df["trial"], y=best_so_far[name],
            mode="lines+markers", name=name,
        )
    fig_samp.update_layout(
        title="Najlepszy RMSE w funkcji numeru próby",
        xaxis_title="Numer próby", yaxis_title="Najlepszy RMSE (CV)",
    )
    fig_samp.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Ćwiczenie 1

    Zmodyfikuj funkcję `objective` tak, by zamiast 3-krotnej walidacji krzyżowej korzystała z większej ilości złożeń (5-krotna albo 10-krotna). Uruchom 20 prób i sprawdź, czy najlepszy RMSE wyznaczony przez Optunę różni się istotnie od poprzedniego. Czy wzrost stabilności estymacji rekompensuje wzrost czasu obliczeń?
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
    ## FLAML: automatyzacja o krok dalej

    Optuna optymalizuje hiperparametry, ale model trzeba wybrać samodzielnie. FLAML łączy oba problemy: dostaje budżet czasowy, listę dopuszczalnych rodzin modeli i sam decyduje, w które warto inwestować czas. Wewnętrznie używa własnego algorytmu (CFO/BlendSearch), który przesuwa się po przestrzeni hiperparametrów uwzględniając koszt obliczeniowy każdej próby.

    Z punktu widzenia użytkownika interfejs jest minimalny: `AutoML().fit(X, y, task=..., time_budget=...)`. Reszta dzieje się pod spodem.
    """)
    return


@app.cell
def _(
    AutoML,
    X_test,
    X_train,
    mean_squared_error,
    np,
    r2_score,
    time,
    y_test,
    y_train,
):
    automl = AutoML()
    settings = {
        "time_budget": 60,
        "task": "regression",
        "metric": "rmse",
        "estimator_list": ["lgbm", "xgboost", "rf", "extra_tree"],
        "seed": 42,
        "verbose": 0,
    }

    _t0 = time.perf_counter()
    automl.fit(X_train, y_train, **settings)
    t_flaml = time.perf_counter() - _t0

    flaml_pred = automl.predict(X_test)
    flaml_rmse = np.sqrt(mean_squared_error(y_test, flaml_pred))
    flaml_r2 = r2_score(y_test, flaml_pred)

    print(f"FLAML (budżet 60s, rzeczywisty czas {t_flaml:.1f}s)")
    print(f"  Wybrany model: {automl.best_estimator}")
    print(f"  RMSE na teście: {flaml_rmse:.2f}")
    print(f"  R² na teście: {flaml_r2:.3f}")
    return (automl,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Co FLAML faktycznie zrobił?

    Wewnątrz obiektu `AutoML` znajdziemy najlepsze konfiguracje dla każdej rodziny modeli i loss osiągnięty w trakcie poszukiwania. Pozwala to ocenić, czy budżet czasowy był wystarczający (czy modele konkurowały o czas) i które rodziny w ogóle miały szansę pokazać swój potencjał.
    """)
    return


@app.cell
def _(automl, go, pd):
    per_est = pd.DataFrame({
        "estimator": list(automl.best_loss_per_estimator.keys()),
        "best_rmse": list(automl.best_loss_per_estimator.values()),
    }).sort_values("best_rmse")

    print("Najlepszy RMSE per rodzina modeli:")
    print(per_est.to_string(index=False))

    fig_flaml = go.Figure()
    fig_flaml.add_bar(x=per_est["estimator"], y=per_est["best_rmse"])
    fig_flaml.update_layout(
        title="FLAML: najlepszy RMSE per rodzina modeli",
        xaxis_title="Rodzina modelu", yaxis_title="Najlepszy RMSE",
    )
    fig_flaml.show()

    print(f"\nNajlepsza konfiguracja ({automl.best_estimator}):")
    for _k, _v in automl.best_config.items():
        print(f"  {_k}: {_v}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Zadania

    ## Zadanie 1 — Optuna z niestandardową metryką

    Zdefiniuj nową funkcję celu dla Optuny, w której zamiast RMSE używasz **RMSLE** (root mean squared logarithmic error).

    Uruchom 30 prób z TPE i porównaj rozkład predykcji najlepszego modelu z modelem optymalizowanym pod RMSE. Czy zmiana metryki wpłynęła na to, jakie wartości hiperparametrów są preferowane?
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
    ## Zadanie 2 — porównanie czasowe FLAML vs Optuna

    Uruchom Optunę z `n_trials=50` i `TPESampler` albo `CMA-ES` zmierz całkowity czas. Następnie uruchom FLAML z tym samym budżetem czasowym (`time_budget` = czas Optuny w sekundach) i `estimator_list=["lgbm"]`. Porównaj końcowy RMSE testowy obu podejść.

    Który framework lepiej wykorzystał ten sam budżet czasowy w sytuacji, gdy obaj mają dostęp wyłącznie do LightGBM? Co się zmieni, jeśli FLAML pozwolisz wybierać z pełnej listy modeli?
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
    ## Zadanie 3

    Wykorzystaj Optunę i FLAML na danych z opendota, które wykorzystujemy w projekcie. Sprawdź, jak ma się nowy wynik względem Twojego poprzedniego na leaderboardzie.
    """)
    return


if __name__ == "__main__":
    app.run()
