import marimo

__generated_with = "0.20.4"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Lab 4: Selekcja cech dla modeli liniowych

    Dopasowanie modelu liniowego do danych wymaga wyboru predyktorów. Gdy liczba
    zmiennych jest duża, model dopasowany do wszystkich może **przeuczyć się**
    (overfitting) i słabo generalizować na nowych danych — nawet jeśli wygląda
    dobrze na danych uczących.

    Podczas tego laboratorium poznamy trzy grupy metod radzenia sobie z tym problemem:

    1. **Selekcja podzbiorów** — przeszukiwanie przestrzeni modeli i wybór
       najlepszego podzbioru predyktorów (selekcja krokowa, best subset selection).
    2. **Regularyzacja (shrinkage)** — karanie za duże współczynniki przez
       dodanie członu penalizującego do funkcji straty (Ridge, Lasso, Elastic-Net).
    3. **Redukcja wymiaru** — projekcja predyktorów na mniejszą przestrzeń
       składowych (PCR — regresja na składowych głównych).

    Wszystkie metody dobierają złożoność modelu tak, by zminimalizować błąd
    generalizacji (czyli np. na zbiorze testowym), nie błąd treningowy.
    """)
    return


@app.cell
def _():
    import warnings
    import marimo as mo
    import numpy as np
    np.Inf = np.inf  # nadpisanie dla zgodności z l0bnb
    import pandas as pd
    from functools import partial
    import plotly.express as px
    import plotly.graph_objects as go
    from statsmodels.api import OLS
    import sklearn.model_selection as skm
    import sklearn.linear_model as skl
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.decomposition import PCA
    from sklearn.cross_decomposition import PLSRegression
    from ISLP import load_data
    from ISLP.models import ModelSpec as MS
    from ISLP.models import (Stepwise,
                             sklearn_selected,
                             sklearn_selection_path)
    from l0bnb import fit_path

    return (
        MS,
        OLS,
        PCA,
        Pipeline,
        StandardScaler,
        Stepwise,
        fit_path,
        go,
        load_data,
        mo,
        np,
        pd,
        px,
        skl,
        sklearn_selected,
        sklearn_selection_path,
        skm,
        warnings,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Zbiór danych Hitters

    Zbiór pochodzi z ligi baseballowej MLB i zawiera statystyki 322 graczy (po
    usunięciu obserwacji z brakującą zmienną objaśnianą). Każda obserwacja opisuje
    gracza 19 predyktorami: liczba uderzeń, zdobyte punkty, lata w lidze, itp.
    Zmienną objaśnianą jest **Salary** — zarobki gracza w tysiącach dolarów za
    sezon 1987.
    """)
    return


@app.cell
def _(load_data, np):
    Hitters = load_data('Hitters')
    print(f"Braki w Salary: {np.isnan(Hitters['Salary']).sum()}")
    Hitters = Hitters.dropna()
    print(f"Rozmiar zbioru po usunięciu NA: {Hitters.shape}")
    Hitters.head()
    return (Hitters,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Selekcja krokowa (Forward Selection)

    Selekcja krokowa do przodu zaczyna od pustego modelu i w każdym kroku dodaje
    tę zmienną, która najbardziej poprawia kryterium dopasowania. Zatrzymuje się,
    gdy żadna kolejna zmienna nie przynosi poprawy.

    Używamy `ISLP.models.Stepwise` wraz z `sklearn_selected()`, które opakowuje
    dopasowanie `OLS` w interfejs kompatybilny ze `sklearn`. Domyślnym kryterium
    jest MSE — przy jego minimalizacji selekcja do przodu wybiera wszystkie 19
    zmiennych (model pełny nigdy nie ma wyższego MSE na danych uczących).
    """)
    return


@app.cell
def _(Hitters, MS, OLS, Stepwise, np, sklearn_selected):
    design = MS(Hitters.columns.drop('Salary')).fit(Hitters)
    Y = np.array(Hitters['Salary'])

    strategy = Stepwise.first_peak(design,
                                   direction='forward',
                                   max_terms=len(design.terms))

    hitters_MSE = sklearn_selected(OLS, strategy)
    hitters_MSE.fit(Hitters, Y)
    print("Zmienne wybrane wg MSE (first_peak):")
    print(hitters_MSE.selected_state_)
    return Y, design


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Ścieżka selekcji i walidacja krzyżowa

    `sklearn_selection_path()` przechodzi całą ścieżkę selekcji krokowej i dla
    każdego kroku (liczby zmiennych) ocenia model przez walidację krzyżową.
    Dzięki temu można wybrać liczbę zmiennych minimalizującą błąd generalizacji,
    a nie błąd treningowy.

    Kluczowa zasada: selekcja zmiennych musi odbywać się **wyłącznie na danych
    uczących** w każdym foldzie — w przeciwnym razie ocena za pomocą walidacji krzyżowej (CV) jest optymistycznie
    zaniżona.
    """)
    return


@app.cell
def _(Hitters, OLS, Stepwise, Y, design, go, np, sklearn_selection_path, skm):
    strategy_full = Stepwise.fixed_steps(design,
                                         len(design.terms),
                                         direction='forward')
    full_path = sklearn_selection_path(OLS, strategy_full)
    full_path.fit(Hitters, Y)
    Yhat_in = full_path.predict(Hitters)

    insample_mse = ((Yhat_in - Y[:, None]) ** 2).mean(0)
    n_steps = insample_mse.shape[0]

    K = 5
    kfold = skm.KFold(K, random_state=0, shuffle=True)
    Yhat_cv = skm.cross_val_predict(full_path, Hitters, Y, cv=kfold)

    cv_mse = []
    for _, test_idx in kfold.split(Y):
        errors = (Yhat_cv[test_idx] - Y[test_idx, None]) ** 2
        cv_mse.append(errors.mean(0))
    cv_mse = np.array(cv_mse).T

    steps = np.arange(n_steps)
    fig_path = go.Figure()
    fig_path.add_scatter(x=steps, y=insample_mse, mode='lines', name='Próba ucząca',
                         line=dict(color='black'))
    fig_path.add_scatter(x=steps, y=cv_mse.mean(1), mode='lines+markers', name='CV (5-krotna)',
                         error_y=dict(array=cv_mse.std(1) / np.sqrt(K)), line=dict(color='red'))
    fig_path.update_layout(xaxis_title='Liczba kroków selekcji do przodu', yaxis_title='MSE',
                           title='Ścieżka selekcji krokowej — MSE treningowe vs CV')
    fig_path.show()
    return K, kfold


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Wybór najlepszego podzbioru (Best Subset Selection)

    Selekcja krokowa jest zachłanną heurystyką — nie gwarantuje znalezienia
    globalnie najlepszego podzbioru. Pakiet `l0bnb` szuka optymalnych podzbiorów
    efektywnie, produkując **ścieżkę rozwiązań** parametryzowaną przez karę
    $\lambda_0$ za liczbę niezerowych współczynników.
    """)
    return


@app.cell
def _(Hitters, MS, Y, fit_path, np):
    D = MS(Hitters.columns.drop('Salary')).fit_transform(Hitters)
    D = D.drop('intercept', axis=1)
    X_bs = np.asarray(D)

    bs_path = fit_path(X_bs, Y, max_nonzeros=X_bs.shape[1])

    print("Przykładowe kroki ścieżki best subset:")
    for step in bs_path[:5]:
        n_nonzero = int((step['B'] != 0).sum())
        print(f"  lambda_0={step['lambda_0']:.4f}, niezerowych współczynników: {n_nonzero}")
    return D, X_bs


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Regularyzacja

    Zamiast wybierać podzbiór zmiennych, metody regularyzacji **kurczą**
    współczynniki w kierunku zera, dodając do funkcji straty człon penalizujący
    ich wielkość. Większy parametr regularyzacji $\lambda$ = silniejsze kurczenie.

    Do dopasowania Ridge i Lasso używamy `skl.ElasticNet` z parametrem `l1_ratio`:
    - `l1_ratio=0` → Ridge (penalizacja $\ell_2$, współczynniki kurczą się, ale nie
      zerują)
    - `l1_ratio=1` → Lasso (penalizacja $\ell_1$, współczynniki mogą być dokładnie
      zero — jednoczesna selekcja zmiennych)

    Modele regularyzowane są wrażliwe na skalę predyktorów — standaryzujemy je
    za pomocą `StandardScaler` w pipeline'ie.

    ### Interpretacja geometryczna regularyzacji

    Regularyzacja ma elegancką interpretację geometryczną: minimalizujemy RSS
    przy ograniczeniu, że współczynniki leżą wewnątrz **obszaru ograniczeń**
    wyznaczonego przez normę. Kształt obszaru zależy od metody.

    Konturami w tle są poziomice RSS — elipsy skupione wokół estymatora MNK $\hat\beta$ (czyli najlepiej dopasowanej wartości parametru).
    Kolorowe linie to granice obszarów ograniczeń dla każdej metody; kolorowe
    punkty to odpowiadające im estymaty regularyzowane.

    Suwak $\lambda$ odpowiada za siłę regularyzacji a suwak $l1\_ratio$ za wagi przy zastosowanych metrykach. Komórkę z kodem do wizualizacji poniżej można wykonać i schować - przy zmianie parametrów na suwakach wykres się zaktualizuje.
    """)
    return


@app.cell
def _(mo):
    lambda_slider = mo.ui.slider(0.0, 3.0, value=0.0, step=0.25,
                                 label="λ (siła regularyzacji)")
    l1_ratio_slider = mo.ui.slider(0.0, 1.0, value=0.5, step=0.1,
                                   label="l1_ratio (0 = Ridge, 1 = Lasso)")
    mo.vstack([lambda_slider, l1_ratio_slider])
    return l1_ratio_slider, lambda_slider


@app.cell
def _(go, l1_ratio_slider, lambda_slider, np):
    lam = lambda_slider.value
    alpha = l1_ratio_slider.value
    b1_hat, b2_hat = 2.2, 1.6  # estymata MNK

    def reg_sol(b):
        return np.sign(b) * np.maximum(np.abs(b) - lam * alpha, 0) / (1 + lam * (1 - alpha))

    b_sol = np.array([reg_sol(b1_hat), reg_sol(b2_hat)])

    # Wartość funkcji kary w punkcie regularyzowanym -> rozmiar obszaru ograniczeń
    def pen_val(b):
        return alpha * np.sum(np.abs(b)) + (1 - alpha) / 2 * np.sum(b ** 2)

    # Granica obszaru ograniczeń:
    def ball_boundary(s, n=500):
        angles = np.linspace(0, 2 * np.pi, n)
        c, ss = np.cos(angles), np.sin(angles)
        L1f = np.abs(c) + np.abs(ss)
        if alpha < 1.0:
            a_c = (1 - alpha) / 2
            b_c = alpha * L1f
            disc = np.maximum(b_c ** 2 + 4 * a_c * s, 0)
            r = (-b_c + np.sqrt(disc)) / (2 * a_c)
        else:
            r = s / L1f
        return r * c, r * ss

    s = pen_val(b_sol)
    bx, by = ball_boundary(s) if s > 0 else (np.array([0.0]), np.array([0.0]))

    if alpha == 0.0:
        label = 'Ridge (l1_ratio=0)'
        color = 'royalblue'
    elif alpha == 1.0:
        label = 'Lasso (l1_ratio=1)'
        color = 'crimson'
    else:
        label = f'Elastic-Net (l1_ratio={alpha:.2f})'
        color = 'seagreen'

    # Poziomice RSS
    b1g = np.linspace(-1.2, 3.8, 300)
    b2g = np.linspace(-1.2, 3.8, 300)
    B1, B2 = np.meshgrid(b1g, b2g)
    RSS = (B1 - b1_hat) ** 2 + 0.6 * (B1 - b1_hat) * (B2 - b2_hat) + (B2 - b2_hat) ** 2

    fig_geo = go.Figure()
    fig_geo.add_trace(go.Contour(
        x=b1g, y=b2g, z=RSS,
        contours=dict(start=0.3, end=9.0, size=1.0, coloring='none'),
        line=dict(color='lightgray', width=1),
        showscale=False, hoverinfo='skip', name='RSS'
    ))
    fig_geo.add_scatter(x=bx, y=by, mode='lines', name=label,
                        line=dict(color=color, width=2.5))
    fig_geo.add_scatter(x=[b_sol[0]], y=[b_sol[1]], mode='markers',
                        marker=dict(size=11, color=color, symbol='circle'),
                        showlegend=False,
                        hovertemplate=f'{label}<br>β₁={b_sol[0]:.3f}, β₂={b_sol[1]:.3f}')
    fig_geo.add_scatter(x=[b1_hat], y=[b2_hat], mode='markers', name='MNK (β̂)',
                        marker=dict(size=14, color='black', symbol='star'))
    fig_geo.add_hline(y=0, line_width=0.8, line_color='black')
    fig_geo.add_vline(x=0, line_width=0.8, line_color='black')
    fig_geo.update_layout(
        title=f'Geometria regularyzacji — λ = {lam:.2f}, l1_ratio = {alpha:.2f}',
        xaxis=dict(title='β₁', range=[-1.5, 3.5]),
        yaxis=dict(title='β₂', range=[-1.5, 3.5]),
        width=640, height=640,
        legend=dict(x=0.02, y=0.98),
    )
    fig_geo.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Regresja grzbietowa (Ridge) — ścieżka współczynników
    """)
    return


@app.cell
def _(D, StandardScaler, X_bs, Y, np, pd, px, skl, warnings):
    warnings.simplefilter("ignore")
    scaler = StandardScaler(with_mean=True, with_std=True)
    Xs = scaler.fit_transform(X_bs)

    lambdas = 10 ** np.linspace(8, -2, 100) / Y.std()

    soln_array = skl.ElasticNet.path(Xs, Y, l1_ratio=0., alphas=lambdas)[1]
    soln_path = pd.DataFrame(soln_array.T,
                             columns=D.columns,
                             index=-np.log(lambdas))
    soln_path.index.name = '-log(lambda)'

    fig_ridge = px.line(soln_path, title='Ridge — ścieżka współczynników',
                        labels={'value': 'Standaryzowane współczynniki',
                                'variable': 'Zmienna'})
    fig_ridge.show()
    return lambdas, scaler


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Przy małym $\lambda$ (prawy koniec osi) współczynniki zbliżają się do estymaty
    MNK. Przy dużym $\lambda$ (lewy koniec) wszystkie kurczą się do zera, lecz
    **żaden nie osiąga dokładnie zera** — Ridge nie zeruje współczynników.

    ### Ridge — dobór $\lambda$ przez walidację krzyżową
    """)
    return


@app.cell
def _(Pipeline, X_bs, Y, kfold, lambdas, np, scaler, skl, warnings):
    warnings.simplefilter("ignore")
    ridgeCV = skl.ElasticNetCV(alphas=lambdas, l1_ratio=0, cv=kfold)
    pipeCV_ridge = Pipeline(steps=[('scaler', scaler), ('ridge', ridgeCV)])
    pipeCV_ridge.fit(X_bs, Y)
    tuned_ridge = pipeCV_ridge.named_steps['ridge']

    print(f"Ridge optymalne lambda: {tuned_ridge.alpha_:.4f}")
    print(f"Ridge minimalne CV MSE: {np.min(tuned_ridge.mse_path_.mean(1)):.0f}")
    print(f"Liczba niezerowych współczynników: {(tuned_ridge.coef_ != 0).sum()} / {len(tuned_ridge.coef_)}")
    return (tuned_ridge,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    CV MSE w funkcji $-\log(\lambda)$ — pionowa linia wskazuje optymalne $\lambda$.
    """)
    return


@app.cell
def _(K, go, lambdas, np, tuned_ridge):
    neg_log_lam = -np.log(lambdas)
    fig_ridgeCV = go.Figure()
    fig_ridgeCV.add_scatter(x=neg_log_lam, y=tuned_ridge.mse_path_.mean(1),
                            error_y=dict(array=tuned_ridge.mse_path_.std(1) / np.sqrt(K)),
                            mode='lines', name='CV MSE')
    fig_ridgeCV.add_vline(x=-np.log(tuned_ridge.alpha_), line_dash='dash',
                          annotation_text=f'opt λ={tuned_ridge.alpha_:.4f}')
    fig_ridgeCV.update_layout(xaxis_title='-log(λ)', yaxis_title='CV MSE',
                              title='Ridge — CV MSE vs λ')
    fig_ridgeCV.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Lasso

    Lasso (`l1_ratio=1`) zeruje współczynniki mniej istotnych zmiennych — wykonuje
    jednocześnie regularyzację i selekcję zmiennych. Poniżej dopasowujemy Lasso
    z doborem $\lambda$ przez CV, a następnie wyświetlamy ścieżkę współczynników.
    """)
    return


@app.cell
def _(D, Pipeline, X_bs, Y, kfold, np, pd, scaler, skl):
    lassoCV = skl.ElasticNetCV(n_alphas=100, l1_ratio=1, cv=kfold)
    pipeCV_lasso = Pipeline(steps=[('scaler', scaler), ('lasso', lassoCV)])
    pipeCV_lasso.fit(X_bs, Y)
    tuned_lasso = pipeCV_lasso.named_steps['lasso']

    print(f"Lasso optymalne lambda: {tuned_lasso.alpha_:.4f}")
    print(f"Lasso minimalne CV MSE: {np.min(tuned_lasso.mse_path_.mean(1)):.0f}")
    print(f"Liczba niezerowych współczynników: {(tuned_lasso.coef_ != 0).sum()} / {len(tuned_lasso.coef_)}")

    lasso_lambdas, lasso_soln = skl.Lasso.path(scaler.transform(X_bs), Y, l1_ratio=1, n_alphas=100)[:2]
    lasso_path_df = pd.DataFrame(lasso_soln.T, columns=D.columns, index=-np.log(lasso_lambdas))
    lasso_path_df.index.name = '-log(lambda)'
    return (lasso_path_df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Ścieżka współczynników Lasso — przy rosnącej regularyzacji (malejące
    $-\log(\lambda)$, lewy koniec) kolejne współczynniki zerują się dokładnie.
    """)
    return


@app.cell
def _(lasso_path_df, px):
    fig_lasso = px.line(lasso_path_df, title='Lasso — ścieżka współczynników',
                        labels={'value': 'Standaryzowane współczynniki',
                                'variable': 'Zmienna'})
    fig_lasso.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Regresja składowych głównych (PCR)

    PCR redukuje przestrzeń predyktorów do $M$ **składowych głównych** —
    kombinacji liniowych predyktorów o maksymalnej wariancji — a następnie
    dopasowuje regresję liniową w tej zmniejszonej przestrzeni. Zamiast selekcji
    zmiennych mamy redukcję wymiaru: wszystkie predyktory uczestniczą w modelu,
    lecz przez pryzmat ich głównych kierunków zmienności.

    Liczbę składowych $M$ dobieramy przez CV.
    """)
    return


@app.cell
def _(PCA, Pipeline, StandardScaler, X_bs, Y, kfold, skl, skm):
    pca = PCA(n_components=2)
    linreg = skl.LinearRegression()
    scaler_pcr = StandardScaler(with_mean=True, with_std=True)
    pipe_pcr = Pipeline([('scaler', scaler_pcr), ('pca', pca), ('linreg', linreg)])

    param_grid_pcr = {'pca__n_components': range(1, 20)}
    grid_pcr = skm.GridSearchCV(pipe_pcr,
                                param_grid_pcr,
                                cv=kfold,
                                scoring='neg_mean_squared_error')
    grid_pcr.fit(X_bs, Y)

    print(f"Optymalna liczba składowych: {grid_pcr.best_params_['pca__n_components']}")
    print(f"PCR minimalne CV MSE: {-grid_pcr.best_score_:.0f}")
    return (grid_pcr,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    CV MSE w funkcji liczby składowych głównych.
    """)
    return


@app.cell
def _(K, go, grid_pcr, np):
    n_comp = list(range(1, 20))
    fig_pcr = go.Figure()
    fig_pcr.add_scatter(x=n_comp,
                        y=-grid_pcr.cv_results_['mean_test_score'],
                        error_y=dict(array=grid_pcr.cv_results_['std_test_score'] / np.sqrt(K)),
                        mode='lines+markers')
    fig_pcr.update_layout(xaxis_title='Liczba składowych głównych', yaxis_title='CV MSE',
                          title='PCR — CV MSE')
    fig_pcr.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    # Zadania

    We wszystkich zadaniach pracujemy na zbiorze Hitters i macierzy predyktorów
    `X_bs` / wektorze odpowiedzi `Y` przygotowanych wcześniej.

    ## Zadanie 1 — porównanie Ridge i Lasso na zbiorze testowym

    Podziel zbiór na część uczącą (80%) i testową (20%) za pomocą
    `skm.train_test_split`.
    Dopasuj **wyłącznie na danych uczących** Ridge i Lasso z doborem $\lambda$
    przez CV (użyj tego samego `kfold` co wcześniej — obiekt `KFold` to tylko
    strategia podziału, wywoła `.split(X_train)` na mniejszym zbiorze, więc
    nie ma wycieku danych). Oblicz MSE na zbiorze testowym i zestaw je z CV MSE
    z danych uczących.

    *Pytanie do przemyślenia:* czy CV MSE dobrze przewiduje błąd testowy?
    Dlaczego może być zawyżone lub zaniżone?
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
    ## Zadanie 2 — regresja składowych ukrytych (PLS)

    Przeczytaj o metodzie **PLS** (`PLSRegression`) i dopasuj model z doborem optymalnej liczby składowych
    przez walidację krzyżową — analogicznie do PCR powyżej. Narysuj CV MSE
    w funkcji liczby składowych **na tym samym wykresie co PCR** i porównaj
    wyniki: która metoda potrzebuje mniej składowych? Która daje niższe minimum?

    Wskazówka: użyj tego samego `kfold` i `grid_pcr` co w sekcji PCR.
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
    ## Zadanie 3 — selekcja krokowa z kryterium $C_p$

    Wraz ze wzrostem ilości predyktorów MSE na zbiorze treningowym jest zawsze tylko lepsze. Zastosujemy zamiast niego kryterium oceny z penalizacją ilości parametrów (jak AIC lub BIC).

    Zaimplementuj $C_p$ jako kryterium oceny i użyj go do selekcji krokowej
    zamiast MSE. Kryterium $C_p$ jest zdefiniowane jako:

    $$C_p = \frac{1}{n}\left(\text{RSS} + 2p\hat{\sigma}^2\right)$$

    gdzie $p$ to liczba predyktorów w bieżącym modelu, a $\hat{\sigma}^2$ to
    wariancja reszt z modelu pełnego (OLS na wszystkich predyktorach):

    ```python
    sigma2 = OLS(Y, design.fit_transform(Hitters)).fit().mse_resid
    ```

    Wskazówka: zdefiniuj `nCp(sigma2, estimator, X, Y)` zwracające ujemne $C_p$
    — w treści funkcji użyj `X.shape` do wyznaczenia $p$ (`sklearn_selected` z ISLP
    przekazuje do scorera tylko kolumny aktualnie wybranych predyktorów, nie cały
    `X_bs`); użyj `partial()`
    do zamrożenia `sigma2`; przekaż kryterium do `sklearn_selected()` jako
    argument `scoring`.

    Porównaj zestaw wybranych zmiennych z wynikiem selekcji wg MSE z początku labu.
    """)
    return


@app.cell
def _():
    # Uzupełnij kod poniżej
    ...
    return


if __name__ == "__main__":
    app.run()
