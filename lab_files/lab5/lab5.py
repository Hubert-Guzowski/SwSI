import marimo

__generated_with = "0.23.1"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Lab 5: Nieliniowe modele predykcyjne

    Liniowe modele regresji są często niewystarczające — w wielu zbiorach
    danych zależność między predyktorami a odpowiedzią jest wyraźnie
    nieliniowa. W tym laboratorium przechodzimy przez rodzinę technik
    o rosnącej elastyczności, starając się jednocześnie zachować
    interpretowalność modelu:

    1. **Regresja wielomianowa** — uogólnienie modelu liniowego na wielomiany
       wyższych stopni.
    2. **Funkcje schodkowe** — dyskretyzacja predyktora numerycznego do
       kategorii.
    3. **Funkcje sklejane (splines)** — ciągłe, odcinkowo-wielomianowe
       krzywe z kontrolowaną gładkością.
    4. **Wygładzające funkcje sklejane** — to samo, co powyżej, ale z karą za krzywiznę albo
       węzeł w każdym punkcie danych.
    5. **Regresja lokalna (LOESS)** — nieparametryczne wygładzanie
       ruchomym oknem (idea podobna do średniej kroczącej).
    6. **Uogólnione modele addytywne (GAM)** — łączenie powyższych technik
       dla wielu predyktorów jednocześnie.

    Wszystkie metody zastosujemy do tego samego, prostego zadania:
    modelowania zarobków `wage` w funkcji wieku `age` w zbiorze **Wage**
    z `ISLR`. Jest to bardzo popularny przykład, w którym zależność jest
    wyraźnie nieliniowa — zarobki rosną z wiekiem, by w okolicach 40–50
    lat osiągnąć plateau i potem maleć.
    """)
    return


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import pandas as pd
    import plotly.graph_objects as go
    import plotly.express as px
    import statsmodels.api as sm
    from statsmodels.stats.anova import anova_lm
    from statsmodels.gam.api import GLMGam, BSplines
    from statsmodels.genmod.families import Gaussian, Binomial
    from statsmodels.nonparametric.smoothers_lowess import lowess
    from ISLP import load_data
    from ISLP.models import ModelSpec as MS
    from ISLP.models import poly, bs, ns

    return (
        BSplines,
        Binomial,
        GLMGam,
        Gaussian,
        MS,
        anova_lm,
        bs,
        go,
        load_data,
        lowess,
        mo,
        np,
        ns,
        pd,
        poly,
        sm,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Dane

    Zbiór `Wage` zawiera 3000 obserwacji pracowników z regionu
    Mid-Atlantic (2003–2009) i 11 zmiennych. Odpowiedzią będzie
    `wage` (roczne zarobki w tysiącach USD), a w roli predyktorów wykorzystamy:

    - `age` — główny predyktor, na którym demonstrujemy każdą
      kolejną rodzinę modeli.
    - `year` — rok (dyskretne, niewielki zakres) — w modelach GAM.
    - `education` — kategoryczna zmienna porządkowa — również w GAM.

    Aby uniknąć powtarzania tego samego kodu, od razu przygotowujemy
    siatkę `age_grid` do predykcji oraz wektor odpowiedzi `y`.
    """)
    return


@app.cell
def _(load_data, np):
    Wage = load_data('Wage')
    Wage.info()
    y = Wage['wage']
    age = Wage['age']
    age_grid = np.linspace(age.min(), age.max(), 200)
    return Wage, age, age_grid, y


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Regresja wielomianowa

    Najprostsze rozszerzenie modelu liniowego: zastępujemy pojedynczy
    predyktor $X$ wielomianem stopnia $d$ czyli bazą $X, X^2, \ldots,
    X^d$. Model pozostaje **liniowy w parametrach**, więc uczymy go
    klasyczną metodą najmniejszych kwadratów.

    ### Wielomiany ortogonalne

    `poly()` z `ISLP.models` tworzy bazę **wielomianów ortogonalnych** —
    kolumny macierzy cech są nieskorelowane. Daje to dwa praktyczne
    zyski:

    - każdy współczynnik można testować niezależnie (t-statystyki nie są
      zafałszowane przez kolinearność);
    - dodanie kolumny wyższego stopnia nie zmienia oszacowań niższych
      stopni, więc test istotności najwyższej potęgi jest testem
      wyboru stopnia wielomianu.

    Obiekt `MS` opakowuje transformację — najpierw `.fit()` wylicza
    statystyki potrzebne do standaryzacji kolumn, a `.transform()`
    stosuje ją do dowolnego zbioru (tu zarówno do `Wage` podczas
    uczenia, jak i później do siatki `age_grid` podczas predykcji).
    """)
    return


@app.cell
def _(MS, Wage, poly, sm, y):
    poly_age = MS([poly('age', degree=4)]).fit(Wage)
    X_poly4 = poly_age.transform(Wage)
    model_poly4 = sm.OLS(y, X_poly4).fit()
    print(model_poly4.summary())
    return X_poly4, poly_age


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    W tabeli widać, że t-statystyki kolejnych stopni maleją: 11, 12, 3,
    2 — a p-value dla najwyższego (czwartego) stopnia to już ok. 5%.
    To sugeruje, że wielomian stopnia 4 może być nadmiarowy. Pokażemy
    to formalnie testem ANOVA za chwilę.

    ### Standardowa baza wielomianowa

    Ten sam model można wyrazić w "surowej" bazie $X, X^2, X^3, X^4$
    (parametr `raw=True`). Współczynniki wychodzą zupełnie inne (kolumny
    są teraz silnie skorelowane), ale **predykcje i $R^2$ są identyczne**
    — to tylko inna parametryzacja tej samej rodziny funkcji.
    Ortogonalność w `poly()` jest wyłącznie wygodą interpretacyjną.
    """)
    return


@app.cell
def _(MS, Wage, poly, sm, y):
    poly_age_raw = MS([poly('age', degree=4, raw=True)]).fit(Wage)
    model_poly4_raw = sm.OLS(y, poly_age_raw.transform(Wage)).fit()
    print(model_poly4_raw.summary())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Wybór stopnia wielomianu (ANOVA)

    Zamiast wybierać stopień wielomianu "na oko", możemy to zrobić
    systematycznie. **Test ANOVA** (analysis of variance) sprawdza, czy
    bardziej złożony model istotnie poprawia dopasowanie względem
    prostszego. Warunek jest taki, że modele muszą być **zagnieżdżone** — prostszy
    musi być szczególnym przypadkiem bardziej złożonego. Dla
    wielomianów stopnia 1, 2, 3, … jest to spełnione z automatu.

    Funkcja `anova_lm()` przyjmuje ciąg dopasowanych modeli i liczy
    sekwencję testów F porównujących kolejne pary.
    """)
    return


@app.cell
def _(MS, Wage, anova_lm, poly, sm, y):
    poly_models = [
        sm.OLS(y, MS([poly('age', degree=d)]).fit_transform(Wage)).fit()
        for d in range(1, 6)
    ]
    print(anova_lm(*poly_models))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Odczytujemy wiersz po wierszu: przejście 1 w 2 i 2 w 3 zdecydowanie
    poprawia model, 3 w 4 jest na granicy istotności
    (około 5 \%), a 4 w 5 już jej nie przekracza.
    W takim razie wybieramy wielomian stopnia 3.

    Zauważmy, że te same p-value pojawiły się w tabeli współczynników
    ortogonalnego `poly(age, 4)` powyżej — dzięki ortogonalności test
    istotności ostatniego stopnia jest równoważny testowi ANOVA.

    Dopasowujemy wybrany model stopnia 3 i przygotowujemy predykcje
    wraz z 95% przedziałem ufności na siatce `age_grid`.
    """)
    return


@app.cell
def _(MS, Wage, age_grid, pd, poly, sm, y):
    poly_age3 = MS([poly('age', degree=3)]).fit(Wage)
    model_poly3 = sm.OLS(y, poly_age3.transform(Wage)).fit()

    age_df = pd.DataFrame({'age': age_grid})
    preds_poly3 = model_poly3.get_prediction(poly_age3.transform(age_df))
    bands_poly3 = preds_poly3.conf_int(alpha=0.05)
    return age_df, bands_poly3, preds_poly3


@app.cell
def _(age, age_grid, bands_poly3, go, preds_poly3, y):
    fig_poly = go.Figure()
    fig_poly.add_scatter(x=age, y=y, mode='markers',
                         opacity=0.3, marker=dict(size=3, color='gray'),
                         name='Dane')
    fig_poly.add_scatter(x=age_grid, y=preds_poly3.predicted_mean,
                         mode='lines', name='Wielomian st. 3',
                         line=dict(color='red', width=2))
    fig_poly.add_scatter(x=age_grid, y=bands_poly3[:, 1],
                         mode='lines', name='CI górny',
                         line=dict(color='red', dash='dash'))
    fig_poly.add_scatter(x=age_grid, y=bands_poly3[:, 0],
                         mode='lines', name='CI dolny',
                         line=dict(color='red', dash='dash'))
    fig_poly.update_layout(title='Regresja wielomianowa (stopień 3)',
                           xaxis_title='Age', yaxis_title='Wage')
    fig_poly.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Krzywa rośnie do ok. 45. roku życia, stabilizuje się, a potem lekko
    spada. Przedziały ufności **rozszerzają się na brzegach zakresu
    `age`** — to typowa wada wielomianów: wysokie potęgi są silnie
    ważone w okolicy krańców i mały zbiór danych nie potrafi ich
    stabilnie oszacować. Tę wadę zaadresują naturalne funkcje
    sklejane w dalszej części laboratorium.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Regresja logistyczna wielomianowa

    Tę samą ideę możemy zastosować w klasyfikacji. Definiujemy binarną
    odpowiedź `wage > 250` (kategoria "wysoko zarabiający", wyrażona
    w tysiącach dolarów rocznie) i dopasowujemy model GLM z rodziną
    **Binomial** (link logit). Ponownie używamy wielomianu stopnia 4 po
    `age` jako macierzy cech — składowe modelu (wybór bazy) i sposób
    dopasowania (metoda największej wiarygodności) są niezależne.

    Zauważ, że pozytywna klasa jest bardzo rzadka — tylko ok. 2.5%
    obserwacji. Będzie to miało konsekwencje dla szerokości przedziałów
    ufności.
    """)
    return


@app.cell
def _(X_poly4, sm, y):
    high_earn = (y > 250).astype(int)
    logit_poly = sm.GLM(high_earn, X_poly4, family=sm.families.Binomial()).fit()
    print(logit_poly.summary())
    return high_earn, logit_poly


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    `get_prediction()` zwraca predykcje na skali logitu wraz
    z błędem standardowym. `conf_int()` liczy przedziały ufności
    również na tej skali, a `statsmodels` automatycznie transformuje
    je funkcją logistyczną przy dostępie przez `predicted_mean`
    — dostajemy więc gotowe prawdopodobieństwa i ich 95% pasmo.
    """)
    return


@app.cell
def _(age_df, logit_poly, poly_age):
    preds_logit = logit_poly.get_prediction(poly_age.transform(age_df))
    bands_logit = preds_logit.conf_int(alpha=0.05)
    return bands_logit, preds_logit


@app.cell
def _(age, age_grid, bands_logit, go, high_earn, preds_logit):
    fig_logit = go.Figure()
    fig_logit.add_scatter(x=age, y=high_earn,
                          mode='markers', opacity=0.3,
                          marker=dict(size=3, color='gray'), name='Dane')
    fig_logit.add_scatter(x=age_grid, y=preds_logit.predicted_mean,
                          mode='lines', name='P(wage > 250)',
                          line=dict(color='red', width=2))
    fig_logit.add_scatter(x=age_grid, y=bands_logit[:, 1],
                          mode='lines', line=dict(color='red', dash='dash'),
                          name='CI')
    fig_logit.add_scatter(x=age_grid, y=bands_logit[:, 0],
                          mode='lines', line=dict(color='red', dash='dash'),
                          showlegend=False)
    fig_logit.update_layout(title='P(wage > 250 | age)',
                            xaxis_title='Age',
                            yaxis_title='Prawdopodobieństwo')
    fig_logit.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Na wykresie powyżej przedział ufności jest bardzo szeroki w okolicy
    środka zakresu wieku, a prawdopodobieństwa pozostają niskie —
    trudno cokolwiek odczytać z pełnej skali 0..1. Dlatego pokazujemy
    to samo w powiększeniu (0..0.2).
    """)
    return


@app.cell
def _(age, age_grid, bands_logit, go, high_earn, preds_logit):
    fig_logit_zoom = go.Figure()
    fig_logit_zoom.add_scatter(x=age, y=high_earn * 0.2,
                               mode='markers', opacity=0.3,
                               marker=dict(size=3, color='gray'), name='Dane')
    fig_logit_zoom.add_scatter(x=age_grid, y=preds_logit.predicted_mean,
                               mode='lines', name='P(wage > 250)',
                               line=dict(color='red', width=2))
    fig_logit_zoom.add_scatter(x=age_grid, y=bands_logit[:, 1],
                               mode='lines', line=dict(color='red', dash='dash'),
                               name='CI')
    fig_logit_zoom.add_scatter(x=age_grid, y=bands_logit[:, 0],
                               mode='lines', line=dict(color='red', dash='dash'),
                               showlegend=False)
    fig_logit_zoom.update_layout(title='P(wage > 250 | age) — zoom',
                                 xaxis_title='Age',
                                 yaxis_title='Prawdopodobieństwo',
                                 yaxis=dict(range=[0, 0.2]))
    fig_logit_zoom.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Funkcje schodkowe

    Alternatywa dla globalnych wielomianów jest dyskretyzacja predyktora.
    Zakres `age` dzielimy na rozłączne przedziały i w każdym dopasowujemy stałą — uzyskujemy funkcję przedziałami stałą zamiast próbować modelować kształt krzywej.
    Brzmi to bardzo prosto, ale jest zaskakująco często stosowane w analizie.
    Dostajemy informację, ile średni zarabia każda grupa wiekowa.

    Używamy `pd.qcut(age, 4)` — czyli podziału na cztery kwantyle, dzięki
    czemu każdy przedział ma zbliżoną liczność (po ok. 750 obserwacji).
    Dla podziału na równe szerokości byłaby to `pd.cut()`.

    `pd.qcut()` zwraca zmienną kategoryczną; przekształcamy ją w macierz
    zerojedynkową przez `pd.get_dummies()`. Uwaga: nie porzucamy
    żadnego poziomu (bez `drop_first=True`) i nie dodajemy jawnego
    wyrazu wolnego — dzięki temu każdy współczynnik jest średnim
    wynagrodzeniem w swoim przedziale.
    """)
    return


@app.cell
def _(age, pd):
    cut_age = pd.qcut(age, 4)
    print(cut_age.value_counts().sort_index())
    return (cut_age,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Punkty cięcia — 33.75, 42 i 51 lat — to 25., 50. i 75. percentyl
    `age`. Teraz dopasowujemy OLS na kolumnach zerojedynkowych.
    """)
    return


@app.cell
def _(cut_age, pd, sm, y):
    X_step = pd.get_dummies(cut_age).astype(float)
    model_step = sm.OLS(y, X_step).fit()
    print(model_step.summary())
    return X_step, model_step


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Interpretacja jest bardzo prosta: współczynnik to mniej więcej średnia `wage` w przedziale.
    Widzimy, że grupa najmłodsza (18–33) zarabia średnio 94 tys., a trzy pozostałe grupy 116–119 tys. — po 33 roku
    życia wzrost zarobków zwalnia, a po 50 nie rośnie już istotnie. Wykres poniżej pokazuje schodkowy charakter
    tego modelu: predykcja zmienia się tylko na granicach przedziałów.
    """)
    return


@app.cell
def _(X_step, age, age_grid, cut_age, go, model_step, pd, y):
    cut_grid = pd.cut(age_grid, bins=cut_age.cat.categories)
    X_step_grid = pd.get_dummies(cut_grid).astype(float)
    X_step_grid = X_step_grid[X_step.columns]
    preds_step = model_step.predict(X_step_grid)

    fig_step = go.Figure()
    fig_step.add_scatter(x=age, y=y, mode='markers',
                         opacity=0.3, marker=dict(size=3, color='gray'),
                         name='Dane')
    fig_step.add_scatter(x=age_grid, y=preds_step, mode='lines',
                         name='Funkcja schodkowa',
                         line=dict(color='red', width=2))
    fig_step.update_layout(title='Regresja — funkcja schodkowa (4 przedziały)',
                           xaxis_title='Age', yaxis_title='Wage')
    fig_step.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Funkcje sklejane (splines)

    Wielomian globalny jest jednolity w całym zakresie i źle kontroluje brzegi.
    Funkcja schodkowa narzuca niegładki, mocno ograniczony kształt.
    **Funkcja sklejana** to połączenie zalet obu: odcinkowo jest wielomianem (domyślnie stopnia 3), ale
    w punktach łączenia zwanych **węzłami** narzucamy warunki gładkości — ciągłość funkcji oraz jej $d-1$ pierwszych pochodnych.
    Dla sześciennej funkcji sklejanej oznacza to ciągłą pierwszą i drugą pochodną w węzłach — krzywa wygląda gładko.

    Bazę B-spline udostępnia `bs()` z `ISLP.models`.

    ### Funkcje sklejane z ustalonymi węzłami

    Najprostszy sposób określenia elastyczności modelu: podajemy wprost listę węzłów wewnętrznych. Węzły powinny znajdować się tam, gdzie
    spodziewamy się zmiany zachowania funkcji — mój startowy wybór to `[25, 40, 60]`.
    """)
    return


@app.cell
def _(MS, Wage, bs, sm, y):
    bs_age_knots = MS([bs('age', internal_knots=[25, 40, 60], name='bs(age)')]).fit(Wage)
    model_bs_knots = sm.OLS(y, bs_age_knots.transform(Wage)).fit()
    print(model_bs_knots.summary())
    return bs_age_knots, model_bs_knots


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Model ma 6 kolumn (stopień 3 + 3 węzły wewnętrzne), plus wyrazcwolny — 7 stopni swobody łącznie.
    Współczynniki B-spline same w sobie są trudne do interpretacji (każdy odpowiada funkcji bazowej o zwartym nośniku), dlatego kluczowym wynikiem jest tu **wykres**. Pionowe przerywane linie zaznaczają położenie węzłów — mniej więcej widać, że krzywa zmienia charakter w tych miejscach.
    """)
    return


@app.cell
def _(age, age_df, age_grid, bs_age_knots, go, model_bs_knots, y):
    preds_bs_knots = model_bs_knots.get_prediction(bs_age_knots.transform(age_df))
    bands_bs_knots = preds_bs_knots.conf_int(alpha=0.05)

    fig_bs = go.Figure()
    fig_bs.add_scatter(x=age, y=y, mode='markers', opacity=0.3,
                       marker=dict(size=3, color='gray'), name='Dane')
    fig_bs.add_scatter(x=age_grid, y=preds_bs_knots.predicted_mean,
                       mode='lines', name='B-spline',
                       line=dict(color='red', width=2))
    fig_bs.add_scatter(x=age_grid, y=bands_bs_knots[:, 1], mode='lines',
                       line=dict(color='red', dash='dash'), name='CI')
    fig_bs.add_scatter(x=age_grid, y=bands_bs_knots[:, 0], mode='lines',
                       line=dict(color='red', dash='dash'), showlegend=False)
    for k in [25, 40, 60]:
        fig_bs.add_vline(x=k, line_dash='dot', line_color='blue')
    fig_bs.update_layout(title='B-spline, węzły: 25, 40, 60',
                         xaxis_title='Age', yaxis_title='Wage')
    fig_bs.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Ćwiczenie 1: wpływ ustawienia węzłów

    Sprawdź jak liczba i położenie węzłów wpływa na dopasowaną krzywą.
    Przetestuj np. `internal_knots=[30, 50]`, `[20, 30, 50, 70]`, `[35]` - wygodnie zrobić to np. sliderem marimo.
    """)
    return


@app.cell
def _():
    # miejsce na Twoje rozwiązanie
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Funkcje sklejane ze stałą liczbą stopni swobody

    Wybór konkretnych węzłów wymaga wiedzy dziedzinowej. Alternatywnie możemy podać liczbę **stopni swobody** (`df`) — wówczas `bs()` rozmieszcza $\text{df} - \text{degree}$ węzłów wewnętrznych na równomiernych **kwantylach** predyktora.
    To rozwiązanie bez założeń - ustawia automatycznie więcej węzłów tam, gdzie jest więcej danych.

    Dla `df=6` i domyślnego `degree=3` dostajemy 3 węzły wewnętrzne (dokładnie tyle, co w poprzednim przykładzie, ale w innych miejscach — na 25., 50. i 75. percentylu `age`).
    """)
    return


@app.cell
def _(MS, Wage, bs, sm, y):
    bs_age_df = MS([bs('age', df=6, name='bs(age,df=6)')]).fit(Wage)
    model_bs_df = sm.OLS(y, bs_age_df.transform(Wage)).fit()
    print(model_bs_df.summary())
    return bs_age_df, model_bs_df


@app.cell
def _(age, age_df, age_grid, bs_age_df, go, model_bs_df, y):
    preds_bs_df = model_bs_df.get_prediction(bs_age_df.transform(age_df))
    bands_bs_df = preds_bs_df.conf_int(alpha=0.05)

    fig_bs_df = go.Figure()
    fig_bs_df.add_scatter(x=age, y=y, mode='markers', opacity=0.3,
                          marker=dict(size=3, color='gray'), name='Dane')
    fig_bs_df.add_scatter(x=age_grid, y=preds_bs_df.predicted_mean,
                          mode='lines', name='B-spline (df=6)',
                          line=dict(color='red', width=2))
    fig_bs_df.add_scatter(x=age_grid, y=bands_bs_df[:, 1], mode='lines',
                          line=dict(color='red', dash='dash'), name='CI')
    fig_bs_df.add_scatter(x=age_grid, y=bands_bs_df[:, 0], mode='lines',
                          line=dict(color='red', dash='dash'), showlegend=False)
    fig_bs_df.update_layout(title='B-spline, df = 6 (węzły na kwantylach)',
                            xaxis_title='Age', yaxis_title='Wage')
    fig_bs_df.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Ćwiczenie 2: wpływ liczby stopni swobody

    Zbadaj jak wartość `df` wpływa na gładkość dopasowania.
    Porównaj `df=3`, `df=6`, `df=10`, `df=20`.
    """)
    return


@app.cell
def _():
    # miejsce na Twoje rozwiązanie
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Ćwiczenie 3: wpływ stopnia wielomianu

    Funkcja `bs()` akceptuje parametr `degree`. Sprawdź jak wygląda krzywa dla `degree=0` (funkcja schodkowa), `degree=1` (łamana), `degree=2` oraz domyślnego `degree=3`.
    """)
    return


@app.cell
def _():
    # miejsce na Twoje rozwiązanie
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Naturalne funkcje sklejane

    Zwykłe B-spline'y mają dalej wadę wielomianów na brzegach zakresu:
    tam gdzie danych jest mało (skrajne wartości `age`), sześcienny ogon ma dużą wariancję, a przedział ufności bardzo się rozszerza.
    Tzw. Naturalna funkcja sklejana narzuca dodatkowe ograniczenie: poza skrajnymi węzłami funkcja jest liniowa co drastycznie redukuje wariancję na brzegach przy minimalnym koszcie zdolności modelowania (zwykle w krańcach zakresu i tak brakuje danych, które uzasadniałyby silną krzywiznę).

    Dodatkowo, dzięki ograniczeniu liniowości na brzegach, ta sama wartość `df` oznacza, że przeznaczamy większą giętkość krzywizny na środek zakresu.
    """)
    return


@app.cell
def _(MS, Wage, ns, sm, y):
    ns_age = MS([ns('age', df=4, name='ns(age,df=4)')]).fit(Wage)
    model_ns = sm.OLS(y, ns_age.transform(Wage)).fit()
    print(model_ns.summary())
    return model_ns, ns_age


@app.cell
def _(age, age_df, age_grid, go, model_ns, ns_age, y):
    preds_ns = model_ns.get_prediction(ns_age.transform(age_df))
    bands_ns = preds_ns.conf_int(alpha=0.05)

    fig_ns = go.Figure()
    fig_ns.add_scatter(x=age, y=y, mode='markers', opacity=0.3,
                       marker=dict(size=3, color='gray'), name='Dane')
    fig_ns.add_scatter(x=age_grid, y=preds_ns.predicted_mean,
                       mode='lines', name='Natural spline (df=4)',
                       line=dict(color='red', width=2))
    fig_ns.add_scatter(x=age_grid, y=bands_ns[:, 1], mode='lines',
                       line=dict(color='red', dash='dash'), name='CI')
    fig_ns.add_scatter(x=age_grid, y=bands_ns[:, 0], mode='lines',
                       line=dict(color='red', dash='dash'), showlegend=False)
    fig_ns.update_layout(title='Naturalna funkcja sklejana (df=4)',
                         xaxis_title='Age', yaxis_title='Wage')
    fig_ns.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Porównaj szerokość przedziału ufności na krańcach z wcześniejszymi
    wykresami `bs()` i wielomianu stopnia 3 — przy tej samej ilości
    danych krzywa pozostaje znacznie lepiej oszacowana.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Ćwiczenie 4: wpływ df dla naturalnych splines

    Porównaj krzywe dla `df=2, 4, 8, 16`. Zwróć uwagę na brzegi zakresu
    wieku — porównaj z `bs()` z sekcji powyżej.
    """)
    return


@app.cell
def _():
    # miejsce na Twoje rozwiązanie
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Wygładzające funkcje sklejane

    Dotychczasowe metody wymagały wyboru liczby i położenia węzłów.
    Wygładzająca funkcja sklejana obchodzi ten problem umieszczając węzeł w **każdym punkcie danych**, ale dodaje karę za nadmierną krzywiznę. Minimalizujemy:

    $$
    \sum_i (y_i - f(x_i))^2 + \lambda \int f''(t)^2\, dt
    $$

    Pierwszy człon to dopasowanie; drugi — całka kwadratu drugiej pochodnej — karze dużą zmienność kierunkową funkcji. Parametr $\lambda$ kontroluje
    siłę penalizacji: 0 daje interpolację wszystkich punktów, wartość nieskończona dałaby regresję liniową.

    W praktyce wygodniej jest sterować nie $\lambda$, lecz efektywną liczbą stopni swobody `df` (od 2 do liczby punktów). Większe `df` = bardziej giętka krzywa.

    W `statsmodels` używamy `GLMGam` z komponentem `BSplines`.
    Uwaga: model wymaga jawnego podania `exog` — aby otrzymać "czystą" wygładzającą krzywą (bez żadnych efektów parametrycznych) podajemy jako `exog` stałą kolumnę jedynek.
    """)
    return


@app.cell
def _(BSplines, GLMGam, Gaussian, age, np, y):
    x_age_col = np.asarray(age).reshape(-1, 1)
    gam_smooth_16 = GLMGam(
        y,
        smoother=BSplines(x_age_col, df=[16], degree=[3]),
        exog=np.ones((len(age), 1)),
        family=Gaussian()
    ).fit()
    return (gam_smooth_16,)


@app.cell
def _(age, age_grid, gam_smooth_16, go, np, y):
    preds_smooth_16 = gam_smooth_16.predict(
        exog=np.ones((len(age_grid), 1)),
        exog_smooth=age_grid[:, None]
    )

    fig_smooth = go.Figure()
    fig_smooth.add_scatter(x=age, y=y, mode='markers', opacity=0.3,
                           marker=dict(size=3, color='gray'), name='Dane')
    fig_smooth.add_scatter(x=age_grid, y=preds_smooth_16,
                           mode='lines', name='Smoothing spline (df=16)',
                           line=dict(color='red', width=2))
    fig_smooth.update_layout(
        title='Wygładzająca funkcja sklejana (df=16)',
        xaxis_title='Age', yaxis_title='Wage',
        yaxis=dict(range=[0, 300])
    )
    fig_smooth.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Regresja lokalna (LOESS)

    LOESS (**locally estimated scatterplot smoothing**) to podejście
    w pełni **nieparametryczne**: dla każdego punktu $x_0$, w którym
    chcemy oszacować $\hat f(x_0)$, liczymy ważoną regresję wielomianową
    (zwykle stopnia 1 lub 2) na najbliższych sąsiadach $x_0$. Wagi
    maleją z odległością, a obserwacje spoza okna są ignorowane.

    Parametr `frac` (span) określa szerokość sąsiedztwa jako ułamek
    całego zbioru. `frac=0.2` oznacza, że do każdego lokalnego
    dopasowania bierzemy 20% najbliższych obserwacji; `frac=0.5` —
    połowę danych. **Mniejsze `frac` = wrażliwsza krzywa**, większe
    = gładsza. Parametr ten odgrywa rolę analogiczną do `df` w splines
    lub $\lambda$ w wygładzających splines.

    Wadą LOESS w porównaniu do metod bazowych jest brak zwartej
    formy modelu — przewidywanie wymaga zapamiętania **całego**
    zbioru uczącego i policzenia nowej regresji dla każdego punktu
    predykcji.
    """)
    return


@app.cell
def _(age, age_grid, go, lowess, y):
    fig_loess = go.Figure()
    fig_loess.add_scatter(x=age, y=y, mode='markers', opacity=0.2,
                          marker=dict(size=3, color='gray'), name='Dane')
    for span_val, color_val in zip([0.2, 0.5], ['red', 'blue']):
        loess_fit = lowess(y, age, frac=span_val, xvals=age_grid)
        fig_loess.add_scatter(x=age_grid, y=loess_fit, mode='lines',
                              name=f'frac = {span_val}',
                              line=dict(color=color_val, width=2))
    fig_loess.update_layout(title='Regresja lokalna LOESS',
                            xaxis_title='Age', yaxis_title='Wage')
    fig_loess.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Uogólnione modele addytywne (GAM)

    Dotąd modelowaliśmy tylko pojedynczy predyktor `age`. GAM uogólnia
    wszystkie wcześniejsze techniki na **wiele predyktorów**
    jednocześnie:

    $$
    y = \beta_0 + f_1(x_1) + f_2(x_2) + \ldots + f_p(x_p) + \varepsilon
    $$

    Każde $f_j$ może być czymś zupełnie innym: splineem, funkcją
    liniową, funkcją schodkową dla kategorycznej zmiennej itp.
    Kluczowa cecha to **addytywność** — nie ma interakcji między
    predyktorami, a wpływ każdego można oglądać osobno (patrz PDP
    w dalszej części). To właśnie addytywność zapewnia
    interpretowalność GAM mimo elastyczności.

    ### GAM metodą najmniejszych kwadratów

    Gdy wszystkie składniki $f_j$ są **parametryczne** (baza o stałym
    wymiarze), GAM redukuje się do zwykłej regresji liniowej — po
    prostu skonkatenowanej z kilku bloków macierzy cech. Zbudujemy
    model:

    $$
    \texttt{wage} \sim \texttt{ns(year, df=4)} + \texttt{ns(age, df=5)} + \texttt{education}
    $$

    Używamy `ModelSpec`, który automatycznie buduje macierz projektową
    z kilku komponentów (spline po `year`, spline po `age`, kodowanie
    zero-jedynkowe `education`).
    """)
    return


@app.cell
def _(MS, Wage, ns, sm, y):
    gam_ls_spec = MS([ns('year', df=4), ns('age', df=5), 'education']).fit(Wage)
    X_gam_ls = gam_ls_spec.transform(Wage)
    gam_ls = sm.OLS(y, X_gam_ls).fit()
    print(gam_ls.summary())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### GAM ze wygładzającymi funkcjami sklejanymi

    Zamiast stałej bazy możemy dla `year` i `age` użyć **wygładzających** splines z karą — wówczas model ma dwa poziomy
    regularyzacji: parametrycznie dla `education` (bez kary) i z karą za krzywiznę dla `year`, `age`. `GLMGam` fittuje to wszystko
    jednocześnie algorytmem **backfitting**: naprzemiennie aktualizuje
    każdy komponent, trzymając pozostałe ustalone, do zbieżności.

    `exog` mieści efekty parametryczne, `smoother` (klasa `BSplines`)
    — gładkie. `df` dla każdej zmiennej wygładzanej zadaje jej
    efektywną złożoność.
    """)
    return


@app.cell
def _(BSplines, GLMGam, Gaussian, MS, Wage, y):
    gam_lin_spec = MS(['education']).fit(Wage)
    X_gam_lin = gam_lin_spec.transform(Wage)
    x_smooth = Wage[['year', 'age']].values

    gam_smooth_model = GLMGam(
        y,
        exog=X_gam_lin,
        smoother=BSplines(x_smooth, df=[5, 5], degree=[3, 3]),
        family=Gaussian()
    ).fit()
    print(gam_smooth_model.summary())
    return X_gam_lin, gam_lin_spec, gam_smooth_model


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Wykresy częściowej zależności (PDP)

    Podsumowanie modelu GAM zawiera współczynniki gładkich bazowych
    komponentów — te są trudne do bezpośredniej interpretacji. Znacznie
    czytelniejszym narzędziem jest **wykres częściowej zależności**
    (*Partial Dependence Plot*), pokazujący jak wskazany predyktor
    wpływa na $\hat y$, gdy pozostałe są "wyzerowane" do wartości
    reprezentatywnych (zazwyczaj średnich).

    Konstrukcja ręczna:

    1. Bierzemy siatkę wartości predyktora, który nas interesuje.
    2. Dla pozostałych predyktorów wstawiamy ich wartości średnie
       (powtórzone wzdłuż siatki).
    3. Przepuszczamy tak spreparowaną macierz przez `model.predict()`.
    4. Centrujemy wynik (odejmujemy średnią) — PDP pokazuje względne
       **odchylenie** od bazowej predykcji, nie wartość absolutną.

    Dla modeli addytywnych PDP dokładnie odtwarza kształt $f_j(x_j)$
    — to kolejna zaleta addytywnej struktury GAM.
    """)
    return


@app.cell
def _(Wage, X_gam_lin, age_grid, np, pd):
    n_grid = len(age_grid)
    year_grid = np.linspace(Wage['year'].min(), Wage['year'].max(), n_grid)
    mean_year = Wage['year'].mean()
    mean_age = Wage['age'].mean()

    X_lin_mean = pd.DataFrame(
        np.tile(X_gam_lin.mean(axis=0).values, (n_grid, 1)),
        columns=X_gam_lin.columns
    )
    smooth_age_grid = np.column_stack([np.full(n_grid, mean_year), age_grid])
    smooth_year_grid = np.column_stack([year_grid, np.full(n_grid, mean_age)])
    return (
        X_lin_mean,
        mean_age,
        mean_year,
        n_grid,
        smooth_age_grid,
        smooth_year_grid,
        year_grid,
    )


@app.cell
def _(X_lin_mean, age_grid, gam_smooth_model, go, smooth_age_grid):
    pdp_age = gam_smooth_model.predict(exog=X_lin_mean, exog_smooth=smooth_age_grid)

    fig_pdp_age = go.Figure()
    fig_pdp_age.add_scatter(x=age_grid, y=pdp_age - pdp_age.mean(),
                            mode='lines', line=dict(color='red', width=2))
    fig_pdp_age.update_layout(
        title='PDP: wpływ age na wage (przy pozostałych cechach uśrednionych)',
        xaxis_title='Age', yaxis_title='Efekt na wage')
    fig_pdp_age.show()
    return


@app.cell
def _(X_lin_mean, gam_smooth_model, go, smooth_year_grid, year_grid):
    pdp_year = gam_smooth_model.predict(exog=X_lin_mean, exog_smooth=smooth_year_grid)

    fig_pdp_year = go.Figure()
    fig_pdp_year.add_scatter(x=year_grid, y=pdp_year - pdp_year.mean(),
                             mode='lines', line=dict(color='red', width=2))
    fig_pdp_year.update_layout(
        title='PDP: wpływ year na wage',
        xaxis_title='Year', yaxis_title='Efekt na wage')
    fig_pdp_year.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Logistyczny GAM

    Tak jak wielomianową regresję przenieśliśmy wcześniej do świata
    klasyfikacji (zmieniając `sm.OLS` na `sm.GLM` z rodziną `Binomial`),
    tak samo GAM dopasowany przez `GLMGam` przyjmuje argument `family`.
    Model pozostaje addytywny, ale teraz na skali **logitu**:

    $$
    \log\frac{P(y=1 \mid x)}{1 - P(y=1 \mid x)} = \beta_0 + f_1(x_1) + \ldots
    $$

    Wewnątrz `predict()` `statsmodels` sam transformuje przewidywania
    do przedziału $[0, 1]$.

    **Suwakiem** poniżej ustawiasz próg $\tau$ zarobków definiujący
    klasę pozytywną `wage > τ`. Model refituje się przy każdej zmianie.
    Obserwuj dwa zjawiska:

    - kształt PDP po `age` zmienia się niewiele (to ta sama struktura
      zarobków), ale jego **amplituda** (wysokość predykcji) maleje
      wraz ze wzrostem progu;
    - przy wysokich progach klasa pozytywna staje się bardzo rzadka
      — model traci wsparcie danych, a predykcje stają się płaskie
      i niepewne.
    """)
    return


@app.cell
def _(mo):
    threshold_slider = mo.ui.slider(
        start=100, stop=400, step=25, value=250,
        label='Próg wage (tys. USD)', show_value=True)
    threshold_slider
    return (threshold_slider,)


@app.cell
def _(BSplines, Binomial, GLMGam, Wage, gam_lin_spec, threshold_slider, y):
    threshold = threshold_slider.value
    high_earn_th = (y > threshold).astype(int)
    X_logit_gam = gam_lin_spec.transform(Wage)
    x_logit_smooth = Wage[['year', 'age']].values

    gam_logit = GLMGam(
        high_earn_th,
        exog=X_logit_gam,
        smoother=BSplines(x_logit_smooth, df=[5, 5], degree=[3, 3]),
        family=Binomial()
    ).fit()
    print(f'Próg: wage > {threshold}')
    print(f'Udział klasy pozytywnej: {high_earn_th.mean():.2%} '
          f'({high_earn_th.sum()} z {len(high_earn_th)} obserwacji)')
    return X_logit_gam, gam_logit, threshold


@app.cell
def _(
    X_logit_gam,
    age_grid,
    gam_logit,
    go,
    mean_year,
    n_grid,
    np,
    pd,
    threshold,
):
    X_logit_mean = pd.DataFrame(
        np.tile(X_logit_gam.mean(axis=0).values, (n_grid, 1)),
        columns=X_logit_gam.columns
    )
    smooth_age_logit = np.column_stack([np.full(n_grid, mean_year), age_grid])
    prob_age = gam_logit.predict(exog=X_logit_mean, exog_smooth=smooth_age_logit)

    fig_logit_age = go.Figure()
    fig_logit_age.add_scatter(x=age_grid, y=prob_age, mode='lines',
                              line=dict(color='blue', width=2))
    fig_logit_age.update_layout(
        title=f'Logistyczny GAM: P(wage > {threshold}) względem age',
        xaxis_title='Age', yaxis_title='Prawdopodobieństwo')
    fig_logit_age.show()
    return (X_logit_mean,)


@app.cell
def _(X_logit_mean, gam_logit, go, mean_age, n_grid, np, threshold, year_grid):
    smooth_year_logit = np.column_stack([year_grid, np.full(n_grid, mean_age)])
    prob_year = gam_logit.predict(exog=X_logit_mean, exog_smooth=smooth_year_logit)

    fig_logit_year = go.Figure()
    fig_logit_year.add_scatter(x=year_grid, y=prob_year, mode='lines',
                               line=dict(color='blue', width=2))
    fig_logit_year.update_layout(
        title=f'Logistyczny GAM: P(wage > {threshold}) względem year',
        xaxis_title='Year', yaxis_title='Prawdopodobieństwo')
    fig_logit_year.show()
    return


if __name__ == "__main__":
    app.run()
