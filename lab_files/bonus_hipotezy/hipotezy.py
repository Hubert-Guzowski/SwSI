import marimo

__generated_with = "0.23.9"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    import numpy as np
    import pandas as pd
    from scipy import stats
    import plotly.graph_objects as go
    from pathlib import Path

    DATA_DIR = Path(__file__).parents[2] / "data"
    return DATA_DIR, go, mo, np, pd, stats


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Testowanie hipotez

    **Test statystyczny sprowadza się do odczytania wartości z odpowiedniego rozkładu**. Jeśli spełnimy założenia testu, to przy prawdziwej hipotezie zerowej $H_0$ statystyka testowa ma *znany* rozkład.
    Czyli uniwersalne kroki do wykonania dla testowania hipotez to:

    1. sprawdzenie założeń,
    2. policzenie statystyki testowej z danych,
    3. naniesienie jej na odpowiedni rozkład,
    4. odczytanie **wartości krytycznej** (kwantyl, `dist.ppf`) lub **$p$-wartości**
       (pole pod ogonem, `dist.cdf` / `dist.sf`),
    5. porównanie z poziomem istotności $\alpha$, który sobie założyliśmy.

    Różne testy różnią się tylko tym, *który* to rozkład (jego formuła wynika z założeń):
    $N(0,1)$, $t$-Studenta, $\chi^2$, $F$, rozkład Kołmogorowa…

    #### Źródła historyczne
    Badania nad testami zostały w dużej mierze domknięte w poprzednim wieku.
    Testy, z których korzystamy mają wyprowadzenia, które są teraz utrzymywane w archiwach różnych instytucji. Przykładowo:

    - **t-Studenta** — W. S. Gosset ("Student"), *The probable error of a mean*
      (1908): [york.ac.uk/.../student.pdf](https://www.york.ac.uk/depts/maths/histstat/student.pdf)
    - **Kołmogorowa** — A. N. Kolmogorov, *Sulla determinazione empirica di una
      legge di distribuzione* (1933):
      [digitale.bnc.roma.sbn.it](http://digitale.bnc.roma.sbn.it/tecadigitale/giornale/CFI0353791/1933/unico) - jego artykuł jest na stronie 93
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Rozkład $t$-Studenta a rozkład normalny

    Test $t$ weryfikuje, czy średnie dwóch próbek są różne. Wyliczamy dla niego wartość:
    $$
    t = \frac{\bar{X}_1 - \bar{X}_2}{\text{SD}(\bar{X}_1 - \bar{X}_2)}
    $$,
    gdzie $\bar{X}$ oznacza średnią z $X$ a SD odchylenie standardowe.
    Następnie wstawiamy ją do rozkładu $t$-Studenta. Ma on cięższe ogony niż $N(0,1)$ - tym cięższe, im mniej stopni swobody (mała próba). Wraz ze wzrostem stopni swobody $t$ zmierza do $N(0,1)$.
    """)
    return


@app.cell(hide_code=True)
def _(go, np, stats):
    x_t = np.linspace(-4, 4, 400)
    fig_t = go.Figure()
    for df_t in [1, 2, 5, 30]:
        fig_t.add_trace(
            go.Scatter(x=x_t, y=stats.t.pdf(x_t, df_t), mode="lines", name=f"t (df={df_t})")
        )
    fig_t.add_trace(
        go.Scatter(
            x=x_t,
            y=stats.norm.pdf(x_t),
            mode="lines",
            name="N(0,1)",
            line=dict(color="black", width=3, dash="dash"),
        )
    )
    fig_t.update_layout(
        title="Rozkład t-Studenta dla różnych stopni swobody vs N(0,1)",
        xaxis_title="t",
        yaxis_title="gęstość",
        template="simple_white",
    )
    fig_t
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Rozkład statystyki Kołmogorowa

    W teście zgodności Kołmogorowa-Smirnowa statystyką jest największa odległość
    między dystrybuantą empiryczną a teoretyczną. Przy prawdziwej $H_0$ ma ona *znany* rozkład
    (`scipy.stats.kstwo`), zależny od liczebności próby $n$. Im większe $n$, tym
    rozkład $D_n$ skupia się bliżej zera - łatwiej wykryć drobne odstępstwa.
    """)
    return


@app.cell(hide_code=True)
def _(go, np, stats):
    x_ks = np.linspace(0.0001, 0.9, 400)
    fig_ks = go.Figure()
    for n_ks in [5, 10, 20, 50]:
        fig_ks.add_trace(
            go.Scatter(
                x=x_ks, y=stats.kstwo.pdf(x_ks, n_ks), mode="lines", name=f"D_n (n={n_ks})"
            )
        )
    fig_ks.update_layout(
        title="Rozkład statystyki Kołmogorowa D_n dla różnych liczebności próby",
        xaxis_title="D_n",
        yaxis_title="gęstość",
        template="simple_white",
    )
    fig_ks
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Jak odczytujemy wynik z rozkładu

    Poniższa funkcja pomocnicza przyjmuje **dowolny rozkład** (z `scipy.stats`)
    oraz **wartość statystyki** i pokazuje na wykresie:

    - obszar krytyczny (zacieniony) wyznaczony przez kwantyl `dist.ppf` przy
      poziomie $\alpha$ - to jest właśnie **wartość krytyczna**,
    - położenie naszej statystyki testowej,
    - obliczoną **$p$-wartość** (pole pod ogonem) i decyzję.

    Można potestować:
    """)
    return


@app.cell
def _(go, np):
    def show_test(dist, stat, alpha=0.05, alternative="two-sided", nazwa="rozkład H0"):
        lo, hi = float(dist.ppf(0.001)), float(dist.ppf(0.999))
        lo, hi = min(lo, stat) - 0.3, max(hi, stat) + 0.3
        xx = np.linspace(lo, hi, 600)
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(x=xx, y=dist.pdf(xx), mode="lines", name=nazwa, line=dict(color="steelblue"))
        )

        def ogon(mask, name):
            fig.add_trace(
                go.Scatter(
                    x=xx[mask],
                    y=dist.pdf(xx[mask]),
                    fill="tozeroy",
                    mode="none",
                    fillcolor="rgba(214,39,40,0.35)",
                    name=name,
                )
            )

        if alternative == "two-sided":
            kl, kr = dist.ppf(alpha / 2), dist.ppf(1 - alpha / 2)
            ogon(xx <= kl, "obszar krytyczny")
            ogon(xx >= kr, None)
            p = 2 * min(float(dist.cdf(stat)), float(dist.sf(stat)))
            kryt = f"±{kr:.3f}"
        elif alternative == "less":
            kl = dist.ppf(alpha)
            ogon(xx <= kl, "obszar krytyczny")
            p = float(dist.cdf(stat))
            kryt = f"{kl:.3f}"
        else:  # greater
            kr = dist.ppf(1 - alpha)
            ogon(xx >= kr, "obszar krytyczny")
            p = float(dist.sf(stat))
            kryt = f"{kr:.3f}"

        fig.add_vline(
            x=stat,
            line=dict(color="black", width=2, dash="dash"),
            annotation_text=f"statystyka = {stat:.3f}",
        )
        decyzja = "odrzucamy H0" if p < alpha else "brak podstaw do odrzucenia H0"
        fig.update_layout(
            title=f"wart. krytyczna: {kryt} | p = {p:.4f} | α = {alpha} → {decyzja}",
            xaxis_title="wartość statystyki",
            yaxis_title="gęstość",
            template="simple_white",
        )
        return fig

    return (show_test,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Przykład

    Rozkład pomiarów głębokości morza jest normalny, $\sigma = 5$ m, $\mu$ nieznane.
    Pomiary: 862, 870, 876, 866, 871. Na poziomie $\alpha = 0.05$ weryfikujemy
    $H_0: \mu = 870$ wobec $H_1: \mu \neq 870$.

    Bo wariancja jest *znana*, statystyka $Z = \frac{\bar{X} - \mu_0}{\sigma/\sqrt{n}}$
    ma przy $H_0$ rozkład $N(0,1)$ — wystarczy ją na nim nanieść.
    """)
    return


@app.cell
def _(np, show_test, stats):
    pomiary = np.array([862, 870, 876, 866, 871])
    sigma_demo = 5
    mu0_demo = 870
    alpha_demo = 0.05

    z_demo = (pomiary.mean() - mu0_demo) / (sigma_demo / np.sqrt(len(pomiary)))
    show_test(stats.norm(), z_demo, alpha=alpha_demo, alternative="two-sided", nazwa="N(0,1)")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Statystyka leży poza obszarem krytycznym (lub $p > \alpha$), czyli brak podstaw do
    odrzucenia $H_0$: dane nie przeczą średniej głębokości 870 m.

    -------

    ## Zadania do samodzielnego rozwiązania

    Pod każdym zadaniem uzupełnij kod tak, by policzyć statystykę i odczytać wynik
    (możesz użyć `show_test(...)` albo gotowych funkcji `scipy`/`statsmodels`).
    Następujące funkcje Pythona implementują testy wykorzystywane w zadaniach:

    - `statsmodels.stats.weightstats.ztest()` - test $Z$;

    - `scipy.stats.ttest_1samp()` - test $t$ dla jednej próby;

    - `scipy.stats.ttest_ind()` - test $t$ dla dwóch niezależnych prób;

    - `statsmodels.stats.proportion.proportions_ztest()` - test równości proporcji;

    - test $\chi^2$ dla wariancji - implementujemy ręcznie przy użyciu `scipy.stats.chi2`;

    - `scipy.stats.levene()` lub ręczny test $F$ przy użyciu `scipy.stats.f` - test $F$.

    W funkcjach scipy parametr `alternative` wyznacza hipotezę
    alternatywną $H_1$ (wartości: `'two-sided'`, `'less'`, `'greater'`).


    Przy jakim poziomie istotności hipoteza zerowa może zostać odrzucona?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Zadanie 1 - test $t$ dla jednej próby (rozkład $t$-Studenta)

    Automat ma produkować blaszki o nominalnej grubości $\mu_0 = 0.04$ mm. W pliku
    `blaszki.csv` zapisano pomiary 24 blaszek. Czy blaszki są **cieńsze** niż
    $0.04$ mm?
    """)
    return


@app.cell
def _(DATA_DIR, pd):
    blaszki = pd.read_csv(DATA_DIR / "blaszki.csv").iloc[:, 0].to_numpy()
    mu0_blaszki = 0.04
    alpha_blaszki = 0.01
    ...
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Zadanie 2 - test równości proporcji (rozkład $N(0,1)$)

    Spośród 97 chorych 51 podano lek, 46 placebo. Poprawę odczuło 12 leczonych i
    5 z placebo. Zweryfikuj hipotezę o braku wpływu leku ($\alpha = 0.05$).

    $$H_0: p_1 = p_2 \qquad H_1: p_1 \neq p_2$$
    """)
    return


@app.cell
def _():
    lek, placebo = 51, 46
    lek_poprawa, placebo_poprawa = 12, 5
    alpha_prop = 0.05
    ...
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Zadanie 3 - test $F$ równości wariancji (rozkład $F$)

    Zbadano wzrost 13 mężczyzn i 12 kobiet. Zakładając rozkłady normalne, sprawdź
    na poziomie $\alpha = 0.1$, że zmienność wzrostu mężczyzn jest **większa** niż
    kobiet.

    $$H_0: \sigma_M^2 = \sigma_K^2 \qquad H_1: \sigma_M^2 > \sigma_K^2$$
    """)
    return


@app.cell
def _(np):
    kobiety = np.array([161, 162, 163, 162, 166, 164, 168, 165, 168, 157, 161, 172])
    mezczyzni = np.array([171, 176, 179, 189, 176, 182, 173, 179, 184, 186, 189, 167, 177])
    alpha_f = 0.1
    ...
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Zadanie 4 - test zgodności Kołmogorowa-Smirnowa

    Poniższa próba ma pochodzić z rozkładu normalnego. Zweryfikuj zgodność z
    $N(\mu, \sigma)$ na poziomie $\alpha = 0.05$.

    $$H_0:\ \text{próba} \sim N \qquad H_1:\ \text{próba} \not\sim N$$
    """)
    return


@app.cell
def _(np):
    dane_ks = np.array(
        [4.9, 5.2, 4.6, 5.5, 5.0, 4.8, 5.3, 4.7, 5.1, 4.4, 5.6, 5.0, 4.9, 5.2, 4.7]
    )
    alpha_ks = 0.05
    ...
    return


if __name__ == "__main__":
    app.run()
