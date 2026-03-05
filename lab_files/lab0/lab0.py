import marimo

__generated_with = "0.20.4"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Lab 0: Wprowadzenie do Marimo

    Niniejszy kurs korzysta z notebooków **Marimo** - niedawno powstałej alternatywy dla Jupytera.

    ## Dlaczego Marimo?

    Wykorzystanie akurat marimo jest powodowane po prostu chęcią przetestowania nowego rozwiązania, ale przemawiają za nią konkretne zalety tych notebooków:

    - Pliki przechowywane są w formacie `.py`, czytelnym w repozytorium git
    - Komórki tworzą acykliczny graf zależności - brak ukrytego stanu
    - Automatyczne przeliczanie zależnych komórek przy zmianie
    - Wbudowane interaktywne elementy (suwaki, listy rozwijane, itp.)
    - Więcej: https://marimo.io
    """)
    return


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Struktura notebooka Marimo

    Każda komórka jest funkcją pythonową. Zmienne zwracane przez komórkę mogą być
    używane przez inne komórki - tworząc w ten sposób graf zależności.

    Komórki mogą zawierać:
    - kod Python,
    - tekst w formacie Markdown (przez `mo.md()`),
    - interaktywne elementy UI.

    Dane do zajęć przechowywane są w folderze `data/` w katalogu głównym projektu.
    Używamy `Path(__file__)` do wyznaczenia ścieżki niezależnie od miejsca uruchomienia notebooka:
    """)
    return


@app.cell
def _():
    import pandas as pd
    import numpy as np
    from pathlib import Path

    DATA_DIR = Path(__file__).parents[2] / "data"

    # Tworzenie przykładowego DataFrame
    cars_df = pd.DataFrame({
        'speed': [4, 4, 7, 7, 8, 9, 10, 10, 10, 11],
        'dist':  [2, 10, 4, 22, 16, 10, 18, 26, 34, 17]
    })

    cars_df.describe()
    return (cars_df,)


@app.cell
def _(cars_df, mo):
    # Gadżety z marimo:
    mo.ui.dataframe(cars_df)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Importowanie kodu z plików .py

    Importowanie działa, jak w standardowych pythonowych projektach, więc możemy doimplementowywać funkcjonalności w plikach innych, niz notebook. Należy uważać, bo zmieniając już po zaimportowaniu kod w pliku źródłowym, nie zaktualizujemy działania zaimportowanej funkcji.
    """)
    return


@app.cell
def _():
    from snippet import utility_function

    utility_function("input")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Interaktywne elementy

    Marimo dostarcza wbudowane elementy interaktywne, które automatycznie
    aktualizują zależne komórki:
    """)
    return


@app.cell
def _(mo):
    slider = mo.ui.slider(start=1, stop=100, value=50, label="Wybierz liczbę:")
    slider
    return (slider,)


@app.cell
def _(slider):
    result = slider.value ** 2
    f"Kwadrat wybranej liczby: {result}"
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Mozna np. zmieniać typ wykresu:
    """)
    return


@app.cell
def _(mo):
    chart_type = mo.ui.dropdown(
        options=["scatter", "bar", "line"],
        value="scatter",
        label="Typ wykresu:",
    )
    chart_type
    return (chart_type,)


@app.cell
def _(cars_df, chart_type):
    import plotly.express as px

    _type = chart_type.value
    if _type == "scatter":
        fig = px.scatter(cars_df, x="speed", y="dist", title="Scatter: speed vs dist")
    elif _type == "bar":
        fig = px.bar(cars_df, x="speed", y="dist", title="Bar: speed vs dist")
    else:
        fig = px.line(cars_df, x="speed", y="dist", title="Line: speed vs dist")
    fig
    return


if __name__ == "__main__":
    app.run()
