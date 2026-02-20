import marimo

__generated_with = "0.19.9"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Lab 0: Wprowadzenie do Marimo

    Niniejszy kurs korzysta z notebooków **Marimo** - nowoczesnej alternatywy dla Jupytera.

    ## Dlaczego Marimo?

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
    """)
    return


@app.cell
def _():
    # Przykład: proste obliczenia
    import pandas as pd
    import numpy as np

    # Tworzenie przykładowego DataFrame (odpowiednik R: data.frame)
    cars_df = pd.DataFrame({
        'speed': [4, 4, 7, 7, 8, 9, 10, 10, 10, 11],
        'dist':  [2, 10, 4, 22, 16, 10, 18, 26, 34, 17]
    })

    cars_df.describe()
    return cars_df, np, pd


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Importowanie kodu z plików .py

    Tak jak w Rmd importowaliśmy funkcje z plików `.r`, w Marimo możemy
    importować z dowolnych plików `.py`:
    """)
    return


@app.cell
def _():
    from snippet import utility_function

    utility_function("input")
    return (utility_function,)


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
    return (result,)


if __name__ == "__main__":
    app.run()
