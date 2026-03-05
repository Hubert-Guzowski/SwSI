### Laboratoria: Statystyka w Sztucznej Inteligencji i Analizie Danych

### 1. Przygotowanie środowiska

#### 1.1 Wymagania:
 - menadżer projektów i zależności uv: https://docs.astral.sh/uv/
 - python>=3.14.2: https://www.python.org
 - zależności określone w pyproject.toml

#### 1.2 Uruchamianie notebooków:

Aby zsynchronizować zainstalowane pakiety i uruchomić notebooki marimo, należy wykonać poniższe kroki:

 1. Sklonowanie repozytorium do wybranego folderu.
 2. Zainstalowanie uv zgodnie z instrukcją: https://docs.astral.sh/uv/getting-started/installation/
 3. Zsynchronizowanie zależności projektu: `uv sync`
 4. Uruchomienie wybranego notebooka marimo: `uv run marimo edit lab_files/lab0/lab0.py`

Po wykonaniu powyższych należy zweryfikować działanie wykonując komórki w notebooku lab0.py
