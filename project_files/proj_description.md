### Założenia
Projekt polega na wykonaniu analizy zalezności pomiędzy zmiennymi i interpretacji, co odróżnia
rozgrywki na różnych poziomach rankingowych gry dota 2. Zadanie można wykonać w oparciu o modele klasyfikacji (zbiór danych jest wysokopoziomowo podzielony na ligi), albo regresji (w informacjach o każdym z graczy znajduje się pole rank_tier o wartości całkowitej). Najważniejszym aspektem projektu jest jednak analiza wyników, do której należy wykorzystać poznane narzędzia, takie jak metody graficzne, macierze korelacji, lime czy shap.

### Zbiór danych
Zbiór danych został pobrany z api open dota (https://docs.opendota.com/#section/Introduction). Zawiera obiekt json opisujących szczegółowo mecze dla różnych lig rankingowych (odpowiedzi z endpointu matches). Na najwyższym poziomie słownik jest podzielony na listy odpowiadające ligom:

- Legend: 178 meczy
- Ancient: 218 meczy
- Archon: 168 meczy
- Guardian: 197 meczy
- Immortal: 100 meczy
- Herald: 267 meczy
- Crusader: 182 meczy
- Divine: 132 meczy

W ramach projektu można pobrać dodatkowe dane, aby wzmocnić wyciągnięte wnioski i nie muszą to być dane kolejnych meczy.

### Analiza
Analiza powinna obejmować:

- ocenę istotności predyktorów;

- ocenę charakteru (dodatni/ujemny, liniowy/nieliniowy) i wielkości wpływu 
istotnych predyktorów na odpowiedź;

- ocenę zależności między predyktorami.

### Prezentacja
Przesłany dokument ma mieć format HTML lub PDF wygenerowany na podstawie pliku R Markdown lub Jupyter Notebook.
Osoby, które jeszcze nie prezentowały poprzednich projektów, zostaną poproszone o prezentację na kolejnych zajęciach.