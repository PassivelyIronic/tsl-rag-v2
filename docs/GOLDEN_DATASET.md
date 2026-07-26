# Golden dataset — format i instrukcja generowania

Dataset mieszka w `evals/golden_dataset/questions.json` i jest walidowany przy każdym
wczytaniu oraz w testach (`tests/unit/test_golden_dataset.py`). Zepsuty wpis wywala eval
od razu, z komunikatem wskazującym pytanie — nie po dwudziestu minutach przebiegu.

Sprawdzenie datasetu bez uruchamiania czegokolwiek:

```powershell
uv run pytest tests/unit/test_golden_dataset.py -v
```

---

## 1. Format

```json
{
  "schema_version": 2,
  "questions": [
    {
      "id": "ec561-czas-jazdy-dzienny",
      "category": "numeric_fact",
      "variant": "standard",
      "question": "Jaki jest maksymalny dzienny czas prowadzenia pojazdu?",
      "expected_answer": "9 godzin, 10 godzin, dwa razy w tygodniu",
      "expected_docs": ["ec_561_2006"],
      "expected_articles": ["6"],
      "source_note": "Art. 6 ust. 1"
    }
  ]
}
```

| Pole | Wymagane | Znaczenie |
|---|---|---|
| `id` | tak | Unikalny, kebab-case, stabilny. Po nim identyfikuje się pytanie między przebiegami — nie zmieniaj go przy przeredagowaniu treści |
| `category` | tak | Jedna z sześciu wartości z §2 |
| `variant` | nie (domyślnie `standard`) | `standard`, `bez_ogonkow` albo `potoczne` — patrz §3 |
| `question` | tak | Pytanie tak, jak zadałby je człowiek |
| `expected_answer` | tak (poza `out_of_scope`) | **Fragmenty faktów rozdzielone przecinkami**, nie zdanie. Patrz §4 — to jest najczęstsze źródło błędów |
| `expected_docs` | tak (poza `out_of_scope`) | Identyfikatory z §5. Dla `out_of_scope` musi być `[]` |
| `expected_articles` | nie | Numery artykułów, np. `["6"]`, `["8", "9"]`. Używane do audytu poprawności cytowań |
| `source_note` | nie | Skąd wzięta odpowiedź, np. `"Art. 6 ust. 1"` albo `"Załącznik 1, lp. 12"`. Do weryfikacji przez człowieka |

---

## 2. Kategorie

| Kategoria | Co sprawdza | Charakterystyka pytania |
|---|---|---|
| `numeric_fact` | Czy system podaje **dokładną liczbę** z przepisu | „Ile godzin…", „Jaki jest maksymalny…". Odpowiedź zawiera konkretne wartości |
| `procedure` | Czy system opisuje **warunki i kroki**, nie tylko liczbę | „Czy można…, i pod jakimi warunkami", „W jaki sposób…" |
| `cross_document` | Czy system **łączy dwa akty** i wskazuje pierwszeństwo | Pytanie, na które nie da się odpowiedzieć z jednego dokumentu. **Wymaga min. 2 wpisów w `expected_docs`** |
| `scope` | Czy system trafia w **właściwy akt** przy aktach o zbliżonej tematyce | „Czy X stosuje się do…", „Kogo obejmuje…". Tu system najczęściej myli AETR z rozporządzeniami UE |
| `penalty` | Czy system trafia we **właściwy taryfikator** (kierowca / przewoźnik / zarządzający) | „Jaka kara grozi kierowcy za…", „Ile zapłaci przewoźnik za…". Podmiot kary musi być jednoznaczny |
| `out_of_scope` | Czy system **odmawia** zamiast halucynować | Pytanie z pogranicza transportu, ale spoza korpusu. `expected_docs: []`, `expected_answer: ""` |

**Rozróżnienie podmiotu w `penalty` jest krytyczne.** System obecnie zawodzi dokładnie tutaj:
zapytany o karę dla kierowcy cytuje taryfikator przewoźnika. Pytania z tej kategorii muszą
mieć w treści jasno wskazany podmiot („kierowca", „przewoźnik", „osoba zarządzająca").

---

## 3. Warianty pytań

Dataset v1 miał wszystkie pytania napisane poprawną polszczyzną z diakrytykami, więc nie był
w stanie zmierzyć rzeczy, które realnie się psują u nietechnicznego użytkownika.

| `variant` | Przykład | Po co |
|---|---|---|
| `standard` | „Jaki jest maksymalny dzienny czas prowadzenia pojazdu?" | Podstawa |
| `bez_ogonkow` | „Jaki jest maksymalny dzienny czas prowadzenia pojazdu bez przerwy?" pisane jako „Jaki jest maks czas jazdy bez przerwy" **bez polskich znaków** | Mierzy składanie diakrytyków w tokenizerze BM25. Realny sposób pisania w pośpiechu |
| `potoczne` | „Ile mogę jechać jednego dnia?" | Bez terminologii z aktu prawnego. Sprawdza, czy retrieval działa, gdy pytanie nie zawiera słów z przepisu |

Ten sam fakt może wystąpić w dwóch wariantach — wtedy `expected_answer` i `expected_docs`
są identyczne, a różni się tylko `question`, `variant` i `id`. To celowe: para
`standard` / `bez_ogonkow` mierzy wprost wpływ zapisu na retrieval.

---

## 4. `expected_answer` — najważniejsza zasada

Ocena keyword-match dzieli `expected_answer` po przecinkach i sprawdza, czy **każdy fragment
występuje w odpowiedzi jako podciąg**, po zamianie na małe litery. Z tego wynika:

✅ **Dobrze** — krótkie, dosłowne fragmenty:
```
"9 godzin, 10 godzin, dwa razy w tygodniu"
"45 minut"
"od 50 do 2000 zł"
"56 godzin prowadzenia, 48 godzin średni czas pracy"
```

❌ **Źle** — całe zdanie. Nigdy nie dopasuje się dosłownie, więc pytanie zawsze dostanie 0
i będzie wyglądać na porażkę modelu:
```
"Kierowca może prowadzić pojazd maksymalnie 9 godzin dziennie, z możliwością
 przedłużenia do 10 godzin nie częściej niż dwa razy w tygodniu."
```

❌ **Źle** — przecinek wewnątrz jednego faktu rozbija go na dwa nieosiągalne fragmenty:
```
"1 500, 00 zł"      → szuka osobno "1 500" i "00 zł"
```
Zapisz jako `"1500 zł"` albo `"1 500 zł"` bez przecinka dziesiętnego.

Dodatkowe zasady:

- **Liczby zapisuj cyframi, tak jak w akcie.** Model odpowiadający „dziewięć godzin" dostanie
  0 przy ocenie keyword-match — to znana słabość tej metody i powód, dla którego istnieje
  tryb `--use-judge`. Nie próbuj tego obchodzić, wpisując oba zapisy.
- **Maksymalnie 60 znaków na fragment** (walidator odrzuca dłuższe).
- **2–4 fragmenty na pytanie.** Jeden fragment daje ocenę zero-jedynkową, więcej niż cztery
  sprawiają, że prawidłowa odpowiedź prawie nigdy nie dostaje 1.0.
- Dla `out_of_scope` zostaw `""` — ocena opiera się wyłącznie na tym, czy system odmówił.

---

## 5. Dozwolone identyfikatory dokumentów

Dokładnie te wartości, małymi literami. Inne odrzuci walidator:

| `expected_docs` | Dokument |
|---|---|
| `ec_561_2006` | Rozporządzenie (WE) 561/2006 — czas prowadzenia, przerwy, odpoczynki |
| `eu_2020_1054` | Rozporządzenie (UE) 2020/1054 — Pakiet Mobilności, zmiany do 561/2006 i 165/2014 |
| `eu_165_2014` | Rozporządzenie (UE) 165/2014 — tachografy |
| `eu_1071_2009` | Rozporządzenie (WE) 1071/2009 — dostęp do zawodu przewoźnika |
| `eu_1072_2009` | Rozporządzenie (WE) 1072/2009 — dostęp do rynku, kabotaż, licencja wspólnotowa |
| `eu_2016_403` | Rozporządzenie (UE) 2016/403 — klasyfikacja poważnych naruszeń, utrata dobrej reputacji |
| `directive_2002_15` | Dyrektywa 2002/15/WE — organizacja czasu pracy |
| `directive_2020_1057` | Dyrektywa (UE) 2020/1057 — delegowanie kierowców |
| `aetr` | Umowa AETR — przewozy międzynarodowe, państwa trzecie |
| `pl_driver_hours_act` | Ustawa z 16.04.2004 o czasie pracy kierowców (PL) |
| `tariff_driver_2022` | **Załącznik nr 1** — grzywny dla **kierowcy** |
| `tariff_company_2022` | **Załącznik nr 3** — kary dla **przewoźnika (pracodawcy)** |
| `tariff_manager_2022` | **Załącznik nr 4** — kary dla **osoby zarządzającej transportem** |

**`TARIFF_EMPLOYER_2022.pdf` nie ma identyfikatora i nie może wystąpić w `expected_docs`.**
Jest bajtowo identyczny z `TARIFF_COMPANY_2022.pdf` — oba to Załącznik nr 3. Pytania o kary
dla pracodawcy/przewoźnika wskazują `tariff_company_2022`.

---

## 6. Ile pytań z którego pliku

Cel: **42 pytania**, min. 5 na kategorię. Przydział uwzględnia objętość dokumentu w korpusie
(liczba chunków) i to, których kategorii dany akt dotyczy.

| Plik | Pytań | Kategorie |
|---|---:|---|
| `EC_561_2006.pdf` | **5** | 4 × `numeric_fact`, 1 × `procedure` |
| `DIRECTIVE_2002_15.pdf` | **3** | 2 × `numeric_fact`, 1 × `procedure` |
| `EU_165_2014.pdf` | **3** | 2 × `numeric_fact`, 1 × `procedure` |
| `EU_2020_1054.pdf` | **2** | 2 × `numeric_fact` |
| `EU_1071_2009.pdf` | **2** | 1 × `procedure`, 1 × `scope` |
| `EU_1072_2009.pdf` | **2** | 2 × `scope` (kabotaż — kogo i czego dotyczy) |
| `DIRECTIVE_2020_1057.pdf` | **2** | 2 × `scope` (delegowanie — kiedy ma zastosowanie) |
| `AETR.pdf` | **3** | 2 × `scope`, 1 × `numeric_fact` |
| `PL_DRIVER_HOURS_ACT.pdf` | **2** | 1 × `numeric_fact`, 1 × `procedure` |
| `EU_2016_403.pdf` | **2** | 2 × `penalty` (kategorie naruszeń, skutek dla reputacji) |
| `TARIFF_DRIVER_2022.pdf` | **2** | 2 × `penalty` — **kara dla kierowcy** |
| `TARIFF_COMPANY_2022.pdf` | **1** | 1 × `penalty` — **kara dla przewoźnika** |
| `TARIFF_MANAGER_2022.pdf` | **1** | 1 × `penalty` — **kara dla zarządzającego** |
| **łącznie z pojedynczych plików** | **30** | |
| `cross_document` (pary aktów) | **6** | patrz niżej |
| `out_of_scope` (bez pliku) | **6** | patrz niżej |
| **RAZEM** | **42** | |

Sugerowane pary dla `cross_document` — pytania, na które nie da się odpowiedzieć z jednego aktu:

1. `ec_561_2006` + `directive_2002_15` — czas prowadzenia a czas pracy
2. `ec_561_2006` + `eu_2020_1054` — co Pakiet Mobilności zmienił w odpoczynkach
3. `ec_561_2006` + `aetr` — przewóz częściowo poza UE
4. `eu_165_2014` + `eu_2020_1054` — wymogi tachografu po zmianach
5. `eu_2016_403` + `tariff_company_2022` — poważne naruszenie a kara pieniężna
6. `eu_1071_2009` + `eu_2016_403` — dobra reputacja a klasyfikacja naruszeń

Dla `out_of_scope` — 6 pytań z pogranicza, **na które w korpusie nie ma odpowiedzi**:
ograniczenia prędkości, płaca minimalna w innym kraju, przepisy celne, ADR/materiały
niebezpieczne, badania techniczne pojazdu, prawo jazdy i jego kategorie.

Z 42 pytań: **6 w wariancie `bez_ogonkow`** i **4 w wariancie `potoczne`** — jako duplikaty
faktów już obecnych w zestawie `standard`, rozłożone po różnych kategoriach.

### Uwaga o limicie zapytań

Pełny przebieg evalu to jedno wywołanie generacji na pytanie. Przy 42 pytaniach i darmowym
limicie OpenRoutera **50 zapytań na dobę** wychodzi jeden przebieg dziennie i to bez zapasu.
Konsekwencje, do rozstrzygnięcia przy rozszerzaniu:

- bramka w CI stanie na `run_retrieval_evals.py`, który **nie wywołuje LLM-a** i nie ma limitu,
- pełny eval generacji uruchamiany świadomie, nie przy każdym commicie,
- jednorazowe doładowanie $10 na OpenRouterze podnosi limit do 1000/dobę i nie wygasa.

---

## 7. Prompt do NotebookLM

Poniższe wklej do NotebookLM z całym korpusem jako źródłami. Generuj **plikami**, nie wszystko
naraz — łatwiej weryfikować i mniejsza szansa, że model pomyli akty.

```
Jesteś ekspertem prawa transportowego UE. Na podstawie WYŁĄCZNIE dokumentu
{NAZWA_PLIKU} wygeneruj {N} pytań testowych do systemu RAG.

Zwróć czysty JSON — tablicę obiektów, bez komentarzy i bez tekstu wokół:

[
  {
    "id": "krotki-identyfikator-kebab-case",
    "category": "numeric_fact",
    "variant": "standard",
    "question": "Pytanie po polsku, tak jak zadałby je człowiek",
    "expected_answer": "fragment 1, fragment 2",
    "expected_docs": ["identyfikator_dokumentu"],
    "expected_articles": ["6"],
    "source_note": "Art. 6 ust. 1"
  }
]

ZASADY BEZWZGLĘDNE:

1. "expected_answer" to KRÓTKIE FRAGMENTY FAKTÓW ROZDZIELONE PRZECINKAMI,
   nie zdanie. Każdy fragment musi być takim ciągiem znaków, który dosłownie
   pojawi się w poprawnej odpowiedzi. Maksymalnie 60 znaków na fragment,
   od 2 do 4 fragmentów na pytanie.
   DOBRZE: "9 godzin, 10 godzin, dwa razy w tygodniu"
   ŹLE:    "Kierowca może prowadzić pojazd maksymalnie 9 godzin dziennie."
2. Liczby zapisuj CYFRAMI, dokładnie tak jak w akcie. Nie zaokrąglaj.
   Nie używaj przecinka wewnątrz jednej liczby — "1500 zł", nie "1 500,00 zł".
3. "expected_docs" to dokładnie: ["{IDENTYFIKATOR}"] — nic innego.
4. "category" wyłącznie z tej listy, w podanych proporcjach: {PROPORCJE}
   - numeric_fact: pytanie o konkretną liczbę z przepisu
   - procedure: pytanie o warunki lub kroki, nie o samą liczbę
   - scope: pytanie o to, kogo lub czego dany akt dotyczy
   - penalty: pytanie o wysokość kary, z JEDNOZNACZNIE wskazanym podmiotem
     (kierowca / przewoźnik / osoba zarządzająca)
5. "source_note" musi wskazywać miejsce w dokumencie, z którego pochodzi
   odpowiedź. Jeśli nie potrafisz go wskazać, POMIŃ CAŁE PYTANIE.
6. Nie zadawaj pytań, na które odpowiedź wymaga wiedzy spoza tego dokumentu.
7. Nie zadawaj pytań o to, czego w dokumencie nie ma. Żadnego wnioskowania
   ani ogólnej wiedzy o transporcie.
```

Podstawiaj:

- `{NAZWA_PLIKU}` i `{IDENTYFIKATOR}` — z tabeli w §5
- `{N}` i `{PROPORCJE}` — z tabeli w §6

Dla `cross_document` i `out_of_scope` uruchom prompt osobno, ze wszystkimi źródłami
podłączonymi naraz, i wskaż pary aktów z §6.

---

## 8. Co zrobić z wynikiem

Wklej otrzymane tablice JSON do rozmowy albo zapisz do pliku — scalę je z istniejącym
`questions.json`, uruchomię walidację i zdejmę `xfail` z testu wymagającego 5 pytań
na kategorię, gdy próg zostanie osiągnięty.

**Przed scaleniem trzeba przejrzeć `source_note` w każdym pytaniu.** To jedyny moment,
w którym da się wychwycić, że model przypisał odpowiedź do złego artykułu — a błąd w golden
datasecie jest gorszy niż brak pytania, bo od tego momentu mierzymy system względem
nieprawdy. Sam tego nie zweryfikuję: nie mam dostępu do treści PDF-ów poza tym, co
znalazło się w bazie jako chunki.
