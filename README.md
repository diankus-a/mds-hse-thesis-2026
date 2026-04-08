# Датасет: Детекция дублей в аналитических таксономиях

**ВКР:** Разработка инструмента для автоматической идентификации дублирующихся событий и проверки стандартов именования в аналитических таксономиях

---

## Структура репозитория

```
├── data/
│   ├── seed_taxonomy.csv     # Seed-таксономия банковских событий (102 записи)
│   ├── pairs_full.csv        # Полный датасет пар (5290 пар)
│   ├── pairs_train.csv       # Обучающая выборка (4232 пар, 80%)
│   └── pairs_test.csv        # Тестовая выборка (1058 пар, 20%)
├── notebooks/
│   └── eda_taxonomy.ipynb    # EDA + генерация датасета + baseline
├── figures/
│   ├── eda_seed.png          # Анализ seed-таксономии
│   ├── eda_violations.png    # Нарушения стандарта именования
│   ├── eda_pairs.png         # Анализ датасета пар
│   ├── eda_corr.png          # Корреляционная матрица признаков
│   ├── eda_baseline_cm.png   # Confusion matrix baseline-методов
│   └── eda_threshold.png     # F1 vs порог сходства
└── README.md
```

---

## Датасет

### Источники данных

| Источник | Тип | Объём | Роль |
|----------|-----|-------|------|
| Таксономия банка-партнёра | Реальные (анонимизированы) | ~1000 событий | Seed для генерации |
| Синтетическая аугментация | Генерируемые | 5290 пар | Обучение / тест моделей |
| Публичные таксономии (Amplitude, Segment) | Открытые | ~300 событий | Расширение seed |

### Seed-таксономия (`seed_taxonomy.csv`)

102 аналитических события из банковской таксономии, покрывающих 9 категорий:

| Категория | Событий | Пример |
|-----------|---------|--------|
| `auth` | 15 | `login_success`, `biometric_auth_failed` |
| `payment` | 16 | `payment_initiated`, `sbp_payment_success` |
| `card` | 14 | `card_blocked`, `virtual_card_created` |
| `account` | 11 | `balance_viewed`, `savings_goal_created` |
| `navigation` | 16 | `home_screen_viewed`, `search_result_clicked` |
| `credit` | 9 | `loan_approved`, `early_repayment_initiated` |
| `onboarding` | 11 | `kyc_submitted`, `registration_completed` |
| `notifications` | 4 | `push_notification_opened` |
| `invest` | 6 | `asset_purchased`, `portfolio_viewed` |

**Атрибуты:**

| Колонка | Тип | Описание |
|---------|-----|----------|
| `event_name` | str | Исходное название события |
| `category` | str | Категория (auth, payment, ...) |
| `subcategory` | str | Подкатегория |
| `naming_format` | str | Определённый формат (snake_case, camelCase, ...) |
| `is_valid` | bool | Соответствие стандарту snake_case |
| `norm_name` | str | Нормализованное название |
| `det_format` | str | Автоматически определённый формат |
| `violations` | list | Список нарушений стандарта |
| `n_tokens` | int | Количество токенов |
| `name_len` | int | Длина названия (символов) |

**Ключевые метрики seed:**
- Валидных событий (snake_case): **88 (86.3%)**
- Нарушающих стандарт: **14 (13.7%)**
- Средняя длина: **17.4 символа**
- Среднее количество токенов: **2.5**

### Датасет пар (`pairs_full.csv`)

5290 пар событий с бинарной разметкой дублей:

| Тип пары | Дубль? | Кол-во | Описание |
|----------|--------|--------|----------|
| `exact` | ✅ | 320 | Точное совпадение |
| `typo` | ✅ | ~315 | Опечатка (удаление/дублирование/замена символа) |
| `case` | ✅ | 320 | Смена регистра (camelCase, PascalCase, UPPER) |
| `permutation` | ✅ | ~245 | Перестановка токенов |
| `synonym` | ✅ | ~215 | Синоним одного токена |
| `semantic` | ✅ | 320 | Семантический дубль (разные слова, одно значение) |
| `not_duplicate` | ❌ | ~3555 | Hard negatives |

**Баланс классов:** label=1 (дубли) : label=0 (не дубли) ≈ **1:5**

**Атрибуты пар:**

| Колонка | Тип | Описание |
|---------|-----|----------|
| `event_1`, `event_2` | str | Названия событий в паре |
| `label` | int | 1 = дубль, 0 = не дубль |
| `dup_type` | str | Тип дубля / `not_duplicate` |
| `char_sim` | float | Символьное сходство (SequenceMatcher) |
| `norm_event_1/2` | str | Нормализованные названия |
| `norm_sim` | float | Символьное сходство после нормализации |
| `len_diff` | int | Разница длин пары |
| `tok_diff` | int | Разница количества токенов |

---

## Основные результаты EDA

### 1. Нарушения стандарта именования

Наиболее частые нарушения в реальных таксономиях:
1. **contains_uppercase** — смешение регистров (`LoginSuccess`, `KYC_submitted`)
2. **camelCase** — неправильный формат (`paymentInitiated`, `registrationCompleted`)
3. **contains_space** — пробелы в названиях (`payment initiated`, `tab switched`)
4. **mixed** — смешанный формат (`Payment_Failed`, `KYC_submitted`)

### 2. Характеристики дублей по типам

| Тип | char_sim (median) | norm_sim (median) | Сложность детекции |
|-----|-------------------|-------------------|-------------------|
| exact | 1.00 | 1.00 | Тривиальная |
| typo | 0.92 | 0.93 | Лёгкая |
| case | 0.75 | 0.98 | Лёгкая (после нормализации) |
| permutation | 0.82 | 0.84 | Средняя |
| synonym | 0.67 | 0.68 | Сложная |
| semantic | 0.42 | 0.43 | **Очень сложная** |

### 3. Baseline-методы

| Метод | Precision | Recall | F1 | AUC |
|-------|-----------|--------|-----|-----|
| B0: Random | ~0.21 | ~0.50 | ~0.30 | ~0.50 |
| B1: Char-sim (θ=0.62) | 0.847 | 0.859 | **0.853** | 0.905 |
| B2: Norm-sim (θ=0.63) | 0.876 | 0.890 | **0.883** | 0.921 |

**Вывод:** Строковые методы достигают F1 ≈ 0.88, однако не способны детектировать семантические дубли (F1 ≈ 0.00 для этого класса). Это обосновывает применение sentence embeddings на следующем этапе.

---

## Воспроизводимость

```bash
# Запустить EDA и регенерировать датасет
jupyter notebook notebooks/eda_taxonomy.ipynb

# Зависимости
pip install pandas numpy matplotlib seaborn scikit-learn
```

Python 3.10+, все зависимости из стандартных библиотек (кроме pandas/numpy/sklearn/matplotlib).

---

## Связанные ссылки

| Ресурс | Ссылка |
|--------|--------|
| Ноутбук EDA | `notebooks/eda_taxonomy.ipynb` |
| Датасет QQP (академический аналог) | https://quoradata.quora.com/First-Quora-Dataset-Release-Question-Pairs |
| MRPC датасет | https://www.microsoft.com/en-us/download/details.aspx?id=52398 |
| Sentence Transformers | https://www.sbert.net/ |
| Amplitude Taxonomy docs | https://amplitude.com/docs/data/amplitude-data-settings |

---

## Следующий этап (Чекпойнт 4)

- [ ] Обучение SBERT bi-encoder (`all-MiniLM-L6-v2`, `all-mpnet-base-v2`)
- [ ] Cross-encoder (DITTO-архитектура на RoBERTa)
- [ ] LLM zero-shot baseline (GPT/Claude API)
- [ ] Сравнение метрик: F1, AUC, F1 по типам дублей
- [ ] Анализ ошибок: какие типы дублей остаются сложными
