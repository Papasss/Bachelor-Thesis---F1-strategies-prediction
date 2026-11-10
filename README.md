# 🏎️ F1 Tyre Strategy Prediction with Deep Learning

![Python](https://img.shields.io/badge/Python-3.9+-blue?logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Keras-FF6F00?logo=tensorflow&logoColor=white)
![FastF1](https://img.shields.io/badge/Data-FastF1-E10600?logo=formula1&logoColor=white)
![Status](https://img.shields.io/badge/Thesis_Project-Unical-green)

Repository ufficiale del progetto di Tesi di Laurea in Informatica presso l'**Università degli Studi di Milano** (A.A. 2024/2025).

> **Titolo Tesi:** *Applicazione di algoritmi di Machine Learning per l’analisi predittiva dell’andamento delle gare di F1*  
> **Autore:** Davide Papasodaro  
> **Relatore:** Prof. Nicola Basilico

---

## 📌 Descrizione del Progetto

In Formula 1 le decisioni strategiche relative ai pit-stop e alla scelta delle gomme determinano spesso l'esito di un Gran Premio. Questo progetto esplora l'utilizzo di architetture di **Deep Learning** per analizzare i dati telemetrici e meteorologici storici e **prevedere la strategia ottimale delle mescole da asciutto (Soft, Medium, Hard) per ciascuno stint di gara**.

L'obiettivo è supportare le decisioni del muretto box simulando il comportamento e la resa degli pneumatici al variare delle condizioni di pista e dell'andamento della corsa.

---

## 📊 Dataset e Data Engineering

I dati sono stati estratti tramite le API della libreria Python **FastF1** e coprono le stagioni dal **2019 al 2024**.

### Feature principali utilizzate
Il dataset finale integra variabili di gara, telemetria e meteo:
* **Dati Gara & Pilota:** `Year`, `Round`, `Event`, `DriverNumber`, `Position`, `LapNumber`, `LapTime` (convertito in secondi), `TrackStatus`.
* **Dati Pneumatici:** `Stint`, `Compound`, `TyreLife`.
* **Dati Meteo & Pista:** `TrackTemp`, `AirTemp`, `Humidity`, `WindSpeed`, `WindDirection`, `Rainfall`.

### Pipeline di Preprocessing
1. **Data Cleaning:** Rimozione di valori nulli/duplicati e filtraggio delle sole sessioni su pista asciutta (`Rainfall == 0`), escludendo le gomme da bagnato (*Intermediate* e *Wet*).
2. **Definizione del Target (`CompoundBest`):** Raggruppamento dei dati per Anno, Evento e Giro per identificare la mescola più performante in ogni fase della gara.
3. **Encoding & Scaling:** Applicazione di *One-Hot Encoding* sulle variabili target categoriche (`Stint` e `CompoundBest`) e normalizzazione delle feature nel range `[0, 1]` tramite `MinMaxScaler`.
4. **Configurazioni del Dataset:**
   * **Dataset Completo:** 117 Gran Premi totali (divisi in 70% Training Set e 30% Test Set).
   * **Dataset "Alto Carico Aerodinamico":** Sottoinsieme filtrato di circuiti ad alto carico per testare la capacità di generalizzazione su gare specifiche (es. Test sul GP del Brasile 2022/2023).

---

## 🧠 Architetture di Deep Learning

Sono state progettate, addestrate e confrontate tre diverse reti neurali utilizzando **TensorFlow / Keras**:

1. **MLP (Multi-Layer Perceptron):** Rete feed-forward densa a più livelli nascosti (256 e 128 neuroni) con attivazione `ReLU` e output `Softmax`.
2. **LSTM (Long Short-Term Memory):** Rete neurale ricorrente (RNN) capace di catturare le dipendenze temporali giro dopo giro (layer da 256 e 128 unità con attivazione `ReLU` e output `Sigmoid`).
3. **GRU (Gated Recurrent Unit):** Variante ricorrente più leggera con 3 layer nascosti da 128 unità intervallati da `Dropout(0.2)` per contrastare l'overfitting.

### Configurazione del Training
* **Optimizer:** Adam (`learning_rate = 0.005` / `0.001`)
* **Loss Function:** Mean Squared Error (`MSE`)
* **Epoche:** Fino a 500 con callback di **EarlyStopping** (`patience = 60`, `min_delta = 0.0003`) e monitoraggio tramite **TensorBoard**.

---

## 📈 Risultati Sperimentali

I modelli sono stati valutati confrontando l'errore quadratico medio (**RMSE**), l'**Accuracy** e la distribuzione delle mescole predette rispetto a quelle reali nei vari stint di gara.

| Modello | Dataset | Train RMSE | Test RMSE | Note principali |
| :--- | :--- | :---: | :---: | :--- |
| **MLP** | 117 Gare (80/20) | `0.2470` | `0.2560` | Tende a sovrastimare l'uso della gomma *Soft* nei primi stint. |
| **GRU** | 117 Gare (80/20) | `0.1563` | `0.1741` | Buon bilanciamento e riduzione dell'errore rispetto a MLP. |
| **LSTM** | 117 Gare (80/20) | **`0.1839`** | **`0.1957`** | **Miglior modello:** riproduce fedelmente la distribuzione reale di *Soft*, *Medium* e *Hard* su tutti gli stint. |
| **LSTM** | Alto Carico (Test: Brasile) | `0.1904` | `0.1995` | Testato su singolo GP ad alto carico aerodinamico. |

> 💡 **Conclusione:** L'architettura **LSTM** si è dimostrata la più efficace nel comprendere l'evoluzione temporale della gara, restituendo conteggi di utilizzo delle mescole nei vari stint estremamente vicini alle strategie reali adottate dai team.

---

## 🛠️ Tecnologie e Librerie

* **Linguaggio:** Python 3.9
* **Data Extraction:** `fastf1`
* **Data Manipulation:** `pandas`, `numpy`
* **Machine & Deep Learning:** `tensorflow`, `keras`, `scikit-learn`
* **Data Visualization & Logging:** `matplotlib`, `tensorboard`

---

## 🚀 Come eseguire il progetto

1. **Clona il repository:**
   ```bash
   git clone [https://github.com/tuo-username/nome-repo.git](https://github.com/tuo-username/nome-repo.git)
   cd nome-repo
