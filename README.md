# ColorForge AI

Trasforma una fotografia in una stampa 3D multicolore a strati, come si fa a mano
con HueForge. Il programma sceglie da solo i filamenti, l'ordine e le quote dei
cambi colore usando un modello ottico fisico (Kubelka-Munk). Si può calibrare.

> Le anteprime sono **previsioni** del modello, non misure. Finché i filamenti non
> sono calibrati (TD sconosciuta), il programma lo segnala e mostra un'affidabilità bassa.

## Installazione (Windows 10/11, Python 3.11+)

```bat
py -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Avvio

```bat
python -m colorforge                       :: interfaccia grafica
python -m colorforge foto.jpg              :: interfaccia con foto già caricata
python -m colorforge.cli foto.jpg -o out   :: riga di comando
```

Opzioni CLI utili: `--width 100` (mm), `--colors 8`, `--quality massima|bilanciata|veloce`,
`--filaments id1,id2,...`, `--printer bambu-x2d --changer ams2pro-x2`.

## Flusso d'uso

1. **Carica fotografia** (PNG, JPEG, TIFF, WebP). Orientamento EXIF e profilo ICC vengono gestiti in automatico.
2. **I miei filamenti…**: inserisci a mano colore (riquadro o HEX), marca, nome, TD e slot AMS.
   Il database personale viene salvato in `%APPDATA%\ColorForge\filaments.json`.
3. Scegli stampante (predefinita: **Bambu Lab X2D, 2× AMS 2 Pro, 8 slot**), larghezza, qualità.
4. **Elabora automaticamente**: confronta originale, previsione e mappa errori ΔE.
5. **Esporta**:
   - `colorforge.stl` + `colorforge_istruzioni.txt`: metodo sempre valido. In Bambu Studio
     aggiungi un cambio filamento alle quote indicate.
   - `colorforge.3mf`: un oggetto per ogni fascia di colore, con lo slot già assegnato.
     **Da verificare** in Bambu Studio 2.8.2.60.
   - immagini di anteprima, errori e report JSON.
6. **File → Salva progetto** (`.cfproj`): si riapre senza ricalcolare.

## Test

```bat
pip install -r requirements-dev.txt
python -m pytest
```

Documentazione tecnica e stato del progetto: [docs/ARCHITETTURA.md](docs/ARCHITETTURA.md).
