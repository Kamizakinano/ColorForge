# Architettura e stato del progetto

## Moduli

| Modulo | Responsabilità |
|---|---|
| `color/spaces.py` | sRGB ↔ lineare ↔ CIELAB, CIEDE2000 |
| `imaging/importer.py` | lettura file, EXIF, ICC → sRGB, trasparenza, errori leggibili |
| `imaging/analysis.py` | luminosità, contrasto, esposizione, rumore, dettagli, soggetto (euristica) |
| `imaging/preprocess.py` | ridimensionamento a risoluzione di stampa, correzioni moderate e registrate |
| `filaments/` | scheda filamento con affidabilità del dato, database JSON con import/export |
| `optics/km.py` | **modello fisico**: Kubelka-Munk per banda, stack dal basso verso l'alto |
| `optimize/search.py` | beam search su sequenza filamenti e quote, affinamento coordinato |
| `optimize/heightmap.py` | quota per pixel (min ΔE00), livellamento accettato solo se non peggiora |
| `geometry/mesh.py` | mesh a colonne chiusa e manifold, correzione "scacchiere", validazione |
| `export/` | STL binario, 3MF (basematerials + model_settings Bambu), report |
| `printers/` | profili stampante e sistemi di cambio (dati da specifiche pubbliche) |
| `project/session.py` | formato `.cfproj` versionato con migrazioni |
| `pipeline.py` | orchestrazione con avanzamento e annullamento |
| `gui/` | PySide6: finestra principale, editor filamenti, elaborazione in thread |

## Modello ottico

Per ogni filamento: `R∞` (colore a spessore infinito) e `S` (scattering).
`K/S = (1-R∞)²/(2R∞)`. Strato di spessore *d* su sfondo *Rg*: formula chiusa di
Kubelka. `S` si ricava da (in ordine di preferenza): calibrazione, TD, valore
generico 2 mm. Il valore generico viene sempre segnalato. La TD è definita come
lo spessore a cui il contrasto bianco/nero residuo scende al 2%.

Limiti: 3 bande RGB (non spettrale), nessuna correzione di Saunderson, nessuna
diffusione laterale. Il modello non è ancora stato validato su stampe reali.

## Ottimizzazione

Come nel flusso HueForge, il filamento di ogni strato è globale. La quota di un
pixel decide il suo colore, cioè la LUT dello stack fino a quella quota.
Obiettivo: ΔE00 medio pesato su 32–96 colori rappresentativi (k-means in Lab) +
penalità per cambio (dipende dalla qualità). Il risultato è deterministico a
parità di input (seed fisso).

## Modalità multicolore per zona (predefinita con AMS)

Ogni colonna di pixel è una ricetta: base comune (filamento più chiaro) + filamento
medio per *a* strati + filamento superiore per *b* strati. Il colore della ricetta si
calcola con lo stack KM. Per ogni pixel si sceglie la ricetta con ΔE00 minimo
(kd-tree in Lab + rifinitura CIEDE2000). I filamenti (≤ capacità AMS) si scelgono
in modo greedy. Gli oggetti 3MF sono uno per (ruolo, filamento). `interval_mesh` è
una superficie di voxel con vertici sdoppiati agli spigoli non-manifold, quindi è
sempre chiusa e 2-manifold. Il prezzo è un numero alto di cambi filamento, che il
report stima.

## Stato

**Completato (prototipo eseguibile, 31 test automatici)**
- Modalità multicolore per zona (AMS), tavolozza dei colori dell'immagine con anteprima automatica
- Import foto, analisi, preparazione adattiva
- Database filamenti con inserimento manuale da interfaccia
- Modello ottico KM, ricerca automatica, mappa quote, mesh validata
- Export STL, 3MF multi-oggetto, istruzioni cambi colore, report, anteprime
- Progetti `.cfproj`, interfaccia grafica con thread, annullamento, zoom, mappa errori
- Profilo Bambu Lab X2D + 2× AMS 2 Pro (8 slot), volume 256×256×260

**Da fare**
- Fase 6 calibrazione: generazione provini, inserimento misure, stima di `S` e `R∞`
- Verifica del 3MF in Bambu Studio 2.8.2.60 (serve una prova dell'utente)
- Costo nella ricerca per preservare dettagli e struttura (oggi SSIM è solo una metrica)
- Separazione soggetto/sfondo con modello AI locale (oggi euristica sul bordo)
- Pacchetto installabile Windows (PyInstaller), accelerazione GPU opzionale

**Problemi aperti**
- Senza TD reali l'errore previsto è alto (ΔE00 ≈ 12 sulla foto di prova): servono i dati dei filamenti
- Le stime di tempo e di spurgo usano ipotesi fisse, documentate nel report
