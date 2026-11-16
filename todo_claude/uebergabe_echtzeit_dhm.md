# Übergabe: Echtzeit-Off-Axis-DHM auf der GPU (OffAxisHolo)

Stand: 2026-11-13 · Zweck: Einstieg für eine neue Worksession ohne Vorwissen.
Zugehörige Dateien: `todo.txt` (Änderungsliste), `anweisung.txt` (Zielbeschreibung),
`holo_gpu_pipeline.py` (Referenz-Skelett), Claude Doc (Roadmap, siehe unten).

---

## 1. Worum es geht

Hannes (PhD, PhoenixD/Leibniz Universität Hannover; DHM/DOE/2PP) baut eine
Echtzeit-Rekonstruktion für Off-Axis-Hologramme, komplett auf der GPU:
Seitenband-Filterung → Hintergrundkompensation → Propagation (Angular Spectrum) →
Mittelung mehrerer Hintergrund- **und** Objekthologramme (gegen zeitliches Rauschen).

Der Mess-PC hat Kamera **und** GPU. Ein anderes Skript auf dem Steuer-PC spricht ihn
später per TCP/IP an. Offene Orchestrierungsfrage wurde mit einer Dienst-Architektur
beantwortet (siehe 3).

Arbeitsrolle in dieser Session: **Planungs-Head** – erst planen, nichts am Code ändern.

## 2. Was bisher entstanden ist

| Ergebnis | Ort / Status |
|---|---|
| Ausführliche Architektur-Zusammenfassung | im Chat |
| Roadmap + Änderungs-Todos + Entscheidungen D1–D8 + Risiken + 6 offene Fragen | Claude Doc: https://claude.ai/artifact/TBeMRT38kvooLK1BtxH8T1 (enthält Architektur- und Roadmap-Diagramm) |
| Referenz-Skelett `holo_gpu_pipeline.py` (xp-agnostisch NumPy/CuPy, Klasse `OffAxisPipeline`, Selbsttest) | an Hannes gesendet; lag nur im Sandbox-Arbeitsbereich der ersten Session |
| Numerische Prüfungen auf der CPU (Linearität, Kolbenphase, Präzision, Shift) | Ergebnisse siehe 3; Skripte nur in der Sandbox |
| `todo.txt` + `anweisung.txt` | als Dateien im Chat geliefert. **Noch NICHT in den Ordner OffAxisHolo geschrieben** (Verbindung zum Rechner war nicht erreichbar) |
| Durchsicht des Ordners `C:\Users\hanne\Documents\Seafile\phoenixd-robben\Code\OffAxisHolo` | read-only, nichts verändert |

## 3. Kernentscheidungen der Architektur

**Physik/Rechenkette**
- Rekonstruktion: Fenster → FFT → Seitenband (−1. Ordnung = `R*·O`) ausschneiden → **IFFT nur n×n**.
  Ausschnitt ändert den Pixelabstand: `dx' = dx·N/n` (muss in die Propagation).
- Alle Schritte sind linear → **erst mitteln, dann einmal kompensieren und propagieren**.
- Komplexe Phasoren mitteln, nie Phasen. Pro Frame **Kolbenphasenkorrektur** (maskiertes
  komplexes Skalarprodukt über eine Referenzregion, ohne Host-Sync). Ohne sie sinkt die
  Amplitude des Mittels mit `exp(−σ²/2)`.
- Kompensation in der Hologrammebene: `E_comp = Ū_obj · conj(Ub)/(|Ub|²+ε)`,
  `ε = eps_rel·max|Ub|²`, Startwert `eps_rel ≈ 1e-3` (Colomb et al., Opt. Express 14, 4300).
  `K = conj(Ub)/(|Ub|²+ε)` einmal vorberechnen.
- Mittelungsmodi: Block, Sliding (Ringpuffer, `S += neu − ältestes`), EMA (`N_eff=(2−α)/α`).
  Hintergrund: Block, danach fest. Rauschmodell: `σ² ≈ σ_obj²/N_obj + σ_bg²/N_bg`
  (Hintergrundboden).
- Propagation: bandbegrenzte ASM, `H(z)` gecacht; Padding-Trick optional (kostet 4× FFT-Fläche;
  bei kleinen z per Abschätzung der Randbreite `|z|·tan(asin(NA/n))/dx'` entscheiden).
- Präzision: complex64/float32 genügt (CPU-Test ~1e-6 rad), Reduktionen in float64.
  Auf echter GPU **noch nicht validiert**.
- Entfalten nicht im Standardpfad: `Fast 2D` nur CPU, `Least Squares` (DCT) GPU-fähig aber
  unvalidiert; zeitliches Entfalten nur fürs Live-Tracking.

**Dienst/Orchestrierung**
- Ein Prozess besitzt Kamera + CUDA-Kontext. Threads: Netzwerk, Akquisition, **ein** GPU-Worker.
  Beschränkte Warteschlangen, verworfene Frames werden gezählt und gemeldet.
- Zustandsautomat: `IDLE / BACKGROUND / READY / MEASURING / ERROR` (+ `LIVE`).
- Protokoll v1: ZeroMQ, JSON + Binärpuffer (dtype/shape-Header), nie pickle, Handshake mit
  Version + Konfig-Hash, asynchron: `measure → job_id`, dann `result/status/abort`.
  Kommandos: `hello, status, config, acquire_background, measure, result, abort,
  live_start, live_stop, shutdown`.
- Client-Bibliothek `dhmclient` (nur numpy + pyzmq) als eigenes Paket; Entscheidungslogik
  bleibt im Steuer-Skript.
- Ergebnis: float32 binär + JSON-Metadaten; Archivierung asynchron als `.zdc`.

## 4. Befund zum Ordner OffAxisHolo (Kurzfassung)

Gut und wiederverwendbar: Backend-Abstraktion NumPy/CuPy (`backend.py`), physikalisch richtige
Reihenfolge (Kompensation vor Propagation), bandbegrenzte ASM mit Padding-Trick, LS-DCT-Entfalten,
CPU/GPU-Paritätstests (`tests/test_backend.py`), Presets, Simulation.

**Nicht echtzeittauglich** (Details in `todo.txt`, Abschnitt A):
1. Objektinstanzen pro Bild (Hologram, ReferenceHologram, HologramProcessor, Plotter).
2. Kein Caching: `angularSpectrum` und `circularMask` bauen Tabellen jedes Mal neu; Hintergrund wird
   jedes Mal neu rekonstruiert.
3. IFFT immer über das volle N×N; fftshift/roll auf voller Fläche.
4. Host-Syncs im Pfad (`bool(xp.min(holo)<0)`, `locate_order` mit `to_numpy`, pro Hologramm).
5. Keine Mittelung, keine Kolbenphase, keine Referenzregion.
6. `Fast 2D`-Entfalten erzwingt GPU→CPU→GPU.
7. `import offaxisholo` zieht tkinter, matplotlib, skimage, cv2 nach.
8. tkinter-Dialoge in `ReconstructionPipeline.__init__`.
9. `backend._state` global, nicht thread-sicher.
10. Kein Kamera-/Dienst-/Protokollcode.
11. Kompensations-eps (`finfo.eps·max|Eref|²`) nur Division-durch-Null-Schutz.

Wichtige Fakten aus dem Code:
- `locate_order` sucht nur auf der linken Spektrumshälfte (negatives fx) → liefert `O·conj(R)`.
  Im Test ist die Ordnung bei `(−128, −128)`.
- Preset Zeiss 63x: `dcRadius=304`, Pixelpitch 0,0869 µm (Objektebene), λ=0,6749 µm,
  `propagationDistance=1.1e-7 m`. Notes nennen einen gemessenen Fokus von ca. −2,6 µm → Widerspruch offen.
- Preset-JSON hat doppelte Schlüssel (`xPixelSizeUm` vs. `pixelPitch`, `wavelengthUm` vs. `wavelength`)
  und `"name": "Zeiss 63X"` vs. Schlüssel `"Zeiss 63x"`.
- pyproject: sehr junge Mindestversionen (numpy ≥ 2.4.5, scipy ≥ 1.17.1); `cupy-cuda12x` als Extra –
  CUDA-Version des Mess-PCs ungeklärt.
- Nicht inhaltlich gelesen: `io/structure.py`, `plotter_class.py`, `simulate.py`, `reconstruction/__init__.py`,
  Material-/20x-Presets, `tests/test/*`, `.gitignore`, `data/`.

## 5. Roadmap (5 Phasen mit Gates, ca. 3,5–6 Wochen)

1. **GPU-Kern** (neues Paket `realtime/`): Demod mit Ausschnitt, Kolbenphase, Hintergrund + `K`,
   Akkumulatoren, Kompensation + Propagation je Ergebnis, Parität zu `HologramProcessor`, Tests,
   Benchmark auf dem Mess-PC.
2. **Kamera**: `FrameSource`, `SimCamera`, `FileReplayCamera`, Pinned-Ringpuffer, echter Treiber.
3. **Dienst + Protokoll**: Zustandsautomat, Worker, ZeroMQ-Server, Live-Modus, Archivierung.
4. **Anbindung**: `dhmclient`, Beispielskript, Integration mit dem echten Steuer-Skript, Referenzmessung.
5. **Betrieb**: Autostart/Watchdog, Logs, Versionsstand in Ergebnissen, Runbook, Langzeittest.

## 6. Nächste Schritte (empfohlene Reihenfolge)

1. `todo.txt` und `anweisung.txt` in den Ordner `OffAxisHolo` legen (siehe 8).
2. Blocker B1 (Lazy-Imports), B3 (Abhängigkeiten aufteilen), B4 (Thread-Regel fürs Backend).
3. **B6 auf dem Mess-PC:** `pytest tests/test_backend.py -v` und
   `python tests/benchmark_backend.py 1024 2048` auf echter CUDA-Hardware laufen lassen.
4. Phase 1 umsetzen (C1–C11), danach messen (C12–C14) und erst dann optimieren.
5. Parallel ohne Hardware: Protokoll, Zustandsautomat, SimCamera, Tests.

## 7. Offene Fragen an Hannes

1. Kamera / SDK?
2. GPU, Speicher, CUDA-Version auf dem Mess-PC?
3. Bildgröße N, Ausschnitt n, Zielbildrate, Ziel-Latenz?
4. Ergebnis in Hologramm- oder Fokusebene? Welches z (0,11 µm Preset vs. ca. −2,6 µm gemessen)?
5. Propagationsmedium (Luft/Harz) und Vorzeichenkonvention (Phase/Höhe)?
6. Entfaltung/Höhenkarte im Dienst oder im Client?
7. Netzwerk: Authentifizierung, ein oder mehrere Clients?
8. Wie lange hält ein Hintergrund, braucht es Drift-Überwachung?
9. Nutzt das Steuer-Skript den Live-Strom (Regelung) oder nur Einzelmessungen?

## 8. Praktische Hinweise für die nächste Session

- Ordner-Zugriff läuft über die Desktop-App-Verbindung. Letzter Stand: Verbindung nicht erreichbar,
  deshalb liegen `todo.txt`/`anweisung.txt` noch nicht im Projektordner.
- Den Ordnerstamm **nie rekursiv listen** (`.git` füllt die Ausgabe); gezielt `src/offaxisholo` und `tests` listen.
- Sandbox-Dateien (`holo_gpu_pipeline.py`, Prüfskripte) gehen mit der Session verloren; das Skelett liegt als
  gesendete Datei beim Nutzer.
- Der Ordner liegt in einem Seafile-Sync-Verzeichnis mit `.git`, `.venv`, `build/` – Konfliktrisiko, siehe `todo.txt` B5.
- Regeln für die Umsetzung (aus `anweisung.txt`): Offline-Bibliothek bleibt unverändert lauffähig und ist
  Golden Reference; neuer Code in neuen Modulen; Test vor Optimierung; nichts löschen; keine erfundenen
  Leistungszahlen – messen.
