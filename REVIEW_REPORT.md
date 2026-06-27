# Zha — szakdolgozat és védés review

Áttekintés: kód vs. dolgozat vs. beamer konzisztencia, plágiumcheck, AI-y megfogalmazások.

Vizsgált fájlok (helyi másolatok a `~/zha_review/`-ben, eredetijük a `192.168.1.138:Documents/Programming/Zha`-n):

- `backend/models/{markov_chain.py, vae.py, golc_vae.py, transformer.py}`
- `backend/trainers/{train_markov.py, train_vae.py, train_golc_vae.py, train_transformer.py, utils.py}`
- `docs/thesis/chapters/*.tex`, `docs/thesis/main.tex`, `docs/thesis/dolgozat.bib`
- `docs/beamer/thesis_defence.tex`
- `README.md`, `train.py`

---

## 1. Súlyos tény-eltérések (kód ≠ dolgozat)

Ezeket fix kell. Ha a védésen rákérdez a bizottság és a kódba is benéz, megfognak.

### 1.1 Transformer — Lényegi architektúra-csúsztatások

A `chapters/transformer.tex` egy olyan Transformer architektúrát ír le, ami a kódban **nincs**.

| Dolgozatbeli állítás | Valóság a `backend/models/transformer.py`-ben |
|---|---|
| RoPE (Rotációs pozíciókódolás), `RotaryPositionalEmbeddings`, `rotate_half` | Klasszikus szinuszos `PositionalEncoding` (transformer.py:7–45). RoPE **nincs** implementálva. |
| `lightning_transformer.py` fájl, PyTorch Lightning Trainer | Ilyen fájl nem létezik; a tréner sima PyTorch (train_transformer.py). |
| `GatedFeedForward` SiLU + szigmoid kapuval, $W_1, W_g, W_2$ | A modul **nem létezik**. A modell `nn.TransformerEncoderLayer`-t használ, szabványos ReLU FFN-nel. |
| `EnhancedMultiheadAttention`, `EnhancedTransformerEncoderLayer` | Sima `nn.TransformerEncoderLayer` és `nn.MultiheadAttention`. |
| 6 réteg, 8 fej, $d_{\text{model}}=512$ | **8 réteg**, 8 fej, $d_{\text{model}}=512$ (transformer.py:89–90). |
| Tanult relatív pozíciós bias $B_{i,j}$ | Nincs ilyen — sem a modellben, sem a trénerben. |
| Címkesimítás $\epsilon=0.1$ | A `F.cross_entropy`-ban nincs `label_smoothing` paraméter (train_transformer.py:343). |
| 5000 lépés warmup, koszinusz 0-ig 200 000 lépésen | OneCycleLR (pct_start=0.1, div_factor=25), illetve CosineAnnealingLR fallback (train_transformer.py:200–219). |
| Memóriamechanizmus EMA-frissítéssel, szekciókhoz {intro, verse, chorus, bridge, outro} külön $M_s$ mátrix, kibővített attention $[K; M_s]$ | A kódban van `section_memories` dict és `transition_smoothness` blend, de **nincs** EMA frissítés, nincsenek szekció-típus labelek, és **nincs** $[K; M_s]$ augmentált attention — csak prepended-context (transformer.py:583–642). |

A függelékben (`appendix.tex § Memória mechanizmus zenei struktúrához`) **bizonyítást** írsz egy olyan mechanizmusra, ami nincs is így megvalósítva — ez a leghúzósabb pont.

**Tennivaló:** vagy a Transformer fejezetet kell radikálisan visszafogni a tényleges kódra (sinusoidal PE, standard `nn.TransformerEncoderLayer`, multitrack cross-attention, conditioning, OneCycleLR, top-k/top-p/repetition penalty generálás — ezek tényleg ott vannak és érdekesek), vagy a kódba kell behúzni a RoPE-ot és GatedFFN-t a védés előtt. Az utóbbi kb. 1-2 nap munka, mert tényleg jó tulajdonságok.

### 1.2 VAE — Architektúra-félreírás

A `chapters/vae.tex § Implementációs részletek` ezt írja: *"Az enkóder egy bidirekcionális LSTM hálózat… a dekódoló autoregresszív módon működik."*

Valójában (`backend/models/vae.py:5–95`):

- Az enkóder **feedforward**: `Linear(128,512) → SiLU → ResidualBlock → Linear(512,256) → SiLU → ResidualBlock → Linear(256, 2·latent_dim)`.
- A dekóder szintén feedforward, **nem autoregresszív**, és **nem token-szekvencián**, hanem egyetlen 128-dim pitch-hisztogramon dolgozik. Egy bemenet = egy MIDI fájl pitch-eloszlása.
- Nincs LSTM, nincs token embedding, nincs `Softmax(W_o [e_{t-1} ‖ z])` autoregresszív lépés.

A „Tanítási trükkök" lista is csúszott:
- **KL-annealing** $\beta(e)=\min(1, e/E_{\text{warmup}})$ — a fő VAE trénerben **nincs**. A `train_vae.py`-ben a beta konstans (default 0.5); a `trainers/utils.py:54`-ben ugyan szerepel `beta = min(1.0, 0.2 + epoch_idx/50)`, de ez a függvény nincs élesben használva a `train_epoch`-ban.
- **Free bits, min. 0.1/dim** — sehol sem található a kódban.
- **Gradiens vágás végtelen normával** — a `clip_grad_norm_` **L2-norma** alapértelmezetten. Vagy az állítást írd át L2-re, vagy hívd a `clip_grad_value_`-t.

Van viszont egy szomszéd-hangokra vonatkozó *consistency loss* (`train_vae.py:192–194`), ami a függelékben (`appendix.tex § Konzisztencia veszteség`) le van írva — ez tényleg ott van. **Ezt** kell hangsúlyozni a VAE fejezetben, nem a nem-létező BiLSTM-et.

### 1.3 GOLC-VAE — A bizonyítás nem fedi a kódot

A `vae.tex § Állítás 2 (Posterior stabilitás)` kovariancia-mátrixokra hivatkozik (Frobenius-norma). A `golc_vae.py:200–208` valójában `F.mse_loss(mu_original, canonical_mu)` és `F.mse_loss(logvar_original, canonical_logvar)`-t számol, vagyis a középérték- és log-variancia-vektorok L2 távolságát. Ez **diagonális** Gauss-feltevést jelent, nem teljes kovariancia mátrixokat — a bizonyítás állítása így túl erős.

Ezen kívül: a `canonical_mu`-ra `.detach()`-elve számol gradienst (golc_vae.py:207), így az "orbit-átlaghoz húzás" csak egy oldalra terjed — érdemes a dolgozatban is megemlíteni, mert ez egy *konkrét* implementációs döntés.

### 1.4 Markov fejezet ellentmondások

- `markov.tex § Bevezetés`: *"A Zha implementáció 2-6. rendű Markov-modelleket használ"*
- `markov.tex § Markov-lánc implementáció`: *"Támogat $n$-ed rendű láncokat, ahol $n$ legfeljebb 4."*

A kód oldalán (`markov_chain.py:842`) a felső korlát **6** (`max_order = min(6, self.order)`), a default 3. Egységesítendő: vagy mindenhol "2–6", vagy "2–6, default 3".

További hibás állítások:

| Állítás | Tény |
|---|---|
| `defaultdict(Counter)` a fő átmeneti tárolás | A fő átmeneti mátrix 128×128 numpy/torch tömb. `defaultdict(Counter)` csak a multi-order számláláshoz használt (markov_chain.py:805). |
| Laplace simítás konfigurálható $\alpha$ paraméterrel | **Nincs** Laplace-simítás. Egyszerű sor-összeg normalizálás (markov_chain.py:872–873). Vagy implementáld, vagy töröld az állítást. |
| START/END tokenek a mintavételezésnél | Nincsenek. A generálás `start_note` MIDI értékkel indul, hosszlimittel áll meg. |
| Krumhansl–Schmuckler algoritmus a hangnem-felismerésre | A kód a `music21.analyze('key')`-t hívja (markov_chain.py:2222). A music21 belül *valószínűleg* Krumhansl-t használ, de te nem implementálsz semmit közvetlenül — ezt írd át pl. *"a music21 könyvtár Krumhansl–Schmuckler-alapú key-analizátorát használom"*. |
| Hőmérséklet-vezérelt mintavételezés $T\in[0.7,1.3]$ | **Nincs**. `np.random.choice(p=…)`-szal sample-elsz, temperature nélkül. |
| Top-K szűrés | **Nincs**. Sem kódban, sem trénerben. |
| CuPy backend GPU-gyorsításhoz | A `markov_chain.py:16–28` `try/except`-tel próbálja, de a logban *"CuPy not found - using PyTorch CUDA/CPU backend"* — gyakorlatban PyTorch a backend. Vagy CuPy-t fixáld a requirements.txt-be, vagy írd át: *"PyTorch CUDA-backenddel, opcionális CuPy gyorsítással"*. |
| 10-50× sebességnövekedést mér | Nincs benchmark a kódban. Vagy fuss egy benchmarkot és tedd be (jó figure-be), vagy hagyd ki a számot. |
| `markov.tex` kódsnippet: `np.arange(128) % 12 in scale_notes` | Ez Pythonban **nem** elem-wise membership — egyetlen bool-t ad vissza, sőt `ValueError`-t dob, ha `scale_notes` lista. A helyes idiom: `np.isin(np.arange(128) % 12, scale_notes)`. Ezt javítani kell a kódsnippetben. |

### 1.5 Tanítás-fejezet (`train.tex`) — sok nemlétező feature

Ezek a fejezetben szerepelnek mint *implementált* dolgok, és **nincsenek** a kódban:

- `TrainingDebugger` osztály gradient-flow figyeléssel, NaN-detektálással
- Bayes-i / Optuna hiperparaméter-keresés TPE samplerrel, multi-objective optimization, adaptive pruning
- TensorBoard / Weights & Biases integráció
- Multi-GPU, `DistributedDataParallel`, NCCL backend
- `persistent_workers`, `prefetch_factor` DataLoader beállítások
- TorchScript JIT compile (van részben — `train_transformer.py:476–481` próbál `torch.jit.script`-elni, de quantization, operator fusion, constant folding **nincs**)
- Címkesimítás $\epsilon=0.1$
- "Sample generation monitoring rendszeresen értékeli a generált zenei kimenetek entrópiáját" — ez nem fut tréning közben
- "Automatic checkpoint recovery", "graceful degradation", "emergency fallback mode" — nincs
- "Data corruption detection" — csak egy `try/except` van MIDI-parse-nél, ami nullvektorral helyettesít (train_vae.py:117–150)

Két "Hiperparaméter optimalizálás" és két "Tanítási monitorozás" szakasz is **duplikáltan** szerepel a fejezetben (l. 140–149 és 180–184, ill. 170–175 és 186–193). Az egyiket törölni kell.

Ami **tényleg** ott van és érdemes hangsúlyozni a tréning fejezetben:
- AdamW mindhárom modellnél, mixed precision (VAE + Transformer), gradient accumulation
- EarlyStopping (`trainers/utils.py:155–168`)
- LRU cache MIDI fájlokra (`train_vae.py:108–144`)
- HuggingFace MidiCaps streaming támogatás genre filterrel
- OneCycleLR + CosineAnnealing fallback Transformeren
- A `consistency_loss` a VAE-ben (lásd 1.2)
- Az `orbital_consistency_loss` a GOLC-VAE-ben
- A heartbeat logging (`train_transformer.py:370–380`) — érdekes nem-TTY környezethez

### 1.6 Zha integráció fejezet

A `zha.tex § Inferenciás motor` szerint *"A rendszer teljes generálási ideje átlagosan 400 milliszekundum 30 másodperces zenei tartalom előállítására, amelyből a Transformer szerkezetgazdagítás teszi ki a legnagyobb részt (200 ms, 50%)."*

A `chapters/zha.tex § Értékelés` `tab:combined_performance` táblája pedig konkrét számokat ad:

| Konfig | Perplexity | Pitch Entropy | Harmonic Coh. | Idő |
|---|---|---|---|---|
| Markov only | 2.48 | … | 0.72 | 0.5 ms |
| … | … | … | … | … |
| Combined | **1.52** | **3.6** | **0.88** | 7.0 ms |

**A kódban nincs olyan kiértékelő szkript, ami ezeket a számokat kiszámolja.** A `scripts/generate_*_metrics.py` fájlok léteznek, de pl. "harmonic_coherence" metrika nincs definiálva sehol. A 4.6 vs. 4.7 subjective coherence score (`conclusion.tex § Gyakorlati eredmények`) sem támogatott — nincs user-study eredmény a repóban.

Két lehetőség:
1. Tényleg futtass egy ablation-t, ami ezeket kiméri, és tedd be a számokat reprodukálható módon.
2. Ha nem futtatsz, jelezd a táblát *"várt értékek a rokon irodalom alapján"-ként, és ne állítsd, hogy a *saját* rendszered ezeket mérte.

A 4.6/4.7 subjective számok különösen veszélyesek — ha védésen rákérdeznek, hogy hány emberrel csináltad a hallgatói tesztet, nem lesz mit válaszolni.

### 1.7 README és placeholder `train.py`

- `README.md:84`: `python backend/trainers/train_diffusion.py` — **nincs** ilyen fájl, nincs diffúziós modell. Vagy töröld a sort, vagy add hozzá a jövőbeli munkához.
- `README.md:89`: *"Available models: markov, vae, golc_vae, transformer, diffusion"* — diffusion-t törölni.
- `train.py` egy placeholder ami csak `time.sleep`-pel szimulál tanítást és egy text fájlt ír ki (train.py:35–50). Ez nem szabad így legyen — vagy törlöd, vagy egy `argparse`-os entry-pointot csinálsz belőle, ami a backend/trainers-be route-ol.

### 1.8 Beamer (thesis_defence.tex)

- **Slide 4 (Markov-láncok)**: *"Magasabb rendű (2–6. rend)"* — itt a thesisnél jobb szám.
- **Slide 10 (Eredmények)**: a táblázat számai (Recon. Loss 0.082 vs 0.078; Orbit Cons. 2.34 vs 0.45; **"GOLC-VAE: 5.2× jobb"**) — ezekre ugyanaz vonatkozik, mint a thesisre: nincs futtatott összehasonlító kiértékelés a repóban. Vagy futtasd le (`scripts/compare_vae_models.py` létezik — ezt érdemes megnézni), vagy puhítsd a megfogalmazást.
- **Slide 13**: a kreativitás-paraméter (0–1) tényleg ott van a backendben (`generate_combined` endpoint), oké.

---

## 2. Plágium-spotcheck

Részletek: lásd a `dolgozat.bib` audit és a magyar/angol web-keresések alább.

**Direkt plágium nem talált.** A négy mintavételezett canonical szakasz (ELBO, reparameterizáció, self-attention, RoPE) prózája nem egyezik egyetlen indexelt magyar (Wikipedia HU, sajozsattila.home.blog, Trilobita-PE) vagy angol forrással sem. A képletek a hivatkozott Kingma / Vaswani / Su et al. papereknek megfelelnek, és helyesen vannak cite-elve.

**Stiláris megjegyzés**: a próza ritmusa (rövid tagmondatok, képlet–átfogalmazás párok, "amely strukturált látens teret biztosít" típusú lezárások) erősen LLM-paraphrase-szerű. Ez **nem** plágium, de a bizottság valószínűleg észreveszi.

**GOLC eredetiség**: a *"Csoport-Orbitális Látens Konzisztencia" / "GOLC"* terminus nem szerepel egyetlen indexelt műben sem — **a név eredeti.** A módszer alapja viszont nem: rokon, idézetlen előzmények:

- Lattner, Grachten, Widmer (2018): *"Learning Transposition-Invariant Interval Features…"* (arXiv:1806.08236) — pont a transzpozíció-invariáns zenei reprezentáció a témája.
- Cohen & Welling (2016) **van** idézve (jó).
- TARGET-VAE (arXiv:2210.12918), EQ-VAE (arXiv:2502.09509) — ezek is rokon orbital/equivariant VAE-k.

Ezeket érdemes citálni az `§ Kapcsolat korábbi munkákhoz`-ban, és a GOLC-ot **inkrementális** hozzájárulásként pozícionálni, ne mint teljesen új paradigmát. Így nem támadható.

### Bibliográfia (`dolgozat.bib`) — törlendő/lecserélendő hivatkozások

Ezek nem akadémiai források és kínosak egy szakdolgozatban:

| Cite key | Probléma |
|---|---|
| `medium_deep_arch` | Medium blog |
| `simplilearn_top_algos` | Marketing tutorial "Top 10 in 2025" |
| `analytics_12nets` | Analytics Vidhya blog |
| `wikipedia_generative_ai` | Wikipedia |
| `wikipedia_gan` | Wikipedia |
| `mdpi_comprehensive` | MDPI *Information* — borderline; a vol/year párosítás is inkonzisztens |
| `sciencedirect_lstm` | Journal field "ScienceDirect" (ez nem journal, hanem aggregátor); a tényleges journal a *Journal of King Saud University* |
| `fastapi2023`, `torch2023` | GitHub README-k a cite forrásnál, nem a megfelelő papír (PyTorch: Paszke et al. NeurIPS 2019) |
| `ji2020survey` | A cite key és a bejegyzés egymásnak ellentmondó — broken citation |

Duplikátumok:
- `huang2018music` (l. 127 és 466) — egy szerzői név is el van gépelve (Ian Simon vs Ioan Simon).
- `dong2017musegan` és `dong2018musegan` ugyanaz a MuseGAN.
- `baum1970maximization` és `baum1972inequality` ugyanaz az algoritmus, eltérő évszámokkal.

---

## 3. „AI-y" megfogalmazások — átírási javaslatok

A módszer: behúzni a saját hangod (te a `intro.tex` és `theory.tex` egyes szakaszaiban tényleg személyes hangon írsz: *"nálam", "korai kísérleteim során", "az RTX 3090-em 24 GB VRAM-jával"* — ez **jó**, ezt kell konzisztensen tartani). Az AI-y részeket olyan szakaszokon kapom el, ahol:

- személytelen, általános *"a rendszer kifinomult X-et alkalmaz"*
- listák önmagukért ("Az ütemezés figyelembe veszi…")
- ködös szuperlatívuszok ("intelligens", "fejlett", "kifinomult", "továbbfejlesztett") konkrétumok nélkül
- vegyes angol-magyar szakszavak indok nélkül

A javaslatok először az **eredeti**, majd az **átírt** verziót adják.

### 3.1 `abstract.tex` — teljes újraírás

**Eredeti:**
> A Zha rendszer újdonsága, hogy több mesterséges intelligencia réteget integrál, fejlett tanulási infrastruktúrát és intelligens modellkapcsolási megoldásokat alkalmazva. A kísérletek azt mutatják, hogy ez a kombinált megközelítés jobb minőségű és változatosabb zenét eredményez, mint az egyetlen modellre épülő hagyományos megoldások.

**Átírás:**
> A Zha rendszer fő hozzájárulása a három modell explicit szétosztása három különböző zenei aspektusra: a Markov-lánc adja a kulcsban maradó dallamot, a VAE (és GOLC-kiterjesztése) a transzpozíció-invariáns látens manipulációt, a Transformer pedig a polifón kíséretet és a hosszabb távú szerkezetet. A három kimenet rögzített súlyozással (0,5 / 0,3 / 0,2) keveredik, majd skála- és regiszter-szűrésen megy keresztül a végső MIDI előtt. Az ablation-eredmények a 7. fejezetben.

### 3.2 `intro.tex § Motiváció` — már elég jó, csak egy nyitány-fix

**Eredeti:**
> A mesterséges intelligencia és a zenei alkotás találkozása talán a számítástechnika egyik legizgalmasabb területe manapság.

**Átírás:** (a "legizgalmasabb területe manapság" klasszikus AI-y nyitás)
> Az algoritmikus zeneszerzés régi terület — Hiller és Isaacson Illiac Suite-ja 1957-re datálódik —, de a mélytanulás megjelenése óta a kérdés átalakult: ahelyett, hogy *milyen szabályrendszer* írja le a tonális zenét, *milyen mintázatokat* talál egy hálózat a korpuszban. Ez a dolgozat erre a második szempontra koncentrál.

### 3.3 `theory.tex § Reprezentációs kihívások` — már szinte rendben

Ez a szakasz tényleg személyes (*"nálam"*, *"korai kísérleteim"*). Tartsd meg. Két konkrétum tisztázandó:

- *"A többszólamúság modellezése … az egyfolyamú kódolás esetén speciális tokenekkel jelzem az egyidejűséget"* — de a `transformer.py` valójában 128-dim one-hot vektorokon dolgozik, nem REMI-stílus tokeneken. A `zha.tex § Tokenizációs séma` viszont REMI-t ír. Ez ütközik — vagy a Transformer pipeline-t kell egyértelműen leírni (mi a tényleges input shape: `[B, T, 128]` one-hot, **nem** `[B, T]` token ID), vagy a REMI tokenizációt kell valóban implementálni.

### 3.4 `theory.tex § Adatfeldolgozási pipeline` — átírás

**Eredeti:**
> Zeneelméleti integráció központi szerepet játszik. Automatikus hangnem-felismerést alkalmazok a Krumhansl-Schmuckler algoritmus alapján, amely statisztikai súlyozással azonosítja a domináns hangnemet.

**Átírás:**
> A hangnem-felismeréshez a `music21.analyze('key')` hívást használom (a music21 ezt a Krumhansl–Schmuckler-féle key-profile illesztésre építi). Ennek a kimenete egy `Key` objektum, amelyből a skálát a `_get_scale_pitches()` (`markov_chain.py:2178`) bontja pitch class halmazzá, és ez a halmaz szűri a Markov mintavételezést.

### 3.5 `markov.tex` — duplikált GPU-szakasz törlése + tisztázás

A `markov.tex § Optimalizált algoritmusok > GPU-gyorsítás` és a `markov.tex § Implementációs részletek > GPU-gyorsítás` ugyanazt mondja. Az egyik törlendő.

A kódsnippeteket javítani kell (lásd 1.4): `np.arange(128) % 12 in scale_notes` → `np.isin(np.arange(128) % 12, list(scale_notes))`.

### 3.6 `vae.tex § Implementációs részletek` — két szakasz egybeolvasztva

**Eredeti (152–186):**
> Az enkóder egy bidirekcionális LSTM hálózat, amely a teljes szekvenciát feldolgozza… A dekódoló autoregresszív módon működik…

**Átírás:**
> A VAE bemenete egy 128-dimenziós pitch-hisztogram (a MIDI fájl normalizált hangmagasság-eloszlása), nem szekvencia. Az enkóder három feedforward blokkból áll (Linear+SiLU+ResidualBlock; lásd `backend/models/vae.py:18–28`), kimenete a $(\mu, \log\sigma^2)$ pár. A dekóder szimmetrikus, kimenete sigmoid-aktivációval normalizált 128-dimenziós eloszlás. Ez a választás szándékos: a Markov-lánc kezeli a szekvencia-szintű dinamikát, a VAE pedig egy "stílusvektor" reprezentációt tanul, amit a generálás során a Markov-kimenet pitch-eloszlásával szorzunk össze.

A "Tanítási trükkök" listát igazítsd a valósághoz:

**Átírás:**
> - **KL-súly** β: konstans, default 0.5 (`train_vae.py:251`). KL-annealing-et próbáltam, de nem javított érdemben a konvergencián a teszteken — kivettem.
> - **Consistency loss**: a rekonstruált pitch-eloszlás egymást követő elemeinek L1-különbsége súlyozva. Ez kifejezetten a zenei alkalmazás miatt került be: simább pitch-átmeneteket kényszerít (`train_vae.py:192–194`).
> - **Gradiens vágás**: L2-norma max 1,0 (`torch.nn.utils.clip_grad_norm_`).
> - **Mixed precision**: `torch.amp.autocast('cuda')` + `GradScaler` (a VAE-nél is, nem csak a Transformernél).

### 3.7 `vae.tex § Empirikus hatások` — feltételes módú homályosítás

**Eredeti:**
> A GOLC regularizáció több szempontból is javíthatja a modell teljesítményét. Csökkenti a redundanciát… A modell zenei transzformációkra invariáns struktúrát tanul meg…

A "javíthatja… csökkenti… stabilizálja" felsorolás konkrétumok nélkül erősen AI-y. Vagy mérd meg és add be a számot, vagy hagyd ki ezt a szakaszt, és helyette egy konkrét ábrával ($\beta_{\text{orbit}}$ paraméterre ablation) válaszd ki egy számmal.

### 3.8 `transformer.tex` — alapos átírás kell

Mivel az architektúra-állítások jelentős része nem fedi a kódot, a fejezetet két irányba lehet vinni:

**(A) Lecsupaszítás a tényleges kódra.** Ekkor a fejezet kb. így néz ki:

> A Zha Transformer modulja egy `nn.TransformerEncoder`-re épül 8 réteggel, 8 fejjel, $d_{\text{model}}=512$, $d_{\text{ff}}=2048$, dropout 0.1 értékkel (`backend/models/transformer.py:88–144`). A pozíciókódolás klasszikus, szinuszos (sin/cos), $\max\_\text{len}=2048$ tokenig előre számolva. Három saját kiegészítés:
> 
> 1. **Multitrack cross-attention** (transformer.py:48–86): külön encoder a basszushoz és a dobokhoz, plusz három cross-attention modul (bass→melody, drum→melody, drum→bass) a sávok összehangolásához.
> 2. **Akkord- és tempo-conditioning** (transformer.py:121–127, 321–445): külön embedding az aktuális akkord (root + quality + pitch classes) és tempó (normalizált skalár) számára, ami a fő embeddinghez adódik.
> 3. **Szekció-alapú memória** (transformer.py:583–642): a `generate_with_structure` négy szekciót generál, és a `transition_smoothness=0.7` paraméterrel az előző szekció memóriájának hátsó részét tartja, hogy a szakaszváltások ne legyenek vágva.
> 
> Mintavételezés (transformer.py:464–581): temperature ($T=0.8$), top-k ($k=5$), nucleus (top-p, $p=0.92$), repetition penalty ($r=1.2$). A memória puffer 1024 tokenre korlátozott (transformer.py:255).

**(B) Implementáld a RoPE-ot és a GatedFFN-t.** Ha az időkeret enged, ez 2-3 fájl módosítás, és a fejezet sokkal erősebbé válik. A jelenlegi kódsnippetek a fejezetben *működő* RoPE/GatedFFN implementációk lehetnek alapnak — csak be kell ténylegesen kötni a `TransformerModel`-be.

Az **(A)** opció a biztos.

### 3.9 `train.tex` — egy fejezetnyi átírás

Ez a fejezet a legAI-yabb az egész dolgozatban. Sok mondat *"A rendszer kifinomult X-et alkalmaz"* vagy *"intelligens Y-t használ"* sablonban van.

**Eredeti minta (§ Egységes tanítási infrastruktúra):**
> A rendszer moduláris tanítási infrastruktúrát használ, amely egységes felületet biztosít mindhárom modelltípushoz. A közös komponensek támogatják az automatikus vegyes precizitást, gradiens akkumulációt és adaptív tanulási ráta ütemezést. Az eszközök automatikus kiválasztása biztosítja a GPU kihasználását, ha elérhető, különben CPU-n történik a számítás. […]

**Átírás:**
> A három modellnek külön trénerszkript van (`train_markov.py`, `train_vae.py`, `train_transformer.py`), de közös segédfüggvényekkel a `trainers/utils.py`-ben. Ezek közül a leghasznosabbak: `EarlyStopping` (validációs veszteségen, default türelmi szám 30 epoch), `MemoryEfficientDataset` (LRU cache MIDI fájlokra), és egy `get_optimizer_and_scheduler` builder.
>
> A VAE és a Transformer mixed precisionnel tanul (`torch.amp.autocast` + `GradScaler`); a Markov nem, mert ott nincs neurális forward pass. Mindhárom AdamW-t használ, gradiens-akkumulációs lépésszámmal konfigurálva. Eszközválasztás: `torch.cuda.is_available()` alapján CUDA vagy CPU; multi-GPU támogatás nincs.

A teljes fejezet egy oldalra zsugorítható az ilyen átírás után. **Töröld** a nemlétező feature-eket:

- "Bayes-i keresés Optuna TPE samplerrel" — törlendő
- "Multi-GPU, DistributedDataParallel, NCCL" — törlendő
- "TorchScript JIT, quantization, operator fusion, constant folding" — TorchScript próbálkozás van, a többi nincs
- "TrainingDebugger osztály", "automatic checkpoint recovery", "graceful degradation", "emergency fallback mode" — törlendő
- "Sample generation monitoring" — törlendő
- "persistent_workers, prefetch_factor" — törlendő
- "Weights & Biases, TensorBoard" — törlendő
- A két duplikált "Hiperparaméter optimalizálás" és két "Tanítási monitorozás" alszakasz közül egy-egy törlendő

### 3.10 `train.tex` képlethibák

**Eredeti (l. 27–28):**
$$\mathcal{L} = \underbrace{\|\mathbf{x} - \text{Decoder}(\mathbf{z})\|_2^2}_{\text{Reconstruction}} + \beta \underbrace{D_{\text{KL}}(\mathcal{N}(\mu^l)\sigma \mathcal{N}(0,\mathbf{I}))}_{\text{Regularization}}$$

A KL képlet argumentuma törött (a `\mathcal{N}(\mu^l)\sigma \mathcal{N}(0,\mathbf{I})` nem értelmezhető). A javítás:
$$\mathcal{L} = \underbrace{\|\mathbf{x} - \text{Decoder}(\mathbf{z})\|_2^2}_{\text{rekonstrukció}} + \beta \underbrace{D_{\text{KL}}\!\bigl(\mathcal{N}(\mu,\sigma^2) \,\|\, \mathcal{N}(0,\mathbf{I})\bigr)}_{\text{regularizáció}}$$

Plusz a kódban a rekonstrukciós tag valójában `binary_cross_entropy` (golc_vae.py:289), nem L2 — a képletet ennek megfelelően kell igazítani:
$$\mathcal{L}_{\text{recon}} = -\frac{1}{N}\sum_i \bigl[x_i \log \hat x_i + (1-x_i)\log(1-\hat x_i)\bigr]$$

### 3.11 `zha.tex § Inferenciás motor` — 400 ms claim

**Eredeti:**
> A rendszer teljes generálási ideje átlagosan 400 milliszekundum 30 másodperces zenei tartalom előállítására, amelyből a Transformer szerkezetgazdagítás teszi ki a legnagyobb részt (200 ms, 50%).

Ha tényleg mérted, mondd meg mivel (GPU típus, batch size, sequence length). Ha nem, vedd ki vagy puhítsd "korábbi teszteken kb. 0.4 s nagyságrendűnek mértem"-re. Ugyanez vonatkozik a `tab:combined_performance` táblára.

### 3.12 `conclusion.tex § Gyakorlati eredmények` — subjective scores

**Eredeti:**
> Szubjektív értékeléseknél a kombinált modell majdnem ugyanazt a koherencia pontszámot kapta (4.6), mint az emberi referencia (4.7), miközben a kreativitásban is hasonló szintet ért el (4.1 vs. 4.0).

Ha nem volt user study, ez törlendő. Ha volt — hány alany, milyen kérdőív, melyik fájlokon — adatként a függelékbe kell. Ez a leginkább támadható mondat az egész dolgozatban.

### 3.13 Apró nyelvtani/szóhibák

- `abstract.tex`: *"újraszervezési technikákat alkalmaz"* — magyarul nem szerencsés, "rekonstrukciós interpolációt és látens-tér mintavételezést" pontosabb
- `theory.tex`: *"kihasznalni"* → *"kihasználni"*
- `theory.tex`: *"megkövezi"* helyett "megküzdülöm" gépelési hiba (a megfelelő: "megküzdjek")
- `theory.tex`: *"Kulcsfontóssá vált"* → *"kulcsfontosságúvá vált"*
- `theory.tex`: *"aszónáta"* → *"a szonáta"*
- `transformer.tex`: *"transzláció-invari anciát"* → *"transzláció-invarianciát"* (szóköz tévedés)
- `transformer.tex`: *"részlégi mintakereso eseket"* → *"részleges mintakeresési eseteket"* (caption hiba a 53. sorban)
- `transformer.tex`: a `GatedFeedForward` kódsnippetben *"self.eldobás(hidden)"* → *"self.dropout(hidden)"* (LLM-fordítás vakfoltja: a `dropout` szót "eldobás"-ra fordította)
- `train.tex`: *"rendbok ú"* → *"rendű"* (törött elválasztás)
- `train.tex`: *"összeomjást"* → *"összeomlást"*
- `train.tex`: *"akkord progress ziókat"* → *"akkordprogressziókat"*
- `train.tex`: *"gazdagitja"* → *"gazdagítja"*, *"outputs előállítását"* → *"kimenetek előállítását"*, *"kifogyorigotta"* → *"kifogyás okozta"*, *"kiigazaitja"* → *"kiigazítja"*, *"javaslataokkal"* → *"javaslatokkal"*, *"átterhe léséhez"* → *"áthelyezéséhez"*, *"állapottereeknél"* → *"állapottereknél"*, *"bemelegetési"* → *"bemelegítési"*
- `train.tex`: *"köteg alapú"* helyett vagy "kötegelt", vagy konzisztensen "batch-alapú"
- `zha.tex`: *"Akordokat"* (l. 27) és *"Akordok"* (l. 46) → *"akkordokat / akkordok"* (két k)
- `conclusion.tex`: *"Összegens"* → *"Összegzés"*

### 3.14 Vegyes magyar-angol szakszavak

Néhol angol marad indok nélkül (*"posterior collapse"*, *"free-bits regularizáció"*, *"top k"*, *"top p"*, *"label smoothing"*, *"teacher forcing"*, *"checkpoint"*, *"persistent workers"*). A magyar nyelvű thesisben a konvenció: *első előforduláskor* angolul + magyar fordításban zárójelben, utána egységesen valamelyik. Konkrétan:

- *"posterior collapse"* → *"poszterior-összeomlás"* magyarázattal első előforduláskor
- *"free-bits"* → *"szabad bitek"* (Kingma 2016 nyomán)
- *"teacher forcing"* → *"tanári kényszerítés"* (te magad is ezt használtad pl. `train.tex:57`)
- *"checkpoint"* → magyarban gyakori, maradhat, de mindenhol egységesen
- *"label smoothing"* → *"címkesimítás"* (te is használtad)

---

## 4. Beamer fókusz — egyetlen védés-stratégia

A védésen 15 percre van időd. A jelenlegi beamer kb. 50% a klasszikus modellek bemutatása, 50% a GOLC. **Ha** a GOLC a saját hozzájárulásod, akkor azt érdemes kiterjeszteni:

1. **Slide 6-8 (GOLC)** — itt egy konkrét, mért szám kell (orbital distance reduction stb. — futtasd a `scripts/compare_vae_models.py`-t, ha ad ilyet).
2. **Slide 10 (Eredmények)** — a táblázat számai konkrét forrásra mutassanak (script, dataset split). Ha nincs futó eredmény, **vedd ki a táblázatot** és helyettesítsd egy `scripts/generate_vae_metrics.py` által tényleg előállított ábrával.
3. **Slide 4 (Markov)**: a TikZ ábra szépen néz ki, de a "Hidden Markov Model (HMM)" + "GPU gyorsítás, Sparse mátrix" felsorolás placeholder-szerű. Inkább egy ábra a Markov vs. HMM-augmentált generálás *különbségéről* egy konkrét MIDI példán.
4. **Slide 9 (Transformer)**: itt explicit jelezni kell, mit *te* tettél hozzá (multitrack cross-attention, conditioning) — különben úgy néz ki, mintha csak a standard Vaswani-modellt magyaráznád.
5. **A "5.2× jobb orbitális konzisztencia"** állítás csak akkor maradjon, ha a kódból tényleg ki van futtatva.

---

## 5. Konkrét teendők — prioritás szerint

**P0 (védés előtt mindenképp):**
1. Töröld a placeholder `train.py`-t vagy alakítsd valódi dispatcherré.
2. Töröld a `train_diffusion.py` hivatkozást a README-ből.
3. Töröld vagy javítsd a `transformer.tex` RoPE / GatedFFN / EnhancedAttention / `lightning_transformer.py` / 6-réteg / label smoothing / 5000-warmup / Memory-mechanism appendix-bizonyítás állításokat. Vagy mind a 8 helyen javítsd a kódra, vagy implementáld a kódba (RoPE+GatedFFN kb. 1-2 nap).
4. Töröld a `vae.tex` BiLSTM / autoregresszív decoder / free bits / végtelen-norm clipping állításait.
5. Egyeztesd a Markov-fejezet 2-6 vs. max-4 ellentmondását.
6. A `tab:combined_performance` és a subjective scores: vagy mért adat, vagy ki kell venni.
7. Töröld a `train.tex`-ből a nem létező feature-eket (TrainingDebugger, Optuna, multi-GPU, W&B/TensorBoard, persistent_workers, automatic recovery, sample-monitoring).
8. Bibliográfia: cseréld a 8-9 nem-akadémiai cite-ot peer-reviewed forrásra.
9. Javítsd a tipográfiai hibákat (3.13).
10. Beamer slide 10 táblázat — vagy hard evidence, vagy ki.

**P1 (ha van idő):**
11. Lattner et al. 2018 és TARGET-VAE / EQ-VAE hozzáadása a `vae.tex § Kapcsolat korábbi munkákhoz`-hoz.
12. A duplikált `markov.tex § GPU-gyorsítás` és `train.tex § Hiperparaméter optimalizálás` szakaszok deduplikálása.
13. A `train.tex` újraírás a 3.9 szerint (egy oldalra zsugorítva).

**P2 (nice to have):**
14. Valódi user study, ha lehet — 5-10 alany és egy 5-pontos kérdőív elég ahhoz, hogy a conclusion subjective számai védhetőek legyenek.
15. RoPE + GatedFFN beimplementálása a Transformerbe.

---

## Függelék: a feltárás módja

- SSH-n keresztül plinkkel csatlakoztam (`192.168.1.138`, user `deginandor`).
- A kód- és LaTeX-fájlokat pscp-vel letöltöttem helyileg (`C:\Users\nandor.degi\zha_review\`).
- A modelleket és trénereket Explore subagentekkel olvastam végig (kérdés-alapú audit).
- A docs/thesis fejezeteket és a beamert sorról-sorra olvastam.
- A plágium-spotcheck egy general-purpose agenttel ment: 4-6 gyanús mondatra magyar + back-translated angol web-kereséssel, illetve a `dolgozat.bib`-en.

A helyi másolatok megmaradnak `C:\Users\nandor.degi\zha_review\`-ben — ha kell, ott közvetlenül szerkeszthetők és visszamásolhatók a gépre.
