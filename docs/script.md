---

## Áttekintés dia
**"A prezentáció során végigmegyünk a rendszer architektúráján, a MIDI adatfeldolgozáson, majd részletesen bemutatom mind a három modellt: Markov-láncot, VAE-t és Transformert, végül összehasonlítom őket."**

---

## Rendszer Architektúra dia

**"A teljes rendszer egy React frontend-ből, FastAPI backend-ből és három különböző modellből áll. A MIDI adatbázis szolgáltatja a tanítási adatokat mind a három modell számára."**

*Mutass a diagramra:*
- **Frontend-Backend kapcsolat:** "A React alkalmazás REST API-n keresztül kommunikál a backend-del"
- **Modellek:** "Mindhárom modell külön endpoint-on érhető el"
- **MIDI I/O:** "Közvetlen MIDI fájl feltöltés és letöltés"

---

## MIDI Adattranszformációs Pipeline dia
**"Minden modell ugyanazt a MIDI feldolgozási pipeline-t használja. A parse_midi függvény 128 dimenziós feature vektort hoz létre."**

*Mutass a kódra:*
- **pretty_midi betöltés:** "PrettyMIDI könyvtár kezeli a bináris MIDI formátumot"
- **feature vektor építés:** "128 elem minden MIDI hangjegyhez (0-127)"
- **velocity súlyozás:** "Hangerő normalizálása 0-1 tartományba"
- **normalizálás:** "Az összeg 1.0-ra normalizálva, valószínűségi eloszlást kapunk"

**"Ez a reprezentáció minden modell számára közös kiindulópont!"**

---

## Markov-lánc Modell Architektúra dia
**"Kezdjük a Markov-lánc modellel. Ez a legegyszerűbb, de nagyon hatékony megközelítés."**

*Mutass a diagramra:*
- **Felső rész:** "293 MIDI fájlt dolgozunk fel párhuzamosan Music21 segítségével hangjegy szekvenciákká"
- **Középső rész:** "Három fajta átmeneti mátrixot építünk fel. Az első a hangjegy átmenetek - egy 128×128-as mátrix, ami minden MIDI hangjegy után következő hangjegy valószínűségét tárolja. A második az intervallum átmenetek - ezek a relatív hangmagasság változásokat tanulják, például +2 félhang azt jelenti, hogy nagy szekund felfelé lépünk. A harmadik pedig a többrendű átmenetek - ezek nem csak az előző hangjegyet, hanem az előző 2-3 hangjegy teljes kontextusát veszik figyelembe a következő hangjegy jóslásához."
- **Alsó rész:** "Ezekből a mátrixokból kinyerjük a magas szintű zenei jellemzőket - akkordmeneteket, ritmus mintákat, hangnemeket és római szám átmeneteket"

**"A kulcs, hogy a többrendű átmenetek zenei frázisokat és motívumokat tanulnak meg, nem csak egyedi hangjegy lépéseket!"**

---

## Markov Implementáció dia
**"A Markov modell három szinten tanulja a zenét: közvetlen átmenetek, intervallum minták, és kontextuális frázisok."**

*Mutass a kódra:*
- **1-2. sor:** "A transitions mátrix az egyszerű hangjegy átmeneteket tárolja - P(hangjegy_j | hangjegy_i) valószínűségeket"
- **3. sor:** "Az interval_transitions a relatív hangmagasság változásokat - így skálafüggetlen mintákat fedez fel"
- **5-12. sor:** "A musical_features pedig a legmagasabb szintű zenei tudást tárolja - teljes akkordmeneteket, komplex ritmus mintákat"

**"A többrendű átmenetek azt jelentik, hogy ha például C-E-G szekvenciát lát, tudja, hogy akkord következik, és ennek megfelelően választ következő hangjegyet!"**

---

## Átmeneti Mátrix Példa dia
**"Itt látható, hogyan működnek a különböző szintű átmenetek a gyakorlatban."**

*Mutass a táblázatra:*
- **C4 sor:** "Egyszerű átmenet: C4 után 25% valószínűséggel D4 következik - ez a diatonikus skála logikája"
- **G4 sor:** "G4 után 35% valószínűséggel tér vissza C4-re - ez a tonikai visszatérés gravitációja"

*Mutass az intervallum részre:*
**"Az intervallum átmenetek univerzálisabbak: +2 félhang bármilyen nagy szekund, függetlenül a konkrét hangmagasságtól."**

**"A többrendű átmenetek pedig így működnek: ha az előző szekvencia C4-E4 volt, akkor nem 20%, hanem például 45% esély van F4-re, mert a modell felismeri az C-E-F motívumot!"**

---

## Markov Tanítási Folyamat dia
**"A tanítás rendkívül hatékony - 5 perc alatt feldolgozza a 293 fájlt, mert párhuzamosan dolgozik."**

*Mutass a kódra:*
- **3-4. sor:** "Párhuzamos feldolgozás multiprocessing.Pool-lal - minden CPU mag egy fájlon dolgozik"
- **8-9. sor:** "Hangjegy szekvenciák kinyerése (pitch, duration) párokban - ezek lesznek a tanítási adatok"
- **12. sor:** "A modell tanítása során építi fel mind a három átmeneti mátrixot egyszerre"

**"A sebesség titka: míg egy mag egy MIDI fájlt elemez, a többi már a következőkön dolgozik!"**

---

## Zenei Jellemzők Kinyerése dia
**"A modell valódi zenei elméletet tanul, nem csak statisztikát."**

*Mutass a features dictionary-re:*
- **2. sor:** "Akkordmeneteket ismer fel - I-V-vi-IV, ii-V-I típusú progressziókat"
- **3. sor:** "Ritmus mintákat tanul - erős és gyenge ütemek kapcsolatait"
- **4. sor:** "Római szám átmeneteket - funkcionális harmónia logikáját: I→V (tónika→domináns)"

**"A Music21 könyvtár segítségével valódi zenei analízist végez - hangnemeket, akkordokat, római számokat ismer fel!"**

---

## Generálási Módszerek dia
**"A generálás többszintű: egyszerű lépésektől komplex harmonikus gondolkodásig."**

*Mutass a kódra:*
- **generate_sequence:** "Alapszintű generálás az átmeneti mátrixból - de már zeneileg értelmes"
- **generate_with_chords:** "Harmonikus generálás - előbb akkordmenetet választ, majd ahhoz illő dallamot"

**"A többrendű átmenetek itt ragyognak: amikor C-E-G után G-t generál, tudja, hogy akkord jön, nem véletlenszerű hangjegy!"**

---

## Expresszi kiszámi generálás dia
**"A komplex generálás kombinálja a ritmus, akkord és komplexitási paramétereket."**

*Mutass a kódra:*
- **complexity paraméter:** "0.3 alatt egyszerű 4/4, felette változatos ütemmutatók"
- **akkordmenet generálás:** "Hangnem alapú progressziók"
- **ritmikus szekvencia:** "Ütemmutatóhoz illő ritmus minták"

**"Az előnyök: villámgyors tanítás, zeneileg értelmes kimenet, valós idejű generálás!"**

---

## VAE Modell Részletes Architektúra dia
**"A VAE teljesen más megközelítés - folytonos látens térben tanulja a zenei stílusokat."**

*Mutass a diagramra végig:*
- **Encoder út:** "128 dimenziós MIDI hisztogram → 512 → 256 reziduális blokkokkal. A ResidualBlock-ok stabilabb tanítást biztosítanak"
- **Látens tér:** "μ és log σ² paraméterek - a reparameterizációs trükk teszi lehetővé a folytonos reprezentációt"
- **Decoder út:** "Vissza 256 → 512 → 128 dimenzióba, Sigmoid aktivációval valószínűségi kimenethez"

**"A LayerNorm + ResidualBlock + SiLU kombináció sokkal stabilabb, mint a hagyományos VAE architektúrák!"**

---

## VAE Pytorch Implementáció dia
**"A ResidualBlock és SiLU aktiváció kombináció az igazi újítás itt."**

*Mutass a kódra:*
- **ResidualBlock:** "LayerNorm + SiLU + reziduális kapcsolat - ez biztosítja a stabil gradiens áramlást"
- **beta paraméter:** "Beta-VAE - a beta=0.5 kontrollált látens tér tanuláshoz, kevésbé káotikus eredményekhez"

**"Ez nem standard VAE - a SiLU aktiváció és reziduális blokkok jelentősen javítják a zenei koherenciát!"**

---

## VAE Tanítási Implementáció dia
**"A consistency loss a legnagyobb újítás - ez biztosítja a simább zenei átmeneteket."**

*Mutass a kódra:*
- **recon_loss:** "Rekonstrukciós veszteség - mennyire jól állítja vissza az eredeti bemenetet"
- **kl_loss:** "KL divergencia - biztosítja, hogy a látens tér normális eloszlású legyen"
- **consistency_loss:** "Saját újítás - bünteti az egymást követő hangjegyek közti nagy ugrásokat"
- **scaler.scale:** "Mixed precision - GPU memória felét spórolja, dupla akkora batch méretet enged"

**"A consistency loss zenei logika: két szomszédos hangjegy között ne legyen hatalmas ugrás!"**

---

## VAE Generálási Képességek dia
**"A VAE legnagyobb ereje: sima interpoláció két zenei stílus között."**

*Mutass a kódra:*
- **sample függvény:** "Temperature vezérelt mintavételezés - alacsony hőmérséklet konzervatív, magas kreatív"
- **interpolate függvény:** "Két dal közötti sima átmenet a látens térben"

**"Mivel a látens tér folytonos, egy jazz és egy klasszikus darab között természetes átmenetet tud létrehozni!"**

---

## Transformer Modell Architektúra dia
**"A Transformer a legkomplexebb - attention mechanizmus plus speciális memória rendszer strukturált zenéhez."**

*Mutass a diagramra:*
- **Input rész:** "MIDI szekvencia → lineáris beágyazás 512 dimenzióba → pozicionális kódolás"
- **Transformer stack:** "8 réteg, mindegyikben 8 fejű attention és 2048 dimenziós feed forward"
- **Memória komponensek:** "A globális memória az aktuális kontextust, a section_memories pedig vers-refrén-híd szakaszokat tárol külön-külön"
- **Kimenet:** "Autoregresszív generálás - minden hangjegy az összes előző alapján"

**"A memória rendszer teszi lehetővé, hogy zenei formákat értsen: ABAB struktúrát, refrén ismétléseket!"**

---

## A Transformer Osztály dia
**"A section_memories dictionary a legnagyobb újítás - zenei szakaszok memóriája."**

*Mutass a kódra:*
- **self.memory:** "Globális memória az aktuális generálási állapothoz"
- **self.section_memories:** "Dictionary: kulcs=szakasz_id (0=vers, 1=refrén), érték=memória tensor"

**"Amikor a refrént generálja, pontosan tudja, hogyan szólt az első refrén, és következetesen alkalmazza!"**

---

## Hol generál dia
**"Itt a teljes memória logika - ez adja a zenei strukturáltságot."**

*Mutass a kódra sorról sorra:*
- **self.reset_memory():** "Tiszta kezdés - törli az összes memóriát"
- **for section_id:** "Minden szakaszhoz (0=vers, 1=refrén, 2=híd, 3=outro)"
- **if section_id in self.section_memories:** "Volt már ilyen szakasz?"
- **self.memory = self.section_memories[section_id]:** "Ha igen, betölti azt a memória állapotot"
- **self._generate_section:** "Generál 16 hangjegyet a memória kontextusával"
- **self.section_memories[section_id] = self.memory:** "Elmenti az új memória állapotot"

**"A lényeg: a section_memories nemcsak hangjegyeket tárol, hanem a Transformer teljes belső állapotát - attention súlyokat, kontextust, mindent. Amikor visszatér a refrénhez, ott folytatja a 'gondolkodást', ahol az első refrén véget ért!"**

---

## Transformer Tanítási Stratégia dia
**"OneCycleLR és mixed precision - a legmodernebb optimalizáció."**

*Mutass a kódra:*
- **AdamW:** "Weight decay regularizáció a túltanulás ellen"
- **OneCycleLR:** "Ciklusos learning rate - 10% warmup, majd cosinus csökkenés"
- **autocast:** "Mixed precision - float16/float32 keverék GPU memória spórolásért"
- **scaler:** "Gradient scaling - megakadályozza a gradiens eltűnést"

**"Ez az optimalizációs kombináció dupla sebességet és fél akkora memóriahasználatot eredményez!"**

---

## Transformer Generálási Funkciók dia
**"A strukturált generálás teljes implementációja - memória + kreativitás."**

*Ismételd meg részletesen:*
**"A section_memories működése: minden section_id saját memóriát kap, ami nemcsak a generált hangjegyeket, hanem a Transformer teljes belső állapotát tárolja. Ez magában foglalja az attention súlyokat, a rejtett reprezentációkat, a kontextuális információkat. Amikor a modell visszatér egy szakaszhoz - például a második refrénhez - nem nulláról kezdi, hanem pontosan ott folytatja, ahol az első refrén memóriája véget ért. Ez biztosítja a zenei koherenciát: a refréneк hasonlóan hangzanak, de nem unalmasan ismétlődnek, mert a temperature paraméter kis variációkat ad hozzá."**

---

## Metrikák dia
**"Végül az eredmények - három teljesen különböző komplexitású megközelítés."**

*Mutass a táblázatra és grafikon:*
- **Markov: 5 perc** "Leggyorsabb - statisztikai számítás és párhuzamos feldolgozás"
- **VAE: 55 perc** "Közepes - neurális hálózat 100 epochon át kevert precizitással"  
- **Transformer: 1 óra 5 perc** "Leglassabb - attention mechanizmus és memória rendszer"

**"A komplexitás-idő trade-off: Markov gyors és zeneileg helyes, VAE kreatív és interpolálható, Transformer strukturált és koherens. Mind a három hozzájárul a végső alkalmazáshoz!"**

---

## Záró dia
**"Összefoglalva: három fundamentálisan különböző megközelítést mutattam be. A Markov-lánc zenei elméleten alapul többrendű átmenetekkel és gyors statisztikai tanulássel. A VAE kontinuus látens térben dolgozik consistency loss-szal a sima átmenetekért. A Transformer pedig attention mechanizmussal és section memories rendszerrel strukturált kompozíciókat alkot. Együtt egy komplett zenei generáló rendszert alkotnak. Köszönöm a figyelmet!"**
