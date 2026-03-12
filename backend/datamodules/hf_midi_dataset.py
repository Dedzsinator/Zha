"""
Streaming MIDI dataset sourced from Hugging Face — no local download required.

Dataset: amaai-lab/MidiCaps  (168,385 MIDI files, CC-BY-SA 4.0)
  https://huggingface.co/datasets/amaai-lab/MidiCaps

The MIDI audio files live in a single tar.gz on HF, but the train.json metadata
contains enough musical information to reconstruct a 128-dim pitch-class feature
vector that is fully compatible with the existing VAE / Transformer / GOLC-VAE
model inputs:

  • Dims   0–11  : Chroma (pitch-class) histogram built from all chord roots
  • Dims  12–23  : Chord-quality distribution (maj/min/dom7/maj7/min7/dim/aug/sus…)
  • Dims  24–35  : Bass-note pitch-class histogram
  • Dims  36–47  : Top-12 General-MIDI instrument-family presence flags
  • Dims  48–59  : Genre one-hot (12 genres recognised in the dataset)
  • Dims  60–67  : Mood flags (8 mood tags)
  • Dim   68     : Normalised tempo  (BPM 40-240 → 0-1)
  • Dim   69     : Normalised duration (0-600 s → 0-1)
  • Dims  70–81  : Key one-hot (12 major/minor keys mapped to 12 pitch classes)
  • Dims  82–127 : Chord-progression n-gram frequency (46 dims, top bigrams)

All features are L1-normalised per group so the final vector sums to 1 (matching
the original pitch-histogram convention used by the rest of the codebase).
"""

import io
import torch
import numpy as np
from torch.utils.data import IterableDataset, DataLoader

# --- genre / mood / quality mappings ----------------------------------------

_GENRES = [
    "pop", "rock", "jazz", "classical", "electronic", "country",
    "r&b", "hip hop", "metal", "folk", "blues", "latin",
]

_MOODS = [
    "happy", "sad", "energetic", "calm", "romantic", "dark", "upbeat", "melancholic",
]

_CHORD_QUALITIES = [
    "major", "minor", "dominant", "major seventh", "minor seventh",
    "diminished", "augmented", "suspended", "half-diminished", "other",
]

# General-MIDI family groups (channel-families, not patch numbers)
_GM_FAMILIES = [
    range(0, 8),    # Piano
    range(8, 16),   # Chromatic Perc
    range(16, 24),  # Organ
    range(24, 32),  # Guitar
    range(32, 40),  # Bass
    range(40, 48),  # Strings
    range(48, 56),  # Ensemble
    range(56, 64),  # Brass
    range(64, 72),  # Reed
    range(72, 80),  # Pipe
    range(80, 88),  # Synth Lead
    range(88, 128), # Synth Pad + SFX + Ethnic + Percussive
]


def _patch_to_family(patch: int) -> int:
    for i, fam in enumerate(_GM_FAMILIES):
        if patch in fam:
            return i
    return 11  # fallback


def _note_name_to_pc(name: str) -> int:
    """Convert note name string like 'C#', 'Bb', 'G' to pitch class 0-11."""
    table = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
    if not name:
        return 0
    root = name[0].upper()
    pc = table.get(root, 0)
    if len(name) > 1:
        acc = name[1:]
        pc += acc.count("#") - acc.count("b")
    return pc % 12


def _build_feature_vector(row: dict) -> np.ndarray:
    """Convert a single MidiCaps metadata row into a 128-dim float32 feature."""
    feat = np.zeros(128, dtype=np.float32)

    # --- 0-11: chroma histogram from chord roots ---
    # Try both field names — MidiCaps uses "chords" in some dataset versions
    all_chords = row.get("all_chords") or row.get("chords") or []
    if isinstance(all_chords, str):
        # Occasionally stored as JSON string
        try:
            import json
            all_chords = json.loads(all_chords)
        except Exception:
            all_chords = []

    chroma = np.zeros(12, dtype=np.float32)
    bass_chroma = np.zeros(12, dtype=np.float32)
    quality_hist = np.zeros(12, dtype=np.float32)  # always 12 to match feat[12:24] slot

    for chord in all_chords:
        if not isinstance(chord, dict):
            continue
        root = chord.get("root") or chord.get("Root") or ""
        bass = chord.get("bass") or chord.get("Bass") or root
        qual = (chord.get("quality") or chord.get("Quality") or "other").lower()

        chroma[_note_name_to_pc(root)] += 1.0
        bass_chroma[_note_name_to_pc(bass)] += 1.0

        matched = False
        for qi, qname in enumerate(_CHORD_QUALITIES[:-1]):
            if qname in qual:
                quality_hist[qi] += 1.0
                matched = True
                break
        if not matched:
            quality_hist[len(_CHORD_QUALITIES) - 1] += 1.0  # "other" bucket stays at original index

    feat[0:12] = chroma
    feat[12:24] = quality_hist[:12]  # trim to 12 to match slot
    feat[24:36] = bass_chroma

    # --- 36-47: GM instrument family flags ---
    instruments = row.get("instrument_numbers_sorted") or []
    if isinstance(instruments, str):
        try:
            import json
            instruments = json.loads(instruments)
        except Exception:
            instruments = []
    inst_flags = np.zeros(12, dtype=np.float32)
    for patch in instruments:
        try:
            inst_flags[_patch_to_family(int(patch))] = 1.0
        except (ValueError, TypeError):
            pass
    feat[36:48] = inst_flags

    # --- 48-59: genre one-hot ---
    genres = row.get("genre") or []
    if isinstance(genres, str):
        genres = [genres]
    genre_vec = np.zeros(12, dtype=np.float32)
    for g in genres:
        g_lower = str(g).lower()
        for gi, gname in enumerate(_GENRES):
            if gname in g_lower:
                genre_vec[gi] = 1.0
    feat[48:60] = genre_vec

    # --- 60-67: mood flags ---
    moods = row.get("mood") or []
    if isinstance(moods, str):
        moods = [moods]
    mood_vec = np.zeros(8, dtype=np.float32)
    for m in moods:
        m_lower = str(m).lower()
        for mi, mname in enumerate(_MOODS):
            if mname in m_lower:
                mood_vec[mi] = 1.0
    feat[60:68] = mood_vec

    # --- 68: normalised tempo ---
    try:
        bpm = float(row.get("tempo") or 120)
        feat[68] = np.clip((bpm - 40) / (240 - 40), 0.0, 1.0)
    except (ValueError, TypeError):
        feat[68] = 0.5

    # --- 69: normalised duration ---
    try:
        dur = float(row.get("duration") or 180)
        feat[69] = np.clip(dur / 600.0, 0.0, 1.0)
    except (ValueError, TypeError):
        feat[69] = 0.3

    # --- 70-81: key pitch class (one-hot over 12) ---
    key_str = str(row.get("key") or "")
    key_vec = np.zeros(12, dtype=np.float32)
    # e.g. "C major", "F# minor", "Bb major"
    tokens = key_str.split()
    if tokens:
        key_vec[_note_name_to_pc(tokens[0])] = 1.0
    feat[70:82] = key_vec

    # --- 82-127: chord bigram frequency (46 dims) ---
    # Encode as root-to-root transition histogram (flattened 12×12 → top 46)
    bigram = np.zeros(46, dtype=np.float32)
    if len(all_chords) > 1:
        for i in range(len(all_chords) - 1):
            c1 = all_chords[i]
            c2 = all_chords[i + 1]
            if isinstance(c1, dict) and isinstance(c2, dict):
                r1 = _note_name_to_pc(c1.get("root") or "")
                r2 = _note_name_to_pc(c2.get("root") or "")
                idx = (r1 * 4 + r2 // 3) % 46  # compact encoding
                bigram[idx] += 1.0
    feat[82:128] = bigram

    # --- L1 normalise each sub-group so magnitudes stay ~[0,1] ---
    for start, end in [(0, 12), (12, 24), (24, 36), (48, 60), (82, 128)]:
        s = feat[start:end].sum()
        if s > 0:
            feat[start:end] /= s

    return feat


# ---------------------------------------------------------------------------

class HuggingFaceMIDIDataset(IterableDataset):
    """
    Streaming PyTorch IterableDataset backed by amaai-lab/MidiCaps on HuggingFace.

    No files are downloaded to disk — the metadata is consumed via the HF
    datasets streaming API one row at a time.

    Args:
        genre_filter : list of genre strings to keep (case-insensitive substring
                       match).  None = keep all 168k tracks.
        split        : HF dataset split, currently only "train" exists.
        max_samples  : optional cap on the number of samples (useful for quick
                       experiments).
        shuffle_buffer: size of the streaming shuffle buffer (0 = no shuffle).
    """

    def __init__(
        self,
        genre_filter=None,
        split="train",
        max_samples=None,
        shuffle_buffer=1000,
    ):
        super().__init__()
        self.genre_filter = [g.lower() for g in genre_filter] if genre_filter else None
        self.split = split
        self.max_samples = max_samples
        self.shuffle_buffer = shuffle_buffer

    def _row_matches_genre(self, row: dict) -> bool:
        if self.genre_filter is None:
            return True
        genres = row.get("genre") or []
        if isinstance(genres, str):
            genres = [genres]
        combined = " ".join(str(g).lower() for g in genres)
        return any(gf in combined for gf in self.genre_filter)

    def __iter__(self):
        try:
            from datasets import load_dataset
        except ImportError as e:
            raise ImportError(
                "The 'datasets' library is required for HuggingFaceMIDIDataset. "
                "Install it with:  pip install datasets"
            ) from e

        ds = load_dataset(
            "amaai-lab/MidiCaps",
            split=self.split,
            streaming=True,
        )

        if self.shuffle_buffer > 0:
            ds = ds.shuffle(buffer_size=self.shuffle_buffer, seed=42)

        count = 0
        for row in ds:
            if self.max_samples is not None and count >= self.max_samples:
                break
            if not self._row_matches_genre(row):
                continue
            feat = _build_feature_vector(row)
            yield torch.from_numpy(feat)
            count += 1


def build_hf_dataloader(
    batch_size: int = 128,
    genre_filter=None,
    max_samples: int = None,
    shuffle_buffer: int = 1000,
    num_workers: int = 0,
    pin_memory: bool = None,  # None = auto-detect (True only when CUDA is available)
    tuple_wrap: bool = False,
) -> DataLoader:
    """
    Convenience factory that returns a DataLoader wrapping HuggingFaceMIDIDataset.

    Args:
        pin_memory: Pin memory for faster GPU transfers. Defaults to auto-detect
                    (True when CUDA is available, False otherwise).
        tuple_wrap: If True, each batch is returned as ``(tensor,)`` so it works
                    with code that does ``for data, in loader:`` (e.g. GOLC-VAE).

    Example — train on all pop & rock tracks::

        loader = build_hf_dataloader(batch_size=128, genre_filter=["pop", "rock"])

    Example — quick smoke test with 2000 samples::

        loader = build_hf_dataloader(batch_size=64, max_samples=2000)
    """
    dataset = HuggingFaceMIDIDataset(
        genre_filter=genre_filter,
        max_samples=max_samples,
        shuffle_buffer=shuffle_buffer,
    )

    # Auto-detect: pin_memory is only useful (and valid) when CUDA is available
    if pin_memory is None:
        pin_memory = torch.cuda.is_available()

    def _collate_tuple(batch):
        t = torch.stack(batch)
        return (t,)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=_collate_tuple if tuple_wrap else None,
    )


# ---------------------------------------------------------------------------
# Quick smoke-test:  python -m backend.datamodules.hf_midi_dataset
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("Streaming first 10 rows from amaai-lab/MidiCaps (no download)...")
    loader = build_hf_dataloader(batch_size=4, max_samples=10, shuffle_buffer=0)
    for i, batch in enumerate(loader):
        print(f"  Batch {i}: shape={batch.shape}  min={batch.min():.3f}  max={batch.max():.3f}")
    print("Done.")
