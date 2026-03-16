# Quick Start: Markov Chain Training

## Basic Usage

```bash
python backend/trainers/train_markov.py
```

## New Features

### Skip Processing (Press 's')

While MIDI files are being processed, press **'s'** to skip the rest and start training immediately:

```
 Processing 5000 MIDI files...
 Processing batch 5/50...
[Press 's']
Skip requested! Stopping after current batch...
 Training ENHANCED Markov model with HMM...
```

**Use cases:**
- Quick testing with partial dataset
- Resume training after interruption
- Preview results before full processing

### Processed File Cache

Files are automatically cached after processing. On subsequent runs:

```
Found 5000 MIDI files
Skipping 4500 already processed files
 Processing 500 new/modified files
```

**Cache location:** `output/trained_models/.processed_cache.json`

**Clear cache to reprocess all:**
```bash
rm output/trained_models/.processed_cache.json
```

### Performance Boost

- **2x faster batches**: 100 files/batch (GPU) or 200 files/batch (CPU)
- **2-3x faster reruns**: Skip already processed files
- **Smart memory**: Same 4-8GB usage, just faster

## Commands

### Standard Training
```bash
python backend/trainers/train_markov.py dataset/midi
```

### Custom Parameters
```bash
python backend/trainers/train_markov.py dataset/midi 4 12 16 true
#                                        ^dir      ^order ^interval ^states ^gpu
```

### Clear Cache and Retrain
```bash
rm output/trained_models/.processed_cache.json
python backend/trainers/train_markov.py
```

## Keyboard Controls

| Key | Action |
|-----|--------|
| `s` | Skip remaining files, finish batch, start training |
| `Ctrl+C` | Cancel training (saves cache first) |

## Tips

### First Run
- Let it complete to build cache
- Press 's' if you want quick preview
- Cache persists between runs

### Adding New Files
- Just add to dataset folder
- Run training again
- Only new files processed

### Testing
- Press 's' after 1-2 batches
- Test with ~100-200 files
- Full run when ready

## Troubleshooting

### "Keyboard listener error"
- Ignore - it's normal in some terminals
- Skip still works if you run in TTY

### Cache not working
- Check `output/trained_models/.processed_cache.json` exists
- Verify file permissions
- Delete and rebuild if corrupted

### Memory still high
- Reduce batch size in `train_markov.py` line ~450
- Default: 100 (GPU) / 200 (CPU)
- Lower to 50 if needed

## Performance Stats

### Before
- 50 files/batch
- ~5 files/sec
- Reprocess all files every run

### After  
- 100-200 files/batch
- ~10-15 files/sec
- Skip processed files (2-3x faster)

## Full Documentation

See `docs/TRAINING_FEATURES.md` for complete details.
