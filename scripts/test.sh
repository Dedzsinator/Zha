#!/bin/bash

echo "Starting multi-stage music generation pipeline..."

# Step 1: Generate with Markov Chain
echo "Step 1: Generating initial Markov Chain material..."
MARKOV_RESPONSE=$(curl -s -X POST http://localhost:8000/generate/markov \
  -F "start_note=60" \
  -F "length=64" \
  -F "key_context=C major" \
  -F "duration=30" \
  -F "instrument=piano" \
  -F "should_generate_audio=false")

MARKOV_MIDI=$(echo $MARKOV_RESPONSE | python3 -c "import sys, json; print(json.load(sys.stdin)['midi_url'].split('/')[-1])")
echo "Generated Markov MIDI: $MARKOV_MIDI"

if [ -z "$MARKOV_MIDI" ]; then
    echo "Failed to generate Markov MIDI"
    exit 1
fi

# Step 2: Add creativity with VAE
echo "Step 2: Adding creative variations with VAE..."
VAE_RESPONSE=$(curl -s -X POST http://localhost:8000/generate/vae \
  -F "midi_file=@generated_files/midi/$MARKOV_MIDI" \
  -F "creativity=0.7" \
  -F "duration=30" \
  -F "instrument=piano" \
  -F "generate_audio=false")

VAE_MIDI=$(echo $VAE_RESPONSE | python3 -c "import sys, json; print(json.load(sys.stdin)['midi_url'].split('/')[-1])")
echo "Generated VAE MIDI: $VAE_MIDI"

if [ -z "$VAE_MIDI" ]; then
    echo "Failed to generate VAE MIDI"
    exit 1
fi

# Step 3: Structure with Transformer
echo "Step 3: Creating structural sections with Transformer..."
FINAL_RESPONSE=$(curl -s -X POST http://localhost:8000/generate/structured_transformer \
  -F "midi_file=@generated_files/midi/$VAE_MIDI" \
  -F "num_sections=4" \
  -F "section_length=16" \
  -F "transition_smoothness=0.7" \
  -F "temperature=0.8" \
  -F "duration=60" \
  -F "instrument=piano" \
  -F "should_generate_audio=true")

FINAL_MIDI=$(echo $FINAL_RESPONSE | python3 -c "import sys, json; print(json.load(sys.stdin).get('midi_file', 'Not found'))")
FINAL_AUDIO=$(echo $FINAL_RESPONSE | python3 -c "import sys, json; print(json.load(sys.stdin).get('audio_file', 'Not found'))")

echo "Pipeline complete!"
echo "Final MIDI: $FINAL_MIDI"
echo "Final audio: $FINAL_AUDIO"