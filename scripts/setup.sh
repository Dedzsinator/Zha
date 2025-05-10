#!/bin/bash

echo "Installing system dependencies..."
sudo pacman -S --needed python python-pip fluidsynth wget python-virtualenv

echo "Creating Python virtual environment..."
python -m virtualenv venv
source venv/bin/activate

echo "Installing Python dependencies..."
pip install torch numpy music21 fastapi uvicorn python-multipart \
    matplotlib tqdm pretty_midi mido scipy requests \
    pydantic

echo "Downloading FluidR3 SoundFont for audio synthesis..."
mkdir -p ~/.soundfonts
if [ ! -f ~/.soundfonts/FluidR3_GM.sf2 ]; then
    wget -O ~/.soundfonts/FluidR3_GM.sf2 https://member.keymusician.com/Member/FluidR3_GM/FluidR3_GM.sf2
fi

echo "Installation complete!"