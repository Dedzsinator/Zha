import React from 'react';

function MusicPlayer({ audioUrl }) {
  return (
    <div className="mt-4">
      <h2>Generated Music</h2>
      <audio controls src={audioUrl} className="w-100" />
      <a href={audioUrl} download="generated_music.wav" className="btn btn-primary mt-2">
        Download Music
      </a>
    </div>
  );
}

export default MusicPlayer;