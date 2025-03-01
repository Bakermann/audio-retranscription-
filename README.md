# Retranscription

## Overview
Retranscription is a Python project that transcribe an audio file. 
The project uses various libraries such as faster whisper and pyannote to transcribe and translate audio from a file.


## Features
- Separates speakers from each other

- Transcribes audio to text

## Requirements
- python 3.11

- faster_whisper

- pyannote.audio

- librosa

- numpy

- torch

## USAGE

```
from transcriptor import transcriptor

tr = Transcritor()
tr.get_transcription("path/to/your/audio.mp3")
```

## Code Overview

VideoCaptioning Class
- __init__(self): Initializes the class by loading models.

- get_audio(self, file_path: str) -> np.ndarray: Load audio file.

- speaker_diarization(self, filename: str, num_speakers: int|None=None) -> list: Separate each speaker speaking time.

- transcribe(self, audio: np.ndarray) -> str: Transcribe audio.

- get_transcription(self, filename, num_speakers=None): Runs the entire captioning process, including audio extraction, vocal separation, transcription, subtitle generation, and translation.

## Contributing
Contributions are welcome! Feel free to submit a Pull Request or raise an issue if you find any bugs or have suggestions for improvements.

## License
This project is licensed under the MIT License. See the LICENSE file for more details.