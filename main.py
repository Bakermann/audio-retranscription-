import librosa
import numpy as np
import torch
from time import time
from faster_whisper import WhisperModel
from pyannote.audio import Pipeline
import logging

logging.getLogger("faster_whisper").setLevel(logging.WARNING)


def convert_seconds(seconds: float)->tuple:
    """Converts time in seconds to SRT format (hh:mm:ss,ms).

    :param seconds: time in seconds
    :return: string containing seconds converted to hours minutes seconds and milliseconds
    """
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    remaining_seconds = seconds % 60
    return hours, minutes, remaining_seconds


class Transcritor:
    """Class to transcribe audio."""

    def __init__(self):
        """Initializes the class by loading models."""

        # select device
        device = "cuda" if torch.cuda.is_available() else "cpu"

        # load whisper model
        self.model = WhisperModel("large", device=device, compute_type="int8")
        self.model_sapling_rate = 16000  # whisper

        # load diarization model
        self.pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token="my_auth_token")

        # send pipeline to GPU (when available)
        self.pipeline.to(torch.device(device))

    def get_audio(self, file_path: str) -> np.ndarray:
        """Load audio file.

        :param file_path: path of the file
        :return: audio in array format
        """
        audio_data, sample_rate = librosa.load(file_path, sr=None)
        audio_data = librosa.resample(audio_data,
                                      orig_sr=sample_rate,
                                      target_sr=self.model_sapling_rate)
        audio_array = np.array(audio_data)

        return audio_array

    def speaker_diarization(self, filename: str, num_speakers: int|None=None) -> list:
        """
        Separate each speaker speaking time.

        :param filename: name of the file to diarize
        :param num_speakers: number of speaker. Default is None
        :return: list containing each speaking period for each speaker
        """
        output = []
        speaker_id = ""
        pos = -1
        diarization = self.pipeline(filename, num_speakers=num_speakers)
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            # put speaker consecutive speeches together
            if speaker == speaker_id:
                output[pos][1] = turn.end
            else:
                output.append([turn.start, turn.end, speaker])
                pos += 1
                speaker_id = speaker

        return output

    def transcribe(self, audio: np.ndarray) -> str:
        """
        Transcribe audio.

        :param audio: audio
        :return: model's transcription
        """

        text = ""
        segments, _ = self.model.transcribe(audio, log_progress=False)
        for segment in segments:
            text += segment.text

        return text

    def get_transcription(self, filename, num_speakers=None):
        """
        Runs the entire captioning process, including audio extraction, vocal separation, transcription, subtitle generation, and translation.

        :param filename: name of the audio to transcribe
        :param num_speakers: number of speaker in the audio
        """

        text = ""
        speaker = ""
        audio_array = self.get_audio(filename)
        diarization = self.speaker_diarization(filename, num_speakers)
        print(
            f"Audio ({round(len(audio_array) / self.model_sapling_rate, 2)}s) got cut into {len(diarization)} chunks.")
        with open(f"transcriptions/{name}.txt", "w", encoding="utf-8") as file:
            for i, diari in enumerate(diarization):
                start, end, speaker_id = diari
                print(f"Processing {i + 1}/{len(diarization)}")
                audio = audio_array[int(self.model_sapling_rate * start):int(self.model_sapling_rate * end)]

                hours, minutes, remaining_seconds = convert_seconds(start)
                start = f"{int(hours)}h{int(minutes)}min{int(remaining_seconds)}s"
                hours, minutes, remaining_seconds = convert_seconds(end)
                end = f"{int(hours)}h{int(minutes)}min{int(remaining_seconds)}s"

                transcription = self.transcribe(audio)
                if speaker != speaker_id:
                    text += f"{speaker_id} ({start} à {end}):\n"
                    speaker = speaker_id
                text += f"{transcription}\n\n"
                file.write(text)
                file.flush()
                text = ""


if __name__ == '__main__':
    file = "tlous.wav"
    name = file.split(".")[0]
    start_time = time()
    tr = Transcritor()
    tr.get_transcription(f"audio/{file}", 9)

    h, m, s = convert_seconds(time() - start_time)
    print(f"Done in {h}hours {int(m)}minutes {int(s)}seconds.")
