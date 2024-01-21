# `pip3 install assemblyai` (macOS)
# `pip install assemblyai` (Windows)

import assemblyai as aai

aai.settings.api_key = "fa76543f131b4f7b9cd931b4b6267461"
transcriber = aai.Transcriber()

transcript = transcriber.transcribe("Sample.wav")
# transcript = transcriber.transcribe("./my-local-audio-file.wav")

print(transcript.text)