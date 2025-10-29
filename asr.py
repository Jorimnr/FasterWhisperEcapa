
from faster_whisper import WhisperModel
import glob
import csv
import time
import os

# time tracking
start_time = time.time()

model = WhisperModel("large-v3", device="cuda", compute_type="float32")

audio_files = glob.glob("Data/*.WAV")
all_segments = []

print(f"Found {len(audio_files)} files to process")

for audio_file in audio_files:
    speaker_name = os.path.splitext(os.path.basename(audio_file))[0]
    file_start = time.time()
    print(f"Processing: {speaker_name}")
    
    segments, info = model.transcribe(
        audio_file, 
        vad_filter=True, 
        vad_parameters=dict(min_silence_duration_ms=2000, min_speech_duration_ms=500),
        word_timestamps=True
    )
    
    for segment in segments:
        all_segments.append({
            'speaker': speaker_name,
            'start_time': segment.start,
            'end_time': segment.end,
            'text': segment.text.strip()
        })
    file_time = time.time() - file_start
    print(f"  Completed in {file_time:.1f} seconds")

all_segments.sort(key=lambda x: x['start_time'])

with open('transcription.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=['speaker', 'start_time', 'end_time', 'text'])
    writer.writeheader()
    writer.writerows(all_segments)

total_time = time.time() - start_time
print(f"\nTotal processing time: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
print(f"Saved {len(all_segments)} segments to transcription.csv")