import glob, csv, time, os, math
from datetime import datetime
import numpy as np
import soundfile as sf
import webrtcvad
from scipy.signal import correlate
import torch
from faster_whisper import WhisperModel
from speechbrain.inference import EncoderClassifier
from scipy.signal import resample_poly
import pandas as pd

# ----------------- Utility -----------------
def load_mono16k(path):
    x, sr = sf.read(path, always_2d=False)
    if x.ndim > 1:
        x = x.mean(axis=1)
    if sr != 16000:
        x = resample_poly(x.astype(np.float32), 16000, sr)
        sr = 16000
    return x.astype(np.float32), sr

def rms_db(x):
    return 20*np.log10(np.sqrt(np.mean(x**2)+1e-12)+1e-12)

def rms_norm(x, target_db=-23.0):
    g = 10**((target_db - rms_db(x))/20)
    y = x * g
    return np.clip(y, -1.0, 1.0)

def align_to_ref(wavs, sr=16000, max_shift_s=0.05):
    """Hard-sync all to wavs[0] by cross-correlation (simple GCC-lite)."""
    ref = wavs[0]
    out = [ref]
    max_shift = int(max_shift_s*sr)
    for i in range(1, len(wavs)):
        a = wavs[i]
        n = min(len(a), len(ref))
        cc = correlate(a[:n], ref[:n], mode="valid")
        # constrain search window
        center = len(cc)//2
        lo = max(0, center - max_shift)
        hi = min(len(cc), center + max_shift)
        k = np.argmax(cc[lo:hi]) + lo
        shift = k - center
        out.append(np.roll(a, -shift))
    L = min(map(len, out))
    return [w[:L] for w in out]

def webrtc_vad_mask(x, sr=16000, frame_ms=20, mode=2):
    """Returns boolean mask with hop=frame_len (20ms default)."""
    vad = webrtcvad.Vad(mode)  # 0-3, 3 = most aggressive
    frame_len = int(sr*frame_ms/1000)
    pcm16 = (np.clip(x, -1, 1)*32767).astype(np.int16).tobytes()
    mask = []
    for i in range(0, len(x)-frame_len, frame_len):
        chunk = pcm16[i*2:(i+frame_len)*2]
        mask.append(vad.is_speech(chunk, sr))
    return np.asarray(mask, bool), frame_len

def frame_energy(x, win, hop):
    return np.array([(x[i:i+win]**2).mean()+1e-12 for i in range(0, len(x)-win, hop)])

def chunk_embed(ecapa, x, sr=16000, sec=1.6, hop=0.8, device=None):
    if device is None:
        device = 'cuda'
    L = len(x); w = int(sec*sr); h = int(hop*sr)
    embs = []
    with torch.no_grad():
        for i in range(0, max(0, L - w), h):
            seg = x[i:i+w]
            if np.abs(seg).mean() < 1e-3:
                continue
            # Ensure proper tensor format [batch_size, samples]
            t = torch.tensor(seg, dtype=torch.float32).unsqueeze(0).to(device)
            try:
                e = ecapa.encode_batch(t)
                # Handle different return formats
                if isinstance(e, tuple):
                    e = e[0]  # Take first element if tuple
                e = torch.nn.functional.normalize(e.squeeze(), dim=-1).cpu().numpy()
                embs.append(e)
            except Exception as ex:
                print(f"Warning: embedding failed for segment: {ex}")
                continue
    if not embs:
        return None
    return np.mean(np.vstack(embs), axis=0)

def cos(a, b):
    return float((a @ b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))

 
#  ----------------- Pipeline -----------------
start_time = time.time()

# Generate timestamp for output files
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# 0) Collect files 
audio_files = sorted(glob.glob("Data/*.WAV"))
assert 2 <= len(audio_files) <= 5, f"Expected 2-5 WAVs in Data/, found {len(audio_files)}"
print(f"Found {len(audio_files)} files: {audio_files}")

# 1) Load, normalize, (optionally) align
wav_list = []
for p in audio_files:
    x, sr = load_mono16k(p)
    wav_list.append(rms_norm(x, -23))
# If your files are same recorder start, you can skip align_to_ref.
# wav_list = align_to_ref(wav_list, sr=16000, max_shift_s=0.05)

# 2) VAD for dominance selection
vad_masks = []
frame_len = None
for x in wav_list:
    m, f = webrtc_vad_mask(x, 16000, frame_ms=20, mode=2)
    vad_masks.append(m)
    frame_len = f
T = min(len(m) for m in vad_masks)
vad_masks = [m[:T] for m in vad_masks]

# 3) Dominance frames: pick frames where a channel's energy >> others (own-voice)
win = frame_len
hop = frame_len
energies = [frame_energy(x, win, hop)[:T] for x in wav_list]
E = np.stack(energies)  # [C,T]
C = E.shape[0]
dom_masks = []
R = 3.0  # ratio threshold; raise to 3.5–4.0 if bleed high
for c in range(C):
    others = (np.sum(E, axis=0) - E[c]) / max(1, C-1)
    dom = (E[c] / (others + 1e-8) >= R)
    dom_masks.append(dom & vad_masks[c])

# 4) Build enrollment audio per channel from dominant regions (~up to 60s)
def gather_audio_from_mask(x, mask, frame_len, cap_sec=60):
    idx = np.where(mask)[0]
    if idx.size == 0:
        return np.zeros(0, dtype=np.float32)
    chunks = []
    max_frames = int((cap_sec*16000)//frame_len)
    for k in idx[:max_frames]:
        s = k*frame_len
        e = s + frame_len
        chunks.append(x[s:e])
    return np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float32)

enroll_audio = [gather_audio_from_mask(wav_list[c], dom_masks[c], frame_len) for c in range(C)]

# # Fallback: if a channel got nothing, take random voiced 30s
# for c in range(C):
#     if len(enroll_audio[c]) < 16000*3:
#         print(f"[warn] Channel {c} had few dominant frames; falling back to voiced chunks.")
#         voiced = np.where(vad_masks[c])[0]
#         enroll_audio[c] = gather_audio_from_mask(wav_list[c], vad_masks[c], frame_len, cap_sec=30)


# 5) ECAPA embeddings (auto-enrollment)
device = "cuda"
# Use the original stable model instead of mel-spec variant
ecapa = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", run_opts={"device": device})
E_seeds = []
for c in range(C):
    e = chunk_embed(ecapa, enroll_audio[c], sr=16000, sec=1.6, hop=0.8, device=device)
    if e is None:
        raise RuntimeError(f"Failed to build embedding for channel {c}")
    E_seeds.append(e)
E_seeds = np.stack(E_seeds)  # [C, D]
print(f"Auto-enrollment done. Built {C} speaker embeddings.")

# 5b) Dynamic enrollment for recurring unknown speakers (e.g., teacher)
# We'll collect embeddings from segments marked as UNKNOWN and cluster them
# If enough similar unknowns exist, we create a "TEACHER" or "OTHER" speaker
unknown_embeddings = []  # Will be populated during transcription
unknown_segments_info = []  # Store segment info for later clustering

# Optional quick refinement: score all voiced frames, keep frames best-matching own seed by margin
# (kept simple for clarity; the seeds are usually good enough)


# 6) Faster-Whisper transcription, but we’ll reassign speaker per segment
fw = WhisperModel("Finnish-NLP/whisper-large-finnish-v3-ct2", device="cuda", compute_type="float32") #Finnish-NLP/whisper-large-finnish-v3-ct2

all_rows = []
file_speaker_stats = {}  # Track speaker statistics per file
 
for audio_file, channel_idx in zip(audio_files, range(C)):
    name = os.path.splitext(os.path.basename(audio_file))[0]
    file_start = time.time()
    print(f"\nProcessing {channel_idx+1}/{C}: {name}")
    
    # Clear CUDA cache before processing each file
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    print(f"  Transcribing audio...")
    
    # Initialize speaker statistics for this file
    file_speaker_stats[name] = {}

    segments, info = fw.transcribe(
        audio_file,
        vad_filter=True,
        vad_parameters=dict(
            min_silence_duration_ms=5000,  # 5 seconds silence before splitting
            min_speech_duration_ms=300,    # Allow short phrases like "yes", "no" (0.3 sec)
            threshold=0.5,                 # Voice activity threshold
            neg_threshold=0.35              # Non-voice threshold
        ),
        word_timestamps=False,  # segment-level first; you can switch to True for per-word
        no_speech_threshold=0.6  # Higher threshold to avoid splitting on brief pauses
    )

    # load raw audio once for slicing
    x, sr = load_mono16k(audio_file)

    for seg in segments:
        s = max(0, int(seg.start * sr))
        e = min(len(x), int(seg.end * sr))
        seg_audio = x[s:e]
        if len(seg_audio) < sr * 0.6:  # too short for stable embedding
            # extend by grabbing a little context
            le = min(len(x), e + int(0.3 * sr))
            ls = max(0, s - int(0.3 * sr))
            seg_audio = x[ls:le]

        emb = chunk_embed(ecapa, seg_audio, sr=sr, sec=1.2, hop=0.6, device=device)
        if emb is None:
            # fallback to channel owner if silent
            assigned = channel_idx
            conf = 0.0
        else:
            sims = np.array([cos(emb, E_seeds[c]) for c in range(C)])
            # Add smaller, balanced priors to avoid exceeding 1.0
            prior = np.full(C, -0.02, dtype=np.float32)  # Small penalty for others
            prior[channel_idx] = 0.02  # Small boost for home channel
            scores = sims + prior
            
            assigned = int(np.argmax(scores))
            conf = float(scores[assigned])

        # Track speaker duration for this file
        segment_duration = seg.end - seg.start
        speaker_name = f"SPK{assigned+1}"
        if speaker_name not in file_speaker_stats[name]:
            file_speaker_stats[name][speaker_name] = 0
        file_speaker_stats[name][speaker_name] += segment_duration
        
        all_rows.append({
            "file_channel": name,
            "assigned_speaker": speaker_name,
            "start_time": seg.start,
            "end_time": seg.end,
            "score": round(conf, 3),
            "text": seg.text.strip()
        })

    file_time = time.time() - file_start
    print(f"  Completed in {file_time:.1f} seconds")

# 6d) Determine main speaker for each file and update file_channel names
file_main_speakers = {}
for original_name, speaker_durations in file_speaker_stats.items():
    if speaker_durations:
        main_speaker = max(speaker_durations, key=speaker_durations.get)
        file_main_speakers[original_name] = main_speaker
        total_duration = sum(speaker_durations.values())
        main_speaker_percentage = (speaker_durations[main_speaker] / total_duration) * 100
        print(f"  File '{original_name}' -> Main speaker: {main_speaker} ({main_speaker_percentage:.1f}% of speech)")
    else:
        file_main_speakers[original_name] = original_name  # Keep original name if no speakers detected

# Update all rows to use main speaker as file_channel
for row in all_rows:
    original_file = row["file_channel"]
    row["file_channel"] = file_main_speakers.get(original_file, original_file)

# 7) (Optional) If your files share a common time origin, you can sort globally:
all_rows.sort(key=lambda r: r["start_time"])

# 8) Write CSV
with open(f"transcription_{timestamp}.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["file_channel","assigned_speaker","start_time","end_time","score","text"])
    writer.writeheader()
    writer.writerows(all_rows)

# 9) Advanced filtering for similar transcriptions

def similarity_score(text1, text2):
    """Calculate similarity between two texts using simple token overlap."""
    tokens1 = set(text1.lower().split())
    tokens2 = set(text2.lower().split())
    if len(tokens1) == 0 and len(tokens2) == 0:
        return 1.0
    if len(tokens1) == 0 or len(tokens2) == 0:
        return 0.0
    intersection = len(tokens1.intersection(tokens2))
    union = len(tokens1.union(tokens2))
    return intersection / union if union > 0 else 0.0

def time_overlap(start1, end1, start2, end2):
    """Calculate overlap ratio between two time intervals."""
    overlap_start = max(start1, start2)
    overlap_end = min(end1, end2)
    if overlap_start >= overlap_end:
        return 0.0
    overlap_duration = overlap_end - overlap_start
    total_duration = max(end1, end2) - min(start1, start2)
    return overlap_duration / total_duration if total_duration > 0 else 0.0

def find_similar_groups(df, time_threshold=0.2, similarity_threshold=0.5):
    """Find groups of similar transcriptions using clustering approach."""
    groups = []
    used = set()
    
    for i in range(len(df)):
        if i in used:
            continue
            
        current_group = [i]
        used.add(i)
        row1 = df.iloc[i]
        
        # Find all similar utterances to this one
        for j in range(i + 1, len(df)):
            if j in used:
                continue
                
            row2 = df.iloc[j]
            
            # Check if they're too far apart in time (>5 seconds gap)
            if row2['start_time'] - row1['end_time'] > 5.0:
                continue
                
            # Check for temporal overlap or proximity
            time_ovl = time_overlap(row1['start_time'], row1['end_time'], 
                                   row2['start_time'], row2['end_time'])
            
            # Also consider proximity (within 2 seconds start time)
            time_proximity = abs(row1['start_time'] - row2['start_time']) < 2.0
            
            # Check for text similarity
            text_sim = similarity_score(row1['text'], row2['text'])
            
            # More aggressive grouping criteria:
            # 1. High text similarity (>0.5) with ANY time overlap or close proximity
            # 2. Medium text similarity (>0.3) with significant time overlap (>0.3)
            # 3. Any overlap (>0.5) with decent text similarity (>0.25)
            should_group = (
                (text_sim > similarity_threshold and (time_ovl > 0.1 or time_proximity)) or
                (text_sim > 0.3 and time_ovl > 0.3) or
                (time_ovl > 0.5 and text_sim > 0.25)
            )
            
            if should_group:
                current_group.append(j)
                used.add(j)
        
        if len(current_group) > 1:
            groups.append(current_group)
    
    return groups

df = pd.DataFrame(all_rows)
print(f"Original segments: {len(df)}")

# Step 1: Remove very low confidence segments
df_filtered = df[df['score'] > 0.15].copy()
print(f"After basic confidence filtering (score > 0.15): {len(df_filtered)}")

# Step 2: Sort by time for better clustering
df_filtered = df_filtered.sort_values('start_time').reset_index(drop=True)

# Step 3: Find groups of similar transcriptions (more aggressive settings)
similar_groups = find_similar_groups(df_filtered, time_threshold=0.1, similarity_threshold=0.35)
print(f"Found {len(similar_groups)} groups of similar transcriptions")

# Step 4: For each group, keep only the one with highest score
indices_to_keep = set(range(len(df_filtered)))

for group in similar_groups:
    if len(group) <= 1:
        continue
        
    # Find the best transcript in this group (highest score, then longest text)
    best_idx = group[0]
    best_row = df_filtered.iloc[best_idx]
    
    for idx in group[1:]:
        row = df_filtered.iloc[idx]
        # Prefer higher score, or if scores are close, prefer longer text
        if (row['score'] > best_row['score'] + 0.05) or \
           (abs(row['score'] - best_row['score']) < 0.05 and len(row['text']) > len(best_row['text'])):
            best_idx = idx
            best_row = row
    
    # Remove all others from this group
    for idx in group:
        if idx != best_idx:
            indices_to_keep.discard(idx)
    
    print(f"Group of {len(group)} similar transcripts -> kept best one (score: {best_row['score']:.3f})")

# Keep only the best from each group
clean_df = df_filtered.iloc[list(indices_to_keep)].reset_index(drop=True)
print(f"After similarity clustering: {len(clean_df)}")

# Step 5: Final quality filter - be more selective
final_df = clean_df[
    (clean_df['score'] > 0.35) |  # Good confidence segments
    ((clean_df['score'] > 0.25) & (clean_df['text'].str.len() > 25))  # Or decent confidence with longer text
].copy()

print(f"After final quality filtering: {len(final_df)}")

# Sort by time for final output
final_df = final_df.sort_values('start_time').reset_index(drop=True)

# Save filtered version
final_df.to_csv(f'Ecapa_transcription_{timestamp}.csv', index=False)

print(f"Saved {len(all_rows)} rows -> transcription_{timestamp}.csv")
print(f"Saved {len(final_df)} clean rows -> Ecapa_transcription_{timestamp}.csv")

total_time = time.time()-start_time
print(f"\nTotal processing time: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
