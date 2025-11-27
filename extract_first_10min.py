import glob
import os
import soundfile as sf
from scipy.signal import resample_poly

def extract_first_n_minutes(input_file, output_file, minutes=10):
    """Extract the first N minutes from an audio file."""
    print(f"Processing: {os.path.basename(input_file)}")
    
    # Read the audio file
    audio, sr = sf.read(input_file, always_2d=False)
    
    # Calculate samples for N minutes
    samples_needed = int(sr * 60 * minutes)
    
    # Extract first N minutes
    if len(audio) > samples_needed:
        audio_trimmed = audio[:samples_needed]
        print(f"  Trimmed from {len(audio)/sr/60:.2f} min to {minutes} min")
    else:
        audio_trimmed = audio
        print(f"  File is only {len(audio)/sr/60:.2f} min, keeping full length")
    
    # Write output file
    sf.write(output_file, audio_trimmed, sr)
    print(f"  Saved: {os.path.basename(output_file)}")

def main():
    # Find all WAV files in Data folder
    input_files = sorted(glob.glob("Data/*.WAV"))
    
    if not input_files:
        print("No WAV files found in Data/ folder")
        return
    
    # Create output directory
    output_dir = "Data_10min"
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Found {len(input_files)} files in Data/")
    print(f"Extracting first 10 minutes to {output_dir}/\n")
    
    for input_file in input_files:
        # Generate output filename with "-10min" suffix
        basename = os.path.basename(input_file)
        name, ext = os.path.splitext(basename)
        output_filename = f"{name}-10min{ext}"
        output_file = os.path.join(output_dir, output_filename)
        
        # Extract first 10 minutes
        extract_first_n_minutes(input_file, output_file, minutes=10)
    
    print(f"\nDone! Created {len(input_files)} files in {output_dir}/")

if __name__ == "__main__":
    main()
