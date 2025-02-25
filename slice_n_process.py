import math
import argparse
from pydub import AudioSegment
from app import inference
from scipy.io.wavfile import write
import numpy as np

# Each segment is 5.12 seconds (i.e. 5120 milliseconds)
SEGMENT_LENGTH_MS = 5120

def process_with_audiosr(segment: AudioSegment) -> AudioSegment:
    # Export the AudioSegment as a WAV file, which feels dumb
    # but I didn't want to have to change all of the underlying code
    # to work with AudioSegment objects.
    segment.export("./segment_tmp.wav", format="wav")
    sr, audio = inference("./segment_tmp.wav", "basic", 3.5, 70)
    return audio

def process_audio_file(input_file: str, output_file: str):
    # Load the input audio file
    audio = AudioSegment.from_file(input_file)
    total_length_ms = len(audio)
    num_segments = math.ceil(total_length_ms / SEGMENT_LENGTH_MS)
    
    processed_segments = []
    
    for i in range(num_segments):
        start_ms = i * SEGMENT_LENGTH_MS
        end_ms = start_ms + SEGMENT_LENGTH_MS
        segment = audio[start_ms:end_ms]
        
        # If the segment is shorter than 5.12 seconds, you can choose to pad it with silence.
        if len(segment) < SEGMENT_LENGTH_MS:
            padding = AudioSegment.silent(duration=SEGMENT_LENGTH_MS - len(segment))
            segment = segment + padding
        
        print(f"Processing segment {i+1}/{num_segments} ...")
        processed_segment = process_with_audiosr(segment)
        processed_segments.append(processed_segment)
    
    # Concatenate all processed segments into one AudioSegment
    final_audio = processed_segments[0]
    for seg in processed_segments[1:]:
        final_audio += seg
    
    # If final_audio is in floating point format (e.g., values between -1.0 and 1.0),
    # you may need to scale it to an integer format. For 16-bit PCM:
    if final_audio.dtype != np.int16:
        # Scale floats to int16 range
        final_audio_int16 = np.int16(final_audio * 32767)
    else:
        final_audio_int16 = final_audio

    # Write the numpy array to disk as a WAV file.
    write(output_file, 48000, final_audio_int16)
    print(f"Final processed audio saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Truncate an input audio file into 5.12-second segments, process them using AudioSR, and concatenate the output."
    )
    parser.add_argument("input_file", help="Path to the input audio file")
    parser.add_argument("output_file", help="Path to save the output audio file")
    
    args = parser.parse_args()
    process_audio_file(args.input_file, args.output_file)
