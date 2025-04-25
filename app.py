import os
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from scipy.ndimage import maximum_filter
import hashlib
import sqlite3
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor


def generate_spectrogram(audio_file, n_fft=1024, hop_length=512):
    """Convert audio file into a spectrogram with optimized FFT size."""
    y, sr = librosa.load(audio_file, sr=None, duration=30)  
    S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length))  # Compute STFT
    S_db = librosa.amplitude_to_db(S, ref=np.max)  # Convert to decibels
    return S_db, sr



def find_peaks(S_db, percentile=95, neighborhood_size=(20, 20), min_freq=100, max_freq=4000, sr=22050, max_peaks_per_time_bin=3):
    """Optimized peak detection: adaptive threshold, frequency filtering, and peak density control."""
    threshold = np.percentile(S_db, percentile)  
    local_max = maximum_filter(S_db, neighborhood_size)  
    peaks = (S_db == local_max) & (S_db > threshold)  

 
    freqs = librosa.fft_frequencies(sr=sr, n_fft=1024)

   
    valid_peaks = [p for p in np.argwhere(peaks) if min_freq <= freqs[p[0]] <= max_freq]
    valid_peaks = np.array(valid_peaks)

    
    if len(valid_peaks) > 0:
        unique_times = np.unique(valid_peaks[:, 1])  
        filtered_peaks = []
        for t in unique_times:
            peaks_at_t = valid_peaks[valid_peaks[:, 1] == t]
            sorted_peaks = sorted(peaks_at_t, key=lambda x: S_db[x[0], x[1]], reverse=True)
            filtered_peaks.extend(sorted_peaks[:max_peaks_per_time_bin])  

        return np.array(filtered_peaks)

    return valid_peaks



def generate_fingerprints(peaks, fan_out=10):
    """Create hashes from anchor-target peak pairs."""
    fingerprints = []
    for i in range(len(peaks)):
        for j in range(1, fan_out):
            if i + j < len(peaks):
                f1, t1 = peaks[i]
                f2, t2 = peaks[i + j]
                delta_t = t2 - t1
                if 0 < delta_t < 200:  
                    hash_string = f"{f1}|{f2}|{delta_t}"
                    hash_value = hashlib.sha1(hash_string.encode()).hexdigest()[:10]
                    fingerprints.append((hash_value, t1))  
    return fingerprints



def create_database():
    """Initialize the SQLite database with indexing."""
    conn = sqlite3.connect('shazam.db')
    cursor = conn.cursor()


    cursor.execute('''
        CREATE TABLE IF NOT EXISTS fingerprints (
            hash TEXT NOT NULL, 
            time_offset INTEGER NOT NULL, 
            song_id INTEGER NOT NULL
        )
    ''')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS songs (
            song_id INTEGER PRIMARY KEY AUTOINCREMENT, 
            song_name TEXT NOT NULL
        )
    ''')

    cursor.execute("CREATE INDEX IF NOT EXISTS idx_fingerprint_hash ON fingerprints (hash)")

    conn.commit()
    conn.close()


def insert_song_fingerprints(song_name, fingerprints):
    """Batch insert fingerprints into the database for speed."""
    conn = sqlite3.connect('shazam.db')
    cursor = conn.cursor()

    cursor.execute("INSERT INTO songs (song_name) VALUES (?)", (song_name,))
    song_id = cursor.lastrowid  

    cursor.executemany("INSERT INTO fingerprints (hash, time_offset, song_id) VALUES (?, ?, ?)",
                       [(hash_value, time_offset, song_id) for hash_value, time_offset in fingerprints])

    conn.commit()
    conn.close()


def process_song(file_path):
    """Process a single song file."""
    try:
        spectrogram, sr = generate_spectrogram(file_path)
        peaks = find_peaks(spectrogram)
        fingerprints = generate_fingerprints(peaks)
        insert_song_fingerprints(os.path.basename(file_path), fingerprints)
    except Exception as e:
        print(f"Error processing {file_path}: {e}")

def process_folder(folder_path):
    """Process all songs in a folder in parallel."""
    create_database()  

    files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.mp3') or f.endswith('.wav')]
    
    if not files:
        print("No audio files found in the folder.")
        return

    print(f"Processing {len(files)} songs...")

    with ProcessPoolExecutor() as executor:
        list(tqdm(executor.map(process_song, files), total=len(files), desc="Processing Songs"))

    print("All songs processed and stored in the database!")



def match_fingerprints(sample_fingerprints):
    """Compare sample fingerprints using database hashing."""
    conn = sqlite3.connect('shazam.db')
    cursor = conn.cursor()

    matches = {}

    for hash_value, sample_time in sample_fingerprints:
        cursor.execute("SELECT time_offset, song_id FROM fingerprints WHERE hash = ?", (hash_value,))
        results = cursor.fetchall()

        for db_time, song_id in results:
            
            if isinstance(db_time, bytes):  
                db_time = int.from_bytes(db_time, byteorder='little')
            else:
                db_time = int(db_time)  # Ensure integer format

            delta_t = db_time - sample_time  # Time offset difference
            matches[(song_id, delta_t)] = matches.get((song_id, delta_t), 0) + 1

    conn.close()

    if not matches:
        return None, 0

    best_match = max(matches, key=matches.get)

    # Fetch the song name
    conn = sqlite3.connect('shazam.db')
    cursor = conn.cursor()
    cursor.execute("SELECT song_name FROM songs WHERE song_id = ?", (best_match[0],))
    result = cursor.fetchone()
    conn.close()

    return (result[0], matches[best_match]) if result else (None, 0)


def recognize_song(sample_path):
    """Recognize a recorded song sample using optimized matching."""
    try:
        spectrogram, sr = generate_spectrogram(sample_path)
        peaks = find_peaks(spectrogram)
        sample_fingerprints = generate_fingerprints(peaks)
        song_name, confidence = match_fingerprints(sample_fingerprints)

        if song_name:
            print(f" Matched Song: {song_name} (Confidence: {confidence})")
        else:
            print(" No match found.")

    except Exception as e:
        print(f" Error processing sample: {e}")

if __name__ == "__main__":
    folder_path = "songs/"  # Change this to your folder path
    process_folder(folder_path)

    sample_path = "Analyze/Analyze.mp3"  # Change this to your sample file path
    recognize_song(sample_path)
