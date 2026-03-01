import librosa
from sklearn.metrics.pairwise import cosine_similarity

# Audio monitoring section replace karo is se:

# ================= AUDIO MONITORING =================
audio_data = stream.read(4096, exception_on_overflow=False)
audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(float)

rms = np.sqrt(np.mean(audio_np**2))

if rms > AUDIO_THRESHOLD:

    # MFCC feature extraction
    mfcc = librosa.feature.mfcc(
        y=audio_np,
        sr=16000,
        n_mfcc=13
    )

    mfcc_mean = np.mean(mfcc.T, axis=0).reshape(1, -1)

    if "previous_mfcc" not in globals():
        previous_mfcc = mfcc_mean
        audio_state = "Speaker1"
        audio_start = current

    else:
        similarity = cosine_similarity(previous_mfcc, mfcc_mean)[0][0]

        # Threshold adjust kar sakti ho (0.75–0.85)
        if similarity < 0.75:
            save_screenshot(frame, "Multiple_Speakers", current)
            timeline_logs.append(f"🔊 Multiple Voices Detected at {current}s")
            previous_mfcc = mfcc_mean
        else:
            previous_mfcc = mfcc_mean