import torch
import os
import sys

print("=" * 60)
print("TTS DIAGNOSTIC TEST")
print("=" * 60)

# 1. Check audio players
print("\n1. Checking audio players...")
players = {
    "sox (play)": "which play",
    "alsa (aplay)": "which aplay", 
    "macOS (afplay)": "which afplay"
}

found_player = False
for name, cmd in players.items():
    if os.system(f"{cmd} > /dev/null 2>&1") == 0:
        print(f"   ✅ {name} - FOUND")
        found_player = True
    else:
        print(f"   ❌ {name} - NOT FOUND")

if not found_player:
    print("\n   ⚠️  NO AUDIO PLAYER FOUND!")
    print("   Install one of these:")
    print("   - Ubuntu/Debian: sudo apt-get install sox")
    print("   - Fedora: sudo dnf install sox")
    print("   - macOS: (afplay is built-in)")
    print("   - Arch: sudo pacman -S sox")

# 2. Check CUDA
print("\n2. Checking CUDA availability...")
print(f"   CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"   CUDA device: {torch.cuda.get_device_name(0)}")

# 3. Test TTS
print("\n3. Testing TTS models...")
print("   (This will download models on first run - may take time)")

try:
    print("\n   Testing Bark TTS...")
    from transformers import pipeline
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"   Using device: {device}")
    
    tts = pipeline(
        "text-to-speech",
        model="suno/bark-small",
        device=0 if device == "cuda" else -1
    )
    
    print("   ✅ Model loaded successfully")
    print("   Generating speech for: 'Hello, this is a test'")
    
    speech = tts("Hello, this is a test", forward_params={"do_sample": True})
    
    print(f"   ✅ Speech generated")
    print(f"   Sample rate: {speech['sampling_rate']}")
    print(f"   Audio shape: {speech['audio'].shape}")
    
    # Save to file
    import scipy.io.wavfile as wavfile
    output_file = "test_audio.wav"
    wavfile.write(
        output_file,
        rate=speech["sampling_rate"],
        data=speech["audio"].squeeze()
    )
    print(f"   ✅ Saved to: {output_file}")
    
    # Try to play
    print("\n4. Attempting playback...")
    if os.system("which play > /dev/null 2>&1") == 0:
        print("   Playing with sox...")
        os.system(f"play {output_file}")
    elif os.system("which aplay > /dev/null 2>&1") == 0:
        print("   Playing with aplay...")
        os.system(f"aplay {output_file}")
    elif os.system("which afplay > /dev/null 2>&1") == 0:
        print("   Playing with afplay...")
        os.system(f"afplay {output_file}")
    else:
        print("   ⚠️  No player available. Audio saved to test_audio.wav")
        print(f"   Try playing manually: play {output_file}")
    
    print(f"\n✅ TTS TEST COMPLETE")
    print(f"   If you didn't hear anything, check your speakers/volume")
    print(f"   Or play the file manually: {output_file}")
    
except Exception as e:
    print(f"   ❌ ERROR: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)