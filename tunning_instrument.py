import numpy as np
import sounddevice as sd
import aubio
import math
import queue
import sys

SAMPLE_RATE = 48000
HOP_SIZE    = 1024            # 프레임 간 겹침 거리(작을수록 반응 빠름)
BUFFER_SIZE = 4               # 오디오 콜백 큐 크기

q = queue.Queue(maxsize=BUFFER_SIZE)

# 마이크 콜백: 들어온 오디오를 큐에 쌓기
def audio_callback(indata, frames, time, status):
    if status:
        print(status, file=sys.stderr)
    # mono로 변환
    mono = np.mean(indata, axis=1).astype(np.float32)
    try:
        q.put_nowait(mono)
    except queue.Full:
        pass

# 피치 검출기: YIN 방식 (정확·안정적)
pitch_o = aubio.pitch("yin", 2048, HOP_SIZE, SAMPLE_RATE)
pitch_o.set_unit("Hz")
pitch_o.set_silence(-40)     # dBFS, 이보다 작으면 무음으로 간주

# 주파수→MIDI→음이름/센트
NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F",
              "F#", "G", "G#", "A", "A#", "B"]

def freq_to_note_and_cents(f):
    if f <= 0:
        return None
    # MIDI 번호: 69 = A4(440Hz)
    midi = 69 + 12 * math.log2(f / 440.0)
    midi_round = int(round(midi))
    note_name = NOTE_NAMES[midi_round % 12]
    octave = midi_round // 12 - 1
    # 해당 MIDI의 기준 주파수
    f_ref = 440.0 * (2 ** ((midi_round - 69) / 12))
    # cent 오차
    cents = 1200 * math.log2(f / f_ref)
    return f"{note_name}{octave}", cents, f_ref

# 스트림 시작
with sd.InputStream(channels=1, samplerate=SAMPLE_RATE,
                    blocksize=HOP_SIZE, callback=audio_callback):
    print("튜너 시작! (Ctrl+C로 종료)")
    while True:
        buf = q.get()
        # aubio는 프레임 단위로 처리
        f0 = pitch_o(buf)[0]
        conf = pitch_o.get_confidence()  # 0~1 신뢰도
        if conf < 0.8 or f0 < 20:        # 너무 낮은 신뢰도/주파수는 무시
            continue
        note = freq_to_note_and_cents(f0)
        if note is None:
            continue
        name, cents, ref = note
        direction = "↑ 올려요" if cents < -3 else ("↓ 내려요" if cents > 3 else "✓ 맞아요")
        print(f"freq={f0:7.2f} Hz | target={name}({ref:7.2f} Hz) | offset={cents:6.1f} cents {direction}")
