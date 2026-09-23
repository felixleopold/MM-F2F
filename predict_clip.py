"""Run the released MM-F2F checkpoint on one recorded conversation clip.

The transcript is supplied by the caller, so this does not load WhisperX. The
video should show the speaker whose face is used for the prediction.
"""

import argparse
import json
import subprocess
import time

import cv2
import imageio_ffmpeg
import numpy as np
import torch

from model.mm import LanguageAudioVisionModel, load_inference_processors


LABELS = ("keep", "turn-taking", "backchannel")
SAMPLE_RATE = 16000
FRAME_COUNT = 16


def choose_device(requested):
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def read_audio(path, end, duration):
    cmd = [imageio_ffmpeg.get_ffmpeg_exe(), "-v", "error", "-i", path]
    if end is not None:
        cmd += ["-to", str(end)]
    cmd += ["-vn", "-ac", "1", "-ar", str(SAMPLE_RATE), "-f", "f32le", "-"]
    result = subprocess.run(cmd, check=True, capture_output=True)
    audio = np.frombuffer(result.stdout, dtype="<f4")
    audio = audio[-int(duration * SAMPLE_RATE):]
    if audio.size == 0:
        raise ValueError("The clip has no audio in the selected window")
    return audio.copy()


def read_frames(path, end):
    cap = cv2.VideoCapture(path)
    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if fps <= 0 or count < FRAME_COUNT:
            raise ValueError("The clip needs at least 16 decodable video frames")
        last = min(count, max(FRAME_COUNT, round(end * fps))) if end is not None else count
        cap.set(cv2.CAP_PROP_POS_FRAMES, last - FRAME_COUNT)
        frames = []
        for _ in range(FRAME_COUNT):
            ok, frame = cap.read()
            if not ok:
                raise ValueError("Could not decode the final 16 frames")
            frames.append(frame)
        return frames
    finally:
        cap.release()


def crop_faces(frames, face_box):
    detector = None
    if face_box is None:
        detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        if detector.empty():
            raise RuntimeError("OpenCV face detector is unavailable")
    cropped = []
    previous_box = None
    for frame in frames:
        height, width = frame.shape[:2]
        if face_box is not None:
            x, y, w, h = face_box
        else:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            boxes = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)
            if len(boxes):
                x, y, w, h = max(boxes, key=lambda box: box[2] * box[3])
                previous_box = (x, y, w, h)
            elif previous_box is not None:
                x, y, w, h = previous_box
            else:
                raise ValueError("No face found; provide --face-box x y width height")
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(width, x + w), min(height, y + h)
        if x2 <= x1 or y2 <= y1:
            raise ValueError("Face box falls outside the video frame")
        cropped.append(cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2RGB))
    return cropped


class Predictor:
    def __init__(self, checkpoint, device="auto"):
        self.device = choose_device(device)
        self.tokenizer, self.clean_text, self.audio_processor, self.video_processor = load_inference_processors()
        device = self.device
        started = time.perf_counter()
        model = LanguageAudioVisionModel(pretrained=False)
        state = torch.load(checkpoint, map_location="cpu", weights_only=True, mmap=True)
        missing, unexpected = model.load_state_dict(state, strict=False)
        unused_heads = {
            f"{name}.out_layer.{part}"
            for name in ("text_model", "audio_model", "vision_model")
            for part in ("weight", "bias")
        }
        if set(missing) != unused_heads or unexpected:
            raise RuntimeError(f"Checkpoint mismatch: missing={missing}, unexpected={unexpected}")
        del state
        self.model = model.to(device).eval()
        self.load_seconds = round(time.perf_counter() - started, 3)

    def predict(self, text, audio, frames, face_box=None, repeat=1, faces_cropped=False):
        started = time.perf_counter()
        if repeat < 1:
            raise ValueError("repeat must be at least 1")
        text = self.clean_text(text)
        if not text:
            raise ValueError("Transcript must contain spoken words")
        if len(frames) != FRAME_COUNT:
            raise ValueError("Exactly 16 video frames are required")
        text_input = self.tokenizer(text, return_tensors="pt").input_ids.to(self.device)
        audio_input = self.audio_processor(audio, sampling_rate=SAMPLE_RATE, return_tensors="pt").input_values.to(self.device)
        faces = ([cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in frames]
                 if faces_cropped else crop_faces(frames, face_box))
        video_input = self.video_processor(faces, return_tensors="pt").pixel_values.to(self.device)
        prepared_s = time.perf_counter() - started
        runs = []
        with torch.inference_mode():
            for _ in range(repeat):
                tick = time.perf_counter()
                logits = self.model(text_input, audio_input, video_input)
                probabilities = torch.softmax(logits, dim=-1)[0].cpu().tolist()
                if self.device == "mps":
                    torch.mps.synchronize()
                runs.append(round(time.perf_counter() - tick, 3))
        return {
            "label": LABELS[int(np.argmax(probabilities))],
            "probabilities": dict(zip(LABELS, probabilities)),
            "device": self.device,
            "seconds": {"prepare": round(prepared_s, 3), "inference_runs": runs},
        }


def predict(args):
    started = time.perf_counter()
    predictor = Predictor(args.checkpoint, args.device)
    load_s = time.perf_counter() - started
    audio = read_audio(args.input, args.end, args.audio_seconds)
    frames = read_frames(args.input, args.end)
    result = predictor.predict(args.text, audio, frames, args.face_box, args.repeat)
    result["seconds"]["load"] = round(load_s, 3)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Video file with audio")
    parser.add_argument("--checkpoint", required=True, help="Released multi-modal-deid.pt")
    parser.add_argument("--text", required=True, help="Transcript up to the prediction point")
    parser.add_argument("--end", type=float, help="Prediction time in seconds; default is clip end")
    parser.add_argument("--audio-seconds", type=float, default=8, help="Trailing audio window (default: 8)")
    parser.add_argument("--face-box", type=int, nargs=4, metavar=("X", "Y", "W", "H"),
                        help="Speaker face box in pixels; default: detect largest face")
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    parser.add_argument("--repeat", type=int, default=1, help="Repeat inference to measure warm latency")
    args = parser.parse_args()
    print(json.dumps(predict(args), indent=2))


if __name__ == "__main__":
    main()
