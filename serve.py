"""Small private HTTP worker for MM-F2F predictions from Reachy."""

import argparse
import base64
import json
import os
from http.server import BaseHTTPRequestHandler, HTTPServer

import cv2
import numpy as np

from predict_clip import FRAME_COUNT, Predictor


MAX_REQUEST_BYTES = 8 * 1024 * 1024


class PredictionHandler(BaseHTTPRequestHandler):
    predictor = None
    token = None

    def do_GET(self):
        if self.path == "/health":
            self.respond(200, {"ready": self.predictor is not None})
        else:
            self.respond(404, {"error": "not found"})

    def do_POST(self):
        if self.path != "/predict":
            return self.respond(404, {"error": "not found"})
        if self.token and self.headers.get("Authorization") != f"Bearer {self.token}":
            return self.respond(401, {"error": "unauthorized"})
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= MAX_REQUEST_BYTES:
                raise ValueError("request size must be between 1 byte and 8 MB")
            payload = json.loads(self.rfile.read(length))
            pcm = base64.b64decode(payload["audio_pcm16_b64"], validate=True)
            audio = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0
            if not 0 < audio.size <= 25 * 16000:
                raise ValueError("audio must contain at most 25 seconds of 16 kHz mono PCM16")
            encoded = payload["frames_jpeg_b64"]
            if len(encoded) != FRAME_COUNT:
                raise ValueError("exactly 16 JPEG frames are required")
            frames = []
            for item in encoded:
                jpg = np.frombuffer(base64.b64decode(item, validate=True), dtype=np.uint8)
                frame = cv2.imdecode(jpg, cv2.IMREAD_COLOR)
                if frame is None:
                    raise ValueError("invalid JPEG frame")
                frames.append(frame)
            result = self.predictor.predict(payload["text"], audio, frames,
                                            faces_cropped=bool(payload.get("faces_cropped", False)))
            self.respond(200, result)
        except (KeyError, ValueError, TypeError, json.JSONDecodeError) as exc:
            self.respond(400, {"error": str(exc)})
        except Exception as exc:
            self.respond(500, {"error": str(exc)})

    def respond(self, status, data):
        body = json.dumps(data).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8788)
    args = parser.parse_args()
    PredictionHandler.predictor = Predictor(args.checkpoint, args.device)
    PredictionHandler.token = os.environ.get("MMF2F_TOKEN")
    server = HTTPServer((args.host, args.port), PredictionHandler)
    print(f"MM-F2F ready on {args.host}:{args.port} ({PredictionHandler.predictor.device})", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
