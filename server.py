import asyncio
import json
import threading
import time
import collections
import os
import numpy as np

from obspy.clients.seedlink.easyseedlink import EasySeedLinkClient
from obspy.signal.trigger import classic_sta_lta, trigger_onset
from scipy import signal as scipy_signal
from scipy.ndimage import zoom
import websockets

from datetime import datetime, timezone

# ─────────────────────────────
# CONFIG
# ─────────────────────────────
STATIONS = [
    {"net": "GE", "sta": "UGM",  "cha": "SHZ", "label": "UGMada", "thr_on": 5.0, "thr_off": 0.8},
    {"net": "GE", "sta": "JAGI", "cha": "BHZ", "label": "Banyuwangi", "thr_on": 5.0, "thr_off": 0.8},
    {"net": "GE", "sta": "BBJI", "cha": "BHZ", "label": "Garut", "thr_on": 5.0, "thr_off": 0.8},
    {"net": "GE", "sta": "SMRI", "cha": "BHZ", "label": "Semarang", "thr_on": 5.0, "thr_off": 0.8},
]

WINDOW_SEC = 120
N_FREQ = 64
N_TIME = 200
PUSH_SEC = 1

GEOFON_HOST = "geofon.gfz-potsdam.de"
GEOFON_PORT = 18000

# ─────────────────────────────
# BUFFER
# ─────────────────────────────
buffers = {
    s["sta"]: {
        "data": collections.deque(maxlen=WINDOW_SEC * 100),
        "status": "init",
        "magnitude": None,
        "triggered": False,
        "sr": 100.0,
    } for s in STATIONS
}

spec_buffers = {
    s["sta"]: collections.deque(maxlen=N_TIME)
    for s in STATIONS
}

lock = threading.Lock()

# ─────────────────────────────
# SEEDLINK
# ─────────────────────────────
class MultiStationClient(EasySeedLinkClient):
    def on_data(self, trace):
        sta = trace.stats.station
        if sta not in buffers:
            return

        with lock:
            buf = buffers[sta]
            sr = float(trace.stats.sampling_rate)
            buf["sr"] = sr

            new_maxlen = int(WINDOW_SEC * sr)
            if buf["data"].maxlen != new_maxlen:
                buf["data"] = collections.deque(buf["data"], maxlen=new_maxlen)

            for sample in trace.data:
                buf["data"].append(float(sample))

            buf["status"] = "live"

            # Trigger
            arr = np.array(buf["data"])
            sr_int = int(sr)

            cfg = next((c for c in STATIONS if c["sta"] == sta), None)

            if cfg and len(arr) > sr_int * 20:
                cft = classic_sta_lta(arr, int(1 * sr_int), int(10 * sr_int))
                on_off = trigger_onset(cft, cfg["thr_on"], cfg["thr_off"])
                buf["triggered"] = len(on_off) > 0

                if buf["triggered"]:
                    peak = float(np.abs(arr).max())
                    if peak > 0:
                        ml = np.log10(peak) + 3 * np.log10(8.0 * 500 / 111.19) - 2.92
                        buf["magnitude"] = round(ml, 2)
                else:
                    buf["magnitude"] = None

    def on_seedlink_error(self):
        print("SeedLink error...")

# ─────────────────────────────
# THREAD SEEDLINK
# ─────────────────────────────
def run_seedlink():
    while True:
        try:
            client = MultiStationClient(f"{GEOFON_HOST}:{GEOFON_PORT}")
            for cfg in STATIONS:
                client.select_stream(cfg["net"], cfg["sta"], cfg["cha"])

            print("SeedLink terhubung")
            client.run()

        except Exception as e:
            print("SeedLink error:", e)
            time.sleep(10)

threading.Thread(target=run_seedlink, daemon=True).start()

# ─────────────────────────────
# SPECTROGRAM (1 KOLOM)
# ─────────────────────────────
def compute_spec_column(data, sr):
    arr = np.array(data, dtype=float)
    sr_int = int(sr)

    if len(arr) < sr_int * 4:
        return None

    try:
        nperseg = int(sr * 4)
        noverlap = int(nperseg * 0.5)

        f, t, Sxx = scipy_signal.spectrogram(
            arr,
            fs=sr,
            nperseg=nperseg,
            noverlap=noverlap,
            nfft=nperseg * 2,
            window="hann"
        )

        freq_mask = (f >= 0.5) & (f <= 10.0)
        Sxx_cut = Sxx[freq_mask, :]

        if Sxx_cut.size == 0:
            return None

        col = Sxx_cut[:, -1]

        col_db = 10 * np.log10(col + 1e-10)

        noise = np.median(col_db)
        col_norm = np.clip((col_db - noise) / 50.0, 0, 1)

        if len(col_norm) != N_FREQ:
            col_norm = zoom(col_norm, N_FREQ / len(col_norm), order=1)

        # balik (low freq di bawah)
        col_norm = col_norm[::-1]

        return (col_norm * 255).astype(np.uint8).tolist()

    except Exception as e:
        print("Spec error:", e)
        return None

# ─────────────────────────────
# WEBSOCKET
# ─────────────────────────────
async def handler(websocket):
    print("Client connected")

    try:
        while True:
            now_ts = datetime.now(timezone.utc).timestamp()

            with lock:
                payload = []

                for cfg in STATIONS:
                    sta = cfg["sta"]
                    buf = buffers[sta]

                    data = list(buf["data"])
                    sr = buf["sr"]

                    col = compute_spec_column(data, sr)

                    if col is not None:
                        spec_buffers[sta].append(col)

                    spec = list(spec_buffers[sta])

                    payload.append({
                        "station": sta,
                        "label": cfg["label"],
                        "spec": spec,
                        "timestamp": now_ts,
                        "status": buf["status"],
                        "triggered": buf["triggered"],
                        "magnitude": buf["magnitude"],
                        "sr": sr
                    })

            await websocket.send(json.dumps(payload))
            await asyncio.sleep(PUSH_SEC)

    except Exception as e:
        print("Client disconnect:", e)

# ─────────────────────────────
# MAIN
# ─────────────────────────────
async def main():
    print("Menunggu data awal...")
    await asyncio.sleep(10)

    port = int(os.environ.get("PORT", 8765))

    async with websockets.serve(handler, "0.0.0.0", port):
        print("WebSocket jalan di port", port)
        await asyncio.Future()

if __name__ == "__main__":
    asyncio.run(main())
