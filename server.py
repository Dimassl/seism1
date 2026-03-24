import asyncio, json, threading, time, collections
import numpy as np
from obspy.clients.seedlink.easyseedlink import EasySeedLinkClient
from obspy.signal.trigger import classic_sta_lta, trigger_onset
from scipy import signal as scipy_signal
from scipy.ndimage import zoom
import websockets
import os
from datetime import datetime, timezone

STATIONS = [
    {"net": "GE", "sta": "UGM",  "cha": "SHZ", "label": "Yogyakarta",   "thr_on": 5.0, "thr_off": 0.8},
    {"net": "GE", "sta": "JAGI", "cha": "BHZ", "label": "Jajag",        "thr_on": 5.0, "thr_off": 0.8},
    {"net": "GE", "sta": "BBJI", "cha": "BHZ", "label": "Garut",        "thr_on": 5.0, "thr_off": 0.8},
    {"net": "GE", "sta": "SMRI", "cha": "BHZ", "label": "Semarang",     "thr_on": 5.0, "thr_off": 0.8},
]

WINDOW_SEC  = 120
GEOFON_HOST = "geofon.gfz-potsdam.de"
GEOFON_PORT = 18000
N_FREQ      = 64
N_NEW_COLS  = 5   # kolom baru per push (tiap 1 detik)

buffers = {s["sta"]: {
    "data"     : collections.deque(maxlen=WINDOW_SEC * 100),
    "status"   : "init",
    "magnitude": None,
    "triggered": False,
    "sr"       : 100.0,
} for s in STATIONS}

lock = threading.Lock()

# ── SeedLink Client ───────────────────────────────────────────
class MultiStationClient(EasySeedLinkClient):
    def on_data(self, trace):
    print(f"Data masuk: {trace.id} | {trace.stats.npts} samples | {trace.stats.sampling_rate} Hz")
    sta = trace.stats.station
    if sta not in buffers:
        print(f"Stasiun {sta} tidak ada di buffers!")
        return
    with lock:
        
            buf = buffers[sta]
            sr  = float(trace.stats.sampling_rate)
            buf["sr"] = sr

            new_maxlen = int(WINDOW_SEC * sr)
            if buf["data"].maxlen != new_maxlen:
                buf["data"] = collections.deque(buf["data"], maxlen=new_maxlen)

            for s in trace.data:
                buf["data"].append(float(s))
            buf["status"] = "live"

            arr    = np.array(buf["data"])
            sr_int = int(sr)
            cfg    = next((c for c in STATIONS if c["sta"] == sta), None)
            if cfg and len(arr) > sr_int * 20:
                cft    = classic_sta_lta(arr, int(1 * sr_int), int(10 * sr_int))
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
        print("SeedLink error, reconnecting...")

# ── SeedLink Runner ───────────────────────────────────────────
def run_seedlink():
    while True:
        try:
            client = MultiStationClient(f"{GEOFON_HOST}:{GEOFON_PORT}")
            for cfg in STATIONS:
                client.select_stream(cfg["net"], cfg["sta"], cfg["cha"])
            print("SeedLink terhubung!")
            client.run()
        except Exception as e:
            print(f"SeedLink error: {e}, reconnect dalam 10 detik...")
            time.sleep(10)

threading.Thread(target=run_seedlink, daemon=True).start()

# ── Spektrogram — kirim kolom baru saja ──────────────────────
def get_latest_spec_cols(data, sr, n_freq=N_FREQ, n_cols=N_NEW_COLS):
    """Hitung spektrogram, kembalikan hanya n_cols kolom terbaru"""
    arr = np.array(data, dtype=float)
    if len(arr) < int(sr) * 2:
        return []
    try:
        nperseg  = min(128, len(arr) // 4)
        noverlap = min(64,  len(arr) // 8)

        f, t, Sxx = scipy_signal.spectrogram(
            arr, fs=sr,
            nperseg  = nperseg,
            noverlap = noverlap,
            nfft     = 256,
        )
        freq_mask = (f >= 0.5) & (f <= 10.0)
        Sxx_cut   = Sxx[freq_mask, :]
        if Sxx_cut.size == 0:
            return []

        Sxx_db     = 10 * np.log10(Sxx_cut + 1e-10)
        vmin, vmax = Sxx_db.min(), Sxx_db.max()
        Sxx_norm   = (Sxx_db - vmin) / (vmax - vmin) if vmax > vmin else np.zeros_like(Sxx_db)

        if Sxx_norm.shape[0] != n_freq:
            Sxx_norm = zoom(Sxx_norm, (n_freq / Sxx_norm.shape[0], 1))

        # Ambil n_cols kolom terakhir, transpose jadi list of columns
        last = Sxx_norm[:, -n_cols:]            # shape: (n_freq, n_cols)
        return np.round(last.T, 3).tolist()     # shape: (n_cols, n_freq)
    except Exception as e:
        print(f"Spektrogram error: {e}")
        return []

# ── WebSocket Handler ─────────────────────────────────────────
async def handler(websocket):
    print(f"Client terhubung: {websocket.remote_address}")
    try:
        while True:
            now_utc = datetime.now(timezone.utc).timestamp()
            with lock:
                payload = [{
                    "station"  : cfg["sta"],
                    "label"    : cfg["label"],
                    "data"     : list(buffers[cfg["sta"]]["data"])[-500:],
                    "spec_new" : get_latest_spec_cols(
                        list(buffers[cfg["sta"]]["data"]),
                        buffers[cfg["sta"]]["sr"],
                    ),
                    "spec_time": now_utc,   # timestamp UTC kolom terbaru
                    "status"   : buffers[cfg["sta"]]["status"],
                    "triggered": buffers[cfg["sta"]]["triggered"],
                    "magnitude": buffers[cfg["sta"]]["magnitude"],
                    "sr"       : buffers[cfg["sta"]]["sr"],
                } for cfg in STATIONS]
            await websocket.send(json.dumps(payload))
            await asyncio.sleep(1)
    except websockets.exceptions.ConnectionClosed:
        print("Client disconnect")

# ── Main ──────────────────────────────────────────────────────
async def main():
    print("Menunggu SeedLink data (10 detik)...")
    await asyncio.sleep(10)
    port = int(os.environ.get("PORT", 8765))
    async with websockets.serve(handler, "0.0.0.0", port):
        print(f"WebSocket server jalan di port {port}")
        await asyncio.Future()

asyncio.run(main())
