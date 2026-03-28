import asyncio, json, threading, time, collections
import numpy as np
from obspy.clients.seedlink.easyseedlink import EasySeedLinkClient
from obspy.signal.trigger import classic_sta_lta, trigger_onset
from scipy import signal as scipy_signal
from scipy.ndimage import zoom
import websockets, os
from datetime import datetime, timezone

STATIONS = [
    {"net": "GE", "sta": "UGM",  "cha": "SHZ", "label": "UGMada", "thr_on": 5.0, "thr_off": 0.8},
    {"net": "GE", "sta": "JAGI", "cha": "BHZ", "label": "Banyuwangi",      "thr_on": 5.0, "thr_off": 0.8},
    {"net": "GE", "sta": "BBJI", "cha": "BHZ", "label": "Garut",      "thr_on": 5.0, "thr_off": 0.8},
    {"net": "GE", "sta": "SMRI", "cha": "BHZ", "label": "Semarang",   "thr_on": 5.0, "thr_off": 0.8},
]

WINDOW_SEC  = 120
GEOFON_HOST = "geofon.gfz-potsdam.de"
GEOFON_PORT = 18000
N_FREQ      = 48
PUSH_SEC    = 3     # hitung spektrogram dari 3 detik terakhir tiap push

buffers = {s["sta"]: {
    "data"     : collections.deque(maxlen=WINDOW_SEC * 100),
    "status"   : "init",
    "magnitude": None,
    "triggered": False,
    "sr"       : 100.0,
} for s in STATIONS}

lock = threading.Lock()

class MultiStationClient(EasySeedLinkClient):
    def on_data(self, trace):
        sta = trace.stats.station
        if sta not in buffers:
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
        print("SeedLink error...")

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

def compute_new_cols(data, sr, push_sec=PUSH_SEC, n_freq=N_FREQ):
    """
    Ambil push_sec detik terakhir data, hitung spektrogram,
    return semua kolomnya — ini yang 'baru' sejak push terakhir
    """
    arr = np.array(data, dtype=float)
    sr_int = int(sr)

    # Butuh minimal 2x push_sec supaya ada konteks untuk normalisasi
    n_context = int(sr * WINDOW_SEC)
    n_new     = int(sr * push_sec)

    if len(arr) < n_new * 2:
        return []

    try:
        # Pakai semua data untuk normalisasi warna yang konsisten
        nperseg  = min(256, sr_int * 2)
        noverlap = int(nperseg * 0.875)

        f, t, Sxx = scipy_signal.spectrogram(
            arr[-n_context:] if len(arr) > n_context else arr,
            fs       = sr,
            nperseg  = nperseg,
            noverlap = noverlap,
            nfft     = 512,
            window   = 'hann',
        )

        freq_mask = (f >= 0.5) & (f <= 10.0)
        Sxx_cut   = Sxx[freq_mask, :]
        if Sxx_cut.size == 0:
            return []

        Sxx_db = 10 * np.log10(Sxx_cut + 1e-10)

        # Normalisasi pakai persentil dari seluruh window — warna konsisten
        vmin = np.percentile(Sxx_db, 5)
        vmax = np.percentile(Sxx_db, 98)
        Sxx_norm = np.clip((Sxx_db - vmin) / (vmax - vmin + 1e-10), 0, 1)

        if Sxx_norm.shape[0] != n_freq:
            Sxx_norm = zoom(Sxx_norm, (n_freq / Sxx_norm.shape[0], 1), order=1)

        # Hitung berapa kolom yang sesuai push_sec detik
        # Setiap kolom = (nperseg - noverlap) / sr detik
        step_sec    = (nperseg - noverlap) / sr
        n_new_cols  = max(1, int(push_sec / step_sec))

        # Ambil hanya n_new_cols kolom terakhir
        new_cols = Sxx_norm[:, -n_new_cols:]   # (n_freq, n_new_cols)

        # Transpose → list of columns, tiap kolom = n_freq nilai
        return np.round(new_cols.T, 3).tolist()  # (n_new_cols, n_freq)

    except Exception as e:
        print(f"Spektrogram error: {e}")
        return []

async def handler(websocket):
    print(f"Client terhubung: {websocket.remote_address}")
    try:
        while True:
            now_ts = datetime.now(timezone.utc).timestamp()
            with lock:
                payload = []
                for cfg in STATIONS:
                    buf  = buffers[cfg["sta"]]
                    data = list(buf["data"])
                    sr   = buf["sr"]

                    cols = compute_new_cols(data, sr)

                    payload.append({
                        "station"  : cfg["sta"],
                        "label"    : cfg["label"],
                        "spec_cols": cols,
                        "timestamp": now_ts,
                        "status"   : buf["status"],
                        "triggered": buf["triggered"],
                        "magnitude": buf["magnitude"],
                        "sr"       : sr,
                    })

            await websocket.send(json.dumps(payload))
            await asyncio.sleep(PUSH_SEC)   # push tiap PUSH_SEC detik

    except websockets.exceptions.ConnectionClosed:
        print("Client disconnect")

async def main():
    print("Menunggu SeedLink data (15 detik)...")
    await asyncio.sleep(15)
    port = int(os.environ.get("PORT", 8765))
    async with websockets.serve(handler, "0.0.0.0", port):
        print(f"WebSocket server jalan di port {port}")
        await asyncio.Future()

asyncio.run(main())
