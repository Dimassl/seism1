import asyncio, json, threading, time, collections
import numpy as np
from obspy.clients.seedlink.easyseedlink import EasySeedLinkClient
from obspy.signal.trigger import classic_sta_lta, trigger_onset
from scipy import signal as scipy_signal
from scipy.ndimage import zoom
import websockets, os
from datetime import datetime, timezone

# ================= CONFIG =================
STATIONS = [
    {"net": "GE", "sta": "UGM",  "cha": "SHZ", "label": "WanaGAMA", "thr_on": 7.0, "thr_off": 0.8},
    {"net": "GE", "sta": "JAGI", "cha": "BHZ", "label": "Banyuwangi",      "thr_on": 7.0, "thr_off": 0.8},
    {"net": "GE", "sta": "BBJI", "cha": "BHZ", "label": "Garut",      "thr_on": 7.0, "thr_off": 0.8},
    {"net": "GE", "sta": "SMRI", "cha": "BHZ", "label": "Semarang",   "thr_on": 7.0, "thr_off": 0.8},
    {"net": "GE", "sta": "PLAI", "cha": "BHZ", "label": "Sumbawa",   "thr_on": 7.0, "thr_off": 0.8},
    {"net": "GE", "sta": "MMRI", "cha": "BHZ", "label": "Maumere",   "thr_on": 7.0, "thr_off": 0.8},
    {"net": "GE", "sta": "SOEI", "cha": "BHZ", "label": "Soe Timor",   "thr_on": 7.0, "thr_off": 0.8},
    {"net": "GE", "sta": "SAUI", "cha": "SHZ", "label": "Tanibar",   "thr_on": 7.0, "thr_off": 0.8},
    {"net": "GE", "sta": "BNDI", "cha": "BHZ", "label": "BandaNeira",   "thr_on": 7.0, "thr_off": 0.8},
    {"net": "GE", "sta": "FAKI", "cha": "BHZ", "label": "FakFak",   "thr_on": 7.0, "thr_off": 0.8},
    {"net": "GE", "sta": "SANI", "cha": "SHZ", "label": "Maluku",   "thr_on": 7.0, "thr_off": 0.8},
    {"net": "GE", "sta": "TNTI", "cha": "BHZ", "label": "Ternate",   "thr_on": 7.0, "thr_off": 0.8},
    {"net": "GE", "sta": "LUWI", "cha": "BHZ", "label": "LhokseumaveSulawesi",   "thr_on": 7.0, "thr_off": 0.8},
    {"net": "GE", "sta": "TOLI2", "cha": "BHZ", "label": "ToliToli",   "thr_on": 7.0, "thr_off": 0.8},
    {"net": "GE", "sta": "LHMI", "cha": "BHZ", "label": "Aceh",   "thr_on": 7.0, "thr_off": 0.8},
    {"net": "GE", "sta": "GSI", "cha": "BHZ", "label": "Nias",   "thr_on": 7.0, "thr_off": 0.8},
    {"net": "GE", "sta": "MNAI", "cha": "BHZ", "label": "Bengkulu",   "thr_on": 7.0, "thr_off": 0.8},
]

WINDOW_SEC = 120
PUSH_SEC   = 1   

N_FREQ = 48
N_TIME = 200

GEOFON_HOST = "geofon.gfz-potsdam.de"
GEOFON_PORT = 18000

# ================= BUFFER =================
buffers = {
    s["sta"]: {
        "data": collections.deque(maxlen=WINDOW_SEC * 100),
        "sr": 100.0,
        "triggered": False,
        "magnitude": None,
        "status": "init"
    } for s in STATIONS
}

lock = threading.Lock()

# ================= SEEDLINK =================
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

            buf["data"].extend(trace.data.tolist())
            buf["status"] = "live"

            # ===== DETEKSI GEMPA =====
            arr = np.array(buf["data"])
            if len(arr) > sr * 20:
                cft = classic_sta_lta(arr, int(1 * sr), int(10 * sr))
                on_off = trigger_onset(cft, 5.0, 0.8)

                buf["triggered"] = len(on_off) > 0

                if buf["triggered"]:
                    peak = np.max(np.abs(arr))
                    if peak > 0:
                        ml = np.log10(peak) + 3 * np.log10(8.0 * 500 / 111.19) - 2.92
                        buf["magnitude"] = round(ml, 2)
                else:
                    buf["magnitude"] = None

# ================= RUN SEEDLINK =================
def run_seedlink():
    while True:
        try:
            client = MultiStationClient(f"{GEOFON_HOST}:{GEOFON_PORT}")

            for cfg in STATIONS:
                client.select_stream(cfg["net"], cfg["sta"], cfg["cha"])

            print("SeedLink connected")
            client.run()

        except Exception as e:
            print("Reconnect:", e)
            time.sleep(10)

threading.Thread(target=run_seedlink, daemon=True).start()

# ================= SPECTROGRAM =================
def compute_spec(data, sr):
    arr = np.array(data)
    if len(arr) < sr * 4:
        return []

    try:
        nperseg = int(sr * 4)
        noverlap = int(nperseg * 0.75)

        f, t, Sxx = scipy_signal.spectrogram(
            arr,
            fs=sr,
            nperseg=nperseg,
            noverlap=noverlap,
            window='hann'
        )

        # FILTER FREKUENSI
        mask = (f >= 0.5) & (f <= 10)
        Sxx = Sxx[mask]

        if Sxx.size == 0:
            return []

        # DB SCALE
        Sxx = 10 * np.log10(Sxx + 1e-10)

        # NORMALISASI SWARM STYLE
        noise = np.median(Sxx)
        Sxx = np.clip((Sxx - noise) / 50.0, 0, 1)

        # RESIZE KE FIX SIZE
        Sxx = zoom(Sxx, (N_FREQ / Sxx.shape[0], N_TIME / Sxx.shape[1]), order=1)

        # OUTPUT: freq × time
        return (Sxx * 255).astype(np.uint8).tolist()

    except Exception as e:
        print("Spec error:", e)
        return []

# ================= WEBSOCKET =================
async def handler(websocket):
    print("Client connected")

    try:
        while True:
            now_ts = datetime.now(timezone.utc).timestamp()

            with lock:
                payload = []

                for cfg in STATIONS:
                    buf = buffers[cfg["sta"]]

                    spec = compute_spec(buf["data"], buf["sr"])

                    payload.append({
                        "station": cfg["sta"],
                        "label": cfg["label"],
                        "spec": spec,              
                        "timestamp": now_ts,       
                        "window_sec": WINDOW_SEC,  
                        "triggered": buf["triggered"],
                        "magnitude": buf["magnitude"],
                        "status": buf["status"],
                        "sr": buf["sr"]
                    })

            await websocket.send(json.dumps(payload))
            await asyncio.sleep(PUSH_SEC)

    except websockets.exceptions.ConnectionClosed:
        print("Client disconnected")

# ================= MAIN =================
async def main():
    print("Waiting for data...")
    await asyncio.sleep(10)

    port = int(os.environ.get("PORT", 8765))

    async with websockets.serve(handler, "0.0.0.0", port):
        print(f"WebSocket running on port {port}")
        await asyncio.Future()

asyncio.run(main())
