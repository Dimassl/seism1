import asyncio, json, threading, time, collections, math
import numpy as np
from obspy.clients.seedlink.easyseedlink import EasySeedLinkClient
from obspy.signal.trigger import classic_sta_lta, trigger_onset
from scipy.optimize import minimize
import websockets, os

# ── Stasiun Indonesia di GEOFON ──────────────────────────────
# Semakin banyak stasiun → lokasi makin akurat
STATIONS = [
    # Jawa
    {"net":"GE","sta":"BBJI","cha":"BHZ","lat":-7.22,"lon":107.85,"label":"Garut"},
    {"net":"GE","sta":"UGM", "cha":"SHZ","lat":-7.77,"lon":110.37,"label":"WanaGAMA"},
    {"net":"GE","sta":"JAGI","cha":"BHZ","lat":-8.46,"lon":114.15,"label":"Banyuwangi"},
    {"net":"GE","sta":"SMRI","cha":"BHZ","lat":-7.05,"lon":110.44,"label":"Semarang"}, 
    # Sumatera
    {"net":"GE","sta":"LHMI","cha":"BHZ","lat": 5.23,"lon": 96.95,"label":"Lhokseumawe"},
    {"net":"GE","sta":"MNAI","cha":"BHZ","lat": 4.36,"lon": 102.96,"label":"Bengkulu"},
    {"net":"GE","sta":"GSI","cha":"BHZ","lat": 1.3,"lon": 97.58,"label":"Gunungistoli"},
    {"net":"GE","sta":"PBKT","cha":"BHZ","lat":-1.05,"lon":114.90,"label":"Palangkaraya"},
    # Nusa Tenggara
    {"net":"GE","sta":"SOEI","cha":"BHZ","lat":-9.76,"lon":124.26,"label":"Soe NTT"},
    {"net":"GE","sta":"MMRI","cha":"BHZ","lat":-8.63,"lon":122.24,"label":"Maumere"},
    {"net":"GE","sta":"PLAI","cha":"BHZ","lat":-8.83,"lon":117.78,"label":"Plampang"},
    # Sulawesi
    {"net":"GE","sta":"TOLI2","cha":"BHZ","lat": 1.12,"lon":120.79,"label":"Toli-Toli"},
    {"net":"GE","sta":"LUWI","cha":"BHZ","lat": 1.04,"lon":122.77,"label":"Luwu"},
    #Maluku
    {"net":"GE","sta":"TNTI","cha":"BHZ","lat": 0.77,"lon":127.37,"label":"Ternate"},
    {"net":"GE","sta":"SAUI","cha":"BHZ","lat": -7.98,"lon":131.3,"label":"Tanibar"}, 
    {"net":"GE","sta":"BNDI","cha":"BHZ","lat": -4.52,"lon":129.9,"label":"BandaNeira"},
    # Papua
    {"net":"GE","sta":"FAKI","cha":"BHZ","lat":-2.92,"lon":132.24,"label":"Fak-Fak"},
    {"net":"GE","sta":"GENI","cha":"BHZ","lat":-2.59,"lon":140.17°,"label":"Jayapura"},
]

GEOFON_HOST     = "geofon.gfz-potsdam.de"
GEOFON_PORT     = 18000
P_VELOCITY      = 6.0        # km/s kecepatan P-wave rata-rata kerak
TRIGGER_THR_ON  = 10.5        # STA/LTA threshold trigger
TRIGGER_THR_OFF = 0.8
MIN_STATIONS    = 3          # minimum stasiun untuk hitung lokasi
ASSOC_WINDOW    = 120        # detik — window asosiasi trigger antar stasiun
EARTH_R         = 6371.0     # km radius bumi

# ── State per stasiun ─────────────────────────────────────────
sta_buffers = {}
for s in STATIONS:
    sta_buffers[s["sta"]] = {
        "data"       : collections.deque(maxlen=6000),
        "sr"         : 20.0,
        "triggered"  : False,
        "trigger_time": None,   # UTC epoch saat P-wave tiba
        "peak_amp"   : 0.0,
    }

lock         = threading.Lock()
active_events = {}     # id_event → dict info
connected_ws  = set()  # WebSocket clients

# ── Jarak great-circle (km) ──────────────────────────────────
def dist_km(lat1, lon1, lat2, lon2):
    r   = math.pi / 180
    dlat = (lat2 - lat1) * r
    dlon = (lon2 - lon1) * r
    a   = math.sin(dlat/2)**2 + \
          math.cos(lat1*r)*math.cos(lat2*r)*math.sin(dlon/2)**2
    return 2 * EARTH_R * math.asin(math.sqrt(a))

# ── Estimasi lokasi episenter (grid search + least squares) ──
def locate_epicenter(triggers):
    """
    triggers: list dict {sta, lat, lon, t_arrive}
    Return: (lat, lon, origin_time, residual_rms)
    """
    if len(triggers) < 3:
        return None

    # Rata-rata lokasi stasiun sebagai titik awal
    lat0 = np.mean([t["lat"] for t in triggers])
    lon0 = np.mean([t["lon"] for t in triggers])
    t0   = min(t["t_arrive"] for t in triggers) - 30  # estimasi OT awal

    def residuals(x):
        elat, elon, ot = x
        res = []
        for tr in triggers:
            d   = dist_km(elat, elon, tr["lat"], tr["lon"])
            t_pred = ot + d / P_VELOCITY
            res.append(t_pred - tr["t_arrive"])
        return np.array(res)

    def objective(x):
        r = residuals(x)
        return np.sum(r**2)

    result = minimize(
        objective,
        x0     = [lat0, lon0, t0],
        method = "Nelder-Mead",
        options= {"xatol": 0.01, "fatol": 0.001, "maxiter": 2000}
    )

    if not result.success and result.fun > 100:
        return None

    elat, elon, ot = result.x
    res = residuals(result.x)
    rms = float(np.sqrt(np.mean(res**2)))

    return {
        "lat"      : float(elat),
        "lon"      : float(elon),
        "origin_t" : float(ot),
        "rms"      : rms,
    }

# ── Estimasi magnitudo (Ml) ───────────────────────────────────
def estimate_magnitude(triggers, epicenter):
    mls = []
    for tr in triggers:
        if tr["peak_amp"] <= 0:
            continue
        d  = dist_km(epicenter["lat"], epicenter["lon"], tr["lat"], tr["lon"])
        d  = max(d, 1.0)
        # Richter ML = log10(A) + 3*log10(8.0*delta) - 2.92
        delta = d / 111.19   # derajat
        ml    = math.log10(tr["peak_amp"]) + 3 * math.log10(8.0 * delta) - 2.92
        mls.append(ml)
    if not mls:
        return None
    return round(float(np.median(mls)), 1)

# ── Estimasi MMI ──────────────────────────────────────────────
def estimate_mmi(magnitude, depth_km=10.0):
    if magnitude is None:
        return "I"
    # Wald et al. 1999 — simplified
    mmi = 1.5 * magnitude - 0.5 * math.log10(max(depth_km, 1)) - 1.0
    mmi = max(1, min(10, round(mmi)))
    mmi_str = ["I","II","III","IV","V","VI","VII","VIII","IX","X"]
    return mmi_str[mmi - 1]

# ── Estimasi nama region dari koordinat ──────────────────────
def region_name(lat, lon):
    # Kasar berdasarkan bounding box pulau utama
    regions = [
        (-8.5, -5.5, 105.0, 115.9, "Jawa"),
        (-5.9,  5.7,  95.0, 108.5, "Sumatera"),
        (-4.2,  7.1, 108.0, 118.6, "Kalimantan"),
        (-5.5,  4.0, 119.0, 125.5, "Sulawesi"),
        (-9.2, -7.9, 124.0, 127.5, "Nusa Tenggara"),
        (-2.5,  1.5, 128.0, 135.0, "Maluku"),
        (-9.1, -0.9, 130.5, 141.0, "Papua"),
    ]
    for latmin, latmax, lonmin, lonmax, name in regions:
        if latmin <= lat <= latmax and lonmin <= lon <= lonmax:
            return name
    # Arah dari pusat Indonesia
    clat, clon = -2.5, 118.0
    bearing = math.degrees(math.atan2(lon - clon, lat - clat))
    dirs    = ["Utara","TimurLaut","Timur","Tenggara",
               "Selatan","BaratDaya","Barat","BaratLaut"]
    idx = int((bearing + 202.5) % 360 / 45)
    return f"Perairan {dirs[idx]} Indonesia"

# ── Proses trigger — cek apakah cukup stasiun & hitung ───────
def process_triggers():
    now = time.time()
    with lock:
        triggers = []
        for s in STATIONS:
            buf = sta_buffers[s["sta"]]
            if buf["triggered"] and buf["trigger_time"] is not None:
                if now - buf["trigger_time"] <= ASSOC_WINDOW:
                    triggers.append({
                        "sta"      : s["sta"],
                        "lat"      : s["lat"],
                        "lon"      : s["lon"],
                        "label"    : s["label"],
                        "t_arrive" : buf["trigger_time"],
                        "peak_amp" : buf["peak_amp"],
                    })

    if len(triggers) < MIN_STATIONS:
        return None

    # Cek apakah sudah ada event aktif yang sama
    event_key = f"{int(min(t['t_arrive'] for t in triggers) / 60)}"
    if event_key in active_events:
        return None  # sudah dikirim

    # Hitung lokasi
    epicenter = locate_epicenter(triggers)
    if epicenter is None or epicenter["rms"] > 15:
        return None

    mag   = estimate_magnitude(triggers, epicenter)
    if mag is None or mag < 2.5:
        return None

    depth = 10.0  # default, bisa dikembangkan dengan fase pS-pP
    mmi   = estimate_mmi(mag, depth)
    region = region_name(epicenter["lat"], epicenter["lon"])

    event = {
        "id"         : event_key,
        "lat"        : round(epicenter["lat"], 3),
        "lon"        : round(epicenter["lon"], 3),
        "depth_km"   : depth,
        "magnitude"  : mag,
        "mmi"        : mmi,
        "region"     : region,
        "origin_time": epicenter["origin_t"],
        "n_stations" : len(triggers),
        "stations"   : [t["sta"] for t in triggers],
        "rms_sec"    : round(epicenter["rms"], 2),
        "timestamp"  : time.time(),
    }

    active_events[event_key] = event
    print(f"[EWS] GEMPA! M{mag} {region} lat={epicenter['lat']:.2f} "
          f"lon={epicenter['lon']:.2f} MMI={mmi} n={len(triggers)}")
    return event

# ── SeedLink client ───────────────────────────────────────────
class EWSClient(EasySeedLinkClient):
    def on_data(self, trace):
        sta = trace.stats.station
        if sta not in sta_buffers:
            return
        with lock:
            buf = sta_buffers[sta]
            buf["sr"] = float(trace.stats.sampling_rate)
            # Batas maxlen sesuai SR
            new_maxlen = int(buf["sr"] * 300)  # 5 menit
            if buf["data"].maxlen != new_maxlen:
                buf["data"] = collections.deque(buf["data"], maxlen=new_maxlen)
            for v in trace.data:
                buf["data"].append(float(v))

            # Jalankan STA/LTA
            arr    = np.array(buf["data"])
            sr_int = int(buf["sr"])
            if len(arr) < sr_int * 20:
                return

            cft    = classic_sta_lta(arr, int(0.5*sr_int), int(10*sr_int))
            on_off = trigger_onset(cft, TRIGGER_THR_ON, TRIGGER_THR_OFF)

            if len(on_off) > 0 and not buf["triggered"]:
                buf["triggered"]   = True
                buf["trigger_time"]= time.time()
                buf["peak_amp"]    = float(np.abs(arr[-sr_int*10:]).max())
                print(f"[TRIGGER] {sta} @ {time.strftime('%H:%M:%S')} "
                      f"amp={buf['peak_amp']:.0f}")
            elif len(on_off) == 0 and buf["triggered"]:
                # Reset setelah 120 detik
                if time.time() - buf["trigger_time"] > 120:
                    buf["triggered"]    = False
                    buf["trigger_time"] = None
                    buf["peak_amp"]     = 0.0

    def on_seedlink_error(self):
        print("SeedLink error...")

def run_seedlink():
    while True:
        try:
            client = EWSClient(f"{GEOFON_HOST}:{GEOFON_PORT}")
            for s in STATIONS:
                try:
                    client.select_stream(s["net"], s["sta"], s["cha"])
                    print(f"  Subscribed: {s['sta']}")
                except Exception as e:
                    print(f"  Skip {s['sta']}: {e}")
            print("SeedLink terhubung!")
            client.run()
        except Exception as e:
            print(f"SeedLink reconnect: {e}")
            time.sleep(10)

# ── Pemroses trigger background ───────────────────────────────
async def trigger_processor():
    while True:
        event = process_triggers()
        if event and connected_ws:
            msg = json.dumps({"type": "ews_alert", "data": event})
            dead = set()
            for ws in connected_ws.copy():
                try:
                    await ws.send(msg)
                except Exception:
                    dead.add(ws)
            connected_ws -= dead
        await asyncio.sleep(2)

# ── WebSocket handler ─────────────────────────────────────────
async def handler(websocket):
    connected_ws.add(websocket)
    print(f"EWS client terhubung: {websocket.remote_address}")
    try:
        # Kirim status stasiun saat pertama konek
        with lock:
            status = [{
                "sta"      : s["sta"],
                "label"    : s["label"],
                "lat"      : s["lat"],
                "lon"      : s["lon"],
                "triggered": sta_buffers[s["sta"]]["triggered"],
            } for s in STATIONS]
        await websocket.send(json.dumps({"type": "station_status", "data": status}))

        # Keep alive + kirim status berkala
        while True:
            with lock:
                status = [{
                    "sta"      : s["sta"],
                    "triggered": sta_buffers[s["sta"]]["triggered"],
                } for s in STATIONS]
            await websocket.send(json.dumps({
                "type": "heartbeat",
                "stations": status,
                "time": time.time()
            }))
            await asyncio.sleep(5)
    except websockets.exceptions.ConnectionClosed:
        pass
    finally:
        connected_ws.discard(websocket)
        print("EWS client disconnect")

async def main():
    print("Menunggu data SeedLink (20 detik)...")
    await asyncio.sleep(20)

    port = int(os.environ.get("PORT", 8766))
    async with websockets.serve(handler, "0.0.0.0", port):
        print(f"EWS WebSocket port {port}")
        asyncio.create_task(trigger_processor())
        await asyncio.Future()

# Mulai SeedLink di thread terpisah
threading.Thread(target=run_seedlink, daemon=True).start()
asyncio.run(main())
