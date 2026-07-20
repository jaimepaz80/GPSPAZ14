import os
import math
import datetime
import urllib.request
import shutil
import ssl
import json
import threading
import re
import tempfile
from flask import Flask, request, jsonify

app = Flask(__name__)

# =============================
# CONFIGURACIÓN Y CONSTANTES
# =============================

BASE_DIR = tempfile.gettempdir()
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'temp_rinex')
REPORT_FOLDER = os.path.join(BASE_DIR, 'informes')
STATE_FILE = os.path.join(UPLOAD_FOLDER, 'estado_proyecto.json')

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(REPORT_FOLDER, exist_ok=True)

STATE_LOCK = threading.Lock()
SP3_LOCK = threading.Lock()

C_LIGHT = 299792458.0
OMEGA_E = 7.2921151467e-5
MU_GPS = 3.986005e14
MU_GALILEO = 3.986004418e14
MU_BDS = 3.986004418e14

FREQ_L1 = 1575.42e6
FREQ_L5 = 1176.45e6
WAVE_L1 = C_LIGHT / FREQ_L1
WAVE_L5 = C_LIGHT / FREQ_L5

MAX_CACHE_SIZE = 2048
SP3_CACHE = {}
SP3_CACHE_KEYS = []


def gps_time_to_tow(year, month, day, hour, minute, second):
    sec_int = int(second)
    sec_frac = second - sec_int
    total = (datetime.datetime(year, month, day, hour, minute, sec_int) - datetime.datetime(1980, 1, 6)).total_seconds() + sec_frac
    return total - (int(total // 604800) * 604800)


# =============================
# UTILIDADES DE ESTADO
# =============================

def guardar_estado(clave, valor):
    with STATE_LOCK:
        estado = {}
        if os.path.exists(STATE_FILE):
            try:
                with open(STATE_FILE, 'r', encoding='utf-8') as f:
                    estado = json.load(f)
            except Exception:
                estado = {}
        estado[clave] = valor
        try:
            with open(STATE_FILE, 'w', encoding='utf-8') as f:
                json.dump(estado, f)
        except Exception:
            pass


def leer_estado(clave):
    with STATE_LOCK:
        if os.path.exists(STATE_FILE):
            try:
                with open(STATE_FILE, 'r', encoding='utf-8') as f:
                    return json.load(f).get(clave)
            except Exception:
                return None
    return None


# =============================
# DESCARGA DE ARCHIVOS
# =============================

def descargar_desde_gdrive(url, filepath):
    match = re.search(r'/file/d/([a-zA-Z0-9_-]+)', url) or re.search(r'id=([a-zA-Z0-9_-]+)', url)
    if not match:
        raise ValueError('URL de Google Drive no reconocida.')
    file_id = match.group(1)
    direct_url = f'https://drive.google.com/uc?export=download&id={file_id}'
    ctx = ssl.create_default_context()
    req = urllib.request.Request(direct_url, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req, context=ctx, timeout=120) as response, open(filepath, 'wb') as out_file:
        shutil.copyfileobj(response, out_file)
    return True


# =============================
# ÁLGEBRA LINEAL
# =============================

def transpose_matrix(M):
    if not M or not M[0]:
        return []
    return [list(row) for row in zip(*M)]


def matmul(A, B):
    if not A or not B or not A[0] or not B[0]:
        return []
    rows, cols, inner = len(A), len(B[0]), len(B)
    result = [[0.0 for _ in range(cols)] for _ in range(rows)]
    for i in range(rows):
        for j in range(cols):
            s = 0.0
            for k in range(inner):
                s += A[i][k] * B[k][j]
            result[i][j] = s
    return result


def matid(n):
    return [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]


def matadd(A, B):
    return [[A[i][j] + B[i][j] for j in range(len(A[0]))] for i in range(len(A))]


def matsub(A, B):
    return [[A[i][j] - B[i][j] for j in range(len(A[0]))] for i in range(len(A))]


def cholesky_decompose(A):
    n = len(A)
    L = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1):
            sum1 = sum(L[i][k] * L[j][k] for k in range(j))
            if i == j:
                val = A[i][i] - sum1
                if val <= 0:
                    raise ValueError('Matriz no definida positiva')
                L[i][j] = math.sqrt(val)
            else:
                L[i][j] = (A[i][j] - sum1) / L[j][j]
    return L


def invert_lower_triangular(L):
    n = len(L)
    inv = [[0.0] * n for _ in range(n)]
    for i in range(n):
        inv[i][i] = 1.0 / L[i][i]
        for j in range(i):
            sum1 = sum(L[i][k] * inv[k][j] for k in range(j, i))
            inv[i][j] = -sum1 / L[i][i]
    return inv


def gauss_jordan_inverse(M):
    n = len(M)
    A = [[float(M[i][j]) for j in range(n)] for i in range(n)]
    I = matid(n)
    for i in range(n):
        max_k = max(range(i, n), key=lambda k: abs(A[k][i]))
        if abs(A[max_k][i]) < 1e-15:
            return None
        if max_k != i:
            A[i], A[max_k] = A[max_k], A[i]
            I[i], I[max_k] = I[max_k], I[i]
        pivot = A[i][i]
        for j in range(n):
            A[i][j] /= pivot
            I[i][j] /= pivot
        for k in range(n):
            if k == i:
                continue
            factor = A[k][i]
            for j in range(n):
                A[k][j] -= factor * A[i][j]
                I[k][j] -= factor * I[i][j]
    return I


def invert_matrix_nxn(M):
    if not M or not M[0]:
        return None
    try:
        L = cholesky_decompose(M)
        L_inv = invert_lower_triangular(L)
        return matmul(transpose_matrix(L_inv), L_inv)
    except Exception:
        return gauss_jordan_inverse(M)


# =============================
# CAPA DE ENTRADA: PARSING
# =============================

def _parse_obs_tokens(line):
    return [x.strip() for x in line[6:60].split() if x.strip()]


def parse_rinex_obs_completo(path):
    obs = {}
    sys_idx = {}
    sys_tokens = {}
    last_sys_char = None
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        in_h = True
        tow = None
        for line in f:
            if in_h:
                if 'SYS / # / OBS TYPES' in line:
                    sys_char = line[0].strip()
                    if sys_char:
                        last_sys_char = sys_char
                    if last_sys_char:
                        sys_tokens.setdefault(last_sys_char, []).extend(_parse_obs_tokens(line))
                elif 'END OF HEADER' in line:
                    in_h = False
                    for sc, t in sys_tokens.items():
                        sys_idx[sc] = {
                            'C1': next((i for i, x in enumerate(t) if x.startswith('C1')), -1),
                            'L1': next((i for i, x in enumerate(t) if x.startswith('L1')), -1),
                            'C5': next((i for i, x in enumerate(t) if x.startswith('C5')), -1),
                            'L5': next((i for i, x in enumerate(t) if x.startswith('L5')), -1),
                            'S1': next((i for i, x in enumerate(t) if x.startswith('S1')), -1),
                            'S5': next((i for i, x in enumerate(t) if x.startswith('S5')), -1),
                        }
            elif line.startswith('>'):
                p = line[1:].split()
                if len(p) >= 6:
                    try:
                        y, m, d, h, mn, sec = int(p[0]), int(p[1]), int(p[2]), int(p[3]), int(p[4]), float(p[5])
                        tow = round(gps_time_to_tow(y, m, d, h, mn, sec), 6)
                        obs[tow] = {'_meta': (y, m, d, h, mn, sec)}
                    except Exception:
                        tow = None
            elif tow is not None and len(line) > 3 and line[0] in 'GRECSJ':
                sys_char = line[0]
                idxs = sys_idx.get(sys_char, {})
                data = {}
                for key in ('C1', 'C5', 'L1', 'L5', 'S1', 'S5'):
                    idx = idxs.get(key, -1)
                    if idx >= 0 and len(line) >= 17 + 16 * idx:
                        v = line[3 + 16 * idx:17 + 16 * idx].strip()
                        if v:
                            try:
                                data[key] = float(v.replace('D', 'E').replace('d', 'e'))
                            except Exception:
                                pass
                valid_p = ('C1' in data and data['C1'] > 15000000.0) or ('C5' in data and data['C5'] > 15000000.0)
                if valid_p:
                    obs.setdefault(tow, {})[line[0:3].strip()] = data
    return obs


def parse_rinex_nav_real(path):
    ephemeris = {'_iono': {'alpha': [0.0] * 4, 'beta': [0.0] * 4}}
    if not path or not os.path.exists(path):
        return ephemeris
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        in_h = True
        sat = None
        data = []
        for line in f:
            if in_h:
                if 'IONOSPHERIC CORR' in line:
                    sys_type = line[0:4].strip()
                    vals = []
                    for i in range(4):
                        try:
                            chunk = line[5 + i * 12:5 + (i + 1) * 12].strip().replace('D', 'E').replace('d', 'e')
                            vals.append(float(chunk) if chunk else 0.0)
                        except Exception:
                            vals.append(0.0)
                    if sys_type == 'GPSA':
                        ephemeris['_iono']['alpha'] = vals
                    elif sys_type == 'GPSB':
                        ephemeris['_iono']['beta'] = vals
                elif 'END OF HEADER' in line:
                    in_h = False
                continue
            if len(line) > 8 and line[0] in 'GECSJ' and line[1:3].isdigit():
                if sat and len(data) >= 20:
                    ephemeris.setdefault(sat, []).append({
                        'af0': data[0], 'af1': data[1], 'af2': data[2],
                        'Crs': data[4], 'Delta_n': data[5], 'M0': data[6],
                        'Cuc': data[7], 'e': data[8], 'Cus': data[9],
                        'sqrtA': data[10], 'Toe': data[11], 'Cic': data[12],
                        'OMEGA': data[13], 'Cis': data[14], 'i0': data[15],
                        'Crc': data[16], 'omega': data[17], 'OMEGA_DOT': data[18],
                        'IDOT': data[19]
                    })
                sat = line[0:3].strip()
                data = [
                    _safe_f(line[23:42]),
                    _safe_f(line[42:61]),
                    _safe_f(line[61:80])
                ]
            elif sat and line.startswith('    '):
                for i in range(4, 80, 19):
                    chunk = line[i:i+19].replace('D', 'E').replace('d', 'e').strip()
                    if chunk:
                        data.append(_safe_f(chunk))
        if sat and len(data) >= 20:
            ephemeris.setdefault(sat, []).append({
                'af0': data[0], 'af1': data[1], 'af2': data[2],
                'Crs': data[4], 'Delta_n': data[5], 'M0': data[6],
                'Cuc': data[7], 'e': data[8], 'Cus': data[9],
                'sqrtA': data[10], 'Toe': data[11], 'Cic': data[12],
                'OMEGA': data[13], 'Cis': data[14], 'i0': data[15],
                'Crc': data[16], 'omega': data[17], 'OMEGA_DOT': data[18],
                'IDOT': data[19]
            })
    return ephemeris


def _safe_f(s):
    try:
        return float(s) if s.strip() else 0.0
    except Exception:
        return 0.0


def parse_sp3_preciso(path):
    sp3_data = {}
    if not path or not os.path.exists(path):
        return sp3_data
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        current_time = None
        for line in f:
            if line.startswith('* '):
                p = line.split()
                if len(p) >= 7:
                    try:
                        y, m, d, h, mn, s = int(p[1]), int(p[2]), int(p[3]), int(p[4]), int(p[5]), float(p[6])
                        current_time = gps_time_to_tow(y, m, d, h, mn, s)
                    except Exception:
                        current_time = None
            elif line.startswith('P') and current_time is not None:
                sys_char = line[1]
                if sys_char in 'GECR':
                    sat_id = line[1:4].strip()
                    try:
                        x = float(line[4:18]) * 1000.0
                        y = float(line[18:32]) * 1000.0
                        z = float(line[32:46]) * 1000.0
                        clk = float(line[46:60]) / 1e6 if len(line) > 46 and line[46:60].strip() else 0.0
                        sp3_data.setdefault(sat_id, []).append((current_time, x, y, z, clk))
                    except Exception:
                        pass
    for sat in sp3_data:
        sp3_data[sat].sort(key=lambda item: item[0])
    return sp3_data


# =============================
# CAPA DE PREPROCESAMIENTO
# =============================

def interpolar_base_a_rover(obs_base, tr, max_gap=0.05):
    tiempos_base = sorted(obs_base.keys())
    if not tiempos_base:
        return None
    idx = min(range(len(tiempos_base)), key=lambda i: abs(tiempos_base[i] - tr))
    if abs(tiempos_base[idx] - tr) <= max_gap:
        return obs_base[tiempos_base[idx]].copy()
    return None


def _select_code_band(d_b, d_r):
    l5_ok = d_b.get('C5') and d_r.get('C5') and d_b.get('L5') and d_r.get('L5')
    l1_ok = d_b.get('C1') and d_r.get('C1') and d_b.get('L1') and d_r.get('L1')
    return 'L5' if l5_ok else ('L1' if l1_ok else None)


def aislar_diferencias_simples_ppk(obs_b, obs_r):
    sd_suavizada = {}
    for tow in sorted(obs_r.keys()):
        if tow not in obs_b:
            continue
        l1_count = 0
        l5_count = 0
        for s, d_r in obs_r[tow].items():
            if s == '_meta' or s not in obs_b[tow]:
                continue
            d_b = obs_b[tow][s]
            if d_b.get('C5') and d_r.get('C5') and d_b.get('L5') and d_r.get('L5'):
                l5_count += 1
            if d_b.get('C1') and d_r.get('C1') and d_b.get('L1') and d_r.get('L1'):
                l1_count += 1
        use_l5 = (l5_count >= 4) or (l5_count >= l1_count and l5_count >= 3)
        sd_epoca = {'_meta': obs_r[tow]['_meta']}
        for s, d_r in obs_r[tow].items():
            if s == '_meta' or s not in obs_b[tow]:
                continue
            d_b = obs_b[tow][s]
            band = _select_code_band(d_b, d_r)
            if use_l5 and band != 'L5':
                continue
            if not use_l5 and band != 'L1':
                continue
            if band == 'L5':
                pr_b, pr_r = d_b['C5'], d_r['C5']
                cp_b, cp_r = d_b['L5'], d_r['L5']
                wave_sys = WAVE_L5
                snr_b = d_b.get('S5', d_b.get('S1', 30.0))
                snr_r = d_r.get('S5', d_r.get('S1', 30.0))
            else:
                pr_b, pr_r = d_b['C1'], d_r['C1']
                cp_b, cp_r = d_b['L1'], d_r['L1']
                wave_sys = WAVE_L1
                snr_b = d_b.get('S1', d_b.get('S5', 30.0))
                snr_r = d_r.get('S1', d_r.get('S5', 30.0))
            sd_epoca[s] = {
                'sd_P': pr_r - pr_b,
                'pr_b': pr_b,
                'pr_r': pr_r,
                'cp_b': cp_b,
                'cp_r': cp_r,
                'wave': wave_sys,
                'snr': min(snr_b, snr_r),
                'sys': s[0]
            }
        if len(sd_epoca) > 1:
            sd_suavizada[tow] = sd_epoca
    return sd_suavizada


# =============================
# MODELO GNSS: ÓRBITA Y RELOJ
# =============================

def lagrange_interpolate(x, x_pts, y_pts):
    n = len(x_pts)
    val = 0.0
    for i in range(n):
        p = 1.0
        for j in range(n):
            if i != j:
                p *= (x - x_pts[j]) / (x_pts[i] - x_pts[j])
        val += y_pts[i] * p
    return val


def interpolate_sp3(sp3_data, sat, t_emision, degree=9):
    cache_key = f'{sat}_{round(t_emision, 3)}'
    with SP3_LOCK:
        if cache_key in SP3_CACHE:
            return SP3_CACHE[cache_key]
    if sat not in sp3_data:
        return None
    data = sp3_data[sat]
    if len(data) < degree + 1:
        return None
    idx = min(range(len(data)), key=lambda i: abs(data[i][0] - t_emision))
    half = degree // 2
    start = max(0, idx - half)
    end = min(len(data), start + degree + 1)
    if end - start < degree + 1:
        start = max(0, end - degree - 1)
    pts = data[start:end]
    t_pts = [p[0] for p in pts]
    x_pts = [p[1] for p in pts]
    y_pts = [p[2] for p in pts]
    z_pts = [p[3] for p in pts]
    start_clk = max(0, idx - 1)
    end_clk = min(len(data), start_clk + 2)
    if end_clk - start_clk < 2:
        start_clk = max(0, end_clk - 2)
    pts_clk = data[start_clk:end_clk]
    t_pts_clk = [p[0] for p in pts_clk]
    clk_pts = [p[4] for p in pts_clk]
    result = (
        lagrange_interpolate(t_emision, t_pts, x_pts),
        lagrange_interpolate(t_emision, t_pts, y_pts),
        lagrange_interpolate(t_emision, t_pts, z_pts),
        lagrange_interpolate(t_emision, t_pts_clk, clk_pts)
    )
    with SP3_LOCK:
        if len(SP3_CACHE) >= MAX_CACHE_SIZE:
            oldest_key = SP3_CACHE_KEYS.pop(0)
            SP3_CACHE.pop(oldest_key, None)
        SP3_CACHE[cache_key] = result
        SP3_CACHE_KEYS.append(cache_key)
    return result


def seleccionar_efemeride_optima(eph_list, t_target):
    if not eph_list:
        return None
    valid_ephs = []
    for eph in eph_list:
        dt = t_target - eph.get('Toe', 0.0)
        if dt > 302400:
            dt -= 604800
        elif dt < -302400:
            dt += 604800
        if abs(dt) <= 7200:
            valid_ephs.append((abs(dt), eph))
    if not valid_ephs:
        return None
    return min(valid_ephs, key=lambda x: x[0])[1]


def calcular_posicion_satelite_wgs84(eph, t_emision, tau_vuelo, sys_char='G'):
    if not eph or eph.get('sqrtA', 0.0) <= 0.0:
        return None
    mu_sys = MU_BDS if sys_char == 'C' else (MU_GALILEO if sys_char == 'E' else MU_GPS)
    omega_e_sys = OMEGA_E
    A = eph['sqrtA'] ** 2
    n0 = math.sqrt(mu_sys / (A ** 3))
    t_k = t_emision - eph['Toe']
    if sys_char == 'C':
        t_k -= 14.0
    if t_k > 302400:
        t_k -= 604800
    elif t_k < -302400:
        t_k += 604800
    M_k = eph['M0'] + (n0 + eph['Delta_n']) * t_k
    E_k = M_k
    for _ in range(8):
        E_k = M_k + eph['e'] * math.sin(E_k)
    dt_sat = eph['af0'] + eph['af1'] * t_k + eph['af2'] * (t_k ** 2)
    nu_k = math.atan2(math.sqrt(1 - eph['e']**2) * math.sin(E_k), math.cos(E_k) - eph['e'])
    phi_k = nu_k + eph['omega']
    u_k = phi_k + eph['Cus'] * math.sin(2 * phi_k) + eph['Cuc'] * math.cos(2 * phi_k)
    r_k = A * (1 - eph['e'] * math.cos(E_k)) + eph['Crs'] * math.sin(2 * phi_k) + eph['Crc'] * math.cos(2 * phi_k)
    i_k = eph['i0'] + eph['Cic'] * math.cos(2 * phi_k) + eph['Cis'] * math.sin(2 * phi_k) + eph['IDOT'] * t_k
    x_k, y_k = r_k * math.cos(u_k), r_k * math.sin(u_k)
    omega_k = eph['OMEGA'] + (eph['OMEGA_DOT'] - omega_e_sys) * t_k - omega_e_sys * eph['Toe']
    xs = x_k * math.cos(omega_k) - y_k * math.cos(i_k) * math.sin(omega_k)
    ys = x_k * math.sin(omega_k) + y_k * math.cos(i_k) * math.cos(omega_k)
    zs = y_k * math.sin(i_k)
    theta = omega_e_sys * tau_vuelo
    return (
        xs * math.cos(theta) + ys * math.sin(theta),
        -xs * math.sin(theta) + ys * math.cos(theta),
        zs,
        dt_sat
    )


# =============================
# MODELO GNSS: CORRECCIONES
# =============================

def correccion_mareas_solidas(X, Y, Z, tow, year, month, day):
    try:
        h2, l2 = 0.609, 0.085
        Re = 6378137.0
        GM_earth, GM_sun, GM_moon = 3.986004418e14, 1.327124e20, 4.902801e12
        jd = 367 * year - (7 * (year + (month + 9) // 12)) // 4 + (275 * month) // 9 + day + 1721013.5
        t_jc = (jd - 2451545.0 + (tow / 86400.0)) / 36525.0
        mean_long_sun = 280.460 + 36000.771 * t_jc
        mean_anom_sun = 357.528 + 35999.050 * t_jc
        ecl_lon_sun = mean_long_sun + 1.915 * math.sin(math.radians(mean_anom_sun)) + 0.020 * math.sin(math.radians(2 * mean_anom_sun))
        dist_sun = 1.495978707e11 * (1.00014 - 0.01671 * math.cos(math.radians(mean_anom_sun)) - 0.00014 * math.cos(math.radians(2 * mean_anom_sun)))
        obliquity = 23.439 - 0.013 * t_jc
        xs_sun = dist_sun * math.cos(math.radians(ecl_lon_sun))
        ys_sun = dist_sun * math.cos(math.radians(obliquity)) * math.sin(math.radians(ecl_lon_sun))
        zs_sun = dist_sun * math.sin(math.radians(obliquity)) * math.sin(math.radians(ecl_lon_sun))
        mean_long_moon = 218.316 + 481267.881 * t_jc
        mean_anom_moon = 134.963 + 477198.867 * t_jc
        mean_dist_moon = 93.272 + 483202.017 * t_jc
        ecl_lon_moon = mean_long_moon + 6.289 * math.sin(math.radians(mean_anom_moon))
        ecl_lat_moon = 5.128 * math.sin(math.radians(mean_dist_moon))
        dist_moon = 385000000.0 - 20905000.0 * math.cos(math.radians(mean_anom_moon))
        xs_moon = dist_moon * math.cos(math.radians(ecl_lon_moon)) * math.cos(math.radians(ecl_lat_moon))
        ys_moon = dist_moon * (math.cos(math.radians(obliquity)) * math.sin(math.radians(ecl_lon_moon)) * math.cos(math.radians(ecl_lat_moon)) - math.sin(math.radians(obliquity)) * math.sin(math.radians(ecl_lat_moon)))
        zs_moon = dist_moon * (math.sin(math.radians(obliquity)) * math.sin(math.radians(ecl_lon_moon)) * math.cos(math.radians(ecl_lat_moon)) + math.cos(math.radians(obliquity)) * math.sin(math.radians(ecl_lat_moon)))
        r_sta = math.sqrt(X**2 + Y**2 + Z**2)
        if r_sta == 0:
            return 0.0, 0.0, 0.0
        rx, ry, rz = X / r_sta, Y / r_sta, Z / r_sta

        def deformacion_cuerpo(mass_ratio, R_body, xs, ys, zs):
            dist_body = math.sqrt(xs**2 + ys**2 + zs**2)
            if dist_body == 0:
                return 0.0, 0.0, 0.0
            ux, uy, uz = xs / dist_body, ys / dist_body, zs / dist_body
            cos_theta = rx * ux + ry * uy + rz * uz
            p2 = 1.5 * cos_theta**2 - 0.5
            p2_prime = 3.0 * cos_theta
            coef = (GM_earth / Re**2) * mass_ratio * (Re / dist_body)**3 * Re
            dr_radial = h2 * coef * p2
            dr_tangent = l2 * coef * p2_prime
            dx = dr_radial * rx + dr_tangent * (ux - cos_theta * rx)
            dy = dr_radial * ry + dr_tangent * (uy - cos_theta * ry)
            dz = dr_radial * rz + dr_tangent * (uz - cos_theta * rz)
            return dx, dy, dz

        dx_sun, dy_sun, dz_sun = deformacion_cuerpo(GM_sun / GM_earth, dist_sun, xs_sun, ys_sun, zs_sun)
        dx_moon, dy_moon, dz_moon = deformacion_cuerpo(GM_moon / GM_earth, dist_moon, xs_moon, ys_moon, zs_moon)
        return dx_sun + dx_moon, dy_sun + dy_moon, dz_sun + dz_moon
    except Exception:
        return 0.0, 0.0, 0.0


def calcular_saastamoinen(lat_deg, alt, elev_deg):
    if elev_deg < 5.0:
        elev_deg = 5.0
    lat_rad = max(math.radians(lat_deg), -math.pi / 2)
    elev_rad = math.radians(elev_deg)
    H = max(0.0, min(alt, 40000.0))
    P = 1013.25 * ((1.0 - 2.2557e-5 * H) ** 5.2568)
    T = 288.15 - 0.0065 * H
    e = 6.11 * 0.5 * (10.0 ** (7.5 * (T - 273.15) / (T - 273.15 + 237.3))) * ((1.0 - 2.2557e-5 * H) ** 5.2568)
    zhd = (0.0022768 * P) / (1.0 - 0.00266 * math.cos(2.0 * lat_rad) - 0.00028 * (H / 1000.0))
    zwd = 0.0022768 * ((1255.0 / T) + 0.05) * e
    return (zhd + zwd) * (1.0 / math.sin(elev_rad))


def calcular_klobuchar(lat_deg, lon_deg, el_deg, az_deg, tow, alpha, beta):
    if not any(alpha) and not any(beta):
        return 0.0
    phi_u, lam_u = lat_deg / 180.0, lon_deg / 180.0
    E, A = el_deg / 180.0, az_deg / 180.0
    psi = 0.0137 / (E + 0.11) - 0.022
    phi_i = phi_u + psi * math.cos(A * math.pi)
    phi_i = max(-0.416, min(0.416, phi_i))
    lam_i = lam_u + (psi * math.sin(A * math.pi)) / max(math.cos(phi_i * math.pi), 1e-12)
    phi_m = phi_i + 0.064 * math.cos((lam_i - 1.617) * math.pi)
    t = (43200.0 * lam_i + tow) % 86400.0
    F = 1.0 + 16.0 * (0.53 - E) ** 3
    PER = max(72000.0, beta[0] + beta[1]*phi_m + beta[2]*(phi_m**2) + beta[3]*(phi_m**3))
    AMP = max(0.0, alpha[0] + alpha[1]*phi_m + alpha[2]*(phi_m**2) + alpha[3]*(phi_m**3))
    x = (2.0 * math.pi * (t - 50400.0)) / PER
    if abs(x) < 1.5707963267948966:
        return F * (5e-9 + AMP * (1.0 - (x**2)/2.0 + (x**4)/24.0)) * C_LIGHT
    return F * 5e-9 * C_LIGHT


# =============================
# GEODESIA: ECEF ↔ LLH ↔ UTM
# =============================

def geodesicas_a_ecef(lat_deg, lon_deg, alt):
    a, e2 = 6378137.0, 0.0066943799901413155
    lat, lon = math.radians(lat_deg), math.radians(lon_deg)
    N = a / math.sqrt(1 - e2 * (math.sin(lat) ** 2))
    return (
        (N + alt) * math.cos(lat) * math.cos(lon),
        (N + alt) * math.cos(lat) * math.sin(lon),
        (N * (1 - e2) + alt) * math.sin(lat)
    )


def ecef_a_geodesicas(x, y, z):
    a, e2 = 6378137.0, 0.0066943799901413155
    b = math.sqrt(a**2 * (1 - e2))
    ep2 = (a**2 - b**2) / b**2
    p = math.sqrt(x**2 + y**2)
    th = math.atan2(a * z, b * p)
    lat = math.atan2((z + ep2 * b * (math.sin(th) ** 3)), (p - e2 * a * (math.cos(th) ** 3)))
    N = a / math.sqrt(1 - e2 * (math.sin(lat) ** 2))
    return math.degrees(lat), math.degrees(math.atan2(y, x)), p / math.cos(lat) - N


def geodesicas_a_utm(lat, lon, force_zone=None):
    a, e2 = 6378137.0, 0.0066943799901413155
    zone = force_zone if force_zone is not None else int((lon + 180) / 6) + 1
    lat_r, lon_r = math.radians(lat), math.radians(lon)
    lon0 = math.radians((zone - 1) * 6 - 180 + 3)
    ep2 = e2 / (1 - e2)
    N = a / math.sqrt(1 - e2 * math.sin(lat_r)**2)
    T = math.tan(lat_r)**2
    C = ep2 * math.cos(lat_r)**2
    A = math.cos(lat_r) * (lon_r - lon0)
    M = a * (
        (1 - e2/4 - 3*e2**2/64 - 5*e2**3/256) * lat_r
        - (3*e2/8 + 3*e2**2/32 + 45*e2**3/1024) * math.sin(2*lat_r)
        + (15*e2**2/256 + 45*e2**3/1024) * math.sin(4*lat_r)
        - (35*e2**3/3072) * math.sin(6*lat_r)
    )
    Easting = 0.9996 * N * (
        A
        + (1 - T + C) * A**3 / 6
        + (5 - 18*T + T**2 + 72*C - 58*ep2) * A**5 / 120
    ) + 500000.0
    Northing = 0.9996 * (
        M
        + N * math.tan(lat_r) * (
            A**2 / 2
            + (5 - T + 9*C + 4*C**2) * A**4 / 24
            + (61 - 58*T + T**2 + 600*C - 330*ep2) * A**6 / 720
        )
    )
    return (Northing + 10000000.0 if lat < 0 else Northing), Easting, zone


def utm_a_geodesicas(easting, northing, zone=19, hemisferio='N'):
    a, e2 = 6378137.0, 0.0066943799901413155
    e1 = (1 - math.sqrt(1 - e2)) / (1 + math.sqrt(1 - e2))
    x, y = easting - 500000.0, northing if hemisferio.upper() == 'N' else northing - 10000000.0
    m = y / 0.9996
    mu = m / (a * (1 - e2/4 - 3*e2**2/64 - 5*e2**3/256))
    phi1_rad = mu + (3*e1/2 - 27*e1**3/32) * math.sin(2*mu) + (21*e1**2/16 - 55*e1**4/32) * math.sin(4*mu)
    n1 = a / math.sqrt(1 - e2 * math.sin(phi1_rad)**2)
    t1 = math.tan(phi1_rad)**2
    c1 = e2 / (1 - e2) * math.cos(phi1_rad)**2
    r1 = a * (1 - e2) / ((1 - e2 * math.sin(phi1_rad)**2) ** 1.5)
    d = x / (n1 * 0.9996)
    lat_rad = phi1_rad - (n1 * math.tan(phi1_rad) / r1) * (d**2 / 2 - (5 + 3*t1 + 10*c1) * d**4 / 24)
    lon_rad = (d - (1 + 2*t1 + c1) * d**3 / 6) / math.cos(phi1_rad)
    lon0 = math.radians((zone - 1) * 6 - 180 + 3)
    return math.degrees(lat_rad), math.degrees(lon_rad + lon0), 0.0


def calcular_topocentricas(xs, ys, zs, X_usr, Y_usr, Z_usr):
    lat_val, lon_val, _ = ecef_a_geodesicas(X_usr, Y_usr, Z_usr)
    lat_r = math.radians(lat_val)
    lon_r = math.radians(lon_val)
    dx, dy, dz = xs - X_usr, ys - Y_usr, zs - Z_usr
    sin_lat, cos_lat = math.sin(lat_r), math.cos(lat_r)
    sin_lon, cos_lon = math.sin(lon_r), math.cos(lon_r)
    e = -sin_lon * dx + cos_lon * dy
    n = -sin_lat * cos_lon * dx - sin_lat * sin_lon * dy + cos_lat * dz
    u = cos_lat * cos_lon * dx + cos_lat * sin_lon * dy + sin_lat * dz
    dist = math.sqrt(dx**2 + dy**2 + dz**2)
    if dist < 1e-6:
        return 0.0, 0.0
    el = math.degrees(math.asin(max(-1.0, min(1.0, u / dist))))
    az = math.degrees(math.atan2(e, n))
    if az < 0:
        az += 360.0
    return el, az


# =============================
# CAPA DE SOLUCIÓN: EKF PPK
# =============================

def _build_design_row(sp_r, sp_b, X_apc, Y_apc, Z_apc):
    dist_r = math.sqrt((sp_r[0] - X_apc)**2 + (sp_r[1] - Y_apc)**2 + (sp_r[2] - Z_apc)**2)
    dist_b = math.sqrt((sp_b[0] - X_apc)**2 + (sp_b[1] - Y_apc)**2 + (sp_b[2] - Z_apc)**2)
    if dist_r < 1e-6 or dist_b < 1e-6:
        return None
    row = [
        -(sp_r[0] - X_apc) / dist_r + (sp_b[0] - X_apc) / dist_b,
        -(sp_r[1] - Y_apc) / dist_r + (sp_b[1] - Y_apc) / dist_b,
        -(sp_r[2] - Z_apc) / dist_r + (sp_b[2] - Z_apc) / dist_b
    ]
    return row, dist_r, dist_b


def procesar_ekf_lambda(sd_epoca, nav, sp3, kf_estado, tr, mask_angle, snr_mask):
    try:
        X_pri = [[kf_estado['X'][0][0]], [kf_estado['X'][1][0]], [kf_estado['X'][2][0]]]
        P_pri = [row[:] for row in kf_estado['P']]
        h_r = kf_estado.get('h_r', 0.0)
        if 'prev_cp' not in kf_estado:
            kf_estado['prev_cp'] = {}

        X_iter, Y_iter, Z_iter = X_pri[0][0], X_pri[1][0], X_pri[2][0]
        lat_r, lon_r, alt_r = ecef_a_geodesicas(X_iter, Y_iter, Z_iter)
        lat_rad, lon_rad = math.radians(lat_r), math.radians(lon_r)
        X_apc = X_iter + h_r * math.cos(lat_rad) * math.cos(lon_rad)
        Y_apc = Y_iter + h_r * math.cos(lat_rad) * math.sin(lon_rad)
        Z_apc = Z_iter + h_r * math.sin(lat_rad)

        alpha = nav.get('_iono', {}).get('alpha', [0.0] * 4)
        beta = nav.get('_iono', {}).get('beta', [0.0] * 4)

        y_m, m_m, d_m, _, _, _ = sd_epoca['_meta']
        dx_tide, dy_tide, dz_tide = correccion_mareas_solidas(
            kf_estado['X_base'][0], kf_estado['X_base'][1], kf_estado['X_base'][2],
            tr, y_m, m_m, d_m
        )
        X_base_corr = kf_estado['X_base'][0] + dx_tide
        Y_base_corr = kf_estado['X_base'][1] + dy_tide
        Z_base_corr = kf_estado['X_base'][2] + dz_tide
        lat_base, lon_base, alt_base = ecef_a_geodesicas(X_base_corr, Y_base_corr, Z_base_corr)

        sat_positions = {}
        for s, d in sd_epoca.items():
            if s == '_meta' or d['sd_P'] is None:
                continue
            tau_r = d['pr_r'] / C_LIGHT
            tau_b = d['pr_b'] / C_LIGHT
            t_emision_r = tr - tau_r
            t_emision_b = tr - tau_b
            sp_r = sp_b = None
            if sp3 and s in sp3:
                sp3_res_r = interpolate_sp3(sp3, s, t_emision_r)
                sp3_res_b = interpolate_sp3(sp3, s, t_emision_b)
                if sp3_res_r and sp3_res_b:
                    theta_r = OMEGA_E * tau_r
                    xs_r = sp3_res_r[0] * math.cos(theta_r) + sp3_res_r[1] * math.sin(theta_r)
                    ys_r = -sp3_res_r[0] * math.sin(theta_r) + sp3_res_r[1] * math.cos(theta_r)
                    sp_r = (xs_r, ys_r, sp3_res_r[2], sp3_res_r[3])
                    theta_b = OMEGA_E * tau_b
                    xs_b = sp3_res_b[0] * math.cos(theta_b) + sp3_res_b[1] * math.sin(theta_b)
                    ys_b = -sp3_res_b[0] * math.sin(theta_b) + sp3_res_b[1] * math.cos(theta_b)
                    sp_b = (xs_b, ys_b, sp3_res_b[2], sp3_res_b[3])
            if not sp_r or not sp_b:
                eph_r = seleccionar_efemeride_optima(nav.get(s), t_emision_r)
                eph_b = seleccionar_efemeride_optima(nav.get(s), t_emision_b)
                sp_r = calcular_posicion_satelite_wgs84(eph_r, t_emision_r, tau_r, s[0]) if eph_r else None
                sp_b = calcular_posicion_satelite_wgs84(eph_b, t_emision_b, tau_b, s[0]) if eph_b else None
            if sp_r and sp_b:
                el_r, _ = calcular_topocentricas(sp_r[0], sp_r[1], sp_r[2], X_apc, Y_apc, Z_apc)
                if el_r >= mask_angle and d.get('snr', 30.0) >= snr_mask:
                    sat_positions[s] = {
                        'sp_r': sp_r, 'sp_b': sp_b,
                        'sd_P': d['sd_P'], 'cp_r': d['cp_r'], 'cp_b': d['cp_b'],
                        'wave': d['wave'], 'snr': d['snr'], 'sys': d['sys']
                    }

        if len(sat_positions) < 4:
            return None, 'FAILED', kf_estado, None

        sat_list_full = list(sat_positions.keys())
        constellations = set(s[0] for s in sat_list_full)
        ref_sats = {}
        sat_list = []
        for c in constellations:
            c_sats = [s for s in sat_list_full if s[0] == c]
            if len(c_sats) >= 2:
                r_candidate = max(
                    c_sats,
                    key=lambda k: calcular_topocentricas(
                        sat_positions[k]['sp_r'][0],
                        sat_positions[k]['sp_r'][1],
                        sat_positions[k]['sp_r'][2],
                        X_apc, Y_apc, Z_apc
                    )[0]
                )
                ref_sats[c] = r_candidate
                c_sats.remove(r_candidate)
                sat_list.extend(c_sats)

        if len(sat_list) < 3:
            return None, 'FAILED', kf_estado, None

        def calc_rho(sp, X, Y, Z, lat, lon, alt, el, az, wave):
            dist = math.sqrt((sp[0]-X)**2 + (sp[1]-Y)**2 + (sp[2]-Z)**2)
            tropo = calcular_saastamoinen(lat, alt, el)
            iono_m = calcular_klobuchar(lat, lon, el, az, tr, alpha, beta)
            if wave == WAVE_L5:
                iono_m *= 1.79327
            return dist + tropo, iono_m, dist

        base_calcs = {}
        for s, data in sat_positions.items():
            el_b, az_b = calcular_topocentricas(data['sp_b'][0], data['sp_b'][1], data['sp_b'][2], X_base_corr, Y_base_corr, Z_base_corr)
            rho_b, iono_b, _ = calc_rho(data['sp_b'], X_base_corr, Y_base_corr, Z_base_corr, lat_base, lon_base, alt_base, el_b, az_b, data['wave'])
            base_calcs[s] = {'P': rho_b + iono_b, 'CP': rho_b - iono_b}

        H = []
        L = []
        R_diag = []
        c_ref = {}
        for c, r_sat in ref_sats.items():
            r_data = sat_positions[r_sat]
            el_r, az_r = calcular_topocentricas(r_data['sp_r'][0], r_data['sp_r'][1], r_data['sp_r'][2], X_apc, Y_apc, Z_apc)
            rho_r, iono_r, dist_r = calc_rho(r_data['sp_r'], X_apc, Y_apc, Z_apc, lat_r, lon_r, alt_r + h_r, el_r, az_r, r_data['wave'])
            SD_P_calc_ref = (rho_r + iono_r) - base_calcs[r_sat]['P']
            c_ref[c] = {
                'dist_r': dist_r,
                'SD_P_calc_ref': SD_P_calc_ref,
                'sp_r': r_data['sp_r'],
                'el_r': el_r,
                'snr': r_data['snr'],
                'sd_P': r_data['sd_P'],
                'cp_r': r_data['cp_r'],
                'cp_b': r_data['cp_b']
            }

        for s in sat_list:
            c = s[0]
            data = sat_positions[s]
            rc = c_ref[c]
            el_i_r, az_i_r = calcular_topocentricas(data['sp_r'][0], data['sp_r'][1], data['sp_r'][2], X_apc, Y_apc, Z_apc)
            rho_i_r, iono_i_r, dist_i_r = calc_rho(data['sp_r'], X_apc, Y_apc, Z_apc, lat_r, lon_r, alt_r + h_r, el_i_r, az_i_r, data['wave'])
            SD_P_calc_i = (rho_i_r + iono_i_r) - base_calcs[s]['P']
            DD_P_calc = SD_P_calc_i - rc['SD_P_calc_ref']
            row = _build_design_row(data['sp_r'], rc['sp_r'], X_apc, Y_apc, Z_apc)
            if not row:
                continue
            dx_geom, dist_i_r, _ = row
            var_base = (10.0 ** (-data['snr'] / 10.0)) * 100.0
            DD_P_obs = data['sd_P'] - rc['sd_P']
            L.append([DD_P_obs - DD_P_calc])
            H.append(dx_geom)
            R_diag.append(var_base * 9.0)

        if len(H) < 3:
            return None, 'FAILED', kf_estado, None

        HT = transpose_matrix(H)
        R = [[R_diag[i] if i == j else 0.0 for j in range(len(R_diag))] for i in range(len(R_diag))]
        P_inv = invert_matrix_nxn(P_pri)
        if not P_inv:
            return None, 'FAILED', kf_estado, None

        HtR = matmul(HT, invert_matrix_nxn(R) or R)
        S = matadd(matmul(HtR, H), P_inv)
        S_inv = invert_matrix_nxn(S)
        if not S_inv:
            return None, 'FAILED', kf_estado, None

        K = matmul(matmul(P_pri, HT), invert_matrix_nxn(R) or R)
        K = matmul(K, invert_matrix_nxn(matadd(matmul(H, matmul(P_pri, HT)), R)) or matid(len(R_diag)))
        X_post = matadd(X_pri, matmul(K, L))
        I = matid(3)
        P_post = matmul(matsub(I, matmul(K, H)), P_pri)

        kf_estado['X'] = X_post
        kf_estado['P'] = P_post
        kf_estado['last_tow'] = tr

        return {
            'X_post': X_post,
            'P_post': P_post,
            'X_pri': X_pri,
            'P_pri': P_pri
        }, 'OK', kf_estado, {'satellites': len(sat_positions), 'used': len(H)}

    except Exception:
        return None, 'FAILED', kf_estado, None


def suavizador_rts_backward(forward_states):
    n = len(forward_states)
    if n == 0:
        return []
    smoothed_states = [None] * n
    smoothed_states[-1] = forward_states[-1]['X_post']
    for k in range(n - 2, -1, -1):
        P_post_k = forward_states[k]['P_post']
        P_pri_k1 = forward_states[k + 1]['P_pri']
        P_pri_inv = invert_matrix_nxn(P_pri_k1)
        if not P_pri_inv:
            smoothed_states[k] = forward_states[k]['X_post']
            continue
        C_k = matmul(P_post_k, P_pri_inv)
        X_smooth_k1 = smoothed_states[k + 1]
        X_pri_k1 = forward_states[k + 1]['X_pri']
        dx = [[X_smooth_k1[i][0] - X_pri_k1[i][0]] for i in range(3)]
        correction = matmul(C_k, dx)
        X_post_k = forward_states[k]['X_post']
        smoothed_states[k] = [[X_post_k[i][0] + correction[i][0]] for i in range(3)]
    return smoothed_states


# =============================
# CAPA DE SALIDA: API
# =============================

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok'})


@app.route('/convert', methods=['POST'])
def convert():
    payload = request.get_json(force=True, silent=True) or {}
    lat = float(payload.get('lat', 0.0))
    lon = float(payload.get('lon', 0.0))
    alt = float(payload.get('alt', 0.0))
    zone = payload.get('zone')
    zone = int(zone) if str(zone).strip() != '' else None
    northing, easting, used_zone = geodesicas_a_utm(lat, lon, zone)
    return jsonify({
        'lat': lat,
        'lon': lon,
        'alt': alt,
        'zone': used_zone,
        'northing': northing,
        'easting': easting
    })


@app.route('/parse_obs', methods=['POST'])
def parse_obs():
    data = request.get_json(force=True, silent=True) or {}
    path = data.get('path')
    if not path or not os.path.exists(path):
        return jsonify({'error': 'path inválido'}), 400
    obs = parse_rinex_obs_completo(path)
    return jsonify({'epochs': len(obs)})


@app.route('/download_gdrive', methods=['POST'])
def download_gdrive():
    data = request.get_json(force=True, silent=True) or {}
    url = data.get('url')
    name = data.get('name', 'input.dat')
    if not url:
        return jsonify({'error': 'url requerida'}), 400
    filepath = os.path.join(UPLOAD_FOLDER, name)
    descargar_desde_gdrive(url, filepath)
    return jsonify({'saved': filepath})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)), debug=False)
