import pygame
import math
from config import WIDTH, HEIGHT, BLACK, WHITE

def create_track():
    track = pygame.Surface((WIDTH, HEIGHT))
    track.fill(BLACK)
    
    # Batas area lintasan
    left, top = 160, 120
    right, bottom = 640, 480
    radius = 55
    
    points = []
    
    # 1. SISI ATAS (Jalur Lurus & Titik Start)
    points.append((left + radius, top))
    points.append((right - radius, top))
    
    # Lengkungan Sudut Kanan Atas (270° ke 360°)
    for deg in range(270, 361, 5):
        rad = math.radians(deg)
        points.append((int(right - radius + math.cos(rad) * radius), int(top + radius + math.sin(rad) * radius)))
        
    # 2. SISI KANAN (Pola Zigzag Tajam)
    zigzag_start_y = top + radius + 40
    zigzag_end_y = bottom - radius - 40
    zigzag_steps = 1
    step_y = (zigzag_end_y - zigzag_start_y) / zigzag_steps
    
    for i in range(zigzag_steps + 1):
        curr_y = zigzag_start_y + (i * step_y)
        # Bergantian masuk ke dalam (-30) dan keluar (+30)
        offset_x = 30 if i % 2 == 1 else -30
        points.append((int(right + offset_x), int(curr_y)))
        
    # Lengkungan Sudut Kanan Bawah (0° ke 90°)
    for deg in range(0, 90, 4):
        rad = math.radians(deg)
        points.append((int(right - radius + math.cos(rad) * radius), int(bottom - radius + math.sin(rad) * radius)))
        
    # 3. SISI BAWAH (Pola Osilasi / Gelombang Sinus Tipis)
    osc_start_x = right - radius - 10
    osc_end_x = left + radius + 10
    osc_steps = 45
    
    for i in range(osc_steps + 1):
        t = i / osc_steps
        curr_x = osc_start_x - t * (osc_start_x - osc_end_x)
        # 3 gelombang penuh dengan tinggi gelombang tipis (14 piksel)
        wave_y = math.sin(t * math.pi * 6) * 10
        points.append((int(curr_x), int(bottom + wave_y)))
        
    # Lengkungan Sudut Kiri Bawah (90° ke 180°)
    for deg in range(90, 180, 5):
        rad = math.radians(deg)
        points.append((int(left + radius + math.cos(rad) * radius), int(bottom - radius + math.sin(rad) * radius)))
        
    # 4. SISI KIRI (Jalur Lurus Kembali ke Atas)
    points.append((left, bottom - radius))
    points.append((left, top + radius))
    
    # Lengkungan Sudut Kiri Atas (180° ke 270°)
    for deg in range(180, 271, 5):
        rad = math.radians(deg)
        points.append((int(left + radius + math.cos(rad) * radius), int(top + radius + math.sin(rad) * radius)))
        
    line_width = 12
    pygame.draw.lines(track, WHITE, True, points, line_width)
    
    # Haluskan setiap sambungan titik agar sudut zigzag tidak terputus
    for p in points:
        pygame.draw.circle(track, WHITE, p, line_width // 2)
        
    return track, points[0]
