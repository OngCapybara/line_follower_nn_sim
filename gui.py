import pygame
from config import TRACK_WIDTH, PANEL_WIDTH, HEIGHT, WHITE, GREEN, RED, BLUE, YELLOW, GRAY, PANEL_BG, CARD_BG

class DashboardGUI:
    def __init__(self):
        self.panel_x = TRACK_WIDTH
        self.font_title = pygame.font.SysFont("Arial", 18, bold=True)
        self.font_reg = pygame.font.SysFont("Arial", 14)
        self.font_small = pygame.font.SysFont("Arial", 12)
        
        btn_x = self.panel_x + 20
        btn_w = PANEL_WIDTH - 40
        self.buttons = {
            "mode": pygame.Rect(btn_x, 310, btn_w, 34),
            "speed": pygame.Rect(btn_x, 354, btn_w, 34),
            "save": pygame.Rect(btn_x, 398, (btn_w // 2) - 5, 34),
            "reset": pygame.Rect(btn_x + (btn_w // 2) + 5, 398, (btn_w // 2) - 5, 34),
        }

    def handle_click(self, pos):
        for action, rect in self.buttons.items():
            if rect.collidepoint(pos):
                return action
        return None

    def draw(self, screen, robot, generation, alive_count, total_pop, solo_mode, sim_speed, fitness_history):
        pygame.draw.rect(screen, PANEL_BG, (self.panel_x, 0, PANEL_WIDTH, HEIGHT))
        pygame.draw.line(screen, GRAY, (self.panel_x, 0), (self.panel_x, HEIGHT), 2)

        x = self.panel_x + 20
        w = PANEL_WIDTH - 40

        # 1. Judul & Status
        title = self.font_title.render("AI TELEMETRY & CONTROL", True, WHITE)
        screen.blit(title, (x, 15))

        pygame.draw.rect(screen, CARD_BG, (x, 45, w, 75), border_radius=8)
        mode_label = "Mode: SOLO (1 Robot)" if solo_mode else f"Mode: LATIHAN ({alive_count}/{total_pop})"
        screen.blit(self.font_reg.render(mode_label, True, GREEN if solo_mode else BLUE), (x + 12, 53))
        screen.blit(self.font_reg.render(f"Generasi: {generation}  |  Speed: {sim_speed}x", True, WHITE), (x + 12, 75))
        screen.blit(self.font_reg.render(f"Skor Fitness: {int(robot.fitness)}", True, YELLOW), (x + 12, 97))

        # 2. Speedometer
        pygame.draw.rect(screen, CARD_BG, (x, 130, w, 80), border_radius=8)
        screen.blit(self.font_small.render("KECEPATAN & PEDAL GAS (OUTPUT 2)", True, GRAY), (x + 12, 138))
        
        speed_txt = self.font_reg.render(f"{robot.current_speed:.2f} px/frame  (Gas: {int(robot.throttle * 100)}%)", True, WHITE)
        screen.blit(speed_txt, (x + 12, 158))
        
        bar_bg = pygame.Rect(x + 12, 184, w - 24, 14)
        pygame.draw.rect(screen, (25, 25, 30), bar_bg, border_radius=4)
        fill_w = int((w - 24) * max(0.0, min(1.0, robot.throttle)))
        pygame.draw.rect(screen, GREEN if robot.throttle > 0.4 else YELLOW, (x + 12, 184, fill_w, 14), border_radius=4)

        # 3. Kemudi
        pygame.draw.rect(screen, CARD_BG, (x, 220, w, 78), border_radius=8)
        screen.blit(self.font_small.render("ARAH KEMUDI / SETIR (OUTPUT 1)", True, GRAY), (x + 12, 228))
        
        steer_norm = robot.current_turn / 0.32
        arah = "LURUS" if abs(steer_norm) < 0.15 else ("BELOK KANAN >>" if steer_norm > 0 else "<< BELOK KIRI")
        screen.blit(self.font_reg.render(f"{arah} ({steer_norm:+.2f})", True, WHITE), (x + 12, 248))

        steer_bg = pygame.Rect(x + 12, 272, w - 24, 14)
        pygame.draw.rect(screen, (25, 25, 30), steer_bg, border_radius=4)
        center_x = x + 12 + (w - 24) // 2
        pygame.draw.line(screen, WHITE, (center_x, 270), (center_x, 288), 2)
        
        bar_len = int(((w - 24) // 2) * min(1.0, abs(steer_norm)))
        if steer_norm >= 0:
            pygame.draw.rect(screen, BLUE, (center_x, 272, bar_len, 14))
        else:
            pygame.draw.rect(screen, RED, (center_x - bar_len, 272, bar_len, 14))

        # 4. Tombol
        self._draw_btn(screen, self.buttons["mode"], "Ganti Mode: SOLO / LATIHAN [T]", BLUE)
        self._draw_btn(screen, self.buttons["speed"], f"Kecepatan Simulasi: {sim_speed}x [SPASI]", (90, 95, 110))
        self._draw_btn(screen, self.buttons["save"], "Simpan [S]", (40, 150, 80))
        self._draw_btn(screen, self.buttons["reset"], "Reset [R]", (170, 50, 50))

        # 5. Grafik
        graph_rect = pygame.Rect(x, 445, w, 135)
        pygame.draw.rect(screen, CARD_BG, graph_rect, border_radius=8)
        screen.blit(self.font_small.render("GRAFIK PERKEMBANGAN SKOR TIAP GENERASI", True, GRAY), (x + 12, 453))

        if len(fitness_history) > 1:
            max_fit = max(max(fitness_history), 1)
            pts = []
            recent = fitness_history[-30:]
            for i, val in enumerate(recent):
                px = x + 15 + int((i / (len(recent) - 1)) * (w - 30))
                py = 565 - int((val / max_fit) * 85)
                pts.append((px, py))
            pygame.draw.lines(screen, GREEN, False, pts, 2)
            for p in pts:
                pygame.draw.circle(screen, YELLOW, p, 3)
        else:
            info = self.font_small.render("Grafik muncul saat Mode Latihan berjalan...", True, GRAY)
            screen.blit(info, (x + 15, 510))

    def _draw_btn(self, screen, rect, text, color):
        pygame.draw.rect(screen, color, rect, border_radius=6)
        txt_surf = self.font_small.render(text, True, WHITE)
        txt_rect = txt_surf.get_rect(center=rect.center)
        screen.blit(txt_surf, txt_rect)
