import pygame
import random
from config import WIDTH, HEIGHT, FPS, POPULATION_SIZE, SENSOR_COUNT
from track import create_track
from robot import Robot
from neural_network import NeuralNetwork
from visualizer import draw_neural_network
from gui import DashboardGUI

def spawn_robots(start_pos, saved_brain, solo_mode):
    # Pastikan otak tersimpan cocok dengan jumlah sensor saat ini (11 sensor)
    valid_brain = saved_brain if (saved_brain and saved_brain.input_size == SENSOR_COUNT) else None

    if solo_mode:
        brain = valid_brain.copy() if valid_brain else None
        return [Robot(start_pos[0], start_pos[1], 0, brain)]
    else:
        if valid_brain:
            robots = [Robot(start_pos[0], start_pos[1], 0, valid_brain.copy())]
            # Sisipkan 4 robot acak baru untuk eksplorasi
            for _ in range(4):
                robots.append(Robot(start_pos[0], start_pos[1], 0))
            while len(robots) < POPULATION_SIZE:
                mutated = valid_brain.copy()
                mutated.mutate(rate=0.25, scale=0.5)
                robots.append(Robot(start_pos[0], start_pos[1], 0, mutated))
            return robots
        else:
            return [Robot(start_pos[0], start_pos[1], 0) for _ in range(POPULATION_SIZE)]

def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("AI Lab: Line Follower Neural Network (11 Sensors)")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("Arial", 16)

    track_surface, start_pos = create_track()
    gui = DashboardGUI()
    
    saved_brain, generation = NeuralNetwork.load("best_brain.npz")
    if saved_brain and saved_brain.input_size != SENSOR_COUNT:
        print("Struktur sensor berubah ke 11, memulai ulang dari Generasi 1...")
        saved_brain = None
        generation = 1

    all_time_best_fitness = 0
    fitness_history = []
    
    # Mulai di mode latihan agar robot 11-sensor langsung belajar
    solo_mode = False
    sim_speeds = [1, 2, 5]
    speed_idx = 0
    
    robots = spawn_robots(start_pos, saved_brain, solo_mode)
    best_robot = robots[0]

    running = True
    while running:
        action = None
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                action = gui.handle_click(event.pos)
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_t:
                    action = "mode"
                elif event.key == pygame.K_SPACE:
                    action = "speed"
                elif event.key == pygame.K_s:
                    action = "save"
                elif event.key == pygame.K_r:
                    action = "reset"

        if action == "mode":
            solo_mode = not solo_mode
            robots = spawn_robots(start_pos, saved_brain, solo_mode)
        elif action == "speed":
            speed_idx = (speed_idx + 1) % len(sim_speeds)
        elif action == "save":
            best_robot.brain.save("best_brain.npz", generation)
            saved_brain = best_robot.brain.copy()
        elif action == "reset":
            generation = 1
            all_time_best_fitness = 0
            fitness_history = []
            saved_brain = None
            robots = spawn_robots(start_pos, saved_brain, solo_mode)

        alive_count = 0
        for _ in range(sim_speeds[speed_idx]):
            alive_count = 0
            best_robot = max(robots, key=lambda r: r.fitness)

            for robot in robots:
                robot.update(track_surface)
                if robot.alive:
                    alive_count += 1

            if alive_count == 0:
                if solo_mode:
                    robots = spawn_robots(start_pos, saved_brain, solo_mode=True)
                else:
                    robots.sort(key=lambda r: r.fitness, reverse=True)
                    top_performers = robots[:5]
                    fitness_history.append(top_performers[0].fitness)

                    if top_performers[0].fitness > all_time_best_fitness:
                        all_time_best_fitness = top_performers[0].fitness
                        saved_brain = top_performers[0].brain.copy()
                        saved_brain.save("best_brain.npz", generation)

                    new_robots = [Robot(start_pos[0], start_pos[1], 0, top_performers[0].brain.copy())]
                    
                    # Sisipkan 4 robot acak baru di setiap generasi
                    for _ in range(4):
                        new_robots.append(Robot(start_pos[0], start_pos[1], 0))

                    while len(new_robots) < POPULATION_SIZE:
                        parent = random.choice(top_performers)
                        child_brain = parent.brain.copy()
                        child_brain.mutate(rate=0.25, scale=0.5)
                        new_robots.append(Robot(start_pos[0], start_pos[1], 0, child_brain))

                    robots = new_robots
                    generation += 1
                best_robot = robots[0]
                break

        screen.blit(track_surface, (0, 0))
        draw_neural_network(screen, best_robot, font)
        
        for robot in robots:
            robot.draw(screen, is_best=(robot == best_robot))

        gui.draw(
            screen, best_robot, generation, alive_count, 
            POPULATION_SIZE, solo_mode, sim_speeds[speed_idx], fitness_history
        )

        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()

if __name__ == "__main__":
    main()
