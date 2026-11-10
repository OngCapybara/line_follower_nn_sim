import pygame
from config import WHITE, GREEN, RED, GRAY

def draw_neural_network(screen, robot, font):
    start_x = 260
    start_y = 175
    layer_spacing = 135
    node_spacing = 20
    
    inputs = robot.sensors
    hiddens = robot.brain.last_hidden
    outputs = robot.brain.last_output
    
    layers = [inputs, hiddens, outputs]
    layer_names = ["Input (11)", "Hidden (8)", "Output (2)"]
    
    node_positions = []
    max_nodes = max(len(l) for l in layers)
    total_height = max_nodes * node_spacing
    
    for l_idx, layer in enumerate(layers):
        x = start_x + (l_idx * layer_spacing)
        layer_height = len(layer) * node_spacing
        offset_y = (total_height - layer_height) / 2
        
        positions = []
        for n_idx in range(len(layer)):
            y = start_y + offset_y + (n_idx * node_spacing)
            positions.append((int(x), int(y)))
        node_positions.append(positions)

    # 1. Garis Bobot
    weights = [robot.brain.w1, robot.brain.w2]
    for l_idx, w_matrix in enumerate(weights):
        for i, start_pos in enumerate(node_positions[l_idx]):
            for j, end_pos in enumerate(node_positions[l_idx + 1]):
                weight_val = w_matrix[i][j]
                color = (40, 180, 40) if weight_val > 0 else (180, 40, 40)
                thickness = max(1, int(abs(weight_val) * 2))
                pygame.draw.line(screen, color, start_pos, end_pos, thickness)

    # 2. Lingkaran Node
    for l_idx, positions in enumerate(node_positions):
        label = font.render(layer_names[l_idx], True, GRAY)
        screen.blit(label, (positions[0][0] - 25, start_y - 25))
        
        for n_idx, pos in enumerate(positions):
            val = layers[l_idx][n_idx]
            if val > 0.2:
                node_color = GREEN
            elif val < -0.2:
                node_color = RED
            else:
                node_color = (60, 60, 60)
                
            pygame.draw.circle(screen, WHITE, pos, 8)
            pygame.draw.circle(screen, node_color, pos, 6)
