import pygame
import math
import random
import os

# --- Colors with modern palette ---
WHITE = (255, 255, 255)
BLACK = (20, 20, 30)
DARK_BG = (25, 28, 38)
CARD_BG = (35, 38, 52)
BLUE = (79, 172, 254)
BLUE_GLOW = (79, 172, 254, 100)
RED = (255, 107, 107)
RED_GLOW = (255, 107, 107, 100)
GREEN = (94, 234, 212)
GREEN_GLOW = (94, 234, 212, 100)
PURPLE = (167, 139, 250)
GOLD = (251, 191, 36)
GRAY = (107, 114, 128)
LIGHT_GRAY = (156, 163, 175)

# --- Screen size ---
SCREEN_WIDTH = 900
SCREEN_HEIGHT = 600

ASSETS_DIR = os.path.join(os.path.dirname(__file__), "..", "assets")

class Particle:
    def __init__(self, x, y, color, velocity):
        self.x = x
        self.y = y
        self.color = color
        self.velocity = velocity
        self.lifetime = 1.0
        self.size = random.uniform(2, 5)
    
    def update(self, dt):
        self.x += self.velocity[0] * dt
        self.y += self.velocity[1] * dt
        self.lifetime -= dt * 0.5
        self.velocity = (self.velocity[0] * 0.98, self.velocity[1] * 0.98)
    
    def is_alive(self):
        return self.lifetime > 0
    
    def draw(self, surface):
        if self.lifetime > 0:
            alpha = int(255 * self.lifetime)
            color_with_alpha = (*self.color[:3], alpha)
            size = int(self.size * self.lifetime)
            if size > 0:
                s = pygame.Surface((size * 2, size * 2), pygame.SRCALPHA)
                pygame.draw.circle(s, color_with_alpha, (size, size), size)
                surface.blit(s, (int(self.x - size), int(self.y - size)))

class AnimatedMetric:
    def __init__(self, target_value):
        self.current = target_value
        self.target = target_value
        self.velocity = 0
    
    def update(self, new_target, dt):
        self.target = new_target
        # Spring physics for smooth animation
        spring_strength = 8.0
        damping = 0.7
        
        force = (self.target - self.current) * spring_strength
        self.velocity += force * dt
        self.velocity *= damping
        self.current += self.velocity * dt
        
        # Clamp to valid range
        self.current = max(0, min(1, self.current))

class PulseEffect:
    def __init__(self):
        self.time = 0
    
    def update(self, dt):
        self.time += dt
    
    def get_scale(self, frequency=2.0, amplitude=0.05):
        return 1.0 + math.sin(self.time * frequency * math.pi) * amplitude

class Renderer:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("Socratic Tutor - AI Learning System")
        self.clock = pygame.time.Clock()
        self.fps =8
        
        # Fonts
        self.title_font = pygame.font.SysFont("Arial", 32, bold=True)
        self.font = pygame.font.SysFont("Arial", 18)
        self.small_font = pygame.font.SysFont("Arial", 16)
        self.metric_font = pygame.font.SysFont("Arial", 16, bold=True)
        
        # Animation states
        self.animated_engagement = AnimatedMetric(0.5)
        self.animated_confusion = AnimatedMetric(0.5)
        self.animated_effort = AnimatedMetric(0.5)
        
        # Particles
        self.particles = []
        
        # Pulse effects
        self.pulse = PulseEffect()
        
        # Connection animation
        self.connection_offset = 0
        
        # Action flash
        self.action_flash_time = 0
        self.last_action = -1
        
        # Load avatars from assets
        try:
            self.student_avatar = pygame.image.load(os.path.join(ASSETS_DIR, "student1.png")).convert_alpha()
            self.tutor_avatar = pygame.image.load(os.path.join(ASSETS_DIR, "teacher.png")).convert_alpha()
            # Scale to larger size for better visibility
            self.student_avatar = pygame.transform.scale(self.student_avatar, (120, 120))
            self.tutor_avatar = pygame.transform.scale(self.tutor_avatar, (120, 120))
        except:
            # Fallback to generated avatars if images not found
            self.student_avatar = self.create_avatar_surface(BLUE, "S")
            self.tutor_avatar = self.create_avatar_surface(PURPLE, "T")
        
        # Background gradient
        self.background = self.create_gradient_background()
        
        self.last_time = pygame.time.get_ticks() / 1000.0

    def create_avatar_surface(self, color, letter):
        """Create a modern circular avatar with letter"""
        size = 100
        surface = pygame.Surface((size, size), pygame.SRCALPHA)
        
        # Outer glow
        for i in range(5):
            alpha = 30 - i * 5
            radius = size // 2 - i
            glow_color = (*color, alpha)
            s = pygame.Surface((size, size), pygame.SRCALPHA)
            pygame.draw.circle(s, glow_color, (size // 2, size // 2), radius)
            surface.blit(s, (0, 0))
        
        # Main circle
        pygame.draw.circle(surface, color, (size // 2, size // 2), size // 2 - 5)
        
        # Inner circle
        inner_color = tuple(min(c + 30, 255) for c in color)
        pygame.draw.circle(surface, inner_color, (size // 2, size // 2), size // 2 - 10)
        
        # Letter
        font = pygame.font.SysFont("Arial", 40, bold=True)
        text = font.render(letter, True, WHITE)
        text_rect = text.get_rect(center=(size // 2, size // 2))
        surface.blit(text, text_rect)
        
        return surface

    def create_gradient_background(self):
        """Create a classroom-themed gradient background"""
        surface = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))
        # Warm classroom colors - beige to light brown gradient
        for y in range(SCREEN_HEIGHT):
            progress = y / SCREEN_HEIGHT
            color = (
                int(245 - progress * 30),  # Beige to tan
                int(240 - progress * 35),
                int(230 - progress * 50)
            )
            pygame.draw.line(surface, color, (0, y), (SCREEN_WIDTH, y))
        
        # Add subtle classroom elements
        # Chalkboard texture at top
        pygame.draw.rect(surface, (45, 55, 50), (0, 0, SCREEN_WIDTH, 70), 0)
        
        # Wood desk texture at bottom
        for i in range(10):
            wood_color = (139 + i * 2, 90 + i, 43 + i // 2)
            pygame.draw.rect(surface, wood_color, (0, SCREEN_HEIGHT - 30 + i * 3, SCREEN_WIDTH, 3))
        
        return surface

    def draw_card(self, surface, rect, title=None):
        """Draw a classroom-themed card"""
        # Shadow
        shadow_rect = rect.move(3, 3)
        s = pygame.Surface((shadow_rect.width, shadow_rect.height), pygame.SRCALPHA)
        pygame.draw.rect(s, (0, 0, 0, 40), s.get_rect(), border_radius=15)
        surface.blit(s, shadow_rect)
        
        # Card background - paper white with slight texture
        pygame.draw.rect(surface, (255, 253, 245), rect, border_radius=15)
        
        # Subtle border
        pygame.draw.rect(surface, (200, 195, 180), rect, border_radius=15, width=2)
        
        if title:
            title_surface = self.small_font.render(title, True, (80, 70, 60))
            surface.blit(title_surface, (rect.x + 15, rect.y + 12))

    def draw_animated_bar(self, surface, x, y, width, height, value, color, glow_color, label):
        """Draw an animated progress bar with glow effect"""
        # Background
        bg_rect = pygame.Rect(x, y, width, height)
        pygame.draw.rect(surface, (50, 53, 65), bg_rect, border_radius=10)
        
        # Progress bar with glow
        if value > 0:
            progress_width = int(width * value)
            progress_rect = pygame.Rect(x, y, progress_width, height)
            
            # Glow effect
            glow_surface = pygame.Surface((progress_width + 20, height + 20), pygame.SRCALPHA)
            pygame.draw.rect(glow_surface, glow_color, 
                           pygame.Rect(10, 10, progress_width, height), border_radius=10)
            
            # Blur effect (simplified)
            for i in range(3):
                offset = 3 - i
                alpha = 60 - i * 20
                glow_rect = pygame.Rect(10 - offset, 10 - offset, 
                                       progress_width + offset * 2, height + offset * 2)
                s = pygame.Surface(glow_surface.get_size(), pygame.SRCALPHA)
                pygame.draw.rect(s, (*color, alpha), glow_rect, border_radius=10)
                glow_surface.blit(s, (0, 0), special_flags=pygame.BLEND_RGBA_ADD)
            
            surface.blit(glow_surface, (x - 10, y - 10))
            
            # Main bar
            pygame.draw.rect(surface, color, progress_rect, border_radius=10)
        
        # Label
        label_surface = self.metric_font.render(label, True, BLACK)
        surface.blit(label_surface, (x, y - 25))
        
        # Percentage
        percentage = f"{int(value * 100)}%"
        perc_surface = self.small_font.render(percentage, True, LIGHT_GRAY)
        surface.blit(perc_surface, (x + width - perc_surface.get_width(), y - 25))

    def emit_particles(self, x, y, color, count=10):
        """Emit particles from a position"""
        for _ in range(count):
            angle = random.uniform(0, math.pi * 2)
            speed = random.uniform(30, 80)
            velocity = (math.cos(angle) * speed, math.sin(angle) * speed)
            self.particles.append(Particle(x, y, color, velocity))

    def draw_connection_line(self, surface, start_pos, end_pos):
        """Draw animated connection between avatars"""
        # Dashed line with animation
        segments = 20
        for i in range(segments):
            progress = i / segments
            offset = (self.connection_offset + progress) % 1.0
            
            if offset < 0.5:  # Draw segment
                x1 = start_pos[0] + (end_pos[0] - start_pos[0]) * progress
                y1 = start_pos[1] + (end_pos[1] - start_pos[1]) * progress
                x2 = start_pos[0] + (end_pos[0] - start_pos[0]) * (progress + 0.03)
                y2 = start_pos[1] + (end_pos[1] - start_pos[1]) * (progress + 0.03)
                
                # Glow
                glow_surface = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA)
                pygame.draw.line(glow_surface, (*BLUE, 60), (x1, y1), (x2, y2), 6)
                surface.blit(glow_surface, (0, 0))
                
                # Main line
                pygame.draw.line(surface, BLUE, (x1, y1), (x2, y2), 3)

    def reset(self, state):
        self.render(state, action=-1, step=0, reward=0)

    def render(self, state, action=-1, step=0, reward=0):
        # Calculate delta time
        current_time = pygame.time.get_ticks() / 1000.0
        dt = current_time - self.last_time
        self.last_time = current_time
        
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return
        
        # Update animations
        self.animated_engagement.update(state["engagement"], dt)
        self.animated_confusion.update(state["confusion"], dt)
        self.animated_effort.update(state["effort"], dt)
        self.pulse.update(dt)
        self.connection_offset += dt * 0.5
        
        # Update particles
        for particle in self.particles[:]:
            particle.update(dt)
            if not particle.is_alive():
                self.particles.remove(particle)
        
        # Action flash effect
        if action != -1 and action != self.last_action:
            self.action_flash_time = 0.5
            self.last_action = action
            # Emit particles at center
            self.emit_particles(SCREEN_WIDTH // 2, 120, GOLD, 15)
        
        self.action_flash_time = max(0, self.action_flash_time - dt)
        
        # Draw background
        self.screen.blit(self.background, (0, 0))
        
        # Draw title like chalk on chalkboard
        title_text = "Socratic Learning Classroom"
        title_surface = self.title_font.render(title_text, True, (240, 240, 230))
        title_rect = title_surface.get_rect(center=(SCREEN_WIDTH // 2, 35))
        
        # Chalk texture effect
        self.screen.blit(title_surface, title_rect)
        
        # Draw action card with flash effect
        action_card_rect = pygame.Rect(SCREEN_WIDTH // 2 - 200, 80, 400, 70)
        self.draw_card(self.screen, action_card_rect)
        
        action_text = self.get_action_text(action)
        action_color = WHITE if self.action_flash_time <= 0 else GOLD
        action_surface = self.font.render(f"Action: {action_text}", True, action_color)
        action_rect = action_surface.get_rect(center=(SCREEN_WIDTH // 2, 115))
        self.screen.blit(action_surface, action_rect)
        
        # Draw avatars with pulse effect
        student_scale = self.pulse.get_scale(2.0, 0.03)
        student_size = int(100 * student_scale)
        student_scaled = pygame.transform.scale(self.student_avatar, (student_size, student_size))
        student_pos = (150 - student_size // 2, 250)
        self.screen.blit(student_scaled, student_pos)
        
        tutor_scale = self.pulse.get_scale(2.5, 0.03)
        tutor_size = int(100 * tutor_scale)
        tutor_scaled = pygame.transform.scale(self.tutor_avatar, (tutor_size, tutor_size))
        tutor_pos = (SCREEN_WIDTH - 150 - tutor_size // 2, 250)
        self.screen.blit(tutor_scaled, tutor_pos)
        
        # Draw connection line
        self.draw_connection_line(self.screen, (200, 300), (SCREEN_WIDTH - 200, 300))
        
        # Labels under avatars
        student_label = self.font.render("Student", True, BLUE)
        student_label_rect = student_label.get_rect(center=(150, 360))
        self.screen.blit(student_label, student_label_rect)
        
        tutor_label = self.font.render("AI Tutor", True, PURPLE)
        tutor_label_rect = tutor_label.get_rect(center=(SCREEN_WIDTH - 150, 360))
        self.screen.blit(tutor_label, tutor_label_rect)
        
        # Draw metrics card
        metrics_card_rect = pygame.Rect(50, 400, SCREEN_WIDTH - 100, 190)
        self.draw_card(self.screen, metrics_card_rect, "Learning Metrics")
        
        # Draw animated metrics bars
        bar_width = 200
        bar_height = 13
        bar_spacing = 38
        start_x = 150
        start_y = 470
        
        self.draw_animated_bar(self.screen, start_x, start_y, bar_width, bar_height,
                              self.animated_engagement.current, BLUE, BLUE_GLOW, "Engagement")
        self.draw_animated_bar(self.screen, start_x, start_y + bar_spacing, bar_width, bar_height,
                              self.animated_confusion.current, RED, RED_GLOW, "Confusion")
        self.draw_animated_bar(self.screen, start_x, start_y + bar_spacing * 2, bar_width, bar_height,
                              self.animated_effort.current, GREEN, GREEN_GLOW, "Effort")
        
        # Stats card
        stats_x = start_x + bar_width + 260
        stats_y = 430
        
        # Step counter
        step_surface = self.metric_font.render(f"Step: {step}", True, WHITE)
        self.screen.blit(step_surface, (stats_x, stats_y))
        
        # Reward with color coding
        reward_color = GREEN if reward >= 0 else RED
        reward_surface = self.metric_font.render(f"Reward: {reward:+.2f}", True, reward_color)
        self.screen.blit(reward_surface, (stats_x, stats_y + 40))
        
        # Learning score (composite metric)
        score = (state["engagement"] * 0.4 + state["effort"] * 0.4 + 
                (1 - state["confusion"]) * 0.2) * 100
        score_surface = self.metric_font.render(f"Score: {score:.0f}/100", True, GOLD)
        self.screen.blit(score_surface, (stats_x, stats_y + 80))
        
        # Draw particles
        for particle in self.particles:
            particle.draw(self.screen)
        
        # Update display
    

        pygame.display.flip()
        self.clock.tick(80)


    def get_action_text(self, action):
        actions = {
            -1: "Initializing...",
            0: "Ask Socratic Question",
            1: "Give Hint",
            2: "Provide Code Example",
            3: "Encourage Reflection",
            4: "Ask Student to Explain"
        }
        return actions.get(action, "Unknown Action")

    def close(self):
        pygame.quit()