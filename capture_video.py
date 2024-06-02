import cv2
import pygame
import random
import numpy as np

# Particle class
class Particle:
    def __init__(self, pos):
        self.x, self.y = pos
        self.vx = random.uniform(-1, 1)
        self.vy = random.uniform(-2, -1)  # Strong upward velocity
        self.alpha = 255
        self.radius = random.randint(1, 4)

    def update(self):
        self.x += self.vx
        self.y += self.vy
        self.alpha -= 2
        if self.alpha < 0:
            self.alpha = 0

    def draw(self, screen):
        if self.alpha > 0:
            s = pygame.Surface((self.radius * 2, self.radius * 2), pygame.SRCALPHA)
            pygame.draw.circle(s, (200, 200, 200, self.alpha), (self.radius, self.radius), self.radius)
            screen.blit(s, (self.x - self.radius, self.y - self.radius))

def capture_video(source=0):
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise ValueError("Error: Could not open video source.")
    ret, frame = cap.read()
    if not ret:
        raise ValueError("Error: Could not read video frame.")
    frame_height, frame_width = frame.shape[:2]
    yield frame_width, frame_height
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        yield frame
    cap.release()

def bezier_curve(t, p0, p1, p2):
    """Compute the quadratic Bezier curve for a given t."""
    x = (1 - t)**2 * p0[0] + 2 * (1 - t) * t * p1[0] + t**2 * p2[0]
    y = (1 - t)**2 * p0[1] + 2 * (1 - t) * t * p1[1] + t**2 * p2[1]
    return x, y

def detect_eyes(frame, face_cascade, eye_cascade):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    eyes_coords = []
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 3)
        for (ex, ey, ew, eh) in eyes:
            # Define points for the quadratic Bezier curve (eyebrow shape)
            p0 = (x + ex, y + ey)
            p1 = (x + ex + ew // 2, y + ey - eh // 2)  # Control point above the eye
            p2 = (x + ex + ew, y + ey)
            for t in np.linspace(0, 1, num=ew):
                bx, by = bezier_curve(t, p0, p1, p2)
                eyes_coords.append((bx, by))
    return eyes_coords

def update_particles(eyes_coords, particles, frame_width):
    for (ex, ey) in eyes_coords:
        if random.random() > 0.9:  # Reduce the frequency of particle emission
            ex = frame_width - ex  # Adjust for the mirrored video feed
            particles.append(Particle((ex, ey)))

def draw_particles(particles, screen):
    for particle in particles:
        particle.update()
        particle.draw(screen)
    particles = [p for p in particles if p.alpha > 0]
    return particles

def main():
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

    particles = []

    pygame.init()

    # Initialize video capture and get frame size
    cap_gen = capture_video(source=1)  # Change to 0 if needed
    frame_width, frame_height = next(cap_gen)
    screen = pygame.display.set_mode((frame_width, frame_height))
    pygame.display.set_caption("Eye Detection with Smoke Particles")

    clock = pygame.time.Clock()

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        try:
            frame = next(cap_gen)
        except StopIteration:
            break

        eyes_coords = detect_eyes(frame, face_cascade, eye_cascade)
        update_particles(eyes_coords, particles, frame_width)

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = np.rot90(frame)
        frame = pygame.surfarray.make_surface(frame)

        screen.blit(frame, (0, 0))
        particles = draw_particles(particles, screen)
        pygame.display.flip()
        clock.tick(30)

    pygame.quit()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
