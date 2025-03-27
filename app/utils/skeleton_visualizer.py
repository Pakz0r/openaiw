import json
import sys
from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *

FORWARD_OFFSET = 0

BONE_MIN_CONFIDENCE = 0.25

BODY25_CONNECTIONS = [
    (0, 1), (1, 8),                 # Spine
    (1, 2), (2, 3), (3, 4),         # Right shoulder and arm
    (1, 5), (5, 6), (6, 7),         # Left shoulder and arm
    (8, 9), (9, 10), (10, 11),      # Hip and right leg
    (8, 12), (12, 13), (13, 14)     # Hip and left leg
]

class SkeletonVisualizer:
    def __init__(self, window_title="3D Skeleton Visualization", window_size=(800, 600)):
        self.window_title = window_title
        self.window_width, self.window_height = window_size
        self.people = []
        self.init_glut()
        self.init_opengl()

    def init_glut(self):
        glutInit(sys.argv)
        glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH)
        glutInitWindowSize(self.window_width, self.window_height)
        glutCreateWindow(self.window_title.encode("utf-8"))
        glutDisplayFunc(self.display)
        glutReshapeFunc(self.reshape)
        glutKeyboardFunc(self.keyboard)

    def init_opengl(self):
        glEnable(GL_DEPTH_TEST)

    def load_data_from_json(self, json_string):
        """Carica i dati dello scheletro da una stringa JSON."""
        data = json.loads(json_string)
        self.people = data.get("People", [])

    def draw_grid(self, size=4, step=0.5):
        """Disegna una griglia 3D per orientarsi nello spazio."""
        glColor3f(0.3, 0.3, 0.3)
        glBegin(GL_LINES)
        for i in range(-size, size + 1):
            glVertex3f(i * step, 0, -size * step)
            glVertex3f(i * step, 0, size * step)
            glVertex3f(-size * step, 0, i * step)
            glVertex3f(size * step, 0, i * step)
        glEnd()

    def draw_sphere(self, position, radius=0.05):
        """Disegna una sfera."""
        glPushMatrix()
        glTranslatef(*position)
        glColor3f(0, 1, 0)  # Verde
        glutSolidSphere(radius, 10, 10)
        glPopMatrix()

    def draw_line(self, p1, p2):
        """Disegna una linea tra due punti."""
        glColor3f(0.8, 0.8, 0.8)  # Grigio chiaro
        glBegin(GL_LINES)
        glVertex3f(*p1)
        glVertex3f(*p2)
        glEnd()

    def draw_head(self, position, rotation):
        """Disegna un cubo orientato per rappresentare la testa."""
        glPushMatrix()
        glTranslatef(*position)
        glRotatef(rotation[0], 1, 0, 0)  # Pitch
        glRotatef(rotation[2], 0, 1, 0)  # Yaw
        glRotatef(rotation[1], 0, 0, 1)  # Roll
        glColor3f(1, 0, 0)  # Rosso
        glutWireCube(0.2)
        glPopMatrix()

    def display(self):
        """Funzione di rendering principale."""
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()
        gluLookAt(0, 2, 6, 0, 1, 0, 0, 1, 0)

        # Disegna la griglia
        self.draw_grid()

        glColor3f(1, 1, 1)
        for person in self.people:
            skeleton = person.get("skeleton", [])
            points = {bone["pointID"]: (bone["x"], bone["y"], -bone["z"] + FORWARD_OFFSET) for bone in skeleton}

            # Disegna i punti dello scheletro come sfere
            for i, point in points.items():
                if skeleton[i]["confidence"] > BONE_MIN_CONFIDENCE:
                    self.draw_sphere(point)

            # Disegna le ossa dello scheletro
            for connection in BODY25_CONNECTIONS:
                i, j = connection
                can_draw = skeleton[i]["confidence"] > BONE_MIN_CONFIDENCE and skeleton[j]["confidence"] > BONE_MIN_CONFIDENCE
                if can_draw and i in points and j in points:
                    self.draw_line(points[i], points[j])

            # Disegna la testa come un cubo orientato
            if skeleton[0]["confidence"] > BONE_MIN_CONFIDENCE:
                head_pos = points[skeleton[0]["pointID"]]
                rotation = (
                    person["face_rotation"]["pitch"],
                    person["face_rotation"]["roll"],
                    person["face_rotation"]["yaw"]
                )
                self.draw_head(head_pos, rotation)

        glutSwapBuffers()

    def reshape(self, w, h):
        """Gestisce il ridimensionamento della finestra."""
        glViewport(0, 0, w, h)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(45, w / h, 0.1, 100.0)
        glMatrixMode(GL_MODELVIEW)

    def keyboard(self, key, x, y):
        self.last_key = key

    def render_frame(self):
        """Renderizza un singolo frame senza bloccare il programma."""
        glutPostRedisplay()
        glutMainLoopEvent()

    def get_key_pressed(self):
        if not hasattr(self, "last_key"): 
            return None
        key = self.last_key
        self.last_key = None
        return key

