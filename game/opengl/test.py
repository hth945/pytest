from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *


def drawFunc():
    glClear(GL_COLOR_BUFFER_BIT)
    glRotatef(1, 0, 1, 0)
    glColor3f(1.0, 1.0, 0.0)
    glutWireTeapot(0.5)
    glFlush()



if __name__ == '__main__':
    glutInit()
    glutInitDisplayMode(GLUT_SINGLE | GLUT_RGBA)
    glutInitWindowSize(3840, 2160)
    # 参数为b类型而不是string。我查资料时，很多网上代码未指出导致报错。
    glutCreateWindow(b"First")
    glutDisplayFunc(drawFunc)
    # glutIdleFunc(drawFunc)
    glutMainLoop()