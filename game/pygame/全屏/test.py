import pygame, sys
from pygame.locals import *


def main():
    global screen, WINDOWWIDTH, WINDOWHEIGHT
    SIZE = WINDOWWIDTH, WINDOWHEIGHT = 500, 500
    fps = pygame.time.Clock()
    isfullscreen = False
    pygame.init()
    screen = pygame.display.set_mode(SIZE, RESIZABLE)
    screen.fill((255,255,255)) # 背景颜色白色
    while True:
        isfullscreen = Resize(isfullscreen)
        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit()
                sys.exit()
            if event.type == KEYDOWN:
                if event.key == K_ESCAPE:
                    pygame.quit()
                    sys.exit()
        pygame.display.update()
        screen.fill((255,255,255))
        fps.tick(30)

def Resize(isfullscreen):
    # 这个函数中必须先判断窗口大小是否变化，在判断是否全屏
    # 否则，在全屏之后，pygame会判定为全屏操作也是改变窗体大小的一个操作，所以会显示一个比较大的窗口但不是全屏模式
    for event in pygame.event.get(VIDEORESIZE):
        size = WINDOWWIDTH, WINDOWHEIGHT = event.size[0], event.size[1]
        screen = pygame.display.set_mode((WINDOWWIDTH, WINDOWHEIGHT), RESIZABLE)
    for event in pygame.event.get(KEYDOWN):
        if event.key == K_F11:
            if not isfullscreen:
                isfullscreen = True
                SIZE = WINDOWWIDTH, WINDOWHEIGHT =  pygame.display.list_modes()[0]
                screen = pygame.display.set_mode(SIZE, FULLSCREEN)
            else:
                isfullscreen = False
                SIZE = WINDOWWIDTH, WINDOWHEIGHT = 1000, 800
                screen = pygame.display.set_mode((WINDOWWIDTH, WINDOWHEIGHT), RESIZABLE)
        pygame.event.post(event)
    return isfullscreen

if __name__ == '__main__':
    main()
