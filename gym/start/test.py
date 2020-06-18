import pyglet

game_window = pyglet.window.Window(
    width=400,
    height=300,
    caption="古明地觉",
    resizable=True
)

# 创建Label对象
label = pyglet.text.Label('Hello, world',
                          font_size=25,  # 字体不指定，使用默认的，大小为25
                          x=game_window.width//2,
                          y=game_window.height//2,
                          anchor_x='center', anchor_y='center'
                          )


# 下面问题来了，我们要如何将字体显示在上面呢？
# 首先显示文本内容,可以通过label.draw()方法
# 但是我们直接写label.draw()是不行的，因为这样无法显示在窗口上面，显示不到窗口上面是无意义的
# 这里我们说一下，当pyglet创建窗口的时候，会调用窗口的on_draw方法，也就是Window这个类的on_draw方法
# 我们只有将label.draw()写到这个on_draw方法里面，才可以实现。
# 一种办法是继承Window这个类，然后重写里面的on_draw方法，用继承Window的类创建窗口，但是这个显然不科学
# 另一种就是通过反射的方式

def show_label():
    # 将初始的窗口内容删除
    game_window.clear()
    # 添加文本，重新绘制窗口
    label.draw()


# 重写on_draw方法，以后就会执行我们在show_label里面指定的代码
setattr(game_window, "on_draw", show_label)


if __name__ == '__main__':
    pyglet.app.run()