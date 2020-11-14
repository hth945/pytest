#%%
import paddle
import paddle.fluid as fluid
import paddle.fluid.compiler as compiler
import numpy
import os

paddle.enable_static()

use_cuda = True
place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
exe = fluid.Executor(place)

train_program = fluid.Program()
startup_program = fluid.Program()
with fluid.program_guard(train_program, startup_program):
    data = fluid.layers.data(name='X', shape=[1], dtype='float32')
    hidden = fluid.layers.fc(input=data, size=10)
    loss = fluid.layers.mean(hidden)
    fluid.optimizer.SGD(learning_rate=0.01).minimize(loss)

# 仅运行一次startup program
# 不需要优化/编译这个startup program
startup_program.random_seed=1
exe.run(startup_program)
#%%
# 无需编译，直接运行main program
x = numpy.random.random(size=(10, 1)).astype('float32')
loss_data, = exe.run(train_program,
                 feed={"X": x},
                 fetch_list=[loss.name])

# 另一种方法是，编译这个main program然后运行。
# 参考CompiledProgram以获取更多信息。
# 注意：如果你使用CPU运行程序，需要具体设置CPU_NUM，
# 否则fluid会把逻辑核的所有数目设为CPU_NUM，
# 在这种情况下，输入的batch size应大于CPU_NUM，
# 否则程序会异常中断。
if not use_cuda:
    os.environ['CPU_NUM'] = str(2)

compiled_prog = compiler.CompiledProgram(
    train_program).with_data_parallel(
    loss_name=loss.name)
loss_data, = exe.run(compiled_prog,
                     feed={"X": x},
                     fetch_list=[loss.name])
                     
# %%


import paddle.fluid as fluid
import numpy

#首先创建执行引擎
place = fluid.CPUPlace() # fluid.CUDAPlace(0)
exe = fluid.Executor(place)

data = fluid.layers.data(name='X', shape=[1], dtype='float32')
hidden = fluid.layers.fc(input=data, size=10)
loss = fluid.layers.mean(hidden)
adam = fluid.optimizer.Adam()
adam.minimize(loss)

#仅运行startup程序一次
exe.run(fluid.default_startup_program())

x = numpy.random.random(size=(10, 1)).astype('float32')
outs = exe.run(feed={'X': x},
               fetch_list=[loss.name])