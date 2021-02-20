from machine import UART

uart = UART(7)                         # init with given baudrate
uart.init(115200, bits=8, parity=None, stop=1,read_buf_len=4096) # init with given parameters
uart.write('abc')   # write the 3 characters
uart.read(10)       # read 10 characters, returns a bytes object
uart.read()         # read all available characters
uart.readline()     # read a line
uart.readinto(buf)  # read and store into the given buffer
uart.write('abc')   # write the 3 characters

