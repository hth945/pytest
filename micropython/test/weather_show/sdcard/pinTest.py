from machine import Pin



p_out = Pin(("PB2", 16*1+2), Pin.OUT_PP)

p_out.value(1)                 # set io high
p_out.value(0)                 # set io low

p_out.init(Pin.IN, Pin.PULL_UP)
print(p_out.value())           # get value, 0 or 1

print(p_out.name())