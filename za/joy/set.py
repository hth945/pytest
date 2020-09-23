#%%

import pyvjoy

#Pythonic API, item-at-a-time

j = pyvjoy.VJoyDevice(1)

j.set_button(1,1)

j.reset()
j.reset_buttons()
j.reset_povs()

j.data.lButtons = 19 # buttons number 1,2 and 5 (1+2+16)
j.data.wAxisX = 0x2000 
j.data.wAxisY= 0x7500

print(j.data)
j.update()



#%%
j.set_button(0,1)
#%%
#turn button 15 off again
j.set_button(3,0)
#%%
j.set_axis(pyvjoy.HID_USAGE_X, 0x1)

j.set_axis(pyvjoy.HID_USAGE_X, 0x8000)

j.reset()
j.reset_buttons()
j.reset_povs()

print(j.data)


j.data.lButtons = 19 # buttons number 1,2 and 5 (1+2+16)
j.data.wAxisX = 0x2000 
j.data.wAxisY= 0x7500

#send data to vJoy device
j.update()


#Lower-level API just wraps the functions in the DLL as thinly as possible, with some attempt to raise exceptions instead of return codes.
# %%
print(j.data)
# %%
j.set_axis(pyvjoy.HID_USAGE_X, 0x1)

j.set_axis(pyvjoy.HID_USAGE_X, 0x8000)

# %%
j.set_axis(pyvjoy.HID_USAGE_X, 3)
# %%
j.data.wAxisX = -0x7fff 
j.data.wAxisY= 0x7fff
j.data.wAxisZ = -0x7fff 
j.data.wAxisXRot= 0
j.data.wAxisYRot = 0x111
j.data.wAxisZRot=  0x111# -0x7fff
j.data.wSlider = 0 
j.data.wAxisVX= 0

j.update()
# %%
for i in range(10):
    print(i)
    j.set_axis(pyvjoy.HID_USAGE_X+i, 0x7fff )
# %%
j.set_axis(pyvjoy.HID_USAGE_WHL, 0x7fff )


# %%
