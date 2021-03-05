#%%

from ctypes import *
from SetupAPI import *

dwSize = DWORD()	# GUID container size.
ptrGUID = GUID()	# GUID container.


bResult = SetupAPI.SetupDiClassGuidsFromNameW("Monitor", None, 0, pointer(dwSize))
if (bResult is False):
    if GetLastError() == ERROR_INSUFFICIENT_BUFFER:
        GUIDs = GUID * dwSize.value
        ptrGUID = GUIDs()
        bResult = SetupAPI.SetupDiClassGuidsFromNameW("Monitor", pointer(ptrGUID[0]), dwSize, pointer(dwSize))

devINFO = HANDLE(-1)
devINFO = SetupAPI.SetupDiGetClassDevsW(pointer(ptrGUID[0]), None, None, 2)

devDATA = SP_DEVINFO_DATA()
devDATA.cbSize = sizeof(SP_DEVINFO_DATA)

devFOUND = True
index = 0


# %%
dwSize.value
# %%
# %%
GetLastError()
# %%
ERROR_INSUFFICIENT_BUFFER
# %%
bResult = SetupAPI.SetupDiClassGuidsFromNameW("Monitor", None, 0, pointer(dwSize))

if GetLastError() == ERROR_INSUFFICIENT_BUFFER:
    print('1')
# %%
