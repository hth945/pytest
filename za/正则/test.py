
# %%

# #op_break
# <PROCESS>B-INSPECTION
# <PROCESS>D-INSPECTION
# <MODEL>X1521
# 708
# 705
# 717
# 706
# 3007

import re
line = "#op_break"

matchObj = re.search( r'^(?i)#.+$', line, re.M|re.I)
if matchObj:
   print ("match --> matchObj.group() : ", matchObj.group())
   print ("matchObj.group(1) : ", matchObj.group(0))
else:
   print ("No match!!")

print ("line : ", line)
num = re.sub(r'^(?i)#\s?', "", line)
print ("num : ", num)
# %%


line = "GADW123456"

matchObj = re.search( r'^(?i)GAD(W|S)\d{6}$', line, re.M|re.I)
if matchObj:
   print ("match --> matchObj.group() : ", matchObj.group())
   print ("matchObj.group(1) : ", matchObj.group(0))

# %%
line = ""
for i in range(ord('A'),ord('Z')+1):
   line += chr(i)
for i in range(ord('0'),ord('9')+1):
   line += chr(i)
for i in range(ord('A'),ord('Z')+1):
   line += chr(i)
for i in range(ord('0'),ord('9')+1):
   line += chr(i)
for i in range(ord('A'),ord('R')+1):
   line += chr(i)
print(line)
print(len(line))
line = line[:17]+"+"+line[17:17+14]+"+"+line[17+14:]
s=""
for i in range(96-len(line)):
   s += "+"
line += s 
print(line)
print(len(line))

matchObj = re.search( r'^(?i)[A-Z0-9]{17}\+[A-Z0-9]{14}\+[A-Z0-9|+]{63}$', line, re.M|re.I)
# matchObj = re.search( r'^(?i)[A-Z0-9]{3}\-[A-Z0-9]{9}\-[A-Z0-9]{4}$', line, re.M|re.I)
if matchObj:
   print ("match --> matchObj.group() : ", matchObj.group())
   print ("matchObj.group(1) : ", matchObj.group(0))
# %%


