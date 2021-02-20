#%%
import struct
a=bytes('hello ',encoding="ascii")
b=bytes('world!',encoding="ascii")
c=2
d=45.123
structBytes=struct.pack('4s5sbf',a,b,c,d)

print(structBytes)
print(len(structBytes))

a, b, c, d = struct.unpack("4s5sbf", structBytes)
print(a, b, c, d)


# %%


# FORMAT	C TYPE	PYTHON TYPE	STANDARD SIZE	NOTES
# x	pad byte	no value		
# c	char	string of length 1	1	
# b	signed char	integer	1	(3)
# B	unsigned char	integer	1	(3)
# ?	_Bool	bool	1	(1)
# h	short	integer	2	(3)
# H	unsigned short	integer	2	(3)
# i	int	integer	4	(3)
# I	unsigned int	integer	4	(3)
# l	long	integer	4	(3)
# L	unsigned long	integer	4	(3)
# q	long long	integer	8	(2), (3)
# Q	unsigned long long	integer	8	(2), (3)
# f	float	float	4	(4)
# d	double	float	8	(4)
# s	char[]	string		
# p	char[]	string		
# P	void *	integer		(5), (3)
