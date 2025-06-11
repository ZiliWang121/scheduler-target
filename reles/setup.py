from distutils.core import setup, Extension

setup(name='mpsched',ext_modules=[Extension('mpsched',['mpsched.c'],
				include_dirs=['/usr/src/linux-headers-5.4.230-custom/include/uapi',
				'/usr/src/linux-headers-5.4.230-custom/include','/usr/include/python3.8'],
				define_macros=[('NUM_SUBFLOWS','2'),('SOL_TCP','6')]) 
				])
