# -*- coding: utf-8 -*-
# Ana Lucia Hernandez
# 17138
# Graficas por Computadora 
#

import module as s
import math 
s.bm = s.Bitmap(2400,2400)
tcat = s.Texture("cattexture.bmp")
light = s.Vector3(0,1,1)
#             obj      mtl   texture  translate      scale      rotate     eye       up      center
#s.bm.load("cat.obj", "cat.mtl", tcat, (-0.5,0,0), (0.45,0.45,0.45), (0,0,0), (0,1,5), (0,1,0),(0,0,0), light)
tduck = s.Texture("duck.bmp")
#s.bm.load("duck.obj", "duck.mtl", tduck, (0.5,0,0), (0.45,0.45,0.45), (0,0,0), (0,1,5), (0,1,0),(0,0,0), light)
tdog = s.Texture("dalmata.bmp")
s.bm.load("dalmata.obj", "dalmata.mtl", tdog, (-0.5,0,0), (0.3,0.3,0.3), (0,0,0), (0,1,5), (0,1,0),(0,0,0), light)
s.glFinish("escena")