# -*- coding: utf-8 -*-
# Ana Lucia Hernandez
# 17138
# Graficas por Computadora 
#

import module as s
import math 
s.bm = s.Bitmap(2400,2400)
light = s.Vector3(0,1,1)
'''
tcat = s.Texture("cattexture.bmp")

#             obj      mtl   texture  translate         scale            rotate         eye       up   center      light
s.bm.load("cat.obj", "cat.mtl", tcat, (-0.5,0,0), (0.45,0.45,0.45), (0,math.pi/4,0), (0,1,5), (0,1,0),(0,0,0), s.Vector3(-1.5,1,5))
'''
tduck = s.Texture("duck.bmp")
s.bm.load("duck.obj", "duck.mtl", tduck, (0.5,0,0), (0.45,0.45,0.45), (0,-math.pi/6,0), (0,1,5), (0,1,0),(0,0,0), light)
'''
tdog = s.Texture("dalmata.bmp")
s.bm.load("dalmata.obj", "dalmata.mtl", tdog, (-0.5,0,0), (0.3,0.3,0.3), (0,0,0), (0,1,5), (0,1,0),(0,0,0), light)

'''
s.glFinish("escena")