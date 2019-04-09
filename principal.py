# -*- coding: utf-8 -*-
# Ana Lucia Hernandez
# 17138
# Graficas por Computadora 
#

import module as s
import math 
s.bm = s.Bitmap(1800,1800)
text = s.Texture("cat-texture.bmp")
#             obj      mtl   texture  translate      scale      rotate     eye       up      center
s.bm.load("cat.obj", "cat.mtl", text, (0,0,0), (0.5, 0.5, 0.5), (0,0,0), (0,1,2), (0,1,0),(0,0,0))
s.glFinish("cube")
