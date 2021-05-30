# fontforge -script export_glyph_pngs.py <font file>

import os
from fontforge import *

with open("glyph_names.txt") as f:
    glyph_names = {line.strip() for line in f.readlines()}

font = open(os.sys.argv[1])
for glyph in font:
    glyph_name = font[glyph].glyphname
    if font[glyph].isWorthOutputting() and glyph_name in glyph_names:
            font[glyph].export(glyph_name + ".png")
