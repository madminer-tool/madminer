# This file was automatically created by FeynRules 2.3.24
# Mathematica version: 11.0.1 for Linux x86 (64-bit) (September 21, 2016)
# Date: Fri 11 Aug 2017 09:48:14



from object_library import all_parameters, Parameter


from function_library import complexconjugate, re, im, csc, sec, acsc, asec, cot

# This is a default parameter object representing 0.
ZERO = Parameter(name = 'ZERO',
                 nature = 'internal',
                 type = 'real',
                 value = '0.0',
                 texname = '0')

# User-defined parameters.
FB = Parameter(name = 'FB',
               nature = 'external',
               type = 'real',
               value = 0.,
               texname = 'f_B',
               lhablock = 'ANOINPUTS',
               lhacode = [ 1 ])

imFB = Parameter(name = 'imFB',
                 nature = 'external',
                 type = 'real',
                 value = 0.,
                 texname = 'f_{\\text{Bim}}',
                 lhablock = 'ANOINPUTS',
                 lhacode = [ 2 ])

FW = Parameter(name = 'FW',
               nature = 'external',
               type = 'real',
               value = 0.,
               texname = 'f_W',
               lhablock = 'ANOINPUTS',
               lhacode = [ 3 ])

imFW = Parameter(name = 'imFW',
                 nature = 'external',
                 type = 'real',
                 value = 0.,
                 texname = 'f_{\\text{Wim}}',
                 lhablock = 'ANOINPUTS',
                 lhacode = [ 4 ])

FBB = Parameter(name = 'FBB',
                nature = 'external',
                type = 'real',
                value = 0.,
                texname = 'f_{\\text{BB}}',
                lhablock = 'ANOINPUTS',
                lhacode = [ 5 ])

imFBB = Parameter(name = 'imFBB',
                  nature = 'external',
                  type = 'real',
                  value = 0.,
                  texname = 'f_{\\text{BBim}}',
                  lhablock = 'ANOINPUTS',
                  lhacode = [ 6 ])

FWW = Parameter(name = 'FWW',
                nature = 'external',
                type = 'real',
                value = 0.,
                texname = 'f_{\\text{WW}}',
                lhablock = 'ANOINPUTS',
                lhacode = [ 7 ])

imFWW = Parameter(name = 'imFWW',
                  nature = 'external',
                  type = 'real',
                  value = 0.,
                  texname = 'f_{\\text{WWim}}',
                  lhablock = 'ANOINPUTS',
                  lhacode = [ 8 ])

FBBtilde = Parameter(name = 'FBBtilde',
                     nature = 'external',
                     type = 'real',
                     value = 0.,
                     texname = 'f_{\\text{BBtilde}}',
                     lhablock = 'ANOINPUTS',
                     lhacode = [ 9 ])

imFBBtilde = Parameter(name = 'imFBBtilde',
                       nature = 'external',
                       type = 'real',
                       value = 0.,
                       texname = 'f_{\\text{BBtildeim}}',
                       lhablock = 'ANOINPUTS',
                       lhacode = [ 10 ])

FWWtilde = Parameter(name = 'FWWtilde',
                     nature = 'external',
                     type = 'real',
                     value = 0.,
                     texname = 'f_{\\text{WWtilde}}',
                     lhablock = 'ANOINPUTS',
                     lhacode = [ 11 ])

imFWWtilde = Parameter(name = 'imFWWtilde',
                       nature = 'external',
                       type = 'real',
                       value = 0.,
                       texname = 'f_{\\text{WWtildeim}}',
                       lhablock = 'ANOINPUTS',
                       lhacode = [ 12 ])

Ftphi = Parameter(name = 'Ftphi',
                  nature = 'external',
                  type = 'real',
                  value = 0.,
                  texname = 'f_{\\text{tphi}}',
                  lhablock = 'ANOINPUTS',
                  lhacode = [ 13 ])

Ftphitilde = Parameter(name = 'Ftphitilde',
                       nature = 'external',
                       type = 'real',
                       value = 0.,
                       texname = 'f_{\\text{tphitilde}}',
                       lhablock = 'ANOINPUTS',
                       lhacode = [ 14 ])

Fphitb1 = Parameter(name = 'Fphitb1',
                    nature = 'external',
                    type = 'real',
                    value = 0.,
                    texname = 'f_{\\text{phitb1}}',
                    lhablock = 'ANOINPUTS',
                    lhacode = [ 15 ])

FphiQ3 = Parameter(name = 'FphiQ3',
                   nature = 'external',
                   type = 'real',
                   value = 0.,
                   texname = 'f_{\\text{phiQ3}}',
                   lhablock = 'ANOINPUTS',
                   lhacode = [ 16 ])

FtW = Parameter(name = 'FtW',
                nature = 'external',
                type = 'real',
                value = 0.,
                texname = 'f_{\\text{tW}}',
                lhablock = 'ANOINPUTS',
                lhacode = [ 17 ])

FbW = Parameter(name = 'FbW',
                nature = 'external',
                type = 'real',
                value = 0.,
                texname = 'f_{\\text{bW}}',
                lhablock = 'ANOINPUTS',
                lhacode = [ 18 ])

aEWM1 = Parameter(name = 'aEWM1',
                  nature = 'external',
                  type = 'real',
                  value = 127.9,
                  texname = '\\text{aEWM1}',
                  lhablock = 'SMINPUTS',
                  lhacode = [ 1 ])

Gf = Parameter(name = 'Gf',
               nature = 'external',
               type = 'real',
               value = 0.0000116637,
               texname = 'G_f',
               lhablock = 'SMINPUTS',
               lhacode = [ 2 ])

aS = Parameter(name = 'aS',
               nature = 'external',
               type = 'real',
               value = 0.1184,
               texname = '\\alpha _s',
               lhablock = 'SMINPUTS',
               lhacode = [ 3 ])

ymb = Parameter(name = 'ymb',
                nature = 'external',
                type = 'real',
                value = 4.7,
                texname = '\\text{ymb}',
                lhablock = 'YUKAWA',
                lhacode = [ 5 ])

ymt = Parameter(name = 'ymt',
                nature = 'external',
                type = 'real',
                value = 172,
                texname = '\\text{ymt}',
                lhablock = 'YUKAWA',
                lhacode = [ 6 ])

ymtau = Parameter(name = 'ymtau',
                  nature = 'external',
                  type = 'real',
                  value = 1.777,
                  texname = '\\text{ymtau}',
                  lhablock = 'YUKAWA',
                  lhacode = [ 15 ])

MZ = Parameter(name = 'MZ',
               nature = 'external',
               type = 'real',
               value = 91.1876,
               texname = '\\text{MZ}',
               lhablock = 'MASS',
               lhacode = [ 23 ])

MMU = Parameter(name = 'MMU',
                nature = 'external',
                type = 'real',
                value = 0.10566,
                texname = '\\text{MMU}',
                lhablock = 'MASS',
                lhacode = [ 13 ])

MTA = Parameter(name = 'MTA',
                nature = 'external',
                type = 'real',
                value = 1.777,
                texname = '\\text{MTA}',
                lhablock = 'MASS',
                lhacode = [ 15 ])

MC = Parameter(name = 'MC',
               nature = 'external',
               type = 'real',
               value = 1.27,
               texname = '\\text{MC}',
               lhablock = 'MASS',
               lhacode = [ 4 ])

MT = Parameter(name = 'MT',
               nature = 'external',
               type = 'real',
               value = 172,
               texname = '\\text{MT}',
               lhablock = 'MASS',
               lhacode = [ 6 ])

MS = Parameter(name = 'MS',
               nature = 'external',
               type = 'real',
               value = 0.101,
               texname = '\\text{MS}',
               lhablock = 'MASS',
               lhacode = [ 3 ])

MB = Parameter(name = 'MB',
               nature = 'external',
               type = 'real',
               value = 4.7,
               texname = '\\text{MB}',
               lhablock = 'MASS',
               lhacode = [ 5 ])

MH = Parameter(name = 'MH',
               nature = 'external',
               type = 'real',
               value = 125,
               texname = '\\text{MH}',
               lhablock = 'MASS',
               lhacode = [ 25 ])

WZ = Parameter(name = 'WZ',
               nature = 'external',
               type = 'real',
               value = 2.4952,
               texname = '\\text{WZ}',
               lhablock = 'DECAY',
               lhacode = [ 23 ])

WW = Parameter(name = 'WW',
               nature = 'external',
               type = 'real',
               value = 2.085,
               texname = '\\text{WW}',
               lhablock = 'DECAY',
               lhacode = [ 24 ])

WT = Parameter(name = 'WT',
               nature = 'external',
               type = 'real',
               value = 1.50833649,
               texname = '\\text{WT}',
               lhablock = 'DECAY',
               lhacode = [ 6 ])

WH = Parameter(name = 'WH',
               nature = 'external',
               type = 'real',
               value = 0.00407,
               texname = '\\text{WH}',
               lhablock = 'DECAY',
               lhacode = [ 25 ])

aEW = Parameter(name = 'aEW',
                nature = 'internal',
                type = 'real',
                value = '1/aEWM1',
                texname = '\\alpha _{\\text{EW}}')

dum = Parameter(name = 'dum',
                nature = 'internal',
                type = 'real',
                value = '1',
                texname = '\\text{}')

G = Parameter(name = 'G',
              nature = 'internal',
              type = 'real',
              value = '2*cmath.sqrt(aS)*cmath.sqrt(cmath.pi)',
              texname = 'G')

MW = Parameter(name = 'MW',
               nature = 'internal',
               type = 'real',
               value = 'cmath.sqrt(MZ**2/2. + cmath.sqrt(MZ**4/4. - (aEW*cmath.pi*MZ**2)/(Gf*cmath.sqrt(2))))',
               texname = 'M_W')

ee = Parameter(name = 'ee',
               nature = 'internal',
               type = 'real',
               value = '2*cmath.sqrt(aEW)*cmath.sqrt(cmath.pi)',
               texname = 'e')

sw2 = Parameter(name = 'sw2',
                nature = 'internal',
                type = 'real',
                value = '1 - MW**2/MZ**2',
                texname = '\\text{sw2}')

cw = Parameter(name = 'cw',
               nature = 'internal',
               type = 'real',
               value = 'cmath.sqrt(1 - sw2)',
               texname = 'c_w')

sw = Parameter(name = 'sw',
               nature = 'internal',
               type = 'real',
               value = 'cmath.sqrt(sw2)',
               texname = 's_w')

g1 = Parameter(name = 'g1',
               nature = 'internal',
               type = 'real',
               value = 'ee/cw',
               texname = 'g_1')

gw = Parameter(name = 'gw',
               nature = 'internal',
               type = 'real',
               value = 'ee/sw',
               texname = 'g_w')

vev = Parameter(name = 'vev',
                nature = 'internal',
                type = 'real',
                value = '(2*MW*sw)/ee',
                texname = '\\text{vev}')

AH = Parameter(name = 'AH',
               nature = 'internal',
               type = 'real',
               value = '(47*ee**2*(1 - (2*MH**4)/(987.*MT**4) - (14*MH**2)/(705.*MT**2) + (213*MH**12)/(2.634632e7*MW**12) + (5*MH**10)/(119756.*MW**10) + (41*MH**8)/(180950.*MW**8) + (87*MH**6)/(65800.*MW**6) + (57*MH**4)/(6580.*MW**4) + (33*MH**2)/(470.*MW**2)))/(72.*cmath.pi**2*vev)',
               texname = 'A_H')

GH = Parameter(name = 'GH',
               nature = 'internal',
               type = 'real',
               value = '-(G**2*(1 + (13*MH**6)/(16800.*MT**6) + MH**4/(168.*MT**4) + (7*MH**2)/(120.*MT**2)))/(12.*cmath.pi**2*vev)',
               texname = 'G_H')

lam = Parameter(name = 'lam',
                nature = 'internal',
                type = 'real',
                value = 'MH**2/(2.*vev**2)',
                texname = '\\text{lam}')

yb = Parameter(name = 'yb',
               nature = 'internal',
               type = 'real',
               value = '(ymb*cmath.sqrt(2))/vev',
               texname = '\\text{yb}')

yt = Parameter(name = 'yt',
               nature = 'internal',
               type = 'complex',
               value = '((Ftphi + complex(0,1)*Ftphitilde)*vev**2)/2. + (ymt*cmath.sqrt(2))/vev',
               texname = '\\text{yt}')

ytau = Parameter(name = 'ytau',
                 nature = 'internal',
                 type = 'real',
                 value = '(ymtau*cmath.sqrt(2))/vev',
                 texname = '\\text{ytau}')

muH = Parameter(name = 'muH',
                nature = 'internal',
                type = 'real',
                value = 'cmath.sqrt(lam*vev**2)',
                texname = '\\mu')

