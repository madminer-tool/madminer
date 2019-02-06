# This file was automatically created by FeynRules 2.4.32
# Mathematica version: 10.0 for Linux x86 (64-bit) (September 9, 2014)
# Date: Thu 7 Jan 2016 02:14:12



from object_library import all_parameters, Parameter


from function_library import complexconjugate, re, im, csc, sec, acsc, asec, cot

# This is a default parameter object representing 0.
ZERO = Parameter(name = 'ZERO',
                 nature = 'internal',
                 type = 'real',
                 value = '0.0',
                 texname = '0')

# User-defined parameters.
cabi = Parameter(name = 'cabi',
                 nature = 'external',
                 type = 'real',
                 value = 0.227736,
                 texname = '\\theta _c',
                 lhablock = 'CKMBLOCK',
                 lhacode = [ 1 ])

cH = Parameter(name = 'cH',
               nature = 'external',
               type = 'real',
               value = 0.1,
               texname = 'C_H',
               lhablock = 'NEWCOUP',
               lhacode = [ 1 ])

cT = Parameter(name = 'cT',
               nature = 'external',
               type = 'real',
               value = 0.1,
               texname = 'C_T',
               lhablock = 'NEWCOUP',
               lhacode = [ 2 ])

c6 = Parameter(name = 'c6',
               nature = 'external',
               type = 'real',
               value = 0.1,
               texname = 'C_6',
               lhablock = 'NEWCOUP',
               lhacode = [ 3 ])

cu = Parameter(name = 'cu',
               nature = 'external',
               type = 'real',
               value = 0.1,
               texname = 'C_u',
               lhablock = 'NEWCOUP',
               lhacode = [ 4 ])

cd = Parameter(name = 'cd',
               nature = 'external',
               type = 'real',
               value = 0.1,
               texname = 'C_d',
               lhablock = 'NEWCOUP',
               lhacode = [ 5 ])

cl = Parameter(name = 'cl',
               nature = 'external',
               type = 'real',
               value = 0.1,
               texname = 'C_l',
               lhablock = 'NEWCOUP',
               lhacode = [ 6 ])

cWW = Parameter(name = 'cWW',
                nature = 'external',
                type = 'real',
                value = 0.1,
                texname = 'C_W',
                lhablock = 'NEWCOUP',
                lhacode = [ 7 ])

cB = Parameter(name = 'cB',
               nature = 'external',
               type = 'real',
               value = 0.1,
               texname = 'C_B',
               lhablock = 'NEWCOUP',
               lhacode = [ 8 ])

cHW = Parameter(name = 'cHW',
                nature = 'external',
                type = 'real',
                value = 0.1,
                texname = 'C_{\\text{HW}}',
                lhablock = 'NEWCOUP',
                lhacode = [ 9 ])

cHB = Parameter(name = 'cHB',
                nature = 'external',
                type = 'real',
                value = 0.1,
                texname = 'C_{\\text{HB}}',
                lhablock = 'NEWCOUP',
                lhacode = [ 10 ])

cA = Parameter(name = 'cA',
               nature = 'external',
               type = 'real',
               value = 0.1,
               texname = 'C_{\\gamma }',
               lhablock = 'NEWCOUP',
               lhacode = [ 11 ])

cG = Parameter(name = 'cG',
               nature = 'external',
               type = 'real',
               value = 0.1,
               texname = 'C_g',
               lhablock = 'NEWCOUP',
               lhacode = [ 12 ])

cHQ = Parameter(name = 'cHQ',
                nature = 'external',
                type = 'real',
                value = 0.1,
                texname = 'C_{\\text{Hq}}',
                lhablock = 'NEWCOUP',
                lhacode = [ 13 ])

cpHQ = Parameter(name = 'cpHQ',
                 nature = 'external',
                 type = 'real',
                 value = 0.1,
                 texname = 'C\'_{\\text{Hq}}',
                 lhablock = 'NEWCOUP',
                 lhacode = [ 14 ])

cHu = Parameter(name = 'cHu',
                nature = 'external',
                type = 'real',
                value = 0.1,
                texname = 'C_{\\text{Hu}}',
                lhablock = 'NEWCOUP',
                lhacode = [ 15 ])

cHd = Parameter(name = 'cHd',
                nature = 'external',
                type = 'real',
                value = 0.1,
                texname = 'C_{\\text{Hd}}',
                lhablock = 'NEWCOUP',
                lhacode = [ 16 ])

cHud = Parameter(name = 'cHud',
                 nature = 'external',
                 type = 'real',
                 value = 0.1,
                 texname = 'C_{\\text{Hud}}',
                 lhablock = 'NEWCOUP',
                 lhacode = [ 17 ])

cHL = Parameter(name = 'cHL',
                nature = 'external',
                type = 'real',
                value = 0.1,
                texname = 'C_{\\text{Hl}}',
                lhablock = 'NEWCOUP',
                lhacode = [ 18 ])

cpHL = Parameter(name = 'cpHL',
                 nature = 'external',
                 type = 'real',
                 value = 0.1,
                 texname = 'C\'_{\\text{Hl}}',
                 lhablock = 'NEWCOUP',
                 lhacode = [ 19 ])

cHe = Parameter(name = 'cHe',
                nature = 'external',
                type = 'real',
                value = 0.1,
                texname = 'C_{\\text{He}}',
                lhablock = 'NEWCOUP',
                lhacode = [ 20 ])

cuB = Parameter(name = 'cuB',
                nature = 'external',
                type = 'real',
                value = 0.1,
                texname = 'C_{\\text{uB}}',
                lhablock = 'NEWCOUP',
                lhacode = [ 21 ])

cuW = Parameter(name = 'cuW',
                nature = 'external',
                type = 'real',
                value = 0.1,
                texname = 'C_{\\text{uW}}',
                lhablock = 'NEWCOUP',
                lhacode = [ 22 ])

cuG = Parameter(name = 'cuG',
                nature = 'external',
                type = 'real',
                value = 0.1,
                texname = 'C_{\\text{uG}}',
                lhablock = 'NEWCOUP',
                lhacode = [ 23 ])

cdB = Parameter(name = 'cdB',
                nature = 'external',
                type = 'real',
                value = 0.1,
                texname = 'C_{\\text{dB}}',
                lhablock = 'NEWCOUP',
                lhacode = [ 24 ])

cdW = Parameter(name = 'cdW',
                nature = 'external',
                type = 'real',
                value = 0.1,
                texname = 'C_{\\text{dW}}',
                lhablock = 'NEWCOUP',
                lhacode = [ 25 ])

cdG = Parameter(name = 'cdG',
                nature = 'external',
                type = 'real',
                value = 0.1,
                texname = 'C_{\\text{dG}}',
                lhablock = 'NEWCOUP',
                lhacode = [ 26 ])

clB = Parameter(name = 'clB',
                nature = 'external',
                type = 'real',
                value = 0.1,
                texname = 'C_{\\text{lB}}',
                lhablock = 'NEWCOUP',
                lhacode = [ 27 ])

clW = Parameter(name = 'clW',
                nature = 'external',
                type = 'real',
                value = 0.1,
                texname = 'C_{\\text{lW}}',
                lhablock = 'NEWCOUP',
                lhacode = [ 28 ])

c3W = Parameter(name = 'c3W',
                nature = 'external',
                type = 'real',
                value = 0.1,
                texname = 'C_{3 W}',
                lhablock = 'NEWCOUP',
                lhacode = [ 29 ])

c3G = Parameter(name = 'c3G',
                nature = 'external',
                type = 'real',
                value = 0.1,
                texname = 'C_{3 G}',
                lhablock = 'NEWCOUP',
                lhacode = [ 30 ])

c2W = Parameter(name = 'c2W',
                nature = 'external',
                type = 'real',
                value = 0.1,
                texname = 'C_{2 W}',
                lhablock = 'NEWCOUP',
                lhacode = [ 31 ])

c2B = Parameter(name = 'c2B',
                nature = 'external',
                type = 'real',
                value = 0.1,
                texname = 'C_{2 B}',
                lhablock = 'NEWCOUP',
                lhacode = [ 32 ])

c2G = Parameter(name = 'c2G',
                nature = 'external',
                type = 'real',
                value = 0.1,
                texname = 'C_{2 G}',
                lhablock = 'NEWCOUP',
                lhacode = [ 33 ])

tcHW = Parameter(name = 'tcHW',
                 nature = 'external',
                 type = 'real',
                 value = 0.1,
                 texname = '\\tilde{C}_{\\text{HW}}',
                 lhablock = 'NEWCOUP',
                 lhacode = [ 34 ])

tcHB = Parameter(name = 'tcHB',
                 nature = 'external',
                 type = 'real',
                 value = 0.1,
                 texname = '\\tilde{C}_{\\text{HB}}',
                 lhablock = 'NEWCOUP',
                 lhacode = [ 35 ])

tcG = Parameter(name = 'tcG',
                nature = 'external',
                type = 'real',
                value = 0.1,
                texname = '\\tilde{C}_G',
                lhablock = 'NEWCOUP',
                lhacode = [ 36 ])

tcA = Parameter(name = 'tcA',
                nature = 'external',
                type = 'real',
                value = 0.1,
                texname = '\\tilde{C}_A',
                lhablock = 'NEWCOUP',
                lhacode = [ 37 ])

tc3W = Parameter(name = 'tc3W',
                 nature = 'external',
                 type = 'real',
                 value = 0.1,
                 texname = '\\tilde{C}_{3 W}',
                 lhablock = 'NEWCOUP',
                 lhacode = [ 38 ])

tc3G = Parameter(name = 'tc3G',
                 nature = 'external',
                 type = 'real',
                 value = 0.1,
                 texname = '\\tilde{C}_{3 G}',
                 lhablock = 'NEWCOUP',
                 lhacode = [ 39 ])

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

ymdo = Parameter(name = 'ymdo',
                 nature = 'external',
                 type = 'real',
                 value = 0.00504,
                 texname = '\\text{ymdo}',
                 lhablock = 'YUKAWA',
                 lhacode = [ 1 ])

ymup = Parameter(name = 'ymup',
                 nature = 'external',
                 type = 'real',
                 value = 0.00255,
                 texname = '\\text{ymup}',
                 lhablock = 'YUKAWA',
                 lhacode = [ 2 ])

yms = Parameter(name = 'yms',
                nature = 'external',
                type = 'real',
                value = 0.101,
                texname = '\\text{yms}',
                lhablock = 'YUKAWA',
                lhacode = [ 3 ])

ymc = Parameter(name = 'ymc',
                nature = 'external',
                type = 'real',
                value = 1.27,
                texname = '\\text{ymc}',
                lhablock = 'YUKAWA',
                lhacode = [ 4 ])

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

yme = Parameter(name = 'yme',
                nature = 'external',
                type = 'real',
                value = 0.000511,
                texname = '\\text{yme}',
                lhablock = 'YUKAWA',
                lhacode = [ 11 ])

ymm = Parameter(name = 'ymm',
                nature = 'external',
                type = 'real',
                value = 0.10566,
                texname = '\\text{ymm}',
                lhablock = 'YUKAWA',
                lhacode = [ 13 ])

ymtau = Parameter(name = 'ymtau',
                  nature = 'external',
                  type = 'real',
                  value = 1.777,
                  texname = '\\text{ymtau}',
                  lhablock = 'YUKAWA',
                  lhacode = [ 15 ])

MW = Parameter(name = 'MW',
               nature = 'external',
               type = 'real',
               value = 80.385,
               texname = '\\text{MW}',
               lhablock = 'MASS',
               lhacode = [ 24 ])

Me = Parameter(name = 'Me',
               nature = 'external',
               type = 'real',
               value = 0.000511,
               texname = '\\text{Me}',
               lhablock = 'MASS',
               lhacode = [ 11 ])

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

MU = Parameter(name = 'MU',
               nature = 'external',
               type = 'real',
               value = 0.00255,
               texname = 'M',
               lhablock = 'MASS',
               lhacode = [ 2 ])

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

MD = Parameter(name = 'MD',
               nature = 'external',
               type = 'real',
               value = 0.00504,
               texname = '\\text{MD}',
               lhablock = 'MASS',
               lhacode = [ 1 ])

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

dum = Parameter(name = 'dum',
                nature = 'internal',
                type = 'real',
                value = '1',
                texname = '\\text{}')

aEW = Parameter(name = 'aEW',
                nature = 'internal',
                type = 'real',
                value = '1/aEWM1',
                texname = '\\alpha _{\\text{EW}}')

vev = Parameter(name = 'vev',
                nature = 'internal',
                type = 'real',
                value = 'cmath.sqrt(1/Gf)/2**0.25',
                texname = '\\text{vev}')

G = Parameter(name = 'G',
              nature = 'internal',
              type = 'real',
              value = '2*cmath.sqrt(aS)*cmath.sqrt(cmath.pi)',
              texname = 'G')

CKM1x1 = Parameter(name = 'CKM1x1',
                   nature = 'internal',
                   type = 'complex',
                   value = 'cmath.cos(cabi)',
                   texname = '\\text{CKM1x1}')

CKM1x2 = Parameter(name = 'CKM1x2',
                   nature = 'internal',
                   type = 'complex',
                   value = 'cmath.sin(cabi)',
                   texname = '\\text{CKM1x2}')

CKM1x3 = Parameter(name = 'CKM1x3',
                   nature = 'internal',
                   type = 'complex',
                   value = '0',
                   texname = '\\text{CKM1x3}')

CKM2x1 = Parameter(name = 'CKM2x1',
                   nature = 'internal',
                   type = 'complex',
                   value = '-cmath.sin(cabi)',
                   texname = '\\text{CKM2x1}')

CKM2x2 = Parameter(name = 'CKM2x2',
                   nature = 'internal',
                   type = 'complex',
                   value = 'cmath.cos(cabi)',
                   texname = '\\text{CKM2x2}')

CKM2x3 = Parameter(name = 'CKM2x3',
                   nature = 'internal',
                   type = 'complex',
                   value = '0',
                   texname = '\\text{CKM2x3}')

CKM3x1 = Parameter(name = 'CKM3x1',
                   nature = 'internal',
                   type = 'complex',
                   value = '0',
                   texname = '\\text{CKM3x1}')

CKM3x2 = Parameter(name = 'CKM3x2',
                   nature = 'internal',
                   type = 'complex',
                   value = '0',
                   texname = '\\text{CKM3x2}')

CKM3x3 = Parameter(name = 'CKM3x3',
                   nature = 'internal',
                   type = 'complex',
                   value = '1',
                   texname = '\\text{CKM3x3}')

ee = Parameter(name = 'ee',
               nature = 'internal',
               type = 'real',
               value = '2*cmath.sqrt(aEW)*cmath.sqrt(cmath.pi)',
               texname = 'e')

GH = Parameter(name = 'GH',
               nature = 'internal',
               type = 'real',
               value = '-G**2/(12.*cmath.pi**2*vev)',
               texname = 'G_H')

lam = Parameter(name = 'lam',
                nature = 'internal',
                type = 'real',
                value = '((1 - (13*c6)/8. + cH)*MH**2)/(2.*vev**2)',
                texname = '\\text{lam}')

yb = Parameter(name = 'yb',
               nature = 'internal',
               type = 'real',
               value = '(ymb*cmath.sqrt(2))/vev',
               texname = '\\text{yb}')

yc = Parameter(name = 'yc',
               nature = 'internal',
               type = 'real',
               value = '(ymc*cmath.sqrt(2))/vev',
               texname = '\\text{yc}')

ydo = Parameter(name = 'ydo',
                nature = 'internal',
                type = 'real',
                value = '(ymdo*cmath.sqrt(2))/vev',
                texname = '\\text{ydo}')

ye = Parameter(name = 'ye',
               nature = 'internal',
               type = 'real',
               value = '(yme*cmath.sqrt(2))/vev',
               texname = '\\text{ye}')

ym = Parameter(name = 'ym',
               nature = 'internal',
               type = 'real',
               value = '(ymm*cmath.sqrt(2))/vev',
               texname = '\\text{ym}')

ys = Parameter(name = 'ys',
               nature = 'internal',
               type = 'real',
               value = '(yms*cmath.sqrt(2))/vev',
               texname = '\\text{ys}')

yt = Parameter(name = 'yt',
               nature = 'internal',
               type = 'real',
               value = '(ymt*cmath.sqrt(2))/vev',
               texname = '\\text{yt}')

ytau = Parameter(name = 'ytau',
                 nature = 'internal',
                 type = 'real',
                 value = '(ymtau*cmath.sqrt(2))/vev',
                 texname = '\\text{ytau}')

yup = Parameter(name = 'yup',
                nature = 'internal',
                type = 'real',
                value = '(ymup*cmath.sqrt(2))/vev',
                texname = '\\text{yup}')

muH = Parameter(name = 'muH',
                nature = 'internal',
                type = 'real',
                value = '(1 + c6/4.)*cmath.sqrt(lam*vev**2)',
                texname = '\\mu')

sw = Parameter(name = 'sw',
               nature = 'internal',
               type = 'real',
               value = '(ee*vev)/(2.*MW)',
               texname = 's_w')

AH = Parameter(name = 'AH',
               nature = 'internal',
               type = 'real',
               value = '(47*ee**2*(1 - (2*MH**4)/(987.*MT**4) - (14*MH**2)/(705.*MT**2) + (213*MH**12)/(2.634632e7*MW**12) + (5*MH**10)/(119756.*MW**10) + (41*MH**8)/(180950.*MW**8) + (87*MH**6)/(65800.*MW**6) + (57*MH**4)/(6580.*MW**4) + (33*MH**2)/(470.*MW**2)))/(72.*cmath.pi**2*vev)',
               texname = 'A_H')

cw = Parameter(name = 'cw',
               nature = 'internal',
               type = 'real',
               value = 'cmath.sqrt(1 - sw**2)',
               texname = 'c_w')

gw = Parameter(name = 'gw',
               nature = 'internal',
               type = 'real',
               value = 'ee/sw',
               texname = 'g_w')

MZ = Parameter(name = 'MZ',
               nature = 'internal',
               type = 'real',
               value = 'cmath.sqrt((gw**2*vev**2*(1 - cT + (ee**2*(cw**2*cWW + cB*sw**2 + 4*cA*sw**4)*vev**2)/(2.*cw**2*MW**2*sw**2)))/cw**2)/2.',
               texname = 'M_Z')

g1 = Parameter(name = 'g1',
               nature = 'internal',
               type = 'real',
               value = 'ee/cw',
               texname = 'g_1')

I1a11 = Parameter(name = 'I1a11',
                  nature = 'internal',
                  type = 'complex',
                  value = 'ydo*complexconjugate(CKM1x1)',
                  texname = '\\text{I1a11}')

I1a12 = Parameter(name = 'I1a12',
                  nature = 'internal',
                  type = 'complex',
                  value = 'ydo*complexconjugate(CKM2x1)',
                  texname = '\\text{I1a12}')

I1a13 = Parameter(name = 'I1a13',
                  nature = 'internal',
                  type = 'complex',
                  value = 'ydo*complexconjugate(CKM3x1)',
                  texname = '\\text{I1a13}')

I1a21 = Parameter(name = 'I1a21',
                  nature = 'internal',
                  type = 'complex',
                  value = 'ys*complexconjugate(CKM1x2)',
                  texname = '\\text{I1a21}')

I1a22 = Parameter(name = 'I1a22',
                  nature = 'internal',
                  type = 'complex',
                  value = 'ys*complexconjugate(CKM2x2)',
                  texname = '\\text{I1a22}')

I1a23 = Parameter(name = 'I1a23',
                  nature = 'internal',
                  type = 'complex',
                  value = 'ys*complexconjugate(CKM3x2)',
                  texname = '\\text{I1a23}')

I1a31 = Parameter(name = 'I1a31',
                  nature = 'internal',
                  type = 'complex',
                  value = 'yb*complexconjugate(CKM1x3)',
                  texname = '\\text{I1a31}')

I1a32 = Parameter(name = 'I1a32',
                  nature = 'internal',
                  type = 'complex',
                  value = 'yb*complexconjugate(CKM2x3)',
                  texname = '\\text{I1a32}')

I1a33 = Parameter(name = 'I1a33',
                  nature = 'internal',
                  type = 'complex',
                  value = 'yb*complexconjugate(CKM3x3)',
                  texname = '\\text{I1a33}')

I2a11 = Parameter(name = 'I2a11',
                  nature = 'internal',
                  type = 'complex',
                  value = 'yup*complexconjugate(CKM1x1)',
                  texname = '\\text{I2a11}')

I2a12 = Parameter(name = 'I2a12',
                  nature = 'internal',
                  type = 'complex',
                  value = 'yc*complexconjugate(CKM2x1)',
                  texname = '\\text{I2a12}')

I2a13 = Parameter(name = 'I2a13',
                  nature = 'internal',
                  type = 'complex',
                  value = 'yt*complexconjugate(CKM3x1)',
                  texname = '\\text{I2a13}')

I2a21 = Parameter(name = 'I2a21',
                  nature = 'internal',
                  type = 'complex',
                  value = 'yup*complexconjugate(CKM1x2)',
                  texname = '\\text{I2a21}')

I2a22 = Parameter(name = 'I2a22',
                  nature = 'internal',
                  type = 'complex',
                  value = 'yc*complexconjugate(CKM2x2)',
                  texname = '\\text{I2a22}')

I2a23 = Parameter(name = 'I2a23',
                  nature = 'internal',
                  type = 'complex',
                  value = 'yt*complexconjugate(CKM3x2)',
                  texname = '\\text{I2a23}')

I2a31 = Parameter(name = 'I2a31',
                  nature = 'internal',
                  type = 'complex',
                  value = 'yup*complexconjugate(CKM1x3)',
                  texname = '\\text{I2a31}')

I2a32 = Parameter(name = 'I2a32',
                  nature = 'internal',
                  type = 'complex',
                  value = 'yc*complexconjugate(CKM2x3)',
                  texname = '\\text{I2a32}')

I2a33 = Parameter(name = 'I2a33',
                  nature = 'internal',
                  type = 'complex',
                  value = 'yt*complexconjugate(CKM3x3)',
                  texname = '\\text{I2a33}')

I3a11 = Parameter(name = 'I3a11',
                  nature = 'internal',
                  type = 'complex',
                  value = 'CKM1x1*yup',
                  texname = '\\text{I3a11}')

I3a12 = Parameter(name = 'I3a12',
                  nature = 'internal',
                  type = 'complex',
                  value = 'CKM1x2*yup',
                  texname = '\\text{I3a12}')

I3a13 = Parameter(name = 'I3a13',
                  nature = 'internal',
                  type = 'complex',
                  value = 'CKM1x3*yup',
                  texname = '\\text{I3a13}')

I3a21 = Parameter(name = 'I3a21',
                  nature = 'internal',
                  type = 'complex',
                  value = 'CKM2x1*yc',
                  texname = '\\text{I3a21}')

I3a22 = Parameter(name = 'I3a22',
                  nature = 'internal',
                  type = 'complex',
                  value = 'CKM2x2*yc',
                  texname = '\\text{I3a22}')

I3a23 = Parameter(name = 'I3a23',
                  nature = 'internal',
                  type = 'complex',
                  value = 'CKM2x3*yc',
                  texname = '\\text{I3a23}')

I3a31 = Parameter(name = 'I3a31',
                  nature = 'internal',
                  type = 'complex',
                  value = 'CKM3x1*yt',
                  texname = '\\text{I3a31}')

I3a32 = Parameter(name = 'I3a32',
                  nature = 'internal',
                  type = 'complex',
                  value = 'CKM3x2*yt',
                  texname = '\\text{I3a32}')

I3a33 = Parameter(name = 'I3a33',
                  nature = 'internal',
                  type = 'complex',
                  value = 'CKM3x3*yt',
                  texname = '\\text{I3a33}')

I4a11 = Parameter(name = 'I4a11',
                  nature = 'internal',
                  type = 'complex',
                  value = 'CKM1x1*ydo',
                  texname = '\\text{I4a11}')

I4a12 = Parameter(name = 'I4a12',
                  nature = 'internal',
                  type = 'complex',
                  value = 'CKM1x2*ys',
                  texname = '\\text{I4a12}')

I4a13 = Parameter(name = 'I4a13',
                  nature = 'internal',
                  type = 'complex',
                  value = 'CKM1x3*yb',
                  texname = '\\text{I4a13}')

I4a21 = Parameter(name = 'I4a21',
                  nature = 'internal',
                  type = 'complex',
                  value = 'CKM2x1*ydo',
                  texname = '\\text{I4a21}')

I4a22 = Parameter(name = 'I4a22',
                  nature = 'internal',
                  type = 'complex',
                  value = 'CKM2x2*ys',
                  texname = '\\text{I4a22}')

I4a23 = Parameter(name = 'I4a23',
                  nature = 'internal',
                  type = 'complex',
                  value = 'CKM2x3*yb',
                  texname = '\\text{I4a23}')

I4a31 = Parameter(name = 'I4a31',
                  nature = 'internal',
                  type = 'complex',
                  value = 'CKM3x1*ydo',
                  texname = '\\text{I4a31}')

I4a32 = Parameter(name = 'I4a32',
                  nature = 'internal',
                  type = 'complex',
                  value = 'CKM3x2*ys',
                  texname = '\\text{I4a32}')

I4a33 = Parameter(name = 'I4a33',
                  nature = 'internal',
                  type = 'complex',
                  value = 'CKM3x3*yb',
                  texname = '\\text{I4a33}')

