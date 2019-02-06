# This file was automatically created by FeynRules 2.4.32
# Mathematica version: 10.0 for Linux x86 (64-bit) (September 9, 2014)
# Date: Thu 7 Jan 2016 02:14:13


from object_library import all_lorentz, Lorentz

from function_library import complexconjugate, re, im, csc, sec, acsc, asec, cot
try:
   import form_factors as ForFac 
except ImportError:
   pass


UUS1 = Lorentz(name = 'UUS1',
               spins = [ -1, -1, 1 ],
               structure = '1')

UUV1 = Lorentz(name = 'UUV1',
               spins = [ -1, -1, 3 ],
               structure = 'P(3,2) + P(3,3)')

SSS1 = Lorentz(name = 'SSS1',
               spins = [ 1, 1, 1 ],
               structure = '1')

SSS2 = Lorentz(name = 'SSS2',
               spins = [ 1, 1, 1 ],
               structure = 'P(-1,1)*P(-1,2) - P(-1,1)*P(-1,3)')

SSS3 = Lorentz(name = 'SSS3',
               spins = [ 1, 1, 1 ],
               structure = 'P(-1,1)*P(-1,2) - (P(-1,1)*P(-1,3))/2. - (P(-1,2)*P(-1,3))/2.')

SSS4 = Lorentz(name = 'SSS4',
               spins = [ 1, 1, 1 ],
               structure = 'P(-1,1)*P(-1,3) + P(-1,2)*P(-1,3)')

SSS5 = Lorentz(name = 'SSS5',
               spins = [ 1, 1, 1 ],
               structure = 'P(-1,1)*P(-1,2) + P(-1,1)*P(-1,3) + P(-1,2)*P(-1,3)')

FFS1 = Lorentz(name = 'FFS1',
               spins = [ 2, 2, 1 ],
               structure = 'ProjM(2,1)')

FFS2 = Lorentz(name = 'FFS2',
               spins = [ 2, 2, 1 ],
               structure = 'P(-1,3)*Gamma(-1,2,-2)*ProjM(-2,1)')

FFS3 = Lorentz(name = 'FFS3',
               spins = [ 2, 2, 1 ],
               structure = 'ProjM(2,1) - ProjP(2,1)')

FFS4 = Lorentz(name = 'FFS4',
               spins = [ 2, 2, 1 ],
               structure = 'ProjP(2,1)')

FFS5 = Lorentz(name = 'FFS5',
               spins = [ 2, 2, 1 ],
               structure = 'ProjM(2,1) + ProjP(2,1)')

FFS6 = Lorentz(name = 'FFS6',
               spins = [ 2, 2, 1 ],
               structure = 'P(-1,3)*Gamma(-1,2,-2)*ProjP(-2,1)')

FFV1 = Lorentz(name = 'FFV1',
               spins = [ 2, 2, 3 ],
               structure = 'Gamma(3,2,-1)*ProjM(-1,1)')

FFV2 = Lorentz(name = 'FFV2',
               spins = [ 2, 2, 3 ],
               structure = 'P(-1,3)*Gamma(-1,2,-3)*Gamma(3,-3,-2)*ProjM(-2,1)')

FFV3 = Lorentz(name = 'FFV3',
               spins = [ 2, 2, 3 ],
               structure = 'P(-1,3)*Gamma(-1,-3,-2)*Gamma(3,2,-3)*ProjM(-2,1)')

FFV4 = Lorentz(name = 'FFV4',
               spins = [ 2, 2, 3 ],
               structure = '-(P(-1,3)*Gamma(-1,2,-3)*Gamma(3,-3,-2)*ProjM(-2,1)) + P(-1,3)*Gamma(-1,-3,-2)*Gamma(3,2,-3)*ProjM(-2,1)')

FFV5 = Lorentz(name = 'FFV5',
               spins = [ 2, 2, 3 ],
               structure = 'Gamma(3,2,-1)*ProjP(-1,1)')

FFV6 = Lorentz(name = 'FFV6',
               spins = [ 2, 2, 3 ],
               structure = 'Gamma(3,2,-1)*ProjM(-1,1) + Gamma(3,2,-1)*ProjP(-1,1)')

FFV7 = Lorentz(name = 'FFV7',
               spins = [ 2, 2, 3 ],
               structure = 'P(-1,3)*Gamma(-1,2,-3)*Gamma(3,-3,-2)*ProjP(-2,1)')

FFV8 = Lorentz(name = 'FFV8',
               spins = [ 2, 2, 3 ],
               structure = 'P(-1,3)*Gamma(-1,-3,-2)*Gamma(3,2,-3)*ProjP(-2,1)')

FFV9 = Lorentz(name = 'FFV9',
               spins = [ 2, 2, 3 ],
               structure = '-(P(-1,3)*Gamma(-1,2,-3)*Gamma(3,-3,-2)*ProjP(-2,1)) + P(-1,3)*Gamma(-1,-3,-2)*Gamma(3,2,-3)*ProjP(-2,1)')

FFV10 = Lorentz(name = 'FFV10',
                spins = [ 2, 2, 3 ],
                structure = '-(P(-1,3)*Gamma(-1,2,-3)*Gamma(3,-3,-2)*ProjM(-2,1)) + P(-1,3)*Gamma(-1,-3,-2)*Gamma(3,2,-3)*ProjM(-2,1) - P(-1,3)*Gamma(-1,2,-3)*Gamma(3,-3,-2)*ProjP(-2,1) + P(-1,3)*Gamma(-1,-3,-2)*Gamma(3,2,-3)*ProjP(-2,1)')

FFV11 = Lorentz(name = 'FFV11',
                spins = [ 2, 2, 3 ],
                structure = 'P(-1,3)*Gamma(-1,-2,-3)*Gamma(3,2,-2)*ProjM(-3,1) - P(-1,3)*Gamma(-1,2,-3)*Gamma(3,-3,-2)*ProjM(-2,1) - P(-1,3)*Gamma(-1,2,-3)*Gamma(3,-3,-2)*ProjP(-2,1) + P(-1,3)*Gamma(-1,-3,-2)*Gamma(3,2,-3)*ProjP(-2,1)')

FFV12 = Lorentz(name = 'FFV12',
                spins = [ 2, 2, 3 ],
                structure = 'P(-1,3)*Gamma(-1,-2,-3)*Gamma(3,2,-2)*ProjP(-3,1) - P(-1,3)*Gamma(-1,2,-3)*Gamma(3,-3,-2)*ProjP(-2,1)')

VSS1 = Lorentz(name = 'VSS1',
               spins = [ 3, 1, 1 ],
               structure = 'P(1,2)')

VSS2 = Lorentz(name = 'VSS2',
               spins = [ 3, 1, 1 ],
               structure = 'P(1,2) - P(1,3)')

VSS3 = Lorentz(name = 'VSS3',
               spins = [ 3, 1, 1 ],
               structure = 'P(1,2) - P(1,3)/3.')

VSS4 = Lorentz(name = 'VSS4',
               spins = [ 3, 1, 1 ],
               structure = '-(P(-1,1)*P(-1,2)*P(1,1)) + P(-1,1)*P(-1,3)*P(1,1) + P(-1,1)**2*P(1,2) - P(-1,1)**2*P(1,3)')

VSS5 = Lorentz(name = 'VSS5',
               spins = [ 3, 1, 1 ],
               structure = '-(P(-1,1)*P(-1,3)*P(1,2)) + P(-1,1)*P(-1,2)*P(1,3)')

VSS6 = Lorentz(name = 'VSS6',
               spins = [ 3, 1, 1 ],
               structure = '-(Epsilon(1,-1,-2,-3)*P(-3,1)*P(-2,3)*P(-1,2))')

VVS1 = Lorentz(name = 'VVS1',
               spins = [ 3, 3, 1 ],
               structure = 'Epsilon(1,2,-1,-2)*P(-2,2)*P(-1,1)')

VVS2 = Lorentz(name = 'VVS2',
               spins = [ 3, 3, 1 ],
               structure = '-(Epsilon(1,2,-1,-2)*P(-2,2)*P(-1,1)) + Epsilon(1,2,-1,-2)*P(-2,1)*P(-1,2)')

VVS3 = Lorentz(name = 'VVS3',
               spins = [ 3, 3, 1 ],
               structure = 'Epsilon(1,2,-1,-2)*P(-2,1)*P(-1,3)')

VVS4 = Lorentz(name = 'VVS4',
               spins = [ 3, 3, 1 ],
               structure = 'Epsilon(1,2,-1,-2)*P(-2,2)*P(-1,3)')

VVS5 = Lorentz(name = 'VVS5',
               spins = [ 3, 3, 1 ],
               structure = 'Epsilon(1,2,-1,-2)*P(-2,1)*P(-1,3) - Epsilon(1,2,-1,-2)*P(-2,2)*P(-1,3)')

VVS6 = Lorentz(name = 'VVS6',
               spins = [ 3, 3, 1 ],
               structure = 'Epsilon(1,2,-1,-2)*P(-2,1)*P(-1,3) + Epsilon(1,2,-1,-2)*P(-2,2)*P(-1,3)')

VVS7 = Lorentz(name = 'VVS7',
               spins = [ 3, 3, 1 ],
               structure = 'Metric(1,2)')

VVS8 = Lorentz(name = 'VVS8',
               spins = [ 3, 3, 1 ],
               structure = 'P(1,1)*P(2,1) - P(-1,1)**2*Metric(1,2)')

VVS9 = Lorentz(name = 'VVS9',
               spins = [ 3, 3, 1 ],
               structure = 'P(1,2)*P(2,1) - P(-1,1)*P(-1,2)*Metric(1,2)')

VVS10 = Lorentz(name = 'VVS10',
                spins = [ 3, 3, 1 ],
                structure = 'P(1,2)*P(2,2) - P(-1,2)**2*Metric(1,2)')

VVS11 = Lorentz(name = 'VVS11',
                spins = [ 3, 3, 1 ],
                structure = 'P(1,1)*P(2,1) + P(1,2)*P(2,2) - P(-1,1)**2*Metric(1,2) - P(-1,2)**2*Metric(1,2)')

VVS12 = Lorentz(name = 'VVS12',
                spins = [ 3, 3, 1 ],
                structure = 'P(1,3)*P(2,1) - P(-1,1)*P(-1,3)*Metric(1,2)')

VVS13 = Lorentz(name = 'VVS13',
                spins = [ 3, 3, 1 ],
                structure = 'P(1,2)*P(2,3) - P(-1,2)*P(-1,3)*Metric(1,2)')

VVS14 = Lorentz(name = 'VVS14',
                spins = [ 3, 3, 1 ],
                structure = 'P(1,3)*P(2,1) + P(1,2)*P(2,3) - P(-1,1)*P(-1,3)*Metric(1,2) - P(-1,2)*P(-1,3)*Metric(1,2)')

VVS15 = Lorentz(name = 'VVS15',
                spins = [ 3, 3, 1 ],
                structure = '-2*P(1,3)*P(2,1) + P(1,2)*P(2,2) - P(1,3)*P(2,2) + P(1,1)*P(2,3) + 2*P(1,2)*P(2,3) - P(-1,2)**2*Metric(1,2) + P(-1,1)*P(-1,3)*Metric(1,2) - P(-1,2)*P(-1,3)*Metric(1,2)')

VVS16 = Lorentz(name = 'VVS16',
                spins = [ 3, 3, 1 ],
                structure = 'P(1,3)*P(2,1) + (P(1,3)*P(2,2))/2. - (P(1,1)*P(2,3))/2. - P(1,2)*P(2,3) - (P(-1,1)*P(-1,3)*Metric(1,2))/2. + (P(-1,2)*P(-1,3)*Metric(1,2))/2.')

VVS17 = Lorentz(name = 'VVS17',
                spins = [ 3, 3, 1 ],
                structure = 'P(1,3)*P(2,1) - P(1,2)*P(2,3) - P(-1,1)*P(-1,3)*Metric(1,2) + P(-1,2)*P(-1,3)*Metric(1,2)')

VVV1 = Lorentz(name = 'VVV1',
               spins = [ 3, 3, 3 ],
               structure = '-(Epsilon(1,2,3,-1)*P(-1,1))')

VVV2 = Lorentz(name = 'VVV2',
               spins = [ 3, 3, 3 ],
               structure = '-(Epsilon(1,2,3,-1)*P(-1,1)) - Epsilon(1,2,3,-1)*P(-1,2)')

VVV3 = Lorentz(name = 'VVV3',
               spins = [ 3, 3, 3 ],
               structure = '-(Epsilon(1,2,3,-1)*P(-1,3))')

VVV4 = Lorentz(name = 'VVV4',
               spins = [ 3, 3, 3 ],
               structure = '-(Epsilon(1,2,3,-1)*P(-1,1)) - Epsilon(1,2,3,-1)*P(-1,2) - Epsilon(1,2,3,-1)*P(-1,3)')

VVV5 = Lorentz(name = 'VVV5',
               spins = [ 3, 3, 3 ],
               structure = 'P(3,1)*Metric(1,2) - P(2,1)*Metric(1,3)')

VVV6 = Lorentz(name = 'VVV6',
               spins = [ 3, 3, 3 ],
               structure = 'P(3,1)*Metric(1,2) + P(3,2)*Metric(1,2) + P(3,3)*Metric(1,2) - P(2,1)*Metric(1,3) - P(2,2)*Metric(1,3) - P(2,3)*Metric(1,3)')

VVV7 = Lorentz(name = 'VVV7',
               spins = [ 3, 3, 3 ],
               structure = 'P(3,1)*Metric(1,2) - P(3,2)*Metric(1,2) - P(2,1)*Metric(1,3) + P(1,2)*Metric(2,3)')

VVV8 = Lorentz(name = 'VVV8',
               spins = [ 3, 3, 3 ],
               structure = 'P(2,3)*Metric(1,3) - P(1,3)*Metric(2,3)')

VVV9 = Lorentz(name = 'VVV9',
               spins = [ 3, 3, 3 ],
               structure = 'P(2,1)*Metric(1,3) + P(2,2)*Metric(1,3) + P(2,3)*Metric(1,3) - P(1,1)*Metric(2,3) - P(1,2)*Metric(2,3) - P(1,3)*Metric(2,3)')

VVV10 = Lorentz(name = 'VVV10',
                spins = [ 3, 3, 3 ],
                structure = 'P(3,1)*Metric(1,2) - P(3,2)*Metric(1,2) - P(2,1)*Metric(1,3) + P(2,3)*Metric(1,3) + P(1,2)*Metric(2,3) - P(1,3)*Metric(2,3)')

VVV11 = Lorentz(name = 'VVV11',
                spins = [ 3, 3, 3 ],
                structure = '-(P(1,2)*P(2,3)*P(3,1)) + P(1,3)*P(2,1)*P(3,2) + P(-1,2)*P(-1,3)*P(3,1)*Metric(1,2) - P(-1,1)*P(-1,3)*P(3,2)*Metric(1,2) - P(-1,2)*P(-1,3)*P(2,1)*Metric(1,3) + P(-1,1)*P(-1,2)*P(2,3)*Metric(1,3) + P(-1,1)*P(-1,3)*P(1,2)*Metric(2,3) - P(-1,1)*P(-1,2)*P(1,3)*Metric(2,3)')

VVV12 = Lorentz(name = 'VVV12',
                spins = [ 3, 3, 3 ],
                structure = 'P(1,1)*P(2,2)*P(3,1) + 2*P(1,2)*P(2,2)*P(3,1) + 2*P(1,1)*P(2,3)*P(3,1) - 2*P(1,1)*P(2,1)*P(3,2) - P(1,1)*P(2,2)*P(3,2) - 2*P(1,3)*P(2,2)*P(3,2) - P(1,1)*P(2,1)*P(3,3) - 2*P(1,3)*P(2,1)*P(3,3) + P(1,2)*P(2,2)*P(3,3) - P(1,3)*P(2,2)*P(3,3) + P(1,1)*P(2,3)*P(3,3) + 2*P(1,2)*P(2,3)*P(3,3) - 2*P(-1,2)**2*P(3,1)*Metric(1,2) - P(-1,3)**2*P(3,1)*Metric(1,2) + 2*P(-1,1)**2*P(3,2)*Metric(1,2) + P(-1,3)**2*P(3,2)*Metric(1,2) + P(-1,1)**2*P(3,3)*Metric(1,2) - P(-1,2)**2*P(3,3)*Metric(1,2) + P(-1,1)*P(-1,3)*P(3,3)*Metric(1,2) - P(-1,2)*P(-1,3)*P(3,3)*Metric(1,2) + P(-1,2)**2*P(2,1)*Metric(1,3) + 2*P(-1,3)**2*P(2,1)*Metric(1,3) - P(-1,1)**2*P(2,2)*Metric(1,3) - P(-1,1)*P(-1,2)*P(2,2)*Metric(1,3) + P(-1,2)*P(-1,3)*P(2,2)*Metric(1,3) + P(-1,3)**2*P(2,2)*Metric(1,3) - 2*P(-1,1)**2*P(2,3)*Metric(1,3) - P(-1,2)**2*P(2,3)*Metric(1,3) + P(-1,1)*P(-1,2)*P(1,1)*Metric(2,3) + P(-1,2)**2*P(1,1)*Metric(2,3) - P(-1,1)*P(-1,3)*P(1,1)*Metric(2,3) - P(-1,3)**2*P(1,1)*Metric(2,3) - P(-1,1)**2*P(1,2)*Metric(2,3) - 2*P(-1,3)**2*P(1,2)*Metric(2,3) + P(-1,1)**2*P(1,3)*Metric(2,3) + 2*P(-1,2)**2*P(1,3)*Metric(2,3)')

VVV13 = Lorentz(name = 'VVV13',
                spins = [ 3, 3, 3 ],
                structure = 'Epsilon(1,2,3,-2)*P(-2,3)*P(-1,1)*P(-1,2) + Epsilon(1,2,3,-2)*P(-2,2)*P(-1,1)*P(-1,3) + Epsilon(1,2,3,-2)*P(-2,1)*P(-1,2)*P(-1,3) - Epsilon(2,3,-1,-2)*P(-2,3)*P(-1,1)*P(1,2) - Epsilon(2,3,-1,-2)*P(-2,2)*P(-1,1)*P(1,3) + Epsilon(1,3,-1,-2)*P(-2,3)*P(-1,2)*P(2,1) - Epsilon(1,3,-1,-2)*P(-2,2)*P(-1,1)*P(2,3) + Epsilon(1,2,-1,-2)*P(-2,3)*P(-1,2)*P(3,1) + Epsilon(1,2,-1,-2)*P(-2,3)*P(-1,1)*P(3,2) + Epsilon(3,-1,-2,-3)*P(-3,3)*P(-2,2)*P(-1,1)*Metric(1,2) + Epsilon(2,-1,-2,-3)*P(-3,3)*P(-2,2)*P(-1,1)*Metric(1,3) + Epsilon(1,-1,-2,-3)*P(-3,3)*P(-2,2)*P(-1,1)*Metric(2,3)')

VVV14 = Lorentz(name = 'VVV14',
                spins = [ 3, 3, 3 ],
                structure = '-4*Epsilon(1,2,3,-2)*P(-2,3)*P(-1,1)*P(-1,2) - 4*Epsilon(1,2,3,-2)*P(-2,2)*P(-1,1)*P(-1,3) - 4*Epsilon(1,2,3,-2)*P(-2,1)*P(-1,2)*P(-1,3) + Epsilon(2,3,-1,-2)*P(-2,3)*P(-1,1)*P(1,2) - 3*Epsilon(2,3,-1,-2)*P(-2,1)*P(-1,3)*P(1,2) + Epsilon(2,3,-1,-2)*P(-2,2)*P(-1,1)*P(1,3) - 3*Epsilon(2,3,-1,-2)*P(-2,1)*P(-1,2)*P(1,3) - Epsilon(1,3,-1,-2)*P(-2,3)*P(-1,2)*P(2,1) + 3*Epsilon(1,3,-1,-2)*P(-2,2)*P(-1,3)*P(2,1) + 3*Epsilon(1,3,-1,-2)*P(-2,2)*P(-1,1)*P(2,3) - Epsilon(1,3,-1,-2)*P(-2,1)*P(-1,2)*P(2,3) - 3*Epsilon(1,2,-1,-2)*P(-2,3)*P(-1,2)*P(3,1) + Epsilon(1,2,-1,-2)*P(-2,2)*P(-1,3)*P(3,1) - 3*Epsilon(1,2,-1,-2)*P(-2,3)*P(-1,1)*P(3,2) + Epsilon(1,2,-1,-2)*P(-2,1)*P(-1,3)*P(3,2) + Epsilon(3,-1,-2,-3)*P(-3,2)*P(-2,3)*P(-1,1)*Metric(1,2) - Epsilon(3,-1,-2,-3)*P(-3,1)*P(-2,3)*P(-1,2)*Metric(1,2) - Epsilon(3,-1,-2,-3)*P(-3,2)*P(-2,1)*P(-1,3)*Metric(1,2) + Epsilon(3,-1,-2,-3)*P(-3,1)*P(-2,2)*P(-1,3)*Metric(1,2) - Epsilon(2,-1,-2,-3)*P(-3,3)*P(-2,2)*P(-1,1)*Metric(1,3) + Epsilon(2,-1,-2,-3)*P(-3,3)*P(-2,1)*P(-1,2)*Metric(1,3) - Epsilon(2,-1,-2,-3)*P(-3,1)*P(-2,3)*P(-1,2)*Metric(1,3) + Epsilon(2,-1,-2,-3)*P(-3,1)*P(-2,2)*P(-1,3)*Metric(1,3) - Epsilon(1,-1,-2,-3)*P(-3,3)*P(-2,2)*P(-1,1)*Metric(2,3) + Epsilon(1,-1,-2,-3)*P(-3,2)*P(-2,3)*P(-1,1)*Metric(2,3) + Epsilon(1,-1,-2,-3)*P(-3,3)*P(-2,1)*P(-1,2)*Metric(2,3) - Epsilon(1,-1,-2,-3)*P(-3,2)*P(-2,1)*P(-1,3)*Metric(2,3)')

SSSS1 = Lorentz(name = 'SSSS1',
                spins = [ 1, 1, 1, 1 ],
                structure = '1')

SSSS2 = Lorentz(name = 'SSSS2',
                spins = [ 1, 1, 1, 1 ],
                structure = 'P(-1,1)*P(-1,3) + P(-1,2)*P(-1,3) + P(-1,1)*P(-1,4) + P(-1,2)*P(-1,4)')

SSSS3 = Lorentz(name = 'SSSS3',
                spins = [ 1, 1, 1, 1 ],
                structure = 'P(-1,1)*P(-1,2) - P(-1,1)*P(-1,3) - P(-1,2)*P(-1,4) + P(-1,3)*P(-1,4)')

SSSS4 = Lorentz(name = 'SSSS4',
                spins = [ 1, 1, 1, 1 ],
                structure = 'P(-1,1)*P(-1,2) - (P(-1,1)*P(-1,3))/2. - (P(-1,2)*P(-1,3))/2. - (P(-1,1)*P(-1,4))/2. - (P(-1,2)*P(-1,4))/2. + P(-1,3)*P(-1,4)')

SSSS5 = Lorentz(name = 'SSSS5',
                spins = [ 1, 1, 1, 1 ],
                structure = 'P(-1,1)*P(-1,2) + (P(-1,1)*P(-1,3))/2. + (P(-1,2)*P(-1,3))/2. + (P(-1,1)*P(-1,4))/2. + (P(-1,2)*P(-1,4))/2. + P(-1,3)*P(-1,4)')

SSSS6 = Lorentz(name = 'SSSS6',
                spins = [ 1, 1, 1, 1 ],
                structure = 'P(-1,1)*P(-1,2) + P(-1,1)*P(-1,3) + P(-1,2)*P(-1,3) + P(-1,1)*P(-1,4) + P(-1,2)*P(-1,4) + P(-1,3)*P(-1,4)')

FFSS1 = Lorentz(name = 'FFSS1',
                spins = [ 2, 2, 1, 1 ],
                structure = 'ProjM(2,1)')

FFSS2 = Lorentz(name = 'FFSS2',
                spins = [ 2, 2, 1, 1 ],
                structure = 'P(-1,3)*Gamma(-1,2,-2)*ProjM(-2,1)')

FFSS3 = Lorentz(name = 'FFSS3',
                spins = [ 2, 2, 1, 1 ],
                structure = 'P(-1,4)*Gamma(-1,2,-2)*ProjM(-2,1)')

FFSS4 = Lorentz(name = 'FFSS4',
                spins = [ 2, 2, 1, 1 ],
                structure = 'P(-1,3)*Gamma(-1,2,-2)*ProjM(-2,1) - P(-1,4)*Gamma(-1,2,-2)*ProjM(-2,1)')

FFSS5 = Lorentz(name = 'FFSS5',
                spins = [ 2, 2, 1, 1 ],
                structure = 'ProjM(2,1) - ProjP(2,1)')

FFSS6 = Lorentz(name = 'FFSS6',
                spins = [ 2, 2, 1, 1 ],
                structure = 'ProjP(2,1)')

FFSS7 = Lorentz(name = 'FFSS7',
                spins = [ 2, 2, 1, 1 ],
                structure = 'ProjM(2,1) + ProjP(2,1)')

FFSS8 = Lorentz(name = 'FFSS8',
                spins = [ 2, 2, 1, 1 ],
                structure = 'P(-1,3)*Gamma(-1,2,-2)*ProjP(-2,1)')

FFSS9 = Lorentz(name = 'FFSS9',
                spins = [ 2, 2, 1, 1 ],
                structure = 'P(-1,4)*Gamma(-1,2,-2)*ProjP(-2,1)')

FFSS10 = Lorentz(name = 'FFSS10',
                 spins = [ 2, 2, 1, 1 ],
                 structure = 'P(-1,3)*Gamma(-1,2,-2)*ProjP(-2,1) - P(-1,4)*Gamma(-1,2,-2)*ProjP(-2,1)')

FFVS1 = Lorentz(name = 'FFVS1',
                spins = [ 2, 2, 3, 1 ],
                structure = 'Gamma(3,2,-1)*ProjM(-1,1)')

FFVS2 = Lorentz(name = 'FFVS2',
                spins = [ 2, 2, 3, 1 ],
                structure = 'P(-1,3)*Gamma(-1,2,-3)*Gamma(3,-3,-2)*ProjM(-2,1)')

FFVS3 = Lorentz(name = 'FFVS3',
                spins = [ 2, 2, 3, 1 ],
                structure = 'P(-1,3)*Gamma(-1,-3,-2)*Gamma(3,2,-3)*ProjM(-2,1)')

FFVS4 = Lorentz(name = 'FFVS4',
                spins = [ 2, 2, 3, 1 ],
                structure = '-(P(-1,3)*Gamma(-1,2,-3)*Gamma(3,-3,-2)*ProjM(-2,1)) + P(-1,3)*Gamma(-1,-3,-2)*Gamma(3,2,-3)*ProjM(-2,1)')

FFVS5 = Lorentz(name = 'FFVS5',
                spins = [ 2, 2, 3, 1 ],
                structure = 'P(-1,3)*Gamma(-1,-2,-3)*Gamma(3,2,-2)*ProjM(-3,1)')

FFVS6 = Lorentz(name = 'FFVS6',
                spins = [ 2, 2, 3, 1 ],
                structure = '-(P(-1,3)*Gamma(-1,2,-2)*Gamma(3,-2,-3)*ProjM(-3,1)) + P(-1,3)*Gamma(-1,-3,-2)*Gamma(3,2,-3)*ProjM(-2,1)')

FFVS7 = Lorentz(name = 'FFVS7',
                spins = [ 2, 2, 3, 1 ],
                structure = 'Gamma(3,2,-1)*ProjP(-1,1)')

FFVS8 = Lorentz(name = 'FFVS8',
                spins = [ 2, 2, 3, 1 ],
                structure = 'P(-1,3)*Gamma(-1,2,-3)*Gamma(3,-3,-2)*ProjP(-2,1)')

FFVS9 = Lorentz(name = 'FFVS9',
                spins = [ 2, 2, 3, 1 ],
                structure = 'P(-1,3)*Gamma(-1,-3,-2)*Gamma(3,2,-3)*ProjP(-2,1)')

FFVS10 = Lorentz(name = 'FFVS10',
                 spins = [ 2, 2, 3, 1 ],
                 structure = '-(P(-1,3)*Gamma(-1,2,-3)*Gamma(3,-3,-2)*ProjP(-2,1)) + P(-1,3)*Gamma(-1,-3,-2)*Gamma(3,2,-3)*ProjP(-2,1)')

FFVS11 = Lorentz(name = 'FFVS11',
                 spins = [ 2, 2, 3, 1 ],
                 structure = 'P(-1,3)*Gamma(-1,-2,-3)*Gamma(3,2,-2)*ProjP(-3,1)')

FFVV1 = Lorentz(name = 'FFVV1',
                spins = [ 2, 2, 3, 3 ],
                structure = 'Gamma(3,-2,-1)*Gamma(4,2,-2)*ProjM(-1,1)')

FFVV2 = Lorentz(name = 'FFVV2',
                spins = [ 2, 2, 3, 3 ],
                structure = 'Gamma(3,2,-2)*Gamma(4,-2,-1)*ProjM(-1,1)')

FFVV3 = Lorentz(name = 'FFVV3',
                spins = [ 2, 2, 3, 3 ],
                structure = 'Gamma(3,2,-2)*Gamma(4,-2,-1)*ProjM(-1,1) - Gamma(3,-2,-1)*Gamma(4,2,-2)*ProjM(-1,1)')

FFVV4 = Lorentz(name = 'FFVV4',
                spins = [ 2, 2, 3, 3 ],
                structure = 'Gamma(3,2,-1)*Gamma(4,-1,-2)*ProjM(-2,1)')

FFVV5 = Lorentz(name = 'FFVV5',
                spins = [ 2, 2, 3, 3 ],
                structure = 'Gamma(3,-2,-1)*Gamma(4,2,-2)*ProjP(-1,1)')

FFVV6 = Lorentz(name = 'FFVV6',
                spins = [ 2, 2, 3, 3 ],
                structure = 'Gamma(3,2,-2)*Gamma(4,-2,-1)*ProjP(-1,1)')

FFVV7 = Lorentz(name = 'FFVV7',
                spins = [ 2, 2, 3, 3 ],
                structure = 'Gamma(3,2,-2)*Gamma(4,-2,-1)*ProjP(-1,1) - Gamma(3,-2,-1)*Gamma(4,2,-2)*ProjP(-1,1)')

VSSS1 = Lorentz(name = 'VSSS1',
                spins = [ 3, 1, 1, 1 ],
                structure = 'P(1,2)')

VSSS2 = Lorentz(name = 'VSSS2',
                spins = [ 3, 1, 1, 1 ],
                structure = 'P(1,2) - P(1,3)')

VSSS3 = Lorentz(name = 'VSSS3',
                spins = [ 3, 1, 1, 1 ],
                structure = 'P(1,2) + P(1,3)')

VSSS4 = Lorentz(name = 'VSSS4',
                spins = [ 3, 1, 1, 1 ],
                structure = 'P(1,2) + P(1,3) - 2*P(1,4)')

VSSS5 = Lorentz(name = 'VSSS5',
                spins = [ 3, 1, 1, 1 ],
                structure = 'P(1,2) - P(1,4)/2.')

VSSS6 = Lorentz(name = 'VSSS6',
                spins = [ 3, 1, 1, 1 ],
                structure = 'P(1,2) - P(1,3)/2. - P(1,4)/2.')

VSSS7 = Lorentz(name = 'VSSS7',
                spins = [ 3, 1, 1, 1 ],
                structure = 'P(1,2) - P(1,3)/3. - P(1,4)/3.')

VSSS8 = Lorentz(name = 'VSSS8',
                spins = [ 3, 1, 1, 1 ],
                structure = 'P(1,2) + P(1,3) + P(1,4)')

VVSS1 = Lorentz(name = 'VVSS1',
                spins = [ 3, 3, 1, 1 ],
                structure = 'Epsilon(1,2,-1,-2)*P(-2,2)*P(-1,1)')

VVSS2 = Lorentz(name = 'VVSS2',
                spins = [ 3, 3, 1, 1 ],
                structure = '-(Epsilon(1,2,-1,-2)*P(-2,2)*P(-1,1)) + Epsilon(1,2,-1,-2)*P(-2,1)*P(-1,2)')

VVSS3 = Lorentz(name = 'VVSS3',
                spins = [ 3, 3, 1, 1 ],
                structure = 'Epsilon(1,2,-1,-2)*P(-2,1)*P(-1,3) + Epsilon(1,2,-1,-2)*P(-2,1)*P(-1,4)')

VVSS4 = Lorentz(name = 'VVSS4',
                spins = [ 3, 3, 1, 1 ],
                structure = 'Epsilon(1,2,-1,-2)*P(-2,1)*P(-1,3) - 2*Epsilon(1,2,-1,-2)*P(-2,2)*P(-1,3) + Epsilon(1,2,-1,-2)*P(-2,1)*P(-1,4) - 2*Epsilon(1,2,-1,-2)*P(-2,2)*P(-1,4)')

VVSS5 = Lorentz(name = 'VVSS5',
                spins = [ 3, 3, 1, 1 ],
                structure = 'Epsilon(1,2,-1,-2)*P(-2,1)*P(-1,3) - Epsilon(1,2,-1,-2)*P(-2,2)*P(-1,3) + Epsilon(1,2,-1,-2)*P(-2,1)*P(-1,4) - Epsilon(1,2,-1,-2)*P(-2,2)*P(-1,4)')

VVSS6 = Lorentz(name = 'VVSS6',
                spins = [ 3, 3, 1, 1 ],
                structure = 'Epsilon(1,2,-1,-2)*P(-2,2)*P(-1,3) + Epsilon(1,2,-1,-2)*P(-2,2)*P(-1,4)')

VVSS7 = Lorentz(name = 'VVSS7',
                spins = [ 3, 3, 1, 1 ],
                structure = 'Epsilon(1,2,-1,-2)*P(-2,2)*P(-1,3) + Epsilon(1,2,-1,-2)*P(-2,4)*P(-1,3) - Epsilon(1,2,-1,-2)*P(-2,1)*P(-1,4)')

VVSS8 = Lorentz(name = 'VVSS8',
                spins = [ 3, 3, 1, 1 ],
                structure = 'Epsilon(1,2,-1,-2)*P(-2,1)*P(-1,3) + 2*Epsilon(1,2,-1,-2)*P(-2,2)*P(-1,3) + 2*Epsilon(1,2,-1,-2)*P(-2,4)*P(-1,3) - Epsilon(1,2,-1,-2)*P(-2,1)*P(-1,4)')

VVSS9 = Lorentz(name = 'VVSS9',
                spins = [ 3, 3, 1, 1 ],
                structure = 'Epsilon(1,2,-1,-2)*P(-2,1)*P(-1,3) + 2*Epsilon(1,2,-1,-2)*P(-2,4)*P(-1,3) - Epsilon(1,2,-1,-2)*P(-2,1)*P(-1,4) - 2*Epsilon(1,2,-1,-2)*P(-2,2)*P(-1,4)')

VVSS10 = Lorentz(name = 'VVSS10',
                 spins = [ 3, 3, 1, 1 ],
                 structure = 'Epsilon(1,2,-1,-2)*P(-2,1)*P(-1,3) + Epsilon(1,2,-1,-2)*P(-2,2)*P(-1,3) + 2*Epsilon(1,2,-1,-2)*P(-2,4)*P(-1,3) - Epsilon(1,2,-1,-2)*P(-2,1)*P(-1,4) - Epsilon(1,2,-1,-2)*P(-2,2)*P(-1,4)')

VVSS11 = Lorentz(name = 'VVSS11',
                 spins = [ 3, 3, 1, 1 ],
                 structure = 'Metric(1,2)')

VVSS12 = Lorentz(name = 'VVSS12',
                 spins = [ 3, 3, 1, 1 ],
                 structure = 'P(1,1)*P(2,1) - P(-1,1)**2*Metric(1,2)')

VVSS13 = Lorentz(name = 'VVSS13',
                 spins = [ 3, 3, 1, 1 ],
                 structure = 'P(1,2)*P(2,1) - P(-1,1)*P(-1,2)*Metric(1,2)')

VVSS14 = Lorentz(name = 'VVSS14',
                 spins = [ 3, 3, 1, 1 ],
                 structure = 'P(1,1)*P(2,1) + 2*P(1,2)*P(2,2) - P(-1,1)**2*Metric(1,2) - 2*P(-1,2)**2*Metric(1,2)')

VVSS15 = Lorentz(name = 'VVSS15',
                 spins = [ 3, 3, 1, 1 ],
                 structure = 'P(1,2)*P(2,2) - P(-1,2)**2*Metric(1,2)')

VVSS16 = Lorentz(name = 'VVSS16',
                 spins = [ 3, 3, 1, 1 ],
                 structure = 'P(1,1)*P(2,1) + P(1,2)*P(2,2) - P(-1,1)**2*Metric(1,2) - P(-1,2)**2*Metric(1,2)')

VVSS17 = Lorentz(name = 'VVSS17',
                 spins = [ 3, 3, 1, 1 ],
                 structure = 'P(1,3)*P(2,1) + P(1,4)*P(2,1) - P(-1,1)*P(-1,3)*Metric(1,2) - P(-1,1)*P(-1,4)*Metric(1,2)')

VVSS18 = Lorentz(name = 'VVSS18',
                 spins = [ 3, 3, 1, 1 ],
                 structure = 'P(1,4)*P(2,1) + P(1,2)*P(2,3) - P(1,4)*P(2,3) + P(1,3)*P(2,4) - P(-1,2)*P(-1,3)*Metric(1,2) - P(-1,1)*P(-1,4)*Metric(1,2)')

VVSS19 = Lorentz(name = 'VVSS19',
                 spins = [ 3, 3, 1, 1 ],
                 structure = 'P(1,3)*P(2,1) - P(1,4)*P(2,1) - 2*P(1,2)*P(2,3) + 2*P(1,4)*P(2,3) - 2*P(1,3)*P(2,4) - P(-1,1)*P(-1,3)*Metric(1,2) + 2*P(-1,2)*P(-1,3)*Metric(1,2) + P(-1,1)*P(-1,4)*Metric(1,2)')

VVSS20 = Lorentz(name = 'VVSS20',
                 spins = [ 3, 3, 1, 1 ],
                 structure = 'P(1,3)*P(2,1) + P(1,4)*P(2,1) + 2*P(1,2)*P(2,3) + 2*P(1,2)*P(2,4) - P(-1,1)*P(-1,3)*Metric(1,2) - 2*P(-1,2)*P(-1,3)*Metric(1,2) - P(-1,1)*P(-1,4)*Metric(1,2) - 2*P(-1,2)*P(-1,4)*Metric(1,2)')

VVSS21 = Lorentz(name = 'VVSS21',
                 spins = [ 3, 3, 1, 1 ],
                 structure = 'P(1,3)*P(2,1) - P(1,4)*P(2,1) + 2*P(1,4)*P(2,3) + 2*P(1,2)*P(2,4) - 2*P(1,3)*P(2,4) - P(-1,1)*P(-1,3)*Metric(1,2) + P(-1,1)*P(-1,4)*Metric(1,2) - 2*P(-1,2)*P(-1,4)*Metric(1,2)')

VVSS22 = Lorentz(name = 'VVSS22',
                 spins = [ 3, 3, 1, 1 ],
                 structure = 'P(1,2)*P(2,3) + P(1,2)*P(2,4) - P(-1,2)*P(-1,3)*Metric(1,2) - P(-1,2)*P(-1,4)*Metric(1,2)')

VVSS23 = Lorentz(name = 'VVSS23',
                 spins = [ 3, 3, 1, 1 ],
                 structure = 'P(1,3)*P(2,1) + P(1,4)*P(2,1) + P(1,2)*P(2,3) + P(1,2)*P(2,4) - P(-1,1)*P(-1,3)*Metric(1,2) - P(-1,2)*P(-1,3)*Metric(1,2) - P(-1,1)*P(-1,4)*Metric(1,2) - P(-1,2)*P(-1,4)*Metric(1,2)')

VVSS24 = Lorentz(name = 'VVSS24',
                 spins = [ 3, 3, 1, 1 ],
                 structure = 'P(1,3)*P(2,1) - P(1,4)*P(2,1) - P(1,2)*P(2,3) + 2*P(1,4)*P(2,3) + P(1,2)*P(2,4) - 2*P(1,3)*P(2,4) - P(-1,1)*P(-1,3)*Metric(1,2) + P(-1,2)*P(-1,3)*Metric(1,2) + P(-1,1)*P(-1,4)*Metric(1,2) - P(-1,2)*P(-1,4)*Metric(1,2)')

VVSS25 = Lorentz(name = 'VVSS25',
                 spins = [ 3, 3, 1, 1 ],
                 structure = '2*P(1,3)*P(2,1) - 2*P(1,4)*P(2,1) + P(1,2)*P(2,2) + P(1,3)*P(2,2) - P(1,4)*P(2,2) - P(1,1)*P(2,3) - 2*P(1,2)*P(2,3) + P(1,1)*P(2,4) + 2*P(1,2)*P(2,4) - P(-1,2)**2*Metric(1,2) - P(-1,1)*P(-1,3)*Metric(1,2) + P(-1,2)*P(-1,3)*Metric(1,2) + P(-1,1)*P(-1,4)*Metric(1,2) - P(-1,2)*P(-1,4)*Metric(1,2)')

VVSS26 = Lorentz(name = 'VVSS26',
                 spins = [ 3, 3, 1, 1 ],
                 structure = 'P(1,1)*P(2,1) + 2*P(1,3)*P(2,1) - 2*P(1,4)*P(2,1) + P(1,2)*P(2,2) + P(1,3)*P(2,2) - P(1,4)*P(2,2) - P(1,1)*P(2,3) - 2*P(1,2)*P(2,3) + P(1,1)*P(2,4) + 2*P(1,2)*P(2,4) - P(-1,1)**2*Metric(1,2) - P(-1,2)**2*Metric(1,2) - P(-1,1)*P(-1,3)*Metric(1,2) + P(-1,2)*P(-1,3)*Metric(1,2) + P(-1,1)*P(-1,4)*Metric(1,2) - P(-1,2)*P(-1,4)*Metric(1,2)')

VVSS27 = Lorentz(name = 'VVSS27',
                 spins = [ 3, 3, 1, 1 ],
                 structure = 'P(1,3)*P(2,1) - P(1,4)*P(2,1) + (P(1,3)*P(2,2))/2. - (P(1,4)*P(2,2))/2. - (P(1,1)*P(2,3))/2. - P(1,2)*P(2,3) + (P(1,1)*P(2,4))/2. + P(1,2)*P(2,4) - (P(-1,1)*P(-1,3)*Metric(1,2))/2. + (P(-1,2)*P(-1,3)*Metric(1,2))/2. + (P(-1,1)*P(-1,4)*Metric(1,2))/2. - (P(-1,2)*P(-1,4)*Metric(1,2))/2.')

VVSS28 = Lorentz(name = 'VVSS28',
                 spins = [ 3, 3, 1, 1 ],
                 structure = '-2*P(1,3)*P(2,1) + 2*P(1,4)*P(2,1) + P(1,2)*P(2,2) - P(1,3)*P(2,2) + P(1,4)*P(2,2) + P(1,1)*P(2,3) + 2*P(1,2)*P(2,3) - P(1,1)*P(2,4) - 2*P(1,2)*P(2,4) - P(-1,2)**2*Metric(1,2) + P(-1,1)*P(-1,3)*Metric(1,2) - P(-1,2)*P(-1,3)*Metric(1,2) - P(-1,1)*P(-1,4)*Metric(1,2) + P(-1,2)*P(-1,4)*Metric(1,2)')

VVVS1 = Lorentz(name = 'VVVS1',
                spins = [ 3, 3, 3, 1 ],
                structure = '-(Epsilon(1,2,3,-1)*P(-1,1))')

VVVS2 = Lorentz(name = 'VVVS2',
                spins = [ 3, 3, 3, 1 ],
                structure = '-(Epsilon(1,2,3,-1)*P(-1,1)) + Epsilon(1,2,3,-1)*P(-1,2)')

VVVS3 = Lorentz(name = 'VVVS3',
                spins = [ 3, 3, 3, 1 ],
                structure = '-(Epsilon(1,2,3,-1)*P(-1,3))')

VVVS4 = Lorentz(name = 'VVVS4',
                spins = [ 3, 3, 3, 1 ],
                structure = '-(Epsilon(1,2,3,-1)*P(-1,2)) + Epsilon(1,2,3,-1)*P(-1,3)')

VVVS5 = Lorentz(name = 'VVVS5',
                spins = [ 3, 3, 3, 1 ],
                structure = '-(Epsilon(1,2,3,-1)*P(-1,1)) - Epsilon(1,2,3,-1)*P(-1,2) - Epsilon(1,2,3,-1)*P(-1,3)')

VVVS6 = Lorentz(name = 'VVVS6',
                spins = [ 3, 3, 3, 1 ],
                structure = '-(Epsilon(1,2,3,-1)*P(-1,1)) - Epsilon(1,2,3,-1)*P(-1,2) - Epsilon(1,2,3,-1)*P(-1,4)')

VVVS7 = Lorentz(name = 'VVVS7',
                spins = [ 3, 3, 3, 1 ],
                structure = '-(Epsilon(1,2,3,-1)*P(-1,2)) - Epsilon(1,2,3,-1)*P(-1,3) - Epsilon(1,2,3,-1)*P(-1,4)')

VVVS8 = Lorentz(name = 'VVVS8',
                spins = [ 3, 3, 3, 1 ],
                structure = '-(Epsilon(1,2,3,-1)*P(-1,1)) - 2*Epsilon(1,2,3,-1)*P(-1,4)')

VVVS9 = Lorentz(name = 'VVVS9',
                spins = [ 3, 3, 3, 1 ],
                structure = '-(Epsilon(1,2,3,-1)*P(-1,1)) - Epsilon(1,2,3,-1)*P(-1,2) - Epsilon(1,2,3,-1)*P(-1,3) - 3*Epsilon(1,2,3,-1)*P(-1,4)')

VVVS10 = Lorentz(name = 'VVVS10',
                 spins = [ 3, 3, 3, 1 ],
                 structure = 'P(3,1)*Metric(1,2) - P(2,1)*Metric(1,3)')

VVVS11 = Lorentz(name = 'VVVS11',
                 spins = [ 3, 3, 3, 1 ],
                 structure = 'P(3,1)*Metric(1,2) - P(3,4)*Metric(1,2) - P(2,1)*Metric(1,3) + P(2,4)*Metric(1,3)')

VVVS12 = Lorentz(name = 'VVVS12',
                 spins = [ 3, 3, 3, 1 ],
                 structure = 'P(3,1)*Metric(1,2) + P(3,2)*Metric(1,2) - P(2,1)*Metric(1,3) - P(1,2)*Metric(2,3)')

VVVS13 = Lorentz(name = 'VVVS13',
                 spins = [ 3, 3, 3, 1 ],
                 structure = 'P(3,1)*Metric(1,2) - P(3,2)*Metric(1,2) - 2*P(2,1)*Metric(1,3) - P(2,2)*Metric(1,3) + P(1,1)*Metric(2,3) + 2*P(1,2)*Metric(2,3)')

VVVS14 = Lorentz(name = 'VVVS14',
                 spins = [ 3, 3, 3, 1 ],
                 structure = 'P(3,1)*Metric(1,2) + P(3,2)*Metric(1,2) + P(3,3)*Metric(1,2) + P(2,1)*Metric(1,3) + P(2,2)*Metric(1,3) + P(2,3)*Metric(1,3) - 2*P(1,1)*Metric(2,3) - 2*P(1,2)*Metric(2,3) - 2*P(1,3)*Metric(2,3)')

VVVS15 = Lorentz(name = 'VVVS15',
                 spins = [ 3, 3, 3, 1 ],
                 structure = 'P(2,3)*Metric(1,3) - P(1,3)*Metric(2,3)')

VVVS16 = Lorentz(name = 'VVVS16',
                 spins = [ 3, 3, 3, 1 ],
                 structure = 'P(3,2)*Metric(1,2) + P(2,3)*Metric(1,3) - P(1,2)*Metric(2,3) - P(1,3)*Metric(2,3)')

VVVS17 = Lorentz(name = 'VVVS17',
                 spins = [ 3, 3, 3, 1 ],
                 structure = 'P(3,1)*Metric(1,2) - P(3,2)*Metric(1,2) - P(2,1)*Metric(1,3) + P(2,3)*Metric(1,3) + P(1,2)*Metric(2,3) - P(1,3)*Metric(2,3)')

VVVS18 = Lorentz(name = 'VVVS18',
                 spins = [ 3, 3, 3, 1 ],
                 structure = 'P(3,1)*Metric(1,2) - (P(3,2)*Metric(1,2))/3. + (P(3,3)*Metric(1,2))/3. - P(2,1)*Metric(1,3) - (P(2,2)*Metric(1,3))/3. + (P(2,3)*Metric(1,3))/3. + (2*P(1,2)*Metric(2,3))/3. - (2*P(1,3)*Metric(2,3))/3.')

VVVS19 = Lorentz(name = 'VVVS19',
                 spins = [ 3, 3, 3, 1 ],
                 structure = 'P(3,4)*Metric(1,2) + P(2,4)*Metric(1,3) - 2*P(1,4)*Metric(2,3)')

VVVS20 = Lorentz(name = 'VVVS20',
                 spins = [ 3, 3, 3, 1 ],
                 structure = 'P(3,1)*Metric(1,2) - P(3,2)*Metric(1,2) - P(2,1)*Metric(1,3) + P(2,4)*Metric(1,3) + P(1,2)*Metric(2,3) - P(1,4)*Metric(2,3)')

VVVS21 = Lorentz(name = 'VVVS21',
                 spins = [ 3, 3, 3, 1 ],
                 structure = 'P(3,4)*Metric(1,2) - (P(2,4)*Metric(1,3))/2. - (P(1,4)*Metric(2,3))/2.')

VVVS22 = Lorentz(name = 'VVVS22',
                 spins = [ 3, 3, 3, 1 ],
                 structure = 'P(3,1)*Metric(1,2) + P(3,2)*Metric(1,2) + P(3,3)*Metric(1,2) + P(3,4)*Metric(1,2) - (P(2,1)*Metric(1,3))/2. - (P(2,2)*Metric(1,3))/2. - (P(2,3)*Metric(1,3))/2. - (P(2,4)*Metric(1,3))/2. - (P(1,1)*Metric(2,3))/2. - (P(1,2)*Metric(2,3))/2. - (P(1,3)*Metric(2,3))/2. - (P(1,4)*Metric(2,3))/2.')

VVVS23 = Lorentz(name = 'VVVS23',
                 spins = [ 3, 3, 3, 1 ],
                 structure = 'P(3,2)*Metric(1,2) + (P(3,3)*Metric(1,2))/2. + (P(3,4)*Metric(1,2))/2. - (P(2,2)*Metric(1,3))/2. - P(2,3)*Metric(1,3) - P(2,4)*Metric(1,3) - (P(1,2)*Metric(2,3))/2. + (P(1,3)*Metric(2,3))/2. + (P(1,4)*Metric(2,3))/2.')

VVVS24 = Lorentz(name = 'VVVS24',
                 spins = [ 3, 3, 3, 1 ],
                 structure = 'P(3,2)*Metric(1,2) - P(2,3)*Metric(1,3) - P(2,4)*Metric(1,3) - P(1,2)*Metric(2,3) + P(1,3)*Metric(2,3) + P(1,4)*Metric(2,3)')

VVVS25 = Lorentz(name = 'VVVS25',
                 spins = [ 3, 3, 3, 1 ],
                 structure = 'P(3,2)*Metric(1,2) - P(3,4)*Metric(1,2) + P(2,3)*Metric(1,3) - P(2,4)*Metric(1,3) - P(1,2)*Metric(2,3) - P(1,3)*Metric(2,3) + 2*P(1,4)*Metric(2,3)')

VVVV1 = Lorentz(name = 'VVVV1',
                spins = [ 3, 3, 3, 3 ],
                structure = 'Metric(1,4)*Metric(2,3) - Metric(1,3)*Metric(2,4)')

VVVV2 = Lorentz(name = 'VVVV2',
                spins = [ 3, 3, 3, 3 ],
                structure = 'P(3,2)*P(4,1)*Metric(1,2) - P(3,1)*P(4,2)*Metric(1,2) + P(2,1)*P(4,2)*Metric(1,3) + P(2,4)*P(4,3)*Metric(1,3) - P(2,1)*P(3,2)*Metric(1,4) - P(2,3)*P(3,4)*Metric(1,4) - P(1,2)*P(4,1)*Metric(2,3) - P(1,4)*P(4,3)*Metric(2,3) + P(-1,1)*P(-1,2)*Metric(1,4)*Metric(2,3) + P(-1,3)*P(-1,4)*Metric(1,4)*Metric(2,3) + P(1,2)*P(3,1)*Metric(2,4) + P(1,3)*P(3,4)*Metric(2,4) - P(-1,1)*P(-1,2)*Metric(1,3)*Metric(2,4) - P(-1,3)*P(-1,4)*Metric(1,3)*Metric(2,4) + P(1,4)*P(2,3)*Metric(3,4) - P(1,3)*P(2,4)*Metric(3,4)')

VVVV3 = Lorentz(name = 'VVVV3',
                spins = [ 3, 3, 3, 3 ],
                structure = '4*Epsilon(1,2,3,4)*P(-1,2)*P(-1,3) + 4*Epsilon(1,2,3,4)*P(-1,1)*P(-1,4) - 4*Epsilon(2,3,4,-1)*P(-1,3)*P(1,2) - 4*Epsilon(2,3,4,-1)*P(-1,2)*P(1,3) + 4*Epsilon(2,3,4,-1)*P(-1,1)*P(1,4) + 4*Epsilon(1,3,4,-1)*P(-1,4)*P(2,1) - 4*Epsilon(1,3,4,-1)*P(-1,2)*P(2,3) + 4*Epsilon(1,3,4,-1)*P(-1,1)*P(2,4) - 4*Epsilon(1,2,4,-1)*P(-1,4)*P(3,1) + 4*Epsilon(1,2,4,-1)*P(-1,3)*P(3,2) - 4*Epsilon(1,2,4,-1)*P(-1,1)*P(3,4) - 4*Epsilon(1,2,3,-1)*P(-1,4)*P(4,1) + 4*Epsilon(1,2,3,-1)*P(-1,3)*P(4,2) + 4*Epsilon(1,2,3,-1)*P(-1,2)*P(4,3) - 2*Epsilon(3,4,-1,-2)*P(-2,4)*P(-1,1)*Metric(1,2) + 2*Epsilon(3,4,-1,-2)*P(-2,3)*P(-1,2)*Metric(1,2) - 2*Epsilon(3,4,-1,-2)*P(-2,2)*P(-1,3)*Metric(1,2) + 2*Epsilon(3,4,-1,-2)*P(-2,1)*P(-1,4)*Metric(1,2) + 2*Epsilon(2,4,-1,-2)*P(-2,4)*P(-1,1)*Metric(1,3) + 2*Epsilon(2,4,-1,-2)*P(-2,3)*P(-1,2)*Metric(1,3) - 2*Epsilon(2,4,-1,-2)*P(-2,2)*P(-1,3)*Metric(1,3) - 2*Epsilon(2,4,-1,-2)*P(-2,1)*P(-1,4)*Metric(1,3) + 2*Epsilon(2,3,-1,-2)*P(-2,4)*P(-1,1)*Metric(1,4) - 2*Epsilon(2,3,-1,-2)*P(-2,1)*P(-1,4)*Metric(1,4) + 2*Epsilon(1,4,-1,-2)*P(-2,3)*P(-1,2)*Metric(2,3) - 2*Epsilon(1,4,-1,-2)*P(-2,2)*P(-1,3)*Metric(2,3) + 2*Epsilon(1,3,-1,-2)*P(-2,4)*P(-1,1)*Metric(2,4) + 2*Epsilon(1,3,-1,-2)*P(-2,3)*P(-1,2)*Metric(2,4) - 2*Epsilon(1,3,-1,-2)*P(-2,2)*P(-1,3)*Metric(2,4) - 2*Epsilon(1,3,-1,-2)*P(-2,1)*P(-1,4)*Metric(2,4) - 2*Epsilon(1,2,-1,-2)*P(-2,4)*P(-1,1)*Metric(3,4) + 2*Epsilon(1,2,-1,-2)*P(-2,3)*P(-1,2)*Metric(3,4) - 2*Epsilon(1,2,-1,-2)*P(-2,2)*P(-1,3)*Metric(3,4) + 2*Epsilon(1,2,-1,-2)*P(-2,1)*P(-1,4)*Metric(3,4)')

VVVV4 = Lorentz(name = 'VVVV4',
                spins = [ 3, 3, 3, 3 ],
                structure = '-(Epsilon(1,2,3,4)*P(-1,1)*P(-1,3)) + Epsilon(1,2,3,4)*P(-1,2)*P(-1,3) + Epsilon(1,2,3,4)*P(-1,1)*P(-1,4) - Epsilon(1,2,3,4)*P(-1,2)*P(-1,4) - Epsilon(2,3,4,-1)*P(-1,3)*P(1,2) + Epsilon(2,3,4,-1)*P(-1,4)*P(1,2) - Epsilon(2,3,4,-1)*P(-1,1)*P(1,3) - Epsilon(2,3,4,-1)*P(-1,2)*P(1,3) + Epsilon(2,3,4,-1)*P(-1,1)*P(1,4) + Epsilon(2,3,4,-1)*P(-1,2)*P(1,4) - Epsilon(1,3,4,-1)*P(-1,3)*P(2,1) + Epsilon(1,3,4,-1)*P(-1,4)*P(2,1) - Epsilon(1,3,4,-1)*P(-1,1)*P(2,3) - Epsilon(1,3,4,-1)*P(-1,2)*P(2,3) + Epsilon(1,3,4,-1)*P(-1,1)*P(2,4) + Epsilon(1,3,4,-1)*P(-1,2)*P(2,4) - Epsilon(1,2,4,-1)*P(-1,3)*P(3,1) - Epsilon(1,2,4,-1)*P(-1,4)*P(3,1) + Epsilon(1,2,4,-1)*P(-1,3)*P(3,2) + Epsilon(1,2,4,-1)*P(-1,4)*P(3,2) - Epsilon(1,2,4,-1)*P(-1,1)*P(3,4) + Epsilon(1,2,4,-1)*P(-1,2)*P(3,4) - Epsilon(1,2,3,-1)*P(-1,3)*P(4,1) - Epsilon(1,2,3,-1)*P(-1,4)*P(4,1) + Epsilon(1,2,3,-1)*P(-1,3)*P(4,2) + Epsilon(1,2,3,-1)*P(-1,4)*P(4,2) - Epsilon(1,2,3,-1)*P(-1,1)*P(4,3) + Epsilon(1,2,3,-1)*P(-1,2)*P(4,3) + Epsilon(3,4,-1,-2)*P(-2,3)*P(-1,1)*Metric(1,2) - Epsilon(3,4,-1,-2)*P(-2,4)*P(-1,1)*Metric(1,2) + Epsilon(3,4,-1,-2)*P(-2,3)*P(-1,2)*Metric(1,2) - Epsilon(3,4,-1,-2)*P(-2,4)*P(-1,2)*Metric(1,2) + Epsilon(2,4,-1,-2)*P(-2,3)*P(-1,1)*Metric(1,3) + Epsilon(2,4,-1,-2)*P(-2,4)*P(-1,1)*Metric(1,3) + Epsilon(2,4,-1,-2)*P(-2,3)*P(-1,2)*Metric(1,3) + Epsilon(2,3,-1,-2)*P(-2,3)*P(-1,1)*Metric(1,4) + Epsilon(2,3,-1,-2)*P(-2,4)*P(-1,1)*Metric(1,4) + Epsilon(2,3,-1,-2)*P(-2,4)*P(-1,2)*Metric(1,4) + Epsilon(1,4,-1,-2)*P(-2,3)*P(-1,1)*Metric(2,3) + Epsilon(1,4,-1,-2)*P(-2,3)*P(-1,2)*Metric(2,3) + Epsilon(1,4,-1,-2)*P(-2,4)*P(-1,2)*Metric(2,3) + Epsilon(1,3,-1,-2)*P(-2,4)*P(-1,1)*Metric(2,4) + Epsilon(1,3,-1,-2)*P(-2,3)*P(-1,2)*Metric(2,4) + Epsilon(1,3,-1,-2)*P(-2,4)*P(-1,2)*Metric(2,4) - Epsilon(1,2,-1,-2)*P(-2,3)*P(-1,1)*Metric(3,4) - Epsilon(1,2,-1,-2)*P(-2,4)*P(-1,1)*Metric(3,4) + Epsilon(1,2,-1,-2)*P(-2,3)*P(-1,2)*Metric(3,4) + Epsilon(1,2,-1,-2)*P(-2,4)*P(-1,2)*Metric(3,4)')

VVVV5 = Lorentz(name = 'VVVV5',
                spins = [ 3, 3, 3, 3 ],
                structure = '-4*Epsilon(1,2,3,4)*P(-1,1)*P(-1,3) - 4*Epsilon(1,2,3,4)*P(-1,2)*P(-1,4) + 4*Epsilon(2,3,4,-1)*P(-1,4)*P(1,2) - 4*Epsilon(2,3,4,-1)*P(-1,1)*P(1,3) + 4*Epsilon(2,3,4,-1)*P(-1,2)*P(1,4) - 4*Epsilon(1,3,4,-1)*P(-1,3)*P(2,1) - 4*Epsilon(1,3,4,-1)*P(-1,1)*P(2,3) + 4*Epsilon(1,3,4,-1)*P(-1,2)*P(2,4) - 4*Epsilon(1,2,4,-1)*P(-1,3)*P(3,1) + 4*Epsilon(1,2,4,-1)*P(-1,4)*P(3,2) + 4*Epsilon(1,2,4,-1)*P(-1,2)*P(3,4) - 4*Epsilon(1,2,3,-1)*P(-1,3)*P(4,1) + 4*Epsilon(1,2,3,-1)*P(-1,4)*P(4,2) - 4*Epsilon(1,2,3,-1)*P(-1,1)*P(4,3) + 2*Epsilon(3,4,-1,-2)*P(-2,3)*P(-1,1)*Metric(1,2) - 2*Epsilon(3,4,-1,-2)*P(-2,4)*P(-1,2)*Metric(1,2) - 2*Epsilon(3,4,-1,-2)*P(-2,1)*P(-1,3)*Metric(1,2) + 2*Epsilon(3,4,-1,-2)*P(-2,2)*P(-1,4)*Metric(1,2) + 2*Epsilon(2,4,-1,-2)*P(-2,3)*P(-1,1)*Metric(1,3) - 2*Epsilon(2,4,-1,-2)*P(-2,1)*P(-1,3)*Metric(1,3) + 2*Epsilon(2,3,-1,-2)*P(-2,3)*P(-1,1)*Metric(1,4) + 2*Epsilon(2,3,-1,-2)*P(-2,4)*P(-1,2)*Metric(1,4) - 2*Epsilon(2,3,-1,-2)*P(-2,1)*P(-1,3)*Metric(1,4) - 2*Epsilon(2,3,-1,-2)*P(-2,2)*P(-1,4)*Metric(1,4) + 2*Epsilon(1,4,-1,-2)*P(-2,3)*P(-1,1)*Metric(2,3) + 2*Epsilon(1,4,-1,-2)*P(-2,4)*P(-1,2)*Metric(2,3) - 2*Epsilon(1,4,-1,-2)*P(-2,1)*P(-1,3)*Metric(2,3) - 2*Epsilon(1,4,-1,-2)*P(-2,2)*P(-1,4)*Metric(2,3) + 2*Epsilon(1,3,-1,-2)*P(-2,4)*P(-1,2)*Metric(2,4) - 2*Epsilon(1,3,-1,-2)*P(-2,2)*P(-1,4)*Metric(2,4) - 2*Epsilon(1,2,-1,-2)*P(-2,3)*P(-1,1)*Metric(3,4) + 2*Epsilon(1,2,-1,-2)*P(-2,4)*P(-1,2)*Metric(3,4) + 2*Epsilon(1,2,-1,-2)*P(-2,1)*P(-1,3)*Metric(3,4) - 2*Epsilon(1,2,-1,-2)*P(-2,2)*P(-1,4)*Metric(3,4)')

VVVV6 = Lorentz(name = 'VVVV6',
                spins = [ 3, 3, 3, 3 ],
                structure = '2*Epsilon(1,2,3,4)*P(-1,1)*P(-1,2) + 2*Epsilon(1,2,3,4)*P(-1,3)*P(-1,4) + 2*Epsilon(2,3,4,-1)*P(-1,1)*P(1,2) - 2*Epsilon(2,3,4,-1)*P(-1,4)*P(1,3) - 2*Epsilon(2,3,4,-1)*P(-1,3)*P(1,4) - 2*Epsilon(1,3,4,-1)*P(-1,2)*P(2,1) + 2*Epsilon(1,3,4,-1)*P(-1,4)*P(2,3) + 2*Epsilon(1,3,4,-1)*P(-1,3)*P(2,4) - 2*Epsilon(1,2,4,-1)*P(-1,2)*P(3,1) - 2*Epsilon(1,2,4,-1)*P(-1,1)*P(3,2) + 2*Epsilon(1,2,4,-1)*P(-1,3)*P(3,4) + 2*Epsilon(1,2,3,-1)*P(-1,2)*P(4,1) + 2*Epsilon(1,2,3,-1)*P(-1,1)*P(4,2) - 2*Epsilon(1,2,3,-1)*P(-1,4)*P(4,3) + Epsilon(3,4,-1,-2)*P(-2,2)*P(-1,1)*Metric(1,2) - Epsilon(3,4,-1,-2)*P(-2,1)*P(-1,2)*Metric(1,2) + Epsilon(2,4,-1,-2)*P(-2,2)*P(-1,1)*Metric(1,3) - Epsilon(2,4,-1,-2)*P(-2,1)*P(-1,2)*Metric(1,3) - Epsilon(2,4,-1,-2)*P(-2,4)*P(-1,3)*Metric(1,3) + Epsilon(2,4,-1,-2)*P(-2,3)*P(-1,4)*Metric(1,3) - Epsilon(2,3,-1,-2)*P(-2,2)*P(-1,1)*Metric(1,4) + Epsilon(2,3,-1,-2)*P(-2,1)*P(-1,2)*Metric(1,4) - Epsilon(2,3,-1,-2)*P(-2,4)*P(-1,3)*Metric(1,4) + Epsilon(2,3,-1,-2)*P(-2,3)*P(-1,4)*Metric(1,4) + Epsilon(1,4,-1,-2)*P(-2,2)*P(-1,1)*Metric(2,3) - Epsilon(1,4,-1,-2)*P(-2,1)*P(-1,2)*Metric(2,3) + Epsilon(1,4,-1,-2)*P(-2,4)*P(-1,3)*Metric(2,3) - Epsilon(1,4,-1,-2)*P(-2,3)*P(-1,4)*Metric(2,3) - Epsilon(1,3,-1,-2)*P(-2,2)*P(-1,1)*Metric(2,4) + Epsilon(1,3,-1,-2)*P(-2,1)*P(-1,2)*Metric(2,4) + Epsilon(1,3,-1,-2)*P(-2,4)*P(-1,3)*Metric(2,4) - Epsilon(1,3,-1,-2)*P(-2,3)*P(-1,4)*Metric(2,4) + Epsilon(1,2,-1,-2)*P(-2,4)*P(-1,3)*Metric(3,4) - Epsilon(1,2,-1,-2)*P(-2,3)*P(-1,4)*Metric(3,4)')

VVVV7 = Lorentz(name = 'VVVV7',
                spins = [ 3, 3, 3, 3 ],
                structure = 'Epsilon(1,2,3,4)*P(-1,1)*P(-1,2) - Epsilon(1,2,3,4)*P(-1,1)*P(-1,3) - Epsilon(1,2,3,4)*P(-1,2)*P(-1,4) + Epsilon(1,2,3,4)*P(-1,3)*P(-1,4) + Epsilon(2,3,4,-1)*P(-1,1)*P(1,2) + Epsilon(2,3,4,-1)*P(-1,4)*P(1,2) - Epsilon(2,3,4,-1)*P(-1,1)*P(1,3) - Epsilon(2,3,4,-1)*P(-1,4)*P(1,3) + Epsilon(2,3,4,-1)*P(-1,2)*P(1,4) - Epsilon(2,3,4,-1)*P(-1,3)*P(1,4) - Epsilon(1,3,4,-1)*P(-1,2)*P(2,1) - Epsilon(1,3,4,-1)*P(-1,3)*P(2,1) - Epsilon(1,3,4,-1)*P(-1,1)*P(2,3) + Epsilon(1,3,4,-1)*P(-1,4)*P(2,3) + Epsilon(1,3,4,-1)*P(-1,2)*P(2,4) + Epsilon(1,3,4,-1)*P(-1,3)*P(2,4) - Epsilon(1,2,4,-1)*P(-1,2)*P(3,1) - Epsilon(1,2,4,-1)*P(-1,3)*P(3,1) - Epsilon(1,2,4,-1)*P(-1,1)*P(3,2) + Epsilon(1,2,4,-1)*P(-1,4)*P(3,2) + Epsilon(1,2,4,-1)*P(-1,2)*P(3,4) + Epsilon(1,2,4,-1)*P(-1,3)*P(3,4) + Epsilon(1,2,3,-1)*P(-1,2)*P(4,1) - Epsilon(1,2,3,-1)*P(-1,3)*P(4,1) + Epsilon(1,2,3,-1)*P(-1,1)*P(4,2) + Epsilon(1,2,3,-1)*P(-1,4)*P(4,2) - Epsilon(1,2,3,-1)*P(-1,1)*P(4,3) - Epsilon(1,2,3,-1)*P(-1,4)*P(4,3) + Epsilon(3,4,-1,-2)*P(-2,2)*P(-1,1)*Metric(1,2) + Epsilon(3,4,-1,-2)*P(-2,3)*P(-1,1)*Metric(1,2) - Epsilon(3,4,-1,-2)*P(-2,4)*P(-1,2)*Metric(1,2) + Epsilon(2,4,-1,-2)*P(-2,2)*P(-1,1)*Metric(1,3) + Epsilon(2,4,-1,-2)*P(-2,3)*P(-1,1)*Metric(1,3) - Epsilon(2,4,-1,-2)*P(-2,4)*P(-1,3)*Metric(1,3) - Epsilon(2,3,-1,-2)*P(-2,2)*P(-1,1)*Metric(1,4) + Epsilon(2,3,-1,-2)*P(-2,3)*P(-1,1)*Metric(1,4) + Epsilon(2,3,-1,-2)*P(-2,4)*P(-1,2)*Metric(1,4) - Epsilon(2,3,-1,-2)*P(-2,4)*P(-1,3)*Metric(1,4) + Epsilon(1,4,-1,-2)*P(-2,2)*P(-1,1)*Metric(2,3) + Epsilon(1,4,-1,-2)*P(-2,3)*P(-1,1)*Metric(2,3) + Epsilon(1,4,-1,-2)*P(-2,4)*P(-1,2)*Metric(2,3) + Epsilon(1,4,-1,-2)*P(-2,4)*P(-1,3)*Metric(2,3) - Epsilon(1,3,-1,-2)*P(-2,2)*P(-1,1)*Metric(2,4) + Epsilon(1,3,-1,-2)*P(-2,4)*P(-1,2)*Metric(2,4) + Epsilon(1,3,-1,-2)*P(-2,4)*P(-1,3)*Metric(2,4) - Epsilon(1,2,-1,-2)*P(-2,3)*P(-1,1)*Metric(3,4) + Epsilon(1,2,-1,-2)*P(-2,4)*P(-1,2)*Metric(3,4) + Epsilon(1,2,-1,-2)*P(-2,4)*P(-1,3)*Metric(3,4)')

VVVV8 = Lorentz(name = 'VVVV8',
                spins = [ 3, 3, 3, 3 ],
                structure = 'Metric(1,4)*Metric(2,3) + Metric(1,3)*Metric(2,4) - 2*Metric(1,2)*Metric(3,4)')

VVVV9 = Lorentz(name = 'VVVV9',
                spins = [ 3, 3, 3, 3 ],
                structure = 'Metric(1,4)*Metric(2,3) - Metric(1,2)*Metric(3,4)')

VVVV10 = Lorentz(name = 'VVVV10',
                 spins = [ 3, 3, 3, 3 ],
                 structure = 'Metric(1,3)*Metric(2,4) - Metric(1,2)*Metric(3,4)')

VVVV11 = Lorentz(name = 'VVVV11',
                 spins = [ 3, 3, 3, 3 ],
                 structure = 'Metric(1,4)*Metric(2,3) - (Metric(1,3)*Metric(2,4))/2. - (Metric(1,2)*Metric(3,4))/2.')

VVVV12 = Lorentz(name = 'VVVV12',
                 spins = [ 3, 3, 3, 3 ],
                 structure = 'P(3,4)*P(4,1)*Metric(1,2) + P(3,2)*P(4,3)*Metric(1,2) - P(2,4)*P(4,1)*Metric(1,3) - P(2,3)*P(4,2)*Metric(1,3) + P(2,4)*P(3,1)*Metric(1,4) - P(2,1)*P(3,4)*Metric(1,4) + P(1,3)*P(4,2)*Metric(2,3) - P(1,2)*P(4,3)*Metric(2,3) - P(1,4)*P(3,1)*Metric(2,4) - P(1,3)*P(3,2)*Metric(2,4) + P(-1,2)*P(-1,3)*Metric(1,3)*Metric(2,4) + P(-1,1)*P(-1,4)*Metric(1,3)*Metric(2,4) + P(1,4)*P(2,1)*Metric(3,4) + P(1,2)*P(2,3)*Metric(3,4) - P(-1,2)*P(-1,3)*Metric(1,2)*Metric(3,4) - P(-1,1)*P(-1,4)*Metric(1,2)*Metric(3,4)')

VVVV13 = Lorentz(name = 'VVVV13',
                 spins = [ 3, 3, 3, 3 ],
                 structure = 'P(3,4)*P(4,2)*Metric(1,2) + P(3,1)*P(4,3)*Metric(1,2) + P(2,3)*P(4,1)*Metric(1,3) - P(2,1)*P(4,3)*Metric(1,3) - P(2,3)*P(3,1)*Metric(1,4) - P(2,4)*P(3,2)*Metric(1,4) - P(1,3)*P(4,1)*Metric(2,3) - P(1,4)*P(4,2)*Metric(2,3) + P(-1,1)*P(-1,3)*Metric(1,4)*Metric(2,3) + P(-1,2)*P(-1,4)*Metric(1,4)*Metric(2,3) + P(1,4)*P(3,2)*Metric(2,4) - P(1,2)*P(3,4)*Metric(2,4) + P(1,3)*P(2,1)*Metric(3,4) + P(1,2)*P(2,4)*Metric(3,4) - P(-1,1)*P(-1,3)*Metric(1,2)*Metric(3,4) - P(-1,2)*P(-1,4)*Metric(1,2)*Metric(3,4)')

VVVV14 = Lorentz(name = 'VVVV14',
                 spins = [ 3, 3, 3, 3 ],
                 structure = 'P(3,2)*P(4,1)*Metric(1,2) - P(3,1)*P(4,2)*Metric(1,2) + P(3,4)*P(4,2)*Metric(1,2) + P(3,1)*P(4,3)*Metric(1,2) + P(2,3)*P(4,1)*Metric(1,3) + P(2,1)*P(4,2)*Metric(1,3) - P(2,1)*P(4,3)*Metric(1,3) + P(2,4)*P(4,3)*Metric(1,3) - P(2,3)*P(3,1)*Metric(1,4) - P(2,1)*P(3,2)*Metric(1,4) - P(2,4)*P(3,2)*Metric(1,4) - P(2,3)*P(3,4)*Metric(1,4) - P(1,2)*P(4,1)*Metric(2,3) - P(1,3)*P(4,1)*Metric(2,3) - P(1,4)*P(4,2)*Metric(2,3) - P(1,4)*P(4,3)*Metric(2,3) + P(-1,1)*P(-1,2)*Metric(1,4)*Metric(2,3) + P(-1,1)*P(-1,3)*Metric(1,4)*Metric(2,3) + P(-1,2)*P(-1,4)*Metric(1,4)*Metric(2,3) + P(-1,3)*P(-1,4)*Metric(1,4)*Metric(2,3) + P(1,2)*P(3,1)*Metric(2,4) + P(1,4)*P(3,2)*Metric(2,4) - P(1,2)*P(3,4)*Metric(2,4) + P(1,3)*P(3,4)*Metric(2,4) - P(-1,1)*P(-1,2)*Metric(1,3)*Metric(2,4) - P(-1,3)*P(-1,4)*Metric(1,3)*Metric(2,4) + P(1,3)*P(2,1)*Metric(3,4) + P(1,4)*P(2,3)*Metric(3,4) + P(1,2)*P(2,4)*Metric(3,4) - P(1,3)*P(2,4)*Metric(3,4) - P(-1,1)*P(-1,3)*Metric(1,2)*Metric(3,4) - P(-1,2)*P(-1,4)*Metric(1,2)*Metric(3,4)')

VVVV15 = Lorentz(name = 'VVVV15',
                 spins = [ 3, 3, 3, 3 ],
                 structure = 'P(3,4)*P(4,1)*Metric(1,2) + P(3,4)*P(4,2)*Metric(1,2) + P(3,1)*P(4,3)*Metric(1,2) + P(3,2)*P(4,3)*Metric(1,2) + P(2,3)*P(4,1)*Metric(1,3) - P(2,4)*P(4,1)*Metric(1,3) - P(2,3)*P(4,2)*Metric(1,3) - P(2,1)*P(4,3)*Metric(1,3) - P(2,3)*P(3,1)*Metric(1,4) + P(2,4)*P(3,1)*Metric(1,4) - P(2,4)*P(3,2)*Metric(1,4) - P(2,1)*P(3,4)*Metric(1,4) - P(1,3)*P(4,1)*Metric(2,3) + P(1,3)*P(4,2)*Metric(2,3) - P(1,4)*P(4,2)*Metric(2,3) - P(1,2)*P(4,3)*Metric(2,3) + P(-1,1)*P(-1,3)*Metric(1,4)*Metric(2,3) + P(-1,2)*P(-1,4)*Metric(1,4)*Metric(2,3) - P(1,4)*P(3,1)*Metric(2,4) - P(1,3)*P(3,2)*Metric(2,4) + P(1,4)*P(3,2)*Metric(2,4) - P(1,2)*P(3,4)*Metric(2,4) + P(-1,2)*P(-1,3)*Metric(1,3)*Metric(2,4) + P(-1,1)*P(-1,4)*Metric(1,3)*Metric(2,4) + P(1,3)*P(2,1)*Metric(3,4) + P(1,4)*P(2,1)*Metric(3,4) + P(1,2)*P(2,3)*Metric(3,4) + P(1,2)*P(2,4)*Metric(3,4) - P(-1,1)*P(-1,3)*Metric(1,2)*Metric(3,4) - P(-1,2)*P(-1,3)*Metric(1,2)*Metric(3,4) - P(-1,1)*P(-1,4)*Metric(1,2)*Metric(3,4) - P(-1,2)*P(-1,4)*Metric(1,2)*Metric(3,4)')

VVVV16 = Lorentz(name = 'VVVV16',
                 spins = [ 3, 3, 3, 3 ],
                 structure = 'P(3,3)*P(4,1)*Metric(1,2) + 2*P(3,4)*P(4,1)*Metric(1,2) - P(3,3)*P(4,2)*Metric(1,2) - 2*P(3,4)*P(4,2)*Metric(1,2) - 2*P(3,1)*P(4,3)*Metric(1,2) + 2*P(3,2)*P(4,3)*Metric(1,2) - P(3,1)*P(4,4)*Metric(1,2) + P(3,2)*P(4,4)*Metric(1,2) + P(2,2)*P(4,2)*Metric(1,3) + 4*P(2,1)*P(4,3)*Metric(1,3) + 2*P(2,2)*P(4,3)*Metric(1,3) + 2*P(2,1)*P(4,4)*Metric(1,3) + P(2,2)*P(4,4)*Metric(1,3) + P(2,4)*P(4,4)*Metric(1,3) - P(2,2)*P(3,2)*Metric(1,4) - 2*P(2,1)*P(3,3)*Metric(1,4) - P(2,2)*P(3,3)*Metric(1,4) - P(2,3)*P(3,3)*Metric(1,4) - 4*P(2,1)*P(3,4)*Metric(1,4) - 2*P(2,2)*P(3,4)*Metric(1,4) - P(1,1)*P(4,1)*Metric(2,3) - 2*P(1,1)*P(4,3)*Metric(2,3) - 4*P(1,2)*P(4,3)*Metric(2,3) - P(1,1)*P(4,4)*Metric(2,3) - 2*P(1,2)*P(4,4)*Metric(2,3) - P(1,4)*P(4,4)*Metric(2,3) + P(-1,1)**2*Metric(1,4)*Metric(2,3) + P(-1,2)**2*Metric(1,4)*Metric(2,3) + P(-1,3)**2*Metric(1,4)*Metric(2,3) + P(-1,4)**2*Metric(1,4)*Metric(2,3) + P(1,1)*P(3,1)*Metric(2,4) + P(1,1)*P(3,3)*Metric(2,4) + 2*P(1,2)*P(3,3)*Metric(2,4) + P(1,3)*P(3,3)*Metric(2,4) + 2*P(1,1)*P(3,4)*Metric(2,4) + 4*P(1,2)*P(3,4)*Metric(2,4) - P(-1,1)**2*Metric(1,3)*Metric(2,4) - P(-1,2)**2*Metric(1,3)*Metric(2,4) - P(-1,3)**2*Metric(1,3)*Metric(2,4) - P(-1,4)**2*Metric(1,3)*Metric(2,4) - 2*P(1,3)*P(2,1)*Metric(3,4) + 2*P(1,4)*P(2,1)*Metric(3,4) - P(1,3)*P(2,2)*Metric(3,4) + P(1,4)*P(2,2)*Metric(3,4) + P(1,1)*P(2,3)*Metric(3,4) + 2*P(1,2)*P(2,3)*Metric(3,4) - P(1,1)*P(2,4)*Metric(3,4) - 2*P(1,2)*P(2,4)*Metric(3,4) + P(-1,1)*P(-1,3)*Metric(1,2)*Metric(3,4) - P(-1,2)*P(-1,3)*Metric(1,2)*Metric(3,4) - P(-1,1)*P(-1,4)*Metric(1,2)*Metric(3,4) + P(-1,2)*P(-1,4)*Metric(1,2)*Metric(3,4)')

VVVV17 = Lorentz(name = 'VVVV17',
                 spins = [ 3, 3, 3, 3 ],
                 structure = 'P(3,2)*P(4,1)*Metric(1,2) + (P(3,3)*P(4,1)*Metric(1,2))/2. + P(3,1)*P(4,2)*Metric(1,2) + (P(3,3)*P(4,2)*Metric(1,2))/2. + (P(3,3)*P(4,3)*Metric(1,2))/2. + (P(3,1)*P(4,4)*Metric(1,2))/2. + (P(3,2)*P(4,4)*Metric(1,2))/2. + (P(3,3)*P(4,4)*Metric(1,2))/2. + (P(3,4)*P(4,4)*Metric(1,2))/2. - (P(2,2)*P(4,1)*Metric(1,3))/4. - P(2,3)*P(4,1)*Metric(1,3) + (P(2,4)*P(4,1)*Metric(1,3))/2. - (P(2,1)*P(4,2)*Metric(1,3))/2. - (P(2,2)*P(4,2)*Metric(1,3))/4. + (P(2,3)*P(4,2)*Metric(1,3))/2. - (P(2,2)*P(4,3)*Metric(1,3))/4. - (P(2,4)*P(4,3)*Metric(1,3))/2. - (P(2,1)*P(4,4)*Metric(1,3))/4. - (P(2,2)*P(4,4)*Metric(1,3))/4. - (P(2,3)*P(4,4)*Metric(1,3))/4. - (P(2,4)*P(4,4)*Metric(1,3))/4. - (P(2,2)*P(3,1)*Metric(1,4))/4. + (P(2,3)*P(3,1)*Metric(1,4))/2. - P(2,4)*P(3,1)*Metric(1,4) - (P(2,1)*P(3,2)*Metric(1,4))/2. - (P(2,2)*P(3,2)*Metric(1,4))/4. + (P(2,4)*P(3,2)*Metric(1,4))/2. - (P(2,1)*P(3,3)*Metric(1,4))/4. - (P(2,2)*P(3,3)*Metric(1,4))/4. - (P(2,3)*P(3,3)*Metric(1,4))/4. - (P(2,4)*P(3,3)*Metric(1,4))/4. - (P(2,2)*P(3,4)*Metric(1,4))/4. - (P(2,3)*P(3,4)*Metric(1,4))/2. - (P(1,1)*P(4,1)*Metric(2,3))/4. - (P(1,2)*P(4,1)*Metric(2,3))/2. + (P(1,3)*P(4,1)*Metric(2,3))/2. - (P(1,1)*P(4,2)*Metric(2,3))/4. - P(1,3)*P(4,2)*Metric(2,3) + (P(1,4)*P(4,2)*Metric(2,3))/2. - (P(1,1)*P(4,3)*Metric(2,3))/4. - (P(1,4)*P(4,3)*Metric(2,3))/2. - (P(1,1)*P(4,4)*Metric(2,3))/4. - (P(1,2)*P(4,4)*Metric(2,3))/4. - (P(1,3)*P(4,4)*Metric(2,3))/4. - (P(1,4)*P(4,4)*Metric(2,3))/4. + (P(-1,1)**2*Metric(1,4)*Metric(2,3))/4. + (P(-1,1)*P(-1,2)*Metric(1,4)*Metric(2,3))/4. + (P(-1,2)**2*Metric(1,4)*Metric(2,3))/4. - (P(-1,1)*P(-1,3)*Metric(1,4)*Metric(2,3))/4. + (P(-1,3)**2*Metric(1,4)*Metric(2,3))/4. - (P(-1,2)*P(-1,4)*Metric(1,4)*Metric(2,3))/4. + (P(-1,3)*P(-1,4)*Metric(1,4)*Metric(2,3))/4. + (P(-1,4)**2*Metric(1,4)*Metric(2,3))/4. - (P(1,1)*P(3,1)*Metric(2,4))/4. - (P(1,2)*P(3,1)*Metric(2,4))/2. + (P(1,4)*P(3,1)*Metric(2,4))/2. - (P(1,1)*P(3,2)*Metric(2,4))/4. + (P(1,3)*P(3,2)*Metric(2,4))/2. - P(1,4)*P(3,2)*Metric(2,4) - (P(1,1)*P(3,3)*Metric(2,4))/4. - (P(1,2)*P(3,3)*Metric(2,4))/4. - (P(1,3)*P(3,3)*Metric(2,4))/4. - (P(1,4)*P(3,3)*Metric(2,4))/4. - (P(1,1)*P(3,4)*Metric(2,4))/4. - (P(1,3)*P(3,4)*Metric(2,4))/2. + (P(-1,1)**2*Metric(1,3)*Metric(2,4))/4. + (P(-1,1)*P(-1,2)*Metric(1,3)*Metric(2,4))/4. + (P(-1,2)**2*Metric(1,3)*Metric(2,4))/4. - (P(-1,2)*P(-1,3)*Metric(1,3)*Metric(2,4))/4. + (P(-1,3)**2*Metric(1,3)*Metric(2,4))/4. - (P(-1,1)*P(-1,4)*Metric(1,3)*Metric(2,4))/4. + (P(-1,3)*P(-1,4)*Metric(1,3)*Metric(2,4))/4. + (P(-1,4)**2*Metric(1,3)*Metric(2,4))/4. + (P(1,1)*P(2,1)*Metric(3,4))/2. + (P(1,1)*P(2,2)*Metric(3,4))/2. + (P(1,2)*P(2,2)*Metric(3,4))/2. + (P(1,3)*P(2,2)*Metric(3,4))/2. + (P(1,4)*P(2,2)*Metric(3,4))/2. + (P(1,1)*P(2,3)*Metric(3,4))/2. + P(1,4)*P(2,3)*Metric(3,4) + (P(1,1)*P(2,4)*Metric(3,4))/2. + P(1,3)*P(2,4)*Metric(3,4) - (P(-1,1)**2*Metric(1,2)*Metric(3,4))/2. - (P(-1,2)**2*Metric(1,2)*Metric(3,4))/2. - (P(-1,3)**2*Metric(1,2)*Metric(3,4))/2. - (P(-1,4)**2*Metric(1,2)*Metric(3,4))/2.')

VVVV18 = Lorentz(name = 'VVVV18',
                 spins = [ 3, 3, 3, 3 ],
                 structure = 'P(3,2)*P(4,1)*Metric(1,2) + (P(3,3)*P(4,1)*Metric(1,2))/2. + (P(3,3)*P(4,3)*Metric(1,2))/4. + (P(3,2)*P(4,4)*Metric(1,2))/2. + (P(3,3)*P(4,4)*Metric(1,2))/4. + (P(3,4)*P(4,4)*Metric(1,2))/4. - (P(2,2)*P(4,1)*Metric(1,3))/2. - P(2,3)*P(4,1)*Metric(1,3) - (P(2,2)*P(4,2)*Metric(1,3))/4. - (P(2,2)*P(4,4)*Metric(1,3))/4. - (P(2,3)*P(4,4)*Metric(1,3))/2. - (P(2,4)*P(4,4)*Metric(1,3))/4. + (P(2,2)*P(3,1)*Metric(1,4))/4. + (P(2,3)*P(3,1)*Metric(1,4))/2. - (P(2,1)*P(3,2)*Metric(1,4))/2. + (P(2,4)*P(3,2)*Metric(1,4))/2. - (P(2,1)*P(3,3)*Metric(1,4))/4. + (P(2,4)*P(3,3)*Metric(1,4))/4. - (P(2,2)*P(3,4)*Metric(1,4))/4. - (P(2,3)*P(3,4)*Metric(1,4))/2. - (P(1,2)*P(4,1)*Metric(2,3))/2. + (P(1,3)*P(4,1)*Metric(2,3))/2. + (P(1,1)*P(4,2)*Metric(2,3))/4. + (P(1,4)*P(4,2)*Metric(2,3))/2. - (P(1,1)*P(4,3)*Metric(2,3))/4. - (P(1,4)*P(4,3)*Metric(2,3))/2. - (P(1,2)*P(4,4)*Metric(2,3))/4. + (P(1,3)*P(4,4)*Metric(2,3))/4. + (P(-1,1)*P(-1,2)*Metric(1,4)*Metric(2,3))/4. - (P(-1,1)*P(-1,3)*Metric(1,4)*Metric(2,3))/4. - (P(-1,2)*P(-1,4)*Metric(1,4)*Metric(2,3))/4. + (P(-1,3)*P(-1,4)*Metric(1,4)*Metric(2,3))/4. - (P(1,1)*P(3,1)*Metric(2,4))/4. - (P(1,1)*P(3,2)*Metric(2,4))/2. - P(1,4)*P(3,2)*Metric(2,4) - (P(1,1)*P(3,3)*Metric(2,4))/4. - (P(1,3)*P(3,3)*Metric(2,4))/4. - (P(1,4)*P(3,3)*Metric(2,4))/2. + (P(-1,1)**2*Metric(1,3)*Metric(2,4))/4. + (P(-1,2)**2*Metric(1,3)*Metric(2,4))/4. + (P(-1,3)**2*Metric(1,3)*Metric(2,4))/4. + (P(-1,4)**2*Metric(1,3)*Metric(2,4))/4. + (P(1,1)*P(2,1)*Metric(3,4))/4. + (P(1,1)*P(2,2)*Metric(3,4))/4. + (P(1,2)*P(2,2)*Metric(3,4))/4. + (P(1,4)*P(2,2)*Metric(3,4))/2. + (P(1,1)*P(2,3)*Metric(3,4))/2. + P(1,4)*P(2,3)*Metric(3,4) - (P(-1,1)**2*Metric(1,2)*Metric(3,4))/4. - (P(-1,2)**2*Metric(1,2)*Metric(3,4))/4. - (P(-1,3)**2*Metric(1,2)*Metric(3,4))/4. - (P(-1,4)**2*Metric(1,2)*Metric(3,4))/4.')

VVVV19 = Lorentz(name = 'VVVV19',
                 spins = [ 3, 3, 3, 3 ],
                 structure = 'P(3,1)*P(4,2)*Metric(1,2) + (P(3,3)*P(4,2)*Metric(1,2))/2. + (P(3,3)*P(4,3)*Metric(1,2))/4. + (P(3,1)*P(4,4)*Metric(1,2))/2. + (P(3,3)*P(4,4)*Metric(1,2))/4. + (P(3,4)*P(4,4)*Metric(1,2))/4. + (P(2,2)*P(4,1)*Metric(1,3))/4. + (P(2,4)*P(4,1)*Metric(1,3))/2. - (P(2,1)*P(4,2)*Metric(1,3))/2. + (P(2,3)*P(4,2)*Metric(1,3))/2. - (P(2,2)*P(4,3)*Metric(1,3))/4. - (P(2,4)*P(4,3)*Metric(1,3))/2. - (P(2,1)*P(4,4)*Metric(1,3))/4. + (P(2,3)*P(4,4)*Metric(1,3))/4. - (P(2,2)*P(3,1)*Metric(1,4))/2. - P(2,4)*P(3,1)*Metric(1,4) - (P(2,2)*P(3,2)*Metric(1,4))/4. - (P(2,2)*P(3,3)*Metric(1,4))/4. - (P(2,3)*P(3,3)*Metric(1,4))/4. - (P(2,4)*P(3,3)*Metric(1,4))/2. - (P(1,1)*P(4,1)*Metric(2,3))/4. - (P(1,1)*P(4,2)*Metric(2,3))/2. - P(1,3)*P(4,2)*Metric(2,3) - (P(1,1)*P(4,4)*Metric(2,3))/4. - (P(1,3)*P(4,4)*Metric(2,3))/2. - (P(1,4)*P(4,4)*Metric(2,3))/4. + (P(-1,1)**2*Metric(1,4)*Metric(2,3))/4. + (P(-1,2)**2*Metric(1,4)*Metric(2,3))/4. + (P(-1,3)**2*Metric(1,4)*Metric(2,3))/4. + (P(-1,4)**2*Metric(1,4)*Metric(2,3))/4. - (P(1,2)*P(3,1)*Metric(2,4))/2. + (P(1,4)*P(3,1)*Metric(2,4))/2. + (P(1,1)*P(3,2)*Metric(2,4))/4. + (P(1,3)*P(3,2)*Metric(2,4))/2. - (P(1,2)*P(3,3)*Metric(2,4))/4. + (P(1,4)*P(3,3)*Metric(2,4))/4. - (P(1,1)*P(3,4)*Metric(2,4))/4. - (P(1,3)*P(3,4)*Metric(2,4))/2. + (P(-1,1)*P(-1,2)*Metric(1,3)*Metric(2,4))/4. - (P(-1,2)*P(-1,3)*Metric(1,3)*Metric(2,4))/4. - (P(-1,1)*P(-1,4)*Metric(1,3)*Metric(2,4))/4. + (P(-1,3)*P(-1,4)*Metric(1,3)*Metric(2,4))/4. + (P(1,1)*P(2,1)*Metric(3,4))/4. + (P(1,1)*P(2,2)*Metric(3,4))/4. + (P(1,2)*P(2,2)*Metric(3,4))/4. + (P(1,3)*P(2,2)*Metric(3,4))/2. + (P(1,1)*P(2,4)*Metric(3,4))/2. + P(1,3)*P(2,4)*Metric(3,4) - (P(-1,1)**2*Metric(1,2)*Metric(3,4))/4. - (P(-1,2)**2*Metric(1,2)*Metric(3,4))/4. - (P(-1,3)**2*Metric(1,2)*Metric(3,4))/4. - (P(-1,4)**2*Metric(1,2)*Metric(3,4))/4.')

VVVV20 = Lorentz(name = 'VVVV20',
                 spins = [ 3, 3, 3, 3 ],
                 structure = '(P(3,3)*P(4,1)*Metric(1,2))/4. + (P(3,4)*P(4,1)*Metric(1,2))/2. + P(3,1)*P(4,2)*Metric(1,2) + (P(3,3)*P(4,2)*Metric(1,2))/4. - (P(3,4)*P(4,2)*Metric(1,2))/2. - (P(3,1)*P(4,3)*Metric(1,2))/2. + (P(3,2)*P(4,3)*Metric(1,2))/2. + (P(3,3)*P(4,3)*Metric(1,2))/4. + (P(3,1)*P(4,4)*Metric(1,2))/4. + (P(3,2)*P(4,4)*Metric(1,2))/4. + (P(3,3)*P(4,4)*Metric(1,2))/4. + (P(3,4)*P(4,4)*Metric(1,2))/4. + (P(2,2)*P(4,1)*Metric(1,3))/4. + (P(2,4)*P(4,1)*Metric(1,3))/2. - (P(2,1)*P(4,2)*Metric(1,3))/2. + (P(2,2)*P(4,2)*Metric(1,3))/4. + (P(2,3)*P(4,2)*Metric(1,3))/2. + P(2,1)*P(4,3)*Metric(1,3) + (P(2,2)*P(4,3)*Metric(1,3))/4. - (P(2,4)*P(4,3)*Metric(1,3))/2. + (P(2,1)*P(4,4)*Metric(1,3))/4. + (P(2,2)*P(4,4)*Metric(1,3))/4. + (P(2,3)*P(4,4)*Metric(1,3))/4. + (P(2,4)*P(4,4)*Metric(1,3))/4. - (P(2,2)*P(3,1)*Metric(1,4))/2. - P(2,4)*P(3,1)*Metric(1,4) - (P(2,2)*P(3,2)*Metric(1,4))/2. - (P(2,1)*P(3,3)*Metric(1,4))/2. - (P(2,2)*P(3,3)*Metric(1,4))/2. - (P(2,3)*P(3,3)*Metric(1,4))/2. - (P(2,4)*P(3,3)*Metric(1,4))/2. - P(2,1)*P(3,4)*Metric(1,4) - (P(2,2)*P(3,4)*Metric(1,4))/2. - (P(1,1)*P(4,1)*Metric(2,3))/2. - (P(1,1)*P(4,2)*Metric(2,3))/2. - P(1,3)*P(4,2)*Metric(2,3) - (P(1,1)*P(4,3)*Metric(2,3))/2. - P(1,2)*P(4,3)*Metric(2,3) - (P(1,1)*P(4,4)*Metric(2,3))/2. - (P(1,2)*P(4,4)*Metric(2,3))/2. - (P(1,3)*P(4,4)*Metric(2,3))/2. - (P(1,4)*P(4,4)*Metric(2,3))/2. + (P(-1,1)**2*Metric(1,4)*Metric(2,3))/2. + (P(-1,2)**2*Metric(1,4)*Metric(2,3))/2. + (P(-1,3)**2*Metric(1,4)*Metric(2,3))/2. + (P(-1,4)**2*Metric(1,4)*Metric(2,3))/2. + (P(1,1)*P(3,1)*Metric(2,4))/4. - (P(1,2)*P(3,1)*Metric(2,4))/2. + (P(1,4)*P(3,1)*Metric(2,4))/2. + (P(1,1)*P(3,2)*Metric(2,4))/4. + (P(1,3)*P(3,2)*Metric(2,4))/2. + (P(1,1)*P(3,3)*Metric(2,4))/4. + (P(1,2)*P(3,3)*Metric(2,4))/4. + (P(1,3)*P(3,3)*Metric(2,4))/4. + (P(1,4)*P(3,3)*Metric(2,4))/4. + (P(1,1)*P(3,4)*Metric(2,4))/4. + P(1,2)*P(3,4)*Metric(2,4) - (P(1,3)*P(3,4)*Metric(2,4))/2. - (P(-1,1)**2*Metric(1,3)*Metric(2,4))/4. + (P(-1,1)*P(-1,2)*Metric(1,3)*Metric(2,4))/4. - (P(-1,2)**2*Metric(1,3)*Metric(2,4))/4. - (P(-1,2)*P(-1,3)*Metric(1,3)*Metric(2,4))/4. - (P(-1,3)**2*Metric(1,3)*Metric(2,4))/4. - (P(-1,1)*P(-1,4)*Metric(1,3)*Metric(2,4))/4. + (P(-1,3)*P(-1,4)*Metric(1,3)*Metric(2,4))/4. - (P(-1,4)**2*Metric(1,3)*Metric(2,4))/4. + (P(1,1)*P(2,1)*Metric(3,4))/4. - (P(1,3)*P(2,1)*Metric(3,4))/2. + (P(1,4)*P(2,1)*Metric(3,4))/2. + (P(1,1)*P(2,2)*Metric(3,4))/4. + (P(1,2)*P(2,2)*Metric(3,4))/4. + (P(1,3)*P(2,2)*Metric(3,4))/4. + (P(1,4)*P(2,2)*Metric(3,4))/4. + (P(1,1)*P(2,3)*Metric(3,4))/4. + (P(1,2)*P(2,3)*Metric(3,4))/2. + (P(1,1)*P(2,4)*Metric(3,4))/4. - (P(1,2)*P(2,4)*Metric(3,4))/2. + P(1,3)*P(2,4)*Metric(3,4) - (P(-1,1)**2*Metric(1,2)*Metric(3,4))/4. - (P(-1,2)**2*Metric(1,2)*Metric(3,4))/4. + (P(-1,1)*P(-1,3)*Metric(1,2)*Metric(3,4))/4. - (P(-1,2)*P(-1,3)*Metric(1,2)*Metric(3,4))/4. - (P(-1,3)**2*Metric(1,2)*Metric(3,4))/4. - (P(-1,1)*P(-1,4)*Metric(1,2)*Metric(3,4))/4. + (P(-1,2)*P(-1,4)*Metric(1,2)*Metric(3,4))/4. - (P(-1,4)**2*Metric(1,2)*Metric(3,4))/4.')

SSSSS1 = Lorentz(name = 'SSSSS1',
                 spins = [ 1, 1, 1, 1, 1 ],
                 structure = '1')

FFSSS1 = Lorentz(name = 'FFSSS1',
                 spins = [ 2, 2, 1, 1, 1 ],
                 structure = 'ProjM(2,1)')

FFSSS2 = Lorentz(name = 'FFSSS2',
                 spins = [ 2, 2, 1, 1, 1 ],
                 structure = 'ProjM(2,1) - ProjP(2,1)')

FFSSS3 = Lorentz(name = 'FFSSS3',
                 spins = [ 2, 2, 1, 1, 1 ],
                 structure = 'ProjP(2,1)')

FFSSS4 = Lorentz(name = 'FFSSS4',
                 spins = [ 2, 2, 1, 1, 1 ],
                 structure = 'ProjM(2,1) + ProjP(2,1)')

FFVSS1 = Lorentz(name = 'FFVSS1',
                 spins = [ 2, 2, 3, 1, 1 ],
                 structure = 'Gamma(3,2,-1)*ProjM(-1,1)')

FFVSS2 = Lorentz(name = 'FFVSS2',
                 spins = [ 2, 2, 3, 1, 1 ],
                 structure = 'Gamma(3,2,-1)*ProjP(-1,1)')

FFVVS1 = Lorentz(name = 'FFVVS1',
                 spins = [ 2, 2, 3, 3, 1 ],
                 structure = 'Gamma(3,-2,-1)*Gamma(4,2,-2)*ProjM(-1,1)')

FFVVS2 = Lorentz(name = 'FFVVS2',
                 spins = [ 2, 2, 3, 3, 1 ],
                 structure = 'Gamma(3,2,-2)*Gamma(4,-2,-1)*ProjM(-1,1)')

FFVVS3 = Lorentz(name = 'FFVVS3',
                 spins = [ 2, 2, 3, 3, 1 ],
                 structure = 'Gamma(3,2,-2)*Gamma(4,-2,-1)*ProjM(-1,1) - Gamma(3,-2,-1)*Gamma(4,2,-2)*ProjM(-1,1)')

FFVVS4 = Lorentz(name = 'FFVVS4',
                 spins = [ 2, 2, 3, 3, 1 ],
                 structure = 'Gamma(3,2,-1)*Gamma(4,-1,-2)*ProjM(-2,1)')

FFVVS5 = Lorentz(name = 'FFVVS5',
                 spins = [ 2, 2, 3, 3, 1 ],
                 structure = '-(Gamma(3,2,-1)*Gamma(4,-1,-2)*ProjM(-2,1)) + Gamma(3,-2,-1)*Gamma(4,2,-2)*ProjM(-1,1)')

FFVVS6 = Lorentz(name = 'FFVVS6',
                 spins = [ 2, 2, 3, 3, 1 ],
                 structure = 'Gamma(3,-2,-1)*Gamma(4,2,-2)*ProjP(-1,1)')

FFVVS7 = Lorentz(name = 'FFVVS7',
                 spins = [ 2, 2, 3, 3, 1 ],
                 structure = 'Gamma(3,2,-2)*Gamma(4,-2,-1)*ProjP(-1,1)')

FFVVS8 = Lorentz(name = 'FFVVS8',
                 spins = [ 2, 2, 3, 3, 1 ],
                 structure = 'Gamma(3,2,-2)*Gamma(4,-2,-1)*ProjP(-1,1) - Gamma(3,-2,-1)*Gamma(4,2,-2)*ProjP(-1,1)')

FFVVS9 = Lorentz(name = 'FFVVS9',
                 spins = [ 2, 2, 3, 3, 1 ],
                 structure = 'Gamma(3,-1,-2)*Gamma(4,2,-1)*ProjP(-2,1)')

FFVVS10 = Lorentz(name = 'FFVVS10',
                  spins = [ 2, 2, 3, 3, 1 ],
                  structure = '-(Gamma(3,-1,-2)*Gamma(4,2,-1)*ProjP(-2,1)) + Gamma(3,2,-2)*Gamma(4,-2,-1)*ProjP(-1,1)')

VSSSS1 = Lorentz(name = 'VSSSS1',
                 spins = [ 3, 1, 1, 1, 1 ],
                 structure = 'P(1,2) - P(1,3)')

VSSSS2 = Lorentz(name = 'VSSSS2',
                 spins = [ 3, 1, 1, 1, 1 ],
                 structure = 'P(1,2) + P(1,3) - 2*P(1,4)')

VSSSS3 = Lorentz(name = 'VSSSS3',
                 spins = [ 3, 1, 1, 1, 1 ],
                 structure = 'P(1,2) - P(1,3)/2. - P(1,4)/2.')

VSSSS4 = Lorentz(name = 'VSSSS4',
                 spins = [ 3, 1, 1, 1, 1 ],
                 structure = 'P(1,2) + P(1,3) + P(1,4) - 3*P(1,5)')

VSSSS5 = Lorentz(name = 'VSSSS5',
                 spins = [ 3, 1, 1, 1, 1 ],
                 structure = 'P(1,2) + P(1,3) - 2*P(1,5)')

VSSSS6 = Lorentz(name = 'VSSSS6',
                 spins = [ 3, 1, 1, 1, 1 ],
                 structure = 'P(1,3) + P(1,4) - 2*P(1,5)')

VSSSS7 = Lorentz(name = 'VSSSS7',
                 spins = [ 3, 1, 1, 1, 1 ],
                 structure = 'P(1,2) - P(1,5)')

VSSSS8 = Lorentz(name = 'VSSSS8',
                 spins = [ 3, 1, 1, 1, 1 ],
                 structure = 'P(1,2) + P(1,3) - P(1,4) - P(1,5)')

VSSSS9 = Lorentz(name = 'VSSSS9',
                 spins = [ 3, 1, 1, 1, 1 ],
                 structure = 'P(1,4) - P(1,5)')

VSSSS10 = Lorentz(name = 'VSSSS10',
                  spins = [ 3, 1, 1, 1, 1 ],
                  structure = 'P(1,2) - P(1,4)/2. - P(1,5)/2.')

VSSSS11 = Lorentz(name = 'VSSSS11',
                  spins = [ 3, 1, 1, 1, 1 ],
                  structure = 'P(1,3) - P(1,4)/2. - P(1,5)/2.')

VSSSS12 = Lorentz(name = 'VSSSS12',
                  spins = [ 3, 1, 1, 1, 1 ],
                  structure = 'P(1,2) - P(1,3)/3. - P(1,4)/3. - P(1,5)/3.')

VVSSS1 = Lorentz(name = 'VVSSS1',
                 spins = [ 3, 3, 1, 1, 1 ],
                 structure = 'Metric(1,2)')

VVVSS1 = Lorentz(name = 'VVVSS1',
                 spins = [ 3, 3, 3, 1, 1 ],
                 structure = '-(Epsilon(1,2,3,-1)*P(-1,1))')

VVVSS2 = Lorentz(name = 'VVVSS2',
                 spins = [ 3, 3, 3, 1, 1 ],
                 structure = '-(Epsilon(1,2,3,-1)*P(-1,1)) + Epsilon(1,2,3,-1)*P(-1,2)')

VVVSS3 = Lorentz(name = 'VVVSS3',
                 spins = [ 3, 3, 3, 1, 1 ],
                 structure = '-(Epsilon(1,2,3,-1)*P(-1,3))')

VVVSS4 = Lorentz(name = 'VVVSS4',
                 spins = [ 3, 3, 3, 1, 1 ],
                 structure = '-(Epsilon(1,2,3,-1)*P(-1,2)) + Epsilon(1,2,3,-1)*P(-1,3)')

VVVSS5 = Lorentz(name = 'VVVSS5',
                 spins = [ 3, 3, 3, 1, 1 ],
                 structure = '-(Epsilon(1,2,3,-1)*P(-1,1)) - Epsilon(1,2,3,-1)*P(-1,2) - Epsilon(1,2,3,-1)*P(-1,3)')

VVVSS6 = Lorentz(name = 'VVVSS6',
                 spins = [ 3, 3, 3, 1, 1 ],
                 structure = '-(Epsilon(1,2,3,-1)*P(-1,1)) - Epsilon(1,2,3,-1)*P(-1,2) - Epsilon(1,2,3,-1)*P(-1,4) - Epsilon(1,2,3,-1)*P(-1,5)')

VVVSS7 = Lorentz(name = 'VVVSS7',
                 spins = [ 3, 3, 3, 1, 1 ],
                 structure = '-(Epsilon(1,2,3,-1)*P(-1,2)) - Epsilon(1,2,3,-1)*P(-1,3) - Epsilon(1,2,3,-1)*P(-1,4) - Epsilon(1,2,3,-1)*P(-1,5)')

VVVSS8 = Lorentz(name = 'VVVSS8',
                 spins = [ 3, 3, 3, 1, 1 ],
                 structure = '-(Epsilon(1,2,3,-1)*P(-1,1)) - 2*Epsilon(1,2,3,-1)*P(-1,4) - 2*Epsilon(1,2,3,-1)*P(-1,5)')

VVVSS9 = Lorentz(name = 'VVVSS9',
                 spins = [ 3, 3, 3, 1, 1 ],
                 structure = '-(Epsilon(1,2,3,-1)*P(-1,1)) - Epsilon(1,2,3,-1)*P(-1,2) - Epsilon(1,2,3,-1)*P(-1,3) - 3*Epsilon(1,2,3,-1)*P(-1,4) - 3*Epsilon(1,2,3,-1)*P(-1,5)')

VVVSS10 = Lorentz(name = 'VVVSS10',
                  spins = [ 3, 3, 3, 1, 1 ],
                  structure = '-(Epsilon(1,2,3,-1)*P(-1,1)) - 2*Epsilon(1,2,3,-1)*P(-1,2) - 2*Epsilon(1,2,3,-1)*P(-1,3) - 4*Epsilon(1,2,3,-1)*P(-1,4) - 4*Epsilon(1,2,3,-1)*P(-1,5)')

VVVSS11 = Lorentz(name = 'VVVSS11',
                  spins = [ 3, 3, 3, 1, 1 ],
                  structure = 'P(3,1)*Metric(1,2) - P(2,1)*Metric(1,3)')

VVVSS12 = Lorentz(name = 'VVVSS12',
                  spins = [ 3, 3, 3, 1, 1 ],
                  structure = 'P(3,1)*Metric(1,2) - P(3,4)*Metric(1,2) - P(3,5)*Metric(1,2) - P(2,1)*Metric(1,3) + P(2,4)*Metric(1,3) + P(2,5)*Metric(1,3)')

VVVSS13 = Lorentz(name = 'VVVSS13',
                  spins = [ 3, 3, 3, 1, 1 ],
                  structure = 'P(3,1)*Metric(1,2) + P(3,2)*Metric(1,2) - P(2,1)*Metric(1,3) - P(1,2)*Metric(2,3)')

VVVSS14 = Lorentz(name = 'VVVSS14',
                  spins = [ 3, 3, 3, 1, 1 ],
                  structure = 'P(3,1)*Metric(1,2) - P(3,2)*Metric(1,2) - 2*P(2,1)*Metric(1,3) - P(2,2)*Metric(1,3) + P(1,1)*Metric(2,3) + 2*P(1,2)*Metric(2,3)')

VVVSS15 = Lorentz(name = 'VVVSS15',
                  spins = [ 3, 3, 3, 1, 1 ],
                  structure = 'P(3,1)*Metric(1,2) + P(3,2)*Metric(1,2) + P(3,3)*Metric(1,2) + P(2,1)*Metric(1,3) + P(2,2)*Metric(1,3) + P(2,3)*Metric(1,3) - 2*P(1,1)*Metric(2,3) - 2*P(1,2)*Metric(2,3) - 2*P(1,3)*Metric(2,3)')

VVVSS16 = Lorentz(name = 'VVVSS16',
                  spins = [ 3, 3, 3, 1, 1 ],
                  structure = 'P(2,3)*Metric(1,3) - P(1,3)*Metric(2,3)')

VVVSS17 = Lorentz(name = 'VVVSS17',
                  spins = [ 3, 3, 3, 1, 1 ],
                  structure = 'P(3,2)*Metric(1,2) + P(2,3)*Metric(1,3) - P(1,2)*Metric(2,3) - P(1,3)*Metric(2,3)')

VVVSS18 = Lorentz(name = 'VVVSS18',
                  spins = [ 3, 3, 3, 1, 1 ],
                  structure = 'P(3,1)*Metric(1,2) - P(3,2)*Metric(1,2) - P(2,1)*Metric(1,3) + P(2,3)*Metric(1,3) + P(1,2)*Metric(2,3) - P(1,3)*Metric(2,3)')

VVVSS19 = Lorentz(name = 'VVVSS19',
                  spins = [ 3, 3, 3, 1, 1 ],
                  structure = 'P(3,1)*Metric(1,2) - (P(3,2)*Metric(1,2))/3. + (P(3,3)*Metric(1,2))/3. - P(2,1)*Metric(1,3) - (P(2,2)*Metric(1,3))/3. + (P(2,3)*Metric(1,3))/3. + (2*P(1,2)*Metric(2,3))/3. - (2*P(1,3)*Metric(2,3))/3.')

VVVSS20 = Lorentz(name = 'VVVSS20',
                  spins = [ 3, 3, 3, 1, 1 ],
                  structure = 'P(3,1)*Metric(1,2) + P(3,2)*Metric(1,2) - 4*P(3,4)*Metric(1,2) - P(2,1)*Metric(1,3) + 2*P(2,4)*Metric(1,3) - P(1,2)*Metric(2,3) + 2*P(1,4)*Metric(2,3)')

VVVSS21 = Lorentz(name = 'VVVSS21',
                  spins = [ 3, 3, 3, 1, 1 ],
                  structure = 'P(3,1)*Metric(1,2) - 2*P(3,2)*Metric(1,2) + 2*P(3,5)*Metric(1,2) - P(2,1)*Metric(1,3) + 2*P(2,3)*Metric(1,3) - 2*P(2,4)*Metric(1,3) + 2*P(1,2)*Metric(2,3) - 2*P(1,3)*Metric(2,3) + 2*P(1,4)*Metric(2,3) - 2*P(1,5)*Metric(2,3)')

VVVSS22 = Lorentz(name = 'VVVSS22',
                  spins = [ 3, 3, 3, 1, 1 ],
                  structure = 'P(3,1)*Metric(1,2) - P(3,2)*Metric(1,2) - P(2,1)*Metric(1,3) + P(2,4)*Metric(1,3) + P(2,5)*Metric(1,3) + P(1,2)*Metric(2,3) - P(1,4)*Metric(2,3) - P(1,5)*Metric(2,3)')

VVVSS23 = Lorentz(name = 'VVVSS23',
                  spins = [ 3, 3, 3, 1, 1 ],
                  structure = 'P(3,2)*Metric(1,2) - 2*P(3,5)*Metric(1,2) - P(2,3)*Metric(1,3) - P(2,4)*Metric(1,3) + 3*P(2,5)*Metric(1,3) - P(1,2)*Metric(2,3) + P(1,3)*Metric(2,3) + P(1,4)*Metric(2,3) - P(1,5)*Metric(2,3)')

VVVSS24 = Lorentz(name = 'VVVSS24',
                  spins = [ 3, 3, 3, 1, 1 ],
                  structure = 'P(3,1)*Metric(1,2) - (5*P(3,2)*Metric(1,2))/3. - (P(3,3)*Metric(1,2))/3. - (P(3,4)*Metric(1,2))/3. + (P(3,5)*Metric(1,2))/3. - P(2,1)*Metric(1,3) + (P(2,2)*Metric(1,3))/3. + (5*P(2,3)*Metric(1,3))/3. - (P(2,4)*Metric(1,3))/3. + (P(2,5)*Metric(1,3))/3. + (4*P(1,2)*Metric(2,3))/3. - (4*P(1,3)*Metric(2,3))/3. + (2*P(1,4)*Metric(2,3))/3. - (2*P(1,5)*Metric(2,3))/3.')

VVVSS25 = Lorentz(name = 'VVVSS25',
                  spins = [ 3, 3, 3, 1, 1 ],
                  structure = 'P(3,1)*Metric(1,2) + P(3,2)*Metric(1,2) + P(3,3)*Metric(1,2) - P(3,4)*Metric(1,2) + P(3,5)*Metric(1,2) - (P(2,1)*Metric(1,3))/2. - (P(2,2)*Metric(1,3))/2. - (P(2,3)*Metric(1,3))/2. + (P(2,4)*Metric(1,3))/2. - (P(2,5)*Metric(1,3))/2. - (P(1,1)*Metric(2,3))/2. - (P(1,2)*Metric(2,3))/2. - (P(1,3)*Metric(2,3))/2. + (P(1,4)*Metric(2,3))/2. - (P(1,5)*Metric(2,3))/2.')

VVVSS26 = Lorentz(name = 'VVVSS26',
                  spins = [ 3, 3, 3, 1, 1 ],
                  structure = 'P(3,2)*Metric(1,2) + (P(3,3)*Metric(1,2))/2. + (P(3,4)*Metric(1,2))/2. - (P(3,5)*Metric(1,2))/2. - (P(2,2)*Metric(1,3))/2. - P(2,3)*Metric(1,3) - P(2,4)*Metric(1,3) + P(2,5)*Metric(1,3) - (P(1,2)*Metric(2,3))/2. + (P(1,3)*Metric(2,3))/2. + (P(1,4)*Metric(2,3))/2. - (P(1,5)*Metric(2,3))/2.')

VVVSS27 = Lorentz(name = 'VVVSS27',
                  spins = [ 3, 3, 3, 1, 1 ],
                  structure = 'P(3,1)*Metric(1,2) - P(3,2)*Metric(1,2) + (2*P(3,4)*Metric(1,2))/3. - (2*P(3,5)*Metric(1,2))/3. - P(2,1)*Metric(1,3) + P(2,3)*Metric(1,3) - (P(2,4)*Metric(1,3))/3. + (P(2,5)*Metric(1,3))/3. + P(1,2)*Metric(2,3) - P(1,3)*Metric(2,3) - (P(1,4)*Metric(2,3))/3. + (P(1,5)*Metric(2,3))/3.')

VVVSS28 = Lorentz(name = 'VVVSS28',
                  spins = [ 3, 3, 3, 1, 1 ],
                  structure = 'P(3,4)*Metric(1,2) - P(3,5)*Metric(1,2) - (P(2,4)*Metric(1,3))/2. + (P(2,5)*Metric(1,3))/2. - (P(1,4)*Metric(2,3))/2. + (P(1,5)*Metric(2,3))/2.')

VVVSS29 = Lorentz(name = 'VVVSS29',
                  spins = [ 3, 3, 3, 1, 1 ],
                  structure = 'P(3,1)*Metric(1,2) + P(3,2)*Metric(1,2) + P(3,3)*Metric(1,2) + P(3,4)*Metric(1,2) - P(3,5)*Metric(1,2) - (P(2,1)*Metric(1,3))/2. - (P(2,2)*Metric(1,3))/2. - (P(2,3)*Metric(1,3))/2. - (P(2,4)*Metric(1,3))/2. + (P(2,5)*Metric(1,3))/2. - (P(1,1)*Metric(2,3))/2. - (P(1,2)*Metric(2,3))/2. - (P(1,3)*Metric(2,3))/2. - (P(1,4)*Metric(2,3))/2. + (P(1,5)*Metric(2,3))/2.')

VVVSS30 = Lorentz(name = 'VVVSS30',
                  spins = [ 3, 3, 3, 1, 1 ],
                  structure = 'P(3,2)*Metric(1,2) + (P(3,3)*Metric(1,2))/2. - (P(3,4)*Metric(1,2))/2. + (P(3,5)*Metric(1,2))/2. - (P(2,2)*Metric(1,3))/2. - P(2,3)*Metric(1,3) + P(2,4)*Metric(1,3) - P(2,5)*Metric(1,3) - (P(1,2)*Metric(2,3))/2. + (P(1,3)*Metric(2,3))/2. - (P(1,4)*Metric(2,3))/2. + (P(1,5)*Metric(2,3))/2.')

VVVSS31 = Lorentz(name = 'VVVSS31',
                  spins = [ 3, 3, 3, 1, 1 ],
                  structure = 'P(3,1)*Metric(1,2) - P(3,2)*Metric(1,2) + 2*P(3,4)*Metric(1,2) - 2*P(3,5)*Metric(1,2) - P(2,1)*Metric(1,3) + P(2,3)*Metric(1,3) - P(2,4)*Metric(1,3) + P(2,5)*Metric(1,3) + P(1,2)*Metric(2,3) - P(1,3)*Metric(2,3) - P(1,4)*Metric(2,3) + P(1,5)*Metric(2,3)')

VVVSS32 = Lorentz(name = 'VVVSS32',
                  spins = [ 3, 3, 3, 1, 1 ],
                  structure = 'P(3,2)*Metric(1,2) - 2*P(3,4)*Metric(1,2) - P(2,3)*Metric(1,3) + 3*P(2,4)*Metric(1,3) - P(2,5)*Metric(1,3) - P(1,2)*Metric(2,3) + P(1,3)*Metric(2,3) - P(1,4)*Metric(2,3) + P(1,5)*Metric(2,3)')

VVVSS33 = Lorentz(name = 'VVVSS33',
                  spins = [ 3, 3, 3, 1, 1 ],
                  structure = 'P(3,1)*Metric(1,2) + P(3,2)*Metric(1,2) - 4*P(3,5)*Metric(1,2) - P(2,1)*Metric(1,3) + 2*P(2,5)*Metric(1,3) - P(1,2)*Metric(2,3) + 2*P(1,5)*Metric(2,3)')

VVVSS34 = Lorentz(name = 'VVVSS34',
                  spins = [ 3, 3, 3, 1, 1 ],
                  structure = 'P(3,4)*Metric(1,2) - P(3,5)*Metric(1,2) + P(2,4)*Metric(1,3) - P(2,5)*Metric(1,3) - 2*P(1,4)*Metric(2,3) + 2*P(1,5)*Metric(2,3)')

VVVSS35 = Lorentz(name = 'VVVSS35',
                  spins = [ 3, 3, 3, 1, 1 ],
                  structure = 'P(3,2)*Metric(1,2) - P(3,4)*Metric(1,2) - P(3,5)*Metric(1,2) + P(2,3)*Metric(1,3) - P(2,4)*Metric(1,3) - P(2,5)*Metric(1,3) - P(1,2)*Metric(2,3) - P(1,3)*Metric(2,3) + 2*P(1,4)*Metric(2,3) + 2*P(1,5)*Metric(2,3)')

VVVVS1 = Lorentz(name = 'VVVVS1',
                 spins = [ 3, 3, 3, 3, 1 ],
                 structure = 'Metric(1,4)*Metric(2,3) - Metric(1,3)*Metric(2,4)')

VVVVS2 = Lorentz(name = 'VVVVS2',
                 spins = [ 3, 3, 3, 3, 1 ],
                 structure = 'Metric(1,4)*Metric(2,3) + Metric(1,3)*Metric(2,4) - 2*Metric(1,2)*Metric(3,4)')

VVVVS3 = Lorentz(name = 'VVVVS3',
                 spins = [ 3, 3, 3, 3, 1 ],
                 structure = 'Metric(1,4)*Metric(2,3) - Metric(1,2)*Metric(3,4)')

VVVVS4 = Lorentz(name = 'VVVVS4',
                 spins = [ 3, 3, 3, 3, 1 ],
                 structure = 'Metric(1,3)*Metric(2,4) - Metric(1,2)*Metric(3,4)')

VVVVS5 = Lorentz(name = 'VVVVS5',
                 spins = [ 3, 3, 3, 3, 1 ],
                 structure = 'Metric(1,4)*Metric(2,3) - (Metric(1,3)*Metric(2,4))/2. - (Metric(1,2)*Metric(3,4))/2.')

VVVVV1 = Lorentz(name = 'VVVVV1',
                 spins = [ 3, 3, 3, 3, 3 ],
                 structure = 'Epsilon(2,3,4,5)*P(1,3) - Epsilon(1,3,4,5)*P(2,3) - Epsilon(2,4,5,-1)*P(-1,3)*Metric(1,3) - (Epsilon(2,3,5,-1)*P(-1,3)*Metric(1,4))/2. + (Epsilon(2,3,4,-1)*P(-1,3)*Metric(1,5))/2. + Epsilon(1,4,5,-1)*P(-1,3)*Metric(2,3) + (Epsilon(1,3,5,-1)*P(-1,3)*Metric(2,4))/2. - (Epsilon(1,3,4,-1)*P(-1,3)*Metric(2,5))/2.')

VVVVV2 = Lorentz(name = 'VVVVV2',
                 spins = [ 3, 3, 3, 3, 3 ],
                 structure = 'Epsilon(2,3,4,5)*P(1,4) - Epsilon(1,3,4,5)*P(2,4) + (Epsilon(2,4,5,-1)*P(-1,4)*Metric(1,3))/2. + Epsilon(2,3,5,-1)*P(-1,4)*Metric(1,4) + (Epsilon(2,3,4,-1)*P(-1,4)*Metric(1,5))/2. - (Epsilon(1,4,5,-1)*P(-1,4)*Metric(2,3))/2. - Epsilon(1,3,5,-1)*P(-1,4)*Metric(2,4) - (Epsilon(1,3,4,-1)*P(-1,4)*Metric(2,5))/2.')

VVVVV3 = Lorentz(name = 'VVVVV3',
                 spins = [ 3, 3, 3, 3, 3 ],
                 structure = 'Epsilon(2,3,4,5)*P(1,5) - Epsilon(1,3,4,5)*P(2,5) + (Epsilon(2,4,5,-1)*P(-1,5)*Metric(1,3))/2. - (Epsilon(2,3,5,-1)*P(-1,5)*Metric(1,4))/2. - Epsilon(2,3,4,-1)*P(-1,5)*Metric(1,5) - (Epsilon(1,4,5,-1)*P(-1,5)*Metric(2,3))/2. + (Epsilon(1,3,5,-1)*P(-1,5)*Metric(2,4))/2. + Epsilon(1,3,4,-1)*P(-1,5)*Metric(2,5)')

VVVVV4 = Lorentz(name = 'VVVVV4',
                 spins = [ 3, 3, 3, 3, 3 ],
                 structure = 'P(4,3)*Metric(1,5)*Metric(2,3) + (P(4,4)*Metric(1,5)*Metric(2,3))/2. - (P(3,3)*Metric(1,5)*Metric(2,4))/2. - P(3,4)*Metric(1,5)*Metric(2,4) - P(4,3)*Metric(1,3)*Metric(2,5) - (P(4,4)*Metric(1,3)*Metric(2,5))/2. + (P(3,3)*Metric(1,4)*Metric(2,5))/2. + P(3,4)*Metric(1,4)*Metric(2,5) - (P(2,3)*Metric(1,5)*Metric(3,4))/2. + (P(2,4)*Metric(1,5)*Metric(3,4))/2. + (P(1,3)*Metric(2,5)*Metric(3,4))/2. - (P(1,4)*Metric(2,5)*Metric(3,4))/2.')

VVVVV5 = Lorentz(name = 'VVVVV5',
                 spins = [ 3, 3, 3, 3, 3 ],
                 structure = 'Epsilon(1,2,3,5)*P(4,1) - Epsilon(1,2,3,4)*P(5,1) + Epsilon(2,3,5,-1)*P(-1,1)*Metric(1,4) - Epsilon(2,3,4,-1)*P(-1,1)*Metric(1,5) + (Epsilon(1,3,5,-1)*P(-1,1)*Metric(2,4))/2. - (Epsilon(1,3,4,-1)*P(-1,1)*Metric(2,5))/2. - (Epsilon(1,2,5,-1)*P(-1,1)*Metric(3,4))/2. + (Epsilon(1,2,4,-1)*P(-1,1)*Metric(3,5))/2.')

VVVVV6 = Lorentz(name = 'VVVVV6',
                 spins = [ 3, 3, 3, 3, 3 ],
                 structure = 'Epsilon(1,3,4,5)*P(2,1) - Epsilon(1,2,4,5)*P(3,1) + Epsilon(3,4,5,-1)*P(-1,1)*Metric(1,2) - Epsilon(2,4,5,-1)*P(-1,1)*Metric(1,3) - (Epsilon(1,3,5,-1)*P(-1,1)*Metric(2,4))/2. + (Epsilon(1,3,4,-1)*P(-1,1)*Metric(2,5))/2. + (Epsilon(1,2,5,-1)*P(-1,1)*Metric(3,4))/2. - (Epsilon(1,2,4,-1)*P(-1,1)*Metric(3,5))/2.')

VVVVV7 = Lorentz(name = 'VVVVV7',
                 spins = [ 3, 3, 3, 3, 3 ],
                 structure = 'Epsilon(2,3,4,5)*P(1,2) + Epsilon(1,2,4,5)*P(3,2) + Epsilon(3,4,5,-1)*P(-1,2)*Metric(1,2) - (Epsilon(2,3,5,-1)*P(-1,2)*Metric(1,4))/2. + (Epsilon(2,3,4,-1)*P(-1,2)*Metric(1,5))/2. - Epsilon(1,4,5,-1)*P(-1,2)*Metric(2,3) - (Epsilon(1,2,5,-1)*P(-1,2)*Metric(3,4))/2. + (Epsilon(1,2,4,-1)*P(-1,2)*Metric(3,5))/2.')

VVVVV8 = Lorentz(name = 'VVVVV8',
                 spins = [ 3, 3, 3, 3, 3 ],
                 structure = 'Epsilon(1,2,3,5)*P(4,2) - Epsilon(1,2,3,4)*P(5,2) - (Epsilon(2,3,5,-1)*P(-1,2)*Metric(1,4))/2. + (Epsilon(2,3,4,-1)*P(-1,2)*Metric(1,5))/2. - Epsilon(1,3,5,-1)*P(-1,2)*Metric(2,4) + Epsilon(1,3,4,-1)*P(-1,2)*Metric(2,5) - (Epsilon(1,2,5,-1)*P(-1,2)*Metric(3,4))/2. + (Epsilon(1,2,4,-1)*P(-1,2)*Metric(3,5))/2.')

VVVVV9 = Lorentz(name = 'VVVVV9',
                 spins = [ 3, 3, 3, 3, 3 ],
                 structure = 'Epsilon(2,3,4,5)*P(1,2) - Epsilon(2,3,4,5)*P(1,3) + Epsilon(1,3,4,5)*P(2,1) - Epsilon(1,3,4,5)*P(2,3) + Epsilon(1,2,4,5)*P(3,1) - Epsilon(1,2,4,5)*P(3,2) + Epsilon(3,4,5,-1)*P(-1,1)*Metric(1,2) + Epsilon(3,4,5,-1)*P(-1,2)*Metric(1,2) + Epsilon(3,4,5,-1)*P(-1,3)*Metric(1,2) + Epsilon(2,4,5,-1)*P(-1,1)*Metric(1,3) + Epsilon(2,4,5,-1)*P(-1,2)*Metric(1,3) + Epsilon(2,4,5,-1)*P(-1,3)*Metric(1,3) - (Epsilon(2,3,5,-1)*P(-1,2)*Metric(1,4))/2. + (Epsilon(2,3,5,-1)*P(-1,3)*Metric(1,4))/2. + (Epsilon(2,3,4,-1)*P(-1,2)*Metric(1,5))/2. - (Epsilon(2,3,4,-1)*P(-1,3)*Metric(1,5))/2. + Epsilon(1,4,5,-1)*P(-1,1)*Metric(2,3) + Epsilon(1,4,5,-1)*P(-1,2)*Metric(2,3) + Epsilon(1,4,5,-1)*P(-1,3)*Metric(2,3) - (Epsilon(1,3,5,-1)*P(-1,1)*Metric(2,4))/2. + (Epsilon(1,3,5,-1)*P(-1,3)*Metric(2,4))/2. + (Epsilon(1,3,4,-1)*P(-1,1)*Metric(2,5))/2. - (Epsilon(1,3,4,-1)*P(-1,3)*Metric(2,5))/2. - (Epsilon(1,2,5,-1)*P(-1,1)*Metric(3,4))/2. + (Epsilon(1,2,5,-1)*P(-1,2)*Metric(3,4))/2. + (Epsilon(1,2,4,-1)*P(-1,1)*Metric(3,5))/2. - (Epsilon(1,2,4,-1)*P(-1,2)*Metric(3,5))/2.')

VVVVV10 = Lorentz(name = 'VVVVV10',
                  spins = [ 3, 3, 3, 3, 3 ],
                  structure = 'Epsilon(1,2,3,5)*P(4,3) - Epsilon(1,2,3,4)*P(5,3) - (Epsilon(2,3,5,-1)*P(-1,3)*Metric(1,4))/2. + (Epsilon(2,3,4,-1)*P(-1,3)*Metric(1,5))/2. + (Epsilon(1,3,5,-1)*P(-1,3)*Metric(2,4))/2. - (Epsilon(1,3,4,-1)*P(-1,3)*Metric(2,5))/2. + Epsilon(1,2,5,-1)*P(-1,3)*Metric(3,4) - Epsilon(1,2,4,-1)*P(-1,3)*Metric(3,5)')

VVVVV11 = Lorentz(name = 'VVVVV11',
                  spins = [ 3, 3, 3, 3, 3 ],
                  structure = 'Epsilon(2,3,4,5)*P(1,4) + Epsilon(1,2,4,5)*P(3,4) - (Epsilon(3,4,5,-1)*P(-1,4)*Metric(1,2))/2. + Epsilon(2,3,5,-1)*P(-1,4)*Metric(1,4) + (Epsilon(2,3,4,-1)*P(-1,4)*Metric(1,5))/2. + (Epsilon(1,4,5,-1)*P(-1,4)*Metric(2,3))/2. + Epsilon(1,2,5,-1)*P(-1,4)*Metric(3,4) + (Epsilon(1,2,4,-1)*P(-1,4)*Metric(3,5))/2.')

VVVVV12 = Lorentz(name = 'VVVVV12',
                  spins = [ 3, 3, 3, 3, 3 ],
                  structure = 'Epsilon(1,3,4,5)*P(2,4) - Epsilon(1,2,4,5)*P(3,4) - (Epsilon(3,4,5,-1)*P(-1,4)*Metric(1,2))/2. + (Epsilon(2,4,5,-1)*P(-1,4)*Metric(1,3))/2. + Epsilon(1,3,5,-1)*P(-1,4)*Metric(2,4) + (Epsilon(1,3,4,-1)*P(-1,4)*Metric(2,5))/2. - Epsilon(1,2,5,-1)*P(-1,4)*Metric(3,4) - (Epsilon(1,2,4,-1)*P(-1,4)*Metric(3,5))/2.')

VVVVV13 = Lorentz(name = 'VVVVV13',
                  spins = [ 3, 3, 3, 3, 3 ],
                  structure = 'Epsilon(1,3,4,5)*P(2,5) - Epsilon(1,2,4,5)*P(3,5) - (Epsilon(3,4,5,-1)*P(-1,5)*Metric(1,2))/2. + (Epsilon(2,4,5,-1)*P(-1,5)*Metric(1,3))/2. - (Epsilon(1,3,5,-1)*P(-1,5)*Metric(2,4))/2. - Epsilon(1,3,4,-1)*P(-1,5)*Metric(2,5) + (Epsilon(1,2,5,-1)*P(-1,5)*Metric(3,4))/2. + Epsilon(1,2,4,-1)*P(-1,5)*Metric(3,5)')

VVVVV14 = Lorentz(name = 'VVVVV14',
                  spins = [ 3, 3, 3, 3, 3 ],
                  structure = 'Epsilon(2,3,4,5)*P(1,5) + Epsilon(1,2,4,5)*P(3,5) - (Epsilon(3,4,5,-1)*P(-1,5)*Metric(1,2))/2. - (Epsilon(2,3,5,-1)*P(-1,5)*Metric(1,4))/2. - Epsilon(2,3,4,-1)*P(-1,5)*Metric(1,5) + (Epsilon(1,4,5,-1)*P(-1,5)*Metric(2,3))/2. - (Epsilon(1,2,5,-1)*P(-1,5)*Metric(3,4))/2. - Epsilon(1,2,4,-1)*P(-1,5)*Metric(3,5)')

VVVVV15 = Lorentz(name = 'VVVVV15',
                  spins = [ 3, 3, 3, 3, 3 ],
                  structure = 'P(5,1)*Metric(1,3)*Metric(2,4) - P(3,1)*Metric(1,5)*Metric(2,4) - P(4,1)*Metric(1,3)*Metric(2,5) + P(3,1)*Metric(1,4)*Metric(2,5) - P(5,1)*Metric(1,2)*Metric(3,4) + P(2,1)*Metric(1,5)*Metric(3,4) + P(4,1)*Metric(1,2)*Metric(3,5) - P(2,1)*Metric(1,4)*Metric(3,5)')

VVVVV16 = Lorentz(name = 'VVVVV16',
                  spins = [ 3, 3, 3, 3, 3 ],
                  structure = 'P(5,2)*Metric(1,4)*Metric(2,3) - P(5,3)*Metric(1,4)*Metric(2,3) - P(4,2)*Metric(1,5)*Metric(2,3) + P(4,3)*Metric(1,5)*Metric(2,3) + 2*P(3,2)*Metric(1,5)*Metric(2,4) + P(3,3)*Metric(1,5)*Metric(2,4) - 2*P(3,2)*Metric(1,4)*Metric(2,5) - P(3,3)*Metric(1,4)*Metric(2,5) - P(2,2)*Metric(1,5)*Metric(3,4) - 2*P(2,3)*Metric(1,5)*Metric(3,4) + P(2,2)*Metric(1,4)*Metric(3,5) + 2*P(2,3)*Metric(1,4)*Metric(3,5)')

VVVVV17 = Lorentz(name = 'VVVVV17',
                  spins = [ 3, 3, 3, 3, 3 ],
                  structure = 'P(5,2)*Metric(1,4)*Metric(2,3) + (P(5,5)*Metric(1,4)*Metric(2,3))/2. - (P(3,2)*Metric(1,4)*Metric(2,5))/2. + (P(3,5)*Metric(1,4)*Metric(2,5))/2. - P(5,2)*Metric(1,2)*Metric(3,4) - (P(5,5)*Metric(1,2)*Metric(3,4))/2. + (P(2,2)*Metric(1,5)*Metric(3,4))/2. + P(2,5)*Metric(1,5)*Metric(3,4) + (P(1,2)*Metric(2,5)*Metric(3,4))/2. - (P(1,5)*Metric(2,5)*Metric(3,4))/2. - (P(2,2)*Metric(1,4)*Metric(3,5))/2. - P(2,5)*Metric(1,4)*Metric(3,5)')

VVVVV18 = Lorentz(name = 'VVVVV18',
                  spins = [ 3, 3, 3, 3, 3 ],
                  structure = 'P(5,1)*Metric(1,2)*Metric(3,4) - P(5,2)*Metric(1,2)*Metric(3,4) - 2*P(2,1)*Metric(1,5)*Metric(3,4) - P(2,2)*Metric(1,5)*Metric(3,4) + P(1,1)*Metric(2,5)*Metric(3,4) + 2*P(1,2)*Metric(2,5)*Metric(3,4) - P(4,1)*Metric(1,2)*Metric(3,5) + P(4,2)*Metric(1,2)*Metric(3,5) + 2*P(2,1)*Metric(1,4)*Metric(3,5) + P(2,2)*Metric(1,4)*Metric(3,5) - P(1,1)*Metric(2,4)*Metric(3,5) - 2*P(1,2)*Metric(2,4)*Metric(3,5)')

VVVVV19 = Lorentz(name = 'VVVVV19',
                  spins = [ 3, 3, 3, 3, 3 ],
                  structure = 'P(5,2)*Metric(1,4)*Metric(2,3) - P(4,2)*Metric(1,5)*Metric(2,3) + P(3,2)*Metric(1,5)*Metric(2,4) - P(3,2)*Metric(1,4)*Metric(2,5) - P(5,2)*Metric(1,2)*Metric(3,4) + P(1,2)*Metric(2,5)*Metric(3,4) + P(4,2)*Metric(1,2)*Metric(3,5) - P(1,2)*Metric(2,4)*Metric(3,5)')

VVVVV20 = Lorentz(name = 'VVVVV20',
                  spins = [ 3, 3, 3, 3, 3 ],
                  structure = 'P(5,1)*Metric(1,4)*Metric(2,3) - (P(5,2)*Metric(1,4)*Metric(2,3))/2. - (P(5,3)*Metric(1,4)*Metric(2,3))/2. - P(4,1)*Metric(1,5)*Metric(2,3) + (P(4,2)*Metric(1,5)*Metric(2,3))/2. + (P(4,3)*Metric(1,5)*Metric(2,3))/2. - (P(5,1)*Metric(1,3)*Metric(2,4))/2. + P(5,2)*Metric(1,3)*Metric(2,4) - (P(5,3)*Metric(1,3)*Metric(2,4))/2. + (P(3,1)*Metric(1,5)*Metric(2,4))/2. - (P(3,2)*Metric(1,5)*Metric(2,4))/2. + (P(4,1)*Metric(1,3)*Metric(2,5))/2. - P(4,2)*Metric(1,3)*Metric(2,5) + (P(4,3)*Metric(1,3)*Metric(2,5))/2. - (P(3,1)*Metric(1,4)*Metric(2,5))/2. + (P(3,2)*Metric(1,4)*Metric(2,5))/2. - (P(5,1)*Metric(1,2)*Metric(3,4))/2. - (P(5,2)*Metric(1,2)*Metric(3,4))/2. + P(5,3)*Metric(1,2)*Metric(3,4) + (P(2,1)*Metric(1,5)*Metric(3,4))/2. - (P(2,3)*Metric(1,5)*Metric(3,4))/2. + (P(1,2)*Metric(2,5)*Metric(3,4))/2. - (P(1,3)*Metric(2,5)*Metric(3,4))/2. + (P(4,1)*Metric(1,2)*Metric(3,5))/2. + (P(4,2)*Metric(1,2)*Metric(3,5))/2. - P(4,3)*Metric(1,2)*Metric(3,5) - (P(2,1)*Metric(1,4)*Metric(3,5))/2. + (P(2,3)*Metric(1,4)*Metric(3,5))/2. - (P(1,2)*Metric(2,4)*Metric(3,5))/2. + (P(1,3)*Metric(2,4)*Metric(3,5))/2.')

VVVVV21 = Lorentz(name = 'VVVVV21',
                  spins = [ 3, 3, 3, 3, 3 ],
                  structure = 'P(5,3)*Metric(1,4)*Metric(2,3) - P(4,3)*Metric(1,5)*Metric(2,3) - P(5,3)*Metric(1,3)*Metric(2,4) + P(4,3)*Metric(1,3)*Metric(2,5) + P(2,3)*Metric(1,5)*Metric(3,4) - P(1,3)*Metric(2,5)*Metric(3,4) - P(2,3)*Metric(1,4)*Metric(3,5) + P(1,3)*Metric(2,4)*Metric(3,5)')

VVVVV22 = Lorentz(name = 'VVVVV22',
                  spins = [ 3, 3, 3, 3, 3 ],
                  structure = 'P(5,1)*Metric(1,3)*Metric(2,4) - P(5,3)*Metric(1,3)*Metric(2,4) - 2*P(3,1)*Metric(1,5)*Metric(2,4) - P(3,3)*Metric(1,5)*Metric(2,4) - P(4,1)*Metric(1,3)*Metric(2,5) + P(4,3)*Metric(1,3)*Metric(2,5) + 2*P(3,1)*Metric(1,4)*Metric(2,5) + P(3,3)*Metric(1,4)*Metric(2,5) - P(1,1)*Metric(2,5)*Metric(3,4) - 2*P(1,3)*Metric(2,5)*Metric(3,4) + P(1,1)*Metric(2,4)*Metric(3,5) + 2*P(1,3)*Metric(2,4)*Metric(3,5)')

VVVVV23 = Lorentz(name = 'VVVVV23',
                  spins = [ 3, 3, 3, 3, 3 ],
                  structure = 'P(4,2)*Metric(1,5)*Metric(2,3) + (P(4,4)*Metric(1,5)*Metric(2,3))/2. - (P(3,2)*Metric(1,5)*Metric(2,4))/2. + (P(3,4)*Metric(1,5)*Metric(2,4))/2. - (P(2,2)*Metric(1,5)*Metric(3,4))/2. - P(2,4)*Metric(1,5)*Metric(3,4) - P(4,2)*Metric(1,2)*Metric(3,5) - (P(4,4)*Metric(1,2)*Metric(3,5))/2. + (P(2,2)*Metric(1,4)*Metric(3,5))/2. + P(2,4)*Metric(1,4)*Metric(3,5) + (P(1,2)*Metric(2,4)*Metric(3,5))/2. - (P(1,4)*Metric(2,4)*Metric(3,5))/2.')

VVVVV24 = Lorentz(name = 'VVVVV24',
                  spins = [ 3, 3, 3, 3, 3 ],
                  structure = 'P(4,1)*Metric(1,3)*Metric(2,5) + (P(4,4)*Metric(1,3)*Metric(2,5))/2. - (P(3,1)*Metric(1,4)*Metric(2,5))/2. + (P(3,4)*Metric(1,4)*Metric(2,5))/2. - (P(1,1)*Metric(2,5)*Metric(3,4))/2. - P(1,4)*Metric(2,5)*Metric(3,4) - P(4,1)*Metric(1,2)*Metric(3,5) - (P(4,4)*Metric(1,2)*Metric(3,5))/2. + (P(2,1)*Metric(1,4)*Metric(3,5))/2. - (P(2,4)*Metric(1,4)*Metric(3,5))/2. + (P(1,1)*Metric(2,4)*Metric(3,5))/2. + P(1,4)*Metric(2,4)*Metric(3,5)')

VVVVV25 = Lorentz(name = 'VVVVV25',
                  spins = [ 3, 3, 3, 3, 3 ],
                  structure = 'P(5,1)*Metric(1,3)*Metric(2,4) + (P(5,5)*Metric(1,3)*Metric(2,4))/2. - (P(3,1)*Metric(1,5)*Metric(2,4))/2. + (P(3,5)*Metric(1,5)*Metric(2,4))/2. - P(5,1)*Metric(1,2)*Metric(3,4) - (P(5,5)*Metric(1,2)*Metric(3,4))/2. + (P(2,1)*Metric(1,5)*Metric(3,4))/2. - (P(2,5)*Metric(1,5)*Metric(3,4))/2. + (P(1,1)*Metric(2,5)*Metric(3,4))/2. + P(1,5)*Metric(2,5)*Metric(3,4) - (P(1,1)*Metric(2,4)*Metric(3,5))/2. - P(1,5)*Metric(2,4)*Metric(3,5)')

VVVVV26 = Lorentz(name = 'VVVVV26',
                  spins = [ 3, 3, 3, 3, 3 ],
                  structure = 'P(5,3)*Metric(1,4)*Metric(2,3) + (P(5,5)*Metric(1,4)*Metric(2,3))/2. - P(5,3)*Metric(1,3)*Metric(2,4) - (P(5,5)*Metric(1,3)*Metric(2,4))/2. + (P(3,3)*Metric(1,5)*Metric(2,4))/2. + P(3,5)*Metric(1,5)*Metric(2,4) - (P(3,3)*Metric(1,4)*Metric(2,5))/2. - P(3,5)*Metric(1,4)*Metric(2,5) - (P(2,3)*Metric(1,4)*Metric(3,5))/2. + (P(2,5)*Metric(1,4)*Metric(3,5))/2. + (P(1,3)*Metric(2,4)*Metric(3,5))/2. - (P(1,5)*Metric(2,4)*Metric(3,5))/2.')

VVVVV27 = Lorentz(name = 'VVVVV27',
                  spins = [ 3, 3, 3, 3, 3 ],
                  structure = 'Epsilon(1,3,4,5)*P(2,1) + Epsilon(1,2,3,5)*P(4,1) + Epsilon(3,4,5,-1)*P(-1,1)*Metric(1,2) + Epsilon(2,3,5,-1)*P(-1,1)*Metric(1,4) + (Epsilon(1,4,5,-1)*P(-1,1)*Metric(2,3))/2. + (Epsilon(1,3,4,-1)*P(-1,1)*Metric(2,5))/2. - (Epsilon(1,2,5,-1)*P(-1,1)*Metric(3,4))/2. + (Epsilon(1,2,3,-1)*P(-1,1)*Metric(4,5))/2.')

VVVVV28 = Lorentz(name = 'VVVVV28',
                  spins = [ 3, 3, 3, 3, 3 ],
                  structure = 'Epsilon(1,2,4,5)*P(3,1) + Epsilon(1,2,3,4)*P(5,1) + Epsilon(2,4,5,-1)*P(-1,1)*Metric(1,3) + Epsilon(2,3,4,-1)*P(-1,1)*Metric(1,5) + (Epsilon(1,4,5,-1)*P(-1,1)*Metric(2,3))/2. + (Epsilon(1,3,4,-1)*P(-1,1)*Metric(2,5))/2. - (Epsilon(1,2,5,-1)*P(-1,1)*Metric(3,4))/2. + (Epsilon(1,2,3,-1)*P(-1,1)*Metric(4,5))/2.')

VVVVV29 = Lorentz(name = 'VVVVV29',
                  spins = [ 3, 3, 3, 3, 3 ],
                  structure = 'Epsilon(1,2,4,5)*P(3,1) - Epsilon(1,2,3,5)*P(4,1) + Epsilon(2,4,5,-1)*P(-1,1)*Metric(1,3) - Epsilon(2,3,5,-1)*P(-1,1)*Metric(1,4) + (Epsilon(1,4,5,-1)*P(-1,1)*Metric(2,3))/2. - (Epsilon(1,3,5,-1)*P(-1,1)*Metric(2,4))/2. + (Epsilon(1,2,4,-1)*P(-1,1)*Metric(3,5))/2. - (Epsilon(1,2,3,-1)*P(-1,1)*Metric(4,5))/2.')

VVVVV30 = Lorentz(name = 'VVVVV30',
                  spins = [ 3, 3, 3, 3, 3 ],
                  structure = 'Epsilon(1,3,4,5)*P(2,1) - Epsilon(1,2,3,4)*P(5,1) + Epsilon(3,4,5,-1)*P(-1,1)*Metric(1,2) - Epsilon(2,3,4,-1)*P(-1,1)*Metric(1,5) + (Epsilon(1,4,5,-1)*P(-1,1)*Metric(2,3))/2. - (Epsilon(1,3,5,-1)*P(-1,1)*Metric(2,4))/2. + (Epsilon(1,2,4,-1)*P(-1,1)*Metric(3,5))/2. - (Epsilon(1,2,3,-1)*P(-1,1)*Metric(4,5))/2.')

VVVVV31 = Lorentz(name = 'VVVVV31',
                  spins = [ 3, 3, 3, 3, 3 ],
                  structure = 'Epsilon(1,2,4,5)*P(3,2) + Epsilon(1,2,3,4)*P(5,2) - (Epsilon(2,4,5,-1)*P(-1,2)*Metric(1,3))/2. - (Epsilon(2,3,4,-1)*P(-1,2)*Metric(1,5))/2. - Epsilon(1,4,5,-1)*P(-1,2)*Metric(2,3) - Epsilon(1,3,4,-1)*P(-1,2)*Metric(2,5) - (Epsilon(1,2,5,-1)*P(-1,2)*Metric(3,4))/2. + (Epsilon(1,2,3,-1)*P(-1,2)*Metric(4,5))/2.')

VVVVV32 = Lorentz(name = 'VVVVV32',
                  spins = [ 3, 3, 3, 3, 3 ],
                  structure = 'Epsilon(2,3,4,5)*P(1,2) + Epsilon(1,2,3,4)*P(5,2) + Epsilon(3,4,5,-1)*P(-1,2)*Metric(1,2) + (Epsilon(2,4,5,-1)*P(-1,2)*Metric(1,3))/2. - (Epsilon(2,3,5,-1)*P(-1,2)*Metric(1,4))/2. - Epsilon(1,3,4,-1)*P(-1,2)*Metric(2,5) - (Epsilon(1,2,4,-1)*P(-1,2)*Metric(3,5))/2. + (Epsilon(1,2,3,-1)*P(-1,2)*Metric(4,5))/2.')

VVVVV33 = Lorentz(name = 'VVVVV33',
                  spins = [ 3, 3, 3, 3, 3 ],
                  structure = 'Epsilon(2,3,4,5)*P(1,2) - Epsilon(1,2,3,5)*P(4,2) + Epsilon(3,4,5,-1)*P(-1,2)*Metric(1,2) + (Epsilon(2,4,5,-1)*P(-1,2)*Metric(1,3))/2. + (Epsilon(2,3,4,-1)*P(-1,2)*Metric(1,5))/2. + Epsilon(1,3,5,-1)*P(-1,2)*Metric(2,4) + (Epsilon(1,2,5,-1)*P(-1,2)*Metric(3,4))/2. - (Epsilon(1,2,3,-1)*P(-1,2)*Metric(4,5))/2.')

VVVVV34 = Lorentz(name = 'VVVVV34',
                  spins = [ 3, 3, 3, 3, 3 ],
                  structure = 'Epsilon(1,2,4,5)*P(3,2) - Epsilon(1,2,3,5)*P(4,2) - (Epsilon(2,4,5,-1)*P(-1,2)*Metric(1,3))/2. + (Epsilon(2,3,5,-1)*P(-1,2)*Metric(1,4))/2. - Epsilon(1,4,5,-1)*P(-1,2)*Metric(2,3) + Epsilon(1,3,5,-1)*P(-1,2)*Metric(2,4) + (Epsilon(1,2,4,-1)*P(-1,2)*Metric(3,5))/2. - (Epsilon(1,2,3,-1)*P(-1,2)*Metric(4,5))/2.')

VVVVV35 = Lorentz(name = 'VVVVV35',
                  spins = [ 3, 3, 3, 3, 3 ],
                  structure = 'Epsilon(2,3,4,5)*P(1,2) - Epsilon(2,3,4,5)*P(1,5) + Epsilon(1,3,4,5)*P(2,1) - Epsilon(1,3,4,5)*P(2,5) + Epsilon(1,2,3,4)*P(5,1) - Epsilon(1,2,3,4)*P(5,2) + Epsilon(3,4,5,-1)*P(-1,1)*Metric(1,2) + Epsilon(3,4,5,-1)*P(-1,2)*Metric(1,2) + Epsilon(3,4,5,-1)*P(-1,5)*Metric(1,2) + (Epsilon(2,4,5,-1)*P(-1,2)*Metric(1,3))/2. - (Epsilon(2,4,5,-1)*P(-1,5)*Metric(1,3))/2. - (Epsilon(2,3,5,-1)*P(-1,2)*Metric(1,4))/2. + (Epsilon(2,3,5,-1)*P(-1,5)*Metric(1,4))/2. + Epsilon(2,3,4,-1)*P(-1,1)*Metric(1,5) + Epsilon(2,3,4,-1)*P(-1,2)*Metric(1,5) + Epsilon(2,3,4,-1)*P(-1,5)*Metric(1,5) + (Epsilon(1,4,5,-1)*P(-1,1)*Metric(2,3))/2. - (Epsilon(1,4,5,-1)*P(-1,5)*Metric(2,3))/2. - (Epsilon(1,3,5,-1)*P(-1,1)*Metric(2,4))/2. + (Epsilon(1,3,5,-1)*P(-1,5)*Metric(2,4))/2. + Epsilon(1,3,4,-1)*P(-1,1)*Metric(2,5) + Epsilon(1,3,4,-1)*P(-1,2)*Metric(2,5) + Epsilon(1,3,4,-1)*P(-1,5)*Metric(2,5) - (Epsilon(1,2,4,-1)*P(-1,1)*Metric(3,5))/2. + (Epsilon(1,2,4,-1)*P(-1,2)*Metric(3,5))/2. + (Epsilon(1,2,3,-1)*P(-1,1)*Metric(4,5))/2. - (Epsilon(1,2,3,-1)*P(-1,2)*Metric(4,5))/2.')

VVVVV36 = Lorentz(name = 'VVVVV36',
                  spins = [ 3, 3, 3, 3, 3 ],
                  structure = '-(Epsilon(2,3,4,5)*P(1,3)) + Epsilon(2,3,4,5)*P(1,4) - Epsilon(1,3,4,5)*P(2,3) + Epsilon(1,3,4,5)*P(2,4) + Epsilon(1,2,4,5)*P(3,1) - Epsilon(1,2,4,5)*P(3,2) + Epsilon(1,2,3,5)*P(4,1) - Epsilon(1,2,3,5)*P(4,2) + Epsilon(3,4,5,-1)*P(-1,3)*Metric(1,2) - Epsilon(3,4,5,-1)*P(-1,4)*Metric(1,2) + Epsilon(2,4,5,-1)*P(-1,1)*Metric(1,3) + (Epsilon(2,4,5,-1)*P(-1,2)*Metric(1,3))/2. + Epsilon(2,4,5,-1)*P(-1,3)*Metric(1,3) + (Epsilon(2,4,5,-1)*P(-1,4)*Metric(1,3))/2. + Epsilon(2,3,5,-1)*P(-1,1)*Metric(1,4) + (Epsilon(2,3,5,-1)*P(-1,2)*Metric(1,4))/2. + (Epsilon(2,3,5,-1)*P(-1,3)*Metric(1,4))/2. + Epsilon(2,3,5,-1)*P(-1,4)*Metric(1,4) - (Epsilon(2,3,4,-1)*P(-1,3)*Metric(1,5))/2. + (Epsilon(2,3,4,-1)*P(-1,4)*Metric(1,5))/2. + (Epsilon(1,4,5,-1)*P(-1,1)*Metric(2,3))/2. + Epsilon(1,4,5,-1)*P(-1,2)*Metric(2,3) + Epsilon(1,4,5,-1)*P(-1,3)*Metric(2,3) + (Epsilon(1,4,5,-1)*P(-1,4)*Metric(2,3))/2. + (Epsilon(1,3,5,-1)*P(-1,1)*Metric(2,4))/2. + Epsilon(1,3,5,-1)*P(-1,2)*Metric(2,4) + (Epsilon(1,3,5,-1)*P(-1,3)*Metric(2,4))/2. + Epsilon(1,3,5,-1)*P(-1,4)*Metric(2,4) - (Epsilon(1,3,4,-1)*P(-1,3)*Metric(2,5))/2. + (Epsilon(1,3,4,-1)*P(-1,4)*Metric(2,5))/2. - Epsilon(1,2,5,-1)*P(-1,1)*Metric(3,4) + Epsilon(1,2,5,-1)*P(-1,2)*Metric(3,4) + (Epsilon(1,2,4,-1)*P(-1,1)*Metric(3,5))/2. - (Epsilon(1,2,4,-1)*P(-1,2)*Metric(3,5))/2. + (Epsilon(1,2,3,-1)*P(-1,1)*Metric(4,5))/2. - (Epsilon(1,2,3,-1)*P(-1,2)*Metric(4,5))/2.')

VVVVV37 = Lorentz(name = 'VVVVV37',
                  spins = [ 3, 3, 3, 3, 3 ],
                  structure = 'Epsilon(1,3,4,5)*P(2,3) + Epsilon(1,2,3,5)*P(4,3) - (Epsilon(3,4,5,-1)*P(-1,3)*Metric(1,2))/2. - (Epsilon(2,3,5,-1)*P(-1,3)*Metric(1,4))/2. - Epsilon(1,4,5,-1)*P(-1,3)*Metric(2,3) + (Epsilon(1,3,4,-1)*P(-1,3)*Metric(2,5))/2. + Epsilon(1,2,5,-1)*P(-1,3)*Metric(3,4) + (Epsilon(1,2,3,-1)*P(-1,3)*Metric(4,5))/2.')

VVVVV38 = Lorentz(name = 'VVVVV38',
                  spins = [ 3, 3, 3, 3, 3 ],
                  structure = 'Epsilon(2,3,4,5)*P(1,3) + Epsilon(1,2,3,4)*P(5,3) - (Epsilon(3,4,5,-1)*P(-1,3)*Metric(1,2))/2. - Epsilon(2,4,5,-1)*P(-1,3)*Metric(1,3) - (Epsilon(2,3,5,-1)*P(-1,3)*Metric(1,4))/2. + (Epsilon(1,3,4,-1)*P(-1,3)*Metric(2,5))/2. + Epsilon(1,2,4,-1)*P(-1,3)*Metric(3,5) + (Epsilon(1,2,3,-1)*P(-1,3)*Metric(4,5))/2.')

VVVVV39 = Lorentz(name = 'VVVVV39',
                  spins = [ 3, 3, 3, 3, 3 ],
                  structure = 'Epsilon(2,3,4,5)*P(1,3) - Epsilon(1,2,3,5)*P(4,3) - (Epsilon(3,4,5,-1)*P(-1,3)*Metric(1,2))/2. - Epsilon(2,4,5,-1)*P(-1,3)*Metric(1,3) + (Epsilon(2,3,4,-1)*P(-1,3)*Metric(1,5))/2. - (Epsilon(1,3,5,-1)*P(-1,3)*Metric(2,4))/2. - Epsilon(1,2,5,-1)*P(-1,3)*Metric(3,4) - (Epsilon(1,2,3,-1)*P(-1,3)*Metric(4,5))/2.')

VVVVV40 = Lorentz(name = 'VVVVV40',
                  spins = [ 3, 3, 3, 3, 3 ],
                  structure = 'Epsilon(1,3,4,5)*P(2,3) - Epsilon(1,2,3,4)*P(5,3) - (Epsilon(3,4,5,-1)*P(-1,3)*Metric(1,2))/2. + (Epsilon(2,3,4,-1)*P(-1,3)*Metric(1,5))/2. - Epsilon(1,4,5,-1)*P(-1,3)*Metric(2,3) - (Epsilon(1,3,5,-1)*P(-1,3)*Metric(2,4))/2. - Epsilon(1,2,4,-1)*P(-1,3)*Metric(3,5) - (Epsilon(1,2,3,-1)*P(-1,3)*Metric(4,5))/2.')

VVVVV41 = Lorentz(name = 'VVVVV41',
                  spins = [ 3, 3, 3, 3, 3 ],
                  structure = '-(Epsilon(1,3,4,5)*P(2,4)) + Epsilon(1,3,4,5)*P(2,5) - Epsilon(1,2,4,5)*P(3,4) + Epsilon(1,2,4,5)*P(3,5) + Epsilon(1,2,3,5)*P(4,2) - Epsilon(1,2,3,5)*P(4,3) + Epsilon(1,2,3,4)*P(5,2) - Epsilon(1,2,3,4)*P(5,3) + (Epsilon(3,4,5,-1)*P(-1,4)*Metric(1,2))/2. - (Epsilon(3,4,5,-1)*P(-1,5)*Metric(1,2))/2. + (Epsilon(2,4,5,-1)*P(-1,4)*Metric(1,3))/2. - (Epsilon(2,4,5,-1)*P(-1,5)*Metric(1,3))/2. - (Epsilon(2,3,5,-1)*P(-1,2)*Metric(1,4))/2. + (Epsilon(2,3,5,-1)*P(-1,3)*Metric(1,4))/2. - (Epsilon(2,3,4,-1)*P(-1,2)*Metric(1,5))/2. + (Epsilon(2,3,4,-1)*P(-1,3)*Metric(1,5))/2. - Epsilon(1,4,5,-1)*P(-1,4)*Metric(2,3) + Epsilon(1,4,5,-1)*P(-1,5)*Metric(2,3) - Epsilon(1,3,5,-1)*P(-1,2)*Metric(2,4) - (Epsilon(1,3,5,-1)*P(-1,3)*Metric(2,4))/2. - Epsilon(1,3,5,-1)*P(-1,4)*Metric(2,4) - (Epsilon(1,3,5,-1)*P(-1,5)*Metric(2,4))/2. - Epsilon(1,3,4,-1)*P(-1,2)*Metric(2,5) - (Epsilon(1,3,4,-1)*P(-1,3)*Metric(2,5))/2. - (Epsilon(1,3,4,-1)*P(-1,4)*Metric(2,5))/2. - Epsilon(1,3,4,-1)*P(-1,5)*Metric(2,5) - (Epsilon(1,2,5,-1)*P(-1,2)*Metric(3,4))/2. - Epsilon(1,2,5,-1)*P(-1,3)*Metric(3,4) - Epsilon(1,2,5,-1)*P(-1,4)*Metric(3,4) - (Epsilon(1,2,5,-1)*P(-1,5)*Metric(3,4))/2. - (Epsilon(1,2,4,-1)*P(-1,2)*Metric(3,5))/2. - Epsilon(1,2,4,-1)*P(-1,3)*Metric(3,5) - (Epsilon(1,2,4,-1)*P(-1,4)*Metric(3,5))/2. - Epsilon(1,2,4,-1)*P(-1,5)*Metric(3,5) + Epsilon(1,2,3,-1)*P(-1,2)*Metric(4,5) - Epsilon(1,2,3,-1)*P(-1,3)*Metric(4,5)')

VVVVV42 = Lorentz(name = 'VVVVV42',
                  spins = [ 3, 3, 3, 3, 3 ],
                  structure = 'Epsilon(1,3,4,5)*P(2,4) - Epsilon(1,2,3,4)*P(5,4) - (Epsilon(3,4,5,-1)*P(-1,4)*Metric(1,2))/2. + (Epsilon(2,3,4,-1)*P(-1,4)*Metric(1,5))/2. + (Epsilon(1,4,5,-1)*P(-1,4)*Metric(2,3))/2. + Epsilon(1,3,5,-1)*P(-1,4)*Metric(2,4) + (Epsilon(1,2,4,-1)*P(-1,4)*Metric(3,5))/2. + Epsilon(1,2,3,-1)*P(-1,4)*Metric(4,5)')

VVVVV43 = Lorentz(name = 'VVVVV43',
                  spins = [ 3, 3, 3, 3, 3 ],
                  structure = 'Epsilon(1,2,4,5)*P(3,4) + Epsilon(1,2,3,4)*P(5,4) - (Epsilon(2,4,5,-1)*P(-1,4)*Metric(1,3))/2. - (Epsilon(2,3,4,-1)*P(-1,4)*Metric(1,5))/2. + (Epsilon(1,4,5,-1)*P(-1,4)*Metric(2,3))/2. + (Epsilon(1,3,4,-1)*P(-1,4)*Metric(2,5))/2. + Epsilon(1,2,5,-1)*P(-1,4)*Metric(3,4) - Epsilon(1,2,3,-1)*P(-1,4)*Metric(4,5)')

VVVVV44 = Lorentz(name = 'VVVVV44',
                  spins = [ 3, 3, 3, 3, 3 ],
                  structure = 'Epsilon(2,3,4,5)*P(1,4) + Epsilon(1,2,3,4)*P(5,4) - (Epsilon(3,4,5,-1)*P(-1,4)*Metric(1,2))/2. + (Epsilon(2,4,5,-1)*P(-1,4)*Metric(1,3))/2. + Epsilon(2,3,5,-1)*P(-1,4)*Metric(1,4) + (Epsilon(1,3,4,-1)*P(-1,4)*Metric(2,5))/2. - (Epsilon(1,2,4,-1)*P(-1,4)*Metric(3,5))/2. - Epsilon(1,2,3,-1)*P(-1,4)*Metric(4,5)')

VVVVV45 = Lorentz(name = 'VVVVV45',
                  spins = [ 3, 3, 3, 3, 3 ],
                  structure = 'Epsilon(2,3,4,5)*P(1,5) - Epsilon(1,2,3,5)*P(4,5) - (Epsilon(3,4,5,-1)*P(-1,5)*Metric(1,2))/2. + (Epsilon(2,4,5,-1)*P(-1,5)*Metric(1,3))/2. - Epsilon(2,3,4,-1)*P(-1,5)*Metric(1,5) - (Epsilon(1,3,5,-1)*P(-1,5)*Metric(2,4))/2. + (Epsilon(1,2,5,-1)*P(-1,5)*Metric(3,4))/2. + Epsilon(1,2,3,-1)*P(-1,5)*Metric(4,5)')

VVVVV46 = Lorentz(name = 'VVVVV46',
                  spins = [ 3, 3, 3, 3, 3 ],
                  structure = 'Epsilon(1,2,4,5)*P(3,5) - Epsilon(1,2,3,5)*P(4,5) - (Epsilon(2,4,5,-1)*P(-1,5)*Metric(1,3))/2. + (Epsilon(2,3,5,-1)*P(-1,5)*Metric(1,4))/2. + (Epsilon(1,4,5,-1)*P(-1,5)*Metric(2,3))/2. - (Epsilon(1,3,5,-1)*P(-1,5)*Metric(2,4))/2. - Epsilon(1,2,4,-1)*P(-1,5)*Metric(3,5) + Epsilon(1,2,3,-1)*P(-1,5)*Metric(4,5)')

VVVVV47 = Lorentz(name = 'VVVVV47',
                  spins = [ 3, 3, 3, 3, 3 ],
                  structure = 'Epsilon(2,3,4,5)*P(1,4) - Epsilon(2,3,4,5)*P(1,5) + Epsilon(1,2,3,5)*P(4,1) - Epsilon(1,2,3,5)*P(4,5) + Epsilon(1,2,3,4)*P(5,1) - Epsilon(1,2,3,4)*P(5,4) - (Epsilon(3,4,5,-1)*P(-1,4)*Metric(1,2))/2. + (Epsilon(3,4,5,-1)*P(-1,5)*Metric(1,2))/2. + (Epsilon(2,4,5,-1)*P(-1,4)*Metric(1,3))/2. - (Epsilon(2,4,5,-1)*P(-1,5)*Metric(1,3))/2. + Epsilon(2,3,5,-1)*P(-1,1)*Metric(1,4) + Epsilon(2,3,5,-1)*P(-1,4)*Metric(1,4) + Epsilon(2,3,5,-1)*P(-1,5)*Metric(1,4) + Epsilon(2,3,4,-1)*P(-1,1)*Metric(1,5) + Epsilon(2,3,4,-1)*P(-1,4)*Metric(1,5) + Epsilon(2,3,4,-1)*P(-1,5)*Metric(1,5) + (Epsilon(1,3,5,-1)*P(-1,1)*Metric(2,4))/2. - (Epsilon(1,3,5,-1)*P(-1,5)*Metric(2,4))/2. + (Epsilon(1,3,4,-1)*P(-1,1)*Metric(2,5))/2. - (Epsilon(1,3,4,-1)*P(-1,4)*Metric(2,5))/2. - (Epsilon(1,2,5,-1)*P(-1,1)*Metric(3,4))/2. + (Epsilon(1,2,5,-1)*P(-1,5)*Metric(3,4))/2. - (Epsilon(1,2,4,-1)*P(-1,1)*Metric(3,5))/2. + (Epsilon(1,2,4,-1)*P(-1,4)*Metric(3,5))/2. + Epsilon(1,2,3,-1)*P(-1,1)*Metric(4,5) + Epsilon(1,2,3,-1)*P(-1,4)*Metric(4,5) + Epsilon(1,2,3,-1)*P(-1,5)*Metric(4,5)')

VVVVV48 = Lorentz(name = 'VVVVV48',
                  spins = [ 3, 3, 3, 3, 3 ],
                  structure = 'Epsilon(1,2,4,5)*P(3,4) - Epsilon(1,2,4,5)*P(3,5) + Epsilon(1,2,3,5)*P(4,3) - Epsilon(1,2,3,5)*P(4,5) + Epsilon(1,2,3,4)*P(5,3) - Epsilon(1,2,3,4)*P(5,4) - (Epsilon(2,4,5,-1)*P(-1,4)*Metric(1,3))/2. + (Epsilon(2,4,5,-1)*P(-1,5)*Metric(1,3))/2. - (Epsilon(2,3,5,-1)*P(-1,3)*Metric(1,4))/2. + (Epsilon(2,3,5,-1)*P(-1,5)*Metric(1,4))/2. - (Epsilon(2,3,4,-1)*P(-1,3)*Metric(1,5))/2. + (Epsilon(2,3,4,-1)*P(-1,4)*Metric(1,5))/2. + (Epsilon(1,4,5,-1)*P(-1,4)*Metric(2,3))/2. - (Epsilon(1,4,5,-1)*P(-1,5)*Metric(2,3))/2. + (Epsilon(1,3,5,-1)*P(-1,3)*Metric(2,4))/2. - (Epsilon(1,3,5,-1)*P(-1,5)*Metric(2,4))/2. + (Epsilon(1,3,4,-1)*P(-1,3)*Metric(2,5))/2. - (Epsilon(1,3,4,-1)*P(-1,4)*Metric(2,5))/2. + Epsilon(1,2,5,-1)*P(-1,3)*Metric(3,4) + Epsilon(1,2,5,-1)*P(-1,4)*Metric(3,4) + Epsilon(1,2,5,-1)*P(-1,5)*Metric(3,4) + Epsilon(1,2,4,-1)*P(-1,3)*Metric(3,5) + Epsilon(1,2,4,-1)*P(-1,4)*Metric(3,5) + Epsilon(1,2,4,-1)*P(-1,5)*Metric(3,5) + Epsilon(1,2,3,-1)*P(-1,3)*Metric(4,5) + Epsilon(1,2,3,-1)*P(-1,4)*Metric(4,5) + Epsilon(1,2,3,-1)*P(-1,5)*Metric(4,5)')

VVVVV49 = Lorentz(name = 'VVVVV49',
                  spins = [ 3, 3, 3, 3, 3 ],
                  structure = 'Epsilon(1,3,4,5)*P(2,5) + Epsilon(1,2,3,5)*P(4,5) - (Epsilon(3,4,5,-1)*P(-1,5)*Metric(1,2))/2. - (Epsilon(2,3,5,-1)*P(-1,5)*Metric(1,4))/2. + (Epsilon(1,4,5,-1)*P(-1,5)*Metric(2,3))/2. - Epsilon(1,3,4,-1)*P(-1,5)*Metric(2,5) - (Epsilon(1,2,5,-1)*P(-1,5)*Metric(3,4))/2. - Epsilon(1,2,3,-1)*P(-1,5)*Metric(4,5)')

VVVVV50 = Lorentz(name = 'VVVVV50',
                  spins = [ 3, 3, 3, 3, 3 ],
                  structure = 'P(4,3)*Metric(1,5)*Metric(2,3) + (P(4,4)*Metric(1,5)*Metric(2,3))/2. - (P(3,3)*Metric(1,5)*Metric(2,4))/2. - P(3,4)*Metric(1,5)*Metric(2,4) + (P(5,3)*Metric(1,2)*Metric(3,4))/2. - (P(5,4)*Metric(1,2)*Metric(3,4))/2. - (P(2,3)*Metric(1,5)*Metric(3,4))/2. + (P(2,4)*Metric(1,5)*Metric(3,4))/2. - P(4,3)*Metric(1,2)*Metric(3,5) - (P(4,4)*Metric(1,2)*Metric(3,5))/2. + (P(3,3)*Metric(1,2)*Metric(4,5))/2. + P(3,4)*Metric(1,2)*Metric(4,5)')

VVVVV51 = Lorentz(name = 'VVVVV51',
                  spins = [ 3, 3, 3, 3, 3 ],
                  structure = 'P(4,3)*Metric(1,3)*Metric(2,5) + (P(4,4)*Metric(1,3)*Metric(2,5))/2. - (P(3,3)*Metric(1,4)*Metric(2,5))/2. - P(3,4)*Metric(1,4)*Metric(2,5) + (P(5,3)*Metric(1,2)*Metric(3,4))/2. - (P(5,4)*Metric(1,2)*Metric(3,4))/2. - (P(1,3)*Metric(2,5)*Metric(3,4))/2. + (P(1,4)*Metric(2,5)*Metric(3,4))/2. - P(4,3)*Metric(1,2)*Metric(3,5) - (P(4,4)*Metric(1,2)*Metric(3,5))/2. + (P(3,3)*Metric(1,2)*Metric(4,5))/2. + P(3,4)*Metric(1,2)*Metric(4,5)')

VVVVV52 = Lorentz(name = 'VVVVV52',
                  spins = [ 3, 3, 3, 3, 3 ],
                  structure = 'P(5,3)*Metric(1,4)*Metric(2,3) + (P(5,5)*Metric(1,4)*Metric(2,3))/2. - (P(3,3)*Metric(1,4)*Metric(2,5))/2. - P(3,5)*Metric(1,4)*Metric(2,5) - P(5,3)*Metric(1,2)*Metric(3,4) - (P(5,5)*Metric(1,2)*Metric(3,4))/2. + (P(4,3)*Metric(1,2)*Metric(3,5))/2. - (P(4,5)*Metric(1,2)*Metric(3,5))/2. - (P(2,3)*Metric(1,4)*Metric(3,5))/2. + (P(2,5)*Metric(1,4)*Metric(3,5))/2. + (P(3,3)*Metric(1,2)*Metric(4,5))/2. + P(3,5)*Metric(1,2)*Metric(4,5)')

VVVVV53 = Lorentz(name = 'VVVVV53',
                  spins = [ 3, 3, 3, 3, 3 ],
                  structure = 'P(5,3)*Metric(1,3)*Metric(2,4) + (P(5,5)*Metric(1,3)*Metric(2,4))/2. - (P(3,3)*Metric(1,5)*Metric(2,4))/2. - P(3,5)*Metric(1,5)*Metric(2,4) - P(5,3)*Metric(1,2)*Metric(3,4) - (P(5,5)*Metric(1,2)*Metric(3,4))/2. + (P(4,3)*Metric(1,2)*Metric(3,5))/2. - (P(4,5)*Metric(1,2)*Metric(3,5))/2. - (P(1,3)*Metric(2,4)*Metric(3,5))/2. + (P(1,5)*Metric(2,4)*Metric(3,5))/2. + (P(3,3)*Metric(1,2)*Metric(4,5))/2. + P(3,5)*Metric(1,2)*Metric(4,5)')

VVVVV54 = Lorentz(name = 'VVVVV54',
                  spins = [ 3, 3, 3, 3, 3 ],
                  structure = 'P(5,1)*Metric(1,4)*Metric(2,3) - P(4,1)*Metric(1,5)*Metric(2,3) + P(4,1)*Metric(1,3)*Metric(2,5) - P(3,1)*Metric(1,4)*Metric(2,5) - P(5,1)*Metric(1,2)*Metric(3,4) + P(2,1)*Metric(1,5)*Metric(3,4) + P(3,1)*Metric(1,2)*Metric(4,5) - P(2,1)*Metric(1,3)*Metric(4,5)')

VVVVV55 = Lorentz(name = 'VVVVV55',
                  spins = [ 3, 3, 3, 3, 3 ],
                  structure = 'P(5,1)*Metric(1,4)*Metric(2,3) - P(4,1)*Metric(1,5)*Metric(2,3) - P(5,1)*Metric(1,3)*Metric(2,4) + P(3,1)*Metric(1,5)*Metric(2,4) + P(4,1)*Metric(1,2)*Metric(3,5) - P(2,1)*Metric(1,4)*Metric(3,5) - P(3,1)*Metric(1,2)*Metric(4,5) + P(2,1)*Metric(1,3)*Metric(4,5)')

VVVVV56 = Lorentz(name = 'VVVVV56',
                  spins = [ 3, 3, 3, 3, 3 ],
                  structure = 'P(5,4)*Metric(1,3)*Metric(2,4) - P(3,4)*Metric(1,4)*Metric(2,5) - P(5,4)*Metric(1,2)*Metric(3,4) + P(1,4)*Metric(2,5)*Metric(3,4) + P(2,4)*Metric(1,4)*Metric(3,5) - P(1,4)*Metric(2,4)*Metric(3,5) + P(3,4)*Metric(1,2)*Metric(4,5) - P(2,4)*Metric(1,3)*Metric(4,5)')

VVVVV57 = Lorentz(name = 'VVVVV57',
                  spins = [ 3, 3, 3, 3, 3 ],
                  structure = 'P(4,2)*Metric(1,5)*Metric(2,3) + (P(4,4)*Metric(1,5)*Metric(2,3))/2. + (P(5,2)*Metric(1,3)*Metric(2,4))/2. - (P(5,4)*Metric(1,3)*Metric(2,4))/2. - (P(3,2)*Metric(1,5)*Metric(2,4))/2. + (P(3,4)*Metric(1,5)*Metric(2,4))/2. - P(4,2)*Metric(1,3)*Metric(2,5) - (P(4,4)*Metric(1,3)*Metric(2,5))/2. - (P(2,2)*Metric(1,5)*Metric(3,4))/2. - P(2,4)*Metric(1,5)*Metric(3,4) + (P(2,2)*Metric(1,3)*Metric(4,5))/2. + P(2,4)*Metric(1,3)*Metric(4,5)')

VVVVV58 = Lorentz(name = 'VVVVV58',
                  spins = [ 3, 3, 3, 3, 3 ],
                  structure = 'P(5,2)*Metric(1,3)*Metric(2,4) - P(5,4)*Metric(1,3)*Metric(2,4) - 2*P(4,2)*Metric(1,3)*Metric(2,5) - P(4,4)*Metric(1,3)*Metric(2,5) + 2*P(4,2)*Metric(1,2)*Metric(3,5) + P(4,4)*Metric(1,2)*Metric(3,5) - P(2,2)*Metric(1,4)*Metric(3,5) - 2*P(2,4)*Metric(1,4)*Metric(3,5) - P(1,2)*Metric(2,4)*Metric(3,5) + P(1,4)*Metric(2,4)*Metric(3,5) + P(2,2)*Metric(1,3)*Metric(4,5) + 2*P(2,4)*Metric(1,3)*Metric(4,5)')

VVVVV59 = Lorentz(name = 'VVVVV59',
                  spins = [ 3, 3, 3, 3, 3 ],
                  structure = 'P(5,2)*Metric(1,3)*Metric(2,4) + (P(5,5)*Metric(1,3)*Metric(2,4))/2. - (P(4,2)*Metric(1,3)*Metric(2,5))/2. + (P(4,5)*Metric(1,3)*Metric(2,5))/2. - P(5,2)*Metric(1,2)*Metric(3,4) - (P(5,5)*Metric(1,2)*Metric(3,4))/2. + (P(2,2)*Metric(1,5)*Metric(3,4))/2. + P(2,5)*Metric(1,5)*Metric(3,4) + (P(1,2)*Metric(2,5)*Metric(3,4))/2. - (P(1,5)*Metric(2,5)*Metric(3,4))/2. - (P(2,2)*Metric(1,3)*Metric(4,5))/2. - P(2,5)*Metric(1,3)*Metric(4,5)')

VVVVV60 = Lorentz(name = 'VVVVV60',
                  spins = [ 3, 3, 3, 3, 3 ],
                  structure = 'P(5,4)*Metric(1,3)*Metric(2,4) + (P(5,5)*Metric(1,3)*Metric(2,4))/2. - (P(4,4)*Metric(1,3)*Metric(2,5))/2. - P(4,5)*Metric(1,3)*Metric(2,5) - P(5,4)*Metric(1,2)*Metric(3,4) - (P(5,5)*Metric(1,2)*Metric(3,4))/2. + (P(4,4)*Metric(1,2)*Metric(3,5))/2. + P(4,5)*Metric(1,2)*Metric(3,5) + (P(3,4)*Metric(1,2)*Metric(4,5))/2. - (P(3,5)*Metric(1,2)*Metric(4,5))/2. - (P(2,4)*Metric(1,3)*Metric(4,5))/2. + (P(2,5)*Metric(1,3)*Metric(4,5))/2.')

VVVVV61 = Lorentz(name = 'VVVVV61',
                  spins = [ 3, 3, 3, 3, 3 ],
                  structure = 'P(3,5)*Metric(1,5)*Metric(2,4) - P(4,5)*Metric(1,3)*Metric(2,5) - P(2,5)*Metric(1,5)*Metric(3,4) + P(1,5)*Metric(2,5)*Metric(3,4) + P(4,5)*Metric(1,2)*Metric(3,5) - P(1,5)*Metric(2,4)*Metric(3,5) - P(3,5)*Metric(1,2)*Metric(4,5) + P(2,5)*Metric(1,3)*Metric(4,5)')

VVVVV62 = Lorentz(name = 'VVVVV62',
                  spins = [ 3, 3, 3, 3, 3 ],
                  structure = 'P(5,2)*Metric(1,4)*Metric(2,3) + (P(5,5)*Metric(1,4)*Metric(2,3))/2. - P(5,2)*Metric(1,3)*Metric(2,4) - (P(5,5)*Metric(1,3)*Metric(2,4))/2. + (P(4,2)*Metric(1,3)*Metric(2,5))/2. - (P(4,5)*Metric(1,3)*Metric(2,5))/2. - (P(3,2)*Metric(1,4)*Metric(2,5))/2. + (P(3,5)*Metric(1,4)*Metric(2,5))/2. - (P(2,2)*Metric(1,4)*Metric(3,5))/2. - P(2,5)*Metric(1,4)*Metric(3,5) + (P(2,2)*Metric(1,3)*Metric(4,5))/2. + P(2,5)*Metric(1,3)*Metric(4,5)')

VVVVV63 = Lorentz(name = 'VVVVV63',
                  spins = [ 3, 3, 3, 3, 3 ],
                  structure = 'P(5,1)*Metric(1,3)*Metric(2,4) - P(5,4)*Metric(1,3)*Metric(2,4) - P(3,1)*Metric(1,5)*Metric(2,4) + 2*P(3,4)*Metric(1,5)*Metric(2,4) - P(3,5)*Metric(1,5)*Metric(2,4) + P(4,1)*Metric(1,3)*Metric(2,5) - P(4,5)*Metric(1,3)*Metric(2,5) - P(3,1)*Metric(1,4)*Metric(2,5) - P(3,4)*Metric(1,4)*Metric(2,5) + 2*P(3,5)*Metric(1,4)*Metric(2,5) - P(5,1)*Metric(1,2)*Metric(3,4) + P(5,4)*Metric(1,2)*Metric(3,4) + P(2,1)*Metric(1,5)*Metric(3,4) - 2*P(2,4)*Metric(1,5)*Metric(3,4) + P(2,5)*Metric(1,5)*Metric(3,4) + P(1,4)*Metric(2,5)*Metric(3,4) - P(1,5)*Metric(2,5)*Metric(3,4) - P(4,1)*Metric(1,2)*Metric(3,5) + P(4,5)*Metric(1,2)*Metric(3,5) + P(2,1)*Metric(1,4)*Metric(3,5) + P(2,4)*Metric(1,4)*Metric(3,5) - 2*P(2,5)*Metric(1,4)*Metric(3,5) - P(1,4)*Metric(2,4)*Metric(3,5) + P(1,5)*Metric(2,4)*Metric(3,5) + 2*P(3,1)*Metric(1,2)*Metric(4,5) - P(3,4)*Metric(1,2)*Metric(4,5) - P(3,5)*Metric(1,2)*Metric(4,5) - 2*P(2,1)*Metric(1,3)*Metric(4,5) + P(2,4)*Metric(1,3)*Metric(4,5) + P(2,5)*Metric(1,3)*Metric(4,5)')

VVVVV64 = Lorentz(name = 'VVVVV64',
                  spins = [ 3, 3, 3, 3, 3 ],
                  structure = 'P(5,1)*Metric(1,2)*Metric(3,4) - P(5,2)*Metric(1,2)*Metric(3,4) - 2*P(2,1)*Metric(1,5)*Metric(3,4) - P(2,2)*Metric(1,5)*Metric(3,4) + P(1,1)*Metric(2,5)*Metric(3,4) + 2*P(1,2)*Metric(2,5)*Metric(3,4) - P(3,1)*Metric(1,2)*Metric(4,5) + P(3,2)*Metric(1,2)*Metric(4,5) + 2*P(2,1)*Metric(1,3)*Metric(4,5) + P(2,2)*Metric(1,3)*Metric(4,5) - P(1,1)*Metric(2,3)*Metric(4,5) - 2*P(1,2)*Metric(2,3)*Metric(4,5)')

VVVVV65 = Lorentz(name = 'VVVVV65',
                  spins = [ 3, 3, 3, 3, 3 ],
                  structure = 'P(4,1)*Metric(1,2)*Metric(3,5) - P(4,2)*Metric(1,2)*Metric(3,5) - 2*P(2,1)*Metric(1,4)*Metric(3,5) - P(2,2)*Metric(1,4)*Metric(3,5) + P(1,1)*Metric(2,4)*Metric(3,5) + 2*P(1,2)*Metric(2,4)*Metric(3,5) - P(3,1)*Metric(1,2)*Metric(4,5) + P(3,2)*Metric(1,2)*Metric(4,5) + 2*P(2,1)*Metric(1,3)*Metric(4,5) + P(2,2)*Metric(1,3)*Metric(4,5) - P(1,1)*Metric(2,3)*Metric(4,5) - 2*P(1,2)*Metric(2,3)*Metric(4,5)')

VVVVV66 = Lorentz(name = 'VVVVV66',
                  spins = [ 3, 3, 3, 3, 3 ],
                  structure = 'P(4,2)*Metric(1,5)*Metric(2,3) + P(5,2)*Metric(1,3)*Metric(2,4) - P(3,2)*Metric(1,5)*Metric(2,4) - P(4,2)*Metric(1,3)*Metric(2,5) - P(5,2)*Metric(1,2)*Metric(3,4) + P(1,2)*Metric(2,5)*Metric(3,4) + P(3,2)*Metric(1,2)*Metric(4,5) - P(1,2)*Metric(2,3)*Metric(4,5)')

VVVVV67 = Lorentz(name = 'VVVVV67',
                  spins = [ 3, 3, 3, 3, 3 ],
                  structure = 'P(5,2)*Metric(1,4)*Metric(2,3) - P(5,2)*Metric(1,3)*Metric(2,4) + P(4,2)*Metric(1,3)*Metric(2,5) - P(3,2)*Metric(1,4)*Metric(2,5) - P(4,2)*Metric(1,2)*Metric(3,5) + P(1,2)*Metric(2,4)*Metric(3,5) + P(3,2)*Metric(1,2)*Metric(4,5) - P(1,2)*Metric(2,3)*Metric(4,5)')

VVVVV68 = Lorentz(name = 'VVVVV68',
                  spins = [ 3, 3, 3, 3, 3 ],
                  structure = 'P(4,1)*Metric(1,3)*Metric(2,5) - P(4,3)*Metric(1,3)*Metric(2,5) - 2*P(3,1)*Metric(1,4)*Metric(2,5) - P(3,3)*Metric(1,4)*Metric(2,5) + P(1,1)*Metric(2,5)*Metric(3,4) + 2*P(1,3)*Metric(2,5)*Metric(3,4) + 2*P(3,1)*Metric(1,2)*Metric(4,5) + P(3,3)*Metric(1,2)*Metric(4,5) - P(2,1)*Metric(1,3)*Metric(4,5) + P(2,3)*Metric(1,3)*Metric(4,5) - P(1,1)*Metric(2,3)*Metric(4,5) - 2*P(1,3)*Metric(2,3)*Metric(4,5)')

VVVVV69 = Lorentz(name = 'VVVVV69',
                  spins = [ 3, 3, 3, 3, 3 ],
                  structure = 'P(5,1)*Metric(1,3)*Metric(2,4) - P(5,3)*Metric(1,3)*Metric(2,4) - 2*P(3,1)*Metric(1,5)*Metric(2,4) - P(3,3)*Metric(1,5)*Metric(2,4) + P(1,1)*Metric(2,4)*Metric(3,5) + 2*P(1,3)*Metric(2,4)*Metric(3,5) + 2*P(3,1)*Metric(1,2)*Metric(4,5) + P(3,3)*Metric(1,2)*Metric(4,5) - P(2,1)*Metric(1,3)*Metric(4,5) + P(2,3)*Metric(1,3)*Metric(4,5) - P(1,1)*Metric(2,3)*Metric(4,5) - 2*P(1,3)*Metric(2,3)*Metric(4,5)')

VVVVV70 = Lorentz(name = 'VVVVV70',
                  spins = [ 3, 3, 3, 3, 3 ],
                  structure = 'P(5,3)*Metric(1,4)*Metric(2,3) - P(4,3)*Metric(1,3)*Metric(2,5) - P(5,3)*Metric(1,2)*Metric(3,4) + P(1,3)*Metric(2,5)*Metric(3,4) + P(4,3)*Metric(1,2)*Metric(3,5) - P(2,3)*Metric(1,4)*Metric(3,5) + P(2,3)*Metric(1,3)*Metric(4,5) - P(1,3)*Metric(2,3)*Metric(4,5)')

VVVVV71 = Lorentz(name = 'VVVVV71',
                  spins = [ 3, 3, 3, 3, 3 ],
                  structure = 'P(4,3)*Metric(1,5)*Metric(2,3) - P(5,3)*Metric(1,3)*Metric(2,4) + P(5,3)*Metric(1,2)*Metric(3,4) - P(2,3)*Metric(1,5)*Metric(3,4) - P(4,3)*Metric(1,2)*Metric(3,5) + P(1,3)*Metric(2,4)*Metric(3,5) + P(2,3)*Metric(1,3)*Metric(4,5) - P(1,3)*Metric(2,3)*Metric(4,5)')

VVVVV72 = Lorentz(name = 'VVVVV72',
                  spins = [ 3, 3, 3, 3, 3 ],
                  structure = 'P(5,2)*Metric(1,4)*Metric(2,3) - P(5,3)*Metric(1,4)*Metric(2,3) + P(4,2)*Metric(1,5)*Metric(2,3) - P(4,3)*Metric(1,5)*Metric(2,3) + (P(5,1)*Metric(1,3)*Metric(2,4))/4. - (P(5,2)*Metric(1,3)*Metric(2,4))/4. + (P(5,3)*Metric(1,3)*Metric(2,4))/4. - (P(5,4)*Metric(1,3)*Metric(2,4))/4. - (P(3,1)*Metric(1,5)*Metric(2,4))/2. - (P(3,2)*Metric(1,5)*Metric(2,4))/2. + (3*P(3,4)*Metric(1,5)*Metric(2,4))/2. - (P(3,5)*Metric(1,5)*Metric(2,4))/2. + (P(4,1)*Metric(1,3)*Metric(2,5))/4. - (P(4,2)*Metric(1,3)*Metric(2,5))/4. + (P(4,3)*Metric(1,3)*Metric(2,5))/4. - (P(4,5)*Metric(1,3)*Metric(2,5))/4. - (P(3,1)*Metric(1,4)*Metric(2,5))/2. - (P(3,2)*Metric(1,4)*Metric(2,5))/2. - (P(3,4)*Metric(1,4)*Metric(2,5))/2. + (3*P(3,5)*Metric(1,4)*Metric(2,5))/2. - (P(5,1)*Metric(1,2)*Metric(3,4))/4. - (P(5,2)*Metric(1,2)*Metric(3,4))/4. + (P(5,3)*Metric(1,2)*Metric(3,4))/4. + (P(5,4)*Metric(1,2)*Metric(3,4))/4. + (P(2,1)*Metric(1,5)*Metric(3,4))/2. + (P(2,3)*Metric(1,5)*Metric(3,4))/2. - (3*P(2,4)*Metric(1,5)*Metric(3,4))/2. + (P(2,5)*Metric(1,5)*Metric(3,4))/2. - (P(1,2)*Metric(2,5)*Metric(3,4))/4. + (P(1,3)*Metric(2,5)*Metric(3,4))/4. + (P(1,4)*Metric(2,5)*Metric(3,4))/4. - (P(1,5)*Metric(2,5)*Metric(3,4))/4. - (P(4,1)*Metric(1,2)*Metric(3,5))/4. - (P(4,2)*Metric(1,2)*Metric(3,5))/4. + (P(4,3)*Metric(1,2)*Metric(3,5))/4. + (P(4,5)*Metric(1,2)*Metric(3,5))/4. + (P(2,1)*Metric(1,4)*Metric(3,5))/2. + (P(2,3)*Metric(1,4)*Metric(3,5))/2. + (P(2,4)*Metric(1,4)*Metric(3,5))/2. - (3*P(2,5)*Metric(1,4)*Metric(3,5))/2. - (P(1,2)*Metric(2,4)*Metric(3,5))/4. + (P(1,3)*Metric(2,4)*Metric(3,5))/4. - (P(1,4)*Metric(2,4)*Metric(3,5))/4. + (P(1,5)*Metric(2,4)*Metric(3,5))/4. + (3*P(3,1)*Metric(1,2)*Metric(4,5))/2. - (P(3,2)*Metric(1,2)*Metric(4,5))/2. - (P(3,4)*Metric(1,2)*Metric(4,5))/2. - (P(3,5)*Metric(1,2)*Metric(4,5))/2. - (3*P(2,1)*Metric(1,3)*Metric(4,5))/2. + (P(2,3)*Metric(1,3)*Metric(4,5))/2. + (P(2,4)*Metric(1,3)*Metric(4,5))/2. + (P(2,5)*Metric(1,3)*Metric(4,5))/2. + P(1,2)*Metric(2,3)*Metric(4,5) - P(1,3)*Metric(2,3)*Metric(4,5)')

VVVVV73 = Lorentz(name = 'VVVVV73',
                  spins = [ 3, 3, 3, 3, 3 ],
                  structure = 'P(4,2)*Metric(1,5)*Metric(2,3) - P(4,3)*Metric(1,5)*Metric(2,3) - 2*P(3,2)*Metric(1,5)*Metric(2,4) - P(3,3)*Metric(1,5)*Metric(2,4) + P(2,2)*Metric(1,5)*Metric(3,4) + 2*P(2,3)*Metric(1,5)*Metric(3,4) + 2*P(3,2)*Metric(1,2)*Metric(4,5) + P(3,3)*Metric(1,2)*Metric(4,5) - P(2,2)*Metric(1,3)*Metric(4,5) - 2*P(2,3)*Metric(1,3)*Metric(4,5) - P(1,2)*Metric(2,3)*Metric(4,5) + P(1,3)*Metric(2,3)*Metric(4,5)')

VVVVV74 = Lorentz(name = 'VVVVV74',
                  spins = [ 3, 3, 3, 3, 3 ],
                  structure = 'P(5,2)*Metric(1,4)*Metric(2,3) - P(5,3)*Metric(1,4)*Metric(2,3) - 2*P(3,2)*Metric(1,4)*Metric(2,5) - P(3,3)*Metric(1,4)*Metric(2,5) + P(2,2)*Metric(1,4)*Metric(3,5) + 2*P(2,3)*Metric(1,4)*Metric(3,5) + 2*P(3,2)*Metric(1,2)*Metric(4,5) + P(3,3)*Metric(1,2)*Metric(4,5) - P(2,2)*Metric(1,3)*Metric(4,5) - 2*P(2,3)*Metric(1,3)*Metric(4,5) - P(1,2)*Metric(2,3)*Metric(4,5) + P(1,3)*Metric(2,3)*Metric(4,5)')

VVVVV75 = Lorentz(name = 'VVVVV75',
                  spins = [ 3, 3, 3, 3, 3 ],
                  structure = 'P(5,4)*Metric(1,4)*Metric(2,3) - P(3,4)*Metric(1,5)*Metric(2,4) - P(5,4)*Metric(1,2)*Metric(3,4) + P(2,4)*Metric(1,5)*Metric(3,4) - P(2,4)*Metric(1,4)*Metric(3,5) + P(1,4)*Metric(2,4)*Metric(3,5) + P(3,4)*Metric(1,2)*Metric(4,5) - P(1,4)*Metric(2,3)*Metric(4,5)')

VVVVV76 = Lorentz(name = 'VVVVV76',
                  spins = [ 3, 3, 3, 3, 3 ],
                  structure = 'P(5,4)*Metric(1,4)*Metric(2,3) - P(5,4)*Metric(1,3)*Metric(2,4) + P(3,4)*Metric(1,5)*Metric(2,4) - P(3,4)*Metric(1,4)*Metric(2,5) - P(2,4)*Metric(1,5)*Metric(3,4) + P(1,4)*Metric(2,5)*Metric(3,4) + P(2,4)*Metric(1,3)*Metric(4,5) - P(1,4)*Metric(2,3)*Metric(4,5)')

VVVVV77 = Lorentz(name = 'VVVVV77',
                  spins = [ 3, 3, 3, 3, 3 ],
                  structure = 'P(5,1)*Metric(1,4)*Metric(2,3) + P(5,2)*Metric(1,4)*Metric(2,3) - P(5,3)*Metric(1,4)*Metric(2,3) - P(5,4)*Metric(1,4)*Metric(2,3) - P(4,1)*Metric(1,5)*Metric(2,3) + P(4,3)*Metric(1,5)*Metric(2,3) + P(5,1)*Metric(1,3)*Metric(2,4) + P(5,2)*Metric(1,3)*Metric(2,4) - P(5,3)*Metric(1,3)*Metric(2,4) - P(5,4)*Metric(1,3)*Metric(2,4) - P(3,1)*Metric(1,5)*Metric(2,4) + P(3,4)*Metric(1,5)*Metric(2,4) - P(4,2)*Metric(1,3)*Metric(2,5) + P(4,3)*Metric(1,3)*Metric(2,5) - P(3,2)*Metric(1,4)*Metric(2,5) + P(3,4)*Metric(1,4)*Metric(2,5) - 2*P(5,1)*Metric(1,2)*Metric(3,4) - 2*P(5,2)*Metric(1,2)*Metric(3,4) + 2*P(5,3)*Metric(1,2)*Metric(3,4) + 2*P(5,4)*Metric(1,2)*Metric(3,4) + 2*P(2,1)*Metric(1,5)*Metric(3,4) - P(2,3)*Metric(1,5)*Metric(3,4) - P(2,4)*Metric(1,5)*Metric(3,4) + 2*P(1,2)*Metric(2,5)*Metric(3,4) - P(1,3)*Metric(2,5)*Metric(3,4) - P(1,4)*Metric(2,5)*Metric(3,4) + P(4,1)*Metric(1,2)*Metric(3,5) + P(4,2)*Metric(1,2)*Metric(3,5) - 2*P(4,3)*Metric(1,2)*Metric(3,5) - P(2,1)*Metric(1,4)*Metric(3,5) + P(2,3)*Metric(1,4)*Metric(3,5) - P(1,2)*Metric(2,4)*Metric(3,5) + P(1,3)*Metric(2,4)*Metric(3,5) + P(3,1)*Metric(1,2)*Metric(4,5) + P(3,2)*Metric(1,2)*Metric(4,5) - 2*P(3,4)*Metric(1,2)*Metric(4,5) - P(2,1)*Metric(1,3)*Metric(4,5) + P(2,4)*Metric(1,3)*Metric(4,5) - P(1,2)*Metric(2,3)*Metric(4,5) + P(1,4)*Metric(2,3)*Metric(4,5)')

VVVVV78 = Lorentz(name = 'VVVVV78',
                  spins = [ 3, 3, 3, 3, 3 ],
                  structure = 'P(5,1)*Metric(1,4)*Metric(2,3) - P(5,4)*Metric(1,4)*Metric(2,3) - 2*P(4,1)*Metric(1,5)*Metric(2,3) - P(4,4)*Metric(1,5)*Metric(2,3) + 2*P(4,1)*Metric(1,3)*Metric(2,5) + P(4,4)*Metric(1,3)*Metric(2,5) - P(3,1)*Metric(1,4)*Metric(2,5) + P(3,4)*Metric(1,4)*Metric(2,5) - P(1,1)*Metric(2,5)*Metric(3,4) - 2*P(1,4)*Metric(2,5)*Metric(3,4) + P(1,1)*Metric(2,3)*Metric(4,5) + 2*P(1,4)*Metric(2,3)*Metric(4,5)')

VVVVV79 = Lorentz(name = 'VVVVV79',
                  spins = [ 3, 3, 3, 3, 3 ],
                  structure = 'P(5,1)*Metric(1,4)*Metric(2,3) - P(5,4)*Metric(1,4)*Metric(2,3) - 2*P(4,1)*Metric(1,5)*Metric(2,3) - P(4,4)*Metric(1,5)*Metric(2,3) + 2*P(4,1)*Metric(1,2)*Metric(3,5) + P(4,4)*Metric(1,2)*Metric(3,5) - P(2,1)*Metric(1,4)*Metric(3,5) + P(2,4)*Metric(1,4)*Metric(3,5) - P(1,1)*Metric(2,4)*Metric(3,5) - 2*P(1,4)*Metric(2,4)*Metric(3,5) + P(1,1)*Metric(2,3)*Metric(4,5) + 2*P(1,4)*Metric(2,3)*Metric(4,5)')

VVVVV80 = Lorentz(name = 'VVVVV80',
                  spins = [ 3, 3, 3, 3, 3 ],
                  structure = 'P(5,1)*Metric(1,4)*Metric(2,3) - P(5,2)*Metric(1,4)*Metric(2,3) - P(5,3)*Metric(1,4)*Metric(2,3) + P(5,4)*Metric(1,4)*Metric(2,3) + P(4,1)*Metric(1,5)*Metric(2,3) - P(4,2)*Metric(1,5)*Metric(2,3) - P(4,3)*Metric(1,5)*Metric(2,3) + P(4,5)*Metric(1,5)*Metric(2,3) - (3*P(5,1)*Metric(1,3)*Metric(2,4))/2. - (P(5,2)*Metric(1,3)*Metric(2,4))/2. + (5*P(5,3)*Metric(1,3)*Metric(2,4))/2. - (P(5,4)*Metric(1,3)*Metric(2,4))/2. + (3*P(3,1)*Metric(1,5)*Metric(2,4))/2. + (P(3,2)*Metric(1,5)*Metric(2,4))/2. + (P(3,4)*Metric(1,5)*Metric(2,4))/2. - (5*P(3,5)*Metric(1,5)*Metric(2,4))/2. - (3*P(4,1)*Metric(1,3)*Metric(2,5))/2. - (P(4,2)*Metric(1,3)*Metric(2,5))/2. + (5*P(4,3)*Metric(1,3)*Metric(2,5))/2. - (P(4,5)*Metric(1,3)*Metric(2,5))/2. + (3*P(3,1)*Metric(1,4)*Metric(2,5))/2. + (P(3,2)*Metric(1,4)*Metric(2,5))/2. - (5*P(3,4)*Metric(1,4)*Metric(2,5))/2. + (P(3,5)*Metric(1,4)*Metric(2,5))/2. - (3*P(5,1)*Metric(1,2)*Metric(3,4))/2. + (5*P(5,2)*Metric(1,2)*Metric(3,4))/2. - (P(5,3)*Metric(1,2)*Metric(3,4))/2. - (P(5,4)*Metric(1,2)*Metric(3,4))/2. + (3*P(2,1)*Metric(1,5)*Metric(3,4))/2. + (P(2,3)*Metric(1,5)*Metric(3,4))/2. + (P(2,4)*Metric(1,5)*Metric(3,4))/2. - (5*P(2,5)*Metric(1,5)*Metric(3,4))/2. - 2*P(1,2)*Metric(2,5)*Metric(3,4) - 2*P(1,3)*Metric(2,5)*Metric(3,4) + 2*P(1,4)*Metric(2,5)*Metric(3,4) + 2*P(1,5)*Metric(2,5)*Metric(3,4) - (3*P(4,1)*Metric(1,2)*Metric(3,5))/2. + (5*P(4,2)*Metric(1,2)*Metric(3,5))/2. - (P(4,3)*Metric(1,2)*Metric(3,5))/2. - (P(4,5)*Metric(1,2)*Metric(3,5))/2. + (3*P(2,1)*Metric(1,4)*Metric(3,5))/2. + (P(2,3)*Metric(1,4)*Metric(3,5))/2. - (5*P(2,4)*Metric(1,4)*Metric(3,5))/2. + (P(2,5)*Metric(1,4)*Metric(3,5))/2. - 2*P(1,2)*Metric(2,4)*Metric(3,5) - 2*P(1,3)*Metric(2,4)*Metric(3,5) + 2*P(1,4)*Metric(2,4)*Metric(3,5) + 2*P(1,5)*Metric(2,4)*Metric(3,5) - P(3,1)*Metric(1,2)*Metric(4,5) - P(3,2)*Metric(1,2)*Metric(4,5) + P(3,4)*Metric(1,2)*Metric(4,5) + P(3,5)*Metric(1,2)*Metric(4,5) - P(2,1)*Metric(1,3)*Metric(4,5) - P(2,3)*Metric(1,3)*Metric(4,5) + P(2,4)*Metric(1,3)*Metric(4,5) + P(2,5)*Metric(1,3)*Metric(4,5) + 2*P(1,2)*Metric(2,3)*Metric(4,5) + 2*P(1,3)*Metric(2,3)*Metric(4,5) - 2*P(1,4)*Metric(2,3)*Metric(4,5) - 2*P(1,5)*Metric(2,3)*Metric(4,5)')

VVVVV81 = Lorentz(name = 'VVVVV81',
                  spins = [ 3, 3, 3, 3, 3 ],
                  structure = 'P(5,1)*Metric(1,4)*Metric(2,3) - P(5,2)*Metric(1,4)*Metric(2,3) - P(5,3)*Metric(1,4)*Metric(2,3) + P(5,4)*Metric(1,4)*Metric(2,3) + P(4,1)*Metric(1,5)*Metric(2,3) - P(4,2)*Metric(1,5)*Metric(2,3) - P(4,3)*Metric(1,5)*Metric(2,3) + P(4,5)*Metric(1,5)*Metric(2,3) + P(5,1)*Metric(1,3)*Metric(2,4) - P(5,2)*Metric(1,3)*Metric(2,4) + P(5,3)*Metric(1,3)*Metric(2,4) - P(5,4)*Metric(1,3)*Metric(2,4) + P(3,1)*Metric(1,5)*Metric(2,4) - P(3,2)*Metric(1,5)*Metric(2,4) - P(3,4)*Metric(1,5)*Metric(2,4) + P(3,5)*Metric(1,5)*Metric(2,4) + P(4,1)*Metric(1,3)*Metric(2,5) - P(4,2)*Metric(1,3)*Metric(2,5) + P(4,3)*Metric(1,3)*Metric(2,5) - P(4,5)*Metric(1,3)*Metric(2,5) + P(3,1)*Metric(1,4)*Metric(2,5) - P(3,2)*Metric(1,4)*Metric(2,5) + P(3,4)*Metric(1,4)*Metric(2,5) - P(3,5)*Metric(1,4)*Metric(2,5) - 4*P(5,1)*Metric(1,2)*Metric(3,4) + 4*P(5,2)*Metric(1,2)*Metric(3,4) + 2*P(2,1)*Metric(1,5)*Metric(3,4) + 2*P(2,3)*Metric(1,5)*Metric(3,4) + 2*P(2,4)*Metric(1,5)*Metric(3,4) - 6*P(2,5)*Metric(1,5)*Metric(3,4) - 2*P(1,2)*Metric(2,5)*Metric(3,4) - 2*P(1,3)*Metric(2,5)*Metric(3,4) - 2*P(1,4)*Metric(2,5)*Metric(3,4) + 6*P(1,5)*Metric(2,5)*Metric(3,4) - 4*P(4,1)*Metric(1,2)*Metric(3,5) + 4*P(4,2)*Metric(1,2)*Metric(3,5) + 2*P(2,1)*Metric(1,4)*Metric(3,5) + 2*P(2,3)*Metric(1,4)*Metric(3,5) - 6*P(2,4)*Metric(1,4)*Metric(3,5) + 2*P(2,5)*Metric(1,4)*Metric(3,5) - 2*P(1,2)*Metric(2,4)*Metric(3,5) - 2*P(1,3)*Metric(2,4)*Metric(3,5) + 6*P(1,4)*Metric(2,4)*Metric(3,5) - 2*P(1,5)*Metric(2,4)*Metric(3,5) - 4*P(3,1)*Metric(1,2)*Metric(4,5) + 4*P(3,2)*Metric(1,2)*Metric(4,5) + 2*P(2,1)*Metric(1,3)*Metric(4,5) - 6*P(2,3)*Metric(1,3)*Metric(4,5) + 2*P(2,4)*Metric(1,3)*Metric(4,5) + 2*P(2,5)*Metric(1,3)*Metric(4,5) - 2*P(1,2)*Metric(2,3)*Metric(4,5) + 6*P(1,3)*Metric(2,3)*Metric(4,5) - 2*P(1,4)*Metric(2,3)*Metric(4,5) - 2*P(1,5)*Metric(2,3)*Metric(4,5)')

VVVVV82 = Lorentz(name = 'VVVVV82',
                  spins = [ 3, 3, 3, 3, 3 ],
                  structure = 'P(4,5)*Metric(1,5)*Metric(2,3) - P(3,5)*Metric(1,4)*Metric(2,5) - P(2,5)*Metric(1,5)*Metric(3,4) + P(1,5)*Metric(2,5)*Metric(3,4) - P(4,5)*Metric(1,2)*Metric(3,5) + P(2,5)*Metric(1,4)*Metric(3,5) + P(3,5)*Metric(1,2)*Metric(4,5) - P(1,5)*Metric(2,3)*Metric(4,5)')

VVVVV83 = Lorentz(name = 'VVVVV83',
                  spins = [ 3, 3, 3, 3, 3 ],
                  structure = 'P(4,5)*Metric(1,5)*Metric(2,3) - P(3,5)*Metric(1,5)*Metric(2,4) - P(4,5)*Metric(1,3)*Metric(2,5) + P(3,5)*Metric(1,4)*Metric(2,5) - P(2,5)*Metric(1,4)*Metric(3,5) + P(1,5)*Metric(2,4)*Metric(3,5) + P(2,5)*Metric(1,3)*Metric(4,5) - P(1,5)*Metric(2,3)*Metric(4,5)')

VVVVV84 = Lorentz(name = 'VVVVV84',
                  spins = [ 3, 3, 3, 3, 3 ],
                  structure = 'P(5,1)*Metric(1,4)*Metric(2,3) + (P(5,5)*Metric(1,4)*Metric(2,3))/2. - (P(4,1)*Metric(1,5)*Metric(2,3))/2. + (P(4,5)*Metric(1,5)*Metric(2,3))/2. - P(5,1)*Metric(1,2)*Metric(3,4) - (P(5,5)*Metric(1,2)*Metric(3,4))/2. + (P(2,1)*Metric(1,5)*Metric(3,4))/2. - (P(2,5)*Metric(1,5)*Metric(3,4))/2. + (P(1,1)*Metric(2,5)*Metric(3,4))/2. + P(1,5)*Metric(2,5)*Metric(3,4) - (P(1,1)*Metric(2,3)*Metric(4,5))/2. - P(1,5)*Metric(2,3)*Metric(4,5)')

VVVVV85 = Lorentz(name = 'VVVVV85',
                  spins = [ 3, 3, 3, 3, 3 ],
                  structure = 'P(5,1)*Metric(1,4)*Metric(2,3) + (P(5,5)*Metric(1,4)*Metric(2,3))/2. - (P(4,1)*Metric(1,5)*Metric(2,3))/2. + (P(4,5)*Metric(1,5)*Metric(2,3))/2. - P(5,1)*Metric(1,3)*Metric(2,4) - (P(5,5)*Metric(1,3)*Metric(2,4))/2. + (P(3,1)*Metric(1,5)*Metric(2,4))/2. - (P(3,5)*Metric(1,5)*Metric(2,4))/2. + (P(1,1)*Metric(2,4)*Metric(3,5))/2. + P(1,5)*Metric(2,4)*Metric(3,5) - (P(1,1)*Metric(2,3)*Metric(4,5))/2. - P(1,5)*Metric(2,3)*Metric(4,5)')

VVVVV86 = Lorentz(name = 'VVVVV86',
                  spins = [ 3, 3, 3, 3, 3 ],
                  structure = 'P(5,1)*Metric(1,4)*Metric(2,3) + P(5,2)*Metric(1,4)*Metric(2,3) - P(5,3)*Metric(1,4)*Metric(2,3) - P(5,4)*Metric(1,4)*Metric(2,3) - (5*P(4,1)*Metric(1,5)*Metric(2,3))/4. + (P(4,2)*Metric(1,5)*Metric(2,3))/4. + (P(4,3)*Metric(1,5)*Metric(2,3))/4. + (3*P(4,5)*Metric(1,5)*Metric(2,3))/4. + P(5,1)*Metric(1,3)*Metric(2,4) + P(5,2)*Metric(1,3)*Metric(2,4) - P(5,3)*Metric(1,3)*Metric(2,4) - P(5,4)*Metric(1,3)*Metric(2,4) - (5*P(3,1)*Metric(1,5)*Metric(2,4))/4. + (P(3,2)*Metric(1,5)*Metric(2,4))/4. + (P(3,4)*Metric(1,5)*Metric(2,4))/4. + (3*P(3,5)*Metric(1,5)*Metric(2,4))/4. + (P(4,1)*Metric(1,3)*Metric(2,5))/4. - (5*P(4,2)*Metric(1,3)*Metric(2,5))/4. + (P(4,3)*Metric(1,3)*Metric(2,5))/4. + (3*P(4,5)*Metric(1,3)*Metric(2,5))/4. + (P(3,1)*Metric(1,4)*Metric(2,5))/4. - (5*P(3,2)*Metric(1,4)*Metric(2,5))/4. + (P(3,4)*Metric(1,4)*Metric(2,5))/4. + (3*P(3,5)*Metric(1,4)*Metric(2,5))/4. - P(5,1)*Metric(1,2)*Metric(3,4) - P(5,2)*Metric(1,2)*Metric(3,4) + P(5,3)*Metric(1,2)*Metric(3,4) + P(5,4)*Metric(1,2)*Metric(3,4) + (P(2,1)*Metric(1,5)*Metric(3,4))/2. - (P(2,3)*Metric(1,5)*Metric(3,4))/2. - (P(2,4)*Metric(1,5)*Metric(3,4))/2. + (P(2,5)*Metric(1,5)*Metric(3,4))/2. + (P(1,2)*Metric(2,5)*Metric(3,4))/2. - (P(1,3)*Metric(2,5)*Metric(3,4))/2. - (P(1,4)*Metric(2,5)*Metric(3,4))/2. + (P(1,5)*Metric(2,5)*Metric(3,4))/2. + (P(4,1)*Metric(1,2)*Metric(3,5))/2. + (P(4,2)*Metric(1,2)*Metric(3,5))/2. - (P(4,3)*Metric(1,2)*Metric(3,5))/2. - (P(4,5)*Metric(1,2)*Metric(3,5))/2. - (P(2,1)*Metric(1,4)*Metric(3,5))/4. + (5*P(2,3)*Metric(1,4)*Metric(3,5))/4. - (P(2,4)*Metric(1,4)*Metric(3,5))/4. - (3*P(2,5)*Metric(1,4)*Metric(3,5))/4. - (P(1,2)*Metric(2,4)*Metric(3,5))/4. + (5*P(1,3)*Metric(2,4)*Metric(3,5))/4. - (P(1,4)*Metric(2,4)*Metric(3,5))/4. - (3*P(1,5)*Metric(2,4)*Metric(3,5))/4. + (P(3,1)*Metric(1,2)*Metric(4,5))/2. + (P(3,2)*Metric(1,2)*Metric(4,5))/2. - (P(3,4)*Metric(1,2)*Metric(4,5))/2. - (P(3,5)*Metric(1,2)*Metric(4,5))/2. - (P(2,1)*Metric(1,3)*Metric(4,5))/4. - (P(2,3)*Metric(1,3)*Metric(4,5))/4. + (5*P(2,4)*Metric(1,3)*Metric(4,5))/4. - (3*P(2,5)*Metric(1,3)*Metric(4,5))/4. - (P(1,2)*Metric(2,3)*Metric(4,5))/4. - (P(1,3)*Metric(2,3)*Metric(4,5))/4. + (5*P(1,4)*Metric(2,3)*Metric(4,5))/4. - (3*P(1,5)*Metric(2,3)*Metric(4,5))/4.')

VVVVV87 = Lorentz(name = 'VVVVV87',
                  spins = [ 3, 3, 3, 3, 3 ],
                  structure = 'P(5,1)*Metric(1,4)*Metric(2,3) - (P(5,2)*Metric(1,4)*Metric(2,3))/3. - (P(5,3)*Metric(1,4)*Metric(2,3))/3. - (P(5,4)*Metric(1,4)*Metric(2,3))/3. - P(4,1)*Metric(1,5)*Metric(2,3) + (P(4,2)*Metric(1,5)*Metric(2,3))/3. + (P(4,3)*Metric(1,5)*Metric(2,3))/3. + (P(4,5)*Metric(1,5)*Metric(2,3))/3. - (P(5,1)*Metric(1,3)*Metric(2,4))/3. + P(5,2)*Metric(1,3)*Metric(2,4) - (P(5,3)*Metric(1,3)*Metric(2,4))/3. - (P(5,4)*Metric(1,3)*Metric(2,4))/3. + (P(3,1)*Metric(1,5)*Metric(2,4))/6. - (P(3,2)*Metric(1,5)*Metric(2,4))/6. - (P(3,4)*Metric(1,5)*Metric(2,4))/6. + (P(3,5)*Metric(1,5)*Metric(2,4))/6. + (P(4,1)*Metric(1,3)*Metric(2,5))/3. - P(4,2)*Metric(1,3)*Metric(2,5) + (P(4,3)*Metric(1,3)*Metric(2,5))/3. + (P(4,5)*Metric(1,3)*Metric(2,5))/3. - (P(3,1)*Metric(1,4)*Metric(2,5))/6. + (P(3,2)*Metric(1,4)*Metric(2,5))/6. - (P(3,4)*Metric(1,4)*Metric(2,5))/6. + (P(3,5)*Metric(1,4)*Metric(2,5))/6. - (P(5,1)*Metric(1,2)*Metric(3,4))/3. - (P(5,2)*Metric(1,2)*Metric(3,4))/3. + P(5,3)*Metric(1,2)*Metric(3,4) - (P(5,4)*Metric(1,2)*Metric(3,4))/3. + (P(2,1)*Metric(1,5)*Metric(3,4))/6. - (P(2,3)*Metric(1,5)*Metric(3,4))/6. - (P(2,4)*Metric(1,5)*Metric(3,4))/6. + (P(2,5)*Metric(1,5)*Metric(3,4))/6. + (P(1,2)*Metric(2,5)*Metric(3,4))/6. - (P(1,3)*Metric(2,5)*Metric(3,4))/6. - (P(1,4)*Metric(2,5)*Metric(3,4))/6. + (P(1,5)*Metric(2,5)*Metric(3,4))/6. + (P(4,1)*Metric(1,2)*Metric(3,5))/3. + (P(4,2)*Metric(1,2)*Metric(3,5))/3. - P(4,3)*Metric(1,2)*Metric(3,5) + (P(4,5)*Metric(1,2)*Metric(3,5))/3. - (P(2,1)*Metric(1,4)*Metric(3,5))/6. + (P(2,3)*Metric(1,4)*Metric(3,5))/6. - (P(2,4)*Metric(1,4)*Metric(3,5))/6. + (P(2,5)*Metric(1,4)*Metric(3,5))/6. - (P(1,2)*Metric(2,4)*Metric(3,5))/6. + (P(1,3)*Metric(2,4)*Metric(3,5))/6. - (P(1,4)*Metric(2,4)*Metric(3,5))/6. + (P(1,5)*Metric(2,4)*Metric(3,5))/6. + (2*P(3,4)*Metric(1,2)*Metric(4,5))/3. - (2*P(3,5)*Metric(1,2)*Metric(4,5))/3. + (2*P(2,4)*Metric(1,3)*Metric(4,5))/3. - (2*P(2,5)*Metric(1,3)*Metric(4,5))/3. + (2*P(1,4)*Metric(2,3)*Metric(4,5))/3. - (2*P(1,5)*Metric(2,3)*Metric(4,5))/3.')

VVVVV88 = Lorentz(name = 'VVVVV88',
                  spins = [ 3, 3, 3, 3, 3 ],
                  structure = 'P(5,4)*Metric(1,4)*Metric(2,3) + (P(5,5)*Metric(1,4)*Metric(2,3))/2. - (P(4,4)*Metric(1,5)*Metric(2,3))/2. - P(4,5)*Metric(1,5)*Metric(2,3) - P(5,4)*Metric(1,2)*Metric(3,4) - (P(5,5)*Metric(1,2)*Metric(3,4))/2. + (P(4,4)*Metric(1,2)*Metric(3,5))/2. + P(4,5)*Metric(1,2)*Metric(3,5) + (P(3,4)*Metric(1,2)*Metric(4,5))/2. - (P(3,5)*Metric(1,2)*Metric(4,5))/2. - (P(1,4)*Metric(2,3)*Metric(4,5))/2. + (P(1,5)*Metric(2,3)*Metric(4,5))/2.')

VVVVV89 = Lorentz(name = 'VVVVV89',
                  spins = [ 3, 3, 3, 3, 3 ],
                  structure = 'P(5,4)*Metric(1,4)*Metric(2,3) + (P(5,5)*Metric(1,4)*Metric(2,3))/2. - (P(4,4)*Metric(1,5)*Metric(2,3))/2. - P(4,5)*Metric(1,5)*Metric(2,3) - P(5,4)*Metric(1,3)*Metric(2,4) - (P(5,5)*Metric(1,3)*Metric(2,4))/2. + (P(4,4)*Metric(1,3)*Metric(2,5))/2. + P(4,5)*Metric(1,3)*Metric(2,5) + (P(2,4)*Metric(1,3)*Metric(4,5))/2. - (P(2,5)*Metric(1,3)*Metric(4,5))/2. - (P(1,4)*Metric(2,3)*Metric(4,5))/2. + (P(1,5)*Metric(2,3)*Metric(4,5))/2.')

VVVVV90 = Lorentz(name = 'VVVVV90',
                  spins = [ 3, 3, 3, 3, 3 ],
                  structure = 'P(5,1)*Metric(1,4)*Metric(2,3) - P(5,2)*Metric(1,4)*Metric(2,3) - P(4,1)*Metric(1,5)*Metric(2,3) + 2*P(4,2)*Metric(1,5)*Metric(2,3) - P(4,5)*Metric(1,5)*Metric(2,3) - P(5,1)*Metric(1,3)*Metric(2,4) + P(5,2)*Metric(1,3)*Metric(2,4) + P(3,1)*Metric(1,5)*Metric(2,4) - 2*P(3,2)*Metric(1,5)*Metric(2,4) + P(3,5)*Metric(1,5)*Metric(2,4) + 2*P(4,1)*Metric(1,3)*Metric(2,5) - P(4,2)*Metric(1,3)*Metric(2,5) - P(4,5)*Metric(1,3)*Metric(2,5) - 2*P(3,1)*Metric(1,4)*Metric(2,5) + P(3,2)*Metric(1,4)*Metric(2,5) + P(3,5)*Metric(1,4)*Metric(2,5) - P(4,1)*Metric(1,2)*Metric(3,5) - P(4,2)*Metric(1,2)*Metric(3,5) + 2*P(4,5)*Metric(1,2)*Metric(3,5) + P(2,1)*Metric(1,4)*Metric(3,5) - P(2,5)*Metric(1,4)*Metric(3,5) + P(1,2)*Metric(2,4)*Metric(3,5) - P(1,5)*Metric(2,4)*Metric(3,5) + P(3,1)*Metric(1,2)*Metric(4,5) + P(3,2)*Metric(1,2)*Metric(4,5) - 2*P(3,5)*Metric(1,2)*Metric(4,5) - P(2,1)*Metric(1,3)*Metric(4,5) + P(2,5)*Metric(1,3)*Metric(4,5) - P(1,2)*Metric(2,3)*Metric(4,5) + P(1,5)*Metric(2,3)*Metric(4,5)')

VVVVV91 = Lorentz(name = 'VVVVV91',
                  spins = [ 3, 3, 3, 3, 3 ],
                  structure = 'P(5,3)*Metric(1,4)*Metric(2,3) - P(5,4)*Metric(1,4)*Metric(2,3) + P(4,3)*Metric(1,5)*Metric(2,3) - P(4,5)*Metric(1,5)*Metric(2,3) - P(5,3)*Metric(1,3)*Metric(2,4) + P(5,4)*Metric(1,3)*Metric(2,4) + P(3,4)*Metric(1,5)*Metric(2,4) - P(3,5)*Metric(1,5)*Metric(2,4) - P(4,3)*Metric(1,3)*Metric(2,5) + P(4,5)*Metric(1,3)*Metric(2,5) - P(3,4)*Metric(1,4)*Metric(2,5) + P(3,5)*Metric(1,4)*Metric(2,5) - P(2,3)*Metric(1,5)*Metric(3,4) - P(2,4)*Metric(1,5)*Metric(3,4) + 2*P(2,5)*Metric(1,5)*Metric(3,4) + P(1,3)*Metric(2,5)*Metric(3,4) + P(1,4)*Metric(2,5)*Metric(3,4) - 2*P(1,5)*Metric(2,5)*Metric(3,4) - P(2,3)*Metric(1,4)*Metric(3,5) + 2*P(2,4)*Metric(1,4)*Metric(3,5) - P(2,5)*Metric(1,4)*Metric(3,5) + P(1,3)*Metric(2,4)*Metric(3,5) - 2*P(1,4)*Metric(2,4)*Metric(3,5) + P(1,5)*Metric(2,4)*Metric(3,5) + 2*P(2,3)*Metric(1,3)*Metric(4,5) - P(2,4)*Metric(1,3)*Metric(4,5) - P(2,5)*Metric(1,3)*Metric(4,5) - 2*P(1,3)*Metric(2,3)*Metric(4,5) + P(1,4)*Metric(2,3)*Metric(4,5) + P(1,5)*Metric(2,3)*Metric(4,5)')

VVVVV92 = Lorentz(name = 'VVVVV92',
                  spins = [ 3, 3, 3, 3, 3 ],
                  structure = 'P(5,1)*Metric(1,4)*Metric(2,3) - P(5,2)*Metric(1,4)*Metric(2,3) - P(5,3)*Metric(1,4)*Metric(2,3) + P(5,4)*Metric(1,4)*Metric(2,3) - 2*P(4,1)*Metric(1,5)*Metric(2,3) + 6*P(4,2)*Metric(1,5)*Metric(2,3) - 2*P(4,3)*Metric(1,5)*Metric(2,3) - 2*P(4,5)*Metric(1,5)*Metric(2,3) - P(5,1)*Metric(1,3)*Metric(2,4) + P(5,2)*Metric(1,3)*Metric(2,4) - P(5,3)*Metric(1,3)*Metric(2,4) + P(5,4)*Metric(1,3)*Metric(2,4) + 2*P(3,1)*Metric(1,5)*Metric(2,4) - 6*P(3,2)*Metric(1,5)*Metric(2,4) + 2*P(3,4)*Metric(1,5)*Metric(2,4) + 2*P(3,5)*Metric(1,5)*Metric(2,4) + 6*P(4,1)*Metric(1,3)*Metric(2,5) - 2*P(4,2)*Metric(1,3)*Metric(2,5) - 2*P(4,3)*Metric(1,3)*Metric(2,5) - 2*P(4,5)*Metric(1,3)*Metric(2,5) - 6*P(3,1)*Metric(1,4)*Metric(2,5) + 2*P(3,2)*Metric(1,4)*Metric(2,5) + 2*P(3,4)*Metric(1,4)*Metric(2,5) + 2*P(3,5)*Metric(1,4)*Metric(2,5) + 4*P(5,3)*Metric(1,2)*Metric(3,4) - 4*P(5,4)*Metric(1,2)*Metric(3,4) + 4*P(2,3)*Metric(1,5)*Metric(3,4) - 4*P(2,4)*Metric(1,5)*Metric(3,4) + 4*P(1,3)*Metric(2,5)*Metric(3,4) - 4*P(1,4)*Metric(2,5)*Metric(3,4) - 2*P(4,1)*Metric(1,2)*Metric(3,5) - 2*P(4,2)*Metric(1,2)*Metric(3,5) - 2*P(4,3)*Metric(1,2)*Metric(3,5) + 6*P(4,5)*Metric(1,2)*Metric(3,5) + P(2,1)*Metric(1,4)*Metric(3,5) - P(2,3)*Metric(1,4)*Metric(3,5) + P(2,4)*Metric(1,4)*Metric(3,5) - P(2,5)*Metric(1,4)*Metric(3,5) + P(1,2)*Metric(2,4)*Metric(3,5) - P(1,3)*Metric(2,4)*Metric(3,5) + P(1,4)*Metric(2,4)*Metric(3,5) - P(1,5)*Metric(2,4)*Metric(3,5) + 2*P(3,1)*Metric(1,2)*Metric(4,5) + 2*P(3,2)*Metric(1,2)*Metric(4,5) + 2*P(3,4)*Metric(1,2)*Metric(4,5) - 6*P(3,5)*Metric(1,2)*Metric(4,5) - P(2,1)*Metric(1,3)*Metric(4,5) - P(2,3)*Metric(1,3)*Metric(4,5) + P(2,4)*Metric(1,3)*Metric(4,5) + P(2,5)*Metric(1,3)*Metric(4,5) - P(1,2)*Metric(2,3)*Metric(4,5) - P(1,3)*Metric(2,3)*Metric(4,5) + P(1,4)*Metric(2,3)*Metric(4,5) + P(1,5)*Metric(2,3)*Metric(4,5)')

VVVVV93 = Lorentz(name = 'VVVVV93',
                  spins = [ 3, 3, 3, 3, 3 ],
                  structure = 'P(5,2)*Metric(1,4)*Metric(2,3) + P(5,3)*Metric(1,4)*Metric(2,3) - 2*P(5,4)*Metric(1,4)*Metric(2,3) + P(4,2)*Metric(1,5)*Metric(2,3) + P(4,3)*Metric(1,5)*Metric(2,3) - 2*P(4,5)*Metric(1,5)*Metric(2,3) - P(5,3)*Metric(1,3)*Metric(2,4) + P(5,4)*Metric(1,3)*Metric(2,4) - P(3,2)*Metric(1,5)*Metric(2,4) + P(3,5)*Metric(1,5)*Metric(2,4) - P(4,3)*Metric(1,3)*Metric(2,5) + P(4,5)*Metric(1,3)*Metric(2,5) - P(3,2)*Metric(1,4)*Metric(2,5) + P(3,4)*Metric(1,4)*Metric(2,5) - P(5,2)*Metric(1,2)*Metric(3,4) + P(5,4)*Metric(1,2)*Metric(3,4) - P(2,3)*Metric(1,5)*Metric(3,4) + P(2,5)*Metric(1,5)*Metric(3,4) + P(1,2)*Metric(2,5)*Metric(3,4) + P(1,3)*Metric(2,5)*Metric(3,4) - P(1,4)*Metric(2,5)*Metric(3,4) - P(1,5)*Metric(2,5)*Metric(3,4) - P(4,2)*Metric(1,2)*Metric(3,5) + P(4,5)*Metric(1,2)*Metric(3,5) - P(2,3)*Metric(1,4)*Metric(3,5) + P(2,4)*Metric(1,4)*Metric(3,5) + P(1,2)*Metric(2,4)*Metric(3,5) + P(1,3)*Metric(2,4)*Metric(3,5) - P(1,4)*Metric(2,4)*Metric(3,5) - P(1,5)*Metric(2,4)*Metric(3,5) + 2*P(3,2)*Metric(1,2)*Metric(4,5) - P(3,4)*Metric(1,2)*Metric(4,5) - P(3,5)*Metric(1,2)*Metric(4,5) + 2*P(2,3)*Metric(1,3)*Metric(4,5) - P(2,4)*Metric(1,3)*Metric(4,5) - P(2,5)*Metric(1,3)*Metric(4,5) - 2*P(1,2)*Metric(2,3)*Metric(4,5) - 2*P(1,3)*Metric(2,3)*Metric(4,5) + 2*P(1,4)*Metric(2,3)*Metric(4,5) + 2*P(1,5)*Metric(2,3)*Metric(4,5)')

SSSSSS1 = Lorentz(name = 'SSSSSS1',
                  spins = [ 1, 1, 1, 1, 1, 1 ],
                  structure = '1')

VVSSSS1 = Lorentz(name = 'VVSSSS1',
                  spins = [ 3, 3, 1, 1, 1, 1 ],
                  structure = 'Metric(1,2)')

VVVVSS1 = Lorentz(name = 'VVVVSS1',
                  spins = [ 3, 3, 3, 3, 1, 1 ],
                  structure = 'Metric(1,4)*Metric(2,3) - Metric(1,3)*Metric(2,4)')

VVVVSS2 = Lorentz(name = 'VVVVSS2',
                  spins = [ 3, 3, 3, 3, 1, 1 ],
                  structure = 'Metric(1,4)*Metric(2,3) + Metric(1,3)*Metric(2,4) - 2*Metric(1,2)*Metric(3,4)')

VVVVSS3 = Lorentz(name = 'VVVVSS3',
                  spins = [ 3, 3, 3, 3, 1, 1 ],
                  structure = 'Metric(1,4)*Metric(2,3) - Metric(1,2)*Metric(3,4)')

VVVVSS4 = Lorentz(name = 'VVVVSS4',
                  spins = [ 3, 3, 3, 3, 1, 1 ],
                  structure = 'Metric(1,3)*Metric(2,4) - Metric(1,2)*Metric(3,4)')

VVVVSS5 = Lorentz(name = 'VVVVSS5',
                  spins = [ 3, 3, 3, 3, 1, 1 ],
                  structure = 'Metric(1,4)*Metric(2,3) - (Metric(1,3)*Metric(2,4))/2. - (Metric(1,2)*Metric(3,4))/2.')

VVVVVV1 = Lorentz(name = 'VVVVVV1',
                  spins = [ 3, 3, 3, 3, 3, 3 ],
                  structure = 'Epsilon(2,4,5,6)*Metric(1,3) - Epsilon(2,3,5,6)*Metric(1,4) - Epsilon(1,4,5,6)*Metric(2,3) + Epsilon(1,3,5,6)*Metric(2,4)')

VVVVVV2 = Lorentz(name = 'VVVVVV2',
                  spins = [ 3, 3, 3, 3, 3, 3 ],
                  structure = 'Epsilon(2,4,5,6)*Metric(1,3) + Epsilon(2,3,4,6)*Metric(1,5) - Epsilon(1,4,5,6)*Metric(2,3) - Epsilon(1,3,4,6)*Metric(2,5)')

VVVVVV3 = Lorentz(name = 'VVVVVV3',
                  spins = [ 3, 3, 3, 3, 3, 3 ],
                  structure = 'Epsilon(2,3,5,6)*Metric(1,4) - Epsilon(2,3,4,6)*Metric(1,5) - Epsilon(1,3,5,6)*Metric(2,4) + Epsilon(1,3,4,6)*Metric(2,5)')

VVVVVV4 = Lorentz(name = 'VVVVVV4',
                  spins = [ 3, 3, 3, 3, 3, 3 ],
                  structure = 'Epsilon(2,3,5,6)*Metric(1,4) + Epsilon(2,3,4,5)*Metric(1,6) - Epsilon(1,3,5,6)*Metric(2,4) - Epsilon(1,3,4,5)*Metric(2,6)')

VVVVVV5 = Lorentz(name = 'VVVVVV5',
                  spins = [ 3, 3, 3, 3, 3, 3 ],
                  structure = 'Epsilon(2,4,5,6)*Metric(1,3) - Epsilon(2,3,4,5)*Metric(1,6) - Epsilon(1,4,5,6)*Metric(2,3) + Epsilon(1,3,4,5)*Metric(2,6)')

VVVVVV6 = Lorentz(name = 'VVVVVV6',
                  spins = [ 3, 3, 3, 3, 3, 3 ],
                  structure = 'Epsilon(2,3,4,6)*Metric(1,5) - Epsilon(2,3,4,5)*Metric(1,6) - Epsilon(1,3,4,6)*Metric(2,5) + Epsilon(1,3,4,5)*Metric(2,6)')

VVVVVV7 = Lorentz(name = 'VVVVVV7',
                  spins = [ 3, 3, 3, 3, 3, 3 ],
                  structure = 'Epsilon(3,4,5,6)*Metric(1,2) - Epsilon(2,4,5,6)*Metric(1,3) + Epsilon(1,3,5,6)*Metric(2,4) - Epsilon(1,2,5,6)*Metric(3,4)')

VVVVVV8 = Lorentz(name = 'VVVVVV8',
                  spins = [ 3, 3, 3, 3, 3, 3 ],
                  structure = 'Epsilon(3,4,5,6)*Metric(1,2) + Epsilon(2,3,5,6)*Metric(1,4) - Epsilon(1,4,5,6)*Metric(2,3) + Epsilon(1,2,5,6)*Metric(3,4)')

VVVVVV9 = Lorentz(name = 'VVVVVV9',
                  spins = [ 3, 3, 3, 3, 3, 3 ],
                  structure = 'Epsilon(3,4,5,6)*Metric(1,2) - Epsilon(2,3,4,6)*Metric(1,5) - Epsilon(1,4,5,6)*Metric(2,3) - Epsilon(1,2,4,6)*Metric(3,5)')

VVVVVV10 = Lorentz(name = 'VVVVVV10',
                   spins = [ 3, 3, 3, 3, 3, 3 ],
                   structure = 'Epsilon(2,3,5,6)*Metric(1,4) - Epsilon(2,3,4,6)*Metric(1,5) + Epsilon(1,2,5,6)*Metric(3,4) - Epsilon(1,2,4,6)*Metric(3,5)')

VVVVVV11 = Lorentz(name = 'VVVVVV11',
                   spins = [ 3, 3, 3, 3, 3, 3 ],
                   structure = 'Epsilon(3,4,5,6)*Metric(1,2) - Epsilon(2,4,5,6)*Metric(1,3) - Epsilon(1,3,4,6)*Metric(2,5) + Epsilon(1,2,4,6)*Metric(3,5)')

VVVVVV12 = Lorentz(name = 'VVVVVV12',
                   spins = [ 3, 3, 3, 3, 3, 3 ],
                   structure = 'Epsilon(1,3,5,6)*Metric(2,4) - Epsilon(1,3,4,6)*Metric(2,5) - Epsilon(1,2,5,6)*Metric(3,4) + Epsilon(1,2,4,6)*Metric(3,5)')

VVVVVV13 = Lorentz(name = 'VVVVVV13',
                   spins = [ 3, 3, 3, 3, 3, 3 ],
                   structure = 'Metric(1,6)*Metric(2,5)*Metric(3,4) - Metric(1,5)*Metric(2,6)*Metric(3,4) - Metric(1,6)*Metric(2,4)*Metric(3,5) + Metric(1,4)*Metric(2,6)*Metric(3,5)')

VVVVVV14 = Lorentz(name = 'VVVVVV14',
                   spins = [ 3, 3, 3, 3, 3, 3 ],
                   structure = 'Epsilon(3,4,5,6)*Metric(1,2) - Epsilon(2,4,5,6)*Metric(1,3) + Epsilon(1,3,4,5)*Metric(2,6) - Epsilon(1,2,4,5)*Metric(3,6)')

VVVVVV15 = Lorentz(name = 'VVVVVV15',
                   spins = [ 3, 3, 3, 3, 3, 3 ],
                   structure = 'Epsilon(1,3,5,6)*Metric(2,4) + Epsilon(1,3,4,5)*Metric(2,6) - Epsilon(1,2,5,6)*Metric(3,4) - Epsilon(1,2,4,5)*Metric(3,6)')

VVVVVV16 = Lorentz(name = 'VVVVVV16',
                   spins = [ 3, 3, 3, 3, 3, 3 ],
                   structure = 'Epsilon(2,3,4,6)*Metric(1,5) - Epsilon(2,3,4,5)*Metric(1,6) + Epsilon(1,2,4,6)*Metric(3,5) - Epsilon(1,2,4,5)*Metric(3,6)')

VVVVVV17 = Lorentz(name = 'VVVVVV17',
                   spins = [ 3, 3, 3, 3, 3, 3 ],
                   structure = 'Epsilon(3,4,5,6)*Metric(1,2) + Epsilon(2,3,4,5)*Metric(1,6) - Epsilon(1,4,5,6)*Metric(2,3) + Epsilon(1,2,4,5)*Metric(3,6)')

VVVVVV18 = Lorentz(name = 'VVVVVV18',
                   spins = [ 3, 3, 3, 3, 3, 3 ],
                   structure = 'Epsilon(2,3,5,6)*Metric(1,4) + Epsilon(2,3,4,5)*Metric(1,6) + Epsilon(1,2,5,6)*Metric(3,4) + Epsilon(1,2,4,5)*Metric(3,6)')

VVVVVV19 = Lorentz(name = 'VVVVVV19',
                   spins = [ 3, 3, 3, 3, 3, 3 ],
                   structure = 'Epsilon(1,3,4,6)*Metric(2,5) - Epsilon(1,3,4,5)*Metric(2,6) - Epsilon(1,2,4,6)*Metric(3,5) + Epsilon(1,2,4,5)*Metric(3,6)')

VVVVVV20 = Lorentz(name = 'VVVVVV20',
                   spins = [ 3, 3, 3, 3, 3, 3 ],
                   structure = 'Metric(1,5)*Metric(2,6)*Metric(3,4) + Metric(1,6)*Metric(2,4)*Metric(3,5) - Metric(1,4)*Metric(2,6)*Metric(3,5) - Metric(1,5)*Metric(2,4)*Metric(3,6)')

VVVVVV21 = Lorentz(name = 'VVVVVV21',
                   spins = [ 3, 3, 3, 3, 3, 3 ],
                   structure = 'Metric(1,6)*Metric(2,5)*Metric(3,4) - Metric(1,5)*Metric(2,6)*Metric(3,4) - Metric(1,6)*Metric(2,4)*Metric(3,5) + Metric(1,5)*Metric(2,4)*Metric(3,6)')

VVVVVV22 = Lorentz(name = 'VVVVVV22',
                   spins = [ 3, 3, 3, 3, 3, 3 ],
                   structure = 'Metric(1,6)*Metric(2,5)*Metric(3,4) - Metric(1,5)*Metric(2,6)*Metric(3,4) + Metric(1,4)*Metric(2,6)*Metric(3,5) - Metric(1,4)*Metric(2,5)*Metric(3,6)')

VVVVVV23 = Lorentz(name = 'VVVVVV23',
                   spins = [ 3, 3, 3, 3, 3, 3 ],
                   structure = 'Metric(1,6)*Metric(2,5)*Metric(3,4) - Metric(1,6)*Metric(2,4)*Metric(3,5) + Metric(1,4)*Metric(2,6)*Metric(3,5) - Metric(1,4)*Metric(2,5)*Metric(3,6)')

VVVVVV24 = Lorentz(name = 'VVVVVV24',
                   spins = [ 3, 3, 3, 3, 3, 3 ],
                   structure = 'Metric(1,6)*Metric(2,5)*Metric(3,4) - Metric(1,5)*Metric(2,6)*Metric(3,4) + Metric(1,5)*Metric(2,4)*Metric(3,6) - Metric(1,4)*Metric(2,5)*Metric(3,6)')

VVVVVV25 = Lorentz(name = 'VVVVVV25',
                   spins = [ 3, 3, 3, 3, 3, 3 ],
                   structure = 'Metric(1,6)*Metric(2,5)*Metric(3,4) - Metric(1,6)*Metric(2,4)*Metric(3,5) + Metric(1,5)*Metric(2,4)*Metric(3,6) - Metric(1,4)*Metric(2,5)*Metric(3,6)')

VVVVVV26 = Lorentz(name = 'VVVVVV26',
                   spins = [ 3, 3, 3, 3, 3, 3 ],
                   structure = 'Metric(1,5)*Metric(2,6)*Metric(3,4) - Metric(1,4)*Metric(2,6)*Metric(3,5) - Metric(1,5)*Metric(2,4)*Metric(3,6) + Metric(1,4)*Metric(2,5)*Metric(3,6)')

VVVVVV27 = Lorentz(name = 'VVVVVV27',
                   spins = [ 3, 3, 3, 3, 3, 3 ],
                   structure = 'Metric(1,6)*Metric(2,4)*Metric(3,5) - Metric(1,4)*Metric(2,6)*Metric(3,5) - Metric(1,5)*Metric(2,4)*Metric(3,6) + Metric(1,4)*Metric(2,5)*Metric(3,6)')

VVVVVV28 = Lorentz(name = 'VVVVVV28',
                   spins = [ 3, 3, 3, 3, 3, 3 ],
                   structure = 'Epsilon(3,4,5,6)*Metric(1,2) + Epsilon(2,3,5,6)*Metric(1,4) - Epsilon(1,3,4,6)*Metric(2,5) - Epsilon(1,2,3,6)*Metric(4,5)')

VVVVVV29 = Lorentz(name = 'VVVVVV29',
                   spins = [ 3, 3, 3, 3, 3, 3 ],
                   structure = 'Epsilon(2,4,5,6)*Metric(1,3) + Epsilon(2,3,4,6)*Metric(1,5) + Epsilon(1,2,5,6)*Metric(3,4) - Epsilon(1,2,3,6)*Metric(4,5)')

VVVVVV30 = Lorentz(name = 'VVVVVV30',
                   spins = [ 3, 3, 3, 3, 3, 3 ],
                   structure = 'Epsilon(1,4,5,6)*Metric(2,3) - Epsilon(1,3,5,6)*Metric(2,4) + Epsilon(1,2,4,6)*Metric(3,5) - Epsilon(1,2,3,6)*Metric(4,5)')

VVVVVV31 = Lorentz(name = 'VVVVVV31',
                   spins = [ 3, 3, 3, 3, 3, 3 ],
                   structure = 'Epsilon(3,4,5,6)*Metric(1,2) - Epsilon(2,3,4,6)*Metric(1,5) + Epsilon(1,3,5,6)*Metric(2,4) + Epsilon(1,2,3,6)*Metric(4,5)')

VVVVVV32 = Lorentz(name = 'VVVVVV32',
                   spins = [ 3, 3, 3, 3, 3, 3 ],
                   structure = 'Epsilon(1,4,5,6)*Metric(2,3) + Epsilon(1,3,4,6)*Metric(2,5) - Epsilon(1,2,5,6)*Metric(3,4) + Epsilon(1,2,3,6)*Metric(4,5)')

VVVVVV33 = Lorentz(name = 'VVVVVV33',
                   spins = [ 3, 3, 3, 3, 3, 3 ],
                   structure = 'Epsilon(2,4,5,6)*Metric(1,3) - Epsilon(2,3,5,6)*Metric(1,4) - Epsilon(1,2,4,6)*Metric(3,5) + Epsilon(1,2,3,6)*Metric(4,5)')

VVVVVV34 = Lorentz(name = 'VVVVVV34',
                   spins = [ 3, 3, 3, 3, 3, 3 ],
                   structure = 'Metric(1,6)*Metric(2,5)*Metric(3,4) - Metric(1,5)*Metric(2,6)*Metric(3,4) - Metric(1,6)*Metric(2,3)*Metric(4,5) + Metric(1,3)*Metric(2,6)*Metric(4,5)')

VVVVVV35 = Lorentz(name = 'VVVVVV35',
                   spins = [ 3, 3, 3, 3, 3, 3 ],
                   structure = 'Metric(1,6)*Metric(2,4)*Metric(3,5) - Metric(1,4)*Metric(2,6)*Metric(3,5) - Metric(1,6)*Metric(2,3)*Metric(4,5) + Metric(1,3)*Metric(2,6)*Metric(4,5)')

VVVVVV36 = Lorentz(name = 'VVVVVV36',
                   spins = [ 3, 3, 3, 3, 3, 3 ],
                   structure = 'Metric(1,6)*Metric(2,4)*Metric(3,5) - Metric(1,5)*Metric(2,4)*Metric(3,6) - Metric(1,6)*Metric(2,3)*Metric(4,5) + Metric(1,2)*Metric(3,6)*Metric(4,5)')

VVVVVV37 = Lorentz(name = 'VVVVVV37',
                   spins = [ 3, 3, 3, 3, 3, 3 ],
                   structure = 'Metric(1,6)*Metric(2,5)*Metric(3,4) - Metric(1,4)*Metric(2,5)*Metric(3,6) - Metric(1,6)*Metric(2,3)*Metric(4,5) + Metric(1,2)*Metric(3,6)*Metric(4,5)')

VVVVVV38 = Lorentz(name = 'VVVVVV38',
                   spins = [ 3, 3, 3, 3, 3, 3 ],
                   structure = 'Metric(1,5)*Metric(2,6)*Metric(3,4) - Metric(1,5)*Metric(2,4)*Metric(3,6) - Metric(1,3)*Metric(2,6)*Metric(4,5) + Metric(1,2)*Metric(3,6)*Metric(4,5)')

VVVVVV39 = Lorentz(name = 'VVVVVV39',
                   spins = [ 3, 3, 3, 3, 3, 3 ],
                   structure = 'Metric(1,4)*Metric(2,6)*Metric(3,5) - Metric(1,4)*Metric(2,5)*Metric(3,6) - Metric(1,3)*Metric(2,6)*Metric(4,5) + Metric(1,2)*Metric(3,6)*Metric(4,5)')

VVVVVV40 = Lorentz(name = 'VVVVVV40',
                   spins = [ 3, 3, 3, 3, 3, 3 ],
                   structure = 'Epsilon(3,4,5,6)*Metric(1,2) + Epsilon(2,3,4,5)*Metric(1,6) + Epsilon(1,3,5,6)*Metric(2,4) - Epsilon(1,2,3,5)*Metric(4,6)')

VVVVVV41 = Lorentz(name = 'VVVVVV41',
                   spins = [ 3, 3, 3, 3, 3, 3 ],
                   structure = 'Epsilon(1,4,5,6)*Metric(2,3) - Epsilon(1,3,4,5)*Metric(2,6) - Epsilon(1,2,5,6)*Metric(3,4) - Epsilon(1,2,3,5)*Metric(4,6)')

VVVVVV42 = Lorentz(name = 'VVVVVV42',
                   spins = [ 3, 3, 3, 3, 3, 3 ],
                   structure = 'Epsilon(2,4,5,6)*Metric(1,3) - Epsilon(2,3,5,6)*Metric(1,4) + Epsilon(1,2,4,5)*Metric(3,6) - Epsilon(1,2,3,5)*Metric(4,6)')

VVVVVV43 = Lorentz(name = 'VVVVVV43',
                   spins = [ 3, 3, 3, 3, 3, 3 ],
                   structure = 'Epsilon(1,3,4,6)*Metric(2,5) - Epsilon(1,3,4,5)*Metric(2,6) + Epsilon(1,2,3,6)*Metric(4,5) - Epsilon(1,2,3,5)*Metric(4,6)')

VVVVVV44 = Lorentz(name = 'VVVVVV44',
                   spins = [ 3, 3, 3, 3, 3, 3 ],
                   structure = 'Epsilon(3,4,5,6)*Metric(1,2) + Epsilon(2,3,5,6)*Metric(1,4) + Epsilon(1,3,4,5)*Metric(2,6) + Epsilon(1,2,3,5)*Metric(4,6)')

VVVVVV45 = Lorentz(name = 'VVVVVV45',
                   spins = [ 3, 3, 3, 3, 3, 3 ],
                   structure = 'Epsilon(2,4,5,6)*Metric(1,3) - Epsilon(2,3,4,5)*Metric(1,6) + Epsilon(1,2,5,6)*Metric(3,4) + Epsilon(1,2,3,5)*Metric(4,6)')

VVVVVV46 = Lorentz(name = 'VVVVVV46',
                   spins = [ 3, 3, 3, 3, 3, 3 ],
                   structure = 'Epsilon(1,4,5,6)*Metric(2,3) - Epsilon(1,3,5,6)*Metric(2,4) - Epsilon(1,2,4,5)*Metric(3,6) + Epsilon(1,2,3,5)*Metric(4,6)')

VVVVVV47 = Lorentz(name = 'VVVVVV47',
                   spins = [ 3, 3, 3, 3, 3, 3 ],
                   structure = 'Epsilon(2,3,4,6)*Metric(1,5) - Epsilon(2,3,4,5)*Metric(1,6) - Epsilon(1,2,3,6)*Metric(4,5) + Epsilon(1,2,3,5)*Metric(4,6)')

VVVVVV48 = Lorentz(name = 'VVVVVV48',
                   spins = [ 3, 3, 3, 3, 3, 3 ],
                   structure = 'Epsilon(1,2,4,6)*Metric(3,5) - Epsilon(1,2,4,5)*Metric(3,6) - Epsilon(1,2,3,6)*Metric(4,5) + Epsilon(1,2,3,5)*Metric(4,6)')

VVVVVV49 = Lorentz(name = 'VVVVVV49',
                   spins = [ 3, 3, 3, 3, 3, 3 ],
                   structure = 'Metric(1,5)*Metric(2,6)*Metric(3,4) + Metric(1,6)*Metric(2,3)*Metric(4,5) - Metric(1,3)*Metric(2,6)*Metric(4,5) - Metric(1,5)*Metric(2,3)*Metric(4,6)')

VVVVVV50 = Lorentz(name = 'VVVVVV50',
                   spins = [ 3, 3, 3, 3, 3, 3 ],
                   structure = 'Metric(1,5)*Metric(2,4)*Metric(3,6) + Metric(1,6)*Metric(2,3)*Metric(4,5) - Metric(1,2)*Metric(3,6)*Metric(4,5) - Metric(1,5)*Metric(2,3)*Metric(4,6)')

VVVVVV51 = Lorentz(name = 'VVVVVV51',
                   spins = [ 3, 3, 3, 3, 3, 3 ],
                   structure = 'Metric(1,6)*Metric(2,5)*Metric(3,4) - Metric(1,5)*Metric(2,6)*Metric(3,4) - Metric(1,6)*Metric(2,3)*Metric(4,5) + Metric(1,5)*Metric(2,3)*Metric(4,6)')

VVVVVV52 = Lorentz(name = 'VVVVVV52',
                   spins = [ 3, 3, 3, 3, 3, 3 ],
                   structure = 'Metric(1,6)*Metric(2,4)*Metric(3,5) - Metric(1,5)*Metric(2,4)*Metric(3,6) - Metric(1,6)*Metric(2,3)*Metric(4,5) + Metric(1,5)*Metric(2,3)*Metric(4,6)')

VVVVVV53 = Lorentz(name = 'VVVVVV53',
                   spins = [ 3, 3, 3, 3, 3, 3 ],
                   structure = 'Metric(1,6)*Metric(2,5)*Metric(3,4) - Metric(1,5)*Metric(2,6)*Metric(3,4) + Metric(1,3)*Metric(2,6)*Metric(4,5) - Metric(1,3)*Metric(2,5)*Metric(4,6)')

VVVVVV54 = Lorentz(name = 'VVVVVV54',
                   spins = [ 3, 3, 3, 3, 3, 3 ],
                   structure = 'Metric(1,6)*Metric(2,5)*Metric(3,4) - Metric(1,6)*Metric(2,3)*Metric(4,5) + Metric(1,3)*Metric(2,6)*Metric(4,5) - Metric(1,3)*Metric(2,5)*Metric(4,6)')

VVVVVV55 = Lorentz(name = 'VVVVVV55',
                   spins = [ 3, 3, 3, 3, 3, 3 ],
                   structure = 'Metric(1,4)*Metric(2,5)*Metric(3,6) + Metric(1,3)*Metric(2,6)*Metric(4,5) - Metric(1,2)*Metric(3,6)*Metric(4,5) - Metric(1,3)*Metric(2,5)*Metric(4,6)')

VVVVVV56 = Lorentz(name = 'VVVVVV56',
                   spins = [ 3, 3, 3, 3, 3, 3 ],
                   structure = 'Metric(1,6)*Metric(2,5)*Metric(3,4) - Metric(1,5)*Metric(2,6)*Metric(3,4) + Metric(1,5)*Metric(2,3)*Metric(4,6) - Metric(1,3)*Metric(2,5)*Metric(4,6)')

VVVVVV57 = Lorentz(name = 'VVVVVV57',
                   spins = [ 3, 3, 3, 3, 3, 3 ],
                   structure = 'Metric(1,6)*Metric(2,5)*Metric(3,4) - Metric(1,6)*Metric(2,3)*Metric(4,5) + Metric(1,5)*Metric(2,3)*Metric(4,6) - Metric(1,3)*Metric(2,5)*Metric(4,6)')

VVVVVV58 = Lorentz(name = 'VVVVVV58',
                   spins = [ 3, 3, 3, 3, 3, 3 ],
                   structure = 'Metric(1,6)*Metric(2,4)*Metric(3,5) - Metric(1,4)*Metric(2,6)*Metric(3,5) - Metric(1,5)*Metric(2,4)*Metric(3,6) + Metric(1,4)*Metric(2,5)*Metric(3,6) - Metric(1,6)*Metric(2,3)*Metric(4,5) + Metric(1,3)*Metric(2,6)*Metric(4,5) + Metric(1,5)*Metric(2,3)*Metric(4,6) - Metric(1,3)*Metric(2,5)*Metric(4,6)')

VVVVVV59 = Lorentz(name = 'VVVVVV59',
                   spins = [ 3, 3, 3, 3, 3, 3 ],
                   structure = 'Metric(1,4)*Metric(2,6)*Metric(3,5) - Metric(1,4)*Metric(2,5)*Metric(3,6) - Metric(1,3)*Metric(2,6)*Metric(4,5) + Metric(1,3)*Metric(2,5)*Metric(4,6)')

VVVVVV60 = Lorentz(name = 'VVVVVV60',
                   spins = [ 3, 3, 3, 3, 3, 3 ],
                   structure = 'Metric(1,5)*Metric(2,4)*Metric(3,6) - Metric(1,4)*Metric(2,5)*Metric(3,6) - Metric(1,5)*Metric(2,3)*Metric(4,6) + Metric(1,3)*Metric(2,5)*Metric(4,6)')

VVVVVV61 = Lorentz(name = 'VVVVVV61',
                   spins = [ 3, 3, 3, 3, 3, 3 ],
                   structure = 'Metric(1,5)*Metric(2,6)*Metric(3,4) - Metric(1,3)*Metric(2,6)*Metric(4,5) - Metric(1,5)*Metric(2,3)*Metric(4,6) + Metric(1,3)*Metric(2,5)*Metric(4,6)')

VVVVVV62 = Lorentz(name = 'VVVVVV62',
                   spins = [ 3, 3, 3, 3, 3, 3 ],
                   structure = 'Metric(1,6)*Metric(2,3)*Metric(4,5) - Metric(1,3)*Metric(2,6)*Metric(4,5) - Metric(1,5)*Metric(2,3)*Metric(4,6) + Metric(1,3)*Metric(2,5)*Metric(4,6)')

VVVVVV63 = Lorentz(name = 'VVVVVV63',
                   spins = [ 3, 3, 3, 3, 3, 3 ],
                   structure = 'Metric(1,6)*Metric(2,4)*Metric(3,5) - Metric(1,5)*Metric(2,4)*Metric(3,6) + Metric(1,2)*Metric(3,6)*Metric(4,5) - Metric(1,2)*Metric(3,5)*Metric(4,6)')

VVVVVV64 = Lorentz(name = 'VVVVVV64',
                   spins = [ 3, 3, 3, 3, 3, 3 ],
                   structure = 'Metric(1,4)*Metric(2,6)*Metric(3,5) - Metric(1,4)*Metric(2,5)*Metric(3,6) + Metric(1,2)*Metric(3,6)*Metric(4,5) - Metric(1,2)*Metric(3,5)*Metric(4,6)')

VVVVVV65 = Lorentz(name = 'VVVVVV65',
                   spins = [ 3, 3, 3, 3, 3, 3 ],
                   structure = 'Metric(1,6)*Metric(2,4)*Metric(3,5) - Metric(1,6)*Metric(2,3)*Metric(4,5) + Metric(1,2)*Metric(3,6)*Metric(4,5) - Metric(1,2)*Metric(3,5)*Metric(4,6)')

VVVVVV66 = Lorentz(name = 'VVVVVV66',
                   spins = [ 3, 3, 3, 3, 3, 3 ],
                   structure = 'Metric(1,4)*Metric(2,6)*Metric(3,5) - Metric(1,3)*Metric(2,6)*Metric(4,5) + Metric(1,2)*Metric(3,6)*Metric(4,5) - Metric(1,2)*Metric(3,5)*Metric(4,6)')

VVVVVV67 = Lorentz(name = 'VVVVVV67',
                   spins = [ 3, 3, 3, 3, 3, 3 ],
                   structure = 'Metric(1,6)*Metric(2,4)*Metric(3,5) - Metric(1,5)*Metric(2,4)*Metric(3,6) + Metric(1,5)*Metric(2,3)*Metric(4,6) - Metric(1,2)*Metric(3,5)*Metric(4,6)')

VVVVVV68 = Lorentz(name = 'VVVVVV68',
                   spins = [ 3, 3, 3, 3, 3, 3 ],
                   structure = 'Metric(1,6)*Metric(2,4)*Metric(3,5) - Metric(1,6)*Metric(2,3)*Metric(4,5) + Metric(1,5)*Metric(2,3)*Metric(4,6) - Metric(1,2)*Metric(3,5)*Metric(4,6)')

VVVVVV69 = Lorentz(name = 'VVVVVV69',
                   spins = [ 3, 3, 3, 3, 3, 3 ],
                   structure = 'Metric(1,6)*Metric(2,5)*Metric(3,4) - Metric(1,5)*Metric(2,6)*Metric(3,4) + Metric(1,4)*Metric(2,6)*Metric(3,5) - Metric(1,4)*Metric(2,5)*Metric(3,6) - Metric(1,6)*Metric(2,3)*Metric(4,5) + Metric(1,2)*Metric(3,6)*Metric(4,5) + Metric(1,5)*Metric(2,3)*Metric(4,6) - Metric(1,2)*Metric(3,5)*Metric(4,6)')

VVVVVV70 = Lorentz(name = 'VVVVVV70',
                   spins = [ 3, 3, 3, 3, 3, 3 ],
                   structure = 'Metric(1,4)*Metric(2,6)*Metric(3,5) - Metric(1,4)*Metric(2,5)*Metric(3,6) + Metric(1,3)*Metric(2,5)*Metric(4,6) - Metric(1,2)*Metric(3,5)*Metric(4,6)')

VVVVVV71 = Lorentz(name = 'VVVVVV71',
                   spins = [ 3, 3, 3, 3, 3, 3 ],
                   structure = 'Metric(1,4)*Metric(2,6)*Metric(3,5) - Metric(1,3)*Metric(2,6)*Metric(4,5) + Metric(1,3)*Metric(2,5)*Metric(4,6) - Metric(1,2)*Metric(3,5)*Metric(4,6)')

VVVVVV72 = Lorentz(name = 'VVVVVV72',
                   spins = [ 3, 3, 3, 3, 3, 3 ],
                   structure = 'Metric(1,5)*Metric(2,6)*Metric(3,4) - Metric(1,4)*Metric(2,6)*Metric(3,5) - Metric(1,5)*Metric(2,3)*Metric(4,6) + Metric(1,2)*Metric(3,5)*Metric(4,6)')

VVVVVV73 = Lorentz(name = 'VVVVVV73',
                   spins = [ 3, 3, 3, 3, 3, 3 ],
                   structure = 'Metric(1,5)*Metric(2,4)*Metric(3,6) - Metric(1,2)*Metric(3,6)*Metric(4,5) - Metric(1,5)*Metric(2,3)*Metric(4,6) + Metric(1,2)*Metric(3,5)*Metric(4,6)')

VVVVVV74 = Lorentz(name = 'VVVVVV74',
                   spins = [ 3, 3, 3, 3, 3, 3 ],
                   structure = 'Metric(1,6)*Metric(2,3)*Metric(4,5) - Metric(1,2)*Metric(3,6)*Metric(4,5) - Metric(1,5)*Metric(2,3)*Metric(4,6) + Metric(1,2)*Metric(3,5)*Metric(4,6)')

VVVVVV75 = Lorentz(name = 'VVVVVV75',
                   spins = [ 3, 3, 3, 3, 3, 3 ],
                   structure = 'Metric(1,6)*Metric(2,5)*Metric(3,4) - Metric(1,6)*Metric(2,4)*Metric(3,5) - Metric(1,3)*Metric(2,5)*Metric(4,6) + Metric(1,2)*Metric(3,5)*Metric(4,6)')

VVVVVV76 = Lorentz(name = 'VVVVVV76',
                   spins = [ 3, 3, 3, 3, 3, 3 ],
                   structure = 'Metric(1,4)*Metric(2,5)*Metric(3,6) - Metric(1,2)*Metric(3,6)*Metric(4,5) - Metric(1,3)*Metric(2,5)*Metric(4,6) + Metric(1,2)*Metric(3,5)*Metric(4,6)')

VVVVVV77 = Lorentz(name = 'VVVVVV77',
                   spins = [ 3, 3, 3, 3, 3, 3 ],
                   structure = 'Metric(1,3)*Metric(2,6)*Metric(4,5) - Metric(1,2)*Metric(3,6)*Metric(4,5) - Metric(1,3)*Metric(2,5)*Metric(4,6) + Metric(1,2)*Metric(3,5)*Metric(4,6)')

VVVVVV78 = Lorentz(name = 'VVVVVV78',
                   spins = [ 3, 3, 3, 3, 3, 3 ],
                   structure = 'Metric(1,6)*Metric(2,5)*Metric(3,4) - Metric(1,5)*Metric(2,6)*Metric(3,4) - Metric(1,6)*Metric(2,4)*Metric(3,5) + Metric(1,5)*Metric(2,4)*Metric(3,6) + Metric(1,3)*Metric(2,6)*Metric(4,5) - Metric(1,2)*Metric(3,6)*Metric(4,5) - Metric(1,3)*Metric(2,5)*Metric(4,6) + Metric(1,2)*Metric(3,5)*Metric(4,6)')

VVVVVV79 = Lorentz(name = 'VVVVVV79',
                   spins = [ 3, 3, 3, 3, 3, 3 ],
                   structure = 'Epsilon(3,4,5,6)*Metric(1,2) - Epsilon(2,3,4,6)*Metric(1,5) + Epsilon(1,3,4,5)*Metric(2,6) - Epsilon(1,2,3,4)*Metric(5,6)')

VVVVVV80 = Lorentz(name = 'VVVVVV80',
                   spins = [ 3, 3, 3, 3, 3, 3 ],
                   structure = 'Epsilon(2,4,5,6)*Metric(1,3) - Epsilon(2,3,4,5)*Metric(1,6) - Epsilon(1,2,4,6)*Metric(3,5) - Epsilon(1,2,3,4)*Metric(5,6)')

VVVVVV81 = Lorentz(name = 'VVVVVV81',
                   spins = [ 3, 3, 3, 3, 3, 3 ],
                   structure = 'Epsilon(1,4,5,6)*Metric(2,3) + Epsilon(1,3,4,6)*Metric(2,5) - Epsilon(1,2,4,5)*Metric(3,6) - Epsilon(1,2,3,4)*Metric(5,6)')

VVVVVV82 = Lorentz(name = 'VVVVVV82',
                   spins = [ 3, 3, 3, 3, 3, 3 ],
                   structure = 'Epsilon(1,3,5,6)*Metric(2,4) + Epsilon(1,3,4,5)*Metric(2,6) + Epsilon(1,2,3,6)*Metric(4,5) - Epsilon(1,2,3,4)*Metric(5,6)')

VVVVVV83 = Lorentz(name = 'VVVVVV83',
                   spins = [ 3, 3, 3, 3, 3, 3 ],
                   structure = 'Epsilon(2,3,5,6)*Metric(1,4) - Epsilon(2,3,4,6)*Metric(1,5) + Epsilon(1,2,3,5)*Metric(4,6) - Epsilon(1,2,3,4)*Metric(5,6)')

VVVVVV84 = Lorentz(name = 'VVVVVV84',
                   spins = [ 3, 3, 3, 3, 3, 3 ],
                   structure = 'Epsilon(1,2,5,6)*Metric(3,4) - Epsilon(1,2,4,6)*Metric(3,5) + Epsilon(1,2,3,5)*Metric(4,6) - Epsilon(1,2,3,4)*Metric(5,6)')

VVVVVV85 = Lorentz(name = 'VVVVVV85',
                   spins = [ 3, 3, 3, 3, 3, 3 ],
                   structure = 'Epsilon(3,4,5,6)*Metric(1,2) + Epsilon(2,3,4,5)*Metric(1,6) - Epsilon(1,3,4,6)*Metric(2,5) + Epsilon(1,2,3,4)*Metric(5,6)')

VVVVVV86 = Lorentz(name = 'VVVVVV86',
                   spins = [ 3, 3, 3, 3, 3, 3 ],
                   structure = 'Epsilon(1,4,5,6)*Metric(2,3) - Epsilon(1,3,4,5)*Metric(2,6) + Epsilon(1,2,4,6)*Metric(3,5) + Epsilon(1,2,3,4)*Metric(5,6)')

VVVVVV87 = Lorentz(name = 'VVVVVV87',
                   spins = [ 3, 3, 3, 3, 3, 3 ],
                   structure = 'Epsilon(2,4,5,6)*Metric(1,3) + Epsilon(2,3,4,6)*Metric(1,5) + Epsilon(1,2,4,5)*Metric(3,6) + Epsilon(1,2,3,4)*Metric(5,6)')

VVVVVV88 = Lorentz(name = 'VVVVVV88',
                   spins = [ 3, 3, 3, 3, 3, 3 ],
                   structure = 'Epsilon(2,3,5,6)*Metric(1,4) + Epsilon(2,3,4,5)*Metric(1,6) - Epsilon(1,2,3,6)*Metric(4,5) + Epsilon(1,2,3,4)*Metric(5,6)')

VVVVVV89 = Lorentz(name = 'VVVVVV89',
                   spins = [ 3, 3, 3, 3, 3, 3 ],
                   structure = 'Epsilon(1,2,5,6)*Metric(3,4) + Epsilon(1,2,4,5)*Metric(3,6) - Epsilon(1,2,3,6)*Metric(4,5) + Epsilon(1,2,3,4)*Metric(5,6)')

VVVVVV90 = Lorentz(name = 'VVVVVV90',
                   spins = [ 3, 3, 3, 3, 3, 3 ],
                   structure = 'Epsilon(1,3,5,6)*Metric(2,4) - Epsilon(1,3,4,6)*Metric(2,5) - Epsilon(1,2,3,5)*Metric(4,6) + Epsilon(1,2,3,4)*Metric(5,6)')

VVVVVV91 = Lorentz(name = 'VVVVVV91',
                   spins = [ 3, 3, 3, 3, 3, 3 ],
                   structure = 'Metric(1,4)*Metric(2,6)*Metric(3,5) + Metric(1,6)*Metric(2,3)*Metric(4,5) - Metric(1,3)*Metric(2,6)*Metric(4,5) - Metric(1,4)*Metric(2,3)*Metric(5,6)')

VVVVVV92 = Lorentz(name = 'VVVVVV92',
                   spins = [ 3, 3, 3, 3, 3, 3 ],
                   structure = 'Metric(1,4)*Metric(2,5)*Metric(3,6) + Metric(1,6)*Metric(2,3)*Metric(4,5) - Metric(1,2)*Metric(3,6)*Metric(4,5) - Metric(1,4)*Metric(2,3)*Metric(5,6)')

VVVVVV93 = Lorentz(name = 'VVVVVV93',
                   spins = [ 3, 3, 3, 3, 3, 3 ],
                   structure = 'Metric(1,4)*Metric(2,5)*Metric(3,6) + Metric(1,5)*Metric(2,3)*Metric(4,6) - Metric(1,3)*Metric(2,5)*Metric(4,6) - Metric(1,4)*Metric(2,3)*Metric(5,6)')

VVVVVV94 = Lorentz(name = 'VVVVVV94',
                   spins = [ 3, 3, 3, 3, 3, 3 ],
                   structure = 'Metric(1,4)*Metric(2,6)*Metric(3,5) + Metric(1,5)*Metric(2,3)*Metric(4,6) - Metric(1,2)*Metric(3,5)*Metric(4,6) - Metric(1,4)*Metric(2,3)*Metric(5,6)')

VVVVVV95 = Lorentz(name = 'VVVVVV95',
                   spins = [ 3, 3, 3, 3, 3, 3 ],
                   structure = 'Metric(1,6)*Metric(2,4)*Metric(3,5) - Metric(1,4)*Metric(2,6)*Metric(3,5) - Metric(1,6)*Metric(2,3)*Metric(4,5) + Metric(1,4)*Metric(2,3)*Metric(5,6)')

VVVVVV96 = Lorentz(name = 'VVVVVV96',
                   spins = [ 3, 3, 3, 3, 3, 3 ],
                   structure = 'Metric(1,6)*Metric(2,5)*Metric(3,4) - Metric(1,4)*Metric(2,5)*Metric(3,6) - Metric(1,6)*Metric(2,3)*Metric(4,5) + Metric(1,4)*Metric(2,3)*Metric(5,6)')

VVVVVV97 = Lorentz(name = 'VVVVVV97',
                   spins = [ 3, 3, 3, 3, 3, 3 ],
                   structure = 'Metric(1,5)*Metric(2,6)*Metric(3,4) - Metric(1,4)*Metric(2,6)*Metric(3,5) - Metric(1,5)*Metric(2,3)*Metric(4,6) + Metric(1,4)*Metric(2,3)*Metric(5,6)')

VVVVVV98 = Lorentz(name = 'VVVVVV98',
                   spins = [ 3, 3, 3, 3, 3, 3 ],
                   structure = 'Metric(1,5)*Metric(2,4)*Metric(3,6) - Metric(1,4)*Metric(2,5)*Metric(3,6) - Metric(1,5)*Metric(2,3)*Metric(4,6) + Metric(1,4)*Metric(2,3)*Metric(5,6)')

VVVVVV99 = Lorentz(name = 'VVVVVV99',
                   spins = [ 3, 3, 3, 3, 3, 3 ],
                   structure = 'Metric(1,6)*Metric(2,4)*Metric(3,5) - Metric(1,4)*Metric(2,6)*Metric(3,5) + Metric(1,3)*Metric(2,6)*Metric(4,5) - Metric(1,3)*Metric(2,4)*Metric(5,6)')

VVVVVV100 = Lorentz(name = 'VVVVVV100',
                    spins = [ 3, 3, 3, 3, 3, 3 ],
                    structure = 'Metric(1,6)*Metric(2,4)*Metric(3,5) - Metric(1,6)*Metric(2,3)*Metric(4,5) + Metric(1,3)*Metric(2,6)*Metric(4,5) - Metric(1,3)*Metric(2,4)*Metric(5,6)')

VVVVVV101 = Lorentz(name = 'VVVVVV101',
                    spins = [ 3, 3, 3, 3, 3, 3 ],
                    structure = 'Metric(1,5)*Metric(2,4)*Metric(3,6) + Metric(1,3)*Metric(2,6)*Metric(4,5) - Metric(1,2)*Metric(3,6)*Metric(4,5) - Metric(1,3)*Metric(2,4)*Metric(5,6)')

VVVVVV102 = Lorentz(name = 'VVVVVV102',
                    spins = [ 3, 3, 3, 3, 3, 3 ],
                    structure = 'Metric(1,5)*Metric(2,4)*Metric(3,6) - Metric(1,4)*Metric(2,5)*Metric(3,6) + Metric(1,3)*Metric(2,5)*Metric(4,6) - Metric(1,3)*Metric(2,4)*Metric(5,6)')

VVVVVV103 = Lorentz(name = 'VVVVVV103',
                    spins = [ 3, 3, 3, 3, 3, 3 ],
                    structure = 'Metric(1,5)*Metric(2,4)*Metric(3,6) - Metric(1,5)*Metric(2,3)*Metric(4,6) + Metric(1,3)*Metric(2,5)*Metric(4,6) - Metric(1,3)*Metric(2,4)*Metric(5,6)')

VVVVVV104 = Lorentz(name = 'VVVVVV104',
                    spins = [ 3, 3, 3, 3, 3, 3 ],
                    structure = 'Metric(1,6)*Metric(2,4)*Metric(3,5) + Metric(1,3)*Metric(2,5)*Metric(4,6) - Metric(1,2)*Metric(3,5)*Metric(4,6) - Metric(1,3)*Metric(2,4)*Metric(5,6)')

VVVVVV105 = Lorentz(name = 'VVVVVV105',
                    spins = [ 3, 3, 3, 3, 3, 3 ],
                    structure = 'Metric(1,6)*Metric(2,4)*Metric(3,5) - Metric(1,4)*Metric(2,6)*Metric(3,5) + Metric(1,4)*Metric(2,3)*Metric(5,6) - Metric(1,3)*Metric(2,4)*Metric(5,6)')

VVVVVV106 = Lorentz(name = 'VVVVVV106',
                    spins = [ 3, 3, 3, 3, 3, 3 ],
                    structure = 'Metric(1,5)*Metric(2,4)*Metric(3,6) - Metric(1,4)*Metric(2,5)*Metric(3,6) + Metric(1,4)*Metric(2,3)*Metric(5,6) - Metric(1,3)*Metric(2,4)*Metric(5,6)')

VVVVVV107 = Lorentz(name = 'VVVVVV107',
                    spins = [ 3, 3, 3, 3, 3, 3 ],
                    structure = 'Metric(1,6)*Metric(2,4)*Metric(3,5) - Metric(1,6)*Metric(2,3)*Metric(4,5) + Metric(1,4)*Metric(2,3)*Metric(5,6) - Metric(1,3)*Metric(2,4)*Metric(5,6)')

VVVVVV108 = Lorentz(name = 'VVVVVV108',
                    spins = [ 3, 3, 3, 3, 3, 3 ],
                    structure = 'Metric(1,6)*Metric(2,5)*Metric(3,4) - Metric(1,5)*Metric(2,6)*Metric(3,4) + Metric(1,5)*Metric(2,4)*Metric(3,6) - Metric(1,4)*Metric(2,5)*Metric(3,6) - Metric(1,6)*Metric(2,3)*Metric(4,5) + Metric(1,3)*Metric(2,6)*Metric(4,5) + Metric(1,4)*Metric(2,3)*Metric(5,6) - Metric(1,3)*Metric(2,4)*Metric(5,6)')

VVVVVV109 = Lorentz(name = 'VVVVVV109',
                    spins = [ 3, 3, 3, 3, 3, 3 ],
                    structure = 'Metric(1,5)*Metric(2,4)*Metric(3,6) - Metric(1,5)*Metric(2,3)*Metric(4,6) + Metric(1,4)*Metric(2,3)*Metric(5,6) - Metric(1,3)*Metric(2,4)*Metric(5,6)')

VVVVVV110 = Lorentz(name = 'VVVVVV110',
                    spins = [ 3, 3, 3, 3, 3, 3 ],
                    structure = 'Metric(1,6)*Metric(2,4)*Metric(3,5) - Metric(1,4)*Metric(2,5)*Metric(3,6) - Metric(1,6)*Metric(2,3)*Metric(4,5) + Metric(1,2)*Metric(3,6)*Metric(4,5) + Metric(1,3)*Metric(2,5)*Metric(4,6) - Metric(1,2)*Metric(3,5)*Metric(4,6) + Metric(1,4)*Metric(2,3)*Metric(5,6) - Metric(1,3)*Metric(2,4)*Metric(5,6)')

VVVVVV111 = Lorentz(name = 'VVVVVV111',
                    spins = [ 3, 3, 3, 3, 3, 3 ],
                    structure = 'Metric(1,5)*Metric(2,6)*Metric(3,4) - Metric(1,5)*Metric(2,4)*Metric(3,6) - Metric(1,3)*Metric(2,6)*Metric(4,5) + Metric(1,3)*Metric(2,4)*Metric(5,6)')

VVVVVV112 = Lorentz(name = 'VVVVVV112',
                    spins = [ 3, 3, 3, 3, 3, 3 ],
                    structure = 'Metric(1,6)*Metric(2,5)*Metric(3,4) - Metric(1,6)*Metric(2,4)*Metric(3,5) - Metric(1,3)*Metric(2,5)*Metric(4,6) + Metric(1,3)*Metric(2,4)*Metric(5,6)')

VVVVVV113 = Lorentz(name = 'VVVVVV113',
                    spins = [ 3, 3, 3, 3, 3, 3 ],
                    structure = 'Metric(1,4)*Metric(2,6)*Metric(3,5) - Metric(1,3)*Metric(2,6)*Metric(4,5) - Metric(1,4)*Metric(2,3)*Metric(5,6) + Metric(1,3)*Metric(2,4)*Metric(5,6)')

VVVVVV114 = Lorentz(name = 'VVVVVV114',
                    spins = [ 3, 3, 3, 3, 3, 3 ],
                    structure = 'Metric(1,6)*Metric(2,3)*Metric(4,5) - Metric(1,3)*Metric(2,6)*Metric(4,5) - Metric(1,4)*Metric(2,3)*Metric(5,6) + Metric(1,3)*Metric(2,4)*Metric(5,6)')

VVVVVV115 = Lorentz(name = 'VVVVVV115',
                    spins = [ 3, 3, 3, 3, 3, 3 ],
                    structure = 'Metric(1,4)*Metric(2,5)*Metric(3,6) - Metric(1,3)*Metric(2,5)*Metric(4,6) - Metric(1,4)*Metric(2,3)*Metric(5,6) + Metric(1,3)*Metric(2,4)*Metric(5,6)')

VVVVVV116 = Lorentz(name = 'VVVVVV116',
                    spins = [ 3, 3, 3, 3, 3, 3 ],
                    structure = 'Metric(1,5)*Metric(2,3)*Metric(4,6) - Metric(1,3)*Metric(2,5)*Metric(4,6) - Metric(1,4)*Metric(2,3)*Metric(5,6) + Metric(1,3)*Metric(2,4)*Metric(5,6)')

VVVVVV117 = Lorentz(name = 'VVVVVV117',
                    spins = [ 3, 3, 3, 3, 3, 3 ],
                    structure = 'Metric(1,6)*Metric(2,5)*Metric(3,4) - Metric(1,5)*Metric(2,6)*Metric(3,4) - Metric(1,6)*Metric(2,4)*Metric(3,5) + Metric(1,4)*Metric(2,6)*Metric(3,5) + Metric(1,5)*Metric(2,3)*Metric(4,6) - Metric(1,3)*Metric(2,5)*Metric(4,6) - Metric(1,4)*Metric(2,3)*Metric(5,6) + Metric(1,3)*Metric(2,4)*Metric(5,6)')

VVVVVV118 = Lorentz(name = 'VVVVVV118',
                    spins = [ 3, 3, 3, 3, 3, 3 ],
                    structure = 'Metric(1,4)*Metric(2,6)*Metric(3,5) - Metric(1,5)*Metric(2,4)*Metric(3,6) - Metric(1,3)*Metric(2,6)*Metric(4,5) + Metric(1,2)*Metric(3,6)*Metric(4,5) + Metric(1,5)*Metric(2,3)*Metric(4,6) - Metric(1,2)*Metric(3,5)*Metric(4,6) - Metric(1,4)*Metric(2,3)*Metric(5,6) + Metric(1,3)*Metric(2,4)*Metric(5,6)')

VVVVVV119 = Lorentz(name = 'VVVVVV119',
                    spins = [ 3, 3, 3, 3, 3, 3 ],
                    structure = 'Metric(1,6)*Metric(2,5)*Metric(3,4) + Metric(1,5)*Metric(2,6)*Metric(3,4) + Metric(1,6)*Metric(2,4)*Metric(3,5) + Metric(1,4)*Metric(2,6)*Metric(3,5) + Metric(1,5)*Metric(2,4)*Metric(3,6) + Metric(1,4)*Metric(2,5)*Metric(3,6) + Metric(1,6)*Metric(2,3)*Metric(4,5) + Metric(1,3)*Metric(2,6)*Metric(4,5) + Metric(1,2)*Metric(3,6)*Metric(4,5) + Metric(1,5)*Metric(2,3)*Metric(4,6) + Metric(1,3)*Metric(2,5)*Metric(4,6) + Metric(1,2)*Metric(3,5)*Metric(4,6) - 4*Metric(1,4)*Metric(2,3)*Metric(5,6) - 4*Metric(1,3)*Metric(2,4)*Metric(5,6) - 4*Metric(1,2)*Metric(3,4)*Metric(5,6)')

VVVVVV120 = Lorentz(name = 'VVVVVV120',
                    spins = [ 3, 3, 3, 3, 3, 3 ],
                    structure = 'Metric(1,6)*Metric(2,5)*Metric(3,4) + Metric(1,5)*Metric(2,6)*Metric(3,4) + Metric(1,6)*Metric(2,4)*Metric(3,5) + Metric(1,4)*Metric(2,6)*Metric(3,5) + Metric(1,5)*Metric(2,4)*Metric(3,6) + Metric(1,4)*Metric(2,5)*Metric(3,6) + Metric(1,6)*Metric(2,3)*Metric(4,5) + Metric(1,3)*Metric(2,6)*Metric(4,5) - 4*Metric(1,2)*Metric(3,6)*Metric(4,5) + Metric(1,5)*Metric(2,3)*Metric(4,6) + Metric(1,3)*Metric(2,5)*Metric(4,6) - 4*Metric(1,2)*Metric(3,5)*Metric(4,6) + Metric(1,4)*Metric(2,3)*Metric(5,6) + Metric(1,3)*Metric(2,4)*Metric(5,6) - 4*Metric(1,2)*Metric(3,4)*Metric(5,6)')

VVVVVV121 = Lorentz(name = 'VVVVVV121',
                    spins = [ 3, 3, 3, 3, 3, 3 ],
                    structure = 'Metric(1,6)*Metric(2,5)*Metric(3,4) + Metric(1,5)*Metric(2,6)*Metric(3,4) - (5*Metric(1,6)*Metric(2,4)*Metric(3,5))/6. - (5*Metric(1,4)*Metric(2,6)*Metric(3,5))/6. - (5*Metric(1,5)*Metric(2,4)*Metric(3,6))/6. - (5*Metric(1,4)*Metric(2,5)*Metric(3,6))/6. - (5*Metric(1,6)*Metric(2,3)*Metric(4,5))/6. - (5*Metric(1,3)*Metric(2,6)*Metric(4,5))/6. + (8*Metric(1,2)*Metric(3,6)*Metric(4,5))/3. - (5*Metric(1,5)*Metric(2,3)*Metric(4,6))/6. - (5*Metric(1,3)*Metric(2,5)*Metric(4,6))/6. + (8*Metric(1,2)*Metric(3,5)*Metric(4,6))/3. + Metric(1,4)*Metric(2,3)*Metric(5,6) + Metric(1,3)*Metric(2,4)*Metric(5,6) - (8*Metric(1,2)*Metric(3,4)*Metric(5,6))/3.')

VVVVVV122 = Lorentz(name = 'VVVVVV122',
                    spins = [ 3, 3, 3, 3, 3, 3 ],
                    structure = 'Metric(1,6)*Metric(2,5)*Metric(3,4) + Metric(1,5)*Metric(2,6)*Metric(3,4) - (5*Metric(1,6)*Metric(2,4)*Metric(3,5))/6. - (5*Metric(1,4)*Metric(2,6)*Metric(3,5))/6. - (5*Metric(1,5)*Metric(2,4)*Metric(3,6))/6. - (5*Metric(1,4)*Metric(2,5)*Metric(3,6))/6. - (5*Metric(1,6)*Metric(2,3)*Metric(4,5))/6. - (5*Metric(1,3)*Metric(2,6)*Metric(4,5))/6. + Metric(1,2)*Metric(3,6)*Metric(4,5) - (5*Metric(1,5)*Metric(2,3)*Metric(4,6))/6. - (5*Metric(1,3)*Metric(2,5)*Metric(4,6))/6. + Metric(1,2)*Metric(3,5)*Metric(4,6) + (8*Metric(1,4)*Metric(2,3)*Metric(5,6))/3. + (8*Metric(1,3)*Metric(2,4)*Metric(5,6))/3. - (8*Metric(1,2)*Metric(3,4)*Metric(5,6))/3.')

VVVVVV123 = Lorentz(name = 'VVVVVV123',
                    spins = [ 3, 3, 3, 3, 3, 3 ],
                    structure = 'Metric(1,6)*Metric(2,5)*Metric(3,4) + Metric(1,5)*Metric(2,6)*Metric(3,4) - (Metric(1,6)*Metric(2,4)*Metric(3,5))/2. - (Metric(1,4)*Metric(2,6)*Metric(3,5))/2. - (Metric(1,5)*Metric(2,4)*Metric(3,6))/2. - (Metric(1,4)*Metric(2,5)*Metric(3,6))/2. - (Metric(1,6)*Metric(2,3)*Metric(4,5))/2. - (Metric(1,3)*Metric(2,6)*Metric(4,5))/2. + Metric(1,2)*Metric(3,6)*Metric(4,5) - (Metric(1,5)*Metric(2,3)*Metric(4,6))/2. - (Metric(1,3)*Metric(2,5)*Metric(4,6))/2. + Metric(1,2)*Metric(3,5)*Metric(4,6) + Metric(1,4)*Metric(2,3)*Metric(5,6) + Metric(1,3)*Metric(2,4)*Metric(5,6) - 2*Metric(1,2)*Metric(3,4)*Metric(5,6)')

VVVVVV124 = Lorentz(name = 'VVVVVV124',
                    spins = [ 3, 3, 3, 3, 3, 3 ],
                    structure = 'Metric(1,5)*Metric(2,6)*Metric(3,4) - Metric(1,5)*Metric(2,4)*Metric(3,6) + Metric(1,2)*Metric(3,6)*Metric(4,5) - Metric(1,2)*Metric(3,4)*Metric(5,6)')

VVVVVV125 = Lorentz(name = 'VVVVVV125',
                    spins = [ 3, 3, 3, 3, 3, 3 ],
                    structure = 'Metric(1,6)*Metric(2,5)*Metric(3,4) - Metric(1,4)*Metric(2,5)*Metric(3,6) + Metric(1,2)*Metric(3,6)*Metric(4,5) - Metric(1,2)*Metric(3,4)*Metric(5,6)')

VVVVVV126 = Lorentz(name = 'VVVVVV126',
                    spins = [ 3, 3, 3, 3, 3, 3 ],
                    structure = 'Metric(1,6)*Metric(2,5)*Metric(3,4) - Metric(1,6)*Metric(2,3)*Metric(4,5) + Metric(1,2)*Metric(3,6)*Metric(4,5) - Metric(1,2)*Metric(3,4)*Metric(5,6)')

VVVVVV127 = Lorentz(name = 'VVVVVV127',
                    spins = [ 3, 3, 3, 3, 3, 3 ],
                    structure = 'Metric(1,5)*Metric(2,6)*Metric(3,4) - Metric(1,3)*Metric(2,6)*Metric(4,5) + Metric(1,2)*Metric(3,6)*Metric(4,5) - Metric(1,2)*Metric(3,4)*Metric(5,6)')

VVVVVV128 = Lorentz(name = 'VVVVVV128',
                    spins = [ 3, 3, 3, 3, 3, 3 ],
                    structure = 'Metric(1,6)*Metric(2,5)*Metric(3,4) - Metric(1,6)*Metric(2,4)*Metric(3,5) + Metric(1,2)*Metric(3,5)*Metric(4,6) - Metric(1,2)*Metric(3,4)*Metric(5,6)')

VVVVVV129 = Lorentz(name = 'VVVVVV129',
                    spins = [ 3, 3, 3, 3, 3, 3 ],
                    structure = 'Metric(1,5)*Metric(2,6)*Metric(3,4) - Metric(1,4)*Metric(2,6)*Metric(3,5) + Metric(1,2)*Metric(3,5)*Metric(4,6) - Metric(1,2)*Metric(3,4)*Metric(5,6)')

VVVVVV130 = Lorentz(name = 'VVVVVV130',
                    spins = [ 3, 3, 3, 3, 3, 3 ],
                    structure = 'Metric(1,5)*Metric(2,6)*Metric(3,4) - Metric(1,5)*Metric(2,3)*Metric(4,6) + Metric(1,2)*Metric(3,5)*Metric(4,6) - Metric(1,2)*Metric(3,4)*Metric(5,6)')

VVVVVV131 = Lorentz(name = 'VVVVVV131',
                    spins = [ 3, 3, 3, 3, 3, 3 ],
                    structure = 'Metric(1,6)*Metric(2,5)*Metric(3,4) - Metric(1,3)*Metric(2,5)*Metric(4,6) + Metric(1,2)*Metric(3,5)*Metric(4,6) - Metric(1,2)*Metric(3,4)*Metric(5,6)')

VVVVVV132 = Lorentz(name = 'VVVVVV132',
                    spins = [ 3, 3, 3, 3, 3, 3 ],
                    structure = 'Metric(1,5)*Metric(2,6)*Metric(3,4) - Metric(1,4)*Metric(2,6)*Metric(3,5) + Metric(1,4)*Metric(2,3)*Metric(5,6) - Metric(1,2)*Metric(3,4)*Metric(5,6)')

VVVVVV133 = Lorentz(name = 'VVVVVV133',
                    spins = [ 3, 3, 3, 3, 3, 3 ],
                    structure = 'Metric(1,6)*Metric(2,5)*Metric(3,4) - Metric(1,4)*Metric(2,5)*Metric(3,6) + Metric(1,4)*Metric(2,3)*Metric(5,6) - Metric(1,2)*Metric(3,4)*Metric(5,6)')

VVVVVV134 = Lorentz(name = 'VVVVVV134',
                    spins = [ 3, 3, 3, 3, 3, 3 ],
                    structure = 'Metric(1,6)*Metric(2,5)*Metric(3,4) - Metric(1,6)*Metric(2,3)*Metric(4,5) + Metric(1,4)*Metric(2,3)*Metric(5,6) - Metric(1,2)*Metric(3,4)*Metric(5,6)')

VVVVVV135 = Lorentz(name = 'VVVVVV135',
                    spins = [ 3, 3, 3, 3, 3, 3 ],
                    structure = 'Metric(1,5)*Metric(2,6)*Metric(3,4) + Metric(1,6)*Metric(2,4)*Metric(3,5) - Metric(1,4)*Metric(2,6)*Metric(3,5) - Metric(1,5)*Metric(2,4)*Metric(3,6) - Metric(1,6)*Metric(2,3)*Metric(4,5) + Metric(1,2)*Metric(3,6)*Metric(4,5) + Metric(1,4)*Metric(2,3)*Metric(5,6) - Metric(1,2)*Metric(3,4)*Metric(5,6)')

VVVVVV136 = Lorentz(name = 'VVVVVV136',
                    spins = [ 3, 3, 3, 3, 3, 3 ],
                    structure = 'Metric(1,5)*Metric(2,6)*Metric(3,4) - Metric(1,5)*Metric(2,3)*Metric(4,6) + Metric(1,4)*Metric(2,3)*Metric(5,6) - Metric(1,2)*Metric(3,4)*Metric(5,6)')

VVVVVV137 = Lorentz(name = 'VVVVVV137',
                    spins = [ 3, 3, 3, 3, 3, 3 ],
                    structure = 'Metric(1,5)*Metric(2,6)*Metric(3,4) - Metric(1,4)*Metric(2,5)*Metric(3,6) - Metric(1,3)*Metric(2,6)*Metric(4,5) + Metric(1,2)*Metric(3,6)*Metric(4,5) - Metric(1,5)*Metric(2,3)*Metric(4,6) + Metric(1,3)*Metric(2,5)*Metric(4,6) + Metric(1,4)*Metric(2,3)*Metric(5,6) - Metric(1,2)*Metric(3,4)*Metric(5,6)')

VVVVVV138 = Lorentz(name = 'VVVVVV138',
                    spins = [ 3, 3, 3, 3, 3, 3 ],
                    structure = 'Metric(1,6)*Metric(2,5)*Metric(3,4) - Metric(1,6)*Metric(2,4)*Metric(3,5) + Metric(1,5)*Metric(2,4)*Metric(3,6) - Metric(1,4)*Metric(2,5)*Metric(3,6) - Metric(1,5)*Metric(2,3)*Metric(4,6) + Metric(1,2)*Metric(3,5)*Metric(4,6) + Metric(1,4)*Metric(2,3)*Metric(5,6) - Metric(1,2)*Metric(3,4)*Metric(5,6)')

VVVVVV139 = Lorentz(name = 'VVVVVV139',
                    spins = [ 3, 3, 3, 3, 3, 3 ],
                    structure = 'Metric(1,6)*Metric(2,5)*Metric(3,4) - Metric(1,4)*Metric(2,6)*Metric(3,5) - Metric(1,6)*Metric(2,3)*Metric(4,5) + Metric(1,3)*Metric(2,6)*Metric(4,5) - Metric(1,3)*Metric(2,5)*Metric(4,6) + Metric(1,2)*Metric(3,5)*Metric(4,6) + Metric(1,4)*Metric(2,3)*Metric(5,6) - Metric(1,2)*Metric(3,4)*Metric(5,6)')

VVVVVV140 = Lorentz(name = 'VVVVVV140',
                    spins = [ 3, 3, 3, 3, 3, 3 ],
                    structure = 'Metric(1,6)*Metric(2,5)*Metric(3,4) - Metric(1,6)*Metric(2,4)*Metric(3,5) + Metric(1,3)*Metric(2,4)*Metric(5,6) - Metric(1,2)*Metric(3,4)*Metric(5,6)')

VVVVVV141 = Lorentz(name = 'VVVVVV141',
                    spins = [ 3, 3, 3, 3, 3, 3 ],
                    structure = 'Metric(1,5)*Metric(2,6)*Metric(3,4) - Metric(1,5)*Metric(2,4)*Metric(3,6) + Metric(1,3)*Metric(2,4)*Metric(5,6) - Metric(1,2)*Metric(3,4)*Metric(5,6)')

VVVVVV142 = Lorentz(name = 'VVVVVV142',
                    spins = [ 3, 3, 3, 3, 3, 3 ],
                    structure = 'Metric(1,5)*Metric(2,6)*Metric(3,4) - Metric(1,3)*Metric(2,6)*Metric(4,5) + Metric(1,3)*Metric(2,4)*Metric(5,6) - Metric(1,2)*Metric(3,4)*Metric(5,6)')

VVVVVV143 = Lorentz(name = 'VVVVVV143',
                    spins = [ 3, 3, 3, 3, 3, 3 ],
                    structure = 'Metric(1,6)*Metric(2,5)*Metric(3,4) - Metric(1,6)*Metric(2,4)*Metric(3,5) + Metric(1,4)*Metric(2,6)*Metric(3,5) - Metric(1,4)*Metric(2,5)*Metric(3,6) - Metric(1,3)*Metric(2,6)*Metric(4,5) + Metric(1,2)*Metric(3,6)*Metric(4,5) + Metric(1,3)*Metric(2,4)*Metric(5,6) - Metric(1,2)*Metric(3,4)*Metric(5,6)')

VVVVVV144 = Lorentz(name = 'VVVVVV144',
                    spins = [ 3, 3, 3, 3, 3, 3 ],
                    structure = 'Metric(1,6)*Metric(2,5)*Metric(3,4) - Metric(1,3)*Metric(2,5)*Metric(4,6) + Metric(1,3)*Metric(2,4)*Metric(5,6) - Metric(1,2)*Metric(3,4)*Metric(5,6)')

VVVVVV145 = Lorentz(name = 'VVVVVV145',
                    spins = [ 3, 3, 3, 3, 3, 3 ],
                    structure = 'Metric(1,6)*Metric(2,5)*Metric(3,4) - Metric(1,5)*Metric(2,4)*Metric(3,6) - Metric(1,6)*Metric(2,3)*Metric(4,5) + Metric(1,2)*Metric(3,6)*Metric(4,5) + Metric(1,5)*Metric(2,3)*Metric(4,6) - Metric(1,3)*Metric(2,5)*Metric(4,6) + Metric(1,3)*Metric(2,4)*Metric(5,6) - Metric(1,2)*Metric(3,4)*Metric(5,6)')

VVVVVV146 = Lorentz(name = 'VVVVVV146',
                    spins = [ 3, 3, 3, 3, 3, 3 ],
                    structure = 'Metric(1,5)*Metric(2,6)*Metric(3,4) - Metric(1,6)*Metric(2,4)*Metric(3,5) + Metric(1,6)*Metric(2,3)*Metric(4,5) - Metric(1,3)*Metric(2,6)*Metric(4,5) - Metric(1,5)*Metric(2,3)*Metric(4,6) + Metric(1,2)*Metric(3,5)*Metric(4,6) + Metric(1,3)*Metric(2,4)*Metric(5,6) - Metric(1,2)*Metric(3,4)*Metric(5,6)')

VVVVVV147 = Lorentz(name = 'VVVVVV147',
                    spins = [ 3, 3, 3, 3, 3, 3 ],
                    structure = 'Metric(1,5)*Metric(2,6)*Metric(3,4) - Metric(1,4)*Metric(2,6)*Metric(3,5) - Metric(1,5)*Metric(2,4)*Metric(3,6) + Metric(1,4)*Metric(2,5)*Metric(3,6) - Metric(1,3)*Metric(2,5)*Metric(4,6) + Metric(1,2)*Metric(3,5)*Metric(4,6) + Metric(1,3)*Metric(2,4)*Metric(5,6) - Metric(1,2)*Metric(3,4)*Metric(5,6)')

VVVVVV148 = Lorentz(name = 'VVVVVV148',
                    spins = [ 3, 3, 3, 3, 3, 3 ],
                    structure = 'Metric(1,6)*Metric(2,5)*Metric(3,4) + Metric(1,5)*Metric(2,6)*Metric(3,4) + Metric(1,6)*Metric(2,4)*Metric(3,5) + Metric(1,4)*Metric(2,6)*Metric(3,5) + Metric(1,5)*Metric(2,4)*Metric(3,6) + Metric(1,4)*Metric(2,5)*Metric(3,6) - (2*Metric(1,6)*Metric(2,3)*Metric(4,5))/3. - (2*Metric(1,3)*Metric(2,6)*Metric(4,5))/3. - (2*Metric(1,2)*Metric(3,6)*Metric(4,5))/3. - (2*Metric(1,5)*Metric(2,3)*Metric(4,6))/3. - (2*Metric(1,3)*Metric(2,5)*Metric(4,6))/3. - (2*Metric(1,2)*Metric(3,5)*Metric(4,6))/3. - (2*Metric(1,4)*Metric(2,3)*Metric(5,6))/3. - (2*Metric(1,3)*Metric(2,4)*Metric(5,6))/3. - (2*Metric(1,2)*Metric(3,4)*Metric(5,6))/3.')

VVVVVV149 = Lorentz(name = 'VVVVVV149',
                    spins = [ 3, 3, 3, 3, 3, 3 ],
                    structure = 'Metric(1,6)*Metric(2,5)*Metric(3,4) - (Metric(1,5)*Metric(2,6)*Metric(3,4))/2. + Metric(1,6)*Metric(2,4)*Metric(3,5) - (Metric(1,4)*Metric(2,6)*Metric(3,5))/2. - (Metric(1,5)*Metric(2,4)*Metric(3,6))/2. - (Metric(1,4)*Metric(2,5)*Metric(3,6))/2. - 2*Metric(1,6)*Metric(2,3)*Metric(4,5) + Metric(1,3)*Metric(2,6)*Metric(4,5) + Metric(1,2)*Metric(3,6)*Metric(4,5) + Metric(1,5)*Metric(2,3)*Metric(4,6) - (Metric(1,3)*Metric(2,5)*Metric(4,6))/2. - (Metric(1,2)*Metric(3,5)*Metric(4,6))/2. + Metric(1,4)*Metric(2,3)*Metric(5,6) - (Metric(1,3)*Metric(2,4)*Metric(5,6))/2. - (Metric(1,2)*Metric(3,4)*Metric(5,6))/2.')

VVVVVV150 = Lorentz(name = 'VVVVVV150',
                    spins = [ 3, 3, 3, 3, 3, 3 ],
                    structure = 'Metric(1,6)*Metric(2,5)*Metric(3,4) - (5*Metric(1,5)*Metric(2,6)*Metric(3,4))/16. + Metric(1,6)*Metric(2,4)*Metric(3,5) - (5*Metric(1,4)*Metric(2,6)*Metric(3,5))/16. - (5*Metric(1,5)*Metric(2,4)*Metric(3,6))/16. - (5*Metric(1,4)*Metric(2,5)*Metric(3,6))/16. - Metric(1,6)*Metric(2,3)*Metric(4,5) + (3*Metric(1,3)*Metric(2,6)*Metric(4,5))/8. + (3*Metric(1,2)*Metric(3,6)*Metric(4,5))/8. + (3*Metric(1,5)*Metric(2,3)*Metric(4,6))/8. - (5*Metric(1,3)*Metric(2,5)*Metric(4,6))/16. - (5*Metric(1,2)*Metric(3,5)*Metric(4,6))/16. + (3*Metric(1,4)*Metric(2,3)*Metric(5,6))/8. - (5*Metric(1,3)*Metric(2,4)*Metric(5,6))/16. - (5*Metric(1,2)*Metric(3,4)*Metric(5,6))/16.')

VVVVVV151 = Lorentz(name = 'VVVVVV151',
                    spins = [ 3, 3, 3, 3, 3, 3 ],
                    structure = 'Metric(1,4)*Metric(2,5)*Metric(3,6) - Metric(1,2)*Metric(3,6)*Metric(4,5) - Metric(1,4)*Metric(2,3)*Metric(5,6) + Metric(1,2)*Metric(3,4)*Metric(5,6)')

VVVVVV152 = Lorentz(name = 'VVVVVV152',
                    spins = [ 3, 3, 3, 3, 3, 3 ],
                    structure = 'Metric(1,6)*Metric(2,3)*Metric(4,5) - Metric(1,2)*Metric(3,6)*Metric(4,5) - Metric(1,4)*Metric(2,3)*Metric(5,6) + Metric(1,2)*Metric(3,4)*Metric(5,6)')

VVVVVV153 = Lorentz(name = 'VVVVVV153',
                    spins = [ 3, 3, 3, 3, 3, 3 ],
                    structure = 'Metric(1,4)*Metric(2,6)*Metric(3,5) - Metric(1,2)*Metric(3,5)*Metric(4,6) - Metric(1,4)*Metric(2,3)*Metric(5,6) + Metric(1,2)*Metric(3,4)*Metric(5,6)')

VVVVVV154 = Lorentz(name = 'VVVVVV154',
                    spins = [ 3, 3, 3, 3, 3, 3 ],
                    structure = 'Metric(1,5)*Metric(2,3)*Metric(4,6) - Metric(1,2)*Metric(3,5)*Metric(4,6) - Metric(1,4)*Metric(2,3)*Metric(5,6) + Metric(1,2)*Metric(3,4)*Metric(5,6)')

VVVVVV155 = Lorentz(name = 'VVVVVV155',
                    spins = [ 3, 3, 3, 3, 3, 3 ],
                    structure = 'Metric(1,5)*Metric(2,4)*Metric(3,6) - Metric(1,2)*Metric(3,6)*Metric(4,5) - Metric(1,3)*Metric(2,4)*Metric(5,6) + Metric(1,2)*Metric(3,4)*Metric(5,6)')

VVVVVV156 = Lorentz(name = 'VVVVVV156',
                    spins = [ 3, 3, 3, 3, 3, 3 ],
                    structure = 'Metric(1,3)*Metric(2,6)*Metric(4,5) - Metric(1,2)*Metric(3,6)*Metric(4,5) - Metric(1,3)*Metric(2,4)*Metric(5,6) + Metric(1,2)*Metric(3,4)*Metric(5,6)')

VVVVVV157 = Lorentz(name = 'VVVVVV157',
                    spins = [ 3, 3, 3, 3, 3, 3 ],
                    structure = 'Metric(1,6)*Metric(2,4)*Metric(3,5) - Metric(1,2)*Metric(3,5)*Metric(4,6) - Metric(1,3)*Metric(2,4)*Metric(5,6) + Metric(1,2)*Metric(3,4)*Metric(5,6)')

VVVVVV158 = Lorentz(name = 'VVVVVV158',
                    spins = [ 3, 3, 3, 3, 3, 3 ],
                    structure = 'Metric(1,3)*Metric(2,5)*Metric(4,6) - Metric(1,2)*Metric(3,5)*Metric(4,6) - Metric(1,3)*Metric(2,4)*Metric(5,6) + Metric(1,2)*Metric(3,4)*Metric(5,6)')

VVVVVV159 = Lorentz(name = 'VVVVVV159',
                    spins = [ 3, 3, 3, 3, 3, 3 ],
                    structure = 'Metric(1,6)*Metric(2,5)*Metric(3,4) + Metric(1,5)*Metric(2,6)*Metric(3,4) - (Metric(1,6)*Metric(2,4)*Metric(3,5))/4. - (Metric(1,4)*Metric(2,6)*Metric(3,5))/4. - (Metric(1,5)*Metric(2,4)*Metric(3,6))/4. - (Metric(1,4)*Metric(2,5)*Metric(3,6))/4. - (Metric(1,6)*Metric(2,3)*Metric(4,5))/4. - (Metric(1,3)*Metric(2,6)*Metric(4,5))/4. - (Metric(1,2)*Metric(3,6)*Metric(4,5))/4. - (Metric(1,5)*Metric(2,3)*Metric(4,6))/4. - (Metric(1,3)*Metric(2,5)*Metric(4,6))/4. - (Metric(1,2)*Metric(3,5)*Metric(4,6))/4. - (Metric(1,4)*Metric(2,3)*Metric(5,6))/4. - (Metric(1,3)*Metric(2,4)*Metric(5,6))/4. + Metric(1,2)*Metric(3,4)*Metric(5,6)')

VVVVVV160 = Lorentz(name = 'VVVVVV160',
                    spins = [ 3, 3, 3, 3, 3, 3 ],
                    structure = 'Metric(1,6)*Metric(2,5)*Metric(3,4) + Metric(1,5)*Metric(2,6)*Metric(3,4) + Metric(1,6)*Metric(2,4)*Metric(3,5) + Metric(1,4)*Metric(2,6)*Metric(3,5) + Metric(1,5)*Metric(2,4)*Metric(3,6) + Metric(1,4)*Metric(2,5)*Metric(3,6) - 4*Metric(1,6)*Metric(2,3)*Metric(4,5) + Metric(1,3)*Metric(2,6)*Metric(4,5) + Metric(1,2)*Metric(3,6)*Metric(4,5) - 4*Metric(1,5)*Metric(2,3)*Metric(4,6) + Metric(1,3)*Metric(2,5)*Metric(4,6) + Metric(1,2)*Metric(3,5)*Metric(4,6) - 4*Metric(1,4)*Metric(2,3)*Metric(5,6) + Metric(1,3)*Metric(2,4)*Metric(5,6) + Metric(1,2)*Metric(3,4)*Metric(5,6)')

VVVVVV161 = Lorentz(name = 'VVVVVV161',
                    spins = [ 3, 3, 3, 3, 3, 3 ],
                    structure = 'Metric(1,6)*Metric(2,5)*Metric(3,4) + Metric(1,5)*Metric(2,6)*Metric(3,4) + Metric(1,6)*Metric(2,4)*Metric(3,5) + Metric(1,4)*Metric(2,6)*Metric(3,5) + Metric(1,5)*Metric(2,4)*Metric(3,6) + Metric(1,4)*Metric(2,5)*Metric(3,6) - 4*Metric(1,6)*Metric(2,3)*Metric(4,5) - 4*Metric(1,3)*Metric(2,6)*Metric(4,5) - 4*Metric(1,2)*Metric(3,6)*Metric(4,5) + Metric(1,5)*Metric(2,3)*Metric(4,6) + Metric(1,3)*Metric(2,5)*Metric(4,6) + Metric(1,2)*Metric(3,5)*Metric(4,6) + Metric(1,4)*Metric(2,3)*Metric(5,6) + Metric(1,3)*Metric(2,4)*Metric(5,6) + Metric(1,2)*Metric(3,4)*Metric(5,6)')

