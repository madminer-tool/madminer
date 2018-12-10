# This file was automatically created by FeynRules 2.3.24
# Mathematica version: 11.0.1 for Linux x86 (64-bit) (September 21, 2016)
# Date: Fri 11 Aug 2017 09:48:14


from object_library import all_couplings, Coupling

from function_library import complexconjugate, re, im, csc, sec, acsc, asec, cot



GC_1 = Coupling(name = 'GC_1',
                value = '-(AH*complex(0,1))',
                order = {'QED':2})

GC_2 = Coupling(name = 'GC_2',
                value = '(ee*complex(0,1))/3.',
                order = {'QED':1})

GC_3 = Coupling(name = 'GC_3',
                value = '(-2*ee*complex(0,1))/3.',
                order = {'QED':1})

GC_4 = Coupling(name = 'GC_4',
                value = '-(ee*complex(0,1))',
                order = {'QED':1})

GC_5 = Coupling(name = 'GC_5',
                value = 'ee*complex(0,1)',
                order = {'QED':1})

GC_6 = Coupling(name = 'GC_6',
                value = 'ee**2*complex(0,1)',
                order = {'QED':2})

GC_7 = Coupling(name = 'GC_7',
                value = '-(complex(0,1)*G)',
                order = {'QCD':1})

GC_8 = Coupling(name = 'GC_8',
                value = 'G',
                order = {'QCD':1})

GC_9 = Coupling(name = 'GC_9',
                value = 'complex(0,1)*G**2',
                order = {'QCD':2})

GC_10 = Coupling(name = 'GC_10',
                 value = '-(complex(0,1)*GH)',
                 order = {'QCD':2})

GC_11 = Coupling(name = 'GC_11',
                 value = 'G*GH',
                 order = {'QCD':3})

GC_12 = Coupling(name = 'GC_12',
                 value = 'complex(0,1)*G**2*GH',
                 order = {'QCD':4})

GC_13 = Coupling(name = 'GC_13',
                 value = '-(ee**2*FBB*complex(0,1)) - ee**2*complex(0,1)*FWW + ee**2*imFBB + ee**2*imFWW',
                 order = {'ANO':1,'QED':2})

GC_14 = Coupling(name = 'GC_14',
                 value = '(ee**2*FBBtilde*complex(0,1))/2. + (ee**2*complex(0,1)*FWWtilde)/2. - (ee**2*imFBBtilde)/2. - (ee**2*imFWWtilde)/2.',
                 order = {'ANO':1,'QED':2})

GC_15 = Coupling(name = 'GC_15',
                 value = '-6*complex(0,1)*lam',
                 order = {'QED':2})

GC_16 = Coupling(name = 'GC_16',
                 value = '-(ee**4*complex(0,1)*FW)/(2.*sw**4) - (ee**4*complex(0,1)*FWW)/sw**4 + (ee**4*imFW)/(2.*sw**4) + (ee**4*imFWW)/sw**4',
                 order = {'ANO':1,'QED':4})

GC_17 = Coupling(name = 'GC_17',
                 value = '-(cw*ee**3*complex(0,1)*FW)/(4.*sw**3) - (cw*ee**3*complex(0,1)*FWW)/sw**3 + (cw*ee**3*imFW)/(4.*sw**3) + (cw*ee**3*imFWW)/sw**3',
                 order = {'ANO':1,'QED':3})

GC_18 = Coupling(name = 'GC_18',
                 value = '(cw*ee**3*complex(0,1)*FWWtilde)/sw**3 - (cw*ee**3*imFWWtilde)/sw**3',
                 order = {'ANO':1,'QED':3})

GC_19 = Coupling(name = 'GC_19',
                 value = '-(ee**3*FB*complex(0,1))/(4.*sw**2) + (ee**3*imFB)/(4.*sw**2)',
                 order = {'ANO':1,'QED':3})

GC_20 = Coupling(name = 'GC_20',
                 value = '-(ee**2*complex(0,1)*FW)/(4.*sw**2) + (ee**2*imFW)/(4.*sw**2)',
                 order = {'ANO':1,'QED':2})

GC_21 = Coupling(name = 'GC_21',
                 value = '-(ee**3*complex(0,1)*FW)/(4.*sw**2) + (ee**3*imFW)/(4.*sw**2)',
                 order = {'ANO':1,'QED':3})

GC_22 = Coupling(name = 'GC_22',
                 value = '(cw**2*ee**4*complex(0,1)*FW)/(2.*sw**4) + (cw**2*ee**4*complex(0,1)*FWW)/sw**4 - (cw**2*ee**4*imFW)/(2.*sw**4) - (cw**2*ee**4*imFWW)/sw**4 + (ee**4*complex(0,1)*FW)/(2.*sw**2) - (ee**4*imFW)/(2.*sw**2)',
                 order = {'ANO':1,'QED':4})

GC_23 = Coupling(name = 'GC_23',
                 value = '-((ee**2*complex(0,1)*FWW)/sw**2) + (ee**2*imFWW)/sw**2',
                 order = {'ANO':1,'QED':2})

GC_24 = Coupling(name = 'GC_24',
                 value = '-((ee**3*complex(0,1)*FWW)/sw**2) + (ee**3*imFWW)/sw**2',
                 order = {'ANO':1,'QED':3})

GC_25 = Coupling(name = 'GC_25',
                 value = '(ee**4*complex(0,1)*FWW)/sw**2 - (ee**4*imFWW)/sw**2',
                 order = {'ANO':1,'QED':4})

GC_26 = Coupling(name = 'GC_26',
                 value = '(ee**2*complex(0,1)*FWWtilde)/(2.*sw**2) - (ee**2*imFWWtilde)/(2.*sw**2)',
                 order = {'ANO':1,'QED':2})

GC_27 = Coupling(name = 'GC_27',
                 value = '(ee**3*complex(0,1)*FWWtilde)/sw**2 - (ee**3*imFWWtilde)/sw**2',
                 order = {'ANO':1,'QED':3})

GC_28 = Coupling(name = 'GC_28',
                 value = '(ee**3*FB*complex(0,1))/(4.*cw*sw) - (ee**3*imFB)/(4.*cw*sw)',
                 order = {'ANO':1,'QED':3})

GC_29 = Coupling(name = 'GC_29',
                 value = '-(ee**3*complex(0,1)*FW)/(4.*cw*sw) + (ee**3*imFW)/(4.*cw*sw)',
                 order = {'ANO':1,'QED':3})

GC_30 = Coupling(name = 'GC_30',
                 value = '-(cw*ee**4*complex(0,1)*FW)/(2.*sw**3) - (2*cw*ee**4*complex(0,1)*FWW)/sw**3 + (cw*ee**4*imFW)/(2.*sw**3) + (2*cw*ee**4*imFWW)/sw**3 - (ee**4*complex(0,1)*FW)/(2.*cw*sw) + (ee**4*imFW)/(2.*cw*sw)',
                 order = {'ANO':1,'QED':4})

GC_31 = Coupling(name = 'GC_31',
                 value = '(ee**2*complex(0,1))/(2.*sw**2)',
                 order = {'QED':2})

GC_32 = Coupling(name = 'GC_32',
                 value = '-((ee**2*complex(0,1))/sw**2)',
                 order = {'QED':2})

GC_33 = Coupling(name = 'GC_33',
                 value = '(cw**2*ee**2*complex(0,1))/sw**2',
                 order = {'QED':2})

GC_34 = Coupling(name = 'GC_34',
                 value = '-((ee*complex(0,1))/(sw*cmath.sqrt(2)))',
                 order = {'QED':1})

GC_35 = Coupling(name = 'GC_35',
                 value = '(cw*ee*complex(0,1))/(2.*sw)',
                 order = {'QED':1})

GC_36 = Coupling(name = 'GC_36',
                 value = '-((cw*ee*complex(0,1))/sw)',
                 order = {'QED':1})

GC_37 = Coupling(name = 'GC_37',
                 value = '(-2*cw*ee**2*complex(0,1))/sw',
                 order = {'QED':2})

GC_38 = Coupling(name = 'GC_38',
                 value = '-(ee*complex(0,1)*sw)/(3.*cw)',
                 order = {'QED':1})

GC_39 = Coupling(name = 'GC_39',
                 value = '-(ee*complex(0,1)*sw)/(2.*cw)',
                 order = {'QED':1})

GC_40 = Coupling(name = 'GC_40',
                 value = '(2*ee*complex(0,1)*sw)/(3.*cw)',
                 order = {'QED':1})

GC_41 = Coupling(name = 'GC_41',
                 value = '-(cw*ee*complex(0,1))/(2.*sw) + (ee*complex(0,1)*sw)/(6.*cw)',
                 order = {'QED':1})

GC_42 = Coupling(name = 'GC_42',
                 value = '(cw*ee*complex(0,1))/(2.*sw) + (ee*complex(0,1)*sw)/(6.*cw)',
                 order = {'QED':1})

GC_43 = Coupling(name = 'GC_43',
                 value = '-(cw*ee*complex(0,1))/(2.*sw) - (ee*complex(0,1)*sw)/(2.*cw)',
                 order = {'QED':1})

GC_44 = Coupling(name = 'GC_44',
                 value = '-((cw*ee**2*complex(0,1)*FWW)/sw) + (cw*ee**2*imFWW)/sw + (ee**2*FBB*complex(0,1)*sw)/cw - (ee**2*imFBB*sw)/cw',
                 order = {'ANO':1,'QED':2})

GC_45 = Coupling(name = 'GC_45',
                 value = '(cw*ee**2*complex(0,1)*FWWtilde)/(2.*sw) - (cw*ee**2*imFWWtilde)/(2.*sw) - (ee**2*FBBtilde*complex(0,1)*sw)/(2.*cw) + (ee**2*imFBBtilde*sw)/(2.*cw)',
                 order = {'ANO':1,'QED':2})

GC_46 = Coupling(name = 'GC_46',
                 value = '(cw*ee**2*FB*complex(0,1))/(4.*sw) - (cw*ee**2*complex(0,1)*FW)/(4.*sw) - (cw*ee**2*imFB)/(4.*sw) + (cw*ee**2*imFW)/(4.*sw) + (ee**2*FB*complex(0,1)*sw)/(4.*cw) - (ee**2*complex(0,1)*FW*sw)/(4.*cw) - (ee**2*imFB*sw)/(4.*cw) + (ee**2*imFW*sw)/(4.*cw)',
                 order = {'ANO':1,'QED':2})

GC_47 = Coupling(name = 'GC_47',
                 value = 'ee**2*complex(0,1) + (cw**2*ee**2*complex(0,1))/(2.*sw**2) + (ee**2*complex(0,1)*sw**2)/(2.*cw**2)',
                 order = {'QED':2})

GC_48 = Coupling(name = 'GC_48',
                 value = '-(ee**2*FB*complex(0,1))/4. - (ee**2*complex(0,1)*FW)/4. + (ee**2*imFB)/4. + (ee**2*imFW)/4. - (cw**2*ee**2*complex(0,1)*FW)/(4.*sw**2) + (cw**2*ee**2*imFW)/(4.*sw**2) - (ee**2*FB*complex(0,1)*sw**2)/(4.*cw**2) + (ee**2*imFB*sw**2)/(4.*cw**2)',
                 order = {'ANO':1,'QED':2})

GC_49 = Coupling(name = 'GC_49',
                 value = '-((cw**2*ee**2*complex(0,1)*FWW)/sw**2) + (cw**2*ee**2*imFWW)/sw**2 - (ee**2*FBB*complex(0,1)*sw**2)/cw**2 + (ee**2*imFBB*sw**2)/cw**2',
                 order = {'ANO':1,'QED':2})

GC_50 = Coupling(name = 'GC_50',
                 value = '(cw**2*ee**2*complex(0,1)*FWWtilde)/(2.*sw**2) - (cw**2*ee**2*imFWWtilde)/(2.*sw**2) + (ee**2*FBBtilde*complex(0,1)*sw**2)/(2.*cw**2) - (ee**2*imFBBtilde*sw**2)/(2.*cw**2)',
                 order = {'ANO':1,'QED':2})

GC_51 = Coupling(name = 'GC_51',
                 value = '-6*complex(0,1)*lam*vev',
                 order = {'QED':1})

GC_52 = Coupling(name = 'GC_52',
                 value = '(ee**2*complex(0,1)*vev)/(2.*sw**2)',
                 order = {'QED':1})

GC_53 = Coupling(name = 'GC_53',
                 value = '-(ee**2*FBB*complex(0,1)*vev) - ee**2*complex(0,1)*FWW*vev + ee**2*imFBB*vev + ee**2*imFWW*vev',
                 order = {'ANO':1,'QED':1})

GC_54 = Coupling(name = 'GC_54',
                 value = '(ee**2*FBBtilde*complex(0,1)*vev)/2. + (ee**2*complex(0,1)*FWWtilde*vev)/2. - (ee**2*imFBBtilde*vev)/2. - (ee**2*imFWWtilde*vev)/2.',
                 order = {'ANO':1,'QED':1})

GC_55 = Coupling(name = 'GC_55',
                 value = '-(ee**4*complex(0,1)*FW*vev)/(2.*sw**4) - (ee**4*complex(0,1)*FWW*vev)/sw**4 + (ee**4*imFW*vev)/(2.*sw**4) + (ee**4*imFWW*vev)/sw**4',
                 order = {'ANO':1,'QED':3})

GC_56 = Coupling(name = 'GC_56',
                 value = '-(cw*ee**3*complex(0,1)*FW*vev)/(4.*sw**3) - (cw*ee**3*complex(0,1)*FWW*vev)/sw**3 + (cw*ee**3*imFW*vev)/(4.*sw**3) + (cw*ee**3*imFWW*vev)/sw**3',
                 order = {'ANO':1,'QED':2})

GC_57 = Coupling(name = 'GC_57',
                 value = '(cw*ee**3*complex(0,1)*FWWtilde*vev)/sw**3 - (cw*ee**3*imFWWtilde*vev)/sw**3',
                 order = {'ANO':1,'QED':2})

GC_58 = Coupling(name = 'GC_58',
                 value = '-(ee**3*FB*complex(0,1)*vev)/(4.*sw**2) + (ee**3*imFB*vev)/(4.*sw**2)',
                 order = {'ANO':1,'QED':2})

GC_59 = Coupling(name = 'GC_59',
                 value = '-(ee**2*complex(0,1)*FW*vev)/(4.*sw**2) + (ee**2*imFW*vev)/(4.*sw**2)',
                 order = {'ANO':1,'QED':1})

GC_60 = Coupling(name = 'GC_60',
                 value = '-(ee**3*complex(0,1)*FW*vev)/(4.*sw**2) + (ee**3*imFW*vev)/(4.*sw**2)',
                 order = {'ANO':1,'QED':2})

GC_61 = Coupling(name = 'GC_61',
                 value = '(cw**2*ee**4*complex(0,1)*FW*vev)/(2.*sw**4) + (cw**2*ee**4*complex(0,1)*FWW*vev)/sw**4 - (cw**2*ee**4*imFW*vev)/(2.*sw**4) - (cw**2*ee**4*imFWW*vev)/sw**4 + (ee**4*complex(0,1)*FW*vev)/(2.*sw**2) - (ee**4*imFW*vev)/(2.*sw**2)',
                 order = {'ANO':1,'QED':3})

GC_62 = Coupling(name = 'GC_62',
                 value = '-((ee**2*complex(0,1)*FWW*vev)/sw**2) + (ee**2*imFWW*vev)/sw**2',
                 order = {'ANO':1,'QED':1})

GC_63 = Coupling(name = 'GC_63',
                 value = '-((ee**3*complex(0,1)*FWW*vev)/sw**2) + (ee**3*imFWW*vev)/sw**2',
                 order = {'ANO':1,'QED':2})

GC_64 = Coupling(name = 'GC_64',
                 value = '(ee**4*complex(0,1)*FWW*vev)/sw**2 - (ee**4*imFWW*vev)/sw**2',
                 order = {'ANO':1,'QED':3})

GC_65 = Coupling(name = 'GC_65',
                 value = '(ee**2*complex(0,1)*FWWtilde*vev)/(2.*sw**2) - (ee**2*imFWWtilde*vev)/(2.*sw**2)',
                 order = {'ANO':1,'QED':1})

GC_66 = Coupling(name = 'GC_66',
                 value = '(ee**3*complex(0,1)*FWWtilde*vev)/sw**2 - (ee**3*imFWWtilde*vev)/sw**2',
                 order = {'ANO':1,'QED':2})

GC_67 = Coupling(name = 'GC_67',
                 value = '(ee**3*FB*complex(0,1)*vev)/(4.*cw*sw) - (ee**3*imFB*vev)/(4.*cw*sw)',
                 order = {'ANO':1,'QED':2})

GC_68 = Coupling(name = 'GC_68',
                 value = '-(ee**3*complex(0,1)*FW*vev)/(4.*cw*sw) + (ee**3*imFW*vev)/(4.*cw*sw)',
                 order = {'ANO':1,'QED':2})

GC_69 = Coupling(name = 'GC_69',
                 value = '-(cw*ee**4*complex(0,1)*FW*vev)/(2.*sw**3) - (2*cw*ee**4*complex(0,1)*FWW*vev)/sw**3 + (cw*ee**4*imFW*vev)/(2.*sw**3) + (2*cw*ee**4*imFWW*vev)/sw**3 - (ee**4*complex(0,1)*FW*vev)/(2.*cw*sw) + (ee**4*imFW*vev)/(2.*cw*sw)',
                 order = {'ANO':1,'QED':3})

GC_70 = Coupling(name = 'GC_70',
                 value = '-((cw*ee**2*complex(0,1)*FWW*vev)/sw) + (cw*ee**2*imFWW*vev)/sw + (ee**2*FBB*complex(0,1)*sw*vev)/cw - (ee**2*imFBB*sw*vev)/cw',
                 order = {'ANO':1,'QED':1})

GC_71 = Coupling(name = 'GC_71',
                 value = '(cw*ee**2*complex(0,1)*FWWtilde*vev)/(2.*sw) - (cw*ee**2*imFWWtilde*vev)/(2.*sw) - (ee**2*FBBtilde*complex(0,1)*sw*vev)/(2.*cw) + (ee**2*imFBBtilde*sw*vev)/(2.*cw)',
                 order = {'ANO':1,'QED':1})

GC_72 = Coupling(name = 'GC_72',
                 value = '(cw*ee**2*FB*complex(0,1)*vev)/(4.*sw) - (cw*ee**2*complex(0,1)*FW*vev)/(4.*sw) - (cw*ee**2*imFB*vev)/(4.*sw) + (cw*ee**2*imFW*vev)/(4.*sw) + (ee**2*FB*complex(0,1)*sw*vev)/(4.*cw) - (ee**2*complex(0,1)*FW*sw*vev)/(4.*cw) - (ee**2*imFB*sw*vev)/(4.*cw) + (ee**2*imFW*sw*vev)/(4.*cw)',
                 order = {'ANO':1,'QED':1})

GC_73 = Coupling(name = 'GC_73',
                 value = 'ee**2*complex(0,1)*vev + (cw**2*ee**2*complex(0,1)*vev)/(2.*sw**2) + (ee**2*complex(0,1)*sw**2*vev)/(2.*cw**2)',
                 order = {'QED':1})

GC_74 = Coupling(name = 'GC_74',
                 value = '-(ee**2*FB*complex(0,1)*vev)/4. - (ee**2*complex(0,1)*FW*vev)/4. + (ee**2*imFB*vev)/4. + (ee**2*imFW*vev)/4. - (cw**2*ee**2*complex(0,1)*FW*vev)/(4.*sw**2) + (cw**2*ee**2*imFW*vev)/(4.*sw**2) - (ee**2*FB*complex(0,1)*sw**2*vev)/(4.*cw**2) + (ee**2*imFB*sw**2*vev)/(4.*cw**2)',
                 order = {'ANO':1,'QED':1})

GC_75 = Coupling(name = 'GC_75',
                 value = '-((cw**2*ee**2*complex(0,1)*FWW*vev)/sw**2) + (cw**2*ee**2*imFWW*vev)/sw**2 - (ee**2*FBB*complex(0,1)*sw**2*vev)/cw**2 + (ee**2*imFBB*sw**2*vev)/cw**2',
                 order = {'ANO':1,'QED':1})

GC_76 = Coupling(name = 'GC_76',
                 value = '(cw**2*ee**2*complex(0,1)*FWWtilde*vev)/(2.*sw**2) - (cw**2*ee**2*imFWWtilde*vev)/(2.*sw**2) + (ee**2*FBBtilde*complex(0,1)*sw**2*vev)/(2.*cw**2) - (ee**2*imFBBtilde*sw**2*vev)/(2.*cw**2)',
                 order = {'ANO':1,'QED':1})

GC_77 = Coupling(name = 'GC_77',
                 value = '-(ee**4*complex(0,1)*FW*vev**2)/(4.*sw**4) + (ee**4*imFW*vev**2)/(4.*sw**4)',
                 order = {'ANO':1,'QED':2})

GC_78 = Coupling(name = 'GC_78',
                 value = '-(cw*ee**3*complex(0,1)*FW*vev**2)/(8.*sw**3) + (cw*ee**3*imFW*vev**2)/(8.*sw**3)',
                 order = {'ANO':1,'QED':1})

GC_79 = Coupling(name = 'GC_79',
                 value = '-(ee**3*FB*complex(0,1)*vev**2)/(8.*sw**2) - (ee**3*complex(0,1)*FW*vev**2)/(8.*sw**2) + (ee**3*imFB*vev**2)/(8.*sw**2) + (ee**3*imFW*vev**2)/(8.*sw**2)',
                 order = {'ANO':1,'QED':1})

GC_80 = Coupling(name = 'GC_80',
                 value = '(cw**2*ee**4*complex(0,1)*FW*vev**2)/(4.*sw**4) - (cw**2*ee**4*imFW*vev**2)/(4.*sw**4) + (ee**4*complex(0,1)*FW*vev**2)/(4.*sw**2) - (ee**4*imFW*vev**2)/(4.*sw**2)',
                 order = {'ANO':1,'QED':2})

GC_81 = Coupling(name = 'GC_81',
                 value = '(ee**3*FB*complex(0,1)*vev**2)/(8.*cw*sw) - (ee**3*imFB*vev**2)/(8.*cw*sw)',
                 order = {'ANO':1,'QED':1})

GC_82 = Coupling(name = 'GC_82',
                 value = '-(ee**3*complex(0,1)*FW*vev**2)/(8.*cw*sw) + (ee**3*imFW*vev**2)/(8.*cw*sw)',
                 order = {'ANO':1,'QED':1})

GC_83 = Coupling(name = 'GC_83',
                 value = '-(cw*ee**4*complex(0,1)*FW*vev**2)/(4.*sw**3) + (cw*ee**4*imFW*vev**2)/(4.*sw**3) - (ee**4*complex(0,1)*FW*vev**2)/(4.*cw*sw) + (ee**4*imFW*vev**2)/(4.*cw*sw)',
                 order = {'ANO':1,'QED':2})

GC_84 = Coupling(name = 'GC_84',
                 value = '-((complex(0,1)*yb)/cmath.sqrt(2))',
                 order = {'QED':1})

GC_85 = Coupling(name = 'GC_85',
                 value = '-((complex(0,1)*yt)/cmath.sqrt(2))',
                 order = {'QED':1})

GC_86 = Coupling(name = 'GC_86',
                 value = '-((complex(0,1)*ytau)/cmath.sqrt(2))',
                 order = {'QED':1})

