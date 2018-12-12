# This file was automatically created by FeynRules 2.3.24
# Mathematica version: 11.0.1 for Linux x86 (64-bit) (September 21, 2016)
# Date: Fri 11 Aug 2017 09:48:14


from object_library import all_vertices, Vertex
import particles as P
import couplings as C
import lorentz as L


V_1 = Vertex(name = 'V_1',
             particles = [ P.H, P.H, P.H, P.H ],
             color = [ '1' ],
             lorentz = [ L.SSSS1 ],
             couplings = {(0,0):C.GC_15})

V_2 = Vertex(name = 'V_2',
             particles = [ P.H, P.H, P.H ],
             color = [ '1' ],
             lorentz = [ L.SSS1 ],
             couplings = {(0,0):C.GC_51})

V_3 = Vertex(name = 'V_3',
             particles = [ P.a, P.a, P.H ],
             color = [ '1' ],
             lorentz = [ L.VVS1, L.VVS3 ],
             couplings = {(0,0):C.GC_54,(0,1):C.GC_1})

V_4 = Vertex(name = 'V_4',
             particles = [ P.a, P.a, P.H ],
             color = [ '1' ],
             lorentz = [ L.VVS3 ],
             couplings = {(0,0):C.GC_53})

V_5 = Vertex(name = 'V_5',
             particles = [ P.a, P.a, P.H, P.H ],
             color = [ '1' ],
             lorentz = [ L.VVSS1, L.VVSS3 ],
             couplings = {(0,0):C.GC_14,(0,1):C.GC_13})

V_6 = Vertex(name = 'V_6',
             particles = [ P.g, P.g, P.H ],
             color = [ 'Identity(1,2)' ],
             lorentz = [ L.VVS3 ],
             couplings = {(0,0):C.GC_10})

V_7 = Vertex(name = 'V_7',
             particles = [ P.W__minus__, P.W__plus__, P.H, P.H ],
             color = [ '1' ],
             lorentz = [ L.VVSS1, L.VVSS2, L.VVSS3, L.VVSS5 ],
             couplings = {(0,0):C.GC_26,(0,2):C.GC_23,(0,3):C.GC_20,(0,1):C.GC_31})

V_8 = Vertex(name = 'V_8',
             particles = [ P.W__minus__, P.W__plus__, P.H ],
             color = [ '1' ],
             lorentz = [ L.VVS1, L.VVS2, L.VVS3, L.VVS5 ],
             couplings = {(0,0):C.GC_65,(0,2):C.GC_62,(0,3):C.GC_59,(0,1):C.GC_52})

V_9 = Vertex(name = 'V_9',
             particles = [ P.a, P.Z, P.H, P.H ],
             color = [ '1' ],
             lorentz = [ L.VVSS1, L.VVSS3, L.VVSS4 ],
             couplings = {(0,0):C.GC_45,(0,1):C.GC_44,(0,2):C.GC_46})

V_10 = Vertex(name = 'V_10',
              particles = [ P.a, P.Z, P.H ],
              color = [ '1' ],
              lorentz = [ L.VVS1, L.VVS3, L.VVS4 ],
              couplings = {(0,0):C.GC_71,(0,1):C.GC_70,(0,2):C.GC_72})

V_11 = Vertex(name = 'V_11',
              particles = [ P.Z, P.Z, P.H, P.H ],
              color = [ '1' ],
              lorentz = [ L.VVSS1, L.VVSS2, L.VVSS3, L.VVSS5 ],
              couplings = {(0,0):C.GC_50,(0,2):C.GC_49,(0,3):C.GC_48,(0,1):C.GC_47})

V_12 = Vertex(name = 'V_12',
              particles = [ P.Z, P.Z, P.H ],
              color = [ '1' ],
              lorentz = [ L.VVS1, L.VVS2, L.VVS3, L.VVS5 ],
              couplings = {(0,0):C.GC_76,(0,2):C.GC_75,(0,3):C.GC_74,(0,1):C.GC_73})

V_13 = Vertex(name = 'V_13',
              particles = [ P.ghG, P.ghG__tilde__, P.g ],
              color = [ 'f(1,2,3)' ],
              lorentz = [ L.UUV1 ],
              couplings = {(0,0):C.GC_8})

V_14 = Vertex(name = 'V_14',
              particles = [ P.g, P.g, P.g ],
              color = [ 'f(1,2,3)' ],
              lorentz = [ L.VVV4 ],
              couplings = {(0,0):C.GC_8})

V_15 = Vertex(name = 'V_15',
              particles = [ P.g, P.g, P.g, P.g ],
              color = [ 'f(-1,1,2)*f(3,4,-1)', 'f(-1,1,3)*f(2,4,-1)', 'f(-1,1,4)*f(2,3,-1)' ],
              lorentz = [ L.VVVV1, L.VVVV3, L.VVVV4 ],
              couplings = {(1,1):C.GC_9,(0,0):C.GC_9,(2,2):C.GC_9})

V_16 = Vertex(name = 'V_16',
              particles = [ P.g, P.g, P.g, P.H ],
              color = [ 'f(1,2,3)' ],
              lorentz = [ L.VVVS5 ],
              couplings = {(0,0):C.GC_11})

V_17 = Vertex(name = 'V_17',
              particles = [ P.g, P.g, P.g, P.g, P.H ],
              color = [ 'f(-1,1,2)*f(3,4,-1)', 'f(-1,1,3)*f(2,4,-1)', 'f(-1,1,4)*f(2,3,-1)' ],
              lorentz = [ L.VVVVS1, L.VVVVS3, L.VVVVS4 ],
              couplings = {(1,1):C.GC_12,(0,0):C.GC_12,(2,2):C.GC_12})

V_18 = Vertex(name = 'V_18',
              particles = [ P.b__tilde__, P.b, P.H ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.FFS1 ],
              couplings = {(0,0):C.GC_84})

V_19 = Vertex(name = 'V_19',
              particles = [ P.ta__plus__, P.ta__minus__, P.H ],
              color = [ '1' ],
              lorentz = [ L.FFS1 ],
              couplings = {(0,0):C.GC_86})

V_20 = Vertex(name = 'V_20',
              particles = [ P.t__tilde__, P.t, P.H ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.FFS1 ],
              couplings = {(0,0):C.GC_85})

V_21 = Vertex(name = 'V_21',
              particles = [ P.a, P.W__minus__, P.W__plus__ ],
              color = [ '1' ],
              lorentz = [ L.VVV1, L.VVV4 ],
              couplings = {(0,1):C.GC_4,(0,0):C.GC_79})

V_22 = Vertex(name = 'V_22',
              particles = [ P.a, P.W__minus__, P.W__plus__, P.H, P.H ],
              color = [ '1' ],
              lorentz = [ L.VVVSS1, L.VVVSS2, L.VVVSS3, L.VVVSS5 ],
              couplings = {(0,0):C.GC_27,(0,1):C.GC_19,(0,2):C.GC_21,(0,3):C.GC_24})

V_23 = Vertex(name = 'V_23',
              particles = [ P.a, P.W__minus__, P.W__plus__, P.H ],
              color = [ '1' ],
              lorentz = [ L.VVVS1, L.VVVS2, L.VVVS3, L.VVVS5 ],
              couplings = {(0,0):C.GC_66,(0,1):C.GC_58,(0,2):C.GC_60,(0,3):C.GC_63})

V_24 = Vertex(name = 'V_24',
              particles = [ P.a, P.a, P.W__minus__, P.W__plus__ ],
              color = [ '1' ],
              lorentz = [ L.VVVV2 ],
              couplings = {(0,0):C.GC_6})

V_25 = Vertex(name = 'V_25',
              particles = [ P.a, P.a, P.W__minus__, P.W__plus__, P.H, P.H ],
              color = [ '1' ],
              lorentz = [ L.VVVVSS1 ],
              couplings = {(0,0):C.GC_25})

V_26 = Vertex(name = 'V_26',
              particles = [ P.a, P.a, P.W__minus__, P.W__plus__, P.H ],
              color = [ '1' ],
              lorentz = [ L.VVVVS2 ],
              couplings = {(0,0):C.GC_64})

V_27 = Vertex(name = 'V_27',
              particles = [ P.W__minus__, P.W__plus__, P.Z, P.H, P.H ],
              color = [ '1' ],
              lorentz = [ L.VVVSS1, L.VVVSS4, L.VVVSS5, L.VVVSS6 ],
              couplings = {(0,0):C.GC_18,(0,2):C.GC_17,(0,3):C.GC_29,(0,1):C.GC_28})

V_28 = Vertex(name = 'V_28',
              particles = [ P.W__minus__, P.W__plus__, P.Z ],
              color = [ '1' ],
              lorentz = [ L.VVV2, L.VVV3, L.VVV4 ],
              couplings = {(0,2):C.GC_36,(0,0):C.GC_82,(0,1):C.GC_81})

V_29 = Vertex(name = 'V_29',
              particles = [ P.W__minus__, P.W__plus__, P.Z ],
              color = [ '1' ],
              lorentz = [ L.VVV4 ],
              couplings = {(0,0):C.GC_78})

V_30 = Vertex(name = 'V_30',
              particles = [ P.W__minus__, P.W__plus__, P.Z, P.H ],
              color = [ '1' ],
              lorentz = [ L.VVVS1, L.VVVS4, L.VVVS5, L.VVVS6 ],
              couplings = {(0,0):C.GC_57,(0,2):C.GC_56,(0,3):C.GC_68,(0,1):C.GC_67})

V_31 = Vertex(name = 'V_31',
              particles = [ P.W__minus__, P.W__minus__, P.W__plus__, P.W__plus__, P.H, P.H ],
              color = [ '1' ],
              lorentz = [ L.VVVVSS1 ],
              couplings = {(0,0):C.GC_16})

V_32 = Vertex(name = 'V_32',
              particles = [ P.W__minus__, P.W__minus__, P.W__plus__, P.W__plus__ ],
              color = [ '1' ],
              lorentz = [ L.VVVV2 ],
              couplings = {(0,0):C.GC_32})

V_33 = Vertex(name = 'V_33',
              particles = [ P.W__minus__, P.W__minus__, P.W__plus__, P.W__plus__ ],
              color = [ '1' ],
              lorentz = [ L.VVVV2 ],
              couplings = {(0,0):C.GC_77})

V_34 = Vertex(name = 'V_34',
              particles = [ P.W__minus__, P.W__minus__, P.W__plus__, P.W__plus__, P.H ],
              color = [ '1' ],
              lorentz = [ L.VVVVS2 ],
              couplings = {(0,0):C.GC_55})

V_35 = Vertex(name = 'V_35',
              particles = [ P.a, P.W__minus__, P.W__plus__, P.Z, P.H, P.H ],
              color = [ '1' ],
              lorentz = [ L.VVVVSS2 ],
              couplings = {(0,0):C.GC_30})

V_36 = Vertex(name = 'V_36',
              particles = [ P.a, P.W__minus__, P.W__plus__, P.Z ],
              color = [ '1' ],
              lorentz = [ L.VVVV5 ],
              couplings = {(0,0):C.GC_37})

V_37 = Vertex(name = 'V_37',
              particles = [ P.a, P.W__minus__, P.W__plus__, P.Z ],
              color = [ '1' ],
              lorentz = [ L.VVVV5 ],
              couplings = {(0,0):C.GC_83})

V_38 = Vertex(name = 'V_38',
              particles = [ P.a, P.W__minus__, P.W__plus__, P.Z, P.H ],
              color = [ '1' ],
              lorentz = [ L.VVVVS5 ],
              couplings = {(0,0):C.GC_69})

V_39 = Vertex(name = 'V_39',
              particles = [ P.W__minus__, P.W__plus__, P.Z, P.Z, P.H, P.H ],
              color = [ '1' ],
              lorentz = [ L.VVVVSS1 ],
              couplings = {(0,0):C.GC_22})

V_40 = Vertex(name = 'V_40',
              particles = [ P.W__minus__, P.W__plus__, P.Z, P.Z ],
              color = [ '1' ],
              lorentz = [ L.VVVV2 ],
              couplings = {(0,0):C.GC_33})

V_41 = Vertex(name = 'V_41',
              particles = [ P.W__minus__, P.W__plus__, P.Z, P.Z ],
              color = [ '1' ],
              lorentz = [ L.VVVV2 ],
              couplings = {(0,0):C.GC_80})

V_42 = Vertex(name = 'V_42',
              particles = [ P.W__minus__, P.W__plus__, P.Z, P.Z, P.H ],
              color = [ '1' ],
              lorentz = [ L.VVVVS2 ],
              couplings = {(0,0):C.GC_61})

V_43 = Vertex(name = 'V_43',
              particles = [ P.e__plus__, P.e__minus__, P.a ],
              color = [ '1' ],
              lorentz = [ L.FFV1 ],
              couplings = {(0,0):C.GC_5})

V_44 = Vertex(name = 'V_44',
              particles = [ P.mu__plus__, P.mu__minus__, P.a ],
              color = [ '1' ],
              lorentz = [ L.FFV1 ],
              couplings = {(0,0):C.GC_5})

V_45 = Vertex(name = 'V_45',
              particles = [ P.ta__plus__, P.ta__minus__, P.a ],
              color = [ '1' ],
              lorentz = [ L.FFV1 ],
              couplings = {(0,0):C.GC_5})

V_46 = Vertex(name = 'V_46',
              particles = [ P.u__tilde__, P.u, P.a ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.FFV1 ],
              couplings = {(0,0):C.GC_3})

V_47 = Vertex(name = 'V_47',
              particles = [ P.c__tilde__, P.c, P.a ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.FFV1 ],
              couplings = {(0,0):C.GC_3})

V_48 = Vertex(name = 'V_48',
              particles = [ P.t__tilde__, P.t, P.a ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.FFV1 ],
              couplings = {(0,0):C.GC_3})

V_49 = Vertex(name = 'V_49',
              particles = [ P.d__tilde__, P.d, P.a ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.FFV1 ],
              couplings = {(0,0):C.GC_2})

V_50 = Vertex(name = 'V_50',
              particles = [ P.s__tilde__, P.s, P.a ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.FFV1 ],
              couplings = {(0,0):C.GC_2})

V_51 = Vertex(name = 'V_51',
              particles = [ P.b__tilde__, P.b, P.a ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.FFV1 ],
              couplings = {(0,0):C.GC_2})

V_52 = Vertex(name = 'V_52',
              particles = [ P.u__tilde__, P.u, P.g ],
              color = [ 'T(3,2,1)' ],
              lorentz = [ L.FFV1 ],
              couplings = {(0,0):C.GC_7})

V_53 = Vertex(name = 'V_53',
              particles = [ P.c__tilde__, P.c, P.g ],
              color = [ 'T(3,2,1)' ],
              lorentz = [ L.FFV1 ],
              couplings = {(0,0):C.GC_7})

V_54 = Vertex(name = 'V_54',
              particles = [ P.t__tilde__, P.t, P.g ],
              color = [ 'T(3,2,1)' ],
              lorentz = [ L.FFV1 ],
              couplings = {(0,0):C.GC_7})

V_55 = Vertex(name = 'V_55',
              particles = [ P.d__tilde__, P.d, P.g ],
              color = [ 'T(3,2,1)' ],
              lorentz = [ L.FFV1 ],
              couplings = {(0,0):C.GC_7})

V_56 = Vertex(name = 'V_56',
              particles = [ P.s__tilde__, P.s, P.g ],
              color = [ 'T(3,2,1)' ],
              lorentz = [ L.FFV1 ],
              couplings = {(0,0):C.GC_7})

V_57 = Vertex(name = 'V_57',
              particles = [ P.b__tilde__, P.b, P.g ],
              color = [ 'T(3,2,1)' ],
              lorentz = [ L.FFV1 ],
              couplings = {(0,0):C.GC_7})

V_58 = Vertex(name = 'V_58',
              particles = [ P.d__tilde__, P.u, P.W__minus__ ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.FFV2 ],
              couplings = {(0,0):C.GC_34})

V_59 = Vertex(name = 'V_59',
              particles = [ P.s__tilde__, P.c, P.W__minus__ ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.FFV2 ],
              couplings = {(0,0):C.GC_34})

V_60 = Vertex(name = 'V_60',
              particles = [ P.b__tilde__, P.t, P.W__minus__ ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.FFV2 ],
              couplings = {(0,0):C.GC_34})

V_61 = Vertex(name = 'V_61',
              particles = [ P.u__tilde__, P.d, P.W__plus__ ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.FFV2 ],
              couplings = {(0,0):C.GC_34})

V_62 = Vertex(name = 'V_62',
              particles = [ P.c__tilde__, P.s, P.W__plus__ ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.FFV2 ],
              couplings = {(0,0):C.GC_34})

V_63 = Vertex(name = 'V_63',
              particles = [ P.t__tilde__, P.b, P.W__plus__ ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.FFV2 ],
              couplings = {(0,0):C.GC_34})

V_64 = Vertex(name = 'V_64',
              particles = [ P.e__plus__, P.ve, P.W__minus__ ],
              color = [ '1' ],
              lorentz = [ L.FFV2 ],
              couplings = {(0,0):C.GC_34})

V_65 = Vertex(name = 'V_65',
              particles = [ P.mu__plus__, P.vm, P.W__minus__ ],
              color = [ '1' ],
              lorentz = [ L.FFV2 ],
              couplings = {(0,0):C.GC_34})

V_66 = Vertex(name = 'V_66',
              particles = [ P.ta__plus__, P.vt, P.W__minus__ ],
              color = [ '1' ],
              lorentz = [ L.FFV2 ],
              couplings = {(0,0):C.GC_34})

V_67 = Vertex(name = 'V_67',
              particles = [ P.ve__tilde__, P.e__minus__, P.W__plus__ ],
              color = [ '1' ],
              lorentz = [ L.FFV2 ],
              couplings = {(0,0):C.GC_34})

V_68 = Vertex(name = 'V_68',
              particles = [ P.vm__tilde__, P.mu__minus__, P.W__plus__ ],
              color = [ '1' ],
              lorentz = [ L.FFV2 ],
              couplings = {(0,0):C.GC_34})

V_69 = Vertex(name = 'V_69',
              particles = [ P.vt__tilde__, P.ta__minus__, P.W__plus__ ],
              color = [ '1' ],
              lorentz = [ L.FFV2 ],
              couplings = {(0,0):C.GC_34})

V_70 = Vertex(name = 'V_70',
              particles = [ P.u__tilde__, P.u, P.Z ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.FFV2, L.FFV3 ],
              couplings = {(0,0):C.GC_41,(0,1):C.GC_40})

V_71 = Vertex(name = 'V_71',
              particles = [ P.c__tilde__, P.c, P.Z ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.FFV2, L.FFV3 ],
              couplings = {(0,0):C.GC_41,(0,1):C.GC_40})

V_72 = Vertex(name = 'V_72',
              particles = [ P.t__tilde__, P.t, P.Z ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.FFV2, L.FFV3 ],
              couplings = {(0,0):C.GC_41,(0,1):C.GC_40})

V_73 = Vertex(name = 'V_73',
              particles = [ P.d__tilde__, P.d, P.Z ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.FFV2, L.FFV3 ],
              couplings = {(0,0):C.GC_42,(0,1):C.GC_38})

V_74 = Vertex(name = 'V_74',
              particles = [ P.s__tilde__, P.s, P.Z ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.FFV2, L.FFV3 ],
              couplings = {(0,0):C.GC_42,(0,1):C.GC_38})

V_75 = Vertex(name = 'V_75',
              particles = [ P.b__tilde__, P.b, P.Z ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.FFV2, L.FFV3 ],
              couplings = {(0,0):C.GC_42,(0,1):C.GC_38})

V_76 = Vertex(name = 'V_76',
              particles = [ P.ve__tilde__, P.ve, P.Z ],
              color = [ '1' ],
              lorentz = [ L.FFV2 ],
              couplings = {(0,0):C.GC_43})

V_77 = Vertex(name = 'V_77',
              particles = [ P.vm__tilde__, P.vm, P.Z ],
              color = [ '1' ],
              lorentz = [ L.FFV2 ],
              couplings = {(0,0):C.GC_43})

V_78 = Vertex(name = 'V_78',
              particles = [ P.vt__tilde__, P.vt, P.Z ],
              color = [ '1' ],
              lorentz = [ L.FFV2 ],
              couplings = {(0,0):C.GC_43})

V_79 = Vertex(name = 'V_79',
              particles = [ P.e__plus__, P.e__minus__, P.Z ],
              color = [ '1' ],
              lorentz = [ L.FFV2, L.FFV4 ],
              couplings = {(0,1):C.GC_39,(0,0):C.GC_35})

V_80 = Vertex(name = 'V_80',
              particles = [ P.mu__plus__, P.mu__minus__, P.Z ],
              color = [ '1' ],
              lorentz = [ L.FFV2, L.FFV4 ],
              couplings = {(0,0):C.GC_35,(0,1):C.GC_39})

V_81 = Vertex(name = 'V_81',
              particles = [ P.ta__plus__, P.ta__minus__, P.Z ],
              color = [ '1' ],
              lorentz = [ L.FFV2, L.FFV4 ],
              couplings = {(0,0):C.GC_35,(0,1):C.GC_39})

