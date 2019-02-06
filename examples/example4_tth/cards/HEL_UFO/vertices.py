# This file was automatically created by FeynRules 2.4.32
# Mathematica version: 10.0 for Linux x86 (64-bit) (September 9, 2014)
# Date: Thu 7 Jan 2016 02:14:12


from object_library import all_vertices, Vertex
import particles as P
import couplings as C
import lorentz as L


V_1 = Vertex(name = 'V_1',
             particles = [ P.G0, P.G0, P.G0, P.G0, P.G0, P.G0 ],
             color = [ '1' ],
             lorentz = [ L.SSSSSS1 ],
             couplings = {(0,0):C.GC_1153})

V_2 = Vertex(name = 'V_2',
             particles = [ P.G0, P.G0, P.G0, P.G0, P.G__minus__, P.G__plus__ ],
             color = [ '1' ],
             lorentz = [ L.SSSSSS1 ],
             couplings = {(0,0):C.GC_1151})

V_3 = Vertex(name = 'V_3',
             particles = [ P.G0, P.G0, P.G__minus__, P.G__minus__, P.G__plus__, P.G__plus__ ],
             color = [ '1' ],
             lorentz = [ L.SSSSSS1 ],
             couplings = {(0,0):C.GC_1150})

V_4 = Vertex(name = 'V_4',
             particles = [ P.G__minus__, P.G__minus__, P.G__minus__, P.G__plus__, P.G__plus__, P.G__plus__ ],
             color = [ '1' ],
             lorentz = [ L.SSSSSS1 ],
             couplings = {(0,0):C.GC_1152})

V_5 = Vertex(name = 'V_5',
             particles = [ P.G0, P.G0, P.G0, P.G0, P.H, P.H ],
             color = [ '1' ],
             lorentz = [ L.SSSSSS1 ],
             couplings = {(0,0):C.GC_1151})

V_6 = Vertex(name = 'V_6',
             particles = [ P.G0, P.G0, P.G__minus__, P.G__plus__, P.H, P.H ],
             color = [ '1' ],
             lorentz = [ L.SSSSSS1 ],
             couplings = {(0,0):C.GC_1149})

V_7 = Vertex(name = 'V_7',
             particles = [ P.G__minus__, P.G__minus__, P.G__plus__, P.G__plus__, P.H, P.H ],
             color = [ '1' ],
             lorentz = [ L.SSSSSS1 ],
             couplings = {(0,0):C.GC_1150})

V_8 = Vertex(name = 'V_8',
             particles = [ P.G0, P.G0, P.H, P.H, P.H, P.H ],
             color = [ '1' ],
             lorentz = [ L.SSSSSS1 ],
             couplings = {(0,0):C.GC_1151})

V_9 = Vertex(name = 'V_9',
             particles = [ P.G__minus__, P.G__plus__, P.H, P.H, P.H, P.H ],
             color = [ '1' ],
             lorentz = [ L.SSSSSS1 ],
             couplings = {(0,0):C.GC_1151})

V_10 = Vertex(name = 'V_10',
              particles = [ P.H, P.H, P.H, P.H, P.H, P.H ],
              color = [ '1' ],
              lorentz = [ L.SSSSSS1 ],
              couplings = {(0,0):C.GC_1153})

V_11 = Vertex(name = 'V_11',
              particles = [ P.H, P.H, P.H, P.H, P.H, P.H ],
              color = [ '1' ],
              lorentz = [ L.SSSSSS1 ],
              couplings = {(0,0):C.GC_1064})

V_12 = Vertex(name = 'V_12',
              particles = [ P.H, P.H, P.H, P.H, P.H, P.H ],
              color = [ '1' ],
              lorentz = [ L.SSSSSS1 ],
              couplings = {(0,0):C.GC_1065})

V_13 = Vertex(name = 'V_13',
              particles = [ P.G0, P.G0, P.G0, P.G0, P.H ],
              color = [ '1' ],
              lorentz = [ L.SSSSS1 ],
              couplings = {(0,0):C.GC_1156})

V_14 = Vertex(name = 'V_14',
              particles = [ P.G0, P.G0, P.G__minus__, P.G__plus__, P.H ],
              color = [ '1' ],
              lorentz = [ L.SSSSS1 ],
              couplings = {(0,0):C.GC_1154})

V_15 = Vertex(name = 'V_15',
              particles = [ P.G__minus__, P.G__minus__, P.G__plus__, P.G__plus__, P.H ],
              color = [ '1' ],
              lorentz = [ L.SSSSS1 ],
              couplings = {(0,0):C.GC_1155})

V_16 = Vertex(name = 'V_16',
              particles = [ P.G0, P.G0, P.H, P.H, P.H ],
              color = [ '1' ],
              lorentz = [ L.SSSSS1 ],
              couplings = {(0,0):C.GC_1156})

V_17 = Vertex(name = 'V_17',
              particles = [ P.G__minus__, P.G__plus__, P.H, P.H, P.H ],
              color = [ '1' ],
              lorentz = [ L.SSSSS1 ],
              couplings = {(0,0):C.GC_1156})

V_18 = Vertex(name = 'V_18',
              particles = [ P.H, P.H, P.H, P.H, P.H ],
              color = [ '1' ],
              lorentz = [ L.SSSSS1 ],
              couplings = {(0,0):C.GC_1157})

V_19 = Vertex(name = 'V_19',
              particles = [ P.H, P.H, P.H, P.H, P.H ],
              color = [ '1' ],
              lorentz = [ L.SSSSS1 ],
              couplings = {(0,0):C.GC_1066})

V_20 = Vertex(name = 'V_20',
              particles = [ P.G0, P.G0, P.G0, P.G0 ],
              color = [ '1' ],
              lorentz = [ L.SSSS1, L.SSSS6 ],
              couplings = {(0,0):C.GC_1302,(0,1):C.GC_1159})

V_21 = Vertex(name = 'V_21',
              particles = [ P.G0, P.G0, P.G0, P.G0 ],
              color = [ '1' ],
              lorentz = [ L.SSSS1 ],
              couplings = {(0,0):C.GC_1085})

V_22 = Vertex(name = 'V_22',
              particles = [ P.G0, P.G0, P.G__minus__, P.G__plus__ ],
              color = [ '1' ],
              lorentz = [ L.SSSS1, L.SSSS2 ],
              couplings = {(0,0):C.GC_1300,(0,1):C.GC_1158})

V_23 = Vertex(name = 'V_23',
              particles = [ P.G0, P.G0, P.G__minus__, P.G__plus__ ],
              color = [ '1' ],
              lorentz = [ L.SSSS1 ],
              couplings = {(0,0):C.GC_1083})

V_24 = Vertex(name = 'V_24',
              particles = [ P.G__minus__, P.G__minus__, P.G__plus__, P.G__plus__ ],
              color = [ '1' ],
              lorentz = [ L.SSSS1, L.SSSS4, L.SSSS5 ],
              couplings = {(0,0):C.GC_1301,(0,2):C.GC_1159,(0,1):C.GC_1195})

V_25 = Vertex(name = 'V_25',
              particles = [ P.G__minus__, P.G__minus__, P.G__plus__, P.G__plus__ ],
              color = [ '1' ],
              lorentz = [ L.SSSS1 ],
              couplings = {(0,0):C.GC_1084})

V_26 = Vertex(name = 'V_26',
              particles = [ P.G0, P.G0, P.H, P.H ],
              color = [ '1' ],
              lorentz = [ L.SSSS1, L.SSSS2, L.SSSS4 ],
              couplings = {(0,0):C.GC_1300,(0,2):C.GC_1196,(0,1):C.GC_1158})

V_27 = Vertex(name = 'V_27',
              particles = [ P.G0, P.G0, P.H, P.H ],
              color = [ '1' ],
              lorentz = [ L.SSSS1 ],
              couplings = {(0,0):C.GC_1303})

V_28 = Vertex(name = 'V_28',
              particles = [ P.G__minus__, P.G__plus__, P.H, P.H ],
              color = [ '1' ],
              lorentz = [ L.SSSS1, L.SSSS2 ],
              couplings = {(0,0):C.GC_1300,(0,1):C.GC_1158})

V_29 = Vertex(name = 'V_29',
              particles = [ P.G__minus__, P.G__plus__, P.H, P.H ],
              color = [ '1' ],
              lorentz = [ L.SSSS1 ],
              couplings = {(0,0):C.GC_1303})

V_30 = Vertex(name = 'V_30',
              particles = [ P.H, P.H, P.H, P.H ],
              color = [ '1' ],
              lorentz = [ L.SSSS1, L.SSSS6 ],
              couplings = {(0,0):C.GC_1302,(0,1):C.GC_1159})

V_31 = Vertex(name = 'V_31',
              particles = [ P.H, P.H, P.H, P.H ],
              color = [ '1' ],
              lorentz = [ L.SSSS1 ],
              couplings = {(0,0):C.GC_1086})

V_32 = Vertex(name = 'V_32',
              particles = [ P.G0, P.G0, P.H ],
              color = [ '1' ],
              lorentz = [ L.SSS1, L.SSS3, L.SSS4 ],
              couplings = {(0,0):C.GC_1439,(0,1):C.GC_1387,(0,2):C.GC_1371})

V_33 = Vertex(name = 'V_33',
              particles = [ P.G0, P.G0, P.H ],
              color = [ '1' ],
              lorentz = [ L.SSS1 ],
              couplings = {(0,0):C.GC_1125})

V_34 = Vertex(name = 'V_34',
              particles = [ P.G__minus__, P.G__plus__, P.H ],
              color = [ '1' ],
              lorentz = [ L.SSS1, L.SSS4 ],
              couplings = {(0,0):C.GC_1439,(0,1):C.GC_1371})

V_35 = Vertex(name = 'V_35',
              particles = [ P.G__minus__, P.G__plus__, P.H ],
              color = [ '1' ],
              lorentz = [ L.SSS1 ],
              couplings = {(0,0):C.GC_1125})

V_36 = Vertex(name = 'V_36',
              particles = [ P.H, P.H, P.H ],
              color = [ '1' ],
              lorentz = [ L.SSS1, L.SSS5 ],
              couplings = {(0,0):C.GC_1440,(0,1):C.GC_1372})

V_37 = Vertex(name = 'V_37',
              particles = [ P.H, P.H, P.H ],
              color = [ '1' ],
              lorentz = [ L.SSS1 ],
              couplings = {(0,0):C.GC_1126})

V_38 = Vertex(name = 'V_38',
              particles = [ P.a, P.a, P.G__minus__, P.G__plus__ ],
              color = [ '1' ],
              lorentz = [ L.VVSS11, L.VVSS13, L.VVSS16, L.VVSS2, L.VVSS23, L.VVSS5 ],
              couplings = {(0,2):C.GC_94,(0,3):C.GC_1004,(0,1):C.GC_133,(0,5):C.GC_1052,(0,4):C.GC_93,(0,0):C.GC_7})

V_39 = Vertex(name = 'V_39',
              particles = [ P.a, P.a, P.G__minus__, P.G__plus__ ],
              color = [ '1' ],
              lorentz = [ L.VVSS11 ],
              couplings = {(0,0):C.GC_1710})

V_40 = Vertex(name = 'V_40',
              particles = [ P.a, P.a, P.G__minus__, P.G__minus__, P.G__plus__, P.G__plus__ ],
              color = [ '1' ],
              lorentz = [ L.VVSSSS1 ],
              couplings = {(0,0):C.GC_1227})

V_41 = Vertex(name = 'V_41',
              particles = [ P.a, P.G0, P.G__minus__, P.G__plus__, P.H ],
              color = [ '1' ],
              lorentz = [ L.VSSSS7 ],
              couplings = {(0,0):C.GC_1225})

V_42 = Vertex(name = 'V_42',
              particles = [ P.a, P.G0, P.G__minus__, P.G__plus__ ],
              color = [ '1' ],
              lorentz = [ L.VSSS1 ],
              couplings = {(0,0):C.GC_1402})

V_43 = Vertex(name = 'V_43',
              particles = [ P.a, P.G__minus__, P.G__plus__ ],
              color = [ '1' ],
              lorentz = [ L.VSS2, L.VSS4, L.VSS5, L.VSS6 ],
              couplings = {(0,0):C.GC_4,(0,3):C.GC_1051,(0,1):C.GC_92,(0,2):C.GC_90})

V_44 = Vertex(name = 'V_44',
              particles = [ P.a, P.G__minus__, P.G__plus__ ],
              color = [ '1' ],
              lorentz = [ L.VSS2 ],
              couplings = {(0,0):C.GC_1707})

V_45 = Vertex(name = 'V_45',
              particles = [ P.a, P.G__minus__, P.G__minus__, P.G__plus__, P.G__plus__ ],
              color = [ '1' ],
              lorentz = [ L.VSSSS8 ],
              couplings = {(0,0):C.GC_1226})

V_46 = Vertex(name = 'V_46',
              particles = [ P.G0, P.G__minus__, P.G__plus__, P.H ],
              color = [ '1' ],
              lorentz = [ L.SSSS3 ],
              couplings = {(0,0):C.GC_1197})

V_47 = Vertex(name = 'V_47',
              particles = [ P.G0, P.G__minus__, P.G__plus__ ],
              color = [ '1' ],
              lorentz = [ L.SSS2 ],
              couplings = {(0,0):C.GC_1388})

V_48 = Vertex(name = 'V_48',
              particles = [ P.a, P.a, P.H ],
              color = [ '1' ],
              lorentz = [ L.VVS2, L.VVS9 ],
              couplings = {(0,0):C.GC_1683,(0,1):C.GC_1})

V_49 = Vertex(name = 'V_49',
              particles = [ P.a, P.a, P.H ],
              color = [ '1' ],
              lorentz = [ L.VVS9 ],
              couplings = {(0,0):C.GC_1489})

V_50 = Vertex(name = 'V_50',
              particles = [ P.a, P.G0, P.H ],
              color = [ '1' ],
              lorentz = [ L.VSS4, L.VSS5, L.VSS6 ],
              couplings = {(0,2):C.GC_1050,(0,0):C.GC_91,(0,1):C.GC_89})

V_51 = Vertex(name = 'V_51',
              particles = [ P.a, P.a, P.G0, P.G0 ],
              color = [ '1' ],
              lorentz = [ L.VVSS13, L.VVSS2 ],
              couplings = {(0,1):C.GC_1004,(0,0):C.GC_133})

V_52 = Vertex(name = 'V_52',
              particles = [ P.a, P.a, P.H, P.H ],
              color = [ '1' ],
              lorentz = [ L.VVSS13, L.VVSS2 ],
              couplings = {(0,1):C.GC_1004,(0,0):C.GC_133})

V_53 = Vertex(name = 'V_53',
              particles = [ P.a, P.W__minus__, P.G0, P.G__plus__ ],
              color = [ '1' ],
              lorentz = [ L.VVSS11, L.VVSS12, L.VVSS17, L.VVSS19, L.VVSS25, L.VVSS3, L.VVSS8 ],
              couplings = {(0,1):C.GC_583,(0,4):C.GC_593,(0,5):C.GC_1021,(0,6):C.GC_1046,(0,2):C.GC_586,(0,3):C.GC_589,(0,0):C.GC_572})

V_54 = Vertex(name = 'V_54',
              particles = [ P.a, P.W__minus__, P.G0, P.G__plus__ ],
              color = [ '1' ],
              lorentz = [ L.VVSS11 ],
              couplings = {(0,0):C.GC_1785})

V_55 = Vertex(name = 'V_55',
              particles = [ P.a, P.W__minus__, P.G__plus__, P.H ],
              color = [ '1' ],
              lorentz = [ L.VVSS11, L.VVSS12, L.VVSS17, L.VVSS21, L.VVSS28, L.VVSS3, L.VVSS9 ],
              couplings = {(0,1):C.GC_584,(0,4):C.GC_594,(0,5):C.GC_1022,(0,6):C.GC_1047,(0,2):C.GC_585,(0,3):C.GC_588,(0,0):C.GC_571})

V_56 = Vertex(name = 'V_56',
              particles = [ P.a, P.W__minus__, P.G__plus__, P.H ],
              color = [ '1' ],
              lorentz = [ L.VVSS11 ],
              couplings = {(0,0):C.GC_1784})

V_57 = Vertex(name = 'V_57',
              particles = [ P.a, P.W__minus__, P.G__plus__ ],
              color = [ '1' ],
              lorentz = [ L.VVS12, L.VVS15, L.VVS3, L.VVS7, L.VVS8 ],
              couplings = {(0,4):C.GC_1563,(0,1):C.GC_1565,(0,2):C.GC_1762,(0,0):C.GC_1747,(0,3):C.GC_1560})

V_58 = Vertex(name = 'V_58',
              particles = [ P.a, P.W__minus__, P.G__plus__ ],
              color = [ '1' ],
              lorentz = [ L.VVS7 ],
              couplings = {(0,0):C.GC_1808})

V_59 = Vertex(name = 'V_59',
              particles = [ P.W__minus__, P.G0, P.G__plus__ ],
              color = [ '1' ],
              lorentz = [ L.VSS1, L.VSS2, L.VSS4, L.VSS5, L.VSS6 ],
              couplings = {(0,1):C.GC_555,(0,0):C.GC_567,(0,4):C.GC_1044,(0,2):C.GC_581,(0,3):C.GC_577})

V_60 = Vertex(name = 'V_60',
              particles = [ P.W__minus__, P.G0, P.G__plus__ ],
              color = [ '1' ],
              lorentz = [ L.VSS2 ],
              couplings = {(0,0):C.GC_1716})

V_61 = Vertex(name = 'V_61',
              particles = [ P.W__minus__, P.G__plus__, P.H ],
              color = [ '1' ],
              lorentz = [ L.VSS2, L.VSS4, L.VSS5, L.VSS6 ],
              couplings = {(0,0):C.GC_553,(0,3):C.GC_1042,(0,1):C.GC_579,(0,2):C.GC_575})

V_62 = Vertex(name = 'V_62',
              particles = [ P.W__minus__, P.G__plus__, P.H ],
              color = [ '1' ],
              lorentz = [ L.VSS2 ],
              couplings = {(0,0):C.GC_1768})

V_63 = Vertex(name = 'V_63',
              particles = [ P.a, P.W__plus__, P.G0, P.G__minus__ ],
              color = [ '1' ],
              lorentz = [ L.VVSS11, L.VVSS12, L.VVSS17, L.VVSS19, L.VVSS25, L.VVSS3, L.VVSS8 ],
              couplings = {(0,1):C.GC_583,(0,4):C.GC_593,(0,5):C.GC_1021,(0,6):C.GC_1046,(0,2):C.GC_586,(0,3):C.GC_589,(0,0):C.GC_572})

V_64 = Vertex(name = 'V_64',
              particles = [ P.a, P.W__plus__, P.G0, P.G__minus__ ],
              color = [ '1' ],
              lorentz = [ L.VVSS11 ],
              couplings = {(0,0):C.GC_1785})

V_65 = Vertex(name = 'V_65',
              particles = [ P.a, P.W__plus__, P.G__minus__, P.H ],
              color = [ '1' ],
              lorentz = [ L.VVSS11, L.VVSS12, L.VVSS17, L.VVSS21, L.VVSS28, L.VVSS3, L.VVSS9 ],
              couplings = {(0,1):C.GC_582,(0,4):C.GC_592,(0,5):C.GC_1020,(0,6):C.GC_1045,(0,2):C.GC_587,(0,3):C.GC_590,(0,0):C.GC_573})

V_66 = Vertex(name = 'V_66',
              particles = [ P.a, P.W__plus__, P.G__minus__, P.H ],
              color = [ '1' ],
              lorentz = [ L.VVSS11 ],
              couplings = {(0,0):C.GC_1786})

V_67 = Vertex(name = 'V_67',
              particles = [ P.a, P.W__plus__, P.G__minus__ ],
              color = [ '1' ],
              lorentz = [ L.VVS12, L.VVS15, L.VVS3, L.VVS7, L.VVS8 ],
              couplings = {(0,4):C.GC_1562,(0,1):C.GC_1564,(0,2):C.GC_1761,(0,0):C.GC_1748,(0,3):C.GC_1561})

V_68 = Vertex(name = 'V_68',
              particles = [ P.a, P.W__plus__, P.G__minus__ ],
              color = [ '1' ],
              lorentz = [ L.VVS7 ],
              couplings = {(0,0):C.GC_1809})

V_69 = Vertex(name = 'V_69',
              particles = [ P.W__plus__, P.G0, P.G__minus__ ],
              color = [ '1' ],
              lorentz = [ L.VSS1, L.VSS2, L.VSS4, L.VSS5, L.VSS6 ],
              couplings = {(0,1):C.GC_554,(0,0):C.GC_568,(0,4):C.GC_1043,(0,2):C.GC_580,(0,3):C.GC_576})

V_70 = Vertex(name = 'V_70',
              particles = [ P.W__plus__, P.G0, P.G__minus__ ],
              color = [ '1' ],
              lorentz = [ L.VSS2 ],
              couplings = {(0,0):C.GC_1715})

V_71 = Vertex(name = 'V_71',
              particles = [ P.W__plus__, P.G__minus__, P.H ],
              color = [ '1' ],
              lorentz = [ L.VSS2, L.VSS4, L.VSS5, L.VSS6 ],
              couplings = {(0,0):C.GC_553,(0,3):C.GC_1042,(0,1):C.GC_579,(0,2):C.GC_575})

V_72 = Vertex(name = 'V_72',
              particles = [ P.W__plus__, P.G__minus__, P.H ],
              color = [ '1' ],
              lorentz = [ L.VSS2 ],
              couplings = {(0,0):C.GC_1768})

V_73 = Vertex(name = 'V_73',
              particles = [ P.a, P.Z, P.G__minus__, P.G__plus__ ],
              color = [ '1' ],
              lorentz = [ L.VVSS1, L.VVSS11, L.VVSS12, L.VVSS13, L.VVSS14, L.VVSS17, L.VVSS20, L.VVSS3, L.VVSS4 ],
              couplings = {(0,2):C.GC_906,(0,4):C.GC_901,(0,0):C.GC_1005,(0,3):C.GC_888,(0,7):C.GC_1063,(0,8):C.GC_1057,(0,5):C.GC_903,(0,6):C.GC_902,(0,1):C.GC_896})

V_74 = Vertex(name = 'V_74',
              particles = [ P.a, P.Z, P.G__minus__, P.G__plus__ ],
              color = [ '1' ],
              lorentz = [ L.VVSS11 ],
              couplings = {(0,0):C.GC_1799})

V_75 = Vertex(name = 'V_75',
              particles = [ P.Z, P.G0, P.H ],
              color = [ '1' ],
              lorentz = [ L.VSS2, L.VSS3, L.VSS4, L.VSS5, L.VSS6 ],
              couplings = {(0,0):C.GC_890,(0,1):C.GC_895,(0,4):C.GC_1055,(0,2):C.GC_897,(0,3):C.GC_899})

V_76 = Vertex(name = 'V_76',
              particles = [ P.Z, P.G0, P.H ],
              color = [ '1' ],
              lorentz = [ L.VSS2 ],
              couplings = {(0,0):C.GC_1798})

V_77 = Vertex(name = 'V_77',
              particles = [ P.Z, P.G__minus__, P.G__plus__ ],
              color = [ '1' ],
              lorentz = [ L.VSS2, L.VSS4, L.VSS5, L.VSS6 ],
              couplings = {(0,0):C.GC_893,(0,3):C.GC_1056,(0,1):C.GC_898,(0,2):C.GC_900})

V_78 = Vertex(name = 'V_78',
              particles = [ P.Z, P.G__minus__, P.G__plus__ ],
              color = [ '1' ],
              lorentz = [ L.VSS2 ],
              couplings = {(0,0):C.GC_1794})

V_79 = Vertex(name = 'V_79',
              particles = [ P.g, P.g, P.H ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.VVS2, L.VVS9 ],
              couplings = {(0,0):C.GC_1686,(0,1):C.GC_14})

V_80 = Vertex(name = 'V_80',
              particles = [ P.g, P.g, P.H ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.VVS9 ],
              couplings = {(0,0):C.GC_1501})

V_81 = Vertex(name = 'V_81',
              particles = [ P.g, P.g, P.G0, P.G0 ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.VVSS13, L.VVSS2 ],
              couplings = {(0,1):C.GC_1007,(0,0):C.GC_160})

V_82 = Vertex(name = 'V_82',
              particles = [ P.g, P.g, P.G__minus__, P.G__plus__ ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.VVSS13, L.VVSS2 ],
              couplings = {(0,1):C.GC_1007,(0,0):C.GC_160})

V_83 = Vertex(name = 'V_83',
              particles = [ P.g, P.g, P.H, P.H ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.VVSS13, L.VVSS2 ],
              couplings = {(0,1):C.GC_1007,(0,0):C.GC_160})

V_84 = Vertex(name = 'V_84',
              particles = [ P.a, P.a, P.W__minus__, P.G0, P.G__plus__ ],
              color = [ '1' ],
              lorentz = [ L.VVVSS13, L.VVVSS2, L.VVVSS20, L.VVVSS25 ],
              couplings = {(0,1):C.GC_1060,(0,0):C.GC_596,(0,2):C.GC_599,(0,3):C.GC_610})

V_85 = Vertex(name = 'V_85',
              particles = [ P.a, P.a, P.W__minus__, P.G__plus__, P.H ],
              color = [ '1' ],
              lorentz = [ L.VVVSS13, L.VVVSS2, L.VVVSS29, L.VVVSS33 ],
              couplings = {(0,1):C.GC_1061,(0,0):C.GC_595,(0,3):C.GC_598,(0,2):C.GC_606})

V_86 = Vertex(name = 'V_86',
              particles = [ P.a, P.a, P.W__minus__, P.G__plus__ ],
              color = [ '1' ],
              lorentz = [ L.VVVS12, L.VVVS2, L.VVVS22 ],
              couplings = {(0,1):C.GC_1764,(0,0):C.GC_1749,(0,2):C.GC_1570})

V_87 = Vertex(name = 'V_87',
              particles = [ P.a, P.W__minus__, P.W__plus__ ],
              color = [ '1' ],
              lorentz = [ L.VVV1, L.VVV10, L.VVV11, L.VVV12, L.VVV13, L.VVV5, L.VVV6 ],
              couplings = {(0,0):C.GC_1804,(0,3):C.GC_131,(0,4):C.GC_1001,(0,2):C.GC_387,(0,1):C.GC_5,(0,5):C.GC_1779,(0,6):C.GC_1719})

V_88 = Vertex(name = 'V_88',
              particles = [ P.a, P.W__minus__, P.W__plus__ ],
              color = [ '1' ],
              lorentz = [ L.VVV10 ],
              couplings = {(0,0):C.GC_1708})

V_89 = Vertex(name = 'V_89',
              particles = [ P.a, P.a, P.W__plus__, P.G0, P.G__minus__ ],
              color = [ '1' ],
              lorentz = [ L.VVVSS13, L.VVVSS2, L.VVVSS20, L.VVVSS25 ],
              couplings = {(0,1):C.GC_1059,(0,0):C.GC_597,(0,2):C.GC_600,(0,3):C.GC_611})

V_90 = Vertex(name = 'V_90',
              particles = [ P.a, P.a, P.W__plus__, P.G__minus__, P.H ],
              color = [ '1' ],
              lorentz = [ L.VVVSS13, L.VVVSS2, L.VVVSS29, L.VVVSS33 ],
              couplings = {(0,1):C.GC_1061,(0,0):C.GC_595,(0,3):C.GC_598,(0,2):C.GC_606})

V_91 = Vertex(name = 'V_91',
              particles = [ P.a, P.a, P.W__plus__, P.G__minus__ ],
              color = [ '1' ],
              lorentz = [ L.VVVS12, L.VVVS2, L.VVVS22 ],
              couplings = {(0,1):C.GC_1764,(0,0):C.GC_1749,(0,2):C.GC_1570})

V_92 = Vertex(name = 'V_92',
              particles = [ P.a, P.a, P.W__minus__, P.W__plus__ ],
              color = [ '1' ],
              lorentz = [ L.VVVV15, L.VVVV17, L.VVVV4, L.VVVV8 ],
              couplings = {(0,2):C.GC_1002,(0,1):C.GC_132,(0,0):C.GC_403,(0,3):C.GC_6})

V_93 = Vertex(name = 'V_93',
              particles = [ P.a, P.a, P.W__minus__, P.W__plus__ ],
              color = [ '1' ],
              lorentz = [ L.VVVV8 ],
              couplings = {(0,0):C.GC_1709})

V_94 = Vertex(name = 'V_94',
              particles = [ P.a, P.Z, P.G0, P.G0 ],
              color = [ '1' ],
              lorentz = [ L.VVSS1, L.VVSS12, L.VVSS13, L.VVSS17, L.VVSS3 ],
              couplings = {(0,1):C.GC_905,(0,0):C.GC_1005,(0,2):C.GC_888,(0,4):C.GC_1062,(0,3):C.GC_904})

V_95 = Vertex(name = 'V_95',
              particles = [ P.a, P.Z, P.H, P.H ],
              color = [ '1' ],
              lorentz = [ L.VVSS1, L.VVSS12, L.VVSS13, L.VVSS17, L.VVSS3 ],
              couplings = {(0,1):C.GC_905,(0,0):C.GC_1005,(0,2):C.GC_888,(0,4):C.GC_1062,(0,3):C.GC_904})

V_96 = Vertex(name = 'V_96',
              particles = [ P.a, P.Z, P.H ],
              color = [ '1' ],
              lorentz = [ L.VVS1, L.VVS12, L.VVS3, L.VVS8, L.VVS9 ],
              couplings = {(0,3):C.GC_1754,(0,0):C.GC_1684,(0,4):C.GC_1681,(0,2):C.GC_1765,(0,1):C.GC_1753})

V_97 = Vertex(name = 'V_97',
              particles = [ P.Z, P.Z, P.G0, P.G0 ],
              color = [ '1' ],
              lorentz = [ L.VVSS11, L.VVSS13, L.VVSS16, L.VVSS2, L.VVSS23, L.VVSS5 ],
              couplings = {(0,2):C.GC_981,(0,3):C.GC_1006,(0,1):C.GC_889,(0,5):C.GC_1053,(0,4):C.GC_984,(0,0):C.GC_980})

V_98 = Vertex(name = 'V_98',
              particles = [ P.Z, P.Z, P.G0, P.G0 ],
              color = [ '1' ],
              lorentz = [ L.VVSS11 ],
              couplings = {(0,0):C.GC_1801})

V_99 = Vertex(name = 'V_99',
              particles = [ P.Z, P.Z, P.G__minus__, P.G__plus__ ],
              color = [ '1' ],
              lorentz = [ L.VVSS11, L.VVSS13, L.VVSS16, L.VVSS2, L.VVSS23, L.VVSS5 ],
              couplings = {(0,2):C.GC_982,(0,3):C.GC_1006,(0,1):C.GC_889,(0,5):C.GC_1054,(0,4):C.GC_983,(0,0):C.GC_979})

V_100 = Vertex(name = 'V_100',
               particles = [ P.Z, P.Z, P.G__minus__, P.G__plus__ ],
               color = [ '1' ],
               lorentz = [ L.VVSS11 ],
               couplings = {(0,0):C.GC_1803})

V_101 = Vertex(name = 'V_101',
               particles = [ P.Z, P.Z, P.H, P.H ],
               color = [ '1' ],
               lorentz = [ L.VVSS11, L.VVSS13, L.VVSS16, L.VVSS2, L.VVSS23, L.VVSS5 ],
               couplings = {(0,2):C.GC_981,(0,3):C.GC_1006,(0,1):C.GC_889,(0,5):C.GC_1053,(0,4):C.GC_984,(0,0):C.GC_980})

V_102 = Vertex(name = 'V_102',
               particles = [ P.Z, P.Z, P.H, P.H ],
               color = [ '1' ],
               lorentz = [ L.VVSS11 ],
               couplings = {(0,0):C.GC_1802})

V_103 = Vertex(name = 'V_103',
               particles = [ P.Z, P.Z, P.H ],
               color = [ '1' ],
               lorentz = [ L.VVS11, L.VVS14, L.VVS2, L.VVS5, L.VVS7, L.VVS9 ],
               couplings = {(0,0):C.GC_1758,(0,2):C.GC_1685,(0,5):C.GC_1682,(0,3):C.GC_1760,(0,1):C.GC_1759,(0,4):C.GC_1756})

V_104 = Vertex(name = 'V_104',
               particles = [ P.Z, P.Z, P.H ],
               color = [ '1' ],
               lorentz = [ L.VVS7 ],
               couplings = {(0,0):C.GC_1810})

V_105 = Vertex(name = 'V_105',
               particles = [ P.W__minus__, P.W__plus__, P.Z ],
               color = [ '1' ],
               lorentz = [ L.VVV10, L.VVV11, L.VVV12, L.VVV13, L.VVV2, L.VVV3, L.VVV4, L.VVV7, L.VVV8, L.VVV9 ],
               couplings = {(0,6):C.GC_1726,(0,4):C.GC_1727,(0,2):C.GC_578,(0,5):C.GC_1725,(0,3):C.GC_998,(0,1):C.GC_361,(0,0):C.GC_570,(0,7):C.GC_1722,(0,9):C.GC_1723,(0,8):C.GC_1721})

V_106 = Vertex(name = 'V_106',
               particles = [ P.W__minus__, P.W__plus__, P.Z ],
               color = [ '1' ],
               lorentz = [ L.VVV10 ],
               couplings = {(0,0):C.GC_1795})

V_107 = Vertex(name = 'V_107',
               particles = [ P.ghA, P.ghWm__tilde__, P.W__minus__ ],
               color = [ '1' ],
               lorentz = [ L.UUV1 ],
               couplings = {(0,0):C.GC_4})

V_108 = Vertex(name = 'V_108',
               particles = [ P.ghA, P.ghWm__tilde__, P.W__minus__ ],
               color = [ '1' ],
               lorentz = [ L.UUV1 ],
               couplings = {(0,0):C.GC_1719})

V_109 = Vertex(name = 'V_109',
               particles = [ P.ghA, P.ghWp__tilde__, P.W__plus__ ],
               color = [ '1' ],
               lorentz = [ L.UUV1 ],
               couplings = {(0,0):C.GC_5})

V_110 = Vertex(name = 'V_110',
               particles = [ P.ghA, P.ghWp__tilde__, P.W__plus__ ],
               color = [ '1' ],
               lorentz = [ L.UUV1 ],
               couplings = {(0,0):C.GC_1720})

V_111 = Vertex(name = 'V_111',
               particles = [ P.ghWm, P.ghA__tilde__, P.G__plus__ ],
               color = [ '1' ],
               lorentz = [ L.UUS1 ],
               couplings = {(0,0):C.GC_1561})

V_112 = Vertex(name = 'V_112',
               particles = [ P.ghWm, P.ghA__tilde__, P.W__plus__ ],
               color = [ '1' ],
               lorentz = [ L.UUV1 ],
               couplings = {(0,0):C.GC_4})

V_113 = Vertex(name = 'V_113',
               particles = [ P.ghWm, P.ghA__tilde__, P.W__plus__ ],
               color = [ '1' ],
               lorentz = [ L.UUV1 ],
               couplings = {(0,0):C.GC_1719})

V_114 = Vertex(name = 'V_114',
               particles = [ P.ghWm, P.ghWm__tilde__, P.G0 ],
               color = [ '1' ],
               lorentz = [ L.UUS1 ],
               couplings = {(0,0):C.GC_1504})

V_115 = Vertex(name = 'V_115',
               particles = [ P.ghWm, P.ghWm__tilde__, P.H ],
               color = [ '1' ],
               lorentz = [ L.UUS1 ],
               couplings = {(0,0):C.GC_1505})

V_116 = Vertex(name = 'V_116',
               particles = [ P.ghWm, P.ghWm__tilde__, P.H ],
               color = [ '1' ],
               lorentz = [ L.UUS1 ],
               couplings = {(0,0):C.GC_1508})

V_117 = Vertex(name = 'V_117',
               particles = [ P.ghWm, P.ghWm__tilde__, P.a ],
               color = [ '1' ],
               lorentz = [ L.UUV1 ],
               couplings = {(0,0):C.GC_5})

V_118 = Vertex(name = 'V_118',
               particles = [ P.ghWm, P.ghWm__tilde__, P.a ],
               color = [ '1' ],
               lorentz = [ L.UUV1 ],
               couplings = {(0,0):C.GC_1708})

V_119 = Vertex(name = 'V_119',
               particles = [ P.ghWm, P.ghWm__tilde__, P.Z ],
               color = [ '1' ],
               lorentz = [ L.UUV1 ],
               couplings = {(0,0):C.GC_570})

V_120 = Vertex(name = 'V_120',
               particles = [ P.ghWm, P.ghWm__tilde__, P.Z ],
               color = [ '1' ],
               lorentz = [ L.UUV1 ],
               couplings = {(0,0):C.GC_1796})

V_121 = Vertex(name = 'V_121',
               particles = [ P.ghWm, P.ghZ__tilde__, P.G__plus__ ],
               color = [ '1' ],
               lorentz = [ L.UUS1 ],
               couplings = {(0,0):C.GC_1739})

V_122 = Vertex(name = 'V_122',
               particles = [ P.ghWm, P.ghZ__tilde__, P.W__plus__ ],
               color = [ '1' ],
               lorentz = [ L.UUV1 ],
               couplings = {(0,0):C.GC_569})

V_123 = Vertex(name = 'V_123',
               particles = [ P.ghWm, P.ghZ__tilde__, P.W__plus__ ],
               color = [ '1' ],
               lorentz = [ L.UUV1 ],
               couplings = {(0,0):C.GC_1717})

V_124 = Vertex(name = 'V_124',
               particles = [ P.ghWp, P.ghA__tilde__, P.G__minus__ ],
               color = [ '1' ],
               lorentz = [ L.UUS1 ],
               couplings = {(0,0):C.GC_1560})

V_125 = Vertex(name = 'V_125',
               particles = [ P.ghWp, P.ghA__tilde__, P.W__minus__ ],
               color = [ '1' ],
               lorentz = [ L.UUV1 ],
               couplings = {(0,0):C.GC_5})

V_126 = Vertex(name = 'V_126',
               particles = [ P.ghWp, P.ghA__tilde__, P.W__minus__ ],
               color = [ '1' ],
               lorentz = [ L.UUV1 ],
               couplings = {(0,0):C.GC_1720})

V_127 = Vertex(name = 'V_127',
               particles = [ P.ghWp, P.ghWp__tilde__, P.G0 ],
               color = [ '1' ],
               lorentz = [ L.UUS1 ],
               couplings = {(0,0):C.GC_1507})

V_128 = Vertex(name = 'V_128',
               particles = [ P.ghWp, P.ghWp__tilde__, P.H ],
               color = [ '1' ],
               lorentz = [ L.UUS1 ],
               couplings = {(0,0):C.GC_1505})

V_129 = Vertex(name = 'V_129',
               particles = [ P.ghWp, P.ghWp__tilde__, P.H ],
               color = [ '1' ],
               lorentz = [ L.UUS1 ],
               couplings = {(0,0):C.GC_1508})

V_130 = Vertex(name = 'V_130',
               particles = [ P.ghWp, P.ghWp__tilde__, P.a ],
               color = [ '1' ],
               lorentz = [ L.UUV1 ],
               couplings = {(0,0):C.GC_4})

V_131 = Vertex(name = 'V_131',
               particles = [ P.ghWp, P.ghWp__tilde__, P.a ],
               color = [ '1' ],
               lorentz = [ L.UUV1 ],
               couplings = {(0,0):C.GC_1707})

V_132 = Vertex(name = 'V_132',
               particles = [ P.ghWp, P.ghWp__tilde__, P.Z ],
               color = [ '1' ],
               lorentz = [ L.UUV1 ],
               couplings = {(0,0):C.GC_569})

V_133 = Vertex(name = 'V_133',
               particles = [ P.ghWp, P.ghWp__tilde__, P.Z ],
               color = [ '1' ],
               lorentz = [ L.UUV1 ],
               couplings = {(0,0):C.GC_1797})

V_134 = Vertex(name = 'V_134',
               particles = [ P.ghWp, P.ghZ__tilde__, P.G__minus__ ],
               color = [ '1' ],
               lorentz = [ L.UUS1 ],
               couplings = {(0,0):C.GC_1738})

V_135 = Vertex(name = 'V_135',
               particles = [ P.ghWp, P.ghZ__tilde__, P.W__minus__ ],
               color = [ '1' ],
               lorentz = [ L.UUV1 ],
               couplings = {(0,0):C.GC_570})

V_136 = Vertex(name = 'V_136',
               particles = [ P.ghWp, P.ghZ__tilde__, P.W__minus__ ],
               color = [ '1' ],
               lorentz = [ L.UUV1 ],
               couplings = {(0,0):C.GC_1718})

V_137 = Vertex(name = 'V_137',
               particles = [ P.ghZ, P.ghWm__tilde__, P.G__minus__ ],
               color = [ '1' ],
               lorentz = [ L.UUS1 ],
               couplings = {(0,0):C.GC_1740})

V_138 = Vertex(name = 'V_138',
               particles = [ P.ghZ, P.ghWm__tilde__, P.W__minus__ ],
               color = [ '1' ],
               lorentz = [ L.UUV1 ],
               couplings = {(0,0):C.GC_569})

V_139 = Vertex(name = 'V_139',
               particles = [ P.ghZ, P.ghWm__tilde__, P.W__minus__ ],
               color = [ '1' ],
               lorentz = [ L.UUV1 ],
               couplings = {(0,0):C.GC_1717})

V_140 = Vertex(name = 'V_140',
               particles = [ P.ghZ, P.ghWp__tilde__, P.G__plus__ ],
               color = [ '1' ],
               lorentz = [ L.UUS1 ],
               couplings = {(0,0):C.GC_1737})

V_141 = Vertex(name = 'V_141',
               particles = [ P.ghZ, P.ghWp__tilde__, P.W__plus__ ],
               color = [ '1' ],
               lorentz = [ L.UUV1 ],
               couplings = {(0,0):C.GC_570})

V_142 = Vertex(name = 'V_142',
               particles = [ P.ghZ, P.ghWp__tilde__, P.W__plus__ ],
               color = [ '1' ],
               lorentz = [ L.UUV1 ],
               couplings = {(0,0):C.GC_1718})

V_143 = Vertex(name = 'V_143',
               particles = [ P.ghZ, P.ghZ__tilde__, P.H ],
               color = [ '1' ],
               lorentz = [ L.UUS1 ],
               couplings = {(0,0):C.GC_1755})

V_144 = Vertex(name = 'V_144',
               particles = [ P.ghZ, P.ghZ__tilde__, P.H ],
               color = [ '1' ],
               lorentz = [ L.UUS1 ],
               couplings = {(0,0):C.GC_1757})

V_145 = Vertex(name = 'V_145',
               particles = [ P.g, P.g, P.g ],
               color = [ 'f(1,2,3)' ],
               lorentz = [ L.VVV10, L.VVV11, L.VVV12, L.VVV14, L.VVV4 ],
               couplings = {(0,4):C.GC_1724,(0,2):C.GC_157,(0,3):C.GC_985,(0,1):C.GC_164,(0,0):C.GC_11})

V_146 = Vertex(name = 'V_146',
               particles = [ P.g, P.g, P.g ],
               color = [ 'f(1,2,3)' ],
               lorentz = [ L.VVV10 ],
               couplings = {(0,0):C.GC_1711})

V_147 = Vertex(name = 'V_147',
               particles = [ P.ghG, P.ghG__tilde__, P.g ],
               color = [ 'f(1,2,3)' ],
               lorentz = [ L.UUV1 ],
               couplings = {(0,0):C.GC_11})

V_148 = Vertex(name = 'V_148',
               particles = [ P.g, P.g, P.g, P.G0, P.G0 ],
               color = [ 'f(1,2,3)' ],
               lorentz = [ L.VVVSS18, L.VVVSS5 ],
               couplings = {(0,1):C.GC_1008,(0,0):C.GC_165})

V_149 = Vertex(name = 'V_149',
               particles = [ P.g, P.g, P.g, P.G__minus__, P.G__plus__ ],
               color = [ 'f(1,2,3)' ],
               lorentz = [ L.VVVSS18, L.VVVSS5 ],
               couplings = {(0,1):C.GC_1008,(0,0):C.GC_165})

V_150 = Vertex(name = 'V_150',
               particles = [ P.g, P.g, P.g, P.H, P.H ],
               color = [ 'f(1,2,3)' ],
               lorentz = [ L.VVVSS18, L.VVVSS5 ],
               couplings = {(0,1):C.GC_1008,(0,0):C.GC_165})

V_151 = Vertex(name = 'V_151',
               particles = [ P.g, P.g, P.g, P.H ],
               color = [ 'f(1,2,3)' ],
               lorentz = [ L.VVVS17, L.VVVS5 ],
               couplings = {(0,1):C.GC_1687,(0,0):C.GC_15})

V_152 = Vertex(name = 'V_152',
               particles = [ P.g, P.g, P.g, P.H ],
               color = [ 'f(1,2,3)' ],
               lorentz = [ L.VVVS17 ],
               couplings = {(0,0):C.GC_1502})

V_153 = Vertex(name = 'V_153',
               particles = [ P.g, P.g, P.g, P.g ],
               color = [ 'f(-1,1,2)*f(3,4,-1)', 'f(-1,1,3)*f(2,4,-1)', 'f(-1,1,4)*f(2,3,-1)' ],
               lorentz = [ L.VVVV1, L.VVVV10, L.VVVV12, L.VVVV13, L.VVVV16, L.VVVV18, L.VVVV19, L.VVVV2, L.VVVV3, L.VVVV5, L.VVVV6, L.VVVV9 ],
               couplings = {(0,10):C.GC_987,(1,9):C.GC_986,(2,8):C.GC_986,(2,5):C.GC_159,(0,7):C.GC_168,(1,6):C.GC_159,(0,4):C.GC_158,(1,3):C.GC_168,(2,2):C.GC_168,(1,11):C.GC_13,(0,0):C.GC_13,(2,1):C.GC_13})

V_154 = Vertex(name = 'V_154',
               particles = [ P.g, P.g, P.g, P.g ],
               color = [ 'f(-1,1,2)*f(3,4,-1)', 'f(-1,1,3)*f(2,4,-1)', 'f(-1,1,4)*f(2,3,-1)' ],
               lorentz = [ L.VVVV1, L.VVVV10, L.VVVV9 ],
               couplings = {(1,2):C.GC_1712,(0,0):C.GC_1712,(2,1):C.GC_1712})

V_155 = Vertex(name = 'V_155',
               particles = [ P.g, P.g, P.g, P.g, P.G0, P.G0 ],
               color = [ 'f(-1,1,2)*f(3,4,-1)', 'f(-1,1,3)*f(2,4,-1)', 'f(-1,1,4)*f(2,3,-1)' ],
               lorentz = [ L.VVVVSS1, L.VVVVSS3, L.VVVVSS4 ],
               couplings = {(1,1):C.GC_169,(0,0):C.GC_169,(2,2):C.GC_169})

V_156 = Vertex(name = 'V_156',
               particles = [ P.g, P.g, P.g, P.g, P.G__minus__, P.G__plus__ ],
               color = [ 'f(-1,1,2)*f(3,4,-1)', 'f(-1,1,3)*f(2,4,-1)', 'f(-1,1,4)*f(2,3,-1)' ],
               lorentz = [ L.VVVVSS1, L.VVVVSS3, L.VVVVSS4 ],
               couplings = {(1,1):C.GC_169,(0,0):C.GC_169,(2,2):C.GC_169})

V_157 = Vertex(name = 'V_157',
               particles = [ P.g, P.g, P.g, P.g, P.H, P.H ],
               color = [ 'f(-1,1,2)*f(3,4,-1)', 'f(-1,1,3)*f(2,4,-1)', 'f(-1,1,4)*f(2,3,-1)' ],
               lorentz = [ L.VVVVSS1, L.VVVVSS3, L.VVVVSS4 ],
               couplings = {(1,1):C.GC_169,(0,0):C.GC_169,(2,2):C.GC_169})

V_158 = Vertex(name = 'V_158',
               particles = [ P.g, P.g, P.g, P.g, P.H ],
               color = [ 'f(-1,1,2)*f(3,4,-1)', 'f(-1,1,3)*f(2,4,-1)', 'f(-1,1,4)*f(2,3,-1)' ],
               lorentz = [ L.VVVVS1, L.VVVVS3, L.VVVVS4 ],
               couplings = {(1,1):C.GC_16,(0,0):C.GC_16,(2,2):C.GC_16})

V_159 = Vertex(name = 'V_159',
               particles = [ P.g, P.g, P.g, P.g, P.H ],
               color = [ 'f(-1,1,2)*f(3,4,-1)', 'f(-1,1,3)*f(2,4,-1)', 'f(-1,1,4)*f(2,3,-1)' ],
               lorentz = [ L.VVVVS1, L.VVVVS3, L.VVVVS4 ],
               couplings = {(1,1):C.GC_1503,(0,0):C.GC_1503,(2,2):C.GC_1503})

V_160 = Vertex(name = 'V_160',
               particles = [ P.g, P.g, P.g, P.g, P.g ],
               color = [ 'f(-2,1,2)*f(-1,-2,3)*f(4,5,-1)', 'f(-2,1,2)*f(-1,-2,4)*f(3,5,-1)', 'f(-2,1,2)*f(-1,-2,5)*f(3,4,-1)', 'f(-2,1,3)*f(-1,-2,2)*f(4,5,-1)', 'f(-2,1,3)*f(-1,-2,4)*f(2,5,-1)', 'f(-2,1,3)*f(-1,-2,5)*f(2,4,-1)', 'f(-2,1,4)*f(-1,-2,2)*f(3,5,-1)', 'f(-2,1,4)*f(-1,-2,3)*f(2,5,-1)', 'f(-2,1,4)*f(-1,-2,5)*f(2,3,-1)', 'f(-2,1,5)*f(-1,-2,2)*f(3,4,-1)', 'f(-2,1,5)*f(-1,-2,3)*f(2,4,-1)', 'f(-2,1,5)*f(-1,-2,4)*f(2,3,-1)', 'f(-2,2,3)*f(-1,-2,1)*f(4,5,-1)', 'f(-2,2,3)*f(-1,-2,4)*f(1,5,-1)', 'f(-2,2,3)*f(-1,-2,5)*f(1,4,-1)', 'f(-2,2,4)*f(-1,-2,1)*f(3,5,-1)', 'f(-2,2,4)*f(-1,-2,3)*f(1,5,-1)', 'f(-2,2,4)*f(-1,-2,5)*f(1,3,-1)', 'f(-2,2,5)*f(-1,-2,1)*f(3,4,-1)', 'f(-2,2,5)*f(-1,-2,3)*f(1,4,-1)', 'f(-2,2,5)*f(-1,-2,4)*f(1,3,-1)', 'f(-2,3,4)*f(-1,-2,1)*f(2,5,-1)', 'f(-2,3,4)*f(-1,-2,2)*f(1,5,-1)', 'f(-2,3,4)*f(-1,-2,5)*f(1,2,-1)', 'f(-2,3,5)*f(-1,-2,1)*f(2,4,-1)', 'f(-2,3,5)*f(-1,-2,2)*f(1,4,-1)', 'f(-2,3,5)*f(-1,-2,4)*f(1,2,-1)', 'f(-2,4,5)*f(-1,-2,1)*f(2,3,-1)', 'f(-2,4,5)*f(-1,-2,2)*f(1,3,-1)', 'f(-2,4,5)*f(-1,-2,3)*f(1,2,-1)' ],
               lorentz = [ L.VVVVV1, L.VVVVV10, L.VVVVV11, L.VVVVV12, L.VVVVV13, L.VVVVV14, L.VVVVV15, L.VVVVV16, L.VVVVV17, L.VVVVV18, L.VVVVV19, L.VVVVV2, L.VVVVV21, L.VVVVV22, L.VVVVV23, L.VVVVV24, L.VVVVV25, L.VVVVV26, L.VVVVV27, L.VVVVV28, L.VVVVV29, L.VVVVV3, L.VVVVV30, L.VVVVV31, L.VVVVV32, L.VVVVV33, L.VVVVV34, L.VVVVV37, L.VVVVV38, L.VVVVV39, L.VVVVV4, L.VVVVV40, L.VVVVV42, L.VVVVV43, L.VVVVV44, L.VVVVV45, L.VVVVV46, L.VVVVV49, L.VVVVV5, L.VVVVV50, L.VVVVV51, L.VVVVV52, L.VVVVV53, L.VVVVV54, L.VVVVV55, L.VVVVV56, L.VVVVV57, L.VVVVV58, L.VVVVV59, L.VVVVV6, L.VVVVV60, L.VVVVV61, L.VVVVV62, L.VVVVV64, L.VVVVV65, L.VVVVV66, L.VVVVV67, L.VVVVV68, L.VVVVV69, L.VVVVV7, L.VVVVV70, L.VVVVV71, L.VVVVV73, L.VVVVV74, L.VVVVV75, L.VVVVV76, L.VVVVV78, L.VVVVV79, L.VVVVV8, L.VVVVV82, L.VVVVV83, L.VVVVV84, L.VVVVV85, L.VVVVV88, L.VVVVV89 ],
               couplings = {(18,22):C.GC_988,(15,18):C.GC_989,(12,49):C.GC_988,(24,19):C.GC_988,(21,20):C.GC_989,(27,38):C.GC_988,(9,24):C.GC_988,(6,25):C.GC_989,(3,59):C.GC_988,(25,23):C.GC_989,(22,26):C.GC_988,(28,68):C.GC_989,(10,28):C.GC_989,(7,29):C.GC_988,(0,0):C.GC_989,(19,31):C.GC_989,(16,27):C.GC_988,(29,1):C.GC_988,(11,34):C.GC_988,(4,2):C.GC_989,(1,11):C.GC_988,(20,32):C.GC_988,(13,3):C.GC_989,(26,33):C.GC_988,(8,35):C.GC_989,(5,5):C.GC_988,(2,21):C.GC_989,(17,37):C.GC_989,(14,4):C.GC_988,(23,36):C.GC_989,(24,43):C.GC_170,(10,71):C.GC_163,(21,44):C.GC_171,(7,67):C.GC_162,(18,44):C.GC_170,(9,72):C.GC_163,(15,43):C.GC_171,(6,66):C.GC_162,(28,10):C.GC_170,(20,8):C.GC_163,(22,56):C.GC_170,(13,63):C.GC_162,(18,52):C.GC_161,(9,56):C.GC_171,(12,7):C.GC_162,(3,10):C.GC_171,(29,12):C.GC_170,(26,17):C.GC_163,(16,60):C.GC_170,(24,41):C.GC_161,(10,60):C.GC_171,(0,12):C.GC_171,(29,74):C.GC_161,(26,65):C.GC_171,(28,73):C.GC_161,(20,64):C.GC_171,(4,64):C.GC_170,(1,65):C.GC_170,(17,14):C.GC_163,(25,55):C.GC_170,(14,62):C.GC_162,(15,46):C.GC_161,(6,55):C.GC_171,(23,30):C.GC_163,(19,61):C.GC_170,(21,39):C.GC_161,(7,61):C.GC_171,(23,70):C.GC_171,(17,69):C.GC_171,(5,69):C.GC_170,(2,70):C.GC_170,(27,6):C.GC_170,(11,16):C.GC_163,(4,58):C.GC_162,(12,6):C.GC_171,(3,13):C.GC_162,(19,48):C.GC_163,(16,47):C.GC_162,(25,42):C.GC_161,(13,45):C.GC_170,(27,50):C.GC_161,(11,45):C.GC_171,(14,51):C.GC_171,(8,51):C.GC_170,(8,15):C.GC_163,(5,57):C.GC_162,(22,40):C.GC_161,(1,53):C.GC_162,(0,9):C.GC_162,(2,54):C.GC_162})

V_161 = Vertex(name = 'V_161',
               particles = [ P.g, P.g, P.g, P.g, P.g, P.g ],
               color = [ 'f(-2,-3,3)*f(-2,2,4)*f(-1,-3,5)*f(1,6,-1)', 'f(-2,-3,3)*f(-2,2,4)*f(-1,-3,6)*f(1,5,-1)', 'f(-2,-3,3)*f(-2,2,5)*f(-1,-3,4)*f(1,6,-1)', 'f(-2,-3,3)*f(-2,2,5)*f(-1,-3,6)*f(1,4,-1)', 'f(-2,-3,3)*f(-2,2,6)*f(-1,-3,4)*f(1,5,-1)', 'f(-2,-3,3)*f(-2,2,6)*f(-1,-3,5)*f(1,4,-1)', 'f(-2,-3,4)*f(-2,2,3)*f(-1,-3,5)*f(1,6,-1)', 'f(-2,-3,4)*f(-2,2,3)*f(-1,-3,6)*f(1,5,-1)', 'f(-2,-3,4)*f(-2,2,5)*f(-1,-3,3)*f(1,6,-1)', 'f(-2,-3,4)*f(-2,2,5)*f(-1,-3,6)*f(1,3,-1)', 'f(-2,-3,4)*f(-2,2,6)*f(-1,-3,3)*f(1,5,-1)', 'f(-2,-3,4)*f(-2,2,6)*f(-1,-3,5)*f(1,3,-1)', 'f(-2,-3,4)*f(-2,3,5)*f(-1,-3,6)*f(1,2,-1)', 'f(-2,-3,4)*f(-2,3,6)*f(-1,-3,5)*f(1,2,-1)', 'f(-2,-3,5)*f(-2,2,3)*f(-1,-3,4)*f(1,6,-1)', 'f(-2,-3,5)*f(-2,2,3)*f(-1,-3,6)*f(1,4,-1)', 'f(-2,-3,5)*f(-2,2,4)*f(-1,-3,3)*f(1,6,-1)', 'f(-2,-3,5)*f(-2,2,4)*f(-1,-3,6)*f(1,3,-1)', 'f(-2,-3,5)*f(-2,2,6)*f(-1,-3,3)*f(1,4,-1)', 'f(-2,-3,5)*f(-2,2,6)*f(-1,-3,4)*f(1,3,-1)', 'f(-2,-3,5)*f(-2,3,4)*f(-1,-3,6)*f(1,2,-1)', 'f(-2,-3,5)*f(-2,3,6)*f(-1,-3,4)*f(1,2,-1)', 'f(-2,-3,6)*f(-2,2,3)*f(-1,-3,4)*f(1,5,-1)', 'f(-2,-3,6)*f(-2,2,3)*f(-1,-3,5)*f(1,4,-1)', 'f(-2,-3,6)*f(-2,2,4)*f(-1,-3,3)*f(1,5,-1)', 'f(-2,-3,6)*f(-2,2,4)*f(-1,-3,5)*f(1,3,-1)', 'f(-2,-3,6)*f(-2,2,5)*f(-1,-3,3)*f(1,4,-1)', 'f(-2,-3,6)*f(-2,2,5)*f(-1,-3,4)*f(1,3,-1)', 'f(-2,-3,6)*f(-2,3,4)*f(-1,-3,5)*f(1,2,-1)', 'f(-2,-3,6)*f(-2,3,5)*f(-1,-3,4)*f(1,2,-1)', 'f(-3,1,2)*f(-2,3,4)*f(-1,-2,-3)*f(5,6,-1)', 'f(-3,1,2)*f(-2,3,5)*f(-1,-2,-3)*f(4,6,-1)', 'f(-3,1,2)*f(-2,3,6)*f(-1,-2,-3)*f(4,5,-1)', 'f(-3,1,2)*f(-2,4,5)*f(-1,-2,-3)*f(3,6,-1)', 'f(-3,1,2)*f(-2,4,6)*f(-1,-2,-3)*f(3,5,-1)', 'f(-3,1,2)*f(-2,5,6)*f(-1,-2,-3)*f(3,4,-1)', 'f(-3,1,3)*f(-2,2,4)*f(-1,-2,-3)*f(5,6,-1)', 'f(-3,1,3)*f(-2,2,5)*f(-1,-2,-3)*f(4,6,-1)', 'f(-3,1,3)*f(-2,2,6)*f(-1,-2,-3)*f(4,5,-1)', 'f(-3,1,3)*f(-2,4,5)*f(-1,-2,-3)*f(2,6,-1)', 'f(-3,1,3)*f(-2,4,6)*f(-1,-2,-3)*f(2,5,-1)', 'f(-3,1,3)*f(-2,5,6)*f(-1,-2,-3)*f(2,4,-1)', 'f(-3,1,4)*f(-2,2,3)*f(-1,-2,-3)*f(5,6,-1)', 'f(-3,1,4)*f(-2,2,5)*f(-1,-2,-3)*f(3,6,-1)', 'f(-3,1,4)*f(-2,2,6)*f(-1,-2,-3)*f(3,5,-1)', 'f(-3,1,4)*f(-2,3,5)*f(-1,-2,-3)*f(2,6,-1)', 'f(-3,1,4)*f(-2,3,6)*f(-1,-2,-3)*f(2,5,-1)', 'f(-3,1,4)*f(-2,5,6)*f(-1,-2,-3)*f(2,3,-1)', 'f(-3,1,5)*f(-2,2,3)*f(-1,-2,-3)*f(4,6,-1)', 'f(-3,1,5)*f(-2,2,4)*f(-1,-2,-3)*f(3,6,-1)', 'f(-3,1,5)*f(-2,2,6)*f(-1,-2,-3)*f(3,4,-1)', 'f(-3,1,5)*f(-2,3,4)*f(-1,-2,-3)*f(2,6,-1)', 'f(-3,1,5)*f(-2,3,6)*f(-1,-2,-3)*f(2,4,-1)', 'f(-3,1,5)*f(-2,4,6)*f(-1,-2,-3)*f(2,3,-1)', 'f(-3,1,6)*f(-2,2,3)*f(-1,-2,-3)*f(4,5,-1)', 'f(-3,1,6)*f(-2,2,4)*f(-1,-2,-3)*f(3,5,-1)', 'f(-3,1,6)*f(-2,2,5)*f(-1,-2,-3)*f(3,4,-1)', 'f(-3,1,6)*f(-2,3,4)*f(-1,-2,-3)*f(2,5,-1)', 'f(-3,1,6)*f(-2,3,5)*f(-1,-2,-3)*f(2,4,-1)', 'f(-3,1,6)*f(-2,4,5)*f(-1,-2,-3)*f(2,3,-1)', 'f(-3,2,3)*f(-2,1,4)*f(-1,-2,-3)*f(5,6,-1)', 'f(-3,2,3)*f(-2,1,5)*f(-1,-2,-3)*f(4,6,-1)', 'f(-3,2,3)*f(-2,1,6)*f(-1,-2,-3)*f(4,5,-1)', 'f(-3,2,3)*f(-2,-3,4)*f(-1,-2,1)*f(5,6,-1)', 'f(-3,2,3)*f(-2,-3,5)*f(-1,-2,1)*f(4,6,-1)', 'f(-3,2,3)*f(-2,-3,6)*f(-1,-2,1)*f(4,5,-1)', 'f(-3,2,3)*f(-2,4,5)*f(-1,-2,-3)*f(1,6,-1)', 'f(-3,2,3)*f(-2,4,6)*f(-1,-2,-3)*f(1,5,-1)', 'f(-3,2,3)*f(-2,5,6)*f(-1,-2,-3)*f(1,4,-1)', 'f(-3,2,4)*f(-2,1,3)*f(-1,-2,-3)*f(5,6,-1)', 'f(-3,2,4)*f(-2,1,5)*f(-1,-2,-3)*f(3,6,-1)', 'f(-3,2,4)*f(-2,1,6)*f(-1,-2,-3)*f(3,5,-1)', 'f(-3,2,4)*f(-2,-3,3)*f(-1,-2,1)*f(5,6,-1)', 'f(-3,2,4)*f(-2,-3,5)*f(-1,-2,1)*f(3,6,-1)', 'f(-3,2,4)*f(-2,3,5)*f(-1,-2,-3)*f(1,6,-1)', 'f(-3,2,4)*f(-2,-3,6)*f(-1,-2,1)*f(3,5,-1)', 'f(-3,2,4)*f(-2,3,6)*f(-1,-2,-3)*f(1,5,-1)', 'f(-3,2,4)*f(-2,5,6)*f(-1,-2,-3)*f(1,3,-1)', 'f(-3,2,5)*f(-2,1,3)*f(-1,-2,-3)*f(4,6,-1)', 'f(-3,2,5)*f(-2,1,4)*f(-1,-2,-3)*f(3,6,-1)', 'f(-3,2,5)*f(-2,1,6)*f(-1,-2,-3)*f(3,4,-1)', 'f(-3,2,5)*f(-2,-3,3)*f(-1,-2,1)*f(4,6,-1)', 'f(-3,2,5)*f(-2,-3,4)*f(-1,-2,1)*f(3,6,-1)', 'f(-3,2,5)*f(-2,3,4)*f(-1,-2,-3)*f(1,6,-1)', 'f(-3,2,5)*f(-2,-3,6)*f(-1,-2,1)*f(3,4,-1)', 'f(-3,2,5)*f(-2,3,6)*f(-1,-2,-3)*f(1,4,-1)', 'f(-3,2,5)*f(-2,4,6)*f(-1,-2,-3)*f(1,3,-1)', 'f(-3,2,6)*f(-2,1,3)*f(-1,-2,-3)*f(4,5,-1)', 'f(-3,2,6)*f(-2,1,4)*f(-1,-2,-3)*f(3,5,-1)', 'f(-3,2,6)*f(-2,1,5)*f(-1,-2,-3)*f(3,4,-1)', 'f(-3,2,6)*f(-2,-3,3)*f(-1,-2,1)*f(4,5,-1)', 'f(-3,2,6)*f(-2,-3,4)*f(-1,-2,1)*f(3,5,-1)', 'f(-3,2,6)*f(-2,3,4)*f(-1,-2,-3)*f(1,5,-1)', 'f(-3,2,6)*f(-2,-3,5)*f(-1,-2,1)*f(3,4,-1)', 'f(-3,2,6)*f(-2,3,5)*f(-1,-2,-3)*f(1,4,-1)', 'f(-3,2,6)*f(-2,4,5)*f(-1,-2,-3)*f(1,3,-1)', 'f(-3,3,4)*f(-2,1,2)*f(-1,-2,-3)*f(5,6,-1)', 'f(-3,3,4)*f(-2,1,5)*f(-1,-2,-3)*f(2,6,-1)', 'f(-3,3,4)*f(-2,1,6)*f(-1,-2,-3)*f(2,5,-1)', 'f(-3,3,4)*f(-2,2,5)*f(-1,-2,-3)*f(1,6,-1)', 'f(-3,3,4)*f(-2,2,6)*f(-1,-2,-3)*f(1,5,-1)', 'f(-3,3,4)*f(-2,-3,2)*f(-1,-2,1)*f(5,6,-1)', 'f(-3,3,4)*f(-2,-3,2)*f(-1,-2,5)*f(1,6,-1)', 'f(-3,3,4)*f(-2,-3,2)*f(-1,-2,6)*f(1,5,-1)', 'f(-3,3,4)*f(-2,-3,5)*f(-1,-2,1)*f(2,6,-1)', 'f(-3,3,4)*f(-2,-3,5)*f(-1,-2,2)*f(1,6,-1)', 'f(-3,3,4)*f(-2,-3,6)*f(-1,-2,1)*f(2,5,-1)', 'f(-3,3,4)*f(-2,-3,6)*f(-1,-2,2)*f(1,5,-1)', 'f(-3,3,4)*f(-2,5,6)*f(-1,-2,-3)*f(1,2,-1)', 'f(-3,3,5)*f(-2,1,2)*f(-1,-2,-3)*f(4,6,-1)', 'f(-3,3,5)*f(-2,1,4)*f(-1,-2,-3)*f(2,6,-1)', 'f(-3,3,5)*f(-2,1,6)*f(-1,-2,-3)*f(2,4,-1)', 'f(-3,3,5)*f(-2,2,4)*f(-1,-2,-3)*f(1,6,-1)', 'f(-3,3,5)*f(-2,2,6)*f(-1,-2,-3)*f(1,4,-1)', 'f(-3,3,5)*f(-2,-3,2)*f(-1,-2,1)*f(4,6,-1)', 'f(-3,3,5)*f(-2,-3,2)*f(-1,-2,4)*f(1,6,-1)', 'f(-3,3,5)*f(-2,-3,2)*f(-1,-2,6)*f(1,4,-1)', 'f(-3,3,5)*f(-2,-3,4)*f(-1,-2,1)*f(2,6,-1)', 'f(-3,3,5)*f(-2,-3,4)*f(-1,-2,2)*f(1,6,-1)', 'f(-3,3,5)*f(-2,-3,6)*f(-1,-2,1)*f(2,4,-1)', 'f(-3,3,5)*f(-2,-3,6)*f(-1,-2,2)*f(1,4,-1)', 'f(-3,3,5)*f(-2,4,6)*f(-1,-2,-3)*f(1,2,-1)', 'f(-3,3,6)*f(-2,1,2)*f(-1,-2,-3)*f(4,5,-1)', 'f(-3,3,6)*f(-2,1,4)*f(-1,-2,-3)*f(2,5,-1)', 'f(-3,3,6)*f(-2,1,5)*f(-1,-2,-3)*f(2,4,-1)', 'f(-3,3,6)*f(-2,2,4)*f(-1,-2,-3)*f(1,5,-1)', 'f(-3,3,6)*f(-2,2,5)*f(-1,-2,-3)*f(1,4,-1)', 'f(-3,3,6)*f(-2,-3,2)*f(-1,-2,1)*f(4,5,-1)', 'f(-3,3,6)*f(-2,-3,2)*f(-1,-2,4)*f(1,5,-1)', 'f(-3,3,6)*f(-2,-3,2)*f(-1,-2,5)*f(1,4,-1)', 'f(-3,3,6)*f(-2,-3,4)*f(-1,-2,1)*f(2,5,-1)', 'f(-3,3,6)*f(-2,-3,4)*f(-1,-2,2)*f(1,5,-1)', 'f(-3,3,6)*f(-2,-3,5)*f(-1,-2,1)*f(2,4,-1)', 'f(-3,3,6)*f(-2,-3,5)*f(-1,-2,2)*f(1,4,-1)', 'f(-3,3,6)*f(-2,4,5)*f(-1,-2,-3)*f(1,2,-1)', 'f(-3,4,5)*f(-2,1,2)*f(-1,-2,-3)*f(3,6,-1)', 'f(-3,4,5)*f(-2,1,3)*f(-1,-2,-3)*f(2,6,-1)', 'f(-3,4,5)*f(-2,1,6)*f(-1,-2,-3)*f(2,3,-1)', 'f(-3,4,5)*f(-2,2,3)*f(-1,-2,-3)*f(1,6,-1)', 'f(-3,4,5)*f(-2,2,6)*f(-1,-2,-3)*f(1,3,-1)', 'f(-3,4,5)*f(-2,-3,2)*f(-1,-2,1)*f(3,6,-1)', 'f(-3,4,5)*f(-2,-3,2)*f(-1,-2,3)*f(1,6,-1)', 'f(-3,4,5)*f(-2,-3,2)*f(-1,-2,6)*f(1,3,-1)', 'f(-3,4,5)*f(-2,-3,3)*f(-1,-2,1)*f(2,6,-1)', 'f(-3,4,5)*f(-2,-3,3)*f(-1,-2,2)*f(1,6,-1)', 'f(-3,4,5)*f(-2,-3,3)*f(-1,-2,6)*f(1,2,-1)', 'f(-3,4,5)*f(-2,-3,6)*f(-1,-2,1)*f(2,3,-1)', 'f(-3,4,5)*f(-2,-3,6)*f(-1,-2,2)*f(1,3,-1)', 'f(-3,4,5)*f(-2,-3,6)*f(-1,-2,3)*f(1,2,-1)', 'f(-3,4,5)*f(-2,3,6)*f(-1,-2,-3)*f(1,2,-1)', 'f(-3,4,6)*f(-2,1,2)*f(-1,-2,-3)*f(3,5,-1)', 'f(-3,4,6)*f(-2,1,3)*f(-1,-2,-3)*f(2,5,-1)', 'f(-3,4,6)*f(-2,1,5)*f(-1,-2,-3)*f(2,3,-1)', 'f(-3,4,6)*f(-2,2,3)*f(-1,-2,-3)*f(1,5,-1)', 'f(-3,4,6)*f(-2,2,5)*f(-1,-2,-3)*f(1,3,-1)', 'f(-3,4,6)*f(-2,-3,2)*f(-1,-2,1)*f(3,5,-1)', 'f(-3,4,6)*f(-2,-3,2)*f(-1,-2,3)*f(1,5,-1)', 'f(-3,4,6)*f(-2,-3,2)*f(-1,-2,5)*f(1,3,-1)', 'f(-3,4,6)*f(-2,-3,3)*f(-1,-2,1)*f(2,5,-1)', 'f(-3,4,6)*f(-2,-3,3)*f(-1,-2,2)*f(1,5,-1)', 'f(-3,4,6)*f(-2,-3,3)*f(-1,-2,5)*f(1,2,-1)', 'f(-3,4,6)*f(-2,-3,5)*f(-1,-2,1)*f(2,3,-1)', 'f(-3,4,6)*f(-2,-3,5)*f(-1,-2,2)*f(1,3,-1)', 'f(-3,4,6)*f(-2,-3,5)*f(-1,-2,3)*f(1,2,-1)', 'f(-3,4,6)*f(-2,3,5)*f(-1,-2,-3)*f(1,2,-1)', 'f(-3,5,6)*f(-2,1,2)*f(-1,-2,-3)*f(3,4,-1)', 'f(-3,5,6)*f(-2,1,3)*f(-1,-2,-3)*f(2,4,-1)', 'f(-3,5,6)*f(-2,1,4)*f(-1,-2,-3)*f(2,3,-1)', 'f(-3,5,6)*f(-2,2,3)*f(-1,-2,-3)*f(1,4,-1)', 'f(-3,5,6)*f(-2,2,4)*f(-1,-2,-3)*f(1,3,-1)', 'f(-3,5,6)*f(-2,-3,2)*f(-1,-2,1)*f(3,4,-1)', 'f(-3,5,6)*f(-2,-3,2)*f(-1,-2,3)*f(1,4,-1)', 'f(-3,5,6)*f(-2,-3,2)*f(-1,-2,4)*f(1,3,-1)', 'f(-3,5,6)*f(-2,-3,3)*f(-1,-2,1)*f(2,4,-1)', 'f(-3,5,6)*f(-2,-3,3)*f(-1,-2,2)*f(1,4,-1)', 'f(-3,5,6)*f(-2,-3,3)*f(-1,-2,4)*f(1,2,-1)', 'f(-3,5,6)*f(-2,-3,4)*f(-1,-2,1)*f(2,3,-1)', 'f(-3,5,6)*f(-2,-3,4)*f(-1,-2,2)*f(1,3,-1)', 'f(-3,5,6)*f(-2,-3,4)*f(-1,-2,3)*f(1,2,-1)', 'f(-3,5,6)*f(-2,3,4)*f(-1,-2,-3)*f(1,2,-1)' ],
               lorentz = [ L.VVVVVV1, L.VVVVVV10, L.VVVVVV100, L.VVVVVV101, L.VVVVVV102, L.VVVVVV103, L.VVVVVV104, L.VVVVVV105, L.VVVVVV106, L.VVVVVV107, L.VVVVVV108, L.VVVVVV109, L.VVVVVV11, L.VVVVVV110, L.VVVVVV111, L.VVVVVV112, L.VVVVVV113, L.VVVVVV114, L.VVVVVV115, L.VVVVVV116, L.VVVVVV117, L.VVVVVV118, L.VVVVVV12, L.VVVVVV124, L.VVVVVV125, L.VVVVVV126, L.VVVVVV127, L.VVVVVV128, L.VVVVVV129, L.VVVVVV13, L.VVVVVV130, L.VVVVVV131, L.VVVVVV132, L.VVVVVV133, L.VVVVVV134, L.VVVVVV135, L.VVVVVV136, L.VVVVVV137, L.VVVVVV138, L.VVVVVV139, L.VVVVVV14, L.VVVVVV140, L.VVVVVV141, L.VVVVVV142, L.VVVVVV143, L.VVVVVV144, L.VVVVVV145, L.VVVVVV146, L.VVVVVV147, L.VVVVVV15, L.VVVVVV151, L.VVVVVV152, L.VVVVVV153, L.VVVVVV154, L.VVVVVV155, L.VVVVVV156, L.VVVVVV157, L.VVVVVV158, L.VVVVVV16, L.VVVVVV17, L.VVVVVV18, L.VVVVVV19, L.VVVVVV2, L.VVVVVV20, L.VVVVVV21, L.VVVVVV22, L.VVVVVV23, L.VVVVVV24, L.VVVVVV25, L.VVVVVV26, L.VVVVVV27, L.VVVVVV28, L.VVVVVV29, L.VVVVVV3, L.VVVVVV30, L.VVVVVV31, L.VVVVVV32, L.VVVVVV33, L.VVVVVV34, L.VVVVVV35, L.VVVVVV36, L.VVVVVV37, L.VVVVVV38, L.VVVVVV39, L.VVVVVV4, L.VVVVVV40, L.VVVVVV41, L.VVVVVV42, L.VVVVVV43, L.VVVVVV44, L.VVVVVV45, L.VVVVVV46, L.VVVVVV47, L.VVVVVV48, L.VVVVVV49, L.VVVVVV5, L.VVVVVV50, L.VVVVVV51, L.VVVVVV52, L.VVVVVV53, L.VVVVVV54, L.VVVVVV55, L.VVVVVV56, L.VVVVVV57, L.VVVVVV58, L.VVVVVV59, L.VVVVVV6, L.VVVVVV60, L.VVVVVV61, L.VVVVVV62, L.VVVVVV63, L.VVVVVV64, L.VVVVVV65, L.VVVVVV66, L.VVVVVV67, L.VVVVVV68, L.VVVVVV69, L.VVVVVV7, L.VVVVVV70, L.VVVVVV71, L.VVVVVV72, L.VVVVVV73, L.VVVVVV74, L.VVVVVV75, L.VVVVVV76, L.VVVVVV77, L.VVVVVV78, L.VVVVVV79, L.VVVVVV8, L.VVVVVV80, L.VVVVVV81, L.VVVVVV82, L.VVVVVV83, L.VVVVVV84, L.VVVVVV85, L.VVVVVV86, L.VVVVVV87, L.VVVVVV88, L.VVVVVV89, L.VVVVVV9, L.VVVVVV90, L.VVVVVV91, L.VVVVVV92, L.VVVVVV93, L.VVVVVV94, L.VVVVVV95, L.VVVVVV96, L.VVVVVV97, L.VVVVVV98, L.VVVVVV99 ],
               couplings = {(50,134):C.GC_991,(56,127):C.GC_990,(80,127):C.GC_991,(89,134):C.GC_990,(44,85):C.GC_990,(55,89):C.GC_991,(71,89):C.GC_990,(88,85):C.GC_991,(43,75):C.GC_991,(49,71):C.GC_990,(70,71):C.GC_991,(79,75):C.GC_990,(38,59):C.GC_991,(54,40):C.GC_990,(62,40):C.GC_991,(87,59):C.GC_990,(37,139):C.GC_990,(48,12):C.GC_991,(61,12):C.GC_990,(78,139):C.GC_991,(36,128):C.GC_991,(42,117):C.GC_990,(60,117):C.GC_991,(69,128):C.GC_990,(52,129):C.GC_991,(58,136):C.GC_990,(111,136):C.GC_991,(124,129):C.GC_990,(46,90):C.GC_990,(57,87):C.GC_991,(98,87):C.GC_990,(123,90):C.GC_991,(45,72):C.GC_991,(51,77):C.GC_990,(97,77):C.GC_991,(110,72):C.GC_990,(32,95):C.GC_991,(122,95):C.GC_990,(31,62):C.GC_990,(109,62):C.GC_991,(30,0):C.GC_991,(96,0):C.GC_990,(53,137):C.GC_991,(59,132):C.GC_990,(137,132):C.GC_991,(152,137):C.GC_990,(40,60):C.GC_990,(151,60):C.GC_991,(39,1):C.GC_991,(136,1):C.GC_990,(34,84):C.GC_991,(150,84):C.GC_990,(33,73):C.GC_990,(135,73):C.GC_991,(47,92):C.GC_991,(167,92):C.GC_990,(41,58):C.GC_990,(166,58):C.GC_991,(35,106):C.GC_991,(165,106):C.GC_990,(85,135):C.GC_991,(94,130):C.GC_990,(113,130):C.GC_991,(126,135):C.GC_990,(76,86):C.GC_990,(92,91):C.GC_991,(100,91):C.GC_990,(125,86):C.GC_991,(74,76):C.GC_991,(83,74):C.GC_990,(99,74):C.GC_991,(112,76):C.GC_990,(86,131):C.GC_991,(95,140):C.GC_990,(139,140):C.GC_991,(154,131):C.GC_990,(67,49):C.GC_990,(153,49):C.GC_991,(66,22):C.GC_991,(138,22):C.GC_990,(77,88):C.GC_991,(169,88):C.GC_990,(68,61):C.GC_990,(168,61):C.GC_991,(121,138):C.GC_991,(134,133):C.GC_990,(149,133):C.GC_991,(164,138):C.GC_990,(121,10):C.GC_172,(134,20):C.GC_173,(149,20):C.GC_172,(164,10):C.GC_173,(160,67):C.GC_167,(145,29):C.GC_166,(13,102):C.GC_166,(12,78):C.GC_167,(77,116):C.GC_172,(95,38):C.GC_172,(139,38):C.GC_173,(169,116):C.GC_173,(172,65):C.GC_167,(142,68):C.GC_167,(19,33):C.GC_166,(17,81):C.GC_167,(68,126):C.GC_172,(94,46):C.GC_172,(113,46):C.GC_173,(168,126):C.GC_173,(171,99):C.GC_167,(116,103):C.GC_167,(18,45):C.GC_166,(15,123):C.GC_167,(67,44):C.GC_173,(76,39):C.GC_173,(125,39):C.GC_172,(153,44):C.GC_172,(131,31):C.GC_167,(159,24):C.GC_167,(7,41):C.GC_166,(1,34):C.GC_166,(47,126):C.GC_173,(53,44):C.GC_172,(152,44):C.GC_173,(167,126):C.GC_172,(41,116):C.GC_173,(52,39):C.GC_172,(124,39):C.GC_173,(166,116):C.GC_172,(39,38):C.GC_173,(45,46):C.GC_173,(110,46):C.GC_172,(136,38):C.GC_172,(117,25):C.GC_167,(143,27):C.GC_167,(34,10):C.GC_173,(44,46):C.GC_172,(88,46):C.GC_173,(150,10):C.GC_172,(33,20):C.GC_172,(49,39):C.GC_173,(70,39):C.GC_172,(135,20):C.GC_173,(73,100):C.GC_167,(140,15):C.GC_166,(32,20):C.GC_173,(38,38):C.GC_172,(87,38):C.GC_173,(122,20):C.GC_172,(31,10):C.GC_172,(48,44):C.GC_173,(61,44):C.GC_172,(109,10):C.GC_173,(64,66):C.GC_167,(114,146):C.GC_166,(36,116):C.GC_172,(42,126):C.GC_172,(60,126):C.GC_173,(69,116):C.GC_173,(63,64):C.GC_166,(72,97):C.GC_166,(86,35):C.GC_172,(154,35):C.GC_173,(157,63):C.GC_167,(27,32):C.GC_166,(25,120):C.GC_167,(85,47):C.GC_172,(126,47):C.GC_173,(129,94):C.GC_167,(26,43):C.GC_166,(23,82):C.GC_167,(66,48):C.GC_173,(74,37):C.GC_173,(112,37):C.GC_172,(138,48):C.GC_172,(118,26):C.GC_167,(144,28):C.GC_167,(6,42):C.GC_166,(0,36):C.GC_166,(59,48):C.GC_172,(137,48):C.GC_173,(58,37):C.GC_172,(111,37):C.GC_173,(40,35):C.GC_173,(46,47):C.GC_173,(123,47):C.GC_172,(151,35):C.GC_172,(130,30):C.GC_167,(158,23):C.GC_167,(55,37):C.GC_173,(71,37):C.GC_172,(75,108):C.GC_167,(155,14):C.GC_166,(43,47):C.GC_172,(79,47):C.GC_173,(54,48):C.GC_173,(62,48):C.GC_172,(65,69):C.GC_167,(127,147):C.GC_166,(37,35):C.GC_172,(78,35):C.GC_173,(108,93):C.GC_991,(179,93):C.GC_990,(108,104):C.GC_172,(179,104):C.GC_173,(175,70):C.GC_167,(21,7):C.GC_166,(20,79):C.GC_167,(11,114):C.GC_166,(9,80):C.GC_167,(133,56):C.GC_167,(174,110):C.GC_167,(3,115):C.GC_166,(92,13):C.GC_172,(100,13):C.GC_173,(156,149):C.GC_167,(103,9):C.GC_167,(10,6):C.GC_166,(51,13):C.GC_173,(97,13):C.GC_172,(104,112):C.GC_167,(35,104):C.GC_173,(50,13):C.GC_172,(89,13):C.GC_173,(165,104):C.GC_172,(82,2):C.GC_167,(81,145):C.GC_166,(30,104):C.GC_172,(96,104):C.GC_173,(101,98):C.GC_166,(128,141):C.GC_167,(24,119):C.GC_166,(22,83):C.GC_167,(83,21):C.GC_173,(99,21):C.GC_172,(105,113):C.GC_167,(14,118):C.GC_166,(2,144):C.GC_166,(132,52):C.GC_167,(173,111):C.GC_167,(57,21):C.GC_172,(98,21):C.GC_173,(56,21):C.GC_173,(80,21):C.GC_172,(84,16):C.GC_167,(170,105):C.GC_166,(29,8):C.GC_166,(28,107):C.GC_167,(120,54):C.GC_167,(5,96):C.GC_166,(141,4):C.GC_167,(102,11):C.GC_167,(8,3):C.GC_166,(106,121):C.GC_167,(91,5):C.GC_167,(90,148):C.GC_166,(107,124):C.GC_167,(4,142):C.GC_166,(115,143):C.GC_167,(16,101):C.GC_166,(119,50):C.GC_167,(93,18):C.GC_167,(163,17):C.GC_167,(178,109):C.GC_167,(162,51):C.GC_167,(177,122):C.GC_167,(161,55):C.GC_167,(176,125):C.GC_167,(148,19):C.GC_167,(147,53):C.GC_167,(146,57):C.GC_167})

V_162 = Vertex(name = 'V_162',
               particles = [ P.d__tilde__, P.d, P.G0 ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFS2, L.FFS3, L.FFS6 ],
               couplings = {(0,0):C.GC_1124,(0,2):C.GC_1373,(0,1):C.GC_1903})

V_163 = Vertex(name = 'V_163',
               particles = [ P.d__tilde__, P.d, P.G0 ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFS3 ],
               couplings = {(0,0):C.GC_1905})

V_164 = Vertex(name = 'V_164',
               particles = [ P.s__tilde__, P.s, P.G0 ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFS2, L.FFS3, L.FFS6 ],
               couplings = {(0,0):C.GC_1124,(0,2):C.GC_1373,(0,1):C.GC_2099})

V_165 = Vertex(name = 'V_165',
               particles = [ P.s__tilde__, P.s, P.G0 ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFS3 ],
               couplings = {(0,0):C.GC_2101})

V_166 = Vertex(name = 'V_166',
               particles = [ P.b__tilde__, P.b, P.G0 ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFS2, L.FFS3, L.FFS6 ],
               couplings = {(0,0):C.GC_1124,(0,2):C.GC_1373,(0,1):C.GC_1811})

V_167 = Vertex(name = 'V_167',
               particles = [ P.b__tilde__, P.b, P.G0 ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFS3 ],
               couplings = {(0,0):C.GC_1813})

V_168 = Vertex(name = 'V_168',
               particles = [ P.d__tilde__, P.d, P.H ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFS5 ],
               couplings = {(0,0):C.GC_1904})

V_169 = Vertex(name = 'V_169',
               particles = [ P.d__tilde__, P.d, P.H ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFS5 ],
               couplings = {(0,0):C.GC_1941})

V_170 = Vertex(name = 'V_170',
               particles = [ P.s__tilde__, P.s, P.H ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFS5 ],
               couplings = {(0,0):C.GC_2100})

V_171 = Vertex(name = 'V_171',
               particles = [ P.s__tilde__, P.s, P.H ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFS5 ],
               couplings = {(0,0):C.GC_2139})

V_172 = Vertex(name = 'V_172',
               particles = [ P.b__tilde__, P.b, P.H ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFS1, L.FFS4 ],
               couplings = {(0,0):C.GC_1812,(0,1):C.GC_1812})

V_173 = Vertex(name = 'V_173',
               particles = [ P.b__tilde__, P.b, P.H ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFS1, L.FFS4 ],
               couplings = {(0,0):C.GC_1847,(0,1):C.GC_1847})

V_174 = Vertex(name = 'V_174',
               particles = [ P.d__tilde__, P.d, P.G0, P.G0, P.G0 ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFSSS1, L.FFSSS3 ],
               couplings = {(0,0):C.GC_1924,(0,1):C.GC_1929})

V_175 = Vertex(name = 'V_175',
               particles = [ P.s__tilde__, P.s, P.G0, P.G0, P.G0 ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFSSS1, L.FFSSS3 ],
               couplings = {(0,0):C.GC_2124,(0,1):C.GC_2129})

V_176 = Vertex(name = 'V_176',
               particles = [ P.b__tilde__, P.b, P.G0, P.G0, P.G0 ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFSSS1, L.FFSSS3 ],
               couplings = {(0,0):C.GC_1832,(0,1):C.GC_1837})

V_177 = Vertex(name = 'V_177',
               particles = [ P.d__tilde__, P.d, P.G0, P.G__minus__, P.G__plus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFSSS1, L.FFSSS3 ],
               couplings = {(0,0):C.GC_1925,(0,1):C.GC_1928})

V_178 = Vertex(name = 'V_178',
               particles = [ P.s__tilde__, P.s, P.G0, P.G__minus__, P.G__plus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFSSS1, L.FFSSS3 ],
               couplings = {(0,0):C.GC_2125,(0,1):C.GC_2128})

V_179 = Vertex(name = 'V_179',
               particles = [ P.b__tilde__, P.b, P.G0, P.G__minus__, P.G__plus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFSSS1, L.FFSSS3 ],
               couplings = {(0,0):C.GC_1833,(0,1):C.GC_1836})

V_180 = Vertex(name = 'V_180',
               particles = [ P.d__tilde__, P.d, P.G0, P.G0, P.H ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFSSS1, L.FFSSS3 ],
               couplings = {(0,0):C.GC_1926,(0,1):C.GC_1926})

V_181 = Vertex(name = 'V_181',
               particles = [ P.s__tilde__, P.s, P.G0, P.G0, P.H ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFSSS1, L.FFSSS3 ],
               couplings = {(0,0):C.GC_2126,(0,1):C.GC_2126})

V_182 = Vertex(name = 'V_182',
               particles = [ P.b__tilde__, P.b, P.G0, P.G0, P.H ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFSSS1, L.FFSSS3 ],
               couplings = {(0,0):C.GC_1834,(0,1):C.GC_1834})

V_183 = Vertex(name = 'V_183',
               particles = [ P.d__tilde__, P.d, P.G__minus__, P.G__plus__, P.H ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFSSS1, L.FFSSS3 ],
               couplings = {(0,0):C.GC_1926,(0,1):C.GC_1926})

V_184 = Vertex(name = 'V_184',
               particles = [ P.s__tilde__, P.s, P.G__minus__, P.G__plus__, P.H ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFSSS1, L.FFSSS3 ],
               couplings = {(0,0):C.GC_2126,(0,1):C.GC_2126})

V_185 = Vertex(name = 'V_185',
               particles = [ P.b__tilde__, P.b, P.G__minus__, P.G__plus__, P.H ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFSSS1, L.FFSSS3 ],
               couplings = {(0,0):C.GC_1834,(0,1):C.GC_1834})

V_186 = Vertex(name = 'V_186',
               particles = [ P.d__tilde__, P.d, P.G0, P.H, P.H ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFSSS1, L.FFSSS3 ],
               couplings = {(0,0):C.GC_1925,(0,1):C.GC_1928})

V_187 = Vertex(name = 'V_187',
               particles = [ P.s__tilde__, P.s, P.G0, P.H, P.H ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFSSS1, L.FFSSS3 ],
               couplings = {(0,0):C.GC_2125,(0,1):C.GC_2128})

V_188 = Vertex(name = 'V_188',
               particles = [ P.b__tilde__, P.b, P.G0, P.H, P.H ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFSSS1, L.FFSSS3 ],
               couplings = {(0,0):C.GC_1833,(0,1):C.GC_1836})

V_189 = Vertex(name = 'V_189',
               particles = [ P.d__tilde__, P.d, P.H, P.H, P.H ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFSSS1, L.FFSSS3 ],
               couplings = {(0,0):C.GC_1927,(0,1):C.GC_1927})

V_190 = Vertex(name = 'V_190',
               particles = [ P.s__tilde__, P.s, P.H, P.H, P.H ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFSSS1, L.FFSSS3 ],
               couplings = {(0,0):C.GC_2127,(0,1):C.GC_2127})

V_191 = Vertex(name = 'V_191',
               particles = [ P.b__tilde__, P.b, P.H, P.H, P.H ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFSSS1, L.FFSSS3 ],
               couplings = {(0,0):C.GC_1835,(0,1):C.GC_1835})

V_192 = Vertex(name = 'V_192',
               particles = [ P.d__tilde__, P.d, P.G0, P.G0 ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFSS1, L.FFSS6 ],
               couplings = {(0,0):C.GC_1931,(0,1):C.GC_1931})

V_193 = Vertex(name = 'V_193',
               particles = [ P.s__tilde__, P.s, P.G0, P.G0 ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFSS1, L.FFSS6 ],
               couplings = {(0,0):C.GC_2131,(0,1):C.GC_2131})

V_194 = Vertex(name = 'V_194',
               particles = [ P.b__tilde__, P.b, P.G0, P.G0 ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFSS1, L.FFSS6 ],
               couplings = {(0,0):C.GC_1839,(0,1):C.GC_1839})

V_195 = Vertex(name = 'V_195',
               particles = [ P.d__tilde__, P.d, P.G__minus__, P.G__plus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFSS1, L.FFSS2, L.FFSS3, L.FFSS6, L.FFSS8, L.FFSS9 ],
               couplings = {(0,1):C.GC_1077,(0,4):C.GC_1161,(0,2):C.GC_1076,(0,5):C.GC_1162,(0,0):C.GC_1931,(0,3):C.GC_1931})

V_196 = Vertex(name = 'V_196',
               particles = [ P.s__tilde__, P.s, P.G__minus__, P.G__plus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFSS1, L.FFSS2, L.FFSS3, L.FFSS6, L.FFSS8, L.FFSS9 ],
               couplings = {(0,1):C.GC_1077,(0,4):C.GC_1161,(0,2):C.GC_1076,(0,5):C.GC_1162,(0,0):C.GC_2131,(0,3):C.GC_2131})

V_197 = Vertex(name = 'V_197',
               particles = [ P.b__tilde__, P.b, P.G__minus__, P.G__plus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFSS1, L.FFSS2, L.FFSS3, L.FFSS6, L.FFSS8, L.FFSS9 ],
               couplings = {(0,1):C.GC_1077,(0,4):C.GC_1161,(0,2):C.GC_1076,(0,5):C.GC_1162,(0,0):C.GC_1839,(0,3):C.GC_1839})

V_198 = Vertex(name = 'V_198',
               particles = [ P.d__tilde__, P.d, P.G0, P.H ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFSS1, L.FFSS2, L.FFSS3, L.FFSS6, L.FFSS8, L.FFSS9 ],
               couplings = {(0,1):C.GC_1078,(0,4):C.GC_1163,(0,2):C.GC_1073,(0,5):C.GC_1160,(0,0):C.GC_1930,(0,3):C.GC_1933})

V_199 = Vertex(name = 'V_199',
               particles = [ P.s__tilde__, P.s, P.G0, P.H ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFSS1, L.FFSS2, L.FFSS3, L.FFSS6, L.FFSS8, L.FFSS9 ],
               couplings = {(0,1):C.GC_1078,(0,4):C.GC_1163,(0,2):C.GC_1073,(0,5):C.GC_1160,(0,0):C.GC_2130,(0,3):C.GC_2133})

V_200 = Vertex(name = 'V_200',
               particles = [ P.b__tilde__, P.b, P.G0, P.H ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFSS1, L.FFSS2, L.FFSS3, L.FFSS6, L.FFSS8, L.FFSS9 ],
               couplings = {(0,1):C.GC_1078,(0,4):C.GC_1163,(0,2):C.GC_1073,(0,5):C.GC_1160,(0,0):C.GC_1838,(0,3):C.GC_1841})

V_201 = Vertex(name = 'V_201',
               particles = [ P.d__tilde__, P.d, P.H, P.H ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFSS1, L.FFSS6 ],
               couplings = {(0,0):C.GC_1932,(0,1):C.GC_1932})

V_202 = Vertex(name = 'V_202',
               particles = [ P.s__tilde__, P.s, P.H, P.H ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFSS1, L.FFSS6 ],
               couplings = {(0,0):C.GC_2132,(0,1):C.GC_2132})

V_203 = Vertex(name = 'V_203',
               particles = [ P.b__tilde__, P.b, P.H, P.H ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFSS1, L.FFSS6 ],
               couplings = {(0,0):C.GC_1840,(0,1):C.GC_1840})

V_204 = Vertex(name = 'V_204',
               particles = [ P.u__tilde__, P.d, P.G__plus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFS1, L.FFS2, L.FFS4, L.FFS6 ],
               couplings = {(0,0):C.GC_17,(0,2):C.GC_35,(0,1):C.GC_2320,(0,3):C.GC_1376})

V_205 = Vertex(name = 'V_205',
               particles = [ P.u__tilde__, P.d, P.G__plus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFS1, L.FFS4 ],
               couplings = {(0,0):C.GC_18,(0,1):C.GC_36})

V_206 = Vertex(name = 'V_206',
               particles = [ P.c__tilde__, P.d, P.G__plus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFS1, L.FFS2, L.FFS4 ],
               couplings = {(0,0):C.GC_19,(0,2):C.GC_37,(0,1):C.GC_2362})

V_207 = Vertex(name = 'V_207',
               particles = [ P.c__tilde__, P.d, P.G__plus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFS1, L.FFS4 ],
               couplings = {(0,0):C.GC_20,(0,1):C.GC_38})

V_208 = Vertex(name = 'V_208',
               particles = [ P.t__tilde__, P.d, P.G__plus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFS1, L.FFS2, L.FFS4 ],
               couplings = {(0,0):C.GC_21,(0,2):C.GC_39,(0,1):C.GC_2404})

V_209 = Vertex(name = 'V_209',
               particles = [ P.t__tilde__, P.d, P.G__plus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFS1, L.FFS4 ],
               couplings = {(0,0):C.GC_22,(0,1):C.GC_40})

V_210 = Vertex(name = 'V_210',
               particles = [ P.u__tilde__, P.s, P.G__plus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFS1, L.FFS2, L.FFS4 ],
               couplings = {(0,0):C.GC_23,(0,2):C.GC_41,(0,1):C.GC_2334})

V_211 = Vertex(name = 'V_211',
               particles = [ P.u__tilde__, P.s, P.G__plus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFS1, L.FFS4 ],
               couplings = {(0,0):C.GC_24,(0,1):C.GC_42})

V_212 = Vertex(name = 'V_212',
               particles = [ P.c__tilde__, P.s, P.G__plus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFS1, L.FFS2, L.FFS4, L.FFS6 ],
               couplings = {(0,0):C.GC_25,(0,2):C.GC_43,(0,1):C.GC_2376,(0,3):C.GC_1376})

V_213 = Vertex(name = 'V_213',
               particles = [ P.c__tilde__, P.s, P.G__plus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFS1, L.FFS4 ],
               couplings = {(0,0):C.GC_26,(0,1):C.GC_44})

V_214 = Vertex(name = 'V_214',
               particles = [ P.t__tilde__, P.s, P.G__plus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFS1, L.FFS2, L.FFS4 ],
               couplings = {(0,0):C.GC_27,(0,2):C.GC_45,(0,1):C.GC_2418})

V_215 = Vertex(name = 'V_215',
               particles = [ P.t__tilde__, P.s, P.G__plus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFS1, L.FFS4 ],
               couplings = {(0,0):C.GC_28,(0,1):C.GC_46})

V_216 = Vertex(name = 'V_216',
               particles = [ P.u__tilde__, P.b, P.G__plus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFS1, L.FFS2, L.FFS4 ],
               couplings = {(0,0):C.GC_29,(0,2):C.GC_47,(0,1):C.GC_2348})

V_217 = Vertex(name = 'V_217',
               particles = [ P.u__tilde__, P.b, P.G__plus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFS1, L.FFS4 ],
               couplings = {(0,0):C.GC_30,(0,1):C.GC_48})

V_218 = Vertex(name = 'V_218',
               particles = [ P.c__tilde__, P.b, P.G__plus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFS1, L.FFS2, L.FFS4 ],
               couplings = {(0,0):C.GC_31,(0,2):C.GC_49,(0,1):C.GC_2390})

V_219 = Vertex(name = 'V_219',
               particles = [ P.c__tilde__, P.b, P.G__plus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFS1, L.FFS4 ],
               couplings = {(0,0):C.GC_32,(0,1):C.GC_50})

V_220 = Vertex(name = 'V_220',
               particles = [ P.t__tilde__, P.b, P.G__plus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFS1, L.FFS2, L.FFS4, L.FFS6 ],
               couplings = {(0,0):C.GC_33,(0,2):C.GC_51,(0,1):C.GC_2432,(0,3):C.GC_1376})

V_221 = Vertex(name = 'V_221',
               particles = [ P.t__tilde__, P.b, P.G__plus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFS1, L.FFS4 ],
               couplings = {(0,0):C.GC_34,(0,1):C.GC_52})

V_222 = Vertex(name = 'V_222',
               particles = [ P.u__tilde__, P.d, P.G0, P.G0, P.G__plus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFSSS1, L.FFSSS3 ],
               couplings = {(0,0):C.GC_1228,(0,1):C.GC_1247})

V_223 = Vertex(name = 'V_223',
               particles = [ P.c__tilde__, P.d, P.G0, P.G0, P.G__plus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFSSS1, L.FFSSS3 ],
               couplings = {(0,0):C.GC_1230,(0,1):C.GC_1249})

V_224 = Vertex(name = 'V_224',
               particles = [ P.t__tilde__, P.d, P.G0, P.G0, P.G__plus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFSSS1, L.FFSSS3 ],
               couplings = {(0,0):C.GC_1232,(0,1):C.GC_1251})

V_225 = Vertex(name = 'V_225',
               particles = [ P.u__tilde__, P.s, P.G0, P.G0, P.G__plus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFSSS1, L.FFSSS3 ],
               couplings = {(0,0):C.GC_1234,(0,1):C.GC_1253})

V_226 = Vertex(name = 'V_226',
               particles = [ P.c__tilde__, P.s, P.G0, P.G0, P.G__plus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFSSS1, L.FFSSS3 ],
               couplings = {(0,0):C.GC_1236,(0,1):C.GC_1255})

V_227 = Vertex(name = 'V_227',
               particles = [ P.t__tilde__, P.s, P.G0, P.G0, P.G__plus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFSSS1, L.FFSSS3 ],
               couplings = {(0,0):C.GC_1238,(0,1):C.GC_1257})

V_228 = Vertex(name = 'V_228',
               particles = [ P.u__tilde__, P.b, P.G0, P.G0, P.G__plus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFSSS1, L.FFSSS3 ],
               couplings = {(0,0):C.GC_1240,(0,1):C.GC_1259})

V_229 = Vertex(name = 'V_229',
               particles = [ P.c__tilde__, P.b, P.G0, P.G0, P.G__plus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFSSS1, L.FFSSS3 ],
               couplings = {(0,0):C.GC_1242,(0,1):C.GC_1261})

V_230 = Vertex(name = 'V_230',
               particles = [ P.t__tilde__, P.b, P.G0, P.G0, P.G__plus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFSSS1, L.FFSSS3 ],
               couplings = {(0,0):C.GC_1244,(0,1):C.GC_1263})

V_231 = Vertex(name = 'V_231',
               particles = [ P.u__tilde__, P.d, P.G__minus__, P.G__plus__, P.G__plus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFSSS1, L.FFSSS3 ],
               couplings = {(0,0):C.GC_1229,(0,1):C.GC_1246})

V_232 = Vertex(name = 'V_232',
               particles = [ P.c__tilde__, P.d, P.G__minus__, P.G__plus__, P.G__plus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFSSS1, L.FFSSS3 ],
               couplings = {(0,0):C.GC_1231,(0,1):C.GC_1248})

V_233 = Vertex(name = 'V_233',
               particles = [ P.t__tilde__, P.d, P.G__minus__, P.G__plus__, P.G__plus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFSSS1, L.FFSSS3 ],
               couplings = {(0,0):C.GC_1233,(0,1):C.GC_1250})

V_234 = Vertex(name = 'V_234',
               particles = [ P.u__tilde__, P.s, P.G__minus__, P.G__plus__, P.G__plus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFSSS1, L.FFSSS3 ],
               couplings = {(0,0):C.GC_1235,(0,1):C.GC_1252})

V_235 = Vertex(name = 'V_235',
               particles = [ P.c__tilde__, P.s, P.G__minus__, P.G__plus__, P.G__plus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFSSS1, L.FFSSS3 ],
               couplings = {(0,0):C.GC_1237,(0,1):C.GC_1254})

V_236 = Vertex(name = 'V_236',
               particles = [ P.t__tilde__, P.s, P.G__minus__, P.G__plus__, P.G__plus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFSSS1, L.FFSSS3 ],
               couplings = {(0,0):C.GC_1239,(0,1):C.GC_1256})

V_237 = Vertex(name = 'V_237',
               particles = [ P.u__tilde__, P.b, P.G__minus__, P.G__plus__, P.G__plus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFSSS1, L.FFSSS3 ],
               couplings = {(0,0):C.GC_1241,(0,1):C.GC_1258})

V_238 = Vertex(name = 'V_238',
               particles = [ P.c__tilde__, P.b, P.G__minus__, P.G__plus__, P.G__plus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFSSS1, L.FFSSS3 ],
               couplings = {(0,0):C.GC_1243,(0,1):C.GC_1260})

V_239 = Vertex(name = 'V_239',
               particles = [ P.t__tilde__, P.b, P.G__minus__, P.G__plus__, P.G__plus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFSSS1, L.FFSSS3 ],
               couplings = {(0,0):C.GC_1245,(0,1):C.GC_1262})

V_240 = Vertex(name = 'V_240',
               particles = [ P.u__tilde__, P.d, P.G__plus__, P.H, P.H ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFSSS1, L.FFSSS3 ],
               couplings = {(0,0):C.GC_1228,(0,1):C.GC_1247})

V_241 = Vertex(name = 'V_241',
               particles = [ P.c__tilde__, P.d, P.G__plus__, P.H, P.H ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFSSS1, L.FFSSS3 ],
               couplings = {(0,0):C.GC_1230,(0,1):C.GC_1249})

V_242 = Vertex(name = 'V_242',
               particles = [ P.t__tilde__, P.d, P.G__plus__, P.H, P.H ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFSSS1, L.FFSSS3 ],
               couplings = {(0,0):C.GC_1232,(0,1):C.GC_1251})

V_243 = Vertex(name = 'V_243',
               particles = [ P.u__tilde__, P.s, P.G__plus__, P.H, P.H ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFSSS1, L.FFSSS3 ],
               couplings = {(0,0):C.GC_1234,(0,1):C.GC_1253})

V_244 = Vertex(name = 'V_244',
               particles = [ P.c__tilde__, P.s, P.G__plus__, P.H, P.H ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFSSS1, L.FFSSS3 ],
               couplings = {(0,0):C.GC_1236,(0,1):C.GC_1255})

V_245 = Vertex(name = 'V_245',
               particles = [ P.t__tilde__, P.s, P.G__plus__, P.H, P.H ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFSSS1, L.FFSSS3 ],
               couplings = {(0,0):C.GC_1238,(0,1):C.GC_1257})

V_246 = Vertex(name = 'V_246',
               particles = [ P.u__tilde__, P.b, P.G__plus__, P.H, P.H ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFSSS1, L.FFSSS3 ],
               couplings = {(0,0):C.GC_1240,(0,1):C.GC_1259})

V_247 = Vertex(name = 'V_247',
               particles = [ P.c__tilde__, P.b, P.G__plus__, P.H, P.H ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFSSS1, L.FFSSS3 ],
               couplings = {(0,0):C.GC_1242,(0,1):C.GC_1261})

V_248 = Vertex(name = 'V_248',
               particles = [ P.t__tilde__, P.b, P.G__plus__, P.H, P.H ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFSSS1, L.FFSSS3 ],
               couplings = {(0,0):C.GC_1244,(0,1):C.GC_1263})

V_249 = Vertex(name = 'V_249',
               particles = [ P.u__tilde__, P.d, P.G__plus__, P.H ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFSS1, L.FFSS2, L.FFSS3, L.FFSS6, L.FFSS8, L.FFSS9 ],
               couplings = {(0,0):C.GC_1403,(0,3):C.GC_1412,(0,1):C.GC_2312,(0,4):C.GC_1170,(0,2):C.GC_2314,(0,5):C.GC_1173})

V_250 = Vertex(name = 'V_250',
               particles = [ P.c__tilde__, P.d, P.G__plus__, P.H ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFSS1, L.FFSS2, L.FFSS3, L.FFSS6 ],
               couplings = {(0,0):C.GC_1404,(0,3):C.GC_1413,(0,1):C.GC_2354,(0,2):C.GC_2356})

V_251 = Vertex(name = 'V_251',
               particles = [ P.t__tilde__, P.d, P.G__plus__, P.H ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFSS1, L.FFSS2, L.FFSS3, L.FFSS6 ],
               couplings = {(0,0):C.GC_1405,(0,3):C.GC_1414,(0,1):C.GC_2396,(0,2):C.GC_2398})

V_252 = Vertex(name = 'V_252',
               particles = [ P.u__tilde__, P.s, P.G__plus__, P.H ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFSS1, L.FFSS2, L.FFSS3, L.FFSS6 ],
               couplings = {(0,0):C.GC_1406,(0,3):C.GC_1415,(0,1):C.GC_2326,(0,2):C.GC_2328})

V_253 = Vertex(name = 'V_253',
               particles = [ P.c__tilde__, P.s, P.G__plus__, P.H ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFSS1, L.FFSS2, L.FFSS3, L.FFSS6, L.FFSS8, L.FFSS9 ],
               couplings = {(0,0):C.GC_1407,(0,3):C.GC_1416,(0,1):C.GC_2368,(0,4):C.GC_1170,(0,2):C.GC_2370,(0,5):C.GC_1173})

V_254 = Vertex(name = 'V_254',
               particles = [ P.t__tilde__, P.s, P.G__plus__, P.H ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFSS1, L.FFSS2, L.FFSS3, L.FFSS6 ],
               couplings = {(0,0):C.GC_1408,(0,3):C.GC_1417,(0,1):C.GC_2410,(0,2):C.GC_2412})

V_255 = Vertex(name = 'V_255',
               particles = [ P.u__tilde__, P.b, P.G__plus__, P.H ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFSS1, L.FFSS2, L.FFSS3, L.FFSS6 ],
               couplings = {(0,0):C.GC_1409,(0,3):C.GC_1418,(0,1):C.GC_2340,(0,2):C.GC_2342})

V_256 = Vertex(name = 'V_256',
               particles = [ P.c__tilde__, P.b, P.G__plus__, P.H ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFSS1, L.FFSS2, L.FFSS3, L.FFSS6 ],
               couplings = {(0,0):C.GC_1410,(0,3):C.GC_1419,(0,1):C.GC_2382,(0,2):C.GC_2384})

V_257 = Vertex(name = 'V_257',
               particles = [ P.t__tilde__, P.b, P.G__plus__, P.H ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFSS1, L.FFSS2, L.FFSS3, L.FFSS6, L.FFSS8, L.FFSS9 ],
               couplings = {(0,0):C.GC_1411,(0,3):C.GC_1420,(0,1):C.GC_2424,(0,4):C.GC_1170,(0,2):C.GC_2426,(0,5):C.GC_1173})

V_258 = Vertex(name = 'V_258',
               particles = [ P.e__plus__, P.e__minus__, P.G0 ],
               color = [ '1' ],
               lorentz = [ L.FFS1, L.FFS2, L.FFS4, L.FFS6 ],
               couplings = {(0,1):C.GC_1122,(0,3):C.GC_1374,(0,0):C.GC_1954,(0,2):C.GC_1956})

V_259 = Vertex(name = 'V_259',
               particles = [ P.e__plus__, P.e__minus__, P.G0 ],
               color = [ '1' ],
               lorentz = [ L.FFS1, L.FFS4 ],
               couplings = {(0,0):C.GC_1959,(0,1):C.GC_1960})

V_260 = Vertex(name = 'V_260',
               particles = [ P.mu__plus__, P.mu__minus__, P.G0 ],
               color = [ '1' ],
               lorentz = [ L.FFS1, L.FFS2, L.FFS4, L.FFS6 ],
               couplings = {(0,1):C.GC_1122,(0,3):C.GC_1374,(0,0):C.GC_2026,(0,2):C.GC_2028})

V_261 = Vertex(name = 'V_261',
               particles = [ P.mu__plus__, P.mu__minus__, P.G0 ],
               color = [ '1' ],
               lorentz = [ L.FFS1, L.FFS4 ],
               couplings = {(0,0):C.GC_2031,(0,1):C.GC_2032})

V_262 = Vertex(name = 'V_262',
               particles = [ P.ta__plus__, P.ta__minus__, P.G0 ],
               color = [ '1' ],
               lorentz = [ L.FFS1, L.FFS2, L.FFS4, L.FFS6 ],
               couplings = {(0,1):C.GC_1122,(0,3):C.GC_1374,(0,0):C.GC_2197,(0,2):C.GC_2199})

V_263 = Vertex(name = 'V_263',
               particles = [ P.ta__plus__, P.ta__minus__, P.G0 ],
               color = [ '1' ],
               lorentz = [ L.FFS1, L.FFS4 ],
               couplings = {(0,0):C.GC_2202,(0,1):C.GC_2203})

V_264 = Vertex(name = 'V_264',
               particles = [ P.e__plus__, P.e__minus__, P.H ],
               color = [ '1' ],
               lorentz = [ L.FFS1, L.FFS4 ],
               couplings = {(0,0):C.GC_1955,(0,1):C.GC_1955})

V_265 = Vertex(name = 'V_265',
               particles = [ P.e__plus__, P.e__minus__, P.H ],
               color = [ '1' ],
               lorentz = [ L.FFS1, L.FFS4 ],
               couplings = {(0,0):C.GC_2008,(0,1):C.GC_2008})

V_266 = Vertex(name = 'V_266',
               particles = [ P.mu__plus__, P.mu__minus__, P.H ],
               color = [ '1' ],
               lorentz = [ L.FFS1, L.FFS4 ],
               couplings = {(0,0):C.GC_2027,(0,1):C.GC_2027})

V_267 = Vertex(name = 'V_267',
               particles = [ P.mu__plus__, P.mu__minus__, P.H ],
               color = [ '1' ],
               lorentz = [ L.FFS1, L.FFS4 ],
               couplings = {(0,0):C.GC_2084,(0,1):C.GC_2084})

V_268 = Vertex(name = 'V_268',
               particles = [ P.ta__plus__, P.ta__minus__, P.H ],
               color = [ '1' ],
               lorentz = [ L.FFS1, L.FFS4 ],
               couplings = {(0,0):C.GC_2198,(0,1):C.GC_2198})

V_269 = Vertex(name = 'V_269',
               particles = [ P.ta__plus__, P.ta__minus__, P.H ],
               color = [ '1' ],
               lorentz = [ L.FFS1, L.FFS4 ],
               couplings = {(0,0):C.GC_2251,(0,1):C.GC_2251})

V_270 = Vertex(name = 'V_270',
               particles = [ P.e__plus__, P.e__minus__, P.G0, P.G0, P.G0 ],
               color = [ '1' ],
               lorentz = [ L.FFSSS1, L.FFSSS3 ],
               couplings = {(0,0):C.GC_1988,(0,1):C.GC_1993})

V_271 = Vertex(name = 'V_271',
               particles = [ P.mu__plus__, P.mu__minus__, P.G0, P.G0, P.G0 ],
               color = [ '1' ],
               lorentz = [ L.FFSSS1, L.FFSSS3 ],
               couplings = {(0,0):C.GC_2062,(0,1):C.GC_2067})

V_272 = Vertex(name = 'V_272',
               particles = [ P.ta__plus__, P.ta__minus__, P.G0, P.G0, P.G0 ],
               color = [ '1' ],
               lorentz = [ L.FFSSS1, L.FFSSS3 ],
               couplings = {(0,0):C.GC_2231,(0,1):C.GC_2236})

V_273 = Vertex(name = 'V_273',
               particles = [ P.e__plus__, P.e__minus__, P.G0, P.G__minus__, P.G__plus__ ],
               color = [ '1' ],
               lorentz = [ L.FFSSS1, L.FFSSS3 ],
               couplings = {(0,0):C.GC_1989,(0,1):C.GC_1992})

V_274 = Vertex(name = 'V_274',
               particles = [ P.mu__plus__, P.mu__minus__, P.G0, P.G__minus__, P.G__plus__ ],
               color = [ '1' ],
               lorentz = [ L.FFSSS1, L.FFSSS3 ],
               couplings = {(0,0):C.GC_2063,(0,1):C.GC_2066})

V_275 = Vertex(name = 'V_275',
               particles = [ P.ta__plus__, P.ta__minus__, P.G0, P.G__minus__, P.G__plus__ ],
               color = [ '1' ],
               lorentz = [ L.FFSSS1, L.FFSSS3 ],
               couplings = {(0,0):C.GC_2232,(0,1):C.GC_2235})

V_276 = Vertex(name = 'V_276',
               particles = [ P.e__plus__, P.e__minus__, P.G0, P.G0, P.H ],
               color = [ '1' ],
               lorentz = [ L.FFSSS1, L.FFSSS3 ],
               couplings = {(0,0):C.GC_1990,(0,1):C.GC_1990})

V_277 = Vertex(name = 'V_277',
               particles = [ P.mu__plus__, P.mu__minus__, P.G0, P.G0, P.H ],
               color = [ '1' ],
               lorentz = [ L.FFSSS1, L.FFSSS3 ],
               couplings = {(0,0):C.GC_2064,(0,1):C.GC_2064})

V_278 = Vertex(name = 'V_278',
               particles = [ P.ta__plus__, P.ta__minus__, P.G0, P.G0, P.H ],
               color = [ '1' ],
               lorentz = [ L.FFSSS1, L.FFSSS3 ],
               couplings = {(0,0):C.GC_2233,(0,1):C.GC_2233})

V_279 = Vertex(name = 'V_279',
               particles = [ P.e__plus__, P.e__minus__, P.G__minus__, P.G__plus__, P.H ],
               color = [ '1' ],
               lorentz = [ L.FFSSS1, L.FFSSS3 ],
               couplings = {(0,0):C.GC_1990,(0,1):C.GC_1990})

V_280 = Vertex(name = 'V_280',
               particles = [ P.mu__plus__, P.mu__minus__, P.G__minus__, P.G__plus__, P.H ],
               color = [ '1' ],
               lorentz = [ L.FFSSS1, L.FFSSS3 ],
               couplings = {(0,0):C.GC_2064,(0,1):C.GC_2064})

V_281 = Vertex(name = 'V_281',
               particles = [ P.ta__plus__, P.ta__minus__, P.G__minus__, P.G__plus__, P.H ],
               color = [ '1' ],
               lorentz = [ L.FFSSS1, L.FFSSS3 ],
               couplings = {(0,0):C.GC_2233,(0,1):C.GC_2233})

V_282 = Vertex(name = 'V_282',
               particles = [ P.e__plus__, P.e__minus__, P.G0, P.H, P.H ],
               color = [ '1' ],
               lorentz = [ L.FFSSS1, L.FFSSS3 ],
               couplings = {(0,0):C.GC_1989,(0,1):C.GC_1992})

V_283 = Vertex(name = 'V_283',
               particles = [ P.mu__plus__, P.mu__minus__, P.G0, P.H, P.H ],
               color = [ '1' ],
               lorentz = [ L.FFSSS1, L.FFSSS3 ],
               couplings = {(0,0):C.GC_2063,(0,1):C.GC_2066})

V_284 = Vertex(name = 'V_284',
               particles = [ P.ta__plus__, P.ta__minus__, P.G0, P.H, P.H ],
               color = [ '1' ],
               lorentz = [ L.FFSSS1, L.FFSSS3 ],
               couplings = {(0,0):C.GC_2232,(0,1):C.GC_2235})

V_285 = Vertex(name = 'V_285',
               particles = [ P.e__plus__, P.e__minus__, P.H, P.H, P.H ],
               color = [ '1' ],
               lorentz = [ L.FFSSS1, L.FFSSS3 ],
               couplings = {(0,0):C.GC_1991,(0,1):C.GC_1991})

V_286 = Vertex(name = 'V_286',
               particles = [ P.mu__plus__, P.mu__minus__, P.H, P.H, P.H ],
               color = [ '1' ],
               lorentz = [ L.FFSSS1, L.FFSSS3 ],
               couplings = {(0,0):C.GC_2065,(0,1):C.GC_2065})

V_287 = Vertex(name = 'V_287',
               particles = [ P.ta__plus__, P.ta__minus__, P.H, P.H, P.H ],
               color = [ '1' ],
               lorentz = [ L.FFSSS1, L.FFSSS3 ],
               couplings = {(0,0):C.GC_2234,(0,1):C.GC_2234})

V_288 = Vertex(name = 'V_288',
               particles = [ P.e__plus__, P.e__minus__, P.G0, P.G0 ],
               color = [ '1' ],
               lorentz = [ L.FFSS1, L.FFSS6 ],
               couplings = {(0,0):C.GC_1997,(0,1):C.GC_1997})

V_289 = Vertex(name = 'V_289',
               particles = [ P.mu__plus__, P.mu__minus__, P.G0, P.G0 ],
               color = [ '1' ],
               lorentz = [ L.FFSS1, L.FFSS6 ],
               couplings = {(0,0):C.GC_2071,(0,1):C.GC_2071})

V_290 = Vertex(name = 'V_290',
               particles = [ P.ta__plus__, P.ta__minus__, P.G0, P.G0 ],
               color = [ '1' ],
               lorentz = [ L.FFSS1, L.FFSS6 ],
               couplings = {(0,0):C.GC_2240,(0,1):C.GC_2240})

V_291 = Vertex(name = 'V_291',
               particles = [ P.e__plus__, P.e__minus__, P.G__minus__, P.G__plus__ ],
               color = [ '1' ],
               lorentz = [ L.FFSS1, L.FFSS2, L.FFSS3, L.FFSS6, L.FFSS8, L.FFSS9 ],
               couplings = {(0,1):C.GC_1071,(0,4):C.GC_1165,(0,2):C.GC_1070,(0,5):C.GC_1166,(0,0):C.GC_1997,(0,3):C.GC_1997})

V_292 = Vertex(name = 'V_292',
               particles = [ P.mu__plus__, P.mu__minus__, P.G__minus__, P.G__plus__ ],
               color = [ '1' ],
               lorentz = [ L.FFSS1, L.FFSS2, L.FFSS3, L.FFSS6, L.FFSS8, L.FFSS9 ],
               couplings = {(0,1):C.GC_1071,(0,4):C.GC_1165,(0,2):C.GC_1070,(0,5):C.GC_1166,(0,0):C.GC_2071,(0,3):C.GC_2071})

V_293 = Vertex(name = 'V_293',
               particles = [ P.ta__plus__, P.ta__minus__, P.G__minus__, P.G__plus__ ],
               color = [ '1' ],
               lorentz = [ L.FFSS1, L.FFSS2, L.FFSS3, L.FFSS6, L.FFSS8, L.FFSS9 ],
               couplings = {(0,1):C.GC_1071,(0,4):C.GC_1165,(0,2):C.GC_1070,(0,5):C.GC_1166,(0,0):C.GC_2240,(0,3):C.GC_2240})

V_294 = Vertex(name = 'V_294',
               particles = [ P.e__plus__, P.e__minus__, P.G0, P.H ],
               color = [ '1' ],
               lorentz = [ L.FFSS1, L.FFSS2, L.FFSS3, L.FFSS6, L.FFSS8, L.FFSS9 ],
               couplings = {(0,1):C.GC_1072,(0,4):C.GC_1167,(0,2):C.GC_1067,(0,5):C.GC_1164,(0,0):C.GC_1996,(0,3):C.GC_1999})

V_295 = Vertex(name = 'V_295',
               particles = [ P.mu__plus__, P.mu__minus__, P.G0, P.H ],
               color = [ '1' ],
               lorentz = [ L.FFSS1, L.FFSS2, L.FFSS3, L.FFSS6, L.FFSS8, L.FFSS9 ],
               couplings = {(0,1):C.GC_1072,(0,4):C.GC_1167,(0,2):C.GC_1067,(0,5):C.GC_1164,(0,0):C.GC_2070,(0,3):C.GC_2073})

V_296 = Vertex(name = 'V_296',
               particles = [ P.ta__plus__, P.ta__minus__, P.G0, P.H ],
               color = [ '1' ],
               lorentz = [ L.FFSS1, L.FFSS2, L.FFSS3, L.FFSS6, L.FFSS8, L.FFSS9 ],
               couplings = {(0,1):C.GC_1072,(0,4):C.GC_1167,(0,2):C.GC_1067,(0,5):C.GC_1164,(0,0):C.GC_2239,(0,3):C.GC_2242})

V_297 = Vertex(name = 'V_297',
               particles = [ P.e__plus__, P.e__minus__, P.H, P.H ],
               color = [ '1' ],
               lorentz = [ L.FFSS1, L.FFSS6 ],
               couplings = {(0,0):C.GC_1998,(0,1):C.GC_1998})

V_298 = Vertex(name = 'V_298',
               particles = [ P.mu__plus__, P.mu__minus__, P.H, P.H ],
               color = [ '1' ],
               lorentz = [ L.FFSS1, L.FFSS6 ],
               couplings = {(0,0):C.GC_2072,(0,1):C.GC_2072})

V_299 = Vertex(name = 'V_299',
               particles = [ P.ta__plus__, P.ta__minus__, P.H, P.H ],
               color = [ '1' ],
               lorentz = [ L.FFSS7 ],
               couplings = {(0,0):C.GC_2241})

V_300 = Vertex(name = 'V_300',
               particles = [ P.ve__tilde__, P.e__minus__, P.G__plus__ ],
               color = [ '1' ],
               lorentz = [ L.FFS1, L.FFS2 ],
               couplings = {(0,0):C.GC_1953,(0,1):C.GC_1377})

V_301 = Vertex(name = 'V_301',
               particles = [ P.ve__tilde__, P.e__minus__, P.G__plus__ ],
               color = [ '1' ],
               lorentz = [ L.FFS1 ],
               couplings = {(0,0):C.GC_1958})

V_302 = Vertex(name = 'V_302',
               particles = [ P.vm__tilde__, P.mu__minus__, P.G__plus__ ],
               color = [ '1' ],
               lorentz = [ L.FFS1, L.FFS2 ],
               couplings = {(0,0):C.GC_2025,(0,1):C.GC_1377})

V_303 = Vertex(name = 'V_303',
               particles = [ P.vm__tilde__, P.mu__minus__, P.G__plus__ ],
               color = [ '1' ],
               lorentz = [ L.FFS1 ],
               couplings = {(0,0):C.GC_2030})

V_304 = Vertex(name = 'V_304',
               particles = [ P.vt__tilde__, P.ta__minus__, P.G__plus__ ],
               color = [ '1' ],
               lorentz = [ L.FFS1, L.FFS2 ],
               couplings = {(0,0):C.GC_2196,(0,1):C.GC_1377})

V_305 = Vertex(name = 'V_305',
               particles = [ P.vt__tilde__, P.ta__minus__, P.G__plus__ ],
               color = [ '1' ],
               lorentz = [ L.FFS1 ],
               couplings = {(0,0):C.GC_2201})

V_306 = Vertex(name = 'V_306',
               particles = [ P.ve__tilde__, P.e__minus__, P.G0, P.G0, P.G__plus__ ],
               color = [ '1' ],
               lorentz = [ L.FFSSS1 ],
               couplings = {(0,0):C.GC_1986})

V_307 = Vertex(name = 'V_307',
               particles = [ P.vm__tilde__, P.mu__minus__, P.G0, P.G0, P.G__plus__ ],
               color = [ '1' ],
               lorentz = [ L.FFSSS1 ],
               couplings = {(0,0):C.GC_2060})

V_308 = Vertex(name = 'V_308',
               particles = [ P.vt__tilde__, P.ta__minus__, P.G0, P.G0, P.G__plus__ ],
               color = [ '1' ],
               lorentz = [ L.FFSSS1 ],
               couplings = {(0,0):C.GC_2229})

V_309 = Vertex(name = 'V_309',
               particles = [ P.ve__tilde__, P.e__minus__, P.G__minus__, P.G__plus__, P.G__plus__ ],
               color = [ '1' ],
               lorentz = [ L.FFSSS1 ],
               couplings = {(0,0):C.GC_1987})

V_310 = Vertex(name = 'V_310',
               particles = [ P.vm__tilde__, P.mu__minus__, P.G__minus__, P.G__plus__, P.G__plus__ ],
               color = [ '1' ],
               lorentz = [ L.FFSSS1 ],
               couplings = {(0,0):C.GC_2061})

V_311 = Vertex(name = 'V_311',
               particles = [ P.vt__tilde__, P.ta__minus__, P.G__minus__, P.G__plus__, P.G__plus__ ],
               color = [ '1' ],
               lorentz = [ L.FFSSS1 ],
               couplings = {(0,0):C.GC_2230})

V_312 = Vertex(name = 'V_312',
               particles = [ P.ve__tilde__, P.e__minus__, P.G__plus__, P.H, P.H ],
               color = [ '1' ],
               lorentz = [ L.FFSSS1 ],
               couplings = {(0,0):C.GC_1986})

V_313 = Vertex(name = 'V_313',
               particles = [ P.vm__tilde__, P.mu__minus__, P.G__plus__, P.H, P.H ],
               color = [ '1' ],
               lorentz = [ L.FFSSS1 ],
               couplings = {(0,0):C.GC_2060})

V_314 = Vertex(name = 'V_314',
               particles = [ P.vt__tilde__, P.ta__minus__, P.G__plus__, P.H, P.H ],
               color = [ '1' ],
               lorentz = [ L.FFSSS1 ],
               couplings = {(0,0):C.GC_2229})

V_315 = Vertex(name = 'V_315',
               particles = [ P.ve__tilde__, P.e__minus__, P.G__plus__, P.H ],
               color = [ '1' ],
               lorentz = [ L.FFSS1, L.FFSS4 ],
               couplings = {(0,0):C.GC_1995,(0,1):C.GC_1174})

V_316 = Vertex(name = 'V_316',
               particles = [ P.vm__tilde__, P.mu__minus__, P.G__plus__, P.H ],
               color = [ '1' ],
               lorentz = [ L.FFSS1, L.FFSS4 ],
               couplings = {(0,0):C.GC_2069,(0,1):C.GC_1174})

V_317 = Vertex(name = 'V_317',
               particles = [ P.vt__tilde__, P.ta__minus__, P.G__plus__, P.H ],
               color = [ '1' ],
               lorentz = [ L.FFSS1, L.FFSS4 ],
               couplings = {(0,0):C.GC_2238,(0,1):C.GC_1174})

V_318 = Vertex(name = 'V_318',
               particles = [ P.u__tilde__, P.u, P.G0 ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFS2, L.FFS3, L.FFS6 ],
               couplings = {(0,0):C.GC_1123,(0,2):C.GC_1375,(0,1):C.GC_2268})

V_319 = Vertex(name = 'V_319',
               particles = [ P.u__tilde__, P.u, P.G0 ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFS3 ],
               couplings = {(0,0):C.GC_2269})

V_320 = Vertex(name = 'V_320',
               particles = [ P.c__tilde__, P.c, P.G0 ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFS2, L.FFS3, L.FFS6 ],
               couplings = {(0,0):C.GC_1123,(0,2):C.GC_1375,(0,1):C.GC_1860})

V_321 = Vertex(name = 'V_321',
               particles = [ P.c__tilde__, P.c, P.G0 ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFS3 ],
               couplings = {(0,0):C.GC_1861})

V_322 = Vertex(name = 'V_322',
               particles = [ P.t__tilde__, P.t, P.G0 ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFS2, L.FFS3, L.FFS6 ],
               couplings = {(0,0):C.GC_1123,(0,2):C.GC_1375,(0,1):C.GC_2152})

V_323 = Vertex(name = 'V_323',
               particles = [ P.t__tilde__, P.t, P.G0 ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFS3 ],
               couplings = {(0,0):C.GC_2153})

V_324 = Vertex(name = 'V_324',
               particles = [ P.u__tilde__, P.u, P.H ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFS5 ],
               couplings = {(0,0):C.GC_2267})

V_325 = Vertex(name = 'V_325',
               particles = [ P.u__tilde__, P.u, P.H ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFS5 ],
               couplings = {(0,0):C.GC_2299})

V_326 = Vertex(name = 'V_326',
               particles = [ P.c__tilde__, P.c, P.H ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFS5 ],
               couplings = {(0,0):C.GC_1859})

V_327 = Vertex(name = 'V_327',
               particles = [ P.c__tilde__, P.c, P.H ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFS5 ],
               couplings = {(0,0):C.GC_1891})

V_328 = Vertex(name = 'V_328',
               particles = [ P.t__tilde__, P.t, P.H ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFS5 ],
               couplings = {(0,0):C.GC_2151})

V_329 = Vertex(name = 'V_329',
               particles = [ P.t__tilde__, P.t, P.H ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFS5 ],
               couplings = {(0,0):C.GC_2183})

V_330 = Vertex(name = 'V_330',
               particles = [ P.u__tilde__, P.u, P.G0, P.G0, P.G0 ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFSSS2 ],
               couplings = {(0,0):C.GC_2290})

V_331 = Vertex(name = 'V_331',
               particles = [ P.c__tilde__, P.c, P.G0, P.G0, P.G0 ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFSSS2 ],
               couplings = {(0,0):C.GC_1882})

V_332 = Vertex(name = 'V_332',
               particles = [ P.t__tilde__, P.t, P.G0, P.G0, P.G0 ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFSSS2 ],
               couplings = {(0,0):C.GC_2174})

V_333 = Vertex(name = 'V_333',
               particles = [ P.u__tilde__, P.u, P.G0, P.G__minus__, P.G__plus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFSSS2 ],
               couplings = {(0,0):C.GC_2289})

V_334 = Vertex(name = 'V_334',
               particles = [ P.c__tilde__, P.c, P.G0, P.G__minus__, P.G__plus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFSSS2 ],
               couplings = {(0,0):C.GC_1881})

V_335 = Vertex(name = 'V_335',
               particles = [ P.t__tilde__, P.t, P.G0, P.G__minus__, P.G__plus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFSSS2 ],
               couplings = {(0,0):C.GC_2173})

V_336 = Vertex(name = 'V_336',
               particles = [ P.u__tilde__, P.u, P.G0, P.G0, P.H ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFSSS4 ],
               couplings = {(0,0):C.GC_2287})

V_337 = Vertex(name = 'V_337',
               particles = [ P.c__tilde__, P.c, P.G0, P.G0, P.H ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFSSS4 ],
               couplings = {(0,0):C.GC_1879})

V_338 = Vertex(name = 'V_338',
               particles = [ P.t__tilde__, P.t, P.G0, P.G0, P.H ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFSSS4 ],
               couplings = {(0,0):C.GC_2171})

V_339 = Vertex(name = 'V_339',
               particles = [ P.u__tilde__, P.u, P.G__minus__, P.G__plus__, P.H ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFSSS4 ],
               couplings = {(0,0):C.GC_2287})

V_340 = Vertex(name = 'V_340',
               particles = [ P.c__tilde__, P.c, P.G__minus__, P.G__plus__, P.H ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFSSS4 ],
               couplings = {(0,0):C.GC_1879})

V_341 = Vertex(name = 'V_341',
               particles = [ P.t__tilde__, P.t, P.G__minus__, P.G__plus__, P.H ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFSSS4 ],
               couplings = {(0,0):C.GC_2171})

V_342 = Vertex(name = 'V_342',
               particles = [ P.u__tilde__, P.u, P.G0, P.H, P.H ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFSSS2 ],
               couplings = {(0,0):C.GC_2289})

V_343 = Vertex(name = 'V_343',
               particles = [ P.c__tilde__, P.c, P.G0, P.H, P.H ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFSSS2 ],
               couplings = {(0,0):C.GC_1881})

V_344 = Vertex(name = 'V_344',
               particles = [ P.t__tilde__, P.t, P.G0, P.H, P.H ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFSSS2 ],
               couplings = {(0,0):C.GC_2173})

V_345 = Vertex(name = 'V_345',
               particles = [ P.u__tilde__, P.u, P.H, P.H, P.H ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFSSS4 ],
               couplings = {(0,0):C.GC_2288})

V_346 = Vertex(name = 'V_346',
               particles = [ P.c__tilde__, P.c, P.H, P.H, P.H ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFSSS4 ],
               couplings = {(0,0):C.GC_1880})

V_347 = Vertex(name = 'V_347',
               particles = [ P.t__tilde__, P.t, P.H, P.H, P.H ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFSSS4 ],
               couplings = {(0,0):C.GC_2172})

V_348 = Vertex(name = 'V_348',
               particles = [ P.u__tilde__, P.u, P.G0, P.G0 ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFSS7 ],
               couplings = {(0,0):C.GC_2291})

V_349 = Vertex(name = 'V_349',
               particles = [ P.c__tilde__, P.c, P.G0, P.G0 ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFSS7 ],
               couplings = {(0,0):C.GC_1883})

V_350 = Vertex(name = 'V_350',
               particles = [ P.t__tilde__, P.t, P.G0, P.G0 ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFSS7 ],
               couplings = {(0,0):C.GC_2175})

V_351 = Vertex(name = 'V_351',
               particles = [ P.u__tilde__, P.u, P.G__minus__, P.G__plus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFSS10, L.FFSS4, L.FFSS7 ],
               couplings = {(0,1):C.GC_1075,(0,0):C.GC_1168,(0,2):C.GC_2291})

V_352 = Vertex(name = 'V_352',
               particles = [ P.c__tilde__, P.c, P.G__minus__, P.G__plus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFSS10, L.FFSS4, L.FFSS7 ],
               couplings = {(0,1):C.GC_1075,(0,0):C.GC_1168,(0,2):C.GC_1883})

V_353 = Vertex(name = 'V_353',
               particles = [ P.t__tilde__, P.t, P.G__minus__, P.G__plus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFSS10, L.FFSS4, L.FFSS7 ],
               couplings = {(0,1):C.GC_1075,(0,0):C.GC_1168,(0,2):C.GC_2175})

V_354 = Vertex(name = 'V_354',
               particles = [ P.u__tilde__, P.u, P.G0, P.H ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFSS10, L.FFSS4, L.FFSS5 ],
               couplings = {(0,1):C.GC_1074,(0,0):C.GC_1169,(0,2):C.GC_2293})

V_355 = Vertex(name = 'V_355',
               particles = [ P.c__tilde__, P.c, P.G0, P.H ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFSS10, L.FFSS4, L.FFSS5 ],
               couplings = {(0,1):C.GC_1074,(0,0):C.GC_1169,(0,2):C.GC_1885})

V_356 = Vertex(name = 'V_356',
               particles = [ P.t__tilde__, P.t, P.G0, P.H ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFSS10, L.FFSS4, L.FFSS5 ],
               couplings = {(0,1):C.GC_1074,(0,0):C.GC_1169,(0,2):C.GC_2177})

V_357 = Vertex(name = 'V_357',
               particles = [ P.u__tilde__, P.u, P.H, P.H ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFSS7 ],
               couplings = {(0,0):C.GC_2292})

V_358 = Vertex(name = 'V_358',
               particles = [ P.c__tilde__, P.c, P.H, P.H ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFSS7 ],
               couplings = {(0,0):C.GC_1884})

V_359 = Vertex(name = 'V_359',
               particles = [ P.t__tilde__, P.t, P.H, P.H ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFSS7 ],
               couplings = {(0,0):C.GC_2176})

V_360 = Vertex(name = 'V_360',
               particles = [ P.d__tilde__, P.u, P.G__minus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFS1, L.FFS2, L.FFS4, L.FFS6 ],
               couplings = {(0,0):C.GC_53,(0,2):C.GC_71,(0,1):C.GC_1378,(0,3):C.GC_1376})

V_361 = Vertex(name = 'V_361',
               particles = [ P.d__tilde__, P.u, P.G__minus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFS1, L.FFS4 ],
               couplings = {(0,0):C.GC_54,(0,1):C.GC_72})

V_362 = Vertex(name = 'V_362',
               particles = [ P.s__tilde__, P.u, P.G__minus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFS1, L.FFS2, L.FFS4 ],
               couplings = {(0,0):C.GC_55,(0,2):C.GC_73,(0,1):C.GC_1379})

V_363 = Vertex(name = 'V_363',
               particles = [ P.s__tilde__, P.u, P.G__minus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFS1, L.FFS4 ],
               couplings = {(0,0):C.GC_56,(0,1):C.GC_74})

V_364 = Vertex(name = 'V_364',
               particles = [ P.b__tilde__, P.u, P.G__minus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFS1, L.FFS2, L.FFS4 ],
               couplings = {(0,0):C.GC_57,(0,2):C.GC_75,(0,1):C.GC_1380})

V_365 = Vertex(name = 'V_365',
               particles = [ P.b__tilde__, P.u, P.G__minus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFS1, L.FFS4 ],
               couplings = {(0,0):C.GC_58,(0,1):C.GC_76})

V_366 = Vertex(name = 'V_366',
               particles = [ P.d__tilde__, P.c, P.G__minus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFS1, L.FFS2, L.FFS4 ],
               couplings = {(0,0):C.GC_59,(0,2):C.GC_77,(0,1):C.GC_1381})

V_367 = Vertex(name = 'V_367',
               particles = [ P.d__tilde__, P.c, P.G__minus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFS1, L.FFS4 ],
               couplings = {(0,0):C.GC_60,(0,1):C.GC_78})

V_368 = Vertex(name = 'V_368',
               particles = [ P.s__tilde__, P.c, P.G__minus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFS1, L.FFS2, L.FFS4, L.FFS6 ],
               couplings = {(0,0):C.GC_61,(0,2):C.GC_79,(0,1):C.GC_1382,(0,3):C.GC_1376})

V_369 = Vertex(name = 'V_369',
               particles = [ P.s__tilde__, P.c, P.G__minus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFS1, L.FFS4 ],
               couplings = {(0,0):C.GC_62,(0,1):C.GC_80})

V_370 = Vertex(name = 'V_370',
               particles = [ P.b__tilde__, P.c, P.G__minus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFS1, L.FFS2, L.FFS4 ],
               couplings = {(0,0):C.GC_63,(0,2):C.GC_81,(0,1):C.GC_1383})

V_371 = Vertex(name = 'V_371',
               particles = [ P.b__tilde__, P.c, P.G__minus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFS1, L.FFS4 ],
               couplings = {(0,0):C.GC_64,(0,1):C.GC_82})

V_372 = Vertex(name = 'V_372',
               particles = [ P.d__tilde__, P.t, P.G__minus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFS1, L.FFS2, L.FFS4 ],
               couplings = {(0,0):C.GC_65,(0,2):C.GC_83,(0,1):C.GC_1384})

V_373 = Vertex(name = 'V_373',
               particles = [ P.d__tilde__, P.t, P.G__minus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFS1, L.FFS4 ],
               couplings = {(0,0):C.GC_66,(0,1):C.GC_84})

V_374 = Vertex(name = 'V_374',
               particles = [ P.s__tilde__, P.t, P.G__minus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFS1, L.FFS2, L.FFS4 ],
               couplings = {(0,0):C.GC_67,(0,2):C.GC_85,(0,1):C.GC_1385})

V_375 = Vertex(name = 'V_375',
               particles = [ P.s__tilde__, P.t, P.G__minus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFS1, L.FFS4 ],
               couplings = {(0,0):C.GC_68,(0,1):C.GC_86})

V_376 = Vertex(name = 'V_376',
               particles = [ P.b__tilde__, P.t, P.G__minus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFS1, L.FFS2, L.FFS4, L.FFS6 ],
               couplings = {(0,0):C.GC_69,(0,2):C.GC_87,(0,1):C.GC_1386,(0,3):C.GC_1376})

V_377 = Vertex(name = 'V_377',
               particles = [ P.b__tilde__, P.t, P.G__minus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFS1, L.FFS4 ],
               couplings = {(0,0):C.GC_70,(0,1):C.GC_88})

V_378 = Vertex(name = 'V_378',
               particles = [ P.d__tilde__, P.u, P.G0, P.G0, P.G__minus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFSSS1, L.FFSSS3 ],
               couplings = {(0,0):C.GC_1264,(0,1):C.GC_1283})

V_379 = Vertex(name = 'V_379',
               particles = [ P.s__tilde__, P.u, P.G0, P.G0, P.G__minus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFSSS1, L.FFSSS3 ],
               couplings = {(0,0):C.GC_1266,(0,1):C.GC_1285})

V_380 = Vertex(name = 'V_380',
               particles = [ P.b__tilde__, P.u, P.G0, P.G0, P.G__minus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFSSS1, L.FFSSS3 ],
               couplings = {(0,0):C.GC_1268,(0,1):C.GC_1287})

V_381 = Vertex(name = 'V_381',
               particles = [ P.d__tilde__, P.c, P.G0, P.G0, P.G__minus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFSSS1, L.FFSSS3 ],
               couplings = {(0,0):C.GC_1270,(0,1):C.GC_1289})

V_382 = Vertex(name = 'V_382',
               particles = [ P.s__tilde__, P.c, P.G0, P.G0, P.G__minus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFSSS1, L.FFSSS3 ],
               couplings = {(0,0):C.GC_1272,(0,1):C.GC_1291})

V_383 = Vertex(name = 'V_383',
               particles = [ P.b__tilde__, P.c, P.G0, P.G0, P.G__minus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFSSS1, L.FFSSS3 ],
               couplings = {(0,0):C.GC_1274,(0,1):C.GC_1293})

V_384 = Vertex(name = 'V_384',
               particles = [ P.d__tilde__, P.t, P.G0, P.G0, P.G__minus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFSSS1, L.FFSSS3 ],
               couplings = {(0,0):C.GC_1276,(0,1):C.GC_1295})

V_385 = Vertex(name = 'V_385',
               particles = [ P.s__tilde__, P.t, P.G0, P.G0, P.G__minus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFSSS1, L.FFSSS3 ],
               couplings = {(0,0):C.GC_1278,(0,1):C.GC_1297})

V_386 = Vertex(name = 'V_386',
               particles = [ P.b__tilde__, P.t, P.G0, P.G0, P.G__minus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFSSS1, L.FFSSS3 ],
               couplings = {(0,0):C.GC_1280,(0,1):C.GC_1299})

V_387 = Vertex(name = 'V_387',
               particles = [ P.d__tilde__, P.u, P.G__minus__, P.G__minus__, P.G__plus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFSSS1, L.FFSSS3 ],
               couplings = {(0,0):C.GC_1265,(0,1):C.GC_1282})

V_388 = Vertex(name = 'V_388',
               particles = [ P.s__tilde__, P.u, P.G__minus__, P.G__minus__, P.G__plus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFSSS1, L.FFSSS3 ],
               couplings = {(0,0):C.GC_1267,(0,1):C.GC_1284})

V_389 = Vertex(name = 'V_389',
               particles = [ P.b__tilde__, P.u, P.G__minus__, P.G__minus__, P.G__plus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFSSS1, L.FFSSS3 ],
               couplings = {(0,0):C.GC_1269,(0,1):C.GC_1286})

V_390 = Vertex(name = 'V_390',
               particles = [ P.d__tilde__, P.c, P.G__minus__, P.G__minus__, P.G__plus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFSSS1, L.FFSSS3 ],
               couplings = {(0,0):C.GC_1271,(0,1):C.GC_1288})

V_391 = Vertex(name = 'V_391',
               particles = [ P.s__tilde__, P.c, P.G__minus__, P.G__minus__, P.G__plus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFSSS1, L.FFSSS3 ],
               couplings = {(0,0):C.GC_1273,(0,1):C.GC_1290})

V_392 = Vertex(name = 'V_392',
               particles = [ P.b__tilde__, P.c, P.G__minus__, P.G__minus__, P.G__plus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFSSS1, L.FFSSS3 ],
               couplings = {(0,0):C.GC_1275,(0,1):C.GC_1292})

V_393 = Vertex(name = 'V_393',
               particles = [ P.d__tilde__, P.t, P.G__minus__, P.G__minus__, P.G__plus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFSSS1, L.FFSSS3 ],
               couplings = {(0,0):C.GC_1277,(0,1):C.GC_1294})

V_394 = Vertex(name = 'V_394',
               particles = [ P.s__tilde__, P.t, P.G__minus__, P.G__minus__, P.G__plus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFSSS1, L.FFSSS3 ],
               couplings = {(0,0):C.GC_1279,(0,1):C.GC_1296})

V_395 = Vertex(name = 'V_395',
               particles = [ P.b__tilde__, P.t, P.G__minus__, P.G__minus__, P.G__plus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFSSS1, L.FFSSS3 ],
               couplings = {(0,0):C.GC_1281,(0,1):C.GC_1298})

V_396 = Vertex(name = 'V_396',
               particles = [ P.d__tilde__, P.u, P.G__minus__, P.H, P.H ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFSSS1, L.FFSSS3 ],
               couplings = {(0,0):C.GC_1264,(0,1):C.GC_1283})

V_397 = Vertex(name = 'V_397',
               particles = [ P.s__tilde__, P.u, P.G__minus__, P.H, P.H ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFSSS1, L.FFSSS3 ],
               couplings = {(0,0):C.GC_1266,(0,1):C.GC_1285})

V_398 = Vertex(name = 'V_398',
               particles = [ P.b__tilde__, P.u, P.G__minus__, P.H, P.H ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFSSS1, L.FFSSS3 ],
               couplings = {(0,0):C.GC_1268,(0,1):C.GC_1287})

V_399 = Vertex(name = 'V_399',
               particles = [ P.d__tilde__, P.c, P.G__minus__, P.H, P.H ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFSSS1, L.FFSSS3 ],
               couplings = {(0,0):C.GC_1270,(0,1):C.GC_1289})

V_400 = Vertex(name = 'V_400',
               particles = [ P.s__tilde__, P.c, P.G__minus__, P.H, P.H ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFSSS1, L.FFSSS3 ],
               couplings = {(0,0):C.GC_1272,(0,1):C.GC_1291})

V_401 = Vertex(name = 'V_401',
               particles = [ P.b__tilde__, P.c, P.G__minus__, P.H, P.H ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFSSS1, L.FFSSS3 ],
               couplings = {(0,0):C.GC_1274,(0,1):C.GC_1293})

V_402 = Vertex(name = 'V_402',
               particles = [ P.d__tilde__, P.t, P.G__minus__, P.H, P.H ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFSSS1, L.FFSSS3 ],
               couplings = {(0,0):C.GC_1276,(0,1):C.GC_1295})

V_403 = Vertex(name = 'V_403',
               particles = [ P.s__tilde__, P.t, P.G__minus__, P.H, P.H ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFSSS1, L.FFSSS3 ],
               couplings = {(0,0):C.GC_1278,(0,1):C.GC_1297})

V_404 = Vertex(name = 'V_404',
               particles = [ P.b__tilde__, P.t, P.G__minus__, P.H, P.H ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFSSS1, L.FFSSS3 ],
               couplings = {(0,0):C.GC_1280,(0,1):C.GC_1299})

V_405 = Vertex(name = 'V_405',
               particles = [ P.d__tilde__, P.u, P.G__minus__, P.H ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFSS1, L.FFSS10, L.FFSS4, L.FFSS6 ],
               couplings = {(0,0):C.GC_1421,(0,3):C.GC_1430,(0,2):C.GC_1177,(0,1):C.GC_1170})

V_406 = Vertex(name = 'V_406',
               particles = [ P.s__tilde__, P.u, P.G__minus__, P.H ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFSS1, L.FFSS4, L.FFSS6 ],
               couplings = {(0,0):C.GC_1422,(0,2):C.GC_1431,(0,1):C.GC_1179})

V_407 = Vertex(name = 'V_407',
               particles = [ P.b__tilde__, P.u, P.G__minus__, P.H ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFSS1, L.FFSS4, L.FFSS6 ],
               couplings = {(0,0):C.GC_1423,(0,2):C.GC_1432,(0,1):C.GC_1181})

V_408 = Vertex(name = 'V_408',
               particles = [ P.d__tilde__, P.c, P.G__minus__, P.H ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFSS1, L.FFSS4, L.FFSS6 ],
               couplings = {(0,0):C.GC_1424,(0,2):C.GC_1433,(0,1):C.GC_1183})

V_409 = Vertex(name = 'V_409',
               particles = [ P.s__tilde__, P.c, P.G__minus__, P.H ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFSS1, L.FFSS10, L.FFSS4, L.FFSS6 ],
               couplings = {(0,0):C.GC_1425,(0,3):C.GC_1434,(0,2):C.GC_1185,(0,1):C.GC_1170})

V_410 = Vertex(name = 'V_410',
               particles = [ P.b__tilde__, P.c, P.G__minus__, P.H ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFSS1, L.FFSS4, L.FFSS6 ],
               couplings = {(0,0):C.GC_1426,(0,2):C.GC_1435,(0,1):C.GC_1187})

V_411 = Vertex(name = 'V_411',
               particles = [ P.d__tilde__, P.t, P.G__minus__, P.H ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFSS1, L.FFSS4, L.FFSS6 ],
               couplings = {(0,0):C.GC_1427,(0,2):C.GC_1436,(0,1):C.GC_1189})

V_412 = Vertex(name = 'V_412',
               particles = [ P.s__tilde__, P.t, P.G__minus__, P.H ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFSS1, L.FFSS4, L.FFSS6 ],
               couplings = {(0,0):C.GC_1428,(0,2):C.GC_1437,(0,1):C.GC_1191})

V_413 = Vertex(name = 'V_413',
               particles = [ P.b__tilde__, P.t, P.G__minus__, P.H ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFSS1, L.FFSS10, L.FFSS4, L.FFSS6 ],
               couplings = {(0,0):C.GC_1429,(0,3):C.GC_1438,(0,2):C.GC_1193,(0,1):C.GC_1170})

V_414 = Vertex(name = 'V_414',
               particles = [ P.a, P.W__minus__, P.G0, P.G__minus__, P.G__plus__, P.G__plus__ ],
               color = [ '1' ],
               lorentz = [ L.VVSSSS1 ],
               couplings = {(0,0):C.GC_1348})

V_415 = Vertex(name = 'V_415',
               particles = [ P.a, P.W__minus__, P.G__minus__, P.G__plus__, P.G__plus__, P.H ],
               color = [ '1' ],
               lorentz = [ L.VVSSSS1 ],
               couplings = {(0,0):C.GC_1349})

V_416 = Vertex(name = 'V_416',
               particles = [ P.a, P.W__minus__, P.G__minus__, P.G__plus__, P.G__plus__ ],
               color = [ '1' ],
               lorentz = [ L.VVSSS1 ],
               couplings = {(0,0):C.GC_1475})

V_417 = Vertex(name = 'V_417',
               particles = [ P.W__minus__, P.G0, P.G0, P.G__plus__, P.H ],
               color = [ '1' ],
               lorentz = [ L.VSSSS5 ],
               couplings = {(0,0):C.GC_1343})

V_418 = Vertex(name = 'V_418',
               particles = [ P.W__minus__, P.G0, P.G__plus__, P.H, P.H ],
               color = [ '1' ],
               lorentz = [ L.VSSSS10 ],
               couplings = {(0,0):C.GC_1341})

V_419 = Vertex(name = 'V_419',
               particles = [ P.W__minus__, P.G0, P.G0, P.G__plus__ ],
               color = [ '1' ],
               lorentz = [ L.VSSS3 ],
               couplings = {(0,0):C.GC_1471})

V_420 = Vertex(name = 'V_420',
               particles = [ P.W__minus__, P.G0, P.G__plus__, P.H ],
               color = [ '1' ],
               lorentz = [ L.VSSS5 ],
               couplings = {(0,0):C.GC_1469})

V_421 = Vertex(name = 'V_421',
               particles = [ P.W__minus__, P.G0, P.G__minus__, P.G__plus__, P.G__plus__ ],
               color = [ '1' ],
               lorentz = [ L.VSSSS11 ],
               couplings = {(0,0):C.GC_1341})

V_422 = Vertex(name = 'V_422',
               particles = [ P.W__minus__, P.G__minus__, P.G__plus__, P.G__plus__, P.H ],
               color = [ '1' ],
               lorentz = [ L.VSSSS3 ],
               couplings = {(0,0):C.GC_1339})

V_423 = Vertex(name = 'V_423',
               particles = [ P.W__minus__, P.G__minus__, P.G__plus__, P.G__plus__ ],
               color = [ '1' ],
               lorentz = [ L.VSSS6 ],
               couplings = {(0,0):C.GC_1468})

V_424 = Vertex(name = 'V_424',
               particles = [ P.W__minus__, P.W__plus__, P.G0, P.G0 ],
               color = [ '1' ],
               lorentz = [ L.VVSS11, L.VVSS16, L.VVSS23, L.VVSS5 ],
               couplings = {(0,1):C.GC_382,(0,3):C.GC_1032,(0,2):C.GC_375,(0,0):C.GC_369})

V_425 = Vertex(name = 'V_425',
               particles = [ P.W__minus__, P.W__plus__, P.G0, P.G0 ],
               color = [ '1' ],
               lorentz = [ L.VVSS11 ],
               couplings = {(0,0):C.GC_1714})

V_426 = Vertex(name = 'V_426',
               particles = [ P.W__minus__, P.W__plus__, P.G__minus__, P.G__plus__ ],
               color = [ '1' ],
               lorentz = [ L.VVSS11, L.VVSS18, L.VVSS26, L.VVSS7 ],
               couplings = {(0,2):C.GC_382,(0,3):C.GC_1033,(0,1):C.GC_376,(0,0):C.GC_369})

V_427 = Vertex(name = 'V_427',
               particles = [ P.W__minus__, P.W__plus__, P.G__minus__, P.G__plus__ ],
               color = [ '1' ],
               lorentz = [ L.VVSS11 ],
               couplings = {(0,0):C.GC_1767})

V_428 = Vertex(name = 'V_428',
               particles = [ P.W__minus__, P.W__plus__, P.H, P.H ],
               color = [ '1' ],
               lorentz = [ L.VVSS11, L.VVSS16, L.VVSS23, L.VVSS5 ],
               couplings = {(0,1):C.GC_382,(0,3):C.GC_1032,(0,2):C.GC_375,(0,0):C.GC_369})

V_429 = Vertex(name = 'V_429',
               particles = [ P.W__minus__, P.W__plus__, P.H, P.H ],
               color = [ '1' ],
               lorentz = [ L.VVSS11 ],
               couplings = {(0,0):C.GC_1766})

V_430 = Vertex(name = 'V_430',
               particles = [ P.W__minus__, P.W__plus__, P.H ],
               color = [ '1' ],
               lorentz = [ L.VVS11, L.VVS14, L.VVS5, L.VVS7 ],
               couplings = {(0,0):C.GC_1514,(0,2):C.GC_1698,(0,1):C.GC_1510,(0,3):C.GC_1506})

V_431 = Vertex(name = 'V_431',
               particles = [ P.W__minus__, P.W__plus__, P.H ],
               color = [ '1' ],
               lorentz = [ L.VVS7 ],
               couplings = {(0,0):C.GC_1805})

V_432 = Vertex(name = 'V_432',
               particles = [ P.W__minus__, P.Z, P.G0, P.G__plus__ ],
               color = [ '1' ],
               lorentz = [ L.VVSS10, L.VVSS11, L.VVSS12, L.VVSS15, L.VVSS17, L.VVSS22, L.VVSS24, L.VVSS27, L.VVSS3, L.VVSS6 ],
               couplings = {(0,2):C.GC_144,(0,3):C.GC_135,(0,8):C.GC_1026,(0,0):C.GC_1036,(0,9):C.GC_1010,(0,4):C.GC_141,(0,6):C.GC_378,(0,7):C.GC_384,(0,5):C.GC_138,(0,1):C.GC_9})

V_433 = Vertex(name = 'V_433',
               particles = [ P.W__minus__, P.Z, P.G0, P.G__plus__ ],
               color = [ '1' ],
               lorentz = [ L.VVSS11 ],
               couplings = {(0,0):C.GC_1782})

V_434 = Vertex(name = 'V_434',
               particles = [ P.W__minus__, P.Z, P.G__plus__, P.H ],
               color = [ '1' ],
               lorentz = [ L.VVSS10, L.VVSS11, L.VVSS12, L.VVSS15, L.VVSS17, L.VVSS22, L.VVSS24, L.VVSS27, L.VVSS3, L.VVSS6 ],
               couplings = {(0,2):C.GC_143,(0,3):C.GC_134,(0,8):C.GC_1025,(0,0):C.GC_1035,(0,9):C.GC_1011,(0,4):C.GC_142,(0,6):C.GC_379,(0,7):C.GC_385,(0,5):C.GC_139,(0,1):C.GC_10})

V_435 = Vertex(name = 'V_435',
               particles = [ P.W__minus__, P.Z, P.G__plus__, P.H ],
               color = [ '1' ],
               lorentz = [ L.VVSS11 ],
               couplings = {(0,0):C.GC_1783})

V_436 = Vertex(name = 'V_436',
               particles = [ P.W__minus__, P.Z, P.G__plus__ ],
               color = [ '1' ],
               lorentz = [ L.VVS10, L.VVS12, L.VVS13, L.VVS16, L.VVS17, L.VVS3, L.VVS4, L.VVS6, L.VVS7, L.VVS8 ],
               couplings = {(0,9):C.GC_1496,(0,0):C.GC_1490,(0,5):C.GC_1694,(0,7):C.GC_1700,(0,6):C.GC_1689,(0,1):C.GC_1495,(0,4):C.GC_1512,(0,3):C.GC_1516,(0,2):C.GC_1493,(0,8):C.GC_1488})

V_437 = Vertex(name = 'V_437',
               particles = [ P.W__minus__, P.Z, P.G__plus__ ],
               color = [ '1' ],
               lorentz = [ L.VVS7 ],
               couplings = {(0,0):C.GC_1807})

V_438 = Vertex(name = 'V_438',
               particles = [ P.a, P.W__minus__, P.W__plus__, P.G0, P.G0 ],
               color = [ '1' ],
               lorentz = [ L.VVVSS1, L.VVVSS11, L.VVVSS12, L.VVVSS19, L.VVVSS8 ],
               couplings = {(0,0):C.GC_1016,(0,4):C.GC_1038,(0,1):C.GC_388,(0,2):C.GC_390,(0,3):C.GC_398})

V_439 = Vertex(name = 'V_439',
               particles = [ P.a, P.W__minus__, P.W__plus__, P.G__minus__, P.G__plus__ ],
               color = [ '1' ],
               lorentz = [ L.VVVSS1, L.VVVSS10, L.VVVSS11, L.VVVSS21, L.VVVSS24 ],
               couplings = {(0,0):C.GC_1015,(0,1):C.GC_1038,(0,2):C.GC_389,(0,3):C.GC_390,(0,4):C.GC_398})

V_440 = Vertex(name = 'V_440',
               particles = [ P.a, P.W__minus__, P.W__plus__, P.H, P.H ],
               color = [ '1' ],
               lorentz = [ L.VVVSS1, L.VVVSS11, L.VVVSS12, L.VVVSS19, L.VVVSS8 ],
               couplings = {(0,0):C.GC_1016,(0,4):C.GC_1038,(0,1):C.GC_388,(0,2):C.GC_390,(0,3):C.GC_398})

V_441 = Vertex(name = 'V_441',
               particles = [ P.a, P.W__minus__, P.W__plus__, P.H ],
               color = [ '1' ],
               lorentz = [ L.VVVS1, L.VVVS10, L.VVVS11, L.VVVS18, L.VVVS8 ],
               couplings = {(0,0):C.GC_1691,(0,4):C.GC_1702,(0,1):C.GC_1517,(0,2):C.GC_1518,(0,3):C.GC_1521})

V_442 = Vertex(name = 'V_442',
               particles = [ P.W__minus__, P.W__plus__, P.G0, P.H ],
               color = [ '1' ],
               lorentz = [ L.VVSS10, L.VVSS24, L.VVSS27 ],
               couplings = {(0,0):C.GC_1034,(0,1):C.GC_374,(0,2):C.GC_381})

V_443 = Vertex(name = 'V_443',
               particles = [ P.W__minus__, P.W__plus__, P.G0 ],
               color = [ '1' ],
               lorentz = [ L.VVS16, L.VVS17, L.VVS6 ],
               couplings = {(0,2):C.GC_1699,(0,1):C.GC_1509,(0,0):C.GC_1513})

V_444 = Vertex(name = 'V_444',
               particles = [ P.a, P.a, P.a, P.W__minus__, P.W__plus__ ],
               color = [ '1' ],
               lorentz = [ L.VVVVV20, L.VVVVV87, L.VVVVV9 ],
               couplings = {(0,2):C.GC_1003,(0,1):C.GC_146,(0,0):C.GC_406})

V_445 = Vertex(name = 'V_445',
               particles = [ P.a, P.W__minus__, P.Z, P.G0, P.G__plus__ ],
               color = [ '1' ],
               lorentz = [ L.VVVSS1, L.VVVSS11, L.VVVSS14, L.VVVSS16, L.VVVSS22, L.VVVSS3, L.VVVSS30, L.VVVSS32, L.VVVSS6, L.VVVSS7 ],
               couplings = {(0,0):C.GC_1018,(0,8):C.GC_1029,(0,9):C.GC_1040,(0,5):C.GC_1013,(0,4):C.GC_151,(0,2):C.GC_154,(0,1):C.GC_392,(0,7):C.GC_395,(0,6):C.GC_400,(0,3):C.GC_148})

V_446 = Vertex(name = 'V_446',
               particles = [ P.a, P.W__minus__, P.Z, P.G__plus__, P.H ],
               color = [ '1' ],
               lorentz = [ L.VVVSS1, L.VVVSS11, L.VVVSS14, L.VVVSS16, L.VVVSS22, L.VVVSS23, L.VVVSS26, L.VVVSS3, L.VVVSS6, L.VVVSS7 ],
               couplings = {(0,0):C.GC_1017,(0,8):C.GC_1030,(0,9):C.GC_1041,(0,7):C.GC_1012,(0,4):C.GC_150,(0,2):C.GC_153,(0,1):C.GC_393,(0,5):C.GC_396,(0,6):C.GC_401,(0,3):C.GC_149})

V_447 = Vertex(name = 'V_447',
               particles = [ P.a, P.W__minus__, P.Z, P.G__plus__ ],
               color = [ '1' ],
               lorentz = [ L.VVVS1, L.VVVS10, L.VVVS13, L.VVVS15, L.VVVS20, L.VVVS23, L.VVVS24, L.VVVS3, L.VVVS6, L.VVVS7 ],
               couplings = {(0,0):C.GC_1692,(0,8):C.GC_1696,(0,9):C.GC_1703,(0,7):C.GC_1690,(0,4):C.GC_1499,(0,2):C.GC_1500,(0,1):C.GC_1519,(0,6):C.GC_1520,(0,5):C.GC_1522,(0,3):C.GC_1498})

V_448 = Vertex(name = 'V_448',
               particles = [ P.a, P.W__minus__, P.W__plus__, P.Z ],
               color = [ '1' ],
               lorentz = [ L.VVVV11, L.VVVV14, L.VVVV20, L.VVVV7 ],
               couplings = {(0,3):C.GC_999,(0,1):C.GC_366,(0,2):C.GC_591,(0,0):C.GC_574})

V_449 = Vertex(name = 'V_449',
               particles = [ P.a, P.W__minus__, P.W__plus__, P.Z ],
               color = [ '1' ],
               lorentz = [ L.VVVV11 ],
               couplings = {(0,0):C.GC_1800})

V_450 = Vertex(name = 'V_450',
               particles = [ P.W__minus__, P.W__minus__, P.G__plus__, P.G__plus__ ],
               color = [ '1' ],
               lorentz = [ L.VVSS11 ],
               couplings = {(0,0):C.GC_371})

V_451 = Vertex(name = 'V_451',
               particles = [ P.W__minus__, P.W__minus__, P.G0, P.G0, P.G__plus__, P.G__plus__ ],
               color = [ '1' ],
               lorentz = [ L.VVSSSS1 ],
               couplings = {(0,0):C.GC_1306})

V_452 = Vertex(name = 'V_452',
               particles = [ P.W__minus__, P.W__minus__, P.G0, P.G__plus__, P.G__plus__, P.H ],
               color = [ '1' ],
               lorentz = [ L.VVSSSS1 ],
               couplings = {(0,0):C.GC_1304})

V_453 = Vertex(name = 'V_453',
               particles = [ P.W__minus__, P.W__minus__, P.G__plus__, P.G__plus__, P.H, P.H ],
               color = [ '1' ],
               lorentz = [ L.VVSSSS1 ],
               couplings = {(0,0):C.GC_1307})

V_454 = Vertex(name = 'V_454',
               particles = [ P.W__minus__, P.W__minus__, P.G0, P.G__plus__, P.G__plus__ ],
               color = [ '1' ],
               lorentz = [ L.VVSSS1 ],
               couplings = {(0,0):C.GC_1441})

V_455 = Vertex(name = 'V_455',
               particles = [ P.W__minus__, P.W__minus__, P.G__plus__, P.G__plus__, P.H ],
               color = [ '1' ],
               lorentz = [ L.VVSSS1 ],
               couplings = {(0,0):C.GC_1443})

V_456 = Vertex(name = 'V_456',
               particles = [ P.W__minus__, P.W__minus__, P.W__plus__, P.W__plus__ ],
               color = [ '1' ],
               lorentz = [ L.VVVV15, L.VVVV17, L.VVVV4, L.VVVV8 ],
               couplings = {(0,2):C.GC_994,(0,1):C.GC_373,(0,0):C.GC_353,(0,3):C.GC_370})

V_457 = Vertex(name = 'V_457',
               particles = [ P.W__minus__, P.W__minus__, P.W__plus__, P.W__plus__ ],
               color = [ '1' ],
               lorentz = [ L.VVVV8 ],
               couplings = {(0,0):C.GC_1713})

V_458 = Vertex(name = 'V_458',
               particles = [ P.W__minus__, P.W__minus__, P.W__plus__, P.G0, P.G__plus__ ],
               color = [ '1' ],
               lorentz = [ L.VVVSS28 ],
               couplings = {(0,0):C.GC_321})

V_459 = Vertex(name = 'V_459',
               particles = [ P.W__minus__, P.W__minus__, P.W__plus__, P.G__plus__, P.H ],
               color = [ '1' ],
               lorentz = [ L.VVVSS28 ],
               couplings = {(0,0):C.GC_322})

V_460 = Vertex(name = 'V_460',
               particles = [ P.W__minus__, P.W__minus__, P.W__plus__, P.G__plus__ ],
               color = [ '1' ],
               lorentz = [ L.VVVS21 ],
               couplings = {(0,0):C.GC_1731})

V_461 = Vertex(name = 'V_461',
               particles = [ P.a, P.W__plus__, P.G0, P.G__minus__, P.G__minus__, P.G__plus__ ],
               color = [ '1' ],
               lorentz = [ L.VVSSSS1 ],
               couplings = {(0,0):C.GC_1348})

V_462 = Vertex(name = 'V_462',
               particles = [ P.a, P.W__plus__, P.G__minus__, P.G__minus__, P.G__plus__, P.H ],
               color = [ '1' ],
               lorentz = [ L.VVSSSS1 ],
               couplings = {(0,0):C.GC_1347})

V_463 = Vertex(name = 'V_463',
               particles = [ P.a, P.W__plus__, P.G__minus__, P.G__minus__, P.G__plus__ ],
               color = [ '1' ],
               lorentz = [ L.VVSSS1 ],
               couplings = {(0,0):C.GC_1474})

V_464 = Vertex(name = 'V_464',
               particles = [ P.W__plus__, P.G0, P.G0, P.G__minus__, P.H ],
               color = [ '1' ],
               lorentz = [ L.VSSSS5 ],
               couplings = {(0,0):C.GC_1343})

V_465 = Vertex(name = 'V_465',
               particles = [ P.W__plus__, P.G0, P.G__minus__, P.H, P.H ],
               color = [ '1' ],
               lorentz = [ L.VSSSS10 ],
               couplings = {(0,0):C.GC_1342})

V_466 = Vertex(name = 'V_466',
               particles = [ P.W__plus__, P.G0, P.G0, P.G__minus__ ],
               color = [ '1' ],
               lorentz = [ L.VSSS3 ],
               couplings = {(0,0):C.GC_1471})

V_467 = Vertex(name = 'V_467',
               particles = [ P.W__plus__, P.G0, P.G__minus__, P.H ],
               color = [ '1' ],
               lorentz = [ L.VSSS5 ],
               couplings = {(0,0):C.GC_1470})

V_468 = Vertex(name = 'V_468',
               particles = [ P.W__plus__, P.G0, P.G__minus__, P.G__minus__, P.G__plus__ ],
               color = [ '1' ],
               lorentz = [ L.VSSSS6 ],
               couplings = {(0,0):C.GC_1340})

V_469 = Vertex(name = 'V_469',
               particles = [ P.W__plus__, P.G__minus__, P.G__minus__, P.G__plus__, P.H ],
               color = [ '1' ],
               lorentz = [ L.VSSSS2 ],
               couplings = {(0,0):C.GC_1343})

V_470 = Vertex(name = 'V_470',
               particles = [ P.W__plus__, P.G__minus__, P.G__minus__, P.G__plus__ ],
               color = [ '1' ],
               lorentz = [ L.VSSS4 ],
               couplings = {(0,0):C.GC_1471})

V_471 = Vertex(name = 'V_471',
               particles = [ P.W__plus__, P.Z, P.G0, P.G__minus__ ],
               color = [ '1' ],
               lorentz = [ L.VVSS10, L.VVSS11, L.VVSS12, L.VVSS15, L.VVSS17, L.VVSS22, L.VVSS24, L.VVSS27, L.VVSS3, L.VVSS6 ],
               couplings = {(0,2):C.GC_144,(0,3):C.GC_135,(0,8):C.GC_1026,(0,0):C.GC_1036,(0,9):C.GC_1010,(0,4):C.GC_141,(0,6):C.GC_378,(0,7):C.GC_384,(0,5):C.GC_138,(0,1):C.GC_9})

V_472 = Vertex(name = 'V_472',
               particles = [ P.W__plus__, P.Z, P.G0, P.G__minus__ ],
               color = [ '1' ],
               lorentz = [ L.VVSS11 ],
               couplings = {(0,0):C.GC_1782})

V_473 = Vertex(name = 'V_473',
               particles = [ P.W__plus__, P.Z, P.G__minus__, P.H ],
               color = [ '1' ],
               lorentz = [ L.VVSS10, L.VVSS11, L.VVSS12, L.VVSS15, L.VVSS17, L.VVSS22, L.VVSS24, L.VVSS27, L.VVSS3, L.VVSS6 ],
               couplings = {(0,2):C.GC_145,(0,3):C.GC_136,(0,8):C.GC_1027,(0,0):C.GC_1037,(0,9):C.GC_1009,(0,4):C.GC_140,(0,6):C.GC_377,(0,7):C.GC_383,(0,5):C.GC_137,(0,1):C.GC_8})

V_474 = Vertex(name = 'V_474',
               particles = [ P.W__plus__, P.Z, P.G__minus__, P.H ],
               color = [ '1' ],
               lorentz = [ L.VVSS11 ],
               couplings = {(0,0):C.GC_1781})

V_475 = Vertex(name = 'V_475',
               particles = [ P.W__plus__, P.Z, P.G__minus__ ],
               color = [ '1' ],
               lorentz = [ L.VVS10, L.VVS12, L.VVS13, L.VVS16, L.VVS17, L.VVS3, L.VVS4, L.VVS6, L.VVS7, L.VVS8 ],
               couplings = {(0,9):C.GC_1497,(0,0):C.GC_1491,(0,5):C.GC_1695,(0,7):C.GC_1701,(0,6):C.GC_1688,(0,1):C.GC_1494,(0,4):C.GC_1511,(0,3):C.GC_1515,(0,2):C.GC_1492,(0,8):C.GC_1487})

V_476 = Vertex(name = 'V_476',
               particles = [ P.W__plus__, P.Z, P.G__minus__ ],
               color = [ '1' ],
               lorentz = [ L.VVS7 ],
               couplings = {(0,0):C.GC_1806})

V_477 = Vertex(name = 'V_477',
               particles = [ P.a, P.W__plus__, P.Z, P.G0, P.G__minus__ ],
               color = [ '1' ],
               lorentz = [ L.VVVSS1, L.VVVSS11, L.VVVSS14, L.VVVSS16, L.VVVSS22, L.VVVSS3, L.VVVSS30, L.VVVSS32, L.VVVSS6, L.VVVSS7 ],
               couplings = {(0,0):C.GC_1019,(0,8):C.GC_1028,(0,9):C.GC_1039,(0,5):C.GC_1014,(0,4):C.GC_152,(0,2):C.GC_155,(0,1):C.GC_391,(0,7):C.GC_394,(0,6):C.GC_399,(0,3):C.GC_147})

V_478 = Vertex(name = 'V_478',
               particles = [ P.a, P.W__plus__, P.Z, P.G__minus__, P.H ],
               color = [ '1' ],
               lorentz = [ L.VVVSS1, L.VVVSS11, L.VVVSS14, L.VVVSS16, L.VVVSS22, L.VVVSS23, L.VVVSS26, L.VVVSS3, L.VVVSS6, L.VVVSS7 ],
               couplings = {(0,0):C.GC_1017,(0,8):C.GC_1030,(0,9):C.GC_1041,(0,7):C.GC_1012,(0,4):C.GC_150,(0,2):C.GC_153,(0,1):C.GC_393,(0,5):C.GC_396,(0,6):C.GC_401,(0,3):C.GC_149})

V_479 = Vertex(name = 'V_479',
               particles = [ P.a, P.W__plus__, P.Z, P.G__minus__ ],
               color = [ '1' ],
               lorentz = [ L.VVVS1, L.VVVS10, L.VVVS13, L.VVVS15, L.VVVS20, L.VVVS23, L.VVVS24, L.VVVS3, L.VVVS6, L.VVVS7 ],
               couplings = {(0,0):C.GC_1692,(0,8):C.GC_1696,(0,9):C.GC_1703,(0,7):C.GC_1690,(0,4):C.GC_1499,(0,2):C.GC_1500,(0,1):C.GC_1519,(0,6):C.GC_1520,(0,5):C.GC_1522,(0,3):C.GC_1498})

V_480 = Vertex(name = 'V_480',
               particles = [ P.W__minus__, P.W__plus__, P.G0, P.G0, P.G__minus__, P.G__plus__ ],
               color = [ '1' ],
               lorentz = [ L.VVSSSS1 ],
               couplings = {(0,0):C.GC_1305})

V_481 = Vertex(name = 'V_481',
               particles = [ P.W__minus__, P.W__plus__, P.G__minus__, P.G__plus__, P.H, P.H ],
               color = [ '1' ],
               lorentz = [ L.VVSSSS1 ],
               couplings = {(0,0):C.GC_1305})

V_482 = Vertex(name = 'V_482',
               particles = [ P.W__minus__, P.W__plus__, P.G__minus__, P.G__plus__, P.H ],
               color = [ '1' ],
               lorentz = [ L.VVSSS1 ],
               couplings = {(0,0):C.GC_1442})

V_483 = Vertex(name = 'V_483',
               particles = [ P.a, P.a, P.W__minus__, P.W__plus__, P.G0, P.G0 ],
               color = [ '1' ],
               lorentz = [ L.VVVVSS2 ],
               couplings = {(0,0):C.GC_405})

V_484 = Vertex(name = 'V_484',
               particles = [ P.a, P.a, P.W__minus__, P.W__plus__, P.G__minus__, P.G__plus__ ],
               color = [ '1' ],
               lorentz = [ L.VVVVSS2 ],
               couplings = {(0,0):C.GC_335})

V_485 = Vertex(name = 'V_485',
               particles = [ P.a, P.a, P.W__minus__, P.W__plus__, P.H, P.H ],
               color = [ '1' ],
               lorentz = [ L.VVVVSS2 ],
               couplings = {(0,0):C.GC_405})

V_486 = Vertex(name = 'V_486',
               particles = [ P.a, P.a, P.W__minus__, P.W__plus__, P.H ],
               color = [ '1' ],
               lorentz = [ L.VVVVS2 ],
               couplings = {(0,0):C.GC_1523})

V_487 = Vertex(name = 'V_487',
               particles = [ P.a, P.a, P.a, P.a, P.W__minus__, P.W__plus__ ],
               color = [ '1' ],
               lorentz = [ L.VVVVVV119 ],
               couplings = {(0,0):C.GC_156})

V_488 = Vertex(name = 'V_488',
               particles = [ P.a, P.W__minus__, P.W__plus__, P.G0, P.H ],
               color = [ '1' ],
               lorentz = [ L.VVVSS34 ],
               couplings = {(0,0):C.GC_332})

V_489 = Vertex(name = 'V_489',
               particles = [ P.a, P.W__minus__, P.W__plus__, P.G0 ],
               color = [ '1' ],
               lorentz = [ L.VVVS19 ],
               couplings = {(0,0):C.GC_1741})

V_490 = Vertex(name = 'V_490',
               particles = [ P.W__minus__, P.W__plus__, P.Z, P.G0, P.G0 ],
               color = [ '1' ],
               lorentz = [ L.VVVSS14, L.VVVSS16, L.VVVSS18, L.VVVSS22, L.VVVSS3, L.VVVSS6, L.VVVSS9 ],
               couplings = {(0,6):C.GC_1031,(0,5):C.GC_1049,(0,4):C.GC_1023,(0,2):C.GC_324,(0,3):C.GC_603,(0,0):C.GC_612,(0,1):C.GC_602})

V_491 = Vertex(name = 'V_491',
               particles = [ P.W__minus__, P.W__plus__, P.Z, P.G__minus__, P.G__plus__ ],
               color = [ '1' ],
               lorentz = [ L.VVVSS14, L.VVVSS16, L.VVVSS22, L.VVVSS27, L.VVVSS3, L.VVVSS31, L.VVVSS6, L.VVVSS9 ],
               couplings = {(0,7):C.GC_1031,(0,6):C.GC_1048,(0,4):C.GC_1024,(0,5):C.GC_362,(0,3):C.GC_364,(0,2):C.GC_604,(0,0):C.GC_613,(0,1):C.GC_601})

V_492 = Vertex(name = 'V_492',
               particles = [ P.W__minus__, P.W__plus__, P.Z, P.H, P.H ],
               color = [ '1' ],
               lorentz = [ L.VVVSS14, L.VVVSS16, L.VVVSS18, L.VVVSS22, L.VVVSS3, L.VVVSS6, L.VVVSS9 ],
               couplings = {(0,6):C.GC_1031,(0,5):C.GC_1049,(0,4):C.GC_1023,(0,2):C.GC_324,(0,3):C.GC_603,(0,0):C.GC_612,(0,1):C.GC_602})

V_493 = Vertex(name = 'V_493',
               particles = [ P.W__minus__, P.W__plus__, P.Z, P.H ],
               color = [ '1' ],
               lorentz = [ L.VVVS13, L.VVVS15, L.VVVS17, L.VVVS20, L.VVVS3, L.VVVS6, L.VVVS9 ],
               couplings = {(0,6):C.GC_1697,(0,5):C.GC_1704,(0,4):C.GC_1693,(0,2):C.GC_1733,(0,3):C.GC_1569,(0,0):C.GC_1572,(0,1):C.GC_1568})

V_494 = Vertex(name = 'V_494',
               particles = [ P.W__minus__, P.W__plus__, P.Z, P.Z ],
               color = [ '1' ],
               lorentz = [ L.VVVV15, L.VVVV17, L.VVVV4, L.VVVV8 ],
               couplings = {(0,2):C.GC_995,(0,1):C.GC_380,(0,0):C.GC_355,(0,3):C.GC_372})

V_495 = Vertex(name = 'V_495',
               particles = [ P.W__minus__, P.W__plus__, P.Z, P.Z ],
               color = [ '1' ],
               lorentz = [ L.VVVV8 ],
               couplings = {(0,0):C.GC_1780})

V_496 = Vertex(name = 'V_496',
               particles = [ P.a, P.W__minus__, P.W__minus__, P.W__plus__, P.G0, P.G__plus__ ],
               color = [ '1' ],
               lorentz = [ L.VVVVSS5 ],
               couplings = {(0,0):C.GC_329})

V_497 = Vertex(name = 'V_497',
               particles = [ P.a, P.W__minus__, P.W__minus__, P.W__plus__, P.G__plus__, P.H ],
               color = [ '1' ],
               lorentz = [ L.VVVVSS5 ],
               couplings = {(0,0):C.GC_331})

V_498 = Vertex(name = 'V_498',
               particles = [ P.a, P.W__minus__, P.W__minus__, P.W__plus__, P.G__plus__ ],
               color = [ '1' ],
               lorentz = [ L.VVVVS5 ],
               couplings = {(0,0):C.GC_1736})

V_499 = Vertex(name = 'V_499',
               particles = [ P.a, P.a, P.W__minus__, P.W__plus__, P.Z ],
               color = [ '1' ],
               lorentz = [ L.VVVVV35, L.VVVVV90, L.VVVVV92 ],
               couplings = {(0,0):C.GC_1000,(0,1):C.GC_368,(0,2):C.GC_605})

V_500 = Vertex(name = 'V_500',
               particles = [ P.a, P.W__minus__, P.W__minus__, P.W__plus__, P.W__plus__ ],
               color = [ '1' ],
               lorentz = [ L.VVVVV41, L.VVVVV80, L.VVVVV93 ],
               couplings = {(0,0):C.GC_996,(0,1):C.GC_386,(0,2):C.GC_357})

V_501 = Vertex(name = 'V_501',
               particles = [ P.W__plus__, P.W__plus__, P.G__minus__, P.G__minus__ ],
               color = [ '1' ],
               lorentz = [ L.VVSS11 ],
               couplings = {(0,0):C.GC_371})

V_502 = Vertex(name = 'V_502',
               particles = [ P.W__plus__, P.W__plus__, P.G0, P.G0, P.G__minus__, P.G__minus__ ],
               color = [ '1' ],
               lorentz = [ L.VVSSSS1 ],
               couplings = {(0,0):C.GC_1306})

V_503 = Vertex(name = 'V_503',
               particles = [ P.W__plus__, P.W__plus__, P.G0, P.G__minus__, P.G__minus__, P.H ],
               color = [ '1' ],
               lorentz = [ L.VVSSSS1 ],
               couplings = {(0,0):C.GC_1308})

V_504 = Vertex(name = 'V_504',
               particles = [ P.W__plus__, P.W__plus__, P.G__minus__, P.G__minus__, P.H, P.H ],
               color = [ '1' ],
               lorentz = [ L.VVSSSS1 ],
               couplings = {(0,0):C.GC_1307})

V_505 = Vertex(name = 'V_505',
               particles = [ P.W__plus__, P.W__plus__, P.G0, P.G__minus__, P.G__minus__ ],
               color = [ '1' ],
               lorentz = [ L.VVSSS1 ],
               couplings = {(0,0):C.GC_1444})

V_506 = Vertex(name = 'V_506',
               particles = [ P.W__plus__, P.W__plus__, P.G__minus__, P.G__minus__, P.H ],
               color = [ '1' ],
               lorentz = [ L.VVSSS1 ],
               couplings = {(0,0):C.GC_1443})

V_507 = Vertex(name = 'V_507',
               particles = [ P.a, P.W__minus__, P.W__plus__, P.W__plus__, P.G0, P.G__minus__ ],
               color = [ '1' ],
               lorentz = [ L.VVVVSS2 ],
               couplings = {(0,0):C.GC_328})

V_508 = Vertex(name = 'V_508',
               particles = [ P.a, P.W__minus__, P.W__plus__, P.W__plus__, P.G__minus__, P.H ],
               color = [ '1' ],
               lorentz = [ L.VVVVSS2 ],
               couplings = {(0,0):C.GC_330})

V_509 = Vertex(name = 'V_509',
               particles = [ P.a, P.W__minus__, P.W__plus__, P.W__plus__, P.G__minus__ ],
               color = [ '1' ],
               lorentz = [ L.VVVVS2 ],
               couplings = {(0,0):C.GC_1735})

V_510 = Vertex(name = 'V_510',
               particles = [ P.W__minus__, P.W__plus__, P.W__plus__, P.G0, P.G__minus__ ],
               color = [ '1' ],
               lorentz = [ L.VVVSS34 ],
               couplings = {(0,0):C.GC_320})

V_511 = Vertex(name = 'V_511',
               particles = [ P.W__minus__, P.W__plus__, P.W__plus__, P.G__minus__, P.H ],
               color = [ '1' ],
               lorentz = [ L.VVVSS34 ],
               couplings = {(0,0):C.GC_319})

V_512 = Vertex(name = 'V_512',
               particles = [ P.W__minus__, P.W__plus__, P.W__plus__, P.G__minus__ ],
               color = [ '1' ],
               lorentz = [ L.VVVS19 ],
               couplings = {(0,0):C.GC_1730})

V_513 = Vertex(name = 'V_513',
               particles = [ P.W__minus__, P.W__minus__, P.W__plus__, P.W__plus__, P.G0, P.G0 ],
               color = [ '1' ],
               lorentz = [ L.VVVVSS2 ],
               couplings = {(0,0):C.GC_318})

V_514 = Vertex(name = 'V_514',
               particles = [ P.W__minus__, P.W__minus__, P.W__plus__, P.W__plus__, P.G__minus__, P.G__plus__ ],
               color = [ '1' ],
               lorentz = [ L.VVVVSS2 ],
               couplings = {(0,0):C.GC_318})

V_515 = Vertex(name = 'V_515',
               particles = [ P.W__minus__, P.W__minus__, P.W__plus__, P.W__plus__, P.H, P.H ],
               color = [ '1' ],
               lorentz = [ L.VVVVSS2 ],
               couplings = {(0,0):C.GC_318})

V_516 = Vertex(name = 'V_516',
               particles = [ P.W__minus__, P.W__minus__, P.W__plus__, P.W__plus__, P.H ],
               color = [ '1' ],
               lorentz = [ L.VVVVS2 ],
               couplings = {(0,0):C.GC_1729})

V_517 = Vertex(name = 'V_517',
               particles = [ P.a, P.a, P.W__minus__, P.W__minus__, P.W__plus__, P.W__plus__ ],
               color = [ '1' ],
               lorentz = [ L.VVVVVV121, L.VVVVVV123 ],
               couplings = {(0,1):C.GC_359,(0,0):C.GC_402})

V_518 = Vertex(name = 'V_518',
               particles = [ P.W__minus__, P.W__minus__, P.W__plus__, P.W__plus__, P.Z ],
               color = [ '1' ],
               lorentz = [ L.VVVVV36, L.VVVVV77, L.VVVVV86 ],
               couplings = {(0,0):C.GC_992,(0,1):C.GC_349,(0,2):C.GC_360})

V_519 = Vertex(name = 'V_519',
               particles = [ P.W__minus__, P.W__minus__, P.W__minus__, P.W__plus__, P.W__plus__, P.W__plus__ ],
               color = [ '1' ],
               lorentz = [ L.VVVVVV148 ],
               couplings = {(0,0):C.GC_352})

V_520 = Vertex(name = 'V_520',
               particles = [ P.W__minus__, P.W__minus__, P.W__minus__, P.W__plus__, P.W__plus__, P.W__plus__ ],
               color = [ '1' ],
               lorentz = [ L.VVVVVV148 ],
               couplings = {(0,0):C.GC_1728})

V_521 = Vertex(name = 'V_521',
               particles = [ P.e__plus__, P.ve, P.G__minus__ ],
               color = [ '1' ],
               lorentz = [ L.FFS2, L.FFS4 ],
               couplings = {(0,1):C.GC_1952,(0,0):C.GC_1377})

V_522 = Vertex(name = 'V_522',
               particles = [ P.e__plus__, P.ve, P.G__minus__ ],
               color = [ '1' ],
               lorentz = [ L.FFS4 ],
               couplings = {(0,0):C.GC_1957})

V_523 = Vertex(name = 'V_523',
               particles = [ P.mu__plus__, P.vm, P.G__minus__ ],
               color = [ '1' ],
               lorentz = [ L.FFS2, L.FFS4 ],
               couplings = {(0,1):C.GC_2024,(0,0):C.GC_1377})

V_524 = Vertex(name = 'V_524',
               particles = [ P.mu__plus__, P.vm, P.G__minus__ ],
               color = [ '1' ],
               lorentz = [ L.FFS4 ],
               couplings = {(0,0):C.GC_2029})

V_525 = Vertex(name = 'V_525',
               particles = [ P.ta__plus__, P.vt, P.G__minus__ ],
               color = [ '1' ],
               lorentz = [ L.FFS2, L.FFS4 ],
               couplings = {(0,1):C.GC_2195,(0,0):C.GC_1377})

V_526 = Vertex(name = 'V_526',
               particles = [ P.ta__plus__, P.vt, P.G__minus__ ],
               color = [ '1' ],
               lorentz = [ L.FFS4 ],
               couplings = {(0,0):C.GC_2200})

V_527 = Vertex(name = 'V_527',
               particles = [ P.e__plus__, P.ve, P.G0, P.G0, P.G__minus__ ],
               color = [ '1' ],
               lorentz = [ L.FFSSS3 ],
               couplings = {(0,0):C.GC_1985})

V_528 = Vertex(name = 'V_528',
               particles = [ P.mu__plus__, P.vm, P.G0, P.G0, P.G__minus__ ],
               color = [ '1' ],
               lorentz = [ L.FFSSS3 ],
               couplings = {(0,0):C.GC_2059})

V_529 = Vertex(name = 'V_529',
               particles = [ P.ta__plus__, P.vt, P.G0, P.G0, P.G__minus__ ],
               color = [ '1' ],
               lorentz = [ L.FFSSS3 ],
               couplings = {(0,0):C.GC_2228})

V_530 = Vertex(name = 'V_530',
               particles = [ P.e__plus__, P.ve, P.G__minus__, P.G__minus__, P.G__plus__ ],
               color = [ '1' ],
               lorentz = [ L.FFSSS3 ],
               couplings = {(0,0):C.GC_1984})

V_531 = Vertex(name = 'V_531',
               particles = [ P.mu__plus__, P.vm, P.G__minus__, P.G__minus__, P.G__plus__ ],
               color = [ '1' ],
               lorentz = [ L.FFSSS3 ],
               couplings = {(0,0):C.GC_2058})

V_532 = Vertex(name = 'V_532',
               particles = [ P.ta__plus__, P.vt, P.G__minus__, P.G__minus__, P.G__plus__ ],
               color = [ '1' ],
               lorentz = [ L.FFSSS3 ],
               couplings = {(0,0):C.GC_2227})

V_533 = Vertex(name = 'V_533',
               particles = [ P.e__plus__, P.ve, P.G__minus__, P.H, P.H ],
               color = [ '1' ],
               lorentz = [ L.FFSSS3 ],
               couplings = {(0,0):C.GC_1985})

V_534 = Vertex(name = 'V_534',
               particles = [ P.mu__plus__, P.vm, P.G__minus__, P.H, P.H ],
               color = [ '1' ],
               lorentz = [ L.FFSSS3 ],
               couplings = {(0,0):C.GC_2059})

V_535 = Vertex(name = 'V_535',
               particles = [ P.ta__plus__, P.vt, P.G__minus__, P.H, P.H ],
               color = [ '1' ],
               lorentz = [ L.FFSSS3 ],
               couplings = {(0,0):C.GC_2228})

V_536 = Vertex(name = 'V_536',
               particles = [ P.e__plus__, P.ve, P.G__minus__, P.H ],
               color = [ '1' ],
               lorentz = [ L.FFSS4, L.FFSS6 ],
               couplings = {(0,1):C.GC_1994,(0,0):C.GC_1174})

V_537 = Vertex(name = 'V_537',
               particles = [ P.mu__plus__, P.vm, P.G__minus__, P.H ],
               color = [ '1' ],
               lorentz = [ L.FFSS4, L.FFSS6 ],
               couplings = {(0,1):C.GC_2068,(0,0):C.GC_1174})

V_538 = Vertex(name = 'V_538',
               particles = [ P.ta__plus__, P.vt, P.G__minus__, P.H ],
               color = [ '1' ],
               lorentz = [ L.FFSS4, L.FFSS6 ],
               couplings = {(0,1):C.GC_2237,(0,0):C.GC_1174})

V_539 = Vertex(name = 'V_539',
               particles = [ P.a, P.Z, P.G0, P.G0, P.G__minus__, P.G__plus__ ],
               color = [ '1' ],
               lorentz = [ L.VVSSSS1 ],
               couplings = {(0,0):C.GC_1115})

V_540 = Vertex(name = 'V_540',
               particles = [ P.a, P.Z, P.G__minus__, P.G__minus__, P.G__plus__, P.G__plus__ ],
               color = [ '1' ],
               lorentz = [ L.VVSSSS1 ],
               couplings = {(0,0):C.GC_1116})

V_541 = Vertex(name = 'V_541',
               particles = [ P.a, P.Z, P.G__minus__, P.G__plus__, P.H, P.H ],
               color = [ '1' ],
               lorentz = [ L.VVSSSS1 ],
               couplings = {(0,0):C.GC_1115})

V_542 = Vertex(name = 'V_542',
               particles = [ P.a, P.Z, P.G__minus__, P.G__plus__, P.H ],
               color = [ '1' ],
               lorentz = [ L.VVSSS1 ],
               couplings = {(0,0):C.GC_1145})

V_543 = Vertex(name = 'V_543',
               particles = [ P.Z, P.G0, P.G0, P.G0, P.H ],
               color = [ '1' ],
               lorentz = [ L.VSSSS4 ],
               couplings = {(0,0):C.GC_1113})

V_544 = Vertex(name = 'V_544',
               particles = [ P.Z, P.G0, P.G__minus__, P.G__plus__, P.H ],
               color = [ '1' ],
               lorentz = [ L.VSSSS7 ],
               couplings = {(0,0):C.GC_1112})

V_545 = Vertex(name = 'V_545',
               particles = [ P.Z, P.G0, P.H, P.H, P.H ],
               color = [ '1' ],
               lorentz = [ L.VSSSS12 ],
               couplings = {(0,0):C.GC_1114})

V_546 = Vertex(name = 'V_546',
               particles = [ P.Z, P.G0, P.G0, P.G0 ],
               color = [ '1' ],
               lorentz = [ L.VSSS8 ],
               couplings = {(0,0):C.GC_1143})

V_547 = Vertex(name = 'V_547',
               particles = [ P.Z, P.G0, P.G__minus__, P.G__plus__ ],
               color = [ '1' ],
               lorentz = [ L.VSSS1 ],
               couplings = {(0,0):C.GC_1142})

V_548 = Vertex(name = 'V_548',
               particles = [ P.Z, P.G0, P.H, P.H ],
               color = [ '1' ],
               lorentz = [ L.VSSS7 ],
               couplings = {(0,0):C.GC_1144})

V_549 = Vertex(name = 'V_549',
               particles = [ P.Z, P.G0, P.G0, P.G__minus__, P.G__plus__ ],
               color = [ '1' ],
               lorentz = [ L.VSSSS9 ],
               couplings = {(0,0):C.GC_1110})

V_550 = Vertex(name = 'V_550',
               particles = [ P.Z, P.G__minus__, P.G__minus__, P.G__plus__, P.G__plus__ ],
               color = [ '1' ],
               lorentz = [ L.VSSSS8 ],
               couplings = {(0,0):C.GC_1111})

V_551 = Vertex(name = 'V_551',
               particles = [ P.Z, P.G__minus__, P.G__plus__, P.H, P.H ],
               color = [ '1' ],
               lorentz = [ L.VSSSS1 ],
               couplings = {(0,0):C.GC_1110})

V_552 = Vertex(name = 'V_552',
               particles = [ P.Z, P.G__minus__, P.G__plus__, P.H ],
               color = [ '1' ],
               lorentz = [ L.VSSS2 ],
               couplings = {(0,0):C.GC_1141})

V_553 = Vertex(name = 'V_553',
               particles = [ P.W__minus__, P.Z, P.G0, P.G0, P.G0, P.G__plus__ ],
               color = [ '1' ],
               lorentz = [ L.VVSSSS1 ],
               couplings = {(0,0):C.GC_1092})

V_554 = Vertex(name = 'V_554',
               particles = [ P.W__minus__, P.Z, P.G0, P.G__minus__, P.G__plus__, P.G__plus__ ],
               color = [ '1' ],
               lorentz = [ L.VVSSSS1 ],
               couplings = {(0,0):C.GC_1091})

V_555 = Vertex(name = 'V_555',
               particles = [ P.W__minus__, P.Z, P.G0, P.G0, P.G__plus__, P.H ],
               color = [ '1' ],
               lorentz = [ L.VVSSSS1 ],
               couplings = {(0,0):C.GC_1089})

V_556 = Vertex(name = 'V_556',
               particles = [ P.W__minus__, P.Z, P.G__minus__, P.G__plus__, P.G__plus__, P.H ],
               color = [ '1' ],
               lorentz = [ L.VVSSSS1 ],
               couplings = {(0,0):C.GC_1094})

V_557 = Vertex(name = 'V_557',
               particles = [ P.W__minus__, P.Z, P.G0, P.G__plus__, P.H, P.H ],
               color = [ '1' ],
               lorentz = [ L.VVSSSS1 ],
               couplings = {(0,0):C.GC_1090})

V_558 = Vertex(name = 'V_558',
               particles = [ P.W__minus__, P.Z, P.G__plus__, P.H, P.H, P.H ],
               color = [ '1' ],
               lorentz = [ L.VVSSSS1 ],
               couplings = {(0,0):C.GC_1087})

V_559 = Vertex(name = 'V_559',
               particles = [ P.W__minus__, P.Z, P.G0, P.G0, P.G__plus__ ],
               color = [ '1' ],
               lorentz = [ L.VVSSS1 ],
               couplings = {(0,0):C.GC_1129})

V_560 = Vertex(name = 'V_560',
               particles = [ P.W__minus__, P.Z, P.G__minus__, P.G__plus__, P.G__plus__ ],
               color = [ '1' ],
               lorentz = [ L.VVSSS1 ],
               couplings = {(0,0):C.GC_1132})

V_561 = Vertex(name = 'V_561',
               particles = [ P.W__minus__, P.Z, P.G0, P.G__plus__, P.H ],
               color = [ '1' ],
               lorentz = [ L.VVSSS1 ],
               couplings = {(0,0):C.GC_1130})

V_562 = Vertex(name = 'V_562',
               particles = [ P.W__minus__, P.Z, P.G__plus__, P.H, P.H ],
               color = [ '1' ],
               lorentz = [ L.VVSSS1 ],
               couplings = {(0,0):C.GC_1127})

V_563 = Vertex(name = 'V_563',
               particles = [ P.a, P.a, P.W__minus__, P.Z, P.G0, P.G__plus__ ],
               color = [ '1' ],
               lorentz = [ L.VVVVSS2 ],
               couplings = {(0,0):C.GC_341})

V_564 = Vertex(name = 'V_564',
               particles = [ P.a, P.a, P.W__minus__, P.Z, P.G__plus__, P.H ],
               color = [ '1' ],
               lorentz = [ L.VVVVSS2 ],
               couplings = {(0,0):C.GC_340})

V_565 = Vertex(name = 'V_565',
               particles = [ P.a, P.a, P.W__minus__, P.Z, P.G__plus__ ],
               color = [ '1' ],
               lorentz = [ L.VVVVS2 ],
               couplings = {(0,0):C.GC_1745})

V_566 = Vertex(name = 'V_566',
               particles = [ P.W__minus__, P.Z, P.Z, P.G0, P.G__plus__ ],
               color = [ '1' ],
               lorentz = [ L.VVVSS15, L.VVVSS17, L.VVVSS34, L.VVVSS35, L.VVVSS4 ],
               couplings = {(0,4):C.GC_1059,(0,0):C.GC_608,(0,1):C.GC_596,(0,3):C.GC_599,(0,2):C.GC_325})

V_567 = Vertex(name = 'V_567',
               particles = [ P.W__minus__, P.Z, P.Z, P.G__plus__, P.H ],
               color = [ '1' ],
               lorentz = [ L.VVVSS15, L.VVVSS17, L.VVVSS34, L.VVVSS35, L.VVVSS4 ],
               couplings = {(0,4):C.GC_1058,(0,0):C.GC_607,(0,1):C.GC_595,(0,3):C.GC_598,(0,2):C.GC_327})

V_568 = Vertex(name = 'V_568',
               particles = [ P.W__minus__, P.Z, P.Z, P.G__plus__ ],
               color = [ '1' ],
               lorentz = [ L.VVVS14, L.VVVS16, L.VVVS19, L.VVVS25, L.VVVS4 ],
               couplings = {(0,4):C.GC_1763,(0,0):C.GC_1571,(0,1):C.GC_1566,(0,3):C.GC_1567,(0,2):C.GC_1734})

V_569 = Vertex(name = 'V_569',
               particles = [ P.W__plus__, P.Z, P.G0, P.G0, P.G0, P.G__minus__ ],
               color = [ '1' ],
               lorentz = [ L.VVSSSS1 ],
               couplings = {(0,0):C.GC_1092})

V_570 = Vertex(name = 'V_570',
               particles = [ P.W__plus__, P.Z, P.G0, P.G__minus__, P.G__minus__, P.G__plus__ ],
               color = [ '1' ],
               lorentz = [ L.VVSSSS1 ],
               couplings = {(0,0):C.GC_1091})

V_571 = Vertex(name = 'V_571',
               particles = [ P.W__plus__, P.Z, P.G0, P.G0, P.G__minus__, P.H ],
               color = [ '1' ],
               lorentz = [ L.VVSSSS1 ],
               couplings = {(0,0):C.GC_1093})

V_572 = Vertex(name = 'V_572',
               particles = [ P.W__plus__, P.Z, P.G__minus__, P.G__minus__, P.G__plus__, P.H ],
               color = [ '1' ],
               lorentz = [ L.VVSSSS1 ],
               couplings = {(0,0):C.GC_1088})

V_573 = Vertex(name = 'V_573',
               particles = [ P.W__plus__, P.Z, P.G0, P.G__minus__, P.H, P.H ],
               color = [ '1' ],
               lorentz = [ L.VVSSSS1 ],
               couplings = {(0,0):C.GC_1090})

V_574 = Vertex(name = 'V_574',
               particles = [ P.W__plus__, P.Z, P.G__minus__, P.H, P.H, P.H ],
               color = [ '1' ],
               lorentz = [ L.VVSSSS1 ],
               couplings = {(0,0):C.GC_1095})

V_575 = Vertex(name = 'V_575',
               particles = [ P.W__plus__, P.Z, P.G0, P.G0, P.G__minus__ ],
               color = [ '1' ],
               lorentz = [ L.VVSSS1 ],
               couplings = {(0,0):C.GC_1131})

V_576 = Vertex(name = 'V_576',
               particles = [ P.W__plus__, P.Z, P.G__minus__, P.G__minus__, P.G__plus__ ],
               color = [ '1' ],
               lorentz = [ L.VVSSS1 ],
               couplings = {(0,0):C.GC_1128})

V_577 = Vertex(name = 'V_577',
               particles = [ P.W__plus__, P.Z, P.G0, P.G__minus__, P.H ],
               color = [ '1' ],
               lorentz = [ L.VVSSS1 ],
               couplings = {(0,0):C.GC_1130})

V_578 = Vertex(name = 'V_578',
               particles = [ P.W__plus__, P.Z, P.G__minus__, P.H, P.H ],
               color = [ '1' ],
               lorentz = [ L.VVSSS1 ],
               couplings = {(0,0):C.GC_1133})

V_579 = Vertex(name = 'V_579',
               particles = [ P.a, P.a, P.W__plus__, P.Z, P.G0, P.G__minus__ ],
               color = [ '1' ],
               lorentz = [ L.VVVVSS2 ],
               couplings = {(0,0):C.GC_341})

V_580 = Vertex(name = 'V_580',
               particles = [ P.a, P.a, P.W__plus__, P.Z, P.G__minus__, P.H ],
               color = [ '1' ],
               lorentz = [ L.VVVVSS2 ],
               couplings = {(0,0):C.GC_342})

V_581 = Vertex(name = 'V_581',
               particles = [ P.a, P.a, P.W__plus__, P.Z, P.G__minus__ ],
               color = [ '1' ],
               lorentz = [ L.VVVVS2 ],
               couplings = {(0,0):C.GC_1746})

V_582 = Vertex(name = 'V_582',
               particles = [ P.W__plus__, P.Z, P.Z, P.G0, P.G__minus__ ],
               color = [ '1' ],
               lorentz = [ L.VVVSS15, L.VVVSS17, L.VVVSS34, L.VVVSS35, L.VVVSS4 ],
               couplings = {(0,4):C.GC_1060,(0,0):C.GC_609,(0,1):C.GC_597,(0,3):C.GC_600,(0,2):C.GC_326})

V_583 = Vertex(name = 'V_583',
               particles = [ P.W__plus__, P.Z, P.Z, P.G__minus__, P.H ],
               color = [ '1' ],
               lorentz = [ L.VVVSS15, L.VVVSS17, L.VVVSS34, L.VVVSS35, L.VVVSS4 ],
               couplings = {(0,4):C.GC_1058,(0,0):C.GC_607,(0,1):C.GC_595,(0,3):C.GC_598,(0,2):C.GC_327})

V_584 = Vertex(name = 'V_584',
               particles = [ P.W__plus__, P.Z, P.Z, P.G__minus__ ],
               color = [ '1' ],
               lorentz = [ L.VVVS14, L.VVVS16, L.VVVS19, L.VVVS25, L.VVVS4 ],
               couplings = {(0,4):C.GC_1763,(0,0):C.GC_1571,(0,1):C.GC_1566,(0,3):C.GC_1567,(0,2):C.GC_1734})

V_585 = Vertex(name = 'V_585',
               particles = [ P.a, P.W__minus__, P.W__plus__, P.Z, P.G0, P.G0 ],
               color = [ '1' ],
               lorentz = [ L.VVVVSS5 ],
               couplings = {(0,0):C.GC_347})

V_586 = Vertex(name = 'V_586',
               particles = [ P.a, P.W__minus__, P.W__plus__, P.Z, P.G__minus__, P.G__plus__ ],
               color = [ '1' ],
               lorentz = [ L.VVVVSS5 ],
               couplings = {(0,0):C.GC_346})

V_587 = Vertex(name = 'V_587',
               particles = [ P.a, P.W__minus__, P.W__plus__, P.Z, P.H, P.H ],
               color = [ '1' ],
               lorentz = [ L.VVVVSS5 ],
               couplings = {(0,0):C.GC_347})

V_588 = Vertex(name = 'V_588',
               particles = [ P.a, P.W__minus__, P.W__plus__, P.Z, P.H ],
               color = [ '1' ],
               lorentz = [ L.VVVVS5 ],
               couplings = {(0,0):C.GC_1752})

V_589 = Vertex(name = 'V_589',
               particles = [ P.W__minus__, P.W__plus__, P.Z, P.G0, P.H ],
               color = [ '1' ],
               lorentz = [ L.VVVSS28 ],
               couplings = {(0,0):C.GC_323})

V_590 = Vertex(name = 'V_590',
               particles = [ P.W__minus__, P.W__plus__, P.Z, P.G0 ],
               color = [ '1' ],
               lorentz = [ L.VVVS21 ],
               couplings = {(0,0):C.GC_1732})

V_591 = Vertex(name = 'V_591',
               particles = [ P.W__minus__, P.W__minus__, P.W__plus__, P.Z, P.G0, P.G__plus__ ],
               color = [ '1' ],
               lorentz = [ L.VVVVSS2 ],
               couplings = {(0,0):C.GC_336})

V_592 = Vertex(name = 'V_592',
               particles = [ P.W__minus__, P.W__minus__, P.W__plus__, P.Z, P.G__plus__, P.H ],
               color = [ '1' ],
               lorentz = [ L.VVVVSS2 ],
               couplings = {(0,0):C.GC_338})

V_593 = Vertex(name = 'V_593',
               particles = [ P.W__minus__, P.W__minus__, P.W__plus__, P.Z, P.G__plus__ ],
               color = [ '1' ],
               lorentz = [ L.VVVVS2 ],
               couplings = {(0,0):C.GC_1743})

V_594 = Vertex(name = 'V_594',
               particles = [ P.a, P.a, P.a, P.W__minus__, P.W__plus__, P.Z ],
               color = [ '1' ],
               lorentz = [ L.VVVVVV161 ],
               couplings = {(0,0):C.GC_614})

V_595 = Vertex(name = 'V_595',
               particles = [ P.a, P.W__minus__, P.W__plus__, P.Z, P.Z ],
               color = [ '1' ],
               lorentz = [ L.VVVVV47, L.VVVVV63, L.VVVVV72 ],
               couplings = {(0,0):C.GC_997,(0,2):C.GC_397,(0,1):C.GC_358})

V_596 = Vertex(name = 'V_596',
               particles = [ P.W__minus__, P.W__plus__, P.W__plus__, P.Z, P.G0, P.G__minus__ ],
               color = [ '1' ],
               lorentz = [ L.VVVVSS5 ],
               couplings = {(0,0):C.GC_337})

V_597 = Vertex(name = 'V_597',
               particles = [ P.W__minus__, P.W__plus__, P.W__plus__, P.Z, P.G__minus__, P.H ],
               color = [ '1' ],
               lorentz = [ L.VVVVSS5 ],
               couplings = {(0,0):C.GC_339})

V_598 = Vertex(name = 'V_598',
               particles = [ P.W__minus__, P.W__plus__, P.W__plus__, P.Z, P.G__minus__ ],
               color = [ '1' ],
               lorentz = [ L.VVVVS5 ],
               couplings = {(0,0):C.GC_1744})

V_599 = Vertex(name = 'V_599',
               particles = [ P.a, P.W__minus__, P.W__minus__, P.W__plus__, P.W__plus__, P.Z ],
               color = [ '1' ],
               lorentz = [ L.VVVVVV149, L.VVVVVV150 ],
               couplings = {(0,0):C.GC_351,(0,1):C.GC_365})

V_600 = Vertex(name = 'V_600',
               particles = [ P.Z, P.Z, P.G0, P.G0, P.G0, P.G0 ],
               color = [ '1' ],
               lorentz = [ L.VVSSSS1 ],
               couplings = {(0,0):C.GC_1120})

V_601 = Vertex(name = 'V_601',
               particles = [ P.Z, P.Z, P.G__minus__, P.G__minus__, P.G__plus__, P.G__plus__ ],
               color = [ '1' ],
               lorentz = [ L.VVSSSS1 ],
               couplings = {(0,0):C.GC_1119})

V_602 = Vertex(name = 'V_602',
               particles = [ P.Z, P.Z, P.G0, P.G0, P.H, P.H ],
               color = [ '1' ],
               lorentz = [ L.VVSSSS1 ],
               couplings = {(0,0):C.GC_1117})

V_603 = Vertex(name = 'V_603',
               particles = [ P.Z, P.Z, P.H, P.H, P.H, P.H ],
               color = [ '1' ],
               lorentz = [ L.VVSSSS1 ],
               couplings = {(0,0):C.GC_1120})

V_604 = Vertex(name = 'V_604',
               particles = [ P.Z, P.Z, P.G0, P.G0, P.G__minus__, P.G__plus__ ],
               color = [ '1' ],
               lorentz = [ L.VVSSSS1 ],
               couplings = {(0,0):C.GC_1118})

V_605 = Vertex(name = 'V_605',
               particles = [ P.Z, P.Z, P.G__minus__, P.G__plus__, P.H, P.H ],
               color = [ '1' ],
               lorentz = [ L.VVSSSS1 ],
               couplings = {(0,0):C.GC_1118})

V_606 = Vertex(name = 'V_606',
               particles = [ P.Z, P.Z, P.G0, P.G0, P.H ],
               color = [ '1' ],
               lorentz = [ L.VVSSS1 ],
               couplings = {(0,0):C.GC_1146})

V_607 = Vertex(name = 'V_607',
               particles = [ P.Z, P.Z, P.H, P.H, P.H ],
               color = [ '1' ],
               lorentz = [ L.VVSSS1 ],
               couplings = {(0,0):C.GC_1148})

V_608 = Vertex(name = 'V_608',
               particles = [ P.Z, P.Z, P.G__minus__, P.G__plus__, P.H ],
               color = [ '1' ],
               lorentz = [ L.VVSSS1 ],
               couplings = {(0,0):C.GC_1147})

V_609 = Vertex(name = 'V_609',
               particles = [ P.a, P.W__minus__, P.Z, P.Z, P.G0, P.G__plus__ ],
               color = [ '1' ],
               lorentz = [ L.VVVVSS2 ],
               couplings = {(0,0):C.GC_344})

V_610 = Vertex(name = 'V_610',
               particles = [ P.a, P.W__minus__, P.Z, P.Z, P.G__plus__, P.H ],
               color = [ '1' ],
               lorentz = [ L.VVVVSS2 ],
               couplings = {(0,0):C.GC_345})

V_611 = Vertex(name = 'V_611',
               particles = [ P.a, P.W__minus__, P.Z, P.Z, P.G__plus__ ],
               color = [ '1' ],
               lorentz = [ L.VVVVS2 ],
               couplings = {(0,0):C.GC_1751})

V_612 = Vertex(name = 'V_612',
               particles = [ P.a, P.W__plus__, P.Z, P.Z, P.G0, P.G__minus__ ],
               color = [ '1' ],
               lorentz = [ L.VVVVSS2 ],
               couplings = {(0,0):C.GC_344})

V_613 = Vertex(name = 'V_613',
               particles = [ P.a, P.W__plus__, P.Z, P.Z, P.G__minus__, P.H ],
               color = [ '1' ],
               lorentz = [ L.VVVVSS2 ],
               couplings = {(0,0):C.GC_343})

V_614 = Vertex(name = 'V_614',
               particles = [ P.a, P.W__plus__, P.Z, P.Z, P.G__minus__ ],
               color = [ '1' ],
               lorentz = [ L.VVVVS2 ],
               couplings = {(0,0):C.GC_1750})

V_615 = Vertex(name = 'V_615',
               particles = [ P.W__minus__, P.W__plus__, P.Z, P.Z, P.G0, P.G0 ],
               color = [ '1' ],
               lorentz = [ L.VVVVSS2 ],
               couplings = {(0,0):C.GC_333})

V_616 = Vertex(name = 'V_616',
               particles = [ P.W__minus__, P.W__plus__, P.Z, P.Z, P.G__minus__, P.G__plus__ ],
               color = [ '1' ],
               lorentz = [ L.VVVVSS2 ],
               couplings = {(0,0):C.GC_334})

V_617 = Vertex(name = 'V_617',
               particles = [ P.W__minus__, P.W__plus__, P.Z, P.Z, P.H, P.H ],
               color = [ '1' ],
               lorentz = [ L.VVVVSS2 ],
               couplings = {(0,0):C.GC_333})

V_618 = Vertex(name = 'V_618',
               particles = [ P.W__minus__, P.W__plus__, P.Z, P.Z, P.H ],
               color = [ '1' ],
               lorentz = [ L.VVVVS2 ],
               couplings = {(0,0):C.GC_1742})

V_619 = Vertex(name = 'V_619',
               particles = [ P.a, P.a, P.W__minus__, P.W__plus__, P.Z, P.Z ],
               color = [ '1' ],
               lorentz = [ L.VVVVVV159 ],
               couplings = {(0,0):C.GC_404})

V_620 = Vertex(name = 'V_620',
               particles = [ P.W__minus__, P.W__minus__, P.W__plus__, P.W__plus__, P.Z, P.Z ],
               color = [ '1' ],
               lorentz = [ L.VVVVVV122, L.VVVVVV123 ],
               couplings = {(0,1):C.GC_348,(0,0):C.GC_354})

V_621 = Vertex(name = 'V_621',
               particles = [ P.W__minus__, P.W__plus__, P.Z, P.Z, P.Z ],
               color = [ '1' ],
               lorentz = [ L.VVVVV48, L.VVVVV81, L.VVVVV91 ],
               couplings = {(0,0):C.GC_993,(0,1):C.GC_363,(0,2):C.GC_350})

V_622 = Vertex(name = 'V_622',
               particles = [ P.a, P.W__minus__, P.W__plus__, P.Z, P.Z, P.Z ],
               color = [ '1' ],
               lorentz = [ L.VVVVVV160 ],
               couplings = {(0,0):C.GC_367})

V_623 = Vertex(name = 'V_623',
               particles = [ P.W__minus__, P.W__plus__, P.Z, P.Z, P.Z, P.Z ],
               color = [ '1' ],
               lorentz = [ L.VVVVVV120 ],
               couplings = {(0,0):C.GC_356})

V_624 = Vertex(name = 'V_624',
               particles = [ P.d__tilde__, P.d, P.a ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFV10, L.FFV11, L.FFV6 ],
               couplings = {(0,2):C.GC_2,(0,1):C.GC_1935,(0,0):C.GC_1934})

V_625 = Vertex(name = 'V_625',
               particles = [ P.d__tilde__, P.d, P.a ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFV6 ],
               couplings = {(0,0):C.GC_1705})

V_626 = Vertex(name = 'V_626',
               particles = [ P.s__tilde__, P.s, P.a ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFV10, L.FFV6 ],
               couplings = {(0,1):C.GC_2,(0,0):C.GC_2148})

V_627 = Vertex(name = 'V_627',
               particles = [ P.s__tilde__, P.s, P.a ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFV6 ],
               couplings = {(0,0):C.GC_1705})

V_628 = Vertex(name = 'V_628',
               particles = [ P.b__tilde__, P.b, P.a ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFV10, L.FFV6 ],
               couplings = {(0,1):C.GC_2,(0,0):C.GC_1856})

V_629 = Vertex(name = 'V_629',
               particles = [ P.b__tilde__, P.b, P.a ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFV6 ],
               couplings = {(0,0):C.GC_1705})

V_630 = Vertex(name = 'V_630',
               particles = [ P.d__tilde__, P.d, P.a, P.G__minus__, P.G__plus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFVSS1, L.FFVSS2 ],
               couplings = {(0,0):C.GC_1081,(0,1):C.GC_1198})

V_631 = Vertex(name = 'V_631',
               particles = [ P.s__tilde__, P.s, P.a, P.G__minus__, P.G__plus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFVSS1, L.FFVSS2 ],
               couplings = {(0,0):C.GC_1081,(0,1):C.GC_1198})

V_632 = Vertex(name = 'V_632',
               particles = [ P.b__tilde__, P.b, P.a, P.G__minus__, P.G__plus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFVSS1, L.FFVSS2 ],
               couplings = {(0,0):C.GC_1081,(0,1):C.GC_1198})

V_633 = Vertex(name = 'V_633',
               particles = [ P.u__tilde__, P.d, P.a, P.G0, P.G__plus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFVSS1, L.FFVSS2 ],
               couplings = {(0,0):C.GC_2316,(0,1):C.GC_1202})

V_634 = Vertex(name = 'V_634',
               particles = [ P.c__tilde__, P.d, P.a, P.G0, P.G__plus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFVSS1 ],
               couplings = {(0,0):C.GC_2358})

V_635 = Vertex(name = 'V_635',
               particles = [ P.t__tilde__, P.d, P.a, P.G0, P.G__plus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFVSS1 ],
               couplings = {(0,0):C.GC_2400})

V_636 = Vertex(name = 'V_636',
               particles = [ P.u__tilde__, P.s, P.a, P.G0, P.G__plus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFVSS1 ],
               couplings = {(0,0):C.GC_2330})

V_637 = Vertex(name = 'V_637',
               particles = [ P.c__tilde__, P.s, P.a, P.G0, P.G__plus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFVSS1, L.FFVSS2 ],
               couplings = {(0,0):C.GC_2372,(0,1):C.GC_1202})

V_638 = Vertex(name = 'V_638',
               particles = [ P.t__tilde__, P.s, P.a, P.G0, P.G__plus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFVSS1 ],
               couplings = {(0,0):C.GC_2414})

V_639 = Vertex(name = 'V_639',
               particles = [ P.u__tilde__, P.b, P.a, P.G0, P.G__plus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFVSS1 ],
               couplings = {(0,0):C.GC_2344})

V_640 = Vertex(name = 'V_640',
               particles = [ P.c__tilde__, P.b, P.a, P.G0, P.G__plus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFVSS1 ],
               couplings = {(0,0):C.GC_2386})

V_641 = Vertex(name = 'V_641',
               particles = [ P.t__tilde__, P.b, P.a, P.G0, P.G__plus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFVSS1, L.FFVSS2 ],
               couplings = {(0,0):C.GC_2428,(0,1):C.GC_1202})

V_642 = Vertex(name = 'V_642',
               particles = [ P.u__tilde__, P.d, P.a, P.G__plus__, P.H ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFVSS1, L.FFVSS2 ],
               couplings = {(0,0):C.GC_2315,(0,1):C.GC_1201})

V_643 = Vertex(name = 'V_643',
               particles = [ P.c__tilde__, P.d, P.a, P.G__plus__, P.H ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFVSS1 ],
               couplings = {(0,0):C.GC_2357})

V_644 = Vertex(name = 'V_644',
               particles = [ P.t__tilde__, P.d, P.a, P.G__plus__, P.H ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFVSS1 ],
               couplings = {(0,0):C.GC_2399})

V_645 = Vertex(name = 'V_645',
               particles = [ P.u__tilde__, P.s, P.a, P.G__plus__, P.H ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFVSS1 ],
               couplings = {(0,0):C.GC_2329})

V_646 = Vertex(name = 'V_646',
               particles = [ P.c__tilde__, P.s, P.a, P.G__plus__, P.H ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFVSS1, L.FFVSS2 ],
               couplings = {(0,0):C.GC_2371,(0,1):C.GC_1201})

V_647 = Vertex(name = 'V_647',
               particles = [ P.t__tilde__, P.s, P.a, P.G__plus__, P.H ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFVSS1 ],
               couplings = {(0,0):C.GC_2413})

V_648 = Vertex(name = 'V_648',
               particles = [ P.u__tilde__, P.b, P.a, P.G__plus__, P.H ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFVSS1 ],
               couplings = {(0,0):C.GC_2343})

V_649 = Vertex(name = 'V_649',
               particles = [ P.c__tilde__, P.b, P.a, P.G__plus__, P.H ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFVSS1 ],
               couplings = {(0,0):C.GC_2385})

V_650 = Vertex(name = 'V_650',
               particles = [ P.t__tilde__, P.b, P.a, P.G__plus__, P.H ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFVSS1, L.FFVSS2 ],
               couplings = {(0,0):C.GC_2427,(0,1):C.GC_1201})

V_651 = Vertex(name = 'V_651',
               particles = [ P.u__tilde__, P.d, P.a, P.G__plus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFVS1, L.FFVS10, L.FFVS4, L.FFVS7 ],
               couplings = {(0,2):C.GC_95,(0,1):C.GC_104,(0,0):C.GC_2321,(0,3):C.GC_1389})

V_652 = Vertex(name = 'V_652',
               particles = [ P.c__tilde__, P.d, P.a, P.G__plus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFVS1, L.FFVS10, L.FFVS4 ],
               couplings = {(0,2):C.GC_96,(0,1):C.GC_105,(0,0):C.GC_2363})

V_653 = Vertex(name = 'V_653',
               particles = [ P.t__tilde__, P.d, P.a, P.G__plus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFVS1, L.FFVS10, L.FFVS4 ],
               couplings = {(0,2):C.GC_97,(0,1):C.GC_106,(0,0):C.GC_2405})

V_654 = Vertex(name = 'V_654',
               particles = [ P.u__tilde__, P.s, P.a, P.G__plus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFVS1, L.FFVS10, L.FFVS4 ],
               couplings = {(0,2):C.GC_98,(0,1):C.GC_107,(0,0):C.GC_2335})

V_655 = Vertex(name = 'V_655',
               particles = [ P.c__tilde__, P.s, P.a, P.G__plus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFVS1, L.FFVS10, L.FFVS4, L.FFVS7 ],
               couplings = {(0,2):C.GC_99,(0,1):C.GC_108,(0,0):C.GC_2377,(0,3):C.GC_1389})

V_656 = Vertex(name = 'V_656',
               particles = [ P.t__tilde__, P.s, P.a, P.G__plus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFVS1, L.FFVS10, L.FFVS4 ],
               couplings = {(0,2):C.GC_100,(0,1):C.GC_109,(0,0):C.GC_2419})

V_657 = Vertex(name = 'V_657',
               particles = [ P.u__tilde__, P.b, P.a, P.G__plus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFVS1, L.FFVS10, L.FFVS4 ],
               couplings = {(0,2):C.GC_101,(0,1):C.GC_110,(0,0):C.GC_2349})

V_658 = Vertex(name = 'V_658',
               particles = [ P.c__tilde__, P.b, P.a, P.G__plus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFVS1, L.FFVS10, L.FFVS4 ],
               couplings = {(0,2):C.GC_102,(0,1):C.GC_111,(0,0):C.GC_2391})

V_659 = Vertex(name = 'V_659',
               particles = [ P.t__tilde__, P.b, P.a, P.G__plus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFVS1, L.FFVS10, L.FFVS4, L.FFVS7 ],
               couplings = {(0,2):C.GC_103,(0,1):C.GC_112,(0,0):C.GC_2433,(0,3):C.GC_1389})

V_660 = Vertex(name = 'V_660',
               particles = [ P.u__tilde__, P.d, P.G0, P.G__plus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFSS10, L.FFSS4 ],
               couplings = {(0,1):C.GC_2313,(0,0):C.GC_1171})

V_661 = Vertex(name = 'V_661',
               particles = [ P.c__tilde__, P.d, P.G0, P.G__plus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFSS4 ],
               couplings = {(0,0):C.GC_2355})

V_662 = Vertex(name = 'V_662',
               particles = [ P.t__tilde__, P.d, P.G0, P.G__plus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFSS4 ],
               couplings = {(0,0):C.GC_2397})

V_663 = Vertex(name = 'V_663',
               particles = [ P.u__tilde__, P.s, P.G0, P.G__plus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFSS4 ],
               couplings = {(0,0):C.GC_2327})

V_664 = Vertex(name = 'V_664',
               particles = [ P.c__tilde__, P.s, P.G0, P.G__plus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFSS10, L.FFSS4 ],
               couplings = {(0,1):C.GC_2369,(0,0):C.GC_1171})

V_665 = Vertex(name = 'V_665',
               particles = [ P.t__tilde__, P.s, P.G0, P.G__plus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFSS4 ],
               couplings = {(0,0):C.GC_2411})

V_666 = Vertex(name = 'V_666',
               particles = [ P.u__tilde__, P.b, P.G0, P.G__plus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFSS4 ],
               couplings = {(0,0):C.GC_2341})

V_667 = Vertex(name = 'V_667',
               particles = [ P.c__tilde__, P.b, P.G0, P.G__plus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFSS4 ],
               couplings = {(0,0):C.GC_2383})

V_668 = Vertex(name = 'V_668',
               particles = [ P.t__tilde__, P.b, P.G0, P.G__plus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFSS10, L.FFSS4 ],
               couplings = {(0,1):C.GC_2425,(0,0):C.GC_1171})

V_669 = Vertex(name = 'V_669',
               particles = [ P.e__plus__, P.e__minus__, P.a ],
               color = [ '1' ],
               lorentz = [ L.FFV10, L.FFV6 ],
               couplings = {(0,1):C.GC_4,(0,0):C.GC_2021})

V_670 = Vertex(name = 'V_670',
               particles = [ P.e__plus__, P.e__minus__, P.a ],
               color = [ '1' ],
               lorentz = [ L.FFV6 ],
               couplings = {(0,0):C.GC_1707})

V_671 = Vertex(name = 'V_671',
               particles = [ P.mu__plus__, P.mu__minus__, P.a ],
               color = [ '1' ],
               lorentz = [ L.FFV10, L.FFV11, L.FFV6 ],
               couplings = {(0,2):C.GC_4,(0,1):C.GC_2075,(0,0):C.GC_2074})

V_672 = Vertex(name = 'V_672',
               particles = [ P.mu__plus__, P.mu__minus__, P.a ],
               color = [ '1' ],
               lorentz = [ L.FFV6 ],
               couplings = {(0,0):C.GC_1707})

V_673 = Vertex(name = 'V_673',
               particles = [ P.ta__plus__, P.ta__minus__, P.a ],
               color = [ '1' ],
               lorentz = [ L.FFV10, L.FFV6 ],
               couplings = {(0,1):C.GC_4,(0,0):C.GC_2264})

V_674 = Vertex(name = 'V_674',
               particles = [ P.ta__plus__, P.ta__minus__, P.a ],
               color = [ '1' ],
               lorentz = [ L.FFV6 ],
               couplings = {(0,0):C.GC_1707})

V_675 = Vertex(name = 'V_675',
               particles = [ P.e__plus__, P.e__minus__, P.a, P.G__minus__, P.G__plus__ ],
               color = [ '1' ],
               lorentz = [ L.FFVSS1, L.FFVSS2 ],
               couplings = {(0,0):C.GC_1079,(0,1):C.GC_1199})

V_676 = Vertex(name = 'V_676',
               particles = [ P.mu__plus__, P.mu__minus__, P.a, P.G__minus__, P.G__plus__ ],
               color = [ '1' ],
               lorentz = [ L.FFVSS1, L.FFVSS2 ],
               couplings = {(0,0):C.GC_1079,(0,1):C.GC_1199})

V_677 = Vertex(name = 'V_677',
               particles = [ P.ta__plus__, P.ta__minus__, P.a, P.G__minus__, P.G__plus__ ],
               color = [ '1' ],
               lorentz = [ L.FFVSS1, L.FFVSS2 ],
               couplings = {(0,0):C.GC_1079,(0,1):C.GC_1199})

V_678 = Vertex(name = 'V_678',
               particles = [ P.ve__tilde__, P.e__minus__, P.a, P.G0, P.G__plus__ ],
               color = [ '1' ],
               lorentz = [ L.FFVSS1 ],
               couplings = {(0,0):C.GC_1205})

V_679 = Vertex(name = 'V_679',
               particles = [ P.vm__tilde__, P.mu__minus__, P.a, P.G0, P.G__plus__ ],
               color = [ '1' ],
               lorentz = [ L.FFVSS1 ],
               couplings = {(0,0):C.GC_1205})

V_680 = Vertex(name = 'V_680',
               particles = [ P.vt__tilde__, P.ta__minus__, P.a, P.G0, P.G__plus__ ],
               color = [ '1' ],
               lorentz = [ L.FFVSS1 ],
               couplings = {(0,0):C.GC_1205})

V_681 = Vertex(name = 'V_681',
               particles = [ P.ve__tilde__, P.e__minus__, P.a, P.G__plus__, P.H ],
               color = [ '1' ],
               lorentz = [ L.FFVSS1 ],
               couplings = {(0,0):C.GC_1204})

V_682 = Vertex(name = 'V_682',
               particles = [ P.vm__tilde__, P.mu__minus__, P.a, P.G__plus__, P.H ],
               color = [ '1' ],
               lorentz = [ L.FFVSS1 ],
               couplings = {(0,0):C.GC_1204})

V_683 = Vertex(name = 'V_683',
               particles = [ P.vt__tilde__, P.ta__minus__, P.a, P.G__plus__, P.H ],
               color = [ '1' ],
               lorentz = [ L.FFVSS1 ],
               couplings = {(0,0):C.GC_1204})

V_684 = Vertex(name = 'V_684',
               particles = [ P.ve__tilde__, P.e__minus__, P.a, P.G__plus__ ],
               color = [ '1' ],
               lorentz = [ L.FFVS1, L.FFVS4 ],
               couplings = {(0,1):C.GC_2009,(0,0):C.GC_1391})

V_685 = Vertex(name = 'V_685',
               particles = [ P.vm__tilde__, P.mu__minus__, P.a, P.G__plus__ ],
               color = [ '1' ],
               lorentz = [ L.FFVS1, L.FFVS4 ],
               couplings = {(0,1):C.GC_2085,(0,0):C.GC_1391})

V_686 = Vertex(name = 'V_686',
               particles = [ P.vt__tilde__, P.ta__minus__, P.a, P.G__plus__ ],
               color = [ '1' ],
               lorentz = [ L.FFVS1, L.FFVS4 ],
               couplings = {(0,1):C.GC_2252,(0,0):C.GC_1391})

V_687 = Vertex(name = 'V_687',
               particles = [ P.ve__tilde__, P.e__minus__, P.G0, P.G__plus__ ],
               color = [ '1' ],
               lorentz = [ L.FFSS4 ],
               couplings = {(0,0):C.GC_1176})

V_688 = Vertex(name = 'V_688',
               particles = [ P.vm__tilde__, P.mu__minus__, P.G0, P.G__plus__ ],
               color = [ '1' ],
               lorentz = [ L.FFSS4 ],
               couplings = {(0,0):C.GC_1176})

V_689 = Vertex(name = 'V_689',
               particles = [ P.vt__tilde__, P.ta__minus__, P.G0, P.G__plus__ ],
               color = [ '1' ],
               lorentz = [ L.FFSS4 ],
               couplings = {(0,0):C.GC_1176})

V_690 = Vertex(name = 'V_690',
               particles = [ P.d__tilde__, P.u, P.a, P.G0, P.G__minus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFVSS1, L.FFVSS2 ],
               couplings = {(0,0):C.GC_1207,(0,1):C.GC_1202})

V_691 = Vertex(name = 'V_691',
               particles = [ P.s__tilde__, P.u, P.a, P.G0, P.G__minus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFVSS1 ],
               couplings = {(0,0):C.GC_1209})

V_692 = Vertex(name = 'V_692',
               particles = [ P.b__tilde__, P.u, P.a, P.G0, P.G__minus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFVSS1 ],
               couplings = {(0,0):C.GC_1211})

V_693 = Vertex(name = 'V_693',
               particles = [ P.d__tilde__, P.c, P.a, P.G0, P.G__minus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFVSS1 ],
               couplings = {(0,0):C.GC_1213})

V_694 = Vertex(name = 'V_694',
               particles = [ P.s__tilde__, P.c, P.a, P.G0, P.G__minus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFVSS1, L.FFVSS2 ],
               couplings = {(0,0):C.GC_1215,(0,1):C.GC_1202})

V_695 = Vertex(name = 'V_695',
               particles = [ P.b__tilde__, P.c, P.a, P.G0, P.G__minus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFVSS1 ],
               couplings = {(0,0):C.GC_1217})

V_696 = Vertex(name = 'V_696',
               particles = [ P.d__tilde__, P.t, P.a, P.G0, P.G__minus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFVSS1 ],
               couplings = {(0,0):C.GC_1219})

V_697 = Vertex(name = 'V_697',
               particles = [ P.s__tilde__, P.t, P.a, P.G0, P.G__minus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFVSS1 ],
               couplings = {(0,0):C.GC_1221})

V_698 = Vertex(name = 'V_698',
               particles = [ P.b__tilde__, P.t, P.a, P.G0, P.G__minus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFVSS1, L.FFVSS2 ],
               couplings = {(0,0):C.GC_1223,(0,1):C.GC_1202})

V_699 = Vertex(name = 'V_699',
               particles = [ P.d__tilde__, P.u, P.a, P.G__minus__, P.H ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFVSS1, L.FFVSS2 ],
               couplings = {(0,0):C.GC_1208,(0,1):C.GC_1203})

V_700 = Vertex(name = 'V_700',
               particles = [ P.s__tilde__, P.u, P.a, P.G__minus__, P.H ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFVSS1 ],
               couplings = {(0,0):C.GC_1210})

V_701 = Vertex(name = 'V_701',
               particles = [ P.b__tilde__, P.u, P.a, P.G__minus__, P.H ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFVSS1 ],
               couplings = {(0,0):C.GC_1212})

V_702 = Vertex(name = 'V_702',
               particles = [ P.d__tilde__, P.c, P.a, P.G__minus__, P.H ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFVSS1 ],
               couplings = {(0,0):C.GC_1214})

V_703 = Vertex(name = 'V_703',
               particles = [ P.s__tilde__, P.c, P.a, P.G__minus__, P.H ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFVSS1, L.FFVSS2 ],
               couplings = {(0,0):C.GC_1216,(0,1):C.GC_1203})

V_704 = Vertex(name = 'V_704',
               particles = [ P.b__tilde__, P.c, P.a, P.G__minus__, P.H ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFVSS1 ],
               couplings = {(0,0):C.GC_1218})

V_705 = Vertex(name = 'V_705',
               particles = [ P.d__tilde__, P.t, P.a, P.G__minus__, P.H ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFVSS1 ],
               couplings = {(0,0):C.GC_1220})

V_706 = Vertex(name = 'V_706',
               particles = [ P.s__tilde__, P.t, P.a, P.G__minus__, P.H ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFVSS1 ],
               couplings = {(0,0):C.GC_1222})

V_707 = Vertex(name = 'V_707',
               particles = [ P.b__tilde__, P.t, P.a, P.G__minus__, P.H ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFVSS1, L.FFVSS2 ],
               couplings = {(0,0):C.GC_1224,(0,1):C.GC_1203})

V_708 = Vertex(name = 'V_708',
               particles = [ P.d__tilde__, P.u, P.a, P.G__minus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFVS1, L.FFVS10, L.FFVS4, L.FFVS7 ],
               couplings = {(0,2):C.GC_113,(0,1):C.GC_122,(0,0):C.GC_1393,(0,3):C.GC_1390})

V_709 = Vertex(name = 'V_709',
               particles = [ P.s__tilde__, P.u, P.a, P.G__minus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFVS1, L.FFVS10, L.FFVS4 ],
               couplings = {(0,2):C.GC_114,(0,1):C.GC_123,(0,0):C.GC_1394})

V_710 = Vertex(name = 'V_710',
               particles = [ P.b__tilde__, P.u, P.a, P.G__minus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFVS1, L.FFVS10, L.FFVS4 ],
               couplings = {(0,2):C.GC_115,(0,1):C.GC_124,(0,0):C.GC_1395})

V_711 = Vertex(name = 'V_711',
               particles = [ P.d__tilde__, P.c, P.a, P.G__minus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFVS1, L.FFVS10, L.FFVS4 ],
               couplings = {(0,2):C.GC_116,(0,1):C.GC_125,(0,0):C.GC_1396})

V_712 = Vertex(name = 'V_712',
               particles = [ P.s__tilde__, P.c, P.a, P.G__minus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFVS1, L.FFVS10, L.FFVS4, L.FFVS7 ],
               couplings = {(0,2):C.GC_117,(0,1):C.GC_126,(0,0):C.GC_1397,(0,3):C.GC_1390})

V_713 = Vertex(name = 'V_713',
               particles = [ P.b__tilde__, P.c, P.a, P.G__minus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFVS1, L.FFVS10, L.FFVS4 ],
               couplings = {(0,2):C.GC_118,(0,1):C.GC_127,(0,0):C.GC_1398})

V_714 = Vertex(name = 'V_714',
               particles = [ P.d__tilde__, P.t, P.a, P.G__minus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFVS1, L.FFVS10, L.FFVS4 ],
               couplings = {(0,2):C.GC_119,(0,1):C.GC_128,(0,0):C.GC_1399})

V_715 = Vertex(name = 'V_715',
               particles = [ P.s__tilde__, P.t, P.a, P.G__minus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFVS1, L.FFVS10, L.FFVS4 ],
               couplings = {(0,2):C.GC_120,(0,1):C.GC_129,(0,0):C.GC_1400})

V_716 = Vertex(name = 'V_716',
               particles = [ P.b__tilde__, P.t, P.a, P.G__minus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFVS1, L.FFVS10, L.FFVS4, L.FFVS7 ],
               couplings = {(0,2):C.GC_121,(0,1):C.GC_130,(0,0):C.GC_1401,(0,3):C.GC_1390})

V_717 = Vertex(name = 'V_717',
               particles = [ P.d__tilde__, P.u, P.G0, P.G__minus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFSS10, L.FFSS4 ],
               couplings = {(0,1):C.GC_1178,(0,0):C.GC_1172})

V_718 = Vertex(name = 'V_718',
               particles = [ P.s__tilde__, P.u, P.G0, P.G__minus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFSS4 ],
               couplings = {(0,0):C.GC_1180})

V_719 = Vertex(name = 'V_719',
               particles = [ P.b__tilde__, P.u, P.G0, P.G__minus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFSS4 ],
               couplings = {(0,0):C.GC_1182})

V_720 = Vertex(name = 'V_720',
               particles = [ P.d__tilde__, P.c, P.G0, P.G__minus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFSS4 ],
               couplings = {(0,0):C.GC_1184})

V_721 = Vertex(name = 'V_721',
               particles = [ P.s__tilde__, P.c, P.G0, P.G__minus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFSS10, L.FFSS4 ],
               couplings = {(0,1):C.GC_1186,(0,0):C.GC_1172})

V_722 = Vertex(name = 'V_722',
               particles = [ P.b__tilde__, P.c, P.G0, P.G__minus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFSS4 ],
               couplings = {(0,0):C.GC_1188})

V_723 = Vertex(name = 'V_723',
               particles = [ P.d__tilde__, P.t, P.G0, P.G__minus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFSS4 ],
               couplings = {(0,0):C.GC_1190})

V_724 = Vertex(name = 'V_724',
               particles = [ P.s__tilde__, P.t, P.G0, P.G__minus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFSS4 ],
               couplings = {(0,0):C.GC_1192})

V_725 = Vertex(name = 'V_725',
               particles = [ P.b__tilde__, P.t, P.G0, P.G__minus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFSS10, L.FFSS4 ],
               couplings = {(0,1):C.GC_1194,(0,0):C.GC_1172})

V_726 = Vertex(name = 'V_726',
               particles = [ P.u__tilde__, P.u, P.a ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFV10, L.FFV6 ],
               couplings = {(0,1):C.GC_3,(0,0):C.GC_2308})

V_727 = Vertex(name = 'V_727',
               particles = [ P.u__tilde__, P.u, P.a ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFV6 ],
               couplings = {(0,0):C.GC_1706})

V_728 = Vertex(name = 'V_728',
               particles = [ P.c__tilde__, P.c, P.a ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFV10, L.FFV6 ],
               couplings = {(0,1):C.GC_3,(0,0):C.GC_1900})

V_729 = Vertex(name = 'V_729',
               particles = [ P.c__tilde__, P.c, P.a ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFV6 ],
               couplings = {(0,0):C.GC_1706})

V_730 = Vertex(name = 'V_730',
               particles = [ P.t__tilde__, P.t, P.a ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFV10, L.FFV6 ],
               couplings = {(0,1):C.GC_3,(0,0):C.GC_2192})

V_731 = Vertex(name = 'V_731',
               particles = [ P.t__tilde__, P.t, P.a ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFV6 ],
               couplings = {(0,0):C.GC_1706})

V_732 = Vertex(name = 'V_732',
               particles = [ P.u__tilde__, P.u, P.a, P.G__minus__, P.G__plus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFVSS1, L.FFVSS2 ],
               couplings = {(0,0):C.GC_1082,(0,1):C.GC_1200})

V_733 = Vertex(name = 'V_733',
               particles = [ P.c__tilde__, P.c, P.a, P.G__minus__, P.G__plus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFVSS1, L.FFVSS2 ],
               couplings = {(0,0):C.GC_1082,(0,1):C.GC_1200})

V_734 = Vertex(name = 'V_734',
               particles = [ P.t__tilde__, P.t, P.a, P.G__minus__, P.G__plus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFVSS1, L.FFVSS2 ],
               couplings = {(0,0):C.GC_1082,(0,1):C.GC_1200})

V_735 = Vertex(name = 'V_735',
               particles = [ P.e__plus__, P.ve, P.a, P.G0, P.G__minus__ ],
               color = [ '1' ],
               lorentz = [ L.FFVSS1 ],
               couplings = {(0,0):C.GC_1205})

V_736 = Vertex(name = 'V_736',
               particles = [ P.mu__plus__, P.vm, P.a, P.G0, P.G__minus__ ],
               color = [ '1' ],
               lorentz = [ L.FFVSS1 ],
               couplings = {(0,0):C.GC_1205})

V_737 = Vertex(name = 'V_737',
               particles = [ P.ta__plus__, P.vt, P.a, P.G0, P.G__minus__ ],
               color = [ '1' ],
               lorentz = [ L.FFVSS1 ],
               couplings = {(0,0):C.GC_1205})

V_738 = Vertex(name = 'V_738',
               particles = [ P.e__plus__, P.ve, P.a, P.G__minus__, P.H ],
               color = [ '1' ],
               lorentz = [ L.FFVSS1 ],
               couplings = {(0,0):C.GC_1206})

V_739 = Vertex(name = 'V_739',
               particles = [ P.mu__plus__, P.vm, P.a, P.G__minus__, P.H ],
               color = [ '1' ],
               lorentz = [ L.FFVSS1 ],
               couplings = {(0,0):C.GC_1206})

V_740 = Vertex(name = 'V_740',
               particles = [ P.ta__plus__, P.vt, P.a, P.G__minus__, P.H ],
               color = [ '1' ],
               lorentz = [ L.FFVSS1 ],
               couplings = {(0,0):C.GC_1206})

V_741 = Vertex(name = 'V_741',
               particles = [ P.e__plus__, P.ve, P.a, P.G__minus__ ],
               color = [ '1' ],
               lorentz = [ L.FFVS1, L.FFVS10 ],
               couplings = {(0,1):C.GC_2010,(0,0):C.GC_1392})

V_742 = Vertex(name = 'V_742',
               particles = [ P.mu__plus__, P.vm, P.a, P.G__minus__ ],
               color = [ '1' ],
               lorentz = [ L.FFVS1, L.FFVS10 ],
               couplings = {(0,1):C.GC_2086,(0,0):C.GC_1392})

V_743 = Vertex(name = 'V_743',
               particles = [ P.ta__plus__, P.vt, P.a, P.G__minus__ ],
               color = [ '1' ],
               lorentz = [ L.FFVS1, L.FFVS10 ],
               couplings = {(0,1):C.GC_2253,(0,0):C.GC_1392})

V_744 = Vertex(name = 'V_744',
               particles = [ P.e__plus__, P.ve, P.G0, P.G__minus__ ],
               color = [ '1' ],
               lorentz = [ L.FFSS4 ],
               couplings = {(0,0):C.GC_1175})

V_745 = Vertex(name = 'V_745',
               particles = [ P.mu__plus__, P.vm, P.G0, P.G__minus__ ],
               color = [ '1' ],
               lorentz = [ L.FFSS4 ],
               couplings = {(0,0):C.GC_1175})

V_746 = Vertex(name = 'V_746',
               particles = [ P.ta__plus__, P.vt, P.G0, P.G__minus__ ],
               color = [ '1' ],
               lorentz = [ L.FFSS4 ],
               couplings = {(0,0):C.GC_1175})

V_747 = Vertex(name = 'V_747',
               particles = [ P.ve__tilde__, P.ve, P.a, P.G__minus__, P.G__plus__ ],
               color = [ '1' ],
               lorentz = [ L.FFVSS1 ],
               couplings = {(0,0):C.GC_1080})

V_748 = Vertex(name = 'V_748',
               particles = [ P.vm__tilde__, P.vm, P.a, P.G__minus__, P.G__plus__ ],
               color = [ '1' ],
               lorentz = [ L.FFVSS1 ],
               couplings = {(0,0):C.GC_1080})

V_749 = Vertex(name = 'V_749',
               particles = [ P.vt__tilde__, P.vt, P.a, P.G__minus__, P.G__plus__ ],
               color = [ '1' ],
               lorentz = [ L.FFVSS1 ],
               couplings = {(0,0):C.GC_1080})

V_750 = Vertex(name = 'V_750',
               particles = [ P.ve__tilde__, P.ve, P.G0, P.H ],
               color = [ '1' ],
               lorentz = [ L.FFSS4 ],
               couplings = {(0,0):C.GC_1068})

V_751 = Vertex(name = 'V_751',
               particles = [ P.vm__tilde__, P.vm, P.G0, P.H ],
               color = [ '1' ],
               lorentz = [ L.FFSS4 ],
               couplings = {(0,0):C.GC_1068})

V_752 = Vertex(name = 'V_752',
               particles = [ P.vt__tilde__, P.vt, P.G0, P.H ],
               color = [ '1' ],
               lorentz = [ L.FFSS4 ],
               couplings = {(0,0):C.GC_1068})

V_753 = Vertex(name = 'V_753',
               particles = [ P.ve__tilde__, P.ve, P.G0 ],
               color = [ '1' ],
               lorentz = [ L.FFS2 ],
               couplings = {(0,0):C.GC_1121})

V_754 = Vertex(name = 'V_754',
               particles = [ P.vm__tilde__, P.vm, P.G0 ],
               color = [ '1' ],
               lorentz = [ L.FFS2 ],
               couplings = {(0,0):C.GC_1121})

V_755 = Vertex(name = 'V_755',
               particles = [ P.vt__tilde__, P.vt, P.G0 ],
               color = [ '1' ],
               lorentz = [ L.FFS2 ],
               couplings = {(0,0):C.GC_1121})

V_756 = Vertex(name = 'V_756',
               particles = [ P.ve__tilde__, P.ve, P.G__minus__, P.G__plus__ ],
               color = [ '1' ],
               lorentz = [ L.FFSS4 ],
               couplings = {(0,0):C.GC_1069})

V_757 = Vertex(name = 'V_757',
               particles = [ P.vm__tilde__, P.vm, P.G__minus__, P.G__plus__ ],
               color = [ '1' ],
               lorentz = [ L.FFSS4 ],
               couplings = {(0,0):C.GC_1069})

V_758 = Vertex(name = 'V_758',
               particles = [ P.vt__tilde__, P.vt, P.G__minus__, P.G__plus__ ],
               color = [ '1' ],
               lorentz = [ L.FFSS4 ],
               couplings = {(0,0):C.GC_1069})

V_759 = Vertex(name = 'V_759',
               particles = [ P.d__tilde__, P.d, P.g ],
               color = [ 'T(3,2,1)' ],
               lorentz = [ L.FFV10, L.FFV6 ],
               couplings = {(0,1):C.GC_12,(0,0):C.GC_1936})

V_760 = Vertex(name = 'V_760',
               particles = [ P.s__tilde__, P.s, P.g ],
               color = [ 'T(3,2,1)' ],
               lorentz = [ L.FFV10, L.FFV6 ],
               couplings = {(0,1):C.GC_12,(0,0):C.GC_2134})

V_761 = Vertex(name = 'V_761',
               particles = [ P.b__tilde__, P.b, P.g ],
               color = [ 'T(3,2,1)' ],
               lorentz = [ L.FFV10, L.FFV6 ],
               couplings = {(0,1):C.GC_12,(0,0):C.GC_1842})

V_762 = Vertex(name = 'V_762',
               particles = [ P.u__tilde__, P.u, P.g ],
               color = [ 'T(3,2,1)' ],
               lorentz = [ L.FFV10, L.FFV6 ],
               couplings = {(0,1):C.GC_12,(0,0):C.GC_2294})

V_763 = Vertex(name = 'V_763',
               particles = [ P.c__tilde__, P.c, P.g ],
               color = [ 'T(3,2,1)' ],
               lorentz = [ L.FFV10, L.FFV6 ],
               couplings = {(0,1):C.GC_12,(0,0):C.GC_1886})

V_764 = Vertex(name = 'V_764',
               particles = [ P.t__tilde__, P.t, P.g ],
               color = [ 'T(3,2,1)' ],
               lorentz = [ L.FFV10, L.FFV6 ],
               couplings = {(0,1):C.GC_12,(0,0):C.GC_2178})

V_765 = Vertex(name = 'V_765',
               particles = [ P.d__tilde__, P.d, P.W__minus__, P.G0, P.G__plus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFVSS1, L.FFVSS2 ],
               couplings = {(0,0):C.GC_1319,(0,1):C.GC_1310})

V_766 = Vertex(name = 'V_766',
               particles = [ P.s__tilde__, P.s, P.W__minus__, P.G0, P.G__plus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFVSS1, L.FFVSS2 ],
               couplings = {(0,0):C.GC_1319,(0,1):C.GC_1310})

V_767 = Vertex(name = 'V_767',
               particles = [ P.b__tilde__, P.b, P.W__minus__, P.G0, P.G__plus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFVSS1, L.FFVSS2 ],
               couplings = {(0,0):C.GC_1319,(0,1):C.GC_1310})

V_768 = Vertex(name = 'V_768',
               particles = [ P.d__tilde__, P.d, P.W__minus__, P.G__plus__, P.H ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFVSS1, L.FFVSS2 ],
               couplings = {(0,0):C.GC_1318,(0,1):C.GC_1309})

V_769 = Vertex(name = 'V_769',
               particles = [ P.s__tilde__, P.s, P.W__minus__, P.G__plus__, P.H ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFVSS1, L.FFVSS2 ],
               couplings = {(0,0):C.GC_1318,(0,1):C.GC_1309})

V_770 = Vertex(name = 'V_770',
               particles = [ P.b__tilde__, P.b, P.W__minus__, P.G__plus__, P.H ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFVSS1, L.FFVSS2 ],
               couplings = {(0,0):C.GC_1318,(0,1):C.GC_1309})

V_771 = Vertex(name = 'V_771',
               particles = [ P.d__tilde__, P.d, P.W__minus__, P.G__plus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFVS1, L.FFVS4, L.FFVS7 ],
               couplings = {(0,0):C.GC_1451,(0,2):C.GC_1445,(0,1):C.GC_1920})

V_772 = Vertex(name = 'V_772',
               particles = [ P.s__tilde__, P.s, P.W__minus__, P.G__plus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFVS1, L.FFVS4, L.FFVS7 ],
               couplings = {(0,0):C.GC_1451,(0,2):C.GC_1445,(0,1):C.GC_2118})

V_773 = Vertex(name = 'V_773',
               particles = [ P.b__tilde__, P.b, P.W__minus__, P.G__plus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFVS1, L.FFVS4, L.FFVS7 ],
               couplings = {(0,0):C.GC_1451,(0,2):C.GC_1445,(0,1):C.GC_1828})

V_774 = Vertex(name = 'V_774',
               particles = [ P.e__plus__, P.e__minus__, P.W__minus__, P.G0, P.G__plus__ ],
               color = [ '1' ],
               lorentz = [ L.FFVSS1, L.FFVSS2 ],
               couplings = {(0,0):C.GC_1316,(0,1):C.GC_1313})

V_775 = Vertex(name = 'V_775',
               particles = [ P.mu__plus__, P.mu__minus__, P.W__minus__, P.G0, P.G__plus__ ],
               color = [ '1' ],
               lorentz = [ L.FFVSS1, L.FFVSS2 ],
               couplings = {(0,0):C.GC_1316,(0,1):C.GC_1313})

V_776 = Vertex(name = 'V_776',
               particles = [ P.ta__plus__, P.ta__minus__, P.W__minus__, P.G0, P.G__plus__ ],
               color = [ '1' ],
               lorentz = [ L.FFVSS1, L.FFVSS2 ],
               couplings = {(0,0):C.GC_1316,(0,1):C.GC_1313})

V_777 = Vertex(name = 'V_777',
               particles = [ P.e__plus__, P.e__minus__, P.W__minus__, P.G__plus__, P.H ],
               color = [ '1' ],
               lorentz = [ L.FFVSS1, L.FFVSS2 ],
               couplings = {(0,0):C.GC_1315,(0,1):C.GC_1312})

V_778 = Vertex(name = 'V_778',
               particles = [ P.mu__plus__, P.mu__minus__, P.W__minus__, P.G__plus__, P.H ],
               color = [ '1' ],
               lorentz = [ L.FFVSS1, L.FFVSS2 ],
               couplings = {(0,0):C.GC_1315,(0,1):C.GC_1312})

V_779 = Vertex(name = 'V_779',
               particles = [ P.ta__plus__, P.ta__minus__, P.W__minus__, P.G__plus__, P.H ],
               color = [ '1' ],
               lorentz = [ L.FFVSS1, L.FFVSS2 ],
               couplings = {(0,0):C.GC_1315,(0,1):C.GC_1312})

V_780 = Vertex(name = 'V_780',
               particles = [ P.e__plus__, P.e__minus__, P.W__minus__, P.G__plus__ ],
               color = [ '1' ],
               lorentz = [ L.FFVS1, L.FFVS4, L.FFVS7 ],
               couplings = {(0,0):C.GC_1449,(0,2):C.GC_1447,(0,1):C.GC_1976})

V_781 = Vertex(name = 'V_781',
               particles = [ P.mu__plus__, P.mu__minus__, P.W__minus__, P.G__plus__ ],
               color = [ '1' ],
               lorentz = [ L.FFVS1, L.FFVS4, L.FFVS7 ],
               couplings = {(0,0):C.GC_1449,(0,2):C.GC_1447,(0,1):C.GC_2048})

V_782 = Vertex(name = 'V_782',
               particles = [ P.ta__plus__, P.ta__minus__, P.W__minus__, P.G__plus__ ],
               color = [ '1' ],
               lorentz = [ L.FFVS1, L.FFVS4, L.FFVS7 ],
               couplings = {(0,0):C.GC_1449,(0,2):C.GC_1447,(0,1):C.GC_2219})

V_783 = Vertex(name = 'V_783',
               particles = [ P.d__tilde__, P.u, P.W__minus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFV1, L.FFV4, L.FFV5, L.FFV9 ],
               couplings = {(0,1):C.GC_1627,(0,3):C.GC_1654,(0,0):C.GC_558,(0,2):C.GC_557})

V_784 = Vertex(name = 'V_784',
               particles = [ P.d__tilde__, P.u, P.W__minus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFV1 ],
               couplings = {(0,0):C.GC_1770})

V_785 = Vertex(name = 'V_785',
               particles = [ P.s__tilde__, P.u, P.W__minus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFV1, L.FFV4, L.FFV9 ],
               couplings = {(0,1):C.GC_1630,(0,2):C.GC_1657,(0,0):C.GC_559})

V_786 = Vertex(name = 'V_786',
               particles = [ P.s__tilde__, P.u, P.W__minus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFV1 ],
               couplings = {(0,0):C.GC_1771})

V_787 = Vertex(name = 'V_787',
               particles = [ P.b__tilde__, P.u, P.W__minus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFV1, L.FFV4, L.FFV9 ],
               couplings = {(0,1):C.GC_1633,(0,2):C.GC_1660,(0,0):C.GC_560})

V_788 = Vertex(name = 'V_788',
               particles = [ P.b__tilde__, P.u, P.W__minus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFV1 ],
               couplings = {(0,0):C.GC_1772})

V_789 = Vertex(name = 'V_789',
               particles = [ P.d__tilde__, P.c, P.W__minus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFV1, L.FFV4, L.FFV9 ],
               couplings = {(0,1):C.GC_1636,(0,2):C.GC_1663,(0,0):C.GC_561})

V_790 = Vertex(name = 'V_790',
               particles = [ P.d__tilde__, P.c, P.W__minus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFV1 ],
               couplings = {(0,0):C.GC_1773})

V_791 = Vertex(name = 'V_791',
               particles = [ P.s__tilde__, P.c, P.W__minus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFV1, L.FFV12, L.FFV4, L.FFV5 ],
               couplings = {(0,1):C.GC_1666,(0,2):C.GC_1639,(0,0):C.GC_562,(0,3):C.GC_557})

V_792 = Vertex(name = 'V_792',
               particles = [ P.s__tilde__, P.c, P.W__minus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFV1 ],
               couplings = {(0,0):C.GC_1774})

V_793 = Vertex(name = 'V_793',
               particles = [ P.b__tilde__, P.c, P.W__minus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFV1, L.FFV4, L.FFV9 ],
               couplings = {(0,1):C.GC_1642,(0,2):C.GC_1669,(0,0):C.GC_563})

V_794 = Vertex(name = 'V_794',
               particles = [ P.b__tilde__, P.c, P.W__minus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFV1 ],
               couplings = {(0,0):C.GC_1775})

V_795 = Vertex(name = 'V_795',
               particles = [ P.d__tilde__, P.t, P.W__minus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFV1, L.FFV4, L.FFV9 ],
               couplings = {(0,1):C.GC_1645,(0,2):C.GC_1672,(0,0):C.GC_564})

V_796 = Vertex(name = 'V_796',
               particles = [ P.d__tilde__, P.t, P.W__minus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFV1 ],
               couplings = {(0,0):C.GC_1776})

V_797 = Vertex(name = 'V_797',
               particles = [ P.s__tilde__, P.t, P.W__minus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFV1, L.FFV4, L.FFV9 ],
               couplings = {(0,1):C.GC_1648,(0,2):C.GC_1675,(0,0):C.GC_565})

V_798 = Vertex(name = 'V_798',
               particles = [ P.s__tilde__, P.t, P.W__minus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFV1 ],
               couplings = {(0,0):C.GC_1777})

V_799 = Vertex(name = 'V_799',
               particles = [ P.b__tilde__, P.t, P.W__minus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFV1, L.FFV4, L.FFV5, L.FFV9 ],
               couplings = {(0,1):C.GC_1651,(0,3):C.GC_1678,(0,0):C.GC_566,(0,2):C.GC_557})

V_800 = Vertex(name = 'V_800',
               particles = [ P.b__tilde__, P.t, P.W__minus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFV1 ],
               couplings = {(0,0):C.GC_1778})

V_801 = Vertex(name = 'V_801',
               particles = [ P.d__tilde__, P.u, P.W__minus__, P.G0, P.G0 ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFVSS1, L.FFVSS2 ],
               couplings = {(0,0):C.GC_1330,(0,1):C.GC_1325})

V_802 = Vertex(name = 'V_802',
               particles = [ P.s__tilde__, P.u, P.W__minus__, P.G0, P.G0 ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFVSS1 ],
               couplings = {(0,0):C.GC_1331})

V_803 = Vertex(name = 'V_803',
               particles = [ P.b__tilde__, P.u, P.W__minus__, P.G0, P.G0 ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFVSS1 ],
               couplings = {(0,0):C.GC_1332})

V_804 = Vertex(name = 'V_804',
               particles = [ P.d__tilde__, P.c, P.W__minus__, P.G0, P.G0 ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFVSS1 ],
               couplings = {(0,0):C.GC_1333})

V_805 = Vertex(name = 'V_805',
               particles = [ P.s__tilde__, P.c, P.W__minus__, P.G0, P.G0 ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFVSS1, L.FFVSS2 ],
               couplings = {(0,0):C.GC_1334,(0,1):C.GC_1325})

V_806 = Vertex(name = 'V_806',
               particles = [ P.b__tilde__, P.c, P.W__minus__, P.G0, P.G0 ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFVSS1 ],
               couplings = {(0,0):C.GC_1335})

V_807 = Vertex(name = 'V_807',
               particles = [ P.d__tilde__, P.t, P.W__minus__, P.G0, P.G0 ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFVSS1 ],
               couplings = {(0,0):C.GC_1336})

V_808 = Vertex(name = 'V_808',
               particles = [ P.s__tilde__, P.t, P.W__minus__, P.G0, P.G0 ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFVSS1 ],
               couplings = {(0,0):C.GC_1337})

V_809 = Vertex(name = 'V_809',
               particles = [ P.b__tilde__, P.t, P.W__minus__, P.G0, P.G0 ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFVSS1, L.FFVSS2 ],
               couplings = {(0,0):C.GC_1338,(0,1):C.GC_1325})

V_810 = Vertex(name = 'V_810',
               particles = [ P.d__tilde__, P.u, P.W__minus__, P.G__minus__, P.G__plus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFVSS1 ],
               couplings = {(0,0):C.GC_1330})

V_811 = Vertex(name = 'V_811',
               particles = [ P.s__tilde__, P.u, P.W__minus__, P.G__minus__, P.G__plus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFVSS1 ],
               couplings = {(0,0):C.GC_1331})

V_812 = Vertex(name = 'V_812',
               particles = [ P.b__tilde__, P.u, P.W__minus__, P.G__minus__, P.G__plus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFVSS1 ],
               couplings = {(0,0):C.GC_1332})

V_813 = Vertex(name = 'V_813',
               particles = [ P.d__tilde__, P.c, P.W__minus__, P.G__minus__, P.G__plus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFVSS1 ],
               couplings = {(0,0):C.GC_1333})

V_814 = Vertex(name = 'V_814',
               particles = [ P.s__tilde__, P.c, P.W__minus__, P.G__minus__, P.G__plus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFVSS1 ],
               couplings = {(0,0):C.GC_1334})

V_815 = Vertex(name = 'V_815',
               particles = [ P.b__tilde__, P.c, P.W__minus__, P.G__minus__, P.G__plus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFVSS1 ],
               couplings = {(0,0):C.GC_1335})

V_816 = Vertex(name = 'V_816',
               particles = [ P.d__tilde__, P.t, P.W__minus__, P.G__minus__, P.G__plus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFVSS1 ],
               couplings = {(0,0):C.GC_1336})

V_817 = Vertex(name = 'V_817',
               particles = [ P.s__tilde__, P.t, P.W__minus__, P.G__minus__, P.G__plus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFVSS1 ],
               couplings = {(0,0):C.GC_1337})

V_818 = Vertex(name = 'V_818',
               particles = [ P.b__tilde__, P.t, P.W__minus__, P.G__minus__, P.G__plus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFVSS1 ],
               couplings = {(0,0):C.GC_1338})

V_819 = Vertex(name = 'V_819',
               particles = [ P.d__tilde__, P.u, P.W__minus__, P.H, P.H ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFVSS1, L.FFVSS2 ],
               couplings = {(0,0):C.GC_1330,(0,1):C.GC_1326})

V_820 = Vertex(name = 'V_820',
               particles = [ P.s__tilde__, P.u, P.W__minus__, P.H, P.H ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFVSS1 ],
               couplings = {(0,0):C.GC_1331})

V_821 = Vertex(name = 'V_821',
               particles = [ P.b__tilde__, P.u, P.W__minus__, P.H, P.H ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFVSS1 ],
               couplings = {(0,0):C.GC_1332})

V_822 = Vertex(name = 'V_822',
               particles = [ P.d__tilde__, P.c, P.W__minus__, P.H, P.H ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFVSS1 ],
               couplings = {(0,0):C.GC_1333})

V_823 = Vertex(name = 'V_823',
               particles = [ P.s__tilde__, P.c, P.W__minus__, P.H, P.H ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFVSS1, L.FFVSS2 ],
               couplings = {(0,0):C.GC_1334,(0,1):C.GC_1326})

V_824 = Vertex(name = 'V_824',
               particles = [ P.b__tilde__, P.c, P.W__minus__, P.H, P.H ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFVSS1 ],
               couplings = {(0,0):C.GC_1335})

V_825 = Vertex(name = 'V_825',
               particles = [ P.d__tilde__, P.t, P.W__minus__, P.H, P.H ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFVSS1 ],
               couplings = {(0,0):C.GC_1336})

V_826 = Vertex(name = 'V_826',
               particles = [ P.s__tilde__, P.t, P.W__minus__, P.H, P.H ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFVSS1 ],
               couplings = {(0,0):C.GC_1337})

V_827 = Vertex(name = 'V_827',
               particles = [ P.b__tilde__, P.t, P.W__minus__, P.H, P.H ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFVSS1, L.FFVSS2 ],
               couplings = {(0,0):C.GC_1338,(0,1):C.GC_1326})

V_828 = Vertex(name = 'V_828',
               particles = [ P.d__tilde__, P.u, P.W__minus__, P.H ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFVS1, L.FFVS10, L.FFVS6, L.FFVS7 ],
               couplings = {(0,2):C.GC_760,(0,1):C.GC_823,(0,0):C.GC_1459,(0,3):C.GC_1456})

V_829 = Vertex(name = 'V_829',
               particles = [ P.s__tilde__, P.u, P.W__minus__, P.H ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFVS1, L.FFVS10, L.FFVS4 ],
               couplings = {(0,2):C.GC_767,(0,1):C.GC_830,(0,0):C.GC_1460})

V_830 = Vertex(name = 'V_830',
               particles = [ P.b__tilde__, P.u, P.W__minus__, P.H ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFVS1, L.FFVS10, L.FFVS4 ],
               couplings = {(0,2):C.GC_774,(0,1):C.GC_837,(0,0):C.GC_1461})

V_831 = Vertex(name = 'V_831',
               particles = [ P.d__tilde__, P.c, P.W__minus__, P.H ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFVS1, L.FFVS10, L.FFVS4 ],
               couplings = {(0,2):C.GC_781,(0,1):C.GC_844,(0,0):C.GC_1462})

V_832 = Vertex(name = 'V_832',
               particles = [ P.s__tilde__, P.c, P.W__minus__, P.H ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFVS1, L.FFVS10, L.FFVS4, L.FFVS7 ],
               couplings = {(0,2):C.GC_788,(0,1):C.GC_851,(0,0):C.GC_1463,(0,3):C.GC_1456})

V_833 = Vertex(name = 'V_833',
               particles = [ P.b__tilde__, P.c, P.W__minus__, P.H ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFVS1, L.FFVS10, L.FFVS4 ],
               couplings = {(0,2):C.GC_795,(0,1):C.GC_858,(0,0):C.GC_1464})

V_834 = Vertex(name = 'V_834',
               particles = [ P.d__tilde__, P.t, P.W__minus__, P.H ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFVS1, L.FFVS10, L.FFVS4 ],
               couplings = {(0,2):C.GC_802,(0,1):C.GC_865,(0,0):C.GC_1465})

V_835 = Vertex(name = 'V_835',
               particles = [ P.s__tilde__, P.t, P.W__minus__, P.H ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFVS1, L.FFVS10, L.FFVS4 ],
               couplings = {(0,2):C.GC_809,(0,1):C.GC_872,(0,0):C.GC_1466})

V_836 = Vertex(name = 'V_836',
               particles = [ P.b__tilde__, P.t, P.W__minus__, P.H ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFVS1, L.FFVS10, L.FFVS4, L.FFVS7 ],
               couplings = {(0,2):C.GC_816,(0,1):C.GC_879,(0,0):C.GC_1467,(0,3):C.GC_1456})

V_837 = Vertex(name = 'V_837',
               particles = [ P.u__tilde__, P.u, P.W__minus__, P.G0, P.G__plus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFVSS1, L.FFVSS2 ],
               couplings = {(0,0):C.GC_1319,(0,1):C.GC_1322})

V_838 = Vertex(name = 'V_838',
               particles = [ P.c__tilde__, P.c, P.W__minus__, P.G0, P.G__plus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFVSS1, L.FFVSS2 ],
               couplings = {(0,0):C.GC_1319,(0,1):C.GC_1322})

V_839 = Vertex(name = 'V_839',
               particles = [ P.t__tilde__, P.t, P.W__minus__, P.G0, P.G__plus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFVSS1, L.FFVSS2 ],
               couplings = {(0,0):C.GC_1319,(0,1):C.GC_1322})

V_840 = Vertex(name = 'V_840',
               particles = [ P.u__tilde__, P.u, P.W__minus__, P.G__plus__, P.H ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFVSS1, L.FFVSS2 ],
               couplings = {(0,0):C.GC_1318,(0,1):C.GC_1321})

V_841 = Vertex(name = 'V_841',
               particles = [ P.c__tilde__, P.c, P.W__minus__, P.G__plus__, P.H ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFVSS1, L.FFVSS2 ],
               couplings = {(0,0):C.GC_1318,(0,1):C.GC_1321})

V_842 = Vertex(name = 'V_842',
               particles = [ P.t__tilde__, P.t, P.W__minus__, P.G__plus__, P.H ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFVSS1, L.FFVSS2 ],
               couplings = {(0,0):C.GC_1318,(0,1):C.GC_1321})

V_843 = Vertex(name = 'V_843',
               particles = [ P.u__tilde__, P.u, P.W__minus__, P.G__plus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFVS1, L.FFVS10, L.FFVS7 ],
               couplings = {(0,0):C.GC_1451,(0,2):C.GC_1453,(0,1):C.GC_2284})

V_844 = Vertex(name = 'V_844',
               particles = [ P.c__tilde__, P.c, P.W__minus__, P.G__plus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFVS1, L.FFVS10, L.FFVS7 ],
               couplings = {(0,0):C.GC_1451,(0,2):C.GC_1453,(0,1):C.GC_1876})

V_845 = Vertex(name = 'V_845',
               particles = [ P.t__tilde__, P.t, P.W__minus__, P.G__plus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFVS1, L.FFVS10, L.FFVS7 ],
               couplings = {(0,0):C.GC_1451,(0,2):C.GC_1453,(0,1):C.GC_2168})

V_846 = Vertex(name = 'V_846',
               particles = [ P.e__plus__, P.ve, P.W__minus__ ],
               color = [ '1' ],
               lorentz = [ L.FFV1, L.FFV9 ],
               couplings = {(0,1):C.GC_2005,(0,0):C.GC_556})

V_847 = Vertex(name = 'V_847',
               particles = [ P.e__plus__, P.ve, P.W__minus__ ],
               color = [ '1' ],
               lorentz = [ L.FFV1 ],
               couplings = {(0,0):C.GC_1769})

V_848 = Vertex(name = 'V_848',
               particles = [ P.mu__plus__, P.vm, P.W__minus__ ],
               color = [ '1' ],
               lorentz = [ L.FFV1, L.FFV9 ],
               couplings = {(0,1):C.GC_2081,(0,0):C.GC_556})

V_849 = Vertex(name = 'V_849',
               particles = [ P.mu__plus__, P.vm, P.W__minus__ ],
               color = [ '1' ],
               lorentz = [ L.FFV1 ],
               couplings = {(0,0):C.GC_1769})

V_850 = Vertex(name = 'V_850',
               particles = [ P.ta__plus__, P.vt, P.W__minus__ ],
               color = [ '1' ],
               lorentz = [ L.FFV1, L.FFV9 ],
               couplings = {(0,1):C.GC_2248,(0,0):C.GC_556})

V_851 = Vertex(name = 'V_851',
               particles = [ P.ta__plus__, P.vt, P.W__minus__ ],
               color = [ '1' ],
               lorentz = [ L.FFV1 ],
               couplings = {(0,0):C.GC_1769})

V_852 = Vertex(name = 'V_852',
               particles = [ P.e__plus__, P.ve, P.W__minus__, P.G0, P.G0 ],
               color = [ '1' ],
               lorentz = [ L.FFVSS1 ],
               couplings = {(0,0):C.GC_1329})

V_853 = Vertex(name = 'V_853',
               particles = [ P.mu__plus__, P.vm, P.W__minus__, P.G0, P.G0 ],
               color = [ '1' ],
               lorentz = [ L.FFVSS1 ],
               couplings = {(0,0):C.GC_1329})

V_854 = Vertex(name = 'V_854',
               particles = [ P.ta__plus__, P.vt, P.W__minus__, P.G0, P.G0 ],
               color = [ '1' ],
               lorentz = [ L.FFVSS1 ],
               couplings = {(0,0):C.GC_1329})

V_855 = Vertex(name = 'V_855',
               particles = [ P.e__plus__, P.ve, P.W__minus__, P.G__minus__, P.G__plus__ ],
               color = [ '1' ],
               lorentz = [ L.FFVSS1 ],
               couplings = {(0,0):C.GC_1329})

V_856 = Vertex(name = 'V_856',
               particles = [ P.mu__plus__, P.vm, P.W__minus__, P.G__minus__, P.G__plus__ ],
               color = [ '1' ],
               lorentz = [ L.FFVSS1 ],
               couplings = {(0,0):C.GC_1329})

V_857 = Vertex(name = 'V_857',
               particles = [ P.ta__plus__, P.vt, P.W__minus__, P.G__minus__, P.G__plus__ ],
               color = [ '1' ],
               lorentz = [ L.FFVSS1 ],
               couplings = {(0,0):C.GC_1329})

V_858 = Vertex(name = 'V_858',
               particles = [ P.e__plus__, P.ve, P.W__minus__, P.H, P.H ],
               color = [ '1' ],
               lorentz = [ L.FFVSS1 ],
               couplings = {(0,0):C.GC_1329})

V_859 = Vertex(name = 'V_859',
               particles = [ P.mu__plus__, P.vm, P.W__minus__, P.H, P.H ],
               color = [ '1' ],
               lorentz = [ L.FFVSS1 ],
               couplings = {(0,0):C.GC_1329})

V_860 = Vertex(name = 'V_860',
               particles = [ P.ta__plus__, P.vt, P.W__minus__, P.H, P.H ],
               color = [ '1' ],
               lorentz = [ L.FFVSS1 ],
               couplings = {(0,0):C.GC_1329})

V_861 = Vertex(name = 'V_861',
               particles = [ P.e__plus__, P.ve, P.W__minus__, P.H ],
               color = [ '1' ],
               lorentz = [ L.FFVS1, L.FFVS10 ],
               couplings = {(0,1):C.GC_1974,(0,0):C.GC_1458})

V_862 = Vertex(name = 'V_862',
               particles = [ P.mu__plus__, P.vm, P.W__minus__, P.H ],
               color = [ '1' ],
               lorentz = [ L.FFVS1, L.FFVS10 ],
               couplings = {(0,1):C.GC_2046,(0,0):C.GC_1458})

V_863 = Vertex(name = 'V_863',
               particles = [ P.ta__plus__, P.vt, P.W__minus__, P.H ],
               color = [ '1' ],
               lorentz = [ L.FFVS1, L.FFVS10 ],
               couplings = {(0,1):C.GC_2217,(0,0):C.GC_1458})

V_864 = Vertex(name = 'V_864',
               particles = [ P.ve__tilde__, P.ve, P.W__minus__, P.G0, P.G__plus__ ],
               color = [ '1' ],
               lorentz = [ L.FFVSS1 ],
               couplings = {(0,0):C.GC_1316})

V_865 = Vertex(name = 'V_865',
               particles = [ P.vm__tilde__, P.vm, P.W__minus__, P.G0, P.G__plus__ ],
               color = [ '1' ],
               lorentz = [ L.FFVSS1 ],
               couplings = {(0,0):C.GC_1316})

V_866 = Vertex(name = 'V_866',
               particles = [ P.vt__tilde__, P.vt, P.W__minus__, P.G0, P.G__plus__ ],
               color = [ '1' ],
               lorentz = [ L.FFVSS1 ],
               couplings = {(0,0):C.GC_1316})

V_867 = Vertex(name = 'V_867',
               particles = [ P.ve__tilde__, P.ve, P.W__minus__, P.G__plus__, P.H ],
               color = [ '1' ],
               lorentz = [ L.FFVSS1 ],
               couplings = {(0,0):C.GC_1315})

V_868 = Vertex(name = 'V_868',
               particles = [ P.vm__tilde__, P.vm, P.W__minus__, P.G__plus__, P.H ],
               color = [ '1' ],
               lorentz = [ L.FFVSS1 ],
               couplings = {(0,0):C.GC_1315})

V_869 = Vertex(name = 'V_869',
               particles = [ P.vt__tilde__, P.vt, P.W__minus__, P.G__plus__, P.H ],
               color = [ '1' ],
               lorentz = [ L.FFVSS1 ],
               couplings = {(0,0):C.GC_1315})

V_870 = Vertex(name = 'V_870',
               particles = [ P.ve__tilde__, P.ve, P.W__minus__, P.G__plus__ ],
               color = [ '1' ],
               lorentz = [ L.FFVS1 ],
               couplings = {(0,0):C.GC_1449})

V_871 = Vertex(name = 'V_871',
               particles = [ P.vm__tilde__, P.vm, P.W__minus__, P.G__plus__ ],
               color = [ '1' ],
               lorentz = [ L.FFVS1 ],
               couplings = {(0,0):C.GC_1449})

V_872 = Vertex(name = 'V_872',
               particles = [ P.vt__tilde__, P.vt, P.W__minus__, P.G__plus__ ],
               color = [ '1' ],
               lorentz = [ L.FFVS1 ],
               couplings = {(0,0):C.GC_1449})

V_873 = Vertex(name = 'V_873',
               particles = [ P.d__tilde__, P.d, P.W__plus__, P.G0, P.G__minus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFVSS1, L.FFVSS2 ],
               couplings = {(0,0):C.GC_1319,(0,1):C.GC_1310})

V_874 = Vertex(name = 'V_874',
               particles = [ P.s__tilde__, P.s, P.W__plus__, P.G0, P.G__minus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFVSS1, L.FFVSS2 ],
               couplings = {(0,0):C.GC_1319,(0,1):C.GC_1310})

V_875 = Vertex(name = 'V_875',
               particles = [ P.b__tilde__, P.b, P.W__plus__, P.G0, P.G__minus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFVSS1, L.FFVSS2 ],
               couplings = {(0,0):C.GC_1319,(0,1):C.GC_1310})

V_876 = Vertex(name = 'V_876',
               particles = [ P.d__tilde__, P.d, P.W__plus__, P.G__minus__, P.H ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFVSS1, L.FFVSS2 ],
               couplings = {(0,0):C.GC_1320,(0,1):C.GC_1311})

V_877 = Vertex(name = 'V_877',
               particles = [ P.s__tilde__, P.s, P.W__plus__, P.G__minus__, P.H ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFVSS1, L.FFVSS2 ],
               couplings = {(0,0):C.GC_1320,(0,1):C.GC_1311})

V_878 = Vertex(name = 'V_878',
               particles = [ P.b__tilde__, P.b, P.W__plus__, P.G__minus__, P.H ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFVSS1, L.FFVSS2 ],
               couplings = {(0,0):C.GC_1320,(0,1):C.GC_1311})

V_879 = Vertex(name = 'V_879',
               particles = [ P.d__tilde__, P.d, P.W__plus__, P.G__minus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFVS1, L.FFVS10, L.FFVS7 ],
               couplings = {(0,0):C.GC_1452,(0,2):C.GC_1446,(0,1):C.GC_1921})

V_880 = Vertex(name = 'V_880',
               particles = [ P.s__tilde__, P.s, P.W__plus__, P.G__minus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFVS1, L.FFVS10, L.FFVS7 ],
               couplings = {(0,0):C.GC_1452,(0,2):C.GC_1446,(0,1):C.GC_2119})

V_881 = Vertex(name = 'V_881',
               particles = [ P.b__tilde__, P.b, P.W__plus__, P.G__minus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFVS1, L.FFVS10, L.FFVS7 ],
               couplings = {(0,0):C.GC_1452,(0,2):C.GC_1446,(0,1):C.GC_1829})

V_882 = Vertex(name = 'V_882',
               particles = [ P.u__tilde__, P.d, P.W__plus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFV1, L.FFV4, L.FFV5, L.FFV9 ],
               couplings = {(0,1):C.GC_1573,(0,3):C.GC_1600,(0,0):C.GC_2311,(0,2):C.GC_557})

V_883 = Vertex(name = 'V_883',
               particles = [ P.u__tilde__, P.d, P.W__plus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFV1 ],
               couplings = {(0,0):C.GC_2324})

V_884 = Vertex(name = 'V_884',
               particles = [ P.c__tilde__, P.d, P.W__plus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFV1, L.FFV4, L.FFV9 ],
               couplings = {(0,1):C.GC_1576,(0,2):C.GC_1603,(0,0):C.GC_2353})

V_885 = Vertex(name = 'V_885',
               particles = [ P.c__tilde__, P.d, P.W__plus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFV1 ],
               couplings = {(0,0):C.GC_2366})

V_886 = Vertex(name = 'V_886',
               particles = [ P.t__tilde__, P.d, P.W__plus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFV1, L.FFV4, L.FFV9 ],
               couplings = {(0,1):C.GC_1579,(0,2):C.GC_1606,(0,0):C.GC_2395})

V_887 = Vertex(name = 'V_887',
               particles = [ P.t__tilde__, P.d, P.W__plus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFV1 ],
               couplings = {(0,0):C.GC_2408})

V_888 = Vertex(name = 'V_888',
               particles = [ P.u__tilde__, P.s, P.W__plus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFV1, L.FFV4, L.FFV9 ],
               couplings = {(0,1):C.GC_1582,(0,2):C.GC_1609,(0,0):C.GC_2325})

V_889 = Vertex(name = 'V_889',
               particles = [ P.u__tilde__, P.s, P.W__plus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFV1 ],
               couplings = {(0,0):C.GC_2338})

V_890 = Vertex(name = 'V_890',
               particles = [ P.c__tilde__, P.s, P.W__plus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFV1, L.FFV4, L.FFV5, L.FFV9 ],
               couplings = {(0,1):C.GC_1585,(0,3):C.GC_1612,(0,0):C.GC_2367,(0,2):C.GC_557})

V_891 = Vertex(name = 'V_891',
               particles = [ P.c__tilde__, P.s, P.W__plus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFV1 ],
               couplings = {(0,0):C.GC_2380})

V_892 = Vertex(name = 'V_892',
               particles = [ P.t__tilde__, P.s, P.W__plus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFV1, L.FFV4, L.FFV9 ],
               couplings = {(0,1):C.GC_1588,(0,2):C.GC_1615,(0,0):C.GC_2409})

V_893 = Vertex(name = 'V_893',
               particles = [ P.t__tilde__, P.s, P.W__plus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFV1 ],
               couplings = {(0,0):C.GC_2422})

V_894 = Vertex(name = 'V_894',
               particles = [ P.u__tilde__, P.b, P.W__plus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFV1, L.FFV4, L.FFV9 ],
               couplings = {(0,1):C.GC_1591,(0,2):C.GC_1618,(0,0):C.GC_2339})

V_895 = Vertex(name = 'V_895',
               particles = [ P.u__tilde__, P.b, P.W__plus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFV1 ],
               couplings = {(0,0):C.GC_2352})

V_896 = Vertex(name = 'V_896',
               particles = [ P.c__tilde__, P.b, P.W__plus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFV1, L.FFV4, L.FFV9 ],
               couplings = {(0,1):C.GC_1594,(0,2):C.GC_1621,(0,0):C.GC_2381})

V_897 = Vertex(name = 'V_897',
               particles = [ P.c__tilde__, P.b, P.W__plus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFV1 ],
               couplings = {(0,0):C.GC_2394})

V_898 = Vertex(name = 'V_898',
               particles = [ P.t__tilde__, P.b, P.W__plus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFV1, L.FFV4, L.FFV5, L.FFV9 ],
               couplings = {(0,1):C.GC_1597,(0,3):C.GC_1624,(0,0):C.GC_2423,(0,2):C.GC_557})

V_899 = Vertex(name = 'V_899',
               particles = [ P.t__tilde__, P.b, P.W__plus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFV1 ],
               couplings = {(0,0):C.GC_2436})

V_900 = Vertex(name = 'V_900',
               particles = [ P.u__tilde__, P.d, P.W__plus__, P.G0, P.G0 ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFVSS1, L.FFVSS2 ],
               couplings = {(0,0):C.GC_2317,(0,1):C.GC_1325})

V_901 = Vertex(name = 'V_901',
               particles = [ P.c__tilde__, P.d, P.W__plus__, P.G0, P.G0 ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFVSS1 ],
               couplings = {(0,0):C.GC_2359})

V_902 = Vertex(name = 'V_902',
               particles = [ P.t__tilde__, P.d, P.W__plus__, P.G0, P.G0 ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFVSS1 ],
               couplings = {(0,0):C.GC_2401})

V_903 = Vertex(name = 'V_903',
               particles = [ P.u__tilde__, P.s, P.W__plus__, P.G0, P.G0 ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFVSS1 ],
               couplings = {(0,0):C.GC_2331})

V_904 = Vertex(name = 'V_904',
               particles = [ P.c__tilde__, P.s, P.W__plus__, P.G0, P.G0 ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFVSS1, L.FFVSS2 ],
               couplings = {(0,0):C.GC_2373,(0,1):C.GC_1325})

V_905 = Vertex(name = 'V_905',
               particles = [ P.t__tilde__, P.s, P.W__plus__, P.G0, P.G0 ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFVSS1 ],
               couplings = {(0,0):C.GC_2415})

V_906 = Vertex(name = 'V_906',
               particles = [ P.u__tilde__, P.b, P.W__plus__, P.G0, P.G0 ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFVSS1 ],
               couplings = {(0,0):C.GC_2345})

V_907 = Vertex(name = 'V_907',
               particles = [ P.c__tilde__, P.b, P.W__plus__, P.G0, P.G0 ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFVSS1 ],
               couplings = {(0,0):C.GC_2387})

V_908 = Vertex(name = 'V_908',
               particles = [ P.t__tilde__, P.b, P.W__plus__, P.G0, P.G0 ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFVSS1, L.FFVSS2 ],
               couplings = {(0,0):C.GC_2429,(0,1):C.GC_1325})

V_909 = Vertex(name = 'V_909',
               particles = [ P.u__tilde__, P.d, P.W__plus__, P.G__minus__, P.G__plus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFVSS1 ],
               couplings = {(0,0):C.GC_2317})

V_910 = Vertex(name = 'V_910',
               particles = [ P.c__tilde__, P.d, P.W__plus__, P.G__minus__, P.G__plus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFVSS1 ],
               couplings = {(0,0):C.GC_2359})

V_911 = Vertex(name = 'V_911',
               particles = [ P.t__tilde__, P.d, P.W__plus__, P.G__minus__, P.G__plus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFVSS1 ],
               couplings = {(0,0):C.GC_2401})

V_912 = Vertex(name = 'V_912',
               particles = [ P.u__tilde__, P.s, P.W__plus__, P.G__minus__, P.G__plus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFVSS1 ],
               couplings = {(0,0):C.GC_2331})

V_913 = Vertex(name = 'V_913',
               particles = [ P.c__tilde__, P.s, P.W__plus__, P.G__minus__, P.G__plus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFVSS1 ],
               couplings = {(0,0):C.GC_2373})

V_914 = Vertex(name = 'V_914',
               particles = [ P.t__tilde__, P.s, P.W__plus__, P.G__minus__, P.G__plus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFVSS1 ],
               couplings = {(0,0):C.GC_2415})

V_915 = Vertex(name = 'V_915',
               particles = [ P.u__tilde__, P.b, P.W__plus__, P.G__minus__, P.G__plus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFVSS1 ],
               couplings = {(0,0):C.GC_2345})

V_916 = Vertex(name = 'V_916',
               particles = [ P.c__tilde__, P.b, P.W__plus__, P.G__minus__, P.G__plus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFVSS1 ],
               couplings = {(0,0):C.GC_2387})

V_917 = Vertex(name = 'V_917',
               particles = [ P.t__tilde__, P.b, P.W__plus__, P.G__minus__, P.G__plus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFVSS1 ],
               couplings = {(0,0):C.GC_2429})

V_918 = Vertex(name = 'V_918',
               particles = [ P.u__tilde__, P.d, P.W__plus__, P.H, P.H ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFVSS1, L.FFVSS2 ],
               couplings = {(0,0):C.GC_2317,(0,1):C.GC_1326})

V_919 = Vertex(name = 'V_919',
               particles = [ P.c__tilde__, P.d, P.W__plus__, P.H, P.H ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFVSS1 ],
               couplings = {(0,0):C.GC_2359})

V_920 = Vertex(name = 'V_920',
               particles = [ P.t__tilde__, P.d, P.W__plus__, P.H, P.H ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFVSS1 ],
               couplings = {(0,0):C.GC_2401})

V_921 = Vertex(name = 'V_921',
               particles = [ P.u__tilde__, P.s, P.W__plus__, P.H, P.H ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFVSS1 ],
               couplings = {(0,0):C.GC_2331})

V_922 = Vertex(name = 'V_922',
               particles = [ P.c__tilde__, P.s, P.W__plus__, P.H, P.H ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFVSS1, L.FFVSS2 ],
               couplings = {(0,0):C.GC_2373,(0,1):C.GC_1326})

V_923 = Vertex(name = 'V_923',
               particles = [ P.t__tilde__, P.s, P.W__plus__, P.H, P.H ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFVSS1 ],
               couplings = {(0,0):C.GC_2415})

V_924 = Vertex(name = 'V_924',
               particles = [ P.u__tilde__, P.b, P.W__plus__, P.H, P.H ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFVSS1 ],
               couplings = {(0,0):C.GC_2345})

V_925 = Vertex(name = 'V_925',
               particles = [ P.c__tilde__, P.b, P.W__plus__, P.H, P.H ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFVSS1 ],
               couplings = {(0,0):C.GC_2387})

V_926 = Vertex(name = 'V_926',
               particles = [ P.t__tilde__, P.b, P.W__plus__, P.H, P.H ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFVSS1, L.FFVSS2 ],
               couplings = {(0,0):C.GC_2429,(0,1):C.GC_1326})

V_927 = Vertex(name = 'V_927',
               particles = [ P.u__tilde__, P.d, P.W__plus__, P.H ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFVS1, L.FFVS2, L.FFVS3, L.FFVS7, L.FFVS8, L.FFVS9 ],
               couplings = {(0,2):C.GC_617,(0,5):C.GC_689,(0,1):C.GC_616,(0,4):C.GC_688,(0,0):C.GC_2322,(0,3):C.GC_1456})

V_928 = Vertex(name = 'V_928',
               particles = [ P.c__tilde__, P.d, P.W__plus__, P.H ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFVS1, L.FFVS2, L.FFVS3, L.FFVS8, L.FFVS9 ],
               couplings = {(0,2):C.GC_625,(0,4):C.GC_697,(0,1):C.GC_624,(0,3):C.GC_696,(0,0):C.GC_2364})

V_929 = Vertex(name = 'V_929',
               particles = [ P.t__tilde__, P.d, P.W__plus__, P.H ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFVS1, L.FFVS2, L.FFVS3, L.FFVS8, L.FFVS9 ],
               couplings = {(0,2):C.GC_633,(0,4):C.GC_705,(0,1):C.GC_632,(0,3):C.GC_704,(0,0):C.GC_2406})

V_930 = Vertex(name = 'V_930',
               particles = [ P.u__tilde__, P.s, P.W__plus__, P.H ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFVS1, L.FFVS2, L.FFVS3, L.FFVS8, L.FFVS9 ],
               couplings = {(0,2):C.GC_641,(0,4):C.GC_713,(0,1):C.GC_640,(0,3):C.GC_712,(0,0):C.GC_2336})

V_931 = Vertex(name = 'V_931',
               particles = [ P.c__tilde__, P.s, P.W__plus__, P.H ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFVS1, L.FFVS2, L.FFVS3, L.FFVS7, L.FFVS8, L.FFVS9 ],
               couplings = {(0,2):C.GC_649,(0,5):C.GC_721,(0,1):C.GC_648,(0,4):C.GC_720,(0,0):C.GC_2378,(0,3):C.GC_1456})

V_932 = Vertex(name = 'V_932',
               particles = [ P.t__tilde__, P.s, P.W__plus__, P.H ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFVS1, L.FFVS2, L.FFVS3, L.FFVS8, L.FFVS9 ],
               couplings = {(0,2):C.GC_657,(0,4):C.GC_729,(0,1):C.GC_656,(0,3):C.GC_728,(0,0):C.GC_2420})

V_933 = Vertex(name = 'V_933',
               particles = [ P.u__tilde__, P.b, P.W__plus__, P.H ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFVS1, L.FFVS2, L.FFVS3, L.FFVS8, L.FFVS9 ],
               couplings = {(0,2):C.GC_665,(0,4):C.GC_737,(0,1):C.GC_664,(0,3):C.GC_736,(0,0):C.GC_2350})

V_934 = Vertex(name = 'V_934',
               particles = [ P.c__tilde__, P.b, P.W__plus__, P.H ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFVS1, L.FFVS2, L.FFVS3, L.FFVS8, L.FFVS9 ],
               couplings = {(0,2):C.GC_673,(0,4):C.GC_745,(0,1):C.GC_672,(0,3):C.GC_744,(0,0):C.GC_2392})

V_935 = Vertex(name = 'V_935',
               particles = [ P.t__tilde__, P.b, P.W__plus__, P.H ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFVS1, L.FFVS2, L.FFVS3, L.FFVS7, L.FFVS8, L.FFVS9 ],
               couplings = {(0,2):C.GC_681,(0,5):C.GC_753,(0,1):C.GC_680,(0,4):C.GC_752,(0,0):C.GC_2434,(0,3):C.GC_1456})

V_936 = Vertex(name = 'V_936',
               particles = [ P.e__plus__, P.e__minus__, P.W__plus__, P.G0, P.G__minus__ ],
               color = [ '1' ],
               lorentz = [ L.FFVSS1, L.FFVSS2 ],
               couplings = {(0,0):C.GC_1316,(0,1):C.GC_1313})

V_937 = Vertex(name = 'V_937',
               particles = [ P.mu__plus__, P.mu__minus__, P.W__plus__, P.G0, P.G__minus__ ],
               color = [ '1' ],
               lorentz = [ L.FFVSS1, L.FFVSS2 ],
               couplings = {(0,0):C.GC_1316,(0,1):C.GC_1313})

V_938 = Vertex(name = 'V_938',
               particles = [ P.ta__plus__, P.ta__minus__, P.W__plus__, P.G0, P.G__minus__ ],
               color = [ '1' ],
               lorentz = [ L.FFVSS1, L.FFVSS2 ],
               couplings = {(0,0):C.GC_1316,(0,1):C.GC_1313})

V_939 = Vertex(name = 'V_939',
               particles = [ P.e__plus__, P.e__minus__, P.W__plus__, P.G__minus__, P.H ],
               color = [ '1' ],
               lorentz = [ L.FFVSS1, L.FFVSS2 ],
               couplings = {(0,0):C.GC_1317,(0,1):C.GC_1314})

V_940 = Vertex(name = 'V_940',
               particles = [ P.mu__plus__, P.mu__minus__, P.W__plus__, P.G__minus__, P.H ],
               color = [ '1' ],
               lorentz = [ L.FFVSS1, L.FFVSS2 ],
               couplings = {(0,0):C.GC_1317,(0,1):C.GC_1314})

V_941 = Vertex(name = 'V_941',
               particles = [ P.ta__plus__, P.ta__minus__, P.W__plus__, P.G__minus__, P.H ],
               color = [ '1' ],
               lorentz = [ L.FFVSS1, L.FFVSS2 ],
               couplings = {(0,0):C.GC_1317,(0,1):C.GC_1314})

V_942 = Vertex(name = 'V_942',
               particles = [ P.e__plus__, P.e__minus__, P.W__plus__, P.G__minus__ ],
               color = [ '1' ],
               lorentz = [ L.FFVS1, L.FFVS7, L.FFVS8, L.FFVS9 ],
               couplings = {(0,0):C.GC_1450,(0,1):C.GC_1448,(0,3):C.GC_1977,(0,2):C.GC_1976})

V_943 = Vertex(name = 'V_943',
               particles = [ P.mu__plus__, P.mu__minus__, P.W__plus__, P.G__minus__ ],
               color = [ '1' ],
               lorentz = [ L.FFVS1, L.FFVS7, L.FFVS8, L.FFVS9 ],
               couplings = {(0,0):C.GC_1450,(0,1):C.GC_1448,(0,3):C.GC_2049,(0,2):C.GC_2048})

V_944 = Vertex(name = 'V_944',
               particles = [ P.ta__plus__, P.ta__minus__, P.W__plus__, P.G__minus__ ],
               color = [ '1' ],
               lorentz = [ L.FFVS1, L.FFVS7, L.FFVS8, L.FFVS9 ],
               couplings = {(0,0):C.GC_1450,(0,1):C.GC_1448,(0,3):C.GC_2220,(0,2):C.GC_2219})

V_945 = Vertex(name = 'V_945',
               particles = [ P.ve__tilde__, P.e__minus__, P.W__plus__ ],
               color = [ '1' ],
               lorentz = [ L.FFV1, L.FFV2, L.FFV3 ],
               couplings = {(0,2):C.GC_2005,(0,1):C.GC_2004,(0,0):C.GC_556})

V_946 = Vertex(name = 'V_946',
               particles = [ P.ve__tilde__, P.e__minus__, P.W__plus__ ],
               color = [ '1' ],
               lorentz = [ L.FFV1 ],
               couplings = {(0,0):C.GC_1769})

V_947 = Vertex(name = 'V_947',
               particles = [ P.vm__tilde__, P.mu__minus__, P.W__plus__ ],
               color = [ '1' ],
               lorentz = [ L.FFV1, L.FFV2, L.FFV3 ],
               couplings = {(0,2):C.GC_2081,(0,1):C.GC_2080,(0,0):C.GC_556})

V_948 = Vertex(name = 'V_948',
               particles = [ P.vm__tilde__, P.mu__minus__, P.W__plus__ ],
               color = [ '1' ],
               lorentz = [ L.FFV1 ],
               couplings = {(0,0):C.GC_1769})

V_949 = Vertex(name = 'V_949',
               particles = [ P.vt__tilde__, P.ta__minus__, P.W__plus__ ],
               color = [ '1' ],
               lorentz = [ L.FFV1, L.FFV2, L.FFV3 ],
               couplings = {(0,2):C.GC_2248,(0,1):C.GC_2247,(0,0):C.GC_556})

V_950 = Vertex(name = 'V_950',
               particles = [ P.vt__tilde__, P.ta__minus__, P.W__plus__ ],
               color = [ '1' ],
               lorentz = [ L.FFV1 ],
               couplings = {(0,0):C.GC_1769})

V_951 = Vertex(name = 'V_951',
               particles = [ P.ve__tilde__, P.e__minus__, P.W__plus__, P.G0, P.G0 ],
               color = [ '1' ],
               lorentz = [ L.FFVSS1 ],
               couplings = {(0,0):C.GC_1329})

V_952 = Vertex(name = 'V_952',
               particles = [ P.vm__tilde__, P.mu__minus__, P.W__plus__, P.G0, P.G0 ],
               color = [ '1' ],
               lorentz = [ L.FFVSS1 ],
               couplings = {(0,0):C.GC_1329})

V_953 = Vertex(name = 'V_953',
               particles = [ P.vt__tilde__, P.ta__minus__, P.W__plus__, P.G0, P.G0 ],
               color = [ '1' ],
               lorentz = [ L.FFVSS1 ],
               couplings = {(0,0):C.GC_1329})

V_954 = Vertex(name = 'V_954',
               particles = [ P.ve__tilde__, P.e__minus__, P.W__plus__, P.G__minus__, P.G__plus__ ],
               color = [ '1' ],
               lorentz = [ L.FFVSS1 ],
               couplings = {(0,0):C.GC_1329})

V_955 = Vertex(name = 'V_955',
               particles = [ P.vm__tilde__, P.mu__minus__, P.W__plus__, P.G__minus__, P.G__plus__ ],
               color = [ '1' ],
               lorentz = [ L.FFVSS1 ],
               couplings = {(0,0):C.GC_1329})

V_956 = Vertex(name = 'V_956',
               particles = [ P.vt__tilde__, P.ta__minus__, P.W__plus__, P.G__minus__, P.G__plus__ ],
               color = [ '1' ],
               lorentz = [ L.FFVSS1 ],
               couplings = {(0,0):C.GC_1329})

V_957 = Vertex(name = 'V_957',
               particles = [ P.ve__tilde__, P.e__minus__, P.W__plus__, P.H, P.H ],
               color = [ '1' ],
               lorentz = [ L.FFVSS1 ],
               couplings = {(0,0):C.GC_1329})

V_958 = Vertex(name = 'V_958',
               particles = [ P.vm__tilde__, P.mu__minus__, P.W__plus__, P.H, P.H ],
               color = [ '1' ],
               lorentz = [ L.FFVSS1 ],
               couplings = {(0,0):C.GC_1329})

V_959 = Vertex(name = 'V_959',
               particles = [ P.vt__tilde__, P.ta__minus__, P.W__plus__, P.H, P.H ],
               color = [ '1' ],
               lorentz = [ L.FFVSS1 ],
               couplings = {(0,0):C.GC_1329})

V_960 = Vertex(name = 'V_960',
               particles = [ P.ve__tilde__, P.e__minus__, P.W__plus__, P.H ],
               color = [ '1' ],
               lorentz = [ L.FFVS1, L.FFVS2, L.FFVS3 ],
               couplings = {(0,2):C.GC_1974,(0,1):C.GC_1973,(0,0):C.GC_1458})

V_961 = Vertex(name = 'V_961',
               particles = [ P.vm__tilde__, P.mu__minus__, P.W__plus__, P.H ],
               color = [ '1' ],
               lorentz = [ L.FFVS1, L.FFVS2, L.FFVS3 ],
               couplings = {(0,2):C.GC_2046,(0,1):C.GC_2045,(0,0):C.GC_1458})

V_962 = Vertex(name = 'V_962',
               particles = [ P.vt__tilde__, P.ta__minus__, P.W__plus__, P.H ],
               color = [ '1' ],
               lorentz = [ L.FFVS1, L.FFVS2, L.FFVS3 ],
               couplings = {(0,2):C.GC_2217,(0,1):C.GC_2216,(0,0):C.GC_1458})

V_963 = Vertex(name = 'V_963',
               particles = [ P.u__tilde__, P.u, P.W__plus__, P.G0, P.G__minus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFVSS1, L.FFVSS2 ],
               couplings = {(0,0):C.GC_1319,(0,1):C.GC_1322})

V_964 = Vertex(name = 'V_964',
               particles = [ P.c__tilde__, P.c, P.W__plus__, P.G0, P.G__minus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFVSS1, L.FFVSS2 ],
               couplings = {(0,0):C.GC_1319,(0,1):C.GC_1322})

V_965 = Vertex(name = 'V_965',
               particles = [ P.t__tilde__, P.t, P.W__plus__, P.G0, P.G__minus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFVSS1, L.FFVSS2 ],
               couplings = {(0,0):C.GC_1319,(0,1):C.GC_1322})

V_966 = Vertex(name = 'V_966',
               particles = [ P.u__tilde__, P.u, P.W__plus__, P.G__minus__, P.H ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFVSS1, L.FFVSS2 ],
               couplings = {(0,0):C.GC_1320,(0,1):C.GC_1323})

V_967 = Vertex(name = 'V_967',
               particles = [ P.c__tilde__, P.c, P.W__plus__, P.G__minus__, P.H ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFVSS1, L.FFVSS2 ],
               couplings = {(0,0):C.GC_1320,(0,1):C.GC_1323})

V_968 = Vertex(name = 'V_968',
               particles = [ P.t__tilde__, P.t, P.W__plus__, P.G__minus__, P.H ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFVSS1, L.FFVSS2 ],
               couplings = {(0,0):C.GC_1320,(0,1):C.GC_1323})

V_969 = Vertex(name = 'V_969',
               particles = [ P.u__tilde__, P.u, P.W__plus__, P.G__minus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFVS1, L.FFVS2, L.FFVS3, L.FFVS7 ],
               couplings = {(0,0):C.GC_1452,(0,3):C.GC_1454,(0,2):C.GC_2283,(0,1):C.GC_2284})

V_970 = Vertex(name = 'V_970',
               particles = [ P.c__tilde__, P.c, P.W__plus__, P.G__minus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFVS1, L.FFVS2, L.FFVS3, L.FFVS7 ],
               couplings = {(0,0):C.GC_1452,(0,3):C.GC_1454,(0,2):C.GC_1875,(0,1):C.GC_1876})

V_971 = Vertex(name = 'V_971',
               particles = [ P.t__tilde__, P.t, P.W__plus__, P.G__minus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFVS1, L.FFVS2, L.FFVS3, L.FFVS7 ],
               couplings = {(0,0):C.GC_1452,(0,3):C.GC_1454,(0,2):C.GC_2167,(0,1):C.GC_2168})

V_972 = Vertex(name = 'V_972',
               particles = [ P.ve__tilde__, P.ve, P.W__plus__, P.G0, P.G__minus__ ],
               color = [ '1' ],
               lorentz = [ L.FFVSS1 ],
               couplings = {(0,0):C.GC_1316})

V_973 = Vertex(name = 'V_973',
               particles = [ P.vm__tilde__, P.vm, P.W__plus__, P.G0, P.G__minus__ ],
               color = [ '1' ],
               lorentz = [ L.FFVSS1 ],
               couplings = {(0,0):C.GC_1316})

V_974 = Vertex(name = 'V_974',
               particles = [ P.vt__tilde__, P.vt, P.W__plus__, P.G0, P.G__minus__ ],
               color = [ '1' ],
               lorentz = [ L.FFVSS1 ],
               couplings = {(0,0):C.GC_1316})

V_975 = Vertex(name = 'V_975',
               particles = [ P.ve__tilde__, P.ve, P.W__plus__, P.G__minus__, P.H ],
               color = [ '1' ],
               lorentz = [ L.FFVSS1 ],
               couplings = {(0,0):C.GC_1317})

V_976 = Vertex(name = 'V_976',
               particles = [ P.vm__tilde__, P.vm, P.W__plus__, P.G__minus__, P.H ],
               color = [ '1' ],
               lorentz = [ L.FFVSS1 ],
               couplings = {(0,0):C.GC_1317})

V_977 = Vertex(name = 'V_977',
               particles = [ P.vt__tilde__, P.vt, P.W__plus__, P.G__minus__, P.H ],
               color = [ '1' ],
               lorentz = [ L.FFVSS1 ],
               couplings = {(0,0):C.GC_1317})

V_978 = Vertex(name = 'V_978',
               particles = [ P.ve__tilde__, P.ve, P.W__plus__, P.G__minus__ ],
               color = [ '1' ],
               lorentz = [ L.FFVS1 ],
               couplings = {(0,0):C.GC_1450})

V_979 = Vertex(name = 'V_979',
               particles = [ P.vm__tilde__, P.vm, P.W__plus__, P.G__minus__ ],
               color = [ '1' ],
               lorentz = [ L.FFVS1 ],
               couplings = {(0,0):C.GC_1450})

V_980 = Vertex(name = 'V_980',
               particles = [ P.vt__tilde__, P.vt, P.W__plus__, P.G__minus__ ],
               color = [ '1' ],
               lorentz = [ L.FFVS1 ],
               couplings = {(0,0):C.GC_1450})

V_981 = Vertex(name = 'V_981',
               particles = [ P.d__tilde__, P.d, P.Z ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFV1, L.FFV2, L.FFV3, L.FFV5, L.FFV7, L.FFV8 ],
               couplings = {(0,0):C.GC_891,(0,3):C.GC_1787,(0,2):C.GC_1950,(0,5):C.GC_1950,(0,1):C.GC_1951,(0,4):C.GC_1951})

V_982 = Vertex(name = 'V_982',
               particles = [ P.d__tilde__, P.d, P.Z ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFV1, L.FFV5 ],
               couplings = {(0,0):C.GC_1790,(0,1):C.GC_885})

V_983 = Vertex(name = 'V_983',
               particles = [ P.s__tilde__, P.s, P.Z ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFV1, L.FFV2, L.FFV3, L.FFV5, L.FFV7, L.FFV8 ],
               couplings = {(0,0):C.GC_891,(0,3):C.GC_1787,(0,2):C.GC_2149,(0,5):C.GC_2149,(0,1):C.GC_2150,(0,4):C.GC_2150})

V_984 = Vertex(name = 'V_984',
               particles = [ P.s__tilde__, P.s, P.Z ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFV1, L.FFV5 ],
               couplings = {(0,0):C.GC_1790,(0,1):C.GC_885})

V_985 = Vertex(name = 'V_985',
               particles = [ P.b__tilde__, P.b, P.Z ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFV1, L.FFV2, L.FFV3, L.FFV5, L.FFV7, L.FFV8 ],
               couplings = {(0,0):C.GC_891,(0,3):C.GC_1787,(0,2):C.GC_1857,(0,5):C.GC_1857,(0,1):C.GC_1858,(0,4):C.GC_1858})

V_986 = Vertex(name = 'V_986',
               particles = [ P.b__tilde__, P.b, P.Z ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFV1, L.FFV5 ],
               couplings = {(0,0):C.GC_1790,(0,1):C.GC_885})

V_987 = Vertex(name = 'V_987',
               particles = [ P.d__tilde__, P.d, P.Z, P.G0, P.G0 ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFVSS1, L.FFVSS2 ],
               couplings = {(0,0):C.GC_1106,(0,1):C.GC_1096})

V_988 = Vertex(name = 'V_988',
               particles = [ P.s__tilde__, P.s, P.Z, P.G0, P.G0 ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFVSS1, L.FFVSS2 ],
               couplings = {(0,0):C.GC_1106,(0,1):C.GC_1096})

V_989 = Vertex(name = 'V_989',
               particles = [ P.b__tilde__, P.b, P.Z, P.G0, P.G0 ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFVSS1, L.FFVSS2 ],
               couplings = {(0,0):C.GC_1106,(0,1):C.GC_1096})

V_990 = Vertex(name = 'V_990',
               particles = [ P.d__tilde__, P.d, P.Z, P.G__minus__, P.G__plus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFVSS1, L.FFVSS2 ],
               couplings = {(0,0):C.GC_1108,(0,1):C.GC_1097})

V_991 = Vertex(name = 'V_991',
               particles = [ P.s__tilde__, P.s, P.Z, P.G__minus__, P.G__plus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFVSS1, L.FFVSS2 ],
               couplings = {(0,0):C.GC_1108,(0,1):C.GC_1097})

V_992 = Vertex(name = 'V_992',
               particles = [ P.b__tilde__, P.b, P.Z, P.G__minus__, P.G__plus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFVSS1, L.FFVSS2 ],
               couplings = {(0,0):C.GC_1108,(0,1):C.GC_1097})

V_993 = Vertex(name = 'V_993',
               particles = [ P.d__tilde__, P.d, P.Z, P.H, P.H ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFVSS1, L.FFVSS2 ],
               couplings = {(0,0):C.GC_1106,(0,1):C.GC_1096})

V_994 = Vertex(name = 'V_994',
               particles = [ P.s__tilde__, P.s, P.Z, P.H, P.H ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFVSS1, L.FFVSS2 ],
               couplings = {(0,0):C.GC_1106,(0,1):C.GC_1096})

V_995 = Vertex(name = 'V_995',
               particles = [ P.b__tilde__, P.b, P.Z, P.H, P.H ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFVSS1, L.FFVSS2 ],
               couplings = {(0,0):C.GC_1106,(0,1):C.GC_1096})

V_996 = Vertex(name = 'V_996',
               particles = [ P.d__tilde__, P.d, P.Z, P.H ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFVS1, L.FFVS2, L.FFVS3, L.FFVS7, L.FFVS8, L.FFVS9 ],
               couplings = {(0,0):C.GC_1139,(0,3):C.GC_1134,(0,2):C.GC_1947,(0,5):C.GC_1947,(0,1):C.GC_1948,(0,4):C.GC_1948})

V_997 = Vertex(name = 'V_997',
               particles = [ P.s__tilde__, P.s, P.Z, P.H ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFVS1, L.FFVS2, L.FFVS3, L.FFVS7, L.FFVS8, L.FFVS9 ],
               couplings = {(0,0):C.GC_1139,(0,3):C.GC_1134,(0,2):C.GC_2145,(0,5):C.GC_2145,(0,1):C.GC_2146,(0,4):C.GC_2146})

V_998 = Vertex(name = 'V_998',
               particles = [ P.b__tilde__, P.b, P.Z, P.H ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFVS1, L.FFVS2, L.FFVS3, L.FFVS7, L.FFVS8, L.FFVS9 ],
               couplings = {(0,0):C.GC_1139,(0,3):C.GC_1134,(0,2):C.GC_1853,(0,5):C.GC_1853,(0,1):C.GC_1854,(0,4):C.GC_1854})

V_999 = Vertex(name = 'V_999',
               particles = [ P.u__tilde__, P.d, P.Z, P.G0, P.G__plus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFVSS1, L.FFVSS2 ],
               couplings = {(0,0):C.GC_2318,(0,1):C.GC_1345})

V_1000 = Vertex(name = 'V_1000',
                particles = [ P.c__tilde__, P.d, P.Z, P.G0, P.G__plus__ ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFVSS1 ],
                couplings = {(0,0):C.GC_2360})

V_1001 = Vertex(name = 'V_1001',
                particles = [ P.t__tilde__, P.d, P.Z, P.G0, P.G__plus__ ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFVSS1 ],
                couplings = {(0,0):C.GC_2402})

V_1002 = Vertex(name = 'V_1002',
                particles = [ P.u__tilde__, P.s, P.Z, P.G0, P.G__plus__ ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFVSS1 ],
                couplings = {(0,0):C.GC_2332})

V_1003 = Vertex(name = 'V_1003',
                particles = [ P.c__tilde__, P.s, P.Z, P.G0, P.G__plus__ ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFVSS1, L.FFVSS2 ],
                couplings = {(0,0):C.GC_2374,(0,1):C.GC_1345})

V_1004 = Vertex(name = 'V_1004',
                particles = [ P.t__tilde__, P.s, P.Z, P.G0, P.G__plus__ ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFVSS1 ],
                couplings = {(0,0):C.GC_2416})

V_1005 = Vertex(name = 'V_1005',
                particles = [ P.u__tilde__, P.b, P.Z, P.G0, P.G__plus__ ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFVSS1 ],
                couplings = {(0,0):C.GC_2346})

V_1006 = Vertex(name = 'V_1006',
                particles = [ P.c__tilde__, P.b, P.Z, P.G0, P.G__plus__ ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFVSS1 ],
                couplings = {(0,0):C.GC_2388})

V_1007 = Vertex(name = 'V_1007',
                particles = [ P.t__tilde__, P.b, P.Z, P.G0, P.G__plus__ ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFVSS1, L.FFVSS2 ],
                couplings = {(0,0):C.GC_2430,(0,1):C.GC_1345})

V_1008 = Vertex(name = 'V_1008',
                particles = [ P.u__tilde__, P.d, P.Z, P.G__plus__, P.H ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFVSS1, L.FFVSS2 ],
                couplings = {(0,0):C.GC_2319,(0,1):C.GC_1344})

V_1009 = Vertex(name = 'V_1009',
                particles = [ P.c__tilde__, P.d, P.Z, P.G__plus__, P.H ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFVSS1 ],
                couplings = {(0,0):C.GC_2361})

V_1010 = Vertex(name = 'V_1010',
                particles = [ P.t__tilde__, P.d, P.Z, P.G__plus__, P.H ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFVSS1 ],
                couplings = {(0,0):C.GC_2403})

V_1011 = Vertex(name = 'V_1011',
                particles = [ P.u__tilde__, P.s, P.Z, P.G__plus__, P.H ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFVSS1 ],
                couplings = {(0,0):C.GC_2333})

V_1012 = Vertex(name = 'V_1012',
                particles = [ P.c__tilde__, P.s, P.Z, P.G__plus__, P.H ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFVSS1, L.FFVSS2 ],
                couplings = {(0,0):C.GC_2375,(0,1):C.GC_1344})

V_1013 = Vertex(name = 'V_1013',
                particles = [ P.t__tilde__, P.s, P.Z, P.G__plus__, P.H ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFVSS1 ],
                couplings = {(0,0):C.GC_2417})

V_1014 = Vertex(name = 'V_1014',
                particles = [ P.u__tilde__, P.b, P.Z, P.G__plus__, P.H ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFVSS1 ],
                couplings = {(0,0):C.GC_2347})

V_1015 = Vertex(name = 'V_1015',
                particles = [ P.c__tilde__, P.b, P.Z, P.G__plus__, P.H ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFVSS1 ],
                couplings = {(0,0):C.GC_2389})

V_1016 = Vertex(name = 'V_1016',
                particles = [ P.t__tilde__, P.b, P.Z, P.G__plus__, P.H ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFVSS1, L.FFVSS2 ],
                couplings = {(0,0):C.GC_2431,(0,1):C.GC_1344})

V_1017 = Vertex(name = 'V_1017',
                particles = [ P.u__tilde__, P.d, P.Z, P.G__plus__ ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFVS1, L.FFVS2, L.FFVS3, L.FFVS7, L.FFVS8, L.FFVS9 ],
                couplings = {(0,2):C.GC_908,(0,5):C.GC_925,(0,1):C.GC_907,(0,4):C.GC_926,(0,0):C.GC_2323,(0,3):C.GC_1472})

V_1018 = Vertex(name = 'V_1018',
                particles = [ P.c__tilde__, P.d, P.Z, P.G__plus__ ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFVS1, L.FFVS2, L.FFVS3, L.FFVS8, L.FFVS9 ],
                couplings = {(0,2):C.GC_910,(0,4):C.GC_927,(0,1):C.GC_909,(0,3):C.GC_928,(0,0):C.GC_2365})

V_1019 = Vertex(name = 'V_1019',
                particles = [ P.t__tilde__, P.d, P.Z, P.G__plus__ ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFVS1, L.FFVS2, L.FFVS3, L.FFVS8, L.FFVS9 ],
                couplings = {(0,2):C.GC_912,(0,4):C.GC_929,(0,1):C.GC_911,(0,3):C.GC_930,(0,0):C.GC_2407})

V_1020 = Vertex(name = 'V_1020',
                particles = [ P.u__tilde__, P.s, P.Z, P.G__plus__ ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFVS1, L.FFVS2, L.FFVS3, L.FFVS8, L.FFVS9 ],
                couplings = {(0,2):C.GC_914,(0,4):C.GC_931,(0,1):C.GC_913,(0,3):C.GC_932,(0,0):C.GC_2337})

V_1021 = Vertex(name = 'V_1021',
                particles = [ P.c__tilde__, P.s, P.Z, P.G__plus__ ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFVS1, L.FFVS2, L.FFVS3, L.FFVS7, L.FFVS8, L.FFVS9 ],
                couplings = {(0,2):C.GC_916,(0,5):C.GC_933,(0,1):C.GC_915,(0,4):C.GC_934,(0,0):C.GC_2379,(0,3):C.GC_1472})

V_1022 = Vertex(name = 'V_1022',
                particles = [ P.t__tilde__, P.s, P.Z, P.G__plus__ ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFVS1, L.FFVS2, L.FFVS3, L.FFVS8, L.FFVS9 ],
                couplings = {(0,2):C.GC_918,(0,4):C.GC_935,(0,1):C.GC_917,(0,3):C.GC_936,(0,0):C.GC_2421})

V_1023 = Vertex(name = 'V_1023',
                particles = [ P.u__tilde__, P.b, P.Z, P.G__plus__ ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFVS1, L.FFVS2, L.FFVS3, L.FFVS8, L.FFVS9 ],
                couplings = {(0,2):C.GC_920,(0,4):C.GC_937,(0,1):C.GC_919,(0,3):C.GC_938,(0,0):C.GC_2351})

V_1024 = Vertex(name = 'V_1024',
                particles = [ P.c__tilde__, P.b, P.Z, P.G__plus__ ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFVS1, L.FFVS2, L.FFVS3, L.FFVS8, L.FFVS9 ],
                couplings = {(0,2):C.GC_922,(0,4):C.GC_939,(0,1):C.GC_921,(0,3):C.GC_940,(0,0):C.GC_2393})

V_1025 = Vertex(name = 'V_1025',
                particles = [ P.t__tilde__, P.b, P.Z, P.G__plus__ ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFVS1, L.FFVS2, L.FFVS3, L.FFVS7, L.FFVS8, L.FFVS9 ],
                couplings = {(0,2):C.GC_924,(0,5):C.GC_941,(0,1):C.GC_923,(0,4):C.GC_942,(0,0):C.GC_2435,(0,3):C.GC_1472})

V_1026 = Vertex(name = 'V_1026',
                particles = [ P.e__plus__, P.e__minus__, P.Z ],
                color = [ '1' ],
                lorentz = [ L.FFV1, L.FFV2, L.FFV3, L.FFV5, L.FFV7, L.FFV8 ],
                couplings = {(0,0):C.GC_893,(0,3):C.GC_1791,(0,2):C.GC_2022,(0,5):C.GC_2022,(0,1):C.GC_2023,(0,4):C.GC_2023})

V_1027 = Vertex(name = 'V_1027',
                particles = [ P.e__plus__, P.e__minus__, P.Z ],
                color = [ '1' ],
                lorentz = [ L.FFV1, L.FFV5 ],
                couplings = {(0,0):C.GC_1793,(0,1):C.GC_887})

V_1028 = Vertex(name = 'V_1028',
                particles = [ P.mu__plus__, P.mu__minus__, P.Z ],
                color = [ '1' ],
                lorentz = [ L.FFV1, L.FFV2, L.FFV3, L.FFV5, L.FFV7, L.FFV8 ],
                couplings = {(0,0):C.GC_893,(0,3):C.GC_1791,(0,2):C.GC_2097,(0,5):C.GC_2097,(0,1):C.GC_2098,(0,4):C.GC_2098})

V_1029 = Vertex(name = 'V_1029',
                particles = [ P.mu__plus__, P.mu__minus__, P.Z ],
                color = [ '1' ],
                lorentz = [ L.FFV1, L.FFV5 ],
                couplings = {(0,0):C.GC_1793,(0,1):C.GC_887})

V_1030 = Vertex(name = 'V_1030',
                particles = [ P.ta__plus__, P.ta__minus__, P.Z ],
                color = [ '1' ],
                lorentz = [ L.FFV1, L.FFV2, L.FFV3, L.FFV5, L.FFV7, L.FFV8 ],
                couplings = {(0,0):C.GC_893,(0,3):C.GC_1791,(0,2):C.GC_2265,(0,5):C.GC_2265,(0,1):C.GC_2266,(0,4):C.GC_2266})

V_1031 = Vertex(name = 'V_1031',
                particles = [ P.ta__plus__, P.ta__minus__, P.Z ],
                color = [ '1' ],
                lorentz = [ L.FFV1, L.FFV5 ],
                couplings = {(0,0):C.GC_1793,(0,1):C.GC_887})

V_1032 = Vertex(name = 'V_1032',
                particles = [ P.e__plus__, P.e__minus__, P.Z, P.G0, P.G0 ],
                color = [ '1' ],
                lorentz = [ L.FFVSS1, L.FFVSS2 ],
                couplings = {(0,0):C.GC_1102,(0,1):C.GC_1098})

V_1033 = Vertex(name = 'V_1033',
                particles = [ P.mu__plus__, P.mu__minus__, P.Z, P.G0, P.G0 ],
                color = [ '1' ],
                lorentz = [ L.FFVSS1, L.FFVSS2 ],
                couplings = {(0,0):C.GC_1102,(0,1):C.GC_1098})

V_1034 = Vertex(name = 'V_1034',
                particles = [ P.ta__plus__, P.ta__minus__, P.Z, P.G0, P.G0 ],
                color = [ '1' ],
                lorentz = [ L.FFVSS1, L.FFVSS2 ],
                couplings = {(0,0):C.GC_1102,(0,1):C.GC_1098})

V_1035 = Vertex(name = 'V_1035',
                particles = [ P.e__plus__, P.e__minus__, P.Z, P.G__minus__, P.G__plus__ ],
                color = [ '1' ],
                lorentz = [ L.FFVSS1, L.FFVSS2 ],
                couplings = {(0,0):C.GC_1104,(0,1):C.GC_1099})

V_1036 = Vertex(name = 'V_1036',
                particles = [ P.mu__plus__, P.mu__minus__, P.Z, P.G__minus__, P.G__plus__ ],
                color = [ '1' ],
                lorentz = [ L.FFVSS1, L.FFVSS2 ],
                couplings = {(0,0):C.GC_1104,(0,1):C.GC_1099})

V_1037 = Vertex(name = 'V_1037',
                particles = [ P.ta__plus__, P.ta__minus__, P.Z, P.G__minus__, P.G__plus__ ],
                color = [ '1' ],
                lorentz = [ L.FFVSS1, L.FFVSS2 ],
                couplings = {(0,0):C.GC_1104,(0,1):C.GC_1099})

V_1038 = Vertex(name = 'V_1038',
                particles = [ P.e__plus__, P.e__minus__, P.Z, P.H, P.H ],
                color = [ '1' ],
                lorentz = [ L.FFVSS1, L.FFVSS2 ],
                couplings = {(0,0):C.GC_1102,(0,1):C.GC_1098})

V_1039 = Vertex(name = 'V_1039',
                particles = [ P.mu__plus__, P.mu__minus__, P.Z, P.H, P.H ],
                color = [ '1' ],
                lorentz = [ L.FFVSS1, L.FFVSS2 ],
                couplings = {(0,0):C.GC_1102,(0,1):C.GC_1098})

V_1040 = Vertex(name = 'V_1040',
                particles = [ P.ta__plus__, P.ta__minus__, P.Z, P.H, P.H ],
                color = [ '1' ],
                lorentz = [ L.FFVSS1, L.FFVSS2 ],
                couplings = {(0,0):C.GC_1102,(0,1):C.GC_1098})

V_1041 = Vertex(name = 'V_1041',
                particles = [ P.e__plus__, P.e__minus__, P.Z, P.H ],
                color = [ '1' ],
                lorentz = [ L.FFVS1, L.FFVS2, L.FFVS3, L.FFVS7, L.FFVS8, L.FFVS9 ],
                couplings = {(0,0):C.GC_1137,(0,3):C.GC_1135,(0,2):C.GC_2018,(0,5):C.GC_2018,(0,1):C.GC_2019,(0,4):C.GC_2019})

V_1042 = Vertex(name = 'V_1042',
                particles = [ P.mu__plus__, P.mu__minus__, P.Z, P.H ],
                color = [ '1' ],
                lorentz = [ L.FFVS1, L.FFVS2, L.FFVS3, L.FFVS7, L.FFVS8, L.FFVS9 ],
                couplings = {(0,0):C.GC_1137,(0,3):C.GC_1135,(0,2):C.GC_2094,(0,5):C.GC_2094,(0,1):C.GC_2095,(0,4):C.GC_2095})

V_1043 = Vertex(name = 'V_1043',
                particles = [ P.ta__plus__, P.ta__minus__, P.Z, P.H ],
                color = [ '1' ],
                lorentz = [ L.FFVS1, L.FFVS2, L.FFVS3, L.FFVS7, L.FFVS8, L.FFVS9 ],
                couplings = {(0,0):C.GC_1137,(0,3):C.GC_1135,(0,2):C.GC_2261,(0,5):C.GC_2261,(0,1):C.GC_2262,(0,4):C.GC_2262})

V_1044 = Vertex(name = 'V_1044',
                particles = [ P.ve__tilde__, P.e__minus__, P.Z, P.G0, P.G__plus__ ],
                color = [ '1' ],
                lorentz = [ L.FFVSS1 ],
                couplings = {(0,0):C.GC_1351})

V_1045 = Vertex(name = 'V_1045',
                particles = [ P.vm__tilde__, P.mu__minus__, P.Z, P.G0, P.G__plus__ ],
                color = [ '1' ],
                lorentz = [ L.FFVSS1 ],
                couplings = {(0,0):C.GC_1351})

V_1046 = Vertex(name = 'V_1046',
                particles = [ P.vt__tilde__, P.ta__minus__, P.Z, P.G0, P.G__plus__ ],
                color = [ '1' ],
                lorentz = [ L.FFVSS1 ],
                couplings = {(0,0):C.GC_1351})

V_1047 = Vertex(name = 'V_1047',
                particles = [ P.ve__tilde__, P.e__minus__, P.Z, P.G__plus__, P.H ],
                color = [ '1' ],
                lorentz = [ L.FFVSS1 ],
                couplings = {(0,0):C.GC_1352})

V_1048 = Vertex(name = 'V_1048',
                particles = [ P.vm__tilde__, P.mu__minus__, P.Z, P.G__plus__, P.H ],
                color = [ '1' ],
                lorentz = [ L.FFVSS1 ],
                couplings = {(0,0):C.GC_1352})

V_1049 = Vertex(name = 'V_1049',
                particles = [ P.vt__tilde__, P.ta__minus__, P.Z, P.G__plus__, P.H ],
                color = [ '1' ],
                lorentz = [ L.FFVSS1 ],
                couplings = {(0,0):C.GC_1352})

V_1050 = Vertex(name = 'V_1050',
                particles = [ P.ve__tilde__, P.e__minus__, P.Z, P.G__plus__ ],
                color = [ '1' ],
                lorentz = [ L.FFVS1, L.FFVS2, L.FFVS3 ],
                couplings = {(0,2):C.GC_2016,(0,1):C.GC_2015,(0,0):C.GC_1477})

V_1051 = Vertex(name = 'V_1051',
                particles = [ P.vm__tilde__, P.mu__minus__, P.Z, P.G__plus__ ],
                color = [ '1' ],
                lorentz = [ L.FFVS1, L.FFVS2, L.FFVS3, L.FFVS5 ],
                couplings = {(0,3):C.GC_2057,(0,2):C.GC_2050,(0,1):C.GC_2091,(0,0):C.GC_1477})

V_1052 = Vertex(name = 'V_1052',
                particles = [ P.vt__tilde__, P.ta__minus__, P.Z, P.G__plus__ ],
                color = [ '1' ],
                lorentz = [ L.FFVS1, L.FFVS2, L.FFVS3 ],
                couplings = {(0,2):C.GC_2259,(0,1):C.GC_2258,(0,0):C.GC_1477})

V_1053 = Vertex(name = 'V_1053',
                particles = [ P.d__tilde__, P.u, P.Z, P.G0, P.G__minus__ ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFVSS1, L.FFVSS2 ],
                couplings = {(0,0):C.GC_1354,(0,1):C.GC_1345})

V_1054 = Vertex(name = 'V_1054',
                particles = [ P.s__tilde__, P.u, P.Z, P.G0, P.G__minus__ ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFVSS1 ],
                couplings = {(0,0):C.GC_1356})

V_1055 = Vertex(name = 'V_1055',
                particles = [ P.b__tilde__, P.u, P.Z, P.G0, P.G__minus__ ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFVSS1 ],
                couplings = {(0,0):C.GC_1358})

V_1056 = Vertex(name = 'V_1056',
                particles = [ P.d__tilde__, P.c, P.Z, P.G0, P.G__minus__ ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFVSS1 ],
                couplings = {(0,0):C.GC_1360})

V_1057 = Vertex(name = 'V_1057',
                particles = [ P.s__tilde__, P.c, P.Z, P.G0, P.G__minus__ ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFVSS1, L.FFVSS2 ],
                couplings = {(0,0):C.GC_1362,(0,1):C.GC_1345})

V_1058 = Vertex(name = 'V_1058',
                particles = [ P.b__tilde__, P.c, P.Z, P.G0, P.G__minus__ ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFVSS1 ],
                couplings = {(0,0):C.GC_1364})

V_1059 = Vertex(name = 'V_1059',
                particles = [ P.d__tilde__, P.t, P.Z, P.G0, P.G__minus__ ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFVSS1 ],
                couplings = {(0,0):C.GC_1366})

V_1060 = Vertex(name = 'V_1060',
                particles = [ P.s__tilde__, P.t, P.Z, P.G0, P.G__minus__ ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFVSS1 ],
                couplings = {(0,0):C.GC_1368})

V_1061 = Vertex(name = 'V_1061',
                particles = [ P.b__tilde__, P.t, P.Z, P.G0, P.G__minus__ ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFVSS1, L.FFVSS2 ],
                couplings = {(0,0):C.GC_1370,(0,1):C.GC_1345})

V_1062 = Vertex(name = 'V_1062',
                particles = [ P.d__tilde__, P.u, P.Z, P.G__minus__, P.H ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFVSS1, L.FFVSS2 ],
                couplings = {(0,0):C.GC_1353,(0,1):C.GC_1346})

V_1063 = Vertex(name = 'V_1063',
                particles = [ P.s__tilde__, P.u, P.Z, P.G__minus__, P.H ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFVSS1 ],
                couplings = {(0,0):C.GC_1355})

V_1064 = Vertex(name = 'V_1064',
                particles = [ P.b__tilde__, P.u, P.Z, P.G__minus__, P.H ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFVSS1 ],
                couplings = {(0,0):C.GC_1357})

V_1065 = Vertex(name = 'V_1065',
                particles = [ P.d__tilde__, P.c, P.Z, P.G__minus__, P.H ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFVSS1 ],
                couplings = {(0,0):C.GC_1359})

V_1066 = Vertex(name = 'V_1066',
                particles = [ P.s__tilde__, P.c, P.Z, P.G__minus__, P.H ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFVSS1, L.FFVSS2 ],
                couplings = {(0,0):C.GC_1361,(0,1):C.GC_1346})

V_1067 = Vertex(name = 'V_1067',
                particles = [ P.b__tilde__, P.c, P.Z, P.G__minus__, P.H ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFVSS1 ],
                couplings = {(0,0):C.GC_1363})

V_1068 = Vertex(name = 'V_1068',
                particles = [ P.d__tilde__, P.t, P.Z, P.G__minus__, P.H ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFVSS1 ],
                couplings = {(0,0):C.GC_1365})

V_1069 = Vertex(name = 'V_1069',
                particles = [ P.s__tilde__, P.t, P.Z, P.G__minus__, P.H ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFVSS1 ],
                couplings = {(0,0):C.GC_1367})

V_1070 = Vertex(name = 'V_1070',
                particles = [ P.b__tilde__, P.t, P.Z, P.G__minus__, P.H ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFVSS1, L.FFVSS2 ],
                couplings = {(0,0):C.GC_1369,(0,1):C.GC_1346})

V_1071 = Vertex(name = 'V_1071',
                particles = [ P.d__tilde__, P.u, P.Z, P.G__minus__ ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFVS1, L.FFVS2, L.FFVS3, L.FFVS7, L.FFVS8, L.FFVS9 ],
                couplings = {(0,2):C.GC_944,(0,5):C.GC_961,(0,1):C.GC_943,(0,4):C.GC_962,(0,0):C.GC_1478,(0,3):C.GC_1473})

V_1072 = Vertex(name = 'V_1072',
                particles = [ P.s__tilde__, P.u, P.Z, P.G__minus__ ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFVS1, L.FFVS2, L.FFVS3, L.FFVS8, L.FFVS9 ],
                couplings = {(0,2):C.GC_946,(0,4):C.GC_963,(0,1):C.GC_945,(0,3):C.GC_964,(0,0):C.GC_1479})

V_1073 = Vertex(name = 'V_1073',
                particles = [ P.b__tilde__, P.u, P.Z, P.G__minus__ ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFVS1, L.FFVS2, L.FFVS3, L.FFVS8, L.FFVS9 ],
                couplings = {(0,2):C.GC_948,(0,4):C.GC_965,(0,1):C.GC_947,(0,3):C.GC_966,(0,0):C.GC_1480})

V_1074 = Vertex(name = 'V_1074',
                particles = [ P.d__tilde__, P.c, P.Z, P.G__minus__ ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFVS1, L.FFVS2, L.FFVS3, L.FFVS8, L.FFVS9 ],
                couplings = {(0,2):C.GC_950,(0,4):C.GC_967,(0,1):C.GC_949,(0,3):C.GC_968,(0,0):C.GC_1481})

V_1075 = Vertex(name = 'V_1075',
                particles = [ P.s__tilde__, P.c, P.Z, P.G__minus__ ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFVS1, L.FFVS2, L.FFVS3, L.FFVS7, L.FFVS8, L.FFVS9 ],
                couplings = {(0,2):C.GC_952,(0,5):C.GC_969,(0,1):C.GC_951,(0,4):C.GC_970,(0,0):C.GC_1482,(0,3):C.GC_1473})

V_1076 = Vertex(name = 'V_1076',
                particles = [ P.b__tilde__, P.c, P.Z, P.G__minus__ ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFVS1, L.FFVS2, L.FFVS3, L.FFVS8, L.FFVS9 ],
                couplings = {(0,2):C.GC_954,(0,4):C.GC_971,(0,1):C.GC_953,(0,3):C.GC_972,(0,0):C.GC_1483})

V_1077 = Vertex(name = 'V_1077',
                particles = [ P.d__tilde__, P.t, P.Z, P.G__minus__ ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFVS1, L.FFVS2, L.FFVS3, L.FFVS8, L.FFVS9 ],
                couplings = {(0,2):C.GC_956,(0,4):C.GC_973,(0,1):C.GC_955,(0,3):C.GC_974,(0,0):C.GC_1484})

V_1078 = Vertex(name = 'V_1078',
                particles = [ P.s__tilde__, P.t, P.Z, P.G__minus__ ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFVS1, L.FFVS2, L.FFVS3, L.FFVS8, L.FFVS9 ],
                couplings = {(0,2):C.GC_958,(0,4):C.GC_975,(0,1):C.GC_957,(0,3):C.GC_976,(0,0):C.GC_1485})

V_1079 = Vertex(name = 'V_1079',
                particles = [ P.b__tilde__, P.t, P.Z, P.G__minus__ ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFVS1, L.FFVS2, L.FFVS3, L.FFVS7, L.FFVS8, L.FFVS9 ],
                couplings = {(0,2):C.GC_960,(0,5):C.GC_977,(0,1):C.GC_959,(0,4):C.GC_978,(0,0):C.GC_1486,(0,3):C.GC_1473})

V_1080 = Vertex(name = 'V_1080',
                particles = [ P.u__tilde__, P.u, P.Z ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFV1, L.FFV2, L.FFV3, L.FFV5, L.FFV7, L.FFV8 ],
                couplings = {(0,0):C.GC_892,(0,3):C.GC_1789,(0,2):C.GC_2309,(0,5):C.GC_2309,(0,1):C.GC_2310,(0,4):C.GC_2310})

V_1081 = Vertex(name = 'V_1081',
                particles = [ P.u__tilde__, P.u, P.Z ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFV1, L.FFV5 ],
                couplings = {(0,0):C.GC_1792,(0,1):C.GC_886})

V_1082 = Vertex(name = 'V_1082',
                particles = [ P.c__tilde__, P.c, P.Z ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFV1, L.FFV2, L.FFV3, L.FFV5, L.FFV7, L.FFV8 ],
                couplings = {(0,0):C.GC_892,(0,3):C.GC_1789,(0,2):C.GC_1901,(0,5):C.GC_1901,(0,1):C.GC_1902,(0,4):C.GC_1902})

V_1083 = Vertex(name = 'V_1083',
                particles = [ P.c__tilde__, P.c, P.Z ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFV1, L.FFV5 ],
                couplings = {(0,0):C.GC_1792,(0,1):C.GC_886})

V_1084 = Vertex(name = 'V_1084',
                particles = [ P.t__tilde__, P.t, P.Z ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFV1, L.FFV2, L.FFV3, L.FFV5, L.FFV7, L.FFV8 ],
                couplings = {(0,0):C.GC_892,(0,3):C.GC_1789,(0,2):C.GC_2193,(0,5):C.GC_2193,(0,1):C.GC_2194,(0,4):C.GC_2194})

V_1085 = Vertex(name = 'V_1085',
                particles = [ P.t__tilde__, P.t, P.Z ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFV1, L.FFV5 ],
                couplings = {(0,0):C.GC_1792,(0,1):C.GC_886})

V_1086 = Vertex(name = 'V_1086',
                particles = [ P.u__tilde__, P.u, P.Z, P.G0, P.G0 ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFVSS1, L.FFVSS2 ],
                couplings = {(0,0):C.GC_1109,(0,1):C.GC_1100})

V_1087 = Vertex(name = 'V_1087',
                particles = [ P.c__tilde__, P.c, P.Z, P.G0, P.G0 ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFVSS1, L.FFVSS2 ],
                couplings = {(0,0):C.GC_1109,(0,1):C.GC_1100})

V_1088 = Vertex(name = 'V_1088',
                particles = [ P.t__tilde__, P.t, P.Z, P.G0, P.G0 ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFVSS1, L.FFVSS2 ],
                couplings = {(0,0):C.GC_1109,(0,1):C.GC_1100})

V_1089 = Vertex(name = 'V_1089',
                particles = [ P.u__tilde__, P.u, P.Z, P.G__minus__, P.G__plus__ ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFVSS1, L.FFVSS2 ],
                couplings = {(0,0):C.GC_1107,(0,1):C.GC_1101})

V_1090 = Vertex(name = 'V_1090',
                particles = [ P.c__tilde__, P.c, P.Z, P.G__minus__, P.G__plus__ ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFVSS1, L.FFVSS2 ],
                couplings = {(0,0):C.GC_1107,(0,1):C.GC_1101})

V_1091 = Vertex(name = 'V_1091',
                particles = [ P.t__tilde__, P.t, P.Z, P.G__minus__, P.G__plus__ ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFVSS1, L.FFVSS2 ],
                couplings = {(0,0):C.GC_1107,(0,1):C.GC_1101})

V_1092 = Vertex(name = 'V_1092',
                particles = [ P.u__tilde__, P.u, P.Z, P.H, P.H ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFVSS1, L.FFVSS2 ],
                couplings = {(0,0):C.GC_1109,(0,1):C.GC_1100})

V_1093 = Vertex(name = 'V_1093',
                particles = [ P.c__tilde__, P.c, P.Z, P.H, P.H ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFVSS1, L.FFVSS2 ],
                couplings = {(0,0):C.GC_1109,(0,1):C.GC_1100})

V_1094 = Vertex(name = 'V_1094',
                particles = [ P.t__tilde__, P.t, P.Z, P.H, P.H ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFVSS1, L.FFVSS2 ],
                couplings = {(0,0):C.GC_1109,(0,1):C.GC_1100})

V_1095 = Vertex(name = 'V_1095',
                particles = [ P.u__tilde__, P.u, P.Z, P.H ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFVS1, L.FFVS2, L.FFVS3, L.FFVS7, L.FFVS8, L.FFVS9 ],
                couplings = {(0,0):C.GC_1140,(0,3):C.GC_1136,(0,2):C.GC_2305,(0,5):C.GC_2305,(0,1):C.GC_2306,(0,4):C.GC_2306})

V_1096 = Vertex(name = 'V_1096',
                particles = [ P.c__tilde__, P.c, P.Z, P.H ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFVS1, L.FFVS2, L.FFVS3, L.FFVS7, L.FFVS8, L.FFVS9 ],
                couplings = {(0,0):C.GC_1140,(0,3):C.GC_1136,(0,2):C.GC_1897,(0,5):C.GC_1897,(0,1):C.GC_1898,(0,4):C.GC_1898})

V_1097 = Vertex(name = 'V_1097',
                particles = [ P.t__tilde__, P.t, P.Z, P.H ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFVS1, L.FFVS2, L.FFVS3, L.FFVS7, L.FFVS8, L.FFVS9 ],
                couplings = {(0,0):C.GC_1140,(0,3):C.GC_1136,(0,2):C.GC_2189,(0,5):C.GC_2189,(0,1):C.GC_2190,(0,4):C.GC_2190})

V_1098 = Vertex(name = 'V_1098',
                particles = [ P.e__plus__, P.ve, P.Z, P.G0, P.G__minus__ ],
                color = [ '1' ],
                lorentz = [ L.FFVSS1 ],
                couplings = {(0,0):C.GC_1351})

V_1099 = Vertex(name = 'V_1099',
                particles = [ P.mu__plus__, P.vm, P.Z, P.G0, P.G__minus__ ],
                color = [ '1' ],
                lorentz = [ L.FFVSS1 ],
                couplings = {(0,0):C.GC_1351})

V_1100 = Vertex(name = 'V_1100',
                particles = [ P.ta__plus__, P.vt, P.Z, P.G0, P.G__minus__ ],
                color = [ '1' ],
                lorentz = [ L.FFVSS1 ],
                couplings = {(0,0):C.GC_1351})

V_1101 = Vertex(name = 'V_1101',
                particles = [ P.e__plus__, P.ve, P.Z, P.G__minus__, P.H ],
                color = [ '1' ],
                lorentz = [ L.FFVSS1 ],
                couplings = {(0,0):C.GC_1350})

V_1102 = Vertex(name = 'V_1102',
                particles = [ P.mu__plus__, P.vm, P.Z, P.G__minus__, P.H ],
                color = [ '1' ],
                lorentz = [ L.FFVSS1 ],
                couplings = {(0,0):C.GC_1350})

V_1103 = Vertex(name = 'V_1103',
                particles = [ P.ta__plus__, P.vt, P.Z, P.G__minus__, P.H ],
                color = [ '1' ],
                lorentz = [ L.FFVSS1 ],
                couplings = {(0,0):C.GC_1350})

V_1104 = Vertex(name = 'V_1104',
                particles = [ P.e__plus__, P.ve, P.Z, P.G__minus__ ],
                color = [ '1' ],
                lorentz = [ L.FFVS1, L.FFVS8, L.FFVS9 ],
                couplings = {(0,2):C.GC_2015,(0,1):C.GC_2016,(0,0):C.GC_1476})

V_1105 = Vertex(name = 'V_1105',
                particles = [ P.mu__plus__, P.vm, P.Z, P.G__minus__ ],
                color = [ '1' ],
                lorentz = [ L.FFVS1, L.FFVS8, L.FFVS9 ],
                couplings = {(0,2):C.GC_2091,(0,1):C.GC_2092,(0,0):C.GC_1476})

V_1106 = Vertex(name = 'V_1106',
                particles = [ P.ta__plus__, P.vt, P.Z, P.G__minus__ ],
                color = [ '1' ],
                lorentz = [ L.FFVS1, L.FFVS8, L.FFVS9 ],
                couplings = {(0,2):C.GC_2258,(0,1):C.GC_2259,(0,0):C.GC_1476})

V_1107 = Vertex(name = 'V_1107',
                particles = [ P.ve__tilde__, P.ve, P.Z ],
                color = [ '1' ],
                lorentz = [ L.FFV1 ],
                couplings = {(0,0):C.GC_894})

V_1108 = Vertex(name = 'V_1108',
                particles = [ P.ve__tilde__, P.ve, P.Z ],
                color = [ '1' ],
                lorentz = [ L.FFV1 ],
                couplings = {(0,0):C.GC_1788})

V_1109 = Vertex(name = 'V_1109',
                particles = [ P.vm__tilde__, P.vm, P.Z ],
                color = [ '1' ],
                lorentz = [ L.FFV1 ],
                couplings = {(0,0):C.GC_894})

V_1110 = Vertex(name = 'V_1110',
                particles = [ P.vm__tilde__, P.vm, P.Z ],
                color = [ '1' ],
                lorentz = [ L.FFV1 ],
                couplings = {(0,0):C.GC_1788})

V_1111 = Vertex(name = 'V_1111',
                particles = [ P.vt__tilde__, P.vt, P.Z ],
                color = [ '1' ],
                lorentz = [ L.FFV1 ],
                couplings = {(0,0):C.GC_894})

V_1112 = Vertex(name = 'V_1112',
                particles = [ P.vt__tilde__, P.vt, P.Z ],
                color = [ '1' ],
                lorentz = [ L.FFV1 ],
                couplings = {(0,0):C.GC_1788})

V_1113 = Vertex(name = 'V_1113',
                particles = [ P.ve__tilde__, P.ve, P.Z, P.G0, P.G0 ],
                color = [ '1' ],
                lorentz = [ L.FFVSS1 ],
                couplings = {(0,0):C.GC_1105})

V_1114 = Vertex(name = 'V_1114',
                particles = [ P.vm__tilde__, P.vm, P.Z, P.G0, P.G0 ],
                color = [ '1' ],
                lorentz = [ L.FFVSS1 ],
                couplings = {(0,0):C.GC_1105})

V_1115 = Vertex(name = 'V_1115',
                particles = [ P.vt__tilde__, P.vt, P.Z, P.G0, P.G0 ],
                color = [ '1' ],
                lorentz = [ L.FFVSS1 ],
                couplings = {(0,0):C.GC_1105})

V_1116 = Vertex(name = 'V_1116',
                particles = [ P.ve__tilde__, P.ve, P.Z, P.G__minus__, P.G__plus__ ],
                color = [ '1' ],
                lorentz = [ L.FFVSS1 ],
                couplings = {(0,0):C.GC_1103})

V_1117 = Vertex(name = 'V_1117',
                particles = [ P.vm__tilde__, P.vm, P.Z, P.G__minus__, P.G__plus__ ],
                color = [ '1' ],
                lorentz = [ L.FFVSS1 ],
                couplings = {(0,0):C.GC_1103})

V_1118 = Vertex(name = 'V_1118',
                particles = [ P.vt__tilde__, P.vt, P.Z, P.G__minus__, P.G__plus__ ],
                color = [ '1' ],
                lorentz = [ L.FFVSS1 ],
                couplings = {(0,0):C.GC_1103})

V_1119 = Vertex(name = 'V_1119',
                particles = [ P.ve__tilde__, P.ve, P.Z, P.H, P.H ],
                color = [ '1' ],
                lorentz = [ L.FFVSS1 ],
                couplings = {(0,0):C.GC_1105})

V_1120 = Vertex(name = 'V_1120',
                particles = [ P.vm__tilde__, P.vm, P.Z, P.H, P.H ],
                color = [ '1' ],
                lorentz = [ L.FFVSS1 ],
                couplings = {(0,0):C.GC_1105})

V_1121 = Vertex(name = 'V_1121',
                particles = [ P.vt__tilde__, P.vt, P.Z, P.H, P.H ],
                color = [ '1' ],
                lorentz = [ L.FFVSS1 ],
                couplings = {(0,0):C.GC_1105})

V_1122 = Vertex(name = 'V_1122',
                particles = [ P.ve__tilde__, P.ve, P.Z, P.H ],
                color = [ '1' ],
                lorentz = [ L.FFVS1 ],
                couplings = {(0,0):C.GC_1138})

V_1123 = Vertex(name = 'V_1123',
                particles = [ P.vm__tilde__, P.vm, P.Z, P.H ],
                color = [ '1' ],
                lorentz = [ L.FFVS1 ],
                couplings = {(0,0):C.GC_1138})

V_1124 = Vertex(name = 'V_1124',
                particles = [ P.vt__tilde__, P.vt, P.Z, P.H ],
                color = [ '1' ],
                lorentz = [ L.FFVS1 ],
                couplings = {(0,0):C.GC_1138})

V_1125 = Vertex(name = 'V_1125',
                particles = [ P.u__tilde__, P.d, P.W__minus__, P.G__plus__, P.G__plus__ ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFVSS2 ],
                couplings = {(0,0):C.GC_1327})

V_1126 = Vertex(name = 'V_1126',
                particles = [ P.c__tilde__, P.s, P.W__minus__, P.G__plus__, P.G__plus__ ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFVSS2 ],
                couplings = {(0,0):C.GC_1327})

V_1127 = Vertex(name = 'V_1127',
                particles = [ P.t__tilde__, P.b, P.W__minus__, P.G__plus__, P.G__plus__ ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFVSS2 ],
                couplings = {(0,0):C.GC_1327})

V_1128 = Vertex(name = 'V_1128',
                particles = [ P.d__tilde__, P.u, P.W__minus__, P.G0, P.H ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFVSS2 ],
                couplings = {(0,0):C.GC_1324})

V_1129 = Vertex(name = 'V_1129',
                particles = [ P.s__tilde__, P.c, P.W__minus__, P.G0, P.H ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFVSS2 ],
                couplings = {(0,0):C.GC_1324})

V_1130 = Vertex(name = 'V_1130',
                particles = [ P.b__tilde__, P.t, P.W__minus__, P.G0, P.H ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFVSS2 ],
                couplings = {(0,0):C.GC_1324})

V_1131 = Vertex(name = 'V_1131',
                particles = [ P.d__tilde__, P.u, P.W__minus__, P.G0 ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFVS2, L.FFVS3, L.FFVS7, L.FFVS8, L.FFVS9 ],
                couplings = {(0,1):C.GC_759,(0,4):C.GC_822,(0,0):C.GC_761,(0,3):C.GC_824,(0,2):C.GC_1455})

V_1132 = Vertex(name = 'V_1132',
                particles = [ P.s__tilde__, P.u, P.W__minus__, P.G0 ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFVS2, L.FFVS3, L.FFVS8, L.FFVS9 ],
                couplings = {(0,1):C.GC_766,(0,3):C.GC_829,(0,0):C.GC_768,(0,2):C.GC_831})

V_1133 = Vertex(name = 'V_1133',
                particles = [ P.b__tilde__, P.u, P.W__minus__, P.G0 ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFVS2, L.FFVS3, L.FFVS8, L.FFVS9 ],
                couplings = {(0,1):C.GC_773,(0,3):C.GC_836,(0,0):C.GC_775,(0,2):C.GC_838})

V_1134 = Vertex(name = 'V_1134',
                particles = [ P.d__tilde__, P.c, P.W__minus__, P.G0 ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFVS2, L.FFVS3, L.FFVS8, L.FFVS9 ],
                couplings = {(0,1):C.GC_780,(0,3):C.GC_843,(0,0):C.GC_782,(0,2):C.GC_845})

V_1135 = Vertex(name = 'V_1135',
                particles = [ P.s__tilde__, P.c, P.W__minus__, P.G0 ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFVS2, L.FFVS3, L.FFVS7, L.FFVS8, L.FFVS9 ],
                couplings = {(0,1):C.GC_787,(0,4):C.GC_850,(0,0):C.GC_789,(0,3):C.GC_852,(0,2):C.GC_1455})

V_1136 = Vertex(name = 'V_1136',
                particles = [ P.b__tilde__, P.c, P.W__minus__, P.G0 ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFVS2, L.FFVS3, L.FFVS8, L.FFVS9 ],
                couplings = {(0,1):C.GC_794,(0,3):C.GC_857,(0,0):C.GC_796,(0,2):C.GC_859})

V_1137 = Vertex(name = 'V_1137',
                particles = [ P.d__tilde__, P.t, P.W__minus__, P.G0 ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFVS2, L.FFVS3, L.FFVS8, L.FFVS9 ],
                couplings = {(0,1):C.GC_801,(0,3):C.GC_864,(0,0):C.GC_803,(0,2):C.GC_866})

V_1138 = Vertex(name = 'V_1138',
                particles = [ P.s__tilde__, P.t, P.W__minus__, P.G0 ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFVS2, L.FFVS3, L.FFVS8, L.FFVS9 ],
                couplings = {(0,1):C.GC_808,(0,3):C.GC_871,(0,0):C.GC_810,(0,2):C.GC_873})

V_1139 = Vertex(name = 'V_1139',
                particles = [ P.b__tilde__, P.t, P.W__minus__, P.G0 ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFVS2, L.FFVS3, L.FFVS7, L.FFVS8, L.FFVS9 ],
                couplings = {(0,1):C.GC_815,(0,4):C.GC_878,(0,0):C.GC_817,(0,3):C.GC_880,(0,2):C.GC_1455})

V_1140 = Vertex(name = 'V_1140',
                particles = [ P.u__tilde__, P.d, P.W__plus__, P.G0, P.H ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFVSS2 ],
                couplings = {(0,0):C.GC_1328})

V_1141 = Vertex(name = 'V_1141',
                particles = [ P.c__tilde__, P.s, P.W__plus__, P.G0, P.H ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFVSS2 ],
                couplings = {(0,0):C.GC_1328})

V_1142 = Vertex(name = 'V_1142',
                particles = [ P.t__tilde__, P.b, P.W__plus__, P.G0, P.H ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFVSS2 ],
                couplings = {(0,0):C.GC_1328})

V_1143 = Vertex(name = 'V_1143',
                particles = [ P.u__tilde__, P.d, P.W__plus__, P.G0 ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFVS2, L.FFVS3, L.FFVS7, L.FFVS8, L.FFVS9 ],
                couplings = {(0,1):C.GC_618,(0,4):C.GC_690,(0,0):C.GC_615,(0,3):C.GC_687,(0,2):C.GC_1457})

V_1144 = Vertex(name = 'V_1144',
                particles = [ P.c__tilde__, P.d, P.W__plus__, P.G0 ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFVS2, L.FFVS3, L.FFVS8, L.FFVS9 ],
                couplings = {(0,1):C.GC_626,(0,3):C.GC_698,(0,0):C.GC_623,(0,2):C.GC_695})

V_1145 = Vertex(name = 'V_1145',
                particles = [ P.t__tilde__, P.d, P.W__plus__, P.G0 ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFVS2, L.FFVS3, L.FFVS8, L.FFVS9 ],
                couplings = {(0,1):C.GC_634,(0,3):C.GC_706,(0,0):C.GC_631,(0,2):C.GC_703})

V_1146 = Vertex(name = 'V_1146',
                particles = [ P.u__tilde__, P.s, P.W__plus__, P.G0 ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFVS2, L.FFVS3, L.FFVS8, L.FFVS9 ],
                couplings = {(0,1):C.GC_642,(0,3):C.GC_714,(0,0):C.GC_639,(0,2):C.GC_711})

V_1147 = Vertex(name = 'V_1147',
                particles = [ P.c__tilde__, P.s, P.W__plus__, P.G0 ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFVS2, L.FFVS3, L.FFVS7, L.FFVS8, L.FFVS9 ],
                couplings = {(0,1):C.GC_650,(0,4):C.GC_722,(0,0):C.GC_647,(0,3):C.GC_719,(0,2):C.GC_1457})

V_1148 = Vertex(name = 'V_1148',
                particles = [ P.t__tilde__, P.s, P.W__plus__, P.G0 ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFVS2, L.FFVS3, L.FFVS8, L.FFVS9 ],
                couplings = {(0,1):C.GC_658,(0,3):C.GC_730,(0,0):C.GC_655,(0,2):C.GC_727})

V_1149 = Vertex(name = 'V_1149',
                particles = [ P.u__tilde__, P.b, P.W__plus__, P.G0 ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFVS2, L.FFVS3, L.FFVS8, L.FFVS9 ],
                couplings = {(0,1):C.GC_666,(0,3):C.GC_738,(0,0):C.GC_663,(0,2):C.GC_735})

V_1150 = Vertex(name = 'V_1150',
                particles = [ P.c__tilde__, P.b, P.W__plus__, P.G0 ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFVS2, L.FFVS3, L.FFVS8, L.FFVS9 ],
                couplings = {(0,1):C.GC_674,(0,3):C.GC_746,(0,0):C.GC_671,(0,2):C.GC_743})

V_1151 = Vertex(name = 'V_1151',
                particles = [ P.t__tilde__, P.b, P.W__plus__, P.G0 ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFVS2, L.FFVS3, L.FFVS7, L.FFVS8, L.FFVS9 ],
                couplings = {(0,1):C.GC_682,(0,4):C.GC_754,(0,0):C.GC_679,(0,3):C.GC_751,(0,2):C.GC_1457})

V_1152 = Vertex(name = 'V_1152',
                particles = [ P.d__tilde__, P.u, P.W__plus__, P.G__minus__, P.G__minus__ ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFVSS2 ],
                couplings = {(0,0):C.GC_1327})

V_1153 = Vertex(name = 'V_1153',
                particles = [ P.s__tilde__, P.c, P.W__plus__, P.G__minus__, P.G__minus__ ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFVSS2 ],
                couplings = {(0,0):C.GC_1327})

V_1154 = Vertex(name = 'V_1154',
                particles = [ P.b__tilde__, P.t, P.W__plus__, P.G__minus__, P.G__minus__ ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFVSS2 ],
                couplings = {(0,0):C.GC_1327})

V_1155 = Vertex(name = 'V_1155',
                particles = [ P.d__tilde__, P.d, P.a, P.G0 ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFVS2, L.FFVS3, L.FFVS8, L.FFVS9 ],
                couplings = {(0,1):C.GC_1942,(0,3):C.GC_1945,(0,0):C.GC_1945,(0,2):C.GC_1942})

V_1156 = Vertex(name = 'V_1156',
                particles = [ P.s__tilde__, P.s, P.a, P.G0 ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFVS11, L.FFVS2, L.FFVS3, L.FFVS8, L.FFVS9 ],
                couplings = {(0,0):C.GC_2103,(0,2):C.GC_2140,(0,4):C.GC_2102,(0,1):C.GC_2143,(0,3):C.GC_2140})

V_1157 = Vertex(name = 'V_1157',
                particles = [ P.b__tilde__, P.b, P.a, P.G0 ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFVS2, L.FFVS3, L.FFVS8, L.FFVS9 ],
                couplings = {(0,1):C.GC_1848,(0,3):C.GC_1851,(0,0):C.GC_1851,(0,2):C.GC_1848})

V_1158 = Vertex(name = 'V_1158',
                particles = [ P.d__tilde__, P.d, P.a, P.H ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFVS2, L.FFVS3, L.FFVS8, L.FFVS9 ],
                couplings = {(0,1):C.GC_1943,(0,3):C.GC_1943,(0,0):C.GC_1944,(0,2):C.GC_1944})

V_1159 = Vertex(name = 'V_1159',
                particles = [ P.s__tilde__, P.s, P.a, P.H ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFVS2, L.FFVS3, L.FFVS8, L.FFVS9 ],
                couplings = {(0,1):C.GC_2141,(0,3):C.GC_2141,(0,0):C.GC_2142,(0,2):C.GC_2142})

V_1160 = Vertex(name = 'V_1160',
                particles = [ P.b__tilde__, P.b, P.a, P.H ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFVS2, L.FFVS3, L.FFVS8, L.FFVS9 ],
                couplings = {(0,1):C.GC_1849,(0,3):C.GC_1849,(0,0):C.GC_1850,(0,2):C.GC_1850})

V_1161 = Vertex(name = 'V_1161',
                particles = [ P.d__tilde__, P.d, P.Z, P.G0 ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFVS2, L.FFVS3, L.FFVS8, L.FFVS9 ],
                couplings = {(0,1):C.GC_1946,(0,3):C.GC_1949,(0,0):C.GC_1949,(0,2):C.GC_1946})

V_1162 = Vertex(name = 'V_1162',
                particles = [ P.s__tilde__, P.s, P.Z, P.G0 ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFVS2, L.FFVS3, L.FFVS5, L.FFVS8, L.FFVS9 ],
                couplings = {(0,2):C.GC_2120,(0,1):C.GC_2123,(0,4):C.GC_2147,(0,0):C.GC_2147,(0,3):C.GC_2144})

V_1163 = Vertex(name = 'V_1163',
                particles = [ P.b__tilde__, P.b, P.Z, P.G0 ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFVS2, L.FFVS3, L.FFVS8, L.FFVS9 ],
                couplings = {(0,1):C.GC_1852,(0,3):C.GC_1855,(0,0):C.GC_1855,(0,2):C.GC_1852})

V_1164 = Vertex(name = 'V_1164',
                particles = [ P.e__plus__, P.e__minus__, P.a, P.G0 ],
                color = [ '1' ],
                lorentz = [ L.FFVS2, L.FFVS3, L.FFVS8, L.FFVS9 ],
                couplings = {(0,1):C.GC_2011,(0,3):C.GC_2014,(0,0):C.GC_2014,(0,2):C.GC_2011})

V_1165 = Vertex(name = 'V_1165',
                particles = [ P.mu__plus__, P.mu__minus__, P.a, P.G0 ],
                color = [ '1' ],
                lorentz = [ L.FFVS2, L.FFVS3, L.FFVS8, L.FFVS9 ],
                couplings = {(0,1):C.GC_2087,(0,3):C.GC_2090,(0,0):C.GC_2090,(0,2):C.GC_2087})

V_1166 = Vertex(name = 'V_1166',
                particles = [ P.ta__plus__, P.ta__minus__, P.a, P.G0 ],
                color = [ '1' ],
                lorentz = [ L.FFVS2, L.FFVS3, L.FFVS8, L.FFVS9 ],
                couplings = {(0,1):C.GC_2254,(0,3):C.GC_2257,(0,0):C.GC_2257,(0,2):C.GC_2254})

V_1167 = Vertex(name = 'V_1167',
                particles = [ P.e__plus__, P.e__minus__, P.a, P.H ],
                color = [ '1' ],
                lorentz = [ L.FFVS2, L.FFVS3, L.FFVS8, L.FFVS9 ],
                couplings = {(0,1):C.GC_2012,(0,3):C.GC_2012,(0,0):C.GC_2013,(0,2):C.GC_2013})

V_1168 = Vertex(name = 'V_1168',
                particles = [ P.mu__plus__, P.mu__minus__, P.a, P.H ],
                color = [ '1' ],
                lorentz = [ L.FFVS2, L.FFVS3, L.FFVS8, L.FFVS9 ],
                couplings = {(0,1):C.GC_2088,(0,3):C.GC_2088,(0,0):C.GC_2089,(0,2):C.GC_2089})

V_1169 = Vertex(name = 'V_1169',
                particles = [ P.ta__plus__, P.ta__minus__, P.a, P.H ],
                color = [ '1' ],
                lorentz = [ L.FFVS2, L.FFVS3, L.FFVS8, L.FFVS9 ],
                couplings = {(0,1):C.GC_2255,(0,3):C.GC_2255,(0,0):C.GC_2256,(0,2):C.GC_2256})

V_1170 = Vertex(name = 'V_1170',
                particles = [ P.e__plus__, P.e__minus__, P.Z, P.G0 ],
                color = [ '1' ],
                lorentz = [ L.FFVS2, L.FFVS3, L.FFVS8, L.FFVS9 ],
                couplings = {(0,1):C.GC_2017,(0,3):C.GC_2020,(0,0):C.GC_2020,(0,2):C.GC_2017})

V_1171 = Vertex(name = 'V_1171',
                particles = [ P.mu__plus__, P.mu__minus__, P.Z, P.G0 ],
                color = [ '1' ],
                lorentz = [ L.FFVS2, L.FFVS3, L.FFVS8, L.FFVS9 ],
                couplings = {(0,1):C.GC_2093,(0,3):C.GC_2096,(0,0):C.GC_2096,(0,2):C.GC_2093})

V_1172 = Vertex(name = 'V_1172',
                particles = [ P.ta__plus__, P.ta__minus__, P.Z, P.G0 ],
                color = [ '1' ],
                lorentz = [ L.FFVS2, L.FFVS3, L.FFVS8, L.FFVS9 ],
                couplings = {(0,1):C.GC_2260,(0,3):C.GC_2263,(0,0):C.GC_2263,(0,2):C.GC_2260})

V_1173 = Vertex(name = 'V_1173',
                particles = [ P.ve__tilde__, P.e__minus__, P.W__plus__, P.G0 ],
                color = [ '1' ],
                lorentz = [ L.FFVS2, L.FFVS3 ],
                couplings = {(0,1):C.GC_1975,(0,0):C.GC_1972})

V_1174 = Vertex(name = 'V_1174',
                particles = [ P.vm__tilde__, P.mu__minus__, P.W__plus__, P.G0 ],
                color = [ '1' ],
                lorentz = [ L.FFVS2, L.FFVS3 ],
                couplings = {(0,1):C.GC_2047,(0,0):C.GC_2044})

V_1175 = Vertex(name = 'V_1175',
                particles = [ P.vt__tilde__, P.ta__minus__, P.W__plus__, P.G0 ],
                color = [ '1' ],
                lorentz = [ L.FFVS2, L.FFVS3 ],
                couplings = {(0,1):C.GC_2218,(0,0):C.GC_2215})

V_1176 = Vertex(name = 'V_1176',
                particles = [ P.u__tilde__, P.u, P.a, P.G0 ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFVS2, L.FFVS3, L.FFVS8, L.FFVS9 ],
                couplings = {(0,1):C.GC_2300,(0,3):C.GC_2303,(0,0):C.GC_2303,(0,2):C.GC_2300})

V_1177 = Vertex(name = 'V_1177',
                particles = [ P.c__tilde__, P.c, P.a, P.G0 ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFVS2, L.FFVS3, L.FFVS8, L.FFVS9 ],
                couplings = {(0,1):C.GC_1892,(0,3):C.GC_1895,(0,0):C.GC_1895,(0,2):C.GC_1892})

V_1178 = Vertex(name = 'V_1178',
                particles = [ P.t__tilde__, P.t, P.a, P.G0 ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFVS2, L.FFVS3, L.FFVS8, L.FFVS9 ],
                couplings = {(0,1):C.GC_2184,(0,3):C.GC_2187,(0,0):C.GC_2187,(0,2):C.GC_2184})

V_1179 = Vertex(name = 'V_1179',
                particles = [ P.u__tilde__, P.u, P.a, P.H ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFVS2, L.FFVS3, L.FFVS8, L.FFVS9 ],
                couplings = {(0,1):C.GC_2302,(0,3):C.GC_2302,(0,0):C.GC_2301,(0,2):C.GC_2301})

V_1180 = Vertex(name = 'V_1180',
                particles = [ P.c__tilde__, P.c, P.a, P.H ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFVS2, L.FFVS3, L.FFVS8, L.FFVS9 ],
                couplings = {(0,1):C.GC_1894,(0,3):C.GC_1894,(0,0):C.GC_1893,(0,2):C.GC_1893})

V_1181 = Vertex(name = 'V_1181',
                particles = [ P.t__tilde__, P.t, P.a, P.H ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFVS2, L.FFVS3, L.FFVS8, L.FFVS9 ],
                couplings = {(0,1):C.GC_2186,(0,3):C.GC_2186,(0,0):C.GC_2185,(0,2):C.GC_2185})

V_1182 = Vertex(name = 'V_1182',
                particles = [ P.u__tilde__, P.u, P.Z, P.G0 ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFVS2, L.FFVS3, L.FFVS8, L.FFVS9 ],
                couplings = {(0,1):C.GC_2307,(0,3):C.GC_2304,(0,0):C.GC_2304,(0,2):C.GC_2307})

V_1183 = Vertex(name = 'V_1183',
                particles = [ P.c__tilde__, P.c, P.Z, P.G0 ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFVS2, L.FFVS3, L.FFVS8, L.FFVS9 ],
                couplings = {(0,1):C.GC_1899,(0,3):C.GC_1896,(0,0):C.GC_1896,(0,2):C.GC_1899})

V_1184 = Vertex(name = 'V_1184',
                particles = [ P.t__tilde__, P.t, P.Z, P.G0 ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFVS2, L.FFVS3, L.FFVS8, L.FFVS9 ],
                couplings = {(0,1):C.GC_2191,(0,3):C.GC_2188,(0,0):C.GC_2188,(0,2):C.GC_2191})

V_1185 = Vertex(name = 'V_1185',
                particles = [ P.d__tilde__, P.d, P.g, P.G0 ],
                color = [ 'T(3,2,1)' ],
                lorentz = [ L.FFVS2, L.FFVS3, L.FFVS8, L.FFVS9 ],
                couplings = {(0,1):C.GC_1909,(0,3):C.GC_1906,(0,0):C.GC_1906,(0,2):C.GC_1909})

V_1186 = Vertex(name = 'V_1186',
                particles = [ P.s__tilde__, P.s, P.g, P.G0 ],
                color = [ 'T(3,2,1)' ],
                lorentz = [ L.FFVS2, L.FFVS3, L.FFVS8, L.FFVS9 ],
                couplings = {(0,1):C.GC_2107,(0,3):C.GC_2104,(0,0):C.GC_2104,(0,2):C.GC_2107})

V_1187 = Vertex(name = 'V_1187',
                particles = [ P.b__tilde__, P.b, P.g, P.G0 ],
                color = [ 'T(3,2,1)' ],
                lorentz = [ L.FFVS2, L.FFVS3, L.FFVS8, L.FFVS9 ],
                couplings = {(0,1):C.GC_1817,(0,3):C.GC_1814,(0,0):C.GC_1814,(0,2):C.GC_1817})

V_1188 = Vertex(name = 'V_1188',
                particles = [ P.d__tilde__, P.d, P.g, P.H ],
                color = [ 'T(3,2,1)' ],
                lorentz = [ L.FFVS2, L.FFVS3, L.FFVS8, L.FFVS9 ],
                couplings = {(0,1):C.GC_1908,(0,3):C.GC_1908,(0,0):C.GC_1907,(0,2):C.GC_1907})

V_1189 = Vertex(name = 'V_1189',
                particles = [ P.s__tilde__, P.s, P.g, P.H ],
                color = [ 'T(3,2,1)' ],
                lorentz = [ L.FFVS2, L.FFVS3, L.FFVS8, L.FFVS9 ],
                couplings = {(0,1):C.GC_2106,(0,3):C.GC_2106,(0,0):C.GC_2105,(0,2):C.GC_2105})

V_1190 = Vertex(name = 'V_1190',
                particles = [ P.b__tilde__, P.b, P.g, P.H ],
                color = [ 'T(3,2,1)' ],
                lorentz = [ L.FFVS2, L.FFVS3, L.FFVS8, L.FFVS9 ],
                couplings = {(0,1):C.GC_1816,(0,3):C.GC_1816,(0,0):C.GC_1815,(0,2):C.GC_1815})

V_1191 = Vertex(name = 'V_1191',
                particles = [ P.u__tilde__, P.d, P.g, P.G__plus__ ],
                color = [ 'T(3,2,1)' ],
                lorentz = [ L.FFVS2, L.FFVS3, L.FFVS8, L.FFVS9 ],
                couplings = {(0,1):C.GC_174,(0,3):C.GC_211,(0,0):C.GC_175,(0,2):C.GC_210})

V_1192 = Vertex(name = 'V_1192',
                particles = [ P.c__tilde__, P.d, P.g, P.G__plus__ ],
                color = [ 'T(3,2,1)' ],
                lorentz = [ L.FFVS2, L.FFVS3, L.FFVS8, L.FFVS9 ],
                couplings = {(0,1):C.GC_178,(0,3):C.GC_215,(0,0):C.GC_179,(0,2):C.GC_214})

V_1193 = Vertex(name = 'V_1193',
                particles = [ P.t__tilde__, P.d, P.g, P.G__plus__ ],
                color = [ 'T(3,2,1)' ],
                lorentz = [ L.FFVS2, L.FFVS3, L.FFVS8, L.FFVS9 ],
                couplings = {(0,1):C.GC_182,(0,3):C.GC_219,(0,0):C.GC_183,(0,2):C.GC_218})

V_1194 = Vertex(name = 'V_1194',
                particles = [ P.u__tilde__, P.s, P.g, P.G__plus__ ],
                color = [ 'T(3,2,1)' ],
                lorentz = [ L.FFVS2, L.FFVS3, L.FFVS8, L.FFVS9 ],
                couplings = {(0,1):C.GC_186,(0,3):C.GC_223,(0,0):C.GC_187,(0,2):C.GC_222})

V_1195 = Vertex(name = 'V_1195',
                particles = [ P.c__tilde__, P.s, P.g, P.G__plus__ ],
                color = [ 'T(3,2,1)' ],
                lorentz = [ L.FFVS2, L.FFVS3, L.FFVS8, L.FFVS9 ],
                couplings = {(0,1):C.GC_190,(0,3):C.GC_227,(0,0):C.GC_191,(0,2):C.GC_226})

V_1196 = Vertex(name = 'V_1196',
                particles = [ P.t__tilde__, P.s, P.g, P.G__plus__ ],
                color = [ 'T(3,2,1)' ],
                lorentz = [ L.FFVS2, L.FFVS3, L.FFVS8, L.FFVS9 ],
                couplings = {(0,1):C.GC_194,(0,3):C.GC_231,(0,0):C.GC_195,(0,2):C.GC_230})

V_1197 = Vertex(name = 'V_1197',
                particles = [ P.u__tilde__, P.b, P.g, P.G__plus__ ],
                color = [ 'T(3,2,1)' ],
                lorentz = [ L.FFVS2, L.FFVS3, L.FFVS8, L.FFVS9 ],
                couplings = {(0,1):C.GC_198,(0,3):C.GC_235,(0,0):C.GC_199,(0,2):C.GC_234})

V_1198 = Vertex(name = 'V_1198',
                particles = [ P.c__tilde__, P.b, P.g, P.G__plus__ ],
                color = [ 'T(3,2,1)' ],
                lorentz = [ L.FFVS2, L.FFVS3, L.FFVS8, L.FFVS9 ],
                couplings = {(0,1):C.GC_202,(0,3):C.GC_239,(0,0):C.GC_203,(0,2):C.GC_238})

V_1199 = Vertex(name = 'V_1199',
                particles = [ P.t__tilde__, P.b, P.g, P.G__plus__ ],
                color = [ 'T(3,2,1)' ],
                lorentz = [ L.FFVS2, L.FFVS3, L.FFVS8, L.FFVS9 ],
                couplings = {(0,1):C.GC_206,(0,3):C.GC_243,(0,0):C.GC_207,(0,2):C.GC_242})

V_1200 = Vertex(name = 'V_1200',
                particles = [ P.u__tilde__, P.u, P.g, P.G0 ],
                color = [ 'T(3,2,1)' ],
                lorentz = [ L.FFVS2, L.FFVS3, L.FFVS8, L.FFVS9 ],
                couplings = {(0,1):C.GC_2270,(0,3):C.GC_2273,(0,0):C.GC_2273,(0,2):C.GC_2270})

V_1201 = Vertex(name = 'V_1201',
                particles = [ P.c__tilde__, P.c, P.g, P.G0 ],
                color = [ 'T(3,2,1)' ],
                lorentz = [ L.FFVS2, L.FFVS3, L.FFVS8, L.FFVS9 ],
                couplings = {(0,1):C.GC_1862,(0,3):C.GC_1865,(0,0):C.GC_1865,(0,2):C.GC_1862})

V_1202 = Vertex(name = 'V_1202',
                particles = [ P.t__tilde__, P.t, P.g, P.G0 ],
                color = [ 'T(3,2,1)' ],
                lorentz = [ L.FFVS2, L.FFVS3, L.FFVS8, L.FFVS9 ],
                couplings = {(0,1):C.GC_2154,(0,3):C.GC_2157,(0,0):C.GC_2157,(0,2):C.GC_2154})

V_1203 = Vertex(name = 'V_1203',
                particles = [ P.u__tilde__, P.u, P.g, P.H ],
                color = [ 'T(3,2,1)' ],
                lorentz = [ L.FFVS2, L.FFVS3, L.FFVS8, L.FFVS9 ],
                couplings = {(0,1):C.GC_2272,(0,3):C.GC_2272,(0,0):C.GC_2271,(0,2):C.GC_2271})

V_1204 = Vertex(name = 'V_1204',
                particles = [ P.c__tilde__, P.c, P.g, P.H ],
                color = [ 'T(3,2,1)' ],
                lorentz = [ L.FFVS2, L.FFVS3, L.FFVS8, L.FFVS9 ],
                couplings = {(0,1):C.GC_1864,(0,3):C.GC_1864,(0,0):C.GC_1863,(0,2):C.GC_1863})

V_1205 = Vertex(name = 'V_1205',
                particles = [ P.t__tilde__, P.t, P.g, P.H ],
                color = [ 'T(3,2,1)' ],
                lorentz = [ L.FFVS2, L.FFVS3, L.FFVS8, L.FFVS9 ],
                couplings = {(0,1):C.GC_2156,(0,3):C.GC_2156,(0,0):C.GC_2155,(0,2):C.GC_2155})

V_1206 = Vertex(name = 'V_1206',
                particles = [ P.d__tilde__, P.u, P.g, P.G__minus__ ],
                color = [ 'T(3,2,1)' ],
                lorentz = [ L.FFVS2, L.FFVS3, L.FFVS8, L.FFVS9 ],
                couplings = {(0,1):C.GC_246,(0,3):C.GC_283,(0,0):C.GC_247,(0,2):C.GC_282})

V_1207 = Vertex(name = 'V_1207',
                particles = [ P.s__tilde__, P.u, P.g, P.G__minus__ ],
                color = [ 'T(3,2,1)' ],
                lorentz = [ L.FFVS2, L.FFVS3, L.FFVS8, L.FFVS9 ],
                couplings = {(0,1):C.GC_250,(0,3):C.GC_287,(0,0):C.GC_251,(0,2):C.GC_286})

V_1208 = Vertex(name = 'V_1208',
                particles = [ P.b__tilde__, P.u, P.g, P.G__minus__ ],
                color = [ 'T(3,2,1)' ],
                lorentz = [ L.FFVS2, L.FFVS3, L.FFVS8, L.FFVS9 ],
                couplings = {(0,1):C.GC_254,(0,3):C.GC_291,(0,0):C.GC_255,(0,2):C.GC_290})

V_1209 = Vertex(name = 'V_1209',
                particles = [ P.d__tilde__, P.c, P.g, P.G__minus__ ],
                color = [ 'T(3,2,1)' ],
                lorentz = [ L.FFVS2, L.FFVS3, L.FFVS8, L.FFVS9 ],
                couplings = {(0,1):C.GC_258,(0,3):C.GC_295,(0,0):C.GC_259,(0,2):C.GC_294})

V_1210 = Vertex(name = 'V_1210',
                particles = [ P.s__tilde__, P.c, P.g, P.G__minus__ ],
                color = [ 'T(3,2,1)' ],
                lorentz = [ L.FFVS2, L.FFVS3, L.FFVS8, L.FFVS9 ],
                couplings = {(0,1):C.GC_262,(0,3):C.GC_299,(0,0):C.GC_263,(0,2):C.GC_298})

V_1211 = Vertex(name = 'V_1211',
                particles = [ P.b__tilde__, P.c, P.g, P.G__minus__ ],
                color = [ 'T(3,2,1)' ],
                lorentz = [ L.FFVS2, L.FFVS3, L.FFVS8, L.FFVS9 ],
                couplings = {(0,1):C.GC_266,(0,3):C.GC_303,(0,0):C.GC_267,(0,2):C.GC_302})

V_1212 = Vertex(name = 'V_1212',
                particles = [ P.d__tilde__, P.t, P.g, P.G__minus__ ],
                color = [ 'T(3,2,1)' ],
                lorentz = [ L.FFVS2, L.FFVS3, L.FFVS8, L.FFVS9 ],
                couplings = {(0,1):C.GC_270,(0,3):C.GC_307,(0,0):C.GC_271,(0,2):C.GC_306})

V_1213 = Vertex(name = 'V_1213',
                particles = [ P.s__tilde__, P.t, P.g, P.G__minus__ ],
                color = [ 'T(3,2,1)' ],
                lorentz = [ L.FFVS2, L.FFVS3, L.FFVS8, L.FFVS9 ],
                couplings = {(0,1):C.GC_274,(0,3):C.GC_311,(0,0):C.GC_275,(0,2):C.GC_310})

V_1214 = Vertex(name = 'V_1214',
                particles = [ P.b__tilde__, P.t, P.g, P.G__minus__ ],
                color = [ 'T(3,2,1)' ],
                lorentz = [ L.FFVS2, L.FFVS3, L.FFVS8, L.FFVS9 ],
                couplings = {(0,1):C.GC_278,(0,3):C.GC_315,(0,0):C.GC_279,(0,2):C.GC_314})

V_1215 = Vertex(name = 'V_1215',
                particles = [ P.d__tilde__, P.d, P.g, P.g, P.G0 ],
                color = [ 'f(-1,3,4)*T(-1,2,1)' ],
                lorentz = [ L.FFVVS1, L.FFVVS2, L.FFVVS6, L.FFVVS7 ],
                couplings = {(0,1):C.GC_1912,(0,0):C.GC_1911,(0,3):C.GC_1911,(0,2):C.GC_1912})

V_1216 = Vertex(name = 'V_1216',
                particles = [ P.s__tilde__, P.s, P.g, P.g, P.G0 ],
                color = [ 'f(-1,3,4)*T(-1,2,1)' ],
                lorentz = [ L.FFVVS1, L.FFVVS2, L.FFVVS6, L.FFVVS7 ],
                couplings = {(0,1):C.GC_2110,(0,0):C.GC_2109,(0,3):C.GC_2109,(0,2):C.GC_2110})

V_1217 = Vertex(name = 'V_1217',
                particles = [ P.b__tilde__, P.b, P.g, P.g, P.G0 ],
                color = [ 'f(-1,3,4)*T(-1,2,1)' ],
                lorentz = [ L.FFVVS1, L.FFVVS2, L.FFVVS6, L.FFVVS7 ],
                couplings = {(0,1):C.GC_1820,(0,0):C.GC_1819,(0,3):C.GC_1819,(0,2):C.GC_1820})

V_1218 = Vertex(name = 'V_1218',
                particles = [ P.d__tilde__, P.d, P.g, P.g, P.H ],
                color = [ 'f(-1,3,4)*T(-1,2,1)' ],
                lorentz = [ L.FFVVS1, L.FFVVS2, L.FFVVS6, L.FFVVS7 ],
                couplings = {(0,1):C.GC_1910,(0,0):C.GC_1913,(0,3):C.GC_1910,(0,2):C.GC_1913})

V_1219 = Vertex(name = 'V_1219',
                particles = [ P.s__tilde__, P.s, P.g, P.g, P.H ],
                color = [ 'f(-1,3,4)*T(-1,2,1)' ],
                lorentz = [ L.FFVVS1, L.FFVVS2, L.FFVVS6, L.FFVVS7 ],
                couplings = {(0,1):C.GC_2108,(0,0):C.GC_2111,(0,3):C.GC_2108,(0,2):C.GC_2111})

V_1220 = Vertex(name = 'V_1220',
                particles = [ P.b__tilde__, P.b, P.g, P.g, P.H ],
                color = [ 'f(-1,3,4)*T(-1,2,1)' ],
                lorentz = [ L.FFVVS1, L.FFVVS2, L.FFVVS6, L.FFVVS7 ],
                couplings = {(0,1):C.GC_1818,(0,0):C.GC_1821,(0,3):C.GC_1818,(0,2):C.GC_1821})

V_1221 = Vertex(name = 'V_1221',
                particles = [ P.d__tilde__, P.d, P.g, P.g ],
                color = [ 'f(-1,3,4)*T(-1,2,1)' ],
                lorentz = [ L.FFVV1, L.FFVV2, L.FFVV5, L.FFVV6 ],
                couplings = {(0,1):C.GC_1937,(0,0):C.GC_1938,(0,3):C.GC_1937,(0,2):C.GC_1938})

V_1222 = Vertex(name = 'V_1222',
                particles = [ P.s__tilde__, P.s, P.g, P.g ],
                color = [ 'f(-1,3,4)*T(-1,2,1)' ],
                lorentz = [ L.FFVV1, L.FFVV2, L.FFVV5, L.FFVV6 ],
                couplings = {(0,1):C.GC_2135,(0,0):C.GC_2136,(0,3):C.GC_2135,(0,2):C.GC_2136})

V_1223 = Vertex(name = 'V_1223',
                particles = [ P.b__tilde__, P.b, P.g, P.g ],
                color = [ 'f(-1,3,4)*T(-1,2,1)' ],
                lorentz = [ L.FFVV1, L.FFVV2, L.FFVV5, L.FFVV6 ],
                couplings = {(0,1):C.GC_1843,(0,0):C.GC_1844,(0,3):C.GC_1843,(0,2):C.GC_1844})

V_1224 = Vertex(name = 'V_1224',
                particles = [ P.u__tilde__, P.d, P.g, P.g, P.G__plus__ ],
                color = [ 'f(-1,3,4)*T(-1,2,1)' ],
                lorentz = [ L.FFVVS1, L.FFVVS2, L.FFVVS6, L.FFVVS7 ],
                couplings = {(0,1):C.GC_176,(0,0):C.GC_177,(0,3):C.GC_213,(0,2):C.GC_212})

V_1225 = Vertex(name = 'V_1225',
                particles = [ P.c__tilde__, P.d, P.g, P.g, P.G__plus__ ],
                color = [ 'f(-1,3,4)*T(-1,2,1)' ],
                lorentz = [ L.FFVVS1, L.FFVVS2, L.FFVVS6, L.FFVVS7 ],
                couplings = {(0,1):C.GC_180,(0,0):C.GC_181,(0,3):C.GC_217,(0,2):C.GC_216})

V_1226 = Vertex(name = 'V_1226',
                particles = [ P.t__tilde__, P.d, P.g, P.g, P.G__plus__ ],
                color = [ 'f(-1,3,4)*T(-1,2,1)' ],
                lorentz = [ L.FFVVS1, L.FFVVS2, L.FFVVS6, L.FFVVS7 ],
                couplings = {(0,1):C.GC_184,(0,0):C.GC_185,(0,3):C.GC_221,(0,2):C.GC_220})

V_1227 = Vertex(name = 'V_1227',
                particles = [ P.u__tilde__, P.s, P.g, P.g, P.G__plus__ ],
                color = [ 'f(-1,3,4)*T(-1,2,1)' ],
                lorentz = [ L.FFVVS1, L.FFVVS2, L.FFVVS6, L.FFVVS7 ],
                couplings = {(0,1):C.GC_188,(0,0):C.GC_189,(0,3):C.GC_225,(0,2):C.GC_224})

V_1228 = Vertex(name = 'V_1228',
                particles = [ P.c__tilde__, P.s, P.g, P.g, P.G__plus__ ],
                color = [ 'f(-1,3,4)*T(-1,2,1)' ],
                lorentz = [ L.FFVVS1, L.FFVVS2, L.FFVVS6, L.FFVVS7 ],
                couplings = {(0,1):C.GC_192,(0,0):C.GC_193,(0,3):C.GC_229,(0,2):C.GC_228})

V_1229 = Vertex(name = 'V_1229',
                particles = [ P.t__tilde__, P.s, P.g, P.g, P.G__plus__ ],
                color = [ 'f(-1,3,4)*T(-1,2,1)' ],
                lorentz = [ L.FFVVS1, L.FFVVS2, L.FFVVS6, L.FFVVS7 ],
                couplings = {(0,1):C.GC_196,(0,0):C.GC_197,(0,3):C.GC_233,(0,2):C.GC_232})

V_1230 = Vertex(name = 'V_1230',
                particles = [ P.u__tilde__, P.b, P.g, P.g, P.G__plus__ ],
                color = [ 'f(-1,3,4)*T(-1,2,1)' ],
                lorentz = [ L.FFVVS1, L.FFVVS2, L.FFVVS6, L.FFVVS7 ],
                couplings = {(0,1):C.GC_200,(0,0):C.GC_201,(0,3):C.GC_237,(0,2):C.GC_236})

V_1231 = Vertex(name = 'V_1231',
                particles = [ P.c__tilde__, P.b, P.g, P.g, P.G__plus__ ],
                color = [ 'f(-1,3,4)*T(-1,2,1)' ],
                lorentz = [ L.FFVVS1, L.FFVVS2, L.FFVVS6, L.FFVVS7 ],
                couplings = {(0,1):C.GC_204,(0,0):C.GC_205,(0,3):C.GC_241,(0,2):C.GC_240})

V_1232 = Vertex(name = 'V_1232',
                particles = [ P.t__tilde__, P.b, P.g, P.g, P.G__plus__ ],
                color = [ 'f(-1,3,4)*T(-1,2,1)' ],
                lorentz = [ L.FFVVS1, L.FFVVS2, L.FFVVS6, L.FFVVS7 ],
                couplings = {(0,1):C.GC_208,(0,0):C.GC_209,(0,3):C.GC_245,(0,2):C.GC_244})

V_1233 = Vertex(name = 'V_1233',
                particles = [ P.u__tilde__, P.u, P.g, P.g, P.G0 ],
                color = [ 'f(-1,3,4)*T(-1,2,1)' ],
                lorentz = [ L.FFVVS1, L.FFVVS2, L.FFVVS6, L.FFVVS7 ],
                couplings = {(0,1):C.GC_2275,(0,0):C.GC_2276,(0,3):C.GC_2276,(0,2):C.GC_2275})

V_1234 = Vertex(name = 'V_1234',
                particles = [ P.c__tilde__, P.c, P.g, P.g, P.G0 ],
                color = [ 'f(-1,3,4)*T(-1,2,1)' ],
                lorentz = [ L.FFVVS1, L.FFVVS2, L.FFVVS6, L.FFVVS7 ],
                couplings = {(0,1):C.GC_1867,(0,0):C.GC_1868,(0,3):C.GC_1868,(0,2):C.GC_1867})

V_1235 = Vertex(name = 'V_1235',
                particles = [ P.t__tilde__, P.t, P.g, P.g, P.G0 ],
                color = [ 'f(-1,3,4)*T(-1,2,1)' ],
                lorentz = [ L.FFVVS1, L.FFVVS2, L.FFVVS6, L.FFVVS7 ],
                couplings = {(0,1):C.GC_2159,(0,0):C.GC_2160,(0,3):C.GC_2160,(0,2):C.GC_2159})

V_1236 = Vertex(name = 'V_1236',
                particles = [ P.u__tilde__, P.u, P.g, P.g, P.H ],
                color = [ 'f(-1,3,4)*T(-1,2,1)' ],
                lorentz = [ L.FFVVS1, L.FFVVS2, L.FFVVS6, L.FFVVS7 ],
                couplings = {(0,1):C.GC_2274,(0,0):C.GC_2277,(0,3):C.GC_2274,(0,2):C.GC_2277})

V_1237 = Vertex(name = 'V_1237',
                particles = [ P.c__tilde__, P.c, P.g, P.g, P.H ],
                color = [ 'f(-1,3,4)*T(-1,2,1)' ],
                lorentz = [ L.FFVVS1, L.FFVVS2, L.FFVVS6, L.FFVVS7 ],
                couplings = {(0,1):C.GC_1866,(0,0):C.GC_1869,(0,3):C.GC_1866,(0,2):C.GC_1869})

V_1238 = Vertex(name = 'V_1238',
                particles = [ P.t__tilde__, P.t, P.g, P.g, P.H ],
                color = [ 'f(-1,3,4)*T(-1,2,1)' ],
                lorentz = [ L.FFVVS1, L.FFVVS2, L.FFVVS6, L.FFVVS7 ],
                couplings = {(0,1):C.GC_2158,(0,0):C.GC_2161,(0,3):C.GC_2158,(0,2):C.GC_2161})

V_1239 = Vertex(name = 'V_1239',
                particles = [ P.u__tilde__, P.u, P.g, P.g ],
                color = [ 'f(-1,3,4)*T(-1,2,1)' ],
                lorentz = [ L.FFVV1, L.FFVV2, L.FFVV5, L.FFVV6 ],
                couplings = {(0,1):C.GC_2295,(0,0):C.GC_2296,(0,3):C.GC_2295,(0,2):C.GC_2296})

V_1240 = Vertex(name = 'V_1240',
                particles = [ P.c__tilde__, P.c, P.g, P.g ],
                color = [ 'f(-1,3,4)*T(-1,2,1)' ],
                lorentz = [ L.FFVV1, L.FFVV2, L.FFVV5, L.FFVV6 ],
                couplings = {(0,1):C.GC_1887,(0,0):C.GC_1888,(0,3):C.GC_1887,(0,2):C.GC_1888})

V_1241 = Vertex(name = 'V_1241',
                particles = [ P.t__tilde__, P.t, P.g, P.g ],
                color = [ 'f(-1,3,4)*T(-1,2,1)' ],
                lorentz = [ L.FFVV1, L.FFVV2, L.FFVV5, L.FFVV6 ],
                couplings = {(0,1):C.GC_2179,(0,0):C.GC_2180,(0,3):C.GC_2179,(0,2):C.GC_2180})

V_1242 = Vertex(name = 'V_1242',
                particles = [ P.d__tilde__, P.u, P.g, P.g, P.G__minus__ ],
                color = [ 'f(-1,3,4)*T(-1,2,1)' ],
                lorentz = [ L.FFVVS1, L.FFVVS2, L.FFVVS6, L.FFVVS7 ],
                couplings = {(0,1):C.GC_248,(0,0):C.GC_249,(0,3):C.GC_285,(0,2):C.GC_284})

V_1243 = Vertex(name = 'V_1243',
                particles = [ P.s__tilde__, P.u, P.g, P.g, P.G__minus__ ],
                color = [ 'f(-1,3,4)*T(-1,2,1)' ],
                lorentz = [ L.FFVVS1, L.FFVVS2, L.FFVVS6, L.FFVVS7 ],
                couplings = {(0,1):C.GC_252,(0,0):C.GC_253,(0,3):C.GC_289,(0,2):C.GC_288})

V_1244 = Vertex(name = 'V_1244',
                particles = [ P.b__tilde__, P.u, P.g, P.g, P.G__minus__ ],
                color = [ 'f(-1,3,4)*T(-1,2,1)' ],
                lorentz = [ L.FFVVS1, L.FFVVS2, L.FFVVS6, L.FFVVS7 ],
                couplings = {(0,1):C.GC_256,(0,0):C.GC_257,(0,3):C.GC_293,(0,2):C.GC_292})

V_1245 = Vertex(name = 'V_1245',
                particles = [ P.d__tilde__, P.c, P.g, P.g, P.G__minus__ ],
                color = [ 'f(-1,3,4)*T(-1,2,1)' ],
                lorentz = [ L.FFVVS1, L.FFVVS2, L.FFVVS6, L.FFVVS7 ],
                couplings = {(0,1):C.GC_260,(0,0):C.GC_261,(0,3):C.GC_297,(0,2):C.GC_296})

V_1246 = Vertex(name = 'V_1246',
                particles = [ P.s__tilde__, P.c, P.g, P.g, P.G__minus__ ],
                color = [ 'f(-1,3,4)*T(-1,2,1)' ],
                lorentz = [ L.FFVVS1, L.FFVVS2, L.FFVVS6, L.FFVVS7 ],
                couplings = {(0,1):C.GC_264,(0,0):C.GC_265,(0,3):C.GC_301,(0,2):C.GC_300})

V_1247 = Vertex(name = 'V_1247',
                particles = [ P.b__tilde__, P.c, P.g, P.g, P.G__minus__ ],
                color = [ 'f(-1,3,4)*T(-1,2,1)' ],
                lorentz = [ L.FFVVS1, L.FFVVS2, L.FFVVS6, L.FFVVS7 ],
                couplings = {(0,1):C.GC_268,(0,0):C.GC_269,(0,3):C.GC_305,(0,2):C.GC_304})

V_1248 = Vertex(name = 'V_1248',
                particles = [ P.d__tilde__, P.t, P.g, P.g, P.G__minus__ ],
                color = [ 'f(-1,3,4)*T(-1,2,1)' ],
                lorentz = [ L.FFVVS1, L.FFVVS2, L.FFVVS6, L.FFVVS7 ],
                couplings = {(0,1):C.GC_272,(0,0):C.GC_273,(0,3):C.GC_309,(0,2):C.GC_308})

V_1249 = Vertex(name = 'V_1249',
                particles = [ P.s__tilde__, P.t, P.g, P.g, P.G__minus__ ],
                color = [ 'f(-1,3,4)*T(-1,2,1)' ],
                lorentz = [ L.FFVVS1, L.FFVVS2, L.FFVVS6, L.FFVVS7 ],
                couplings = {(0,1):C.GC_276,(0,0):C.GC_277,(0,3):C.GC_313,(0,2):C.GC_312})

V_1250 = Vertex(name = 'V_1250',
                particles = [ P.b__tilde__, P.t, P.g, P.g, P.G__minus__ ],
                color = [ 'f(-1,3,4)*T(-1,2,1)' ],
                lorentz = [ L.FFVVS1, L.FFVVS2, L.FFVVS6, L.FFVVS7 ],
                couplings = {(0,1):C.GC_280,(0,0):C.GC_281,(0,3):C.GC_317,(0,2):C.GC_316})

V_1251 = Vertex(name = 'V_1251',
                particles = [ P.d__tilde__, P.d, P.a, P.W__minus__, P.G__plus__ ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFVVS1, L.FFVVS2 ],
                couplings = {(0,1):C.GC_1922,(0,0):C.GC_1923})

V_1252 = Vertex(name = 'V_1252',
                particles = [ P.s__tilde__, P.s, P.a, P.W__minus__, P.G__plus__ ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFVVS1, L.FFVVS2 ],
                couplings = {(0,1):C.GC_2121,(0,0):C.GC_2122})

V_1253 = Vertex(name = 'V_1253',
                particles = [ P.b__tilde__, P.b, P.a, P.W__minus__, P.G__plus__ ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFVVS1, L.FFVVS2 ],
                couplings = {(0,1):C.GC_1830,(0,0):C.GC_1831})

V_1254 = Vertex(name = 'V_1254',
                particles = [ P.e__plus__, P.e__minus__, P.a, P.W__minus__, P.G__plus__ ],
                color = [ '1' ],
                lorentz = [ L.FFVVS1, L.FFVVS2 ],
                couplings = {(0,1):C.GC_1982,(0,0):C.GC_1983})

V_1255 = Vertex(name = 'V_1255',
                particles = [ P.mu__plus__, P.mu__minus__, P.a, P.W__minus__, P.G__plus__ ],
                color = [ '1' ],
                lorentz = [ L.FFVVS1, L.FFVVS2 ],
                couplings = {(0,1):C.GC_2055,(0,0):C.GC_2056})

V_1256 = Vertex(name = 'V_1256',
                particles = [ P.ta__plus__, P.ta__minus__, P.a, P.W__minus__, P.G__plus__ ],
                color = [ '1' ],
                lorentz = [ L.FFVVS1, L.FFVVS2 ],
                couplings = {(0,1):C.GC_2225,(0,0):C.GC_2226})

V_1257 = Vertex(name = 'V_1257',
                particles = [ P.d__tilde__, P.u, P.a, P.W__minus__, P.G0 ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFVVS1, L.FFVVS2, L.FFVVS6, L.FFVVS7 ],
                couplings = {(0,1):C.GC_762,(0,0):C.GC_765,(0,3):C.GC_825,(0,2):C.GC_828})

V_1258 = Vertex(name = 'V_1258',
                particles = [ P.s__tilde__, P.u, P.a, P.W__minus__, P.G0 ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFVVS1, L.FFVVS2, L.FFVVS6, L.FFVVS7 ],
                couplings = {(0,1):C.GC_769,(0,0):C.GC_772,(0,3):C.GC_832,(0,2):C.GC_835})

V_1259 = Vertex(name = 'V_1259',
                particles = [ P.b__tilde__, P.u, P.a, P.W__minus__, P.G0 ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFVVS1, L.FFVVS4, L.FFVVS6, L.FFVVS7 ],
                couplings = {(0,0):C.GC_779,(0,1):C.GC_776,(0,3):C.GC_839,(0,2):C.GC_842})

V_1260 = Vertex(name = 'V_1260',
                particles = [ P.d__tilde__, P.c, P.a, P.W__minus__, P.G0 ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFVVS1, L.FFVVS2, L.FFVVS6, L.FFVVS7 ],
                couplings = {(0,1):C.GC_783,(0,0):C.GC_786,(0,3):C.GC_846,(0,2):C.GC_849})

V_1261 = Vertex(name = 'V_1261',
                particles = [ P.s__tilde__, P.c, P.a, P.W__minus__, P.G0 ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFVVS1, L.FFVVS2, L.FFVVS6, L.FFVVS7 ],
                couplings = {(0,1):C.GC_790,(0,0):C.GC_793,(0,3):C.GC_853,(0,2):C.GC_856})

V_1262 = Vertex(name = 'V_1262',
                particles = [ P.b__tilde__, P.c, P.a, P.W__minus__, P.G0 ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFVVS1, L.FFVVS2, L.FFVVS6, L.FFVVS7 ],
                couplings = {(0,1):C.GC_797,(0,0):C.GC_800,(0,3):C.GC_860,(0,2):C.GC_863})

V_1263 = Vertex(name = 'V_1263',
                particles = [ P.d__tilde__, P.t, P.a, P.W__minus__, P.G0 ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFVVS1, L.FFVVS2, L.FFVVS6, L.FFVVS7 ],
                couplings = {(0,1):C.GC_804,(0,0):C.GC_807,(0,3):C.GC_867,(0,2):C.GC_870})

V_1264 = Vertex(name = 'V_1264',
                particles = [ P.s__tilde__, P.t, P.a, P.W__minus__, P.G0 ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFVVS1, L.FFVVS2, L.FFVVS6, L.FFVVS7 ],
                couplings = {(0,1):C.GC_811,(0,0):C.GC_814,(0,3):C.GC_874,(0,2):C.GC_877})

V_1265 = Vertex(name = 'V_1265',
                particles = [ P.b__tilde__, P.t, P.a, P.W__minus__, P.G0 ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFVVS1, L.FFVVS2, L.FFVVS6, L.FFVVS7 ],
                couplings = {(0,1):C.GC_818,(0,0):C.GC_821,(0,3):C.GC_881,(0,2):C.GC_884})

V_1266 = Vertex(name = 'V_1266',
                particles = [ P.d__tilde__, P.u, P.a, P.W__minus__, P.H ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFVVS1, L.FFVVS2, L.FFVVS6, L.FFVVS7 ],
                couplings = {(0,1):C.GC_764,(0,0):C.GC_763,(0,3):C.GC_827,(0,2):C.GC_826})

V_1267 = Vertex(name = 'V_1267',
                particles = [ P.s__tilde__, P.u, P.a, P.W__minus__, P.H ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFVVS1, L.FFVVS2, L.FFVVS7, L.FFVVS9 ],
                couplings = {(0,1):C.GC_771,(0,0):C.GC_770,(0,2):C.GC_834,(0,3):C.GC_833})

V_1268 = Vertex(name = 'V_1268',
                particles = [ P.b__tilde__, P.u, P.a, P.W__minus__, P.H ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFVVS1, L.FFVVS2, L.FFVVS6, L.FFVVS7 ],
                couplings = {(0,1):C.GC_778,(0,0):C.GC_777,(0,3):C.GC_841,(0,2):C.GC_840})

V_1269 = Vertex(name = 'V_1269',
                particles = [ P.d__tilde__, P.c, P.a, P.W__minus__, P.H ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFVVS1, L.FFVVS2, L.FFVVS6, L.FFVVS7 ],
                couplings = {(0,1):C.GC_785,(0,0):C.GC_784,(0,3):C.GC_848,(0,2):C.GC_847})

V_1270 = Vertex(name = 'V_1270',
                particles = [ P.s__tilde__, P.c, P.a, P.W__minus__, P.H ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFVVS1, L.FFVVS2, L.FFVVS6, L.FFVVS7 ],
                couplings = {(0,1):C.GC_792,(0,0):C.GC_791,(0,3):C.GC_855,(0,2):C.GC_854})

V_1271 = Vertex(name = 'V_1271',
                particles = [ P.b__tilde__, P.c, P.a, P.W__minus__, P.H ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFVVS1, L.FFVVS2, L.FFVVS6, L.FFVVS7 ],
                couplings = {(0,1):C.GC_799,(0,0):C.GC_798,(0,3):C.GC_862,(0,2):C.GC_861})

V_1272 = Vertex(name = 'V_1272',
                particles = [ P.d__tilde__, P.t, P.a, P.W__minus__, P.H ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFVVS1, L.FFVVS2, L.FFVVS6, L.FFVVS7 ],
                couplings = {(0,1):C.GC_806,(0,0):C.GC_805,(0,3):C.GC_869,(0,2):C.GC_868})

V_1273 = Vertex(name = 'V_1273',
                particles = [ P.s__tilde__, P.t, P.a, P.W__minus__, P.H ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFVVS1, L.FFVVS2, L.FFVVS6, L.FFVVS7 ],
                couplings = {(0,1):C.GC_813,(0,0):C.GC_812,(0,3):C.GC_876,(0,2):C.GC_875})

V_1274 = Vertex(name = 'V_1274',
                particles = [ P.b__tilde__, P.t, P.a, P.W__minus__, P.H ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFVVS1, L.FFVVS2, L.FFVVS6, L.FFVVS7 ],
                couplings = {(0,1):C.GC_820,(0,0):C.GC_819,(0,3):C.GC_883,(0,2):C.GC_882})

V_1275 = Vertex(name = 'V_1275',
                particles = [ P.d__tilde__, P.u, P.a, P.W__minus__ ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFVV1, L.FFVV2, L.FFVV5, L.FFVV6 ],
                couplings = {(0,1):C.GC_1629,(0,0):C.GC_1628,(0,3):C.GC_1656,(0,2):C.GC_1655})

V_1276 = Vertex(name = 'V_1276',
                particles = [ P.s__tilde__, P.u, P.a, P.W__minus__ ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFVV1, L.FFVV2, L.FFVV5, L.FFVV6 ],
                couplings = {(0,1):C.GC_1632,(0,0):C.GC_1631,(0,3):C.GC_1659,(0,2):C.GC_1658})

V_1277 = Vertex(name = 'V_1277',
                particles = [ P.b__tilde__, P.u, P.a, P.W__minus__ ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFVV1, L.FFVV2, L.FFVV5, L.FFVV6 ],
                couplings = {(0,1):C.GC_1635,(0,0):C.GC_1634,(0,3):C.GC_1662,(0,2):C.GC_1661})

V_1278 = Vertex(name = 'V_1278',
                particles = [ P.d__tilde__, P.c, P.a, P.W__minus__ ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFVV1, L.FFVV2, L.FFVV5, L.FFVV6 ],
                couplings = {(0,1):C.GC_1638,(0,0):C.GC_1637,(0,3):C.GC_1665,(0,2):C.GC_1664})

V_1279 = Vertex(name = 'V_1279',
                particles = [ P.s__tilde__, P.c, P.a, P.W__minus__ ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFVV1, L.FFVV4, L.FFVV5, L.FFVV6 ],
                couplings = {(0,0):C.GC_1640,(0,1):C.GC_1641,(0,3):C.GC_1668,(0,2):C.GC_1667})

V_1280 = Vertex(name = 'V_1280',
                particles = [ P.b__tilde__, P.c, P.a, P.W__minus__ ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFVV1, L.FFVV2, L.FFVV5, L.FFVV6 ],
                couplings = {(0,1):C.GC_1644,(0,0):C.GC_1643,(0,3):C.GC_1671,(0,2):C.GC_1670})

V_1281 = Vertex(name = 'V_1281',
                particles = [ P.d__tilde__, P.t, P.a, P.W__minus__ ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFVV1, L.FFVV2, L.FFVV5, L.FFVV6 ],
                couplings = {(0,1):C.GC_1647,(0,0):C.GC_1646,(0,3):C.GC_1674,(0,2):C.GC_1673})

V_1282 = Vertex(name = 'V_1282',
                particles = [ P.s__tilde__, P.t, P.a, P.W__minus__ ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFVV1, L.FFVV2, L.FFVV5, L.FFVV6 ],
                couplings = {(0,1):C.GC_1650,(0,0):C.GC_1649,(0,3):C.GC_1677,(0,2):C.GC_1676})

V_1283 = Vertex(name = 'V_1283',
                particles = [ P.b__tilde__, P.t, P.a, P.W__minus__ ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFVV1, L.FFVV2, L.FFVV5, L.FFVV6 ],
                couplings = {(0,1):C.GC_1653,(0,0):C.GC_1652,(0,3):C.GC_1680,(0,2):C.GC_1679})

V_1284 = Vertex(name = 'V_1284',
                particles = [ P.u__tilde__, P.d, P.a, P.W__plus__, P.G0 ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFVVS1, L.FFVVS2, L.FFVVS6, L.FFVVS7 ],
                couplings = {(0,1):C.GC_619,(0,0):C.GC_622,(0,3):C.GC_691,(0,2):C.GC_694})

V_1285 = Vertex(name = 'V_1285',
                particles = [ P.c__tilde__, P.d, P.a, P.W__plus__, P.G0 ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFVVS1, L.FFVVS2, L.FFVVS6, L.FFVVS7 ],
                couplings = {(0,1):C.GC_627,(0,0):C.GC_630,(0,3):C.GC_699,(0,2):C.GC_702})

V_1286 = Vertex(name = 'V_1286',
                particles = [ P.t__tilde__, P.d, P.a, P.W__plus__, P.G0 ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFVVS1, L.FFVVS2, L.FFVVS6, L.FFVVS7 ],
                couplings = {(0,1):C.GC_635,(0,0):C.GC_638,(0,3):C.GC_707,(0,2):C.GC_710})

V_1287 = Vertex(name = 'V_1287',
                particles = [ P.u__tilde__, P.s, P.a, P.W__plus__, P.G0 ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFVVS1, L.FFVVS2, L.FFVVS6, L.FFVVS7 ],
                couplings = {(0,1):C.GC_643,(0,0):C.GC_646,(0,3):C.GC_715,(0,2):C.GC_718})

V_1288 = Vertex(name = 'V_1288',
                particles = [ P.c__tilde__, P.s, P.a, P.W__plus__, P.G0 ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFVVS1, L.FFVVS2, L.FFVVS6, L.FFVVS7 ],
                couplings = {(0,1):C.GC_651,(0,0):C.GC_654,(0,3):C.GC_723,(0,2):C.GC_726})

V_1289 = Vertex(name = 'V_1289',
                particles = [ P.t__tilde__, P.s, P.a, P.W__plus__, P.G0 ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFVVS1, L.FFVVS2, L.FFVVS6, L.FFVVS7 ],
                couplings = {(0,1):C.GC_659,(0,0):C.GC_662,(0,3):C.GC_731,(0,2):C.GC_734})

V_1290 = Vertex(name = 'V_1290',
                particles = [ P.u__tilde__, P.b, P.a, P.W__plus__, P.G0 ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFVVS1, L.FFVVS2, L.FFVVS6, L.FFVVS7 ],
                couplings = {(0,1):C.GC_667,(0,0):C.GC_670,(0,3):C.GC_739,(0,2):C.GC_742})

V_1291 = Vertex(name = 'V_1291',
                particles = [ P.c__tilde__, P.b, P.a, P.W__plus__, P.G0 ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFVVS1, L.FFVVS2, L.FFVVS6, L.FFVVS7 ],
                couplings = {(0,1):C.GC_675,(0,0):C.GC_678,(0,3):C.GC_747,(0,2):C.GC_750})

V_1292 = Vertex(name = 'V_1292',
                particles = [ P.t__tilde__, P.b, P.a, P.W__plus__, P.G0 ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFVVS1, L.FFVVS2, L.FFVVS6, L.FFVVS7 ],
                couplings = {(0,1):C.GC_683,(0,0):C.GC_686,(0,3):C.GC_755,(0,2):C.GC_758})

V_1293 = Vertex(name = 'V_1293',
                particles = [ P.u__tilde__, P.d, P.a, P.W__plus__, P.H ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFVVS1, L.FFVVS2, L.FFVVS6, L.FFVVS7 ],
                couplings = {(0,1):C.GC_620,(0,0):C.GC_621,(0,3):C.GC_692,(0,2):C.GC_693})

V_1294 = Vertex(name = 'V_1294',
                particles = [ P.c__tilde__, P.d, P.a, P.W__plus__, P.H ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFVVS1, L.FFVVS2, L.FFVVS6, L.FFVVS7 ],
                couplings = {(0,1):C.GC_628,(0,0):C.GC_629,(0,3):C.GC_700,(0,2):C.GC_701})

V_1295 = Vertex(name = 'V_1295',
                particles = [ P.t__tilde__, P.d, P.a, P.W__plus__, P.H ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFVVS1, L.FFVVS2, L.FFVVS6, L.FFVVS7 ],
                couplings = {(0,1):C.GC_636,(0,0):C.GC_637,(0,3):C.GC_708,(0,2):C.GC_709})

V_1296 = Vertex(name = 'V_1296',
                particles = [ P.u__tilde__, P.s, P.a, P.W__plus__, P.H ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFVVS1, L.FFVVS2, L.FFVVS6, L.FFVVS7 ],
                couplings = {(0,1):C.GC_644,(0,0):C.GC_645,(0,3):C.GC_716,(0,2):C.GC_717})

V_1297 = Vertex(name = 'V_1297',
                particles = [ P.c__tilde__, P.s, P.a, P.W__plus__, P.H ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFVVS1, L.FFVVS2, L.FFVVS6, L.FFVVS7 ],
                couplings = {(0,1):C.GC_652,(0,0):C.GC_653,(0,3):C.GC_724,(0,2):C.GC_725})

V_1298 = Vertex(name = 'V_1298',
                particles = [ P.t__tilde__, P.s, P.a, P.W__plus__, P.H ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFVVS1, L.FFVVS2, L.FFVVS6, L.FFVVS7 ],
                couplings = {(0,1):C.GC_660,(0,0):C.GC_661,(0,3):C.GC_732,(0,2):C.GC_733})

V_1299 = Vertex(name = 'V_1299',
                particles = [ P.u__tilde__, P.b, P.a, P.W__plus__, P.H ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFVVS1, L.FFVVS2, L.FFVVS6, L.FFVVS7 ],
                couplings = {(0,1):C.GC_668,(0,0):C.GC_669,(0,3):C.GC_740,(0,2):C.GC_741})

V_1300 = Vertex(name = 'V_1300',
                particles = [ P.c__tilde__, P.b, P.a, P.W__plus__, P.H ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFVVS1, L.FFVVS2, L.FFVVS6, L.FFVVS7 ],
                couplings = {(0,1):C.GC_676,(0,0):C.GC_677,(0,3):C.GC_748,(0,2):C.GC_749})

V_1301 = Vertex(name = 'V_1301',
                particles = [ P.t__tilde__, P.b, P.a, P.W__plus__, P.H ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFVVS1, L.FFVVS2, L.FFVVS6, L.FFVVS7 ],
                couplings = {(0,1):C.GC_684,(0,0):C.GC_685,(0,3):C.GC_756,(0,2):C.GC_757})

V_1302 = Vertex(name = 'V_1302',
                particles = [ P.u__tilde__, P.d, P.a, P.W__plus__ ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFVV1, L.FFVV2, L.FFVV5, L.FFVV6 ],
                couplings = {(0,1):C.GC_1574,(0,0):C.GC_1575,(0,3):C.GC_1601,(0,2):C.GC_1602})

V_1303 = Vertex(name = 'V_1303',
                particles = [ P.c__tilde__, P.d, P.a, P.W__plus__ ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFVV1, L.FFVV2, L.FFVV5, L.FFVV6 ],
                couplings = {(0,1):C.GC_1577,(0,0):C.GC_1578,(0,3):C.GC_1604,(0,2):C.GC_1605})

V_1304 = Vertex(name = 'V_1304',
                particles = [ P.t__tilde__, P.d, P.a, P.W__plus__ ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFVV1, L.FFVV2, L.FFVV5, L.FFVV6 ],
                couplings = {(0,1):C.GC_1580,(0,0):C.GC_1581,(0,3):C.GC_1607,(0,2):C.GC_1608})

V_1305 = Vertex(name = 'V_1305',
                particles = [ P.u__tilde__, P.s, P.a, P.W__plus__ ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFVV1, L.FFVV2, L.FFVV5, L.FFVV6 ],
                couplings = {(0,1):C.GC_1583,(0,0):C.GC_1584,(0,3):C.GC_1610,(0,2):C.GC_1611})

V_1306 = Vertex(name = 'V_1306',
                particles = [ P.c__tilde__, P.s, P.a, P.W__plus__ ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFVV1, L.FFVV2, L.FFVV5, L.FFVV6 ],
                couplings = {(0,1):C.GC_1586,(0,0):C.GC_1587,(0,3):C.GC_1613,(0,2):C.GC_1614})

V_1307 = Vertex(name = 'V_1307',
                particles = [ P.t__tilde__, P.s, P.a, P.W__plus__ ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFVV1, L.FFVV2, L.FFVV5, L.FFVV6 ],
                couplings = {(0,1):C.GC_1589,(0,0):C.GC_1590,(0,3):C.GC_1616,(0,2):C.GC_1617})

V_1308 = Vertex(name = 'V_1308',
                particles = [ P.u__tilde__, P.b, P.a, P.W__plus__ ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFVV1, L.FFVV2, L.FFVV5, L.FFVV6 ],
                couplings = {(0,1):C.GC_1592,(0,0):C.GC_1593,(0,3):C.GC_1619,(0,2):C.GC_1620})

V_1309 = Vertex(name = 'V_1309',
                particles = [ P.c__tilde__, P.b, P.a, P.W__plus__ ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFVV1, L.FFVV2, L.FFVV5, L.FFVV6 ],
                couplings = {(0,1):C.GC_1595,(0,0):C.GC_1596,(0,3):C.GC_1622,(0,2):C.GC_1623})

V_1310 = Vertex(name = 'V_1310',
                particles = [ P.t__tilde__, P.b, P.a, P.W__plus__ ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFVV1, L.FFVV2, L.FFVV5, L.FFVV6 ],
                couplings = {(0,1):C.GC_1598,(0,0):C.GC_1599,(0,3):C.GC_1625,(0,2):C.GC_1626})

V_1311 = Vertex(name = 'V_1311',
                particles = [ P.ve__tilde__, P.e__minus__, P.a, P.W__plus__, P.G0 ],
                color = [ '1' ],
                lorentz = [ L.FFVVS1, L.FFVVS2 ],
                couplings = {(0,1):C.GC_1978,(0,0):C.GC_1981})

V_1312 = Vertex(name = 'V_1312',
                particles = [ P.vm__tilde__, P.mu__minus__, P.a, P.W__plus__, P.G0 ],
                color = [ '1' ],
                lorentz = [ L.FFVVS1, L.FFVVS2 ],
                couplings = {(0,1):C.GC_2051,(0,0):C.GC_2054})

V_1313 = Vertex(name = 'V_1313',
                particles = [ P.vt__tilde__, P.ta__minus__, P.a, P.W__plus__, P.G0 ],
                color = [ '1' ],
                lorentz = [ L.FFVVS1, L.FFVVS2 ],
                couplings = {(0,1):C.GC_2221,(0,0):C.GC_2224})

V_1314 = Vertex(name = 'V_1314',
                particles = [ P.ve__tilde__, P.e__minus__, P.a, P.W__plus__, P.H ],
                color = [ '1' ],
                lorentz = [ L.FFVVS1, L.FFVVS2 ],
                couplings = {(0,1):C.GC_1979,(0,0):C.GC_1980})

V_1315 = Vertex(name = 'V_1315',
                particles = [ P.vm__tilde__, P.mu__minus__, P.a, P.W__plus__, P.H ],
                color = [ '1' ],
                lorentz = [ L.FFVVS1, L.FFVVS2 ],
                couplings = {(0,1):C.GC_2052,(0,0):C.GC_2053})

V_1316 = Vertex(name = 'V_1316',
                particles = [ P.vt__tilde__, P.ta__minus__, P.a, P.W__plus__, P.H ],
                color = [ '1' ],
                lorentz = [ L.FFVVS1, L.FFVVS2 ],
                couplings = {(0,1):C.GC_2222,(0,0):C.GC_2223})

V_1317 = Vertex(name = 'V_1317',
                particles = [ P.ve__tilde__, P.e__minus__, P.a, P.W__plus__ ],
                color = [ '1' ],
                lorentz = [ L.FFVV1, L.FFVV2 ],
                couplings = {(0,1):C.GC_2006,(0,0):C.GC_2007})

V_1318 = Vertex(name = 'V_1318',
                particles = [ P.vm__tilde__, P.mu__minus__, P.a, P.W__plus__ ],
                color = [ '1' ],
                lorentz = [ L.FFVV1, L.FFVV2 ],
                couplings = {(0,1):C.GC_2082,(0,0):C.GC_2083})

V_1319 = Vertex(name = 'V_1319',
                particles = [ P.vt__tilde__, P.ta__minus__, P.a, P.W__plus__ ],
                color = [ '1' ],
                lorentz = [ L.FFVV1, L.FFVV2 ],
                couplings = {(0,1):C.GC_2249,(0,0):C.GC_2250})

V_1320 = Vertex(name = 'V_1320',
                particles = [ P.u__tilde__, P.u, P.a, P.W__plus__, P.G__minus__ ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFVVS1, L.FFVVS2 ],
                couplings = {(0,1):C.GC_2286,(0,0):C.GC_2285})

V_1321 = Vertex(name = 'V_1321',
                particles = [ P.c__tilde__, P.c, P.a, P.W__plus__, P.G__minus__ ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFVVS1, L.FFVVS2 ],
                couplings = {(0,1):C.GC_1878,(0,0):C.GC_1877})

V_1322 = Vertex(name = 'V_1322',
                particles = [ P.t__tilde__, P.t, P.a, P.W__plus__, P.G__minus__ ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFVVS1, L.FFVVS2 ],
                couplings = {(0,1):C.GC_2170,(0,0):C.GC_2169})

V_1323 = Vertex(name = 'V_1323',
                particles = [ P.d__tilde__, P.d, P.W__minus__, P.W__plus__, P.G0 ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFVVS1, L.FFVVS2, L.FFVVS6, L.FFVVS7 ],
                couplings = {(0,1):C.GC_1914,(0,0):C.GC_1917,(0,3):C.GC_1917,(0,2):C.GC_1914})

V_1324 = Vertex(name = 'V_1324',
                particles = [ P.s__tilde__, P.s, P.W__minus__, P.W__plus__, P.G0 ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFVVS1, L.FFVVS2, L.FFVVS6, L.FFVVS7 ],
                couplings = {(0,1):C.GC_2112,(0,0):C.GC_2115,(0,3):C.GC_2115,(0,2):C.GC_2112})

V_1325 = Vertex(name = 'V_1325',
                particles = [ P.b__tilde__, P.b, P.W__minus__, P.W__plus__, P.G0 ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFVVS1, L.FFVVS2, L.FFVVS6, L.FFVVS7 ],
                couplings = {(0,1):C.GC_1822,(0,0):C.GC_1825,(0,3):C.GC_1825,(0,2):C.GC_1822})

V_1326 = Vertex(name = 'V_1326',
                particles = [ P.d__tilde__, P.d, P.W__minus__, P.W__plus__, P.H ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFVVS1, L.FFVVS2, L.FFVVS6, L.FFVVS7 ],
                couplings = {(0,1):C.GC_1915,(0,0):C.GC_1916,(0,3):C.GC_1915,(0,2):C.GC_1916})

V_1327 = Vertex(name = 'V_1327',
                particles = [ P.s__tilde__, P.s, P.W__minus__, P.W__plus__, P.H ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFVVS1, L.FFVVS2, L.FFVVS6, L.FFVVS7 ],
                couplings = {(0,1):C.GC_2113,(0,0):C.GC_2114,(0,3):C.GC_2113,(0,2):C.GC_2114})

V_1328 = Vertex(name = 'V_1328',
                particles = [ P.b__tilde__, P.b, P.W__minus__, P.W__plus__, P.H ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFVVS1, L.FFVVS2, L.FFVVS6, L.FFVVS7 ],
                couplings = {(0,1):C.GC_1823,(0,0):C.GC_1824,(0,3):C.GC_1823,(0,2):C.GC_1824})

V_1329 = Vertex(name = 'V_1329',
                particles = [ P.d__tilde__, P.d, P.W__minus__, P.W__plus__ ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFVV1, L.FFVV2, L.FFVV5, L.FFVV6 ],
                couplings = {(0,1):C.GC_1939,(0,0):C.GC_1940,(0,3):C.GC_1939,(0,2):C.GC_1940})

V_1330 = Vertex(name = 'V_1330',
                particles = [ P.s__tilde__, P.s, P.W__minus__, P.W__plus__ ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFVV1, L.FFVV2, L.FFVV5, L.FFVV6 ],
                couplings = {(0,1):C.GC_2137,(0,0):C.GC_2138,(0,3):C.GC_2137,(0,2):C.GC_2138})

V_1331 = Vertex(name = 'V_1331',
                particles = [ P.b__tilde__, P.b, P.W__minus__, P.W__plus__ ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFVV1, L.FFVV2, L.FFVV5, L.FFVV6 ],
                couplings = {(0,1):C.GC_1845,(0,0):C.GC_1846,(0,3):C.GC_1845,(0,2):C.GC_1846})

V_1332 = Vertex(name = 'V_1332',
                particles = [ P.u__tilde__, P.d, P.W__minus__, P.W__plus__, P.G__plus__ ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFVVS1, L.FFVVS2, L.FFVVS6, L.FFVVS7 ],
                couplings = {(0,1):C.GC_407,(0,0):C.GC_408,(0,3):C.GC_443,(0,2):C.GC_444})

V_1333 = Vertex(name = 'V_1333',
                particles = [ P.c__tilde__, P.d, P.W__minus__, P.W__plus__, P.G__plus__ ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFVVS1, L.FFVVS2, L.FFVVS6, L.FFVVS7 ],
                couplings = {(0,1):C.GC_411,(0,0):C.GC_412,(0,3):C.GC_447,(0,2):C.GC_448})

V_1334 = Vertex(name = 'V_1334',
                particles = [ P.t__tilde__, P.d, P.W__minus__, P.W__plus__, P.G__plus__ ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFVVS1, L.FFVVS2, L.FFVVS6, L.FFVVS7 ],
                couplings = {(0,1):C.GC_415,(0,0):C.GC_416,(0,3):C.GC_451,(0,2):C.GC_452})

V_1335 = Vertex(name = 'V_1335',
                particles = [ P.u__tilde__, P.s, P.W__minus__, P.W__plus__, P.G__plus__ ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFVVS1, L.FFVVS2, L.FFVVS6, L.FFVVS7 ],
                couplings = {(0,1):C.GC_419,(0,0):C.GC_420,(0,3):C.GC_455,(0,2):C.GC_456})

V_1336 = Vertex(name = 'V_1336',
                particles = [ P.c__tilde__, P.s, P.W__minus__, P.W__plus__, P.G__plus__ ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFVVS1, L.FFVVS2, L.FFVVS6, L.FFVVS7 ],
                couplings = {(0,1):C.GC_423,(0,0):C.GC_424,(0,3):C.GC_459,(0,2):C.GC_460})

V_1337 = Vertex(name = 'V_1337',
                particles = [ P.t__tilde__, P.s, P.W__minus__, P.W__plus__, P.G__plus__ ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFVVS1, L.FFVVS2, L.FFVVS6, L.FFVVS7 ],
                couplings = {(0,1):C.GC_427,(0,0):C.GC_428,(0,3):C.GC_463,(0,2):C.GC_464})

V_1338 = Vertex(name = 'V_1338',
                particles = [ P.u__tilde__, P.b, P.W__minus__, P.W__plus__, P.G__plus__ ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFVVS1, L.FFVVS2, L.FFVVS6, L.FFVVS7 ],
                couplings = {(0,1):C.GC_431,(0,0):C.GC_432,(0,3):C.GC_467,(0,2):C.GC_468})

V_1339 = Vertex(name = 'V_1339',
                particles = [ P.c__tilde__, P.b, P.W__minus__, P.W__plus__, P.G__plus__ ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFVVS1, L.FFVVS2, L.FFVVS6, L.FFVVS7 ],
                couplings = {(0,1):C.GC_435,(0,0):C.GC_436,(0,3):C.GC_471,(0,2):C.GC_472})

V_1340 = Vertex(name = 'V_1340',
                particles = [ P.t__tilde__, P.b, P.W__minus__, P.W__plus__, P.G__plus__ ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFVVS1, L.FFVVS2, L.FFVVS6, L.FFVVS7 ],
                couplings = {(0,1):C.GC_439,(0,0):C.GC_440,(0,3):C.GC_475,(0,2):C.GC_476})

V_1341 = Vertex(name = 'V_1341',
                particles = [ P.e__plus__, P.e__minus__, P.W__minus__, P.W__plus__, P.G0 ],
                color = [ '1' ],
                lorentz = [ L.FFVVS1, L.FFVVS2, L.FFVVS6, L.FFVVS7 ],
                couplings = {(0,1):C.GC_1963,(0,0):C.GC_1966,(0,3):C.GC_1966,(0,2):C.GC_1963})

V_1342 = Vertex(name = 'V_1342',
                particles = [ P.mu__plus__, P.mu__minus__, P.W__minus__, P.W__plus__, P.G0 ],
                color = [ '1' ],
                lorentz = [ L.FFVVS1, L.FFVVS2, L.FFVVS6, L.FFVVS7 ],
                couplings = {(0,1):C.GC_2035,(0,0):C.GC_2038,(0,3):C.GC_2038,(0,2):C.GC_2035})

V_1343 = Vertex(name = 'V_1343',
                particles = [ P.ta__plus__, P.ta__minus__, P.W__minus__, P.W__plus__, P.G0 ],
                color = [ '1' ],
                lorentz = [ L.FFVVS1, L.FFVVS2, L.FFVVS6, L.FFVVS7 ],
                couplings = {(0,1):C.GC_2206,(0,0):C.GC_2209,(0,3):C.GC_2209,(0,2):C.GC_2206})

V_1344 = Vertex(name = 'V_1344',
                particles = [ P.e__plus__, P.e__minus__, P.W__minus__, P.W__plus__, P.H ],
                color = [ '1' ],
                lorentz = [ L.FFVVS1, L.FFVVS2, L.FFVVS6, L.FFVVS7 ],
                couplings = {(0,1):C.GC_1964,(0,0):C.GC_1965,(0,3):C.GC_1964,(0,2):C.GC_1965})

V_1345 = Vertex(name = 'V_1345',
                particles = [ P.mu__plus__, P.mu__minus__, P.W__minus__, P.W__plus__, P.H ],
                color = [ '1' ],
                lorentz = [ L.FFVVS1, L.FFVVS2, L.FFVVS6, L.FFVVS7 ],
                couplings = {(0,1):C.GC_2036,(0,0):C.GC_2037,(0,3):C.GC_2036,(0,2):C.GC_2037})

V_1346 = Vertex(name = 'V_1346',
                particles = [ P.ta__plus__, P.ta__minus__, P.W__minus__, P.W__plus__, P.H ],
                color = [ '1' ],
                lorentz = [ L.FFVVS1, L.FFVVS2, L.FFVVS6, L.FFVVS7 ],
                couplings = {(0,1):C.GC_2207,(0,0):C.GC_2208,(0,3):C.GC_2207,(0,2):C.GC_2208})

V_1347 = Vertex(name = 'V_1347',
                particles = [ P.e__plus__, P.e__minus__, P.W__minus__, P.W__plus__ ],
                color = [ '1' ],
                lorentz = [ L.FFVV1, L.FFVV2, L.FFVV5, L.FFVV6 ],
                couplings = {(0,1):C.GC_2000,(0,0):C.GC_2001,(0,3):C.GC_2000,(0,2):C.GC_2001})

V_1348 = Vertex(name = 'V_1348',
                particles = [ P.mu__plus__, P.mu__minus__, P.W__minus__, P.W__plus__ ],
                color = [ '1' ],
                lorentz = [ L.FFVV1, L.FFVV2, L.FFVV5, L.FFVV6 ],
                couplings = {(0,1):C.GC_2076,(0,0):C.GC_2077,(0,3):C.GC_2076,(0,2):C.GC_2077})

V_1349 = Vertex(name = 'V_1349',
                particles = [ P.ta__plus__, P.ta__minus__, P.W__minus__, P.W__plus__ ],
                color = [ '1' ],
                lorentz = [ L.FFVV1, L.FFVV2, L.FFVV5, L.FFVV6 ],
                couplings = {(0,1):C.GC_2243,(0,0):C.GC_2244,(0,3):C.GC_2243,(0,2):C.GC_2244})

V_1350 = Vertex(name = 'V_1350',
                particles = [ P.ve__tilde__, P.e__minus__, P.W__minus__, P.W__plus__, P.G__plus__ ],
                color = [ '1' ],
                lorentz = [ L.FFVVS1, L.FFVVS2 ],
                couplings = {(0,1):C.GC_1961,(0,0):C.GC_1962})

V_1351 = Vertex(name = 'V_1351',
                particles = [ P.vm__tilde__, P.mu__minus__, P.W__minus__, P.W__plus__, P.G__plus__ ],
                color = [ '1' ],
                lorentz = [ L.FFVVS1, L.FFVVS2 ],
                couplings = {(0,1):C.GC_2033,(0,0):C.GC_2034})

V_1352 = Vertex(name = 'V_1352',
                particles = [ P.vt__tilde__, P.ta__minus__, P.W__minus__, P.W__plus__, P.G__plus__ ],
                color = [ '1' ],
                lorentz = [ L.FFVVS1, L.FFVVS2 ],
                couplings = {(0,1):C.GC_2204,(0,0):C.GC_2205})

V_1353 = Vertex(name = 'V_1353',
                particles = [ P.u__tilde__, P.u, P.W__minus__, P.W__plus__, P.G0 ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFVVS1, L.FFVVS2, L.FFVVS6, L.FFVVS7 ],
                couplings = {(0,1):C.GC_2278,(0,0):C.GC_2281,(0,3):C.GC_2281,(0,2):C.GC_2278})

V_1354 = Vertex(name = 'V_1354',
                particles = [ P.c__tilde__, P.c, P.W__minus__, P.W__plus__, P.G0 ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFVVS1, L.FFVVS2, L.FFVVS6, L.FFVVS7 ],
                couplings = {(0,1):C.GC_1870,(0,0):C.GC_1873,(0,3):C.GC_1873,(0,2):C.GC_1870})

V_1355 = Vertex(name = 'V_1355',
                particles = [ P.t__tilde__, P.t, P.W__minus__, P.W__plus__, P.G0 ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFVVS1, L.FFVVS2, L.FFVVS6, L.FFVVS7 ],
                couplings = {(0,1):C.GC_2162,(0,0):C.GC_2165,(0,3):C.GC_2165,(0,2):C.GC_2162})

V_1356 = Vertex(name = 'V_1356',
                particles = [ P.u__tilde__, P.u, P.W__minus__, P.W__plus__, P.H ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFVVS1, L.FFVVS2, L.FFVVS6, L.FFVVS7 ],
                couplings = {(0,1):C.GC_2280,(0,0):C.GC_2279,(0,3):C.GC_2280,(0,2):C.GC_2279})

V_1357 = Vertex(name = 'V_1357',
                particles = [ P.c__tilde__, P.c, P.W__minus__, P.W__plus__, P.H ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFVVS1, L.FFVVS2, L.FFVVS6, L.FFVVS7 ],
                couplings = {(0,1):C.GC_1872,(0,0):C.GC_1871,(0,3):C.GC_1872,(0,2):C.GC_1871})

V_1358 = Vertex(name = 'V_1358',
                particles = [ P.t__tilde__, P.t, P.W__minus__, P.W__plus__, P.H ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFVVS1, L.FFVVS2, L.FFVVS6, L.FFVVS7 ],
                couplings = {(0,1):C.GC_2164,(0,0):C.GC_2163,(0,3):C.GC_2164,(0,2):C.GC_2163})

V_1359 = Vertex(name = 'V_1359',
                particles = [ P.u__tilde__, P.u, P.W__minus__, P.W__plus__ ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFVV1, L.FFVV2, L.FFVV5, L.FFVV6 ],
                couplings = {(0,1):C.GC_2298,(0,0):C.GC_2297,(0,3):C.GC_2298,(0,2):C.GC_2297})

V_1360 = Vertex(name = 'V_1360',
                particles = [ P.c__tilde__, P.c, P.W__minus__, P.W__plus__ ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFVV1, L.FFVV2, L.FFVV5, L.FFVV6 ],
                couplings = {(0,1):C.GC_1890,(0,0):C.GC_1889,(0,3):C.GC_1890,(0,2):C.GC_1889})

V_1361 = Vertex(name = 'V_1361',
                particles = [ P.t__tilde__, P.t, P.W__minus__, P.W__plus__ ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFVV1, L.FFVV2, L.FFVV5, L.FFVV6 ],
                couplings = {(0,1):C.GC_2182,(0,0):C.GC_2181,(0,3):C.GC_2182,(0,2):C.GC_2181})

V_1362 = Vertex(name = 'V_1362',
                particles = [ P.d__tilde__, P.u, P.W__minus__, P.W__plus__, P.G__minus__ ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFVVS1, L.FFVVS2, L.FFVVS6, L.FFVVS7 ],
                couplings = {(0,1):C.GC_480,(0,0):C.GC_479,(0,3):C.GC_517,(0,2):C.GC_516})

V_1363 = Vertex(name = 'V_1363',
                particles = [ P.s__tilde__, P.u, P.W__minus__, P.W__plus__, P.G__minus__ ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFVVS1, L.FFVVS2, L.FFVVS6, L.FFVVS7 ],
                couplings = {(0,1):C.GC_485,(0,0):C.GC_484,(0,3):C.GC_522,(0,2):C.GC_521})

V_1364 = Vertex(name = 'V_1364',
                particles = [ P.b__tilde__, P.u, P.W__minus__, P.W__plus__, P.G__minus__ ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFVVS1, L.FFVVS2, L.FFVVS6, L.FFVVS7 ],
                couplings = {(0,1):C.GC_489,(0,0):C.GC_488,(0,3):C.GC_526,(0,2):C.GC_525})

V_1365 = Vertex(name = 'V_1365',
                particles = [ P.d__tilde__, P.c, P.W__minus__, P.W__plus__, P.G__minus__ ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFVVS1, L.FFVVS2, L.FFVVS6, L.FFVVS7 ],
                couplings = {(0,1):C.GC_493,(0,0):C.GC_492,(0,3):C.GC_530,(0,2):C.GC_529})

V_1366 = Vertex(name = 'V_1366',
                particles = [ P.s__tilde__, P.c, P.W__minus__, P.W__plus__, P.G__minus__ ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFVVS1, L.FFVVS2, L.FFVVS6, L.FFVVS7 ],
                couplings = {(0,1):C.GC_497,(0,0):C.GC_496,(0,3):C.GC_534,(0,2):C.GC_533})

V_1367 = Vertex(name = 'V_1367',
                particles = [ P.b__tilde__, P.c, P.W__minus__, P.W__plus__, P.G__minus__ ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFVVS1, L.FFVVS2, L.FFVVS6, L.FFVVS7 ],
                couplings = {(0,1):C.GC_501,(0,0):C.GC_500,(0,3):C.GC_538,(0,2):C.GC_537})

V_1368 = Vertex(name = 'V_1368',
                particles = [ P.d__tilde__, P.t, P.W__minus__, P.W__plus__, P.G__minus__ ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFVVS1, L.FFVVS2, L.FFVVS6, L.FFVVS7 ],
                couplings = {(0,1):C.GC_505,(0,0):C.GC_504,(0,3):C.GC_542,(0,2):C.GC_541})

V_1369 = Vertex(name = 'V_1369',
                particles = [ P.s__tilde__, P.t, P.W__minus__, P.W__plus__, P.G__minus__ ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFVVS1, L.FFVVS2, L.FFVVS6, L.FFVVS7 ],
                couplings = {(0,1):C.GC_509,(0,0):C.GC_508,(0,3):C.GC_546,(0,2):C.GC_545})

V_1370 = Vertex(name = 'V_1370',
                particles = [ P.b__tilde__, P.t, P.W__minus__, P.W__plus__, P.G__minus__ ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFVVS1, L.FFVVS2, L.FFVVS6, L.FFVVS7 ],
                couplings = {(0,1):C.GC_513,(0,0):C.GC_512,(0,3):C.GC_550,(0,2):C.GC_549})

V_1371 = Vertex(name = 'V_1371',
                particles = [ P.d__tilde__, P.d, P.W__minus__, P.Z, P.G__plus__ ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFVVS1, L.FFVVS2 ],
                couplings = {(0,1):C.GC_1919,(0,0):C.GC_1918})

V_1372 = Vertex(name = 'V_1372',
                particles = [ P.s__tilde__, P.s, P.W__minus__, P.Z, P.G__plus__ ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFVVS1, L.FFVVS2 ],
                couplings = {(0,1):C.GC_2117,(0,0):C.GC_2116})

V_1373 = Vertex(name = 'V_1373',
                particles = [ P.b__tilde__, P.b, P.W__minus__, P.Z, P.G__plus__ ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFVVS1, L.FFVVS2 ],
                couplings = {(0,1):C.GC_1827,(0,0):C.GC_1826})

V_1374 = Vertex(name = 'V_1374',
                particles = [ P.e__plus__, P.e__minus__, P.W__minus__, P.Z, P.G__plus__ ],
                color = [ '1' ],
                lorentz = [ L.FFVVS1, L.FFVVS2 ],
                couplings = {(0,1):C.GC_1971,(0,0):C.GC_1970})

V_1375 = Vertex(name = 'V_1375',
                particles = [ P.mu__plus__, P.mu__minus__, P.W__minus__, P.Z, P.G__plus__ ],
                color = [ '1' ],
                lorentz = [ L.FFVVS1, L.FFVVS2 ],
                couplings = {(0,1):C.GC_2043,(0,0):C.GC_2042})

V_1376 = Vertex(name = 'V_1376',
                particles = [ P.ta__plus__, P.ta__minus__, P.W__minus__, P.Z, P.G__plus__ ],
                color = [ '1' ],
                lorentz = [ L.FFVVS1, L.FFVVS2 ],
                couplings = {(0,1):C.GC_2214,(0,0):C.GC_2213})

V_1377 = Vertex(name = 'V_1377',
                particles = [ P.d__tilde__, P.u, P.W__minus__, P.Z, P.G0 ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFVVS1, L.FFVVS2, L.FFVVS6, L.FFVVS7 ],
                couplings = {(0,1):C.GC_483,(0,0):C.GC_481,(0,3):C.GC_520,(0,2):C.GC_518})

V_1378 = Vertex(name = 'V_1378',
                particles = [ P.s__tilde__, P.u, P.W__minus__, P.Z, P.G0 ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFVVS10, L.FFVVS3 ],
                couplings = {(0,1):C.GC_487,(0,0):C.GC_524})

V_1379 = Vertex(name = 'V_1379',
                particles = [ P.b__tilde__, P.u, P.W__minus__, P.Z, P.G0 ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFVVS3, L.FFVVS8 ],
                couplings = {(0,0):C.GC_491,(0,1):C.GC_528})

V_1380 = Vertex(name = 'V_1380',
                particles = [ P.d__tilde__, P.c, P.W__minus__, P.Z, P.G0 ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFVVS3, L.FFVVS8 ],
                couplings = {(0,0):C.GC_495,(0,1):C.GC_532})

V_1381 = Vertex(name = 'V_1381',
                particles = [ P.s__tilde__, P.c, P.W__minus__, P.Z, P.G0 ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFVVS3, L.FFVVS8 ],
                couplings = {(0,0):C.GC_499,(0,1):C.GC_536})

V_1382 = Vertex(name = 'V_1382',
                particles = [ P.b__tilde__, P.c, P.W__minus__, P.Z, P.G0 ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFVVS3, L.FFVVS8 ],
                couplings = {(0,0):C.GC_503,(0,1):C.GC_540})

V_1383 = Vertex(name = 'V_1383',
                particles = [ P.d__tilde__, P.t, P.W__minus__, P.Z, P.G0 ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFVVS3, L.FFVVS8 ],
                couplings = {(0,0):C.GC_507,(0,1):C.GC_544})

V_1384 = Vertex(name = 'V_1384',
                particles = [ P.s__tilde__, P.t, P.W__minus__, P.Z, P.G0 ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFVVS3, L.FFVVS8 ],
                couplings = {(0,0):C.GC_511,(0,1):C.GC_548})

V_1385 = Vertex(name = 'V_1385',
                particles = [ P.b__tilde__, P.t, P.W__minus__, P.Z, P.G0 ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFVVS3, L.FFVVS8 ],
                couplings = {(0,0):C.GC_515,(0,1):C.GC_552})

V_1386 = Vertex(name = 'V_1386',
                particles = [ P.d__tilde__, P.u, P.W__minus__, P.Z, P.H ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFVVS3, L.FFVVS8 ],
                couplings = {(0,0):C.GC_482,(0,1):C.GC_519})

V_1387 = Vertex(name = 'V_1387',
                particles = [ P.s__tilde__, P.u, P.W__minus__, P.Z, P.H ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFVVS3, L.FFVVS8 ],
                couplings = {(0,0):C.GC_486,(0,1):C.GC_523})

V_1388 = Vertex(name = 'V_1388',
                particles = [ P.b__tilde__, P.u, P.W__minus__, P.Z, P.H ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFVVS3, L.FFVVS8 ],
                couplings = {(0,0):C.GC_490,(0,1):C.GC_527})

V_1389 = Vertex(name = 'V_1389',
                particles = [ P.d__tilde__, P.c, P.W__minus__, P.Z, P.H ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFVVS3, L.FFVVS8 ],
                couplings = {(0,0):C.GC_494,(0,1):C.GC_531})

V_1390 = Vertex(name = 'V_1390',
                particles = [ P.s__tilde__, P.c, P.W__minus__, P.Z, P.H ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFVVS5, L.FFVVS8 ],
                couplings = {(0,0):C.GC_498,(0,1):C.GC_535})

V_1391 = Vertex(name = 'V_1391',
                particles = [ P.b__tilde__, P.c, P.W__minus__, P.Z, P.H ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFVVS3, L.FFVVS8 ],
                couplings = {(0,0):C.GC_502,(0,1):C.GC_539})

V_1392 = Vertex(name = 'V_1392',
                particles = [ P.d__tilde__, P.t, P.W__minus__, P.Z, P.H ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFVVS3, L.FFVVS8 ],
                couplings = {(0,0):C.GC_506,(0,1):C.GC_543})

V_1393 = Vertex(name = 'V_1393',
                particles = [ P.s__tilde__, P.t, P.W__minus__, P.Z, P.H ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFVVS3, L.FFVVS8 ],
                couplings = {(0,0):C.GC_510,(0,1):C.GC_547})

V_1394 = Vertex(name = 'V_1394',
                particles = [ P.b__tilde__, P.t, P.W__minus__, P.Z, P.H ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFVVS3, L.FFVVS8 ],
                couplings = {(0,0):C.GC_514,(0,1):C.GC_551})

V_1395 = Vertex(name = 'V_1395',
                particles = [ P.d__tilde__, P.u, P.W__minus__, P.Z ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFVV3, L.FFVV7 ],
                couplings = {(0,0):C.GC_1542,(0,1):C.GC_1551})

V_1396 = Vertex(name = 'V_1396',
                particles = [ P.s__tilde__, P.u, P.W__minus__, P.Z ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFVV3, L.FFVV7 ],
                couplings = {(0,0):C.GC_1543,(0,1):C.GC_1552})

V_1397 = Vertex(name = 'V_1397',
                particles = [ P.b__tilde__, P.u, P.W__minus__, P.Z ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFVV3, L.FFVV7 ],
                couplings = {(0,0):C.GC_1544,(0,1):C.GC_1553})

V_1398 = Vertex(name = 'V_1398',
                particles = [ P.d__tilde__, P.c, P.W__minus__, P.Z ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFVV3, L.FFVV7 ],
                couplings = {(0,0):C.GC_1545,(0,1):C.GC_1554})

V_1399 = Vertex(name = 'V_1399',
                particles = [ P.s__tilde__, P.c, P.W__minus__, P.Z ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFVV3, L.FFVV7 ],
                couplings = {(0,0):C.GC_1546,(0,1):C.GC_1555})

V_1400 = Vertex(name = 'V_1400',
                particles = [ P.b__tilde__, P.c, P.W__minus__, P.Z ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFVV3, L.FFVV7 ],
                couplings = {(0,0):C.GC_1547,(0,1):C.GC_1556})

V_1401 = Vertex(name = 'V_1401',
                particles = [ P.d__tilde__, P.t, P.W__minus__, P.Z ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFVV3, L.FFVV7 ],
                couplings = {(0,0):C.GC_1548,(0,1):C.GC_1557})

V_1402 = Vertex(name = 'V_1402',
                particles = [ P.s__tilde__, P.t, P.W__minus__, P.Z ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFVV3, L.FFVV7 ],
                couplings = {(0,0):C.GC_1549,(0,1):C.GC_1558})

V_1403 = Vertex(name = 'V_1403',
                particles = [ P.b__tilde__, P.t, P.W__minus__, P.Z ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFVV3, L.FFVV7 ],
                couplings = {(0,0):C.GC_1550,(0,1):C.GC_1559})

V_1404 = Vertex(name = 'V_1404',
                particles = [ P.u__tilde__, P.d, P.W__plus__, P.Z, P.G0 ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFVVS3, L.FFVVS8 ],
                couplings = {(0,0):C.GC_410,(0,1):C.GC_446})

V_1405 = Vertex(name = 'V_1405',
                particles = [ P.c__tilde__, P.d, P.W__plus__, P.Z, P.G0 ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFVVS3, L.FFVVS8 ],
                couplings = {(0,0):C.GC_414,(0,1):C.GC_450})

V_1406 = Vertex(name = 'V_1406',
                particles = [ P.t__tilde__, P.d, P.W__plus__, P.Z, P.G0 ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFVVS3, L.FFVVS8 ],
                couplings = {(0,0):C.GC_418,(0,1):C.GC_454})

V_1407 = Vertex(name = 'V_1407',
                particles = [ P.u__tilde__, P.s, P.W__plus__, P.Z, P.G0 ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFVVS3, L.FFVVS8 ],
                couplings = {(0,0):C.GC_422,(0,1):C.GC_458})

V_1408 = Vertex(name = 'V_1408',
                particles = [ P.c__tilde__, P.s, P.W__plus__, P.Z, P.G0 ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFVVS3, L.FFVVS8 ],
                couplings = {(0,0):C.GC_426,(0,1):C.GC_462})

V_1409 = Vertex(name = 'V_1409',
                particles = [ P.t__tilde__, P.s, P.W__plus__, P.Z, P.G0 ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFVVS3, L.FFVVS8 ],
                couplings = {(0,0):C.GC_430,(0,1):C.GC_466})

V_1410 = Vertex(name = 'V_1410',
                particles = [ P.u__tilde__, P.b, P.W__plus__, P.Z, P.G0 ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFVVS3, L.FFVVS8 ],
                couplings = {(0,0):C.GC_434,(0,1):C.GC_470})

V_1411 = Vertex(name = 'V_1411',
                particles = [ P.c__tilde__, P.b, P.W__plus__, P.Z, P.G0 ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFVVS3, L.FFVVS8 ],
                couplings = {(0,0):C.GC_438,(0,1):C.GC_474})

V_1412 = Vertex(name = 'V_1412',
                particles = [ P.t__tilde__, P.b, P.W__plus__, P.Z, P.G0 ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFVVS3, L.FFVVS8 ],
                couplings = {(0,0):C.GC_442,(0,1):C.GC_478})

V_1413 = Vertex(name = 'V_1413',
                particles = [ P.u__tilde__, P.d, P.W__plus__, P.Z, P.H ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFVVS3, L.FFVVS8 ],
                couplings = {(0,0):C.GC_409,(0,1):C.GC_445})

V_1414 = Vertex(name = 'V_1414',
                particles = [ P.c__tilde__, P.d, P.W__plus__, P.Z, P.H ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFVVS3, L.FFVVS8 ],
                couplings = {(0,0):C.GC_413,(0,1):C.GC_449})

V_1415 = Vertex(name = 'V_1415',
                particles = [ P.t__tilde__, P.d, P.W__plus__, P.Z, P.H ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFVVS3, L.FFVVS8 ],
                couplings = {(0,0):C.GC_417,(0,1):C.GC_453})

V_1416 = Vertex(name = 'V_1416',
                particles = [ P.u__tilde__, P.s, P.W__plus__, P.Z, P.H ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFVVS3, L.FFVVS8 ],
                couplings = {(0,0):C.GC_421,(0,1):C.GC_457})

V_1417 = Vertex(name = 'V_1417',
                particles = [ P.c__tilde__, P.s, P.W__plus__, P.Z, P.H ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFVVS3, L.FFVVS8 ],
                couplings = {(0,0):C.GC_425,(0,1):C.GC_461})

V_1418 = Vertex(name = 'V_1418',
                particles = [ P.t__tilde__, P.s, P.W__plus__, P.Z, P.H ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFVVS3, L.FFVVS8 ],
                couplings = {(0,0):C.GC_429,(0,1):C.GC_465})

V_1419 = Vertex(name = 'V_1419',
                particles = [ P.u__tilde__, P.b, P.W__plus__, P.Z, P.H ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFVVS3, L.FFVVS8 ],
                couplings = {(0,0):C.GC_433,(0,1):C.GC_469})

V_1420 = Vertex(name = 'V_1420',
                particles = [ P.c__tilde__, P.b, P.W__plus__, P.Z, P.H ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFVVS3, L.FFVVS8 ],
                couplings = {(0,0):C.GC_437,(0,1):C.GC_473})

V_1421 = Vertex(name = 'V_1421',
                particles = [ P.t__tilde__, P.b, P.W__plus__, P.Z, P.H ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFVVS3, L.FFVVS8 ],
                couplings = {(0,0):C.GC_441,(0,1):C.GC_477})

V_1422 = Vertex(name = 'V_1422',
                particles = [ P.u__tilde__, P.d, P.W__plus__, P.Z ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFVV3, L.FFVV7 ],
                couplings = {(0,0):C.GC_1524,(0,1):C.GC_1533})

V_1423 = Vertex(name = 'V_1423',
                particles = [ P.c__tilde__, P.d, P.W__plus__, P.Z ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFVV3, L.FFVV7 ],
                couplings = {(0,0):C.GC_1525,(0,1):C.GC_1534})

V_1424 = Vertex(name = 'V_1424',
                particles = [ P.t__tilde__, P.d, P.W__plus__, P.Z ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFVV3, L.FFVV7 ],
                couplings = {(0,0):C.GC_1526,(0,1):C.GC_1535})

V_1425 = Vertex(name = 'V_1425',
                particles = [ P.u__tilde__, P.s, P.W__plus__, P.Z ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFVV3, L.FFVV7 ],
                couplings = {(0,0):C.GC_1527,(0,1):C.GC_1536})

V_1426 = Vertex(name = 'V_1426',
                particles = [ P.c__tilde__, P.s, P.W__plus__, P.Z ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFVV3, L.FFVV7 ],
                couplings = {(0,0):C.GC_1528,(0,1):C.GC_1537})

V_1427 = Vertex(name = 'V_1427',
                particles = [ P.t__tilde__, P.s, P.W__plus__, P.Z ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFVV3, L.FFVV7 ],
                couplings = {(0,0):C.GC_1529,(0,1):C.GC_1538})

V_1428 = Vertex(name = 'V_1428',
                particles = [ P.u__tilde__, P.b, P.W__plus__, P.Z ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFVV3, L.FFVV7 ],
                couplings = {(0,0):C.GC_1530,(0,1):C.GC_1539})

V_1429 = Vertex(name = 'V_1429',
                particles = [ P.c__tilde__, P.b, P.W__plus__, P.Z ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFVV3, L.FFVV7 ],
                couplings = {(0,0):C.GC_1531,(0,1):C.GC_1540})

V_1430 = Vertex(name = 'V_1430',
                particles = [ P.t__tilde__, P.b, P.W__plus__, P.Z ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFVV3, L.FFVV7 ],
                couplings = {(0,0):C.GC_1532,(0,1):C.GC_1541})

V_1431 = Vertex(name = 'V_1431',
                particles = [ P.ve__tilde__, P.e__minus__, P.W__plus__, P.Z, P.G0 ],
                color = [ '1' ],
                lorentz = [ L.FFVVS3 ],
                couplings = {(0,0):C.GC_1969})

V_1432 = Vertex(name = 'V_1432',
                particles = [ P.vm__tilde__, P.mu__minus__, P.W__plus__, P.Z, P.G0 ],
                color = [ '1' ],
                lorentz = [ L.FFVVS3 ],
                couplings = {(0,0):C.GC_2041})

V_1433 = Vertex(name = 'V_1433',
                particles = [ P.vt__tilde__, P.ta__minus__, P.W__plus__, P.Z, P.G0 ],
                color = [ '1' ],
                lorentz = [ L.FFVVS3 ],
                couplings = {(0,0):C.GC_2212})

V_1434 = Vertex(name = 'V_1434',
                particles = [ P.ve__tilde__, P.e__minus__, P.W__plus__, P.Z, P.H ],
                color = [ '1' ],
                lorentz = [ L.FFVVS3 ],
                couplings = {(0,0):C.GC_1968})

V_1435 = Vertex(name = 'V_1435',
                particles = [ P.vm__tilde__, P.mu__minus__, P.W__plus__, P.Z, P.H ],
                color = [ '1' ],
                lorentz = [ L.FFVVS3 ],
                couplings = {(0,0):C.GC_2040})

V_1436 = Vertex(name = 'V_1436',
                particles = [ P.vt__tilde__, P.ta__minus__, P.W__plus__, P.Z, P.H ],
                color = [ '1' ],
                lorentz = [ L.FFVVS3 ],
                couplings = {(0,0):C.GC_2211})

V_1437 = Vertex(name = 'V_1437',
                particles = [ P.ve__tilde__, P.e__minus__, P.W__plus__, P.Z ],
                color = [ '1' ],
                lorentz = [ L.FFVV3 ],
                couplings = {(0,0):C.GC_2003})

V_1438 = Vertex(name = 'V_1438',
                particles = [ P.vm__tilde__, P.mu__minus__, P.W__plus__, P.Z ],
                color = [ '1' ],
                lorentz = [ L.FFVV3 ],
                couplings = {(0,0):C.GC_2079})

V_1439 = Vertex(name = 'V_1439',
                particles = [ P.vt__tilde__, P.ta__minus__, P.W__plus__, P.Z ],
                color = [ '1' ],
                lorentz = [ L.FFVV3 ],
                couplings = {(0,0):C.GC_2246})

V_1440 = Vertex(name = 'V_1440',
                particles = [ P.u__tilde__, P.u, P.W__plus__, P.Z, P.G__minus__ ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFVVS3 ],
                couplings = {(0,0):C.GC_2282})

V_1441 = Vertex(name = 'V_1441',
                particles = [ P.c__tilde__, P.c, P.W__plus__, P.Z, P.G__minus__ ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFVVS3 ],
                couplings = {(0,0):C.GC_1874})

V_1442 = Vertex(name = 'V_1442',
                particles = [ P.t__tilde__, P.t, P.W__plus__, P.Z, P.G__minus__ ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFVVS3 ],
                couplings = {(0,0):C.GC_2166})

V_1443 = Vertex(name = 'V_1443',
                particles = [ P.d__tilde__, P.d, P.a, P.W__plus__, P.G__minus__ ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFVVS8 ],
                couplings = {(0,0):C.GC_1922})

V_1444 = Vertex(name = 'V_1444',
                particles = [ P.s__tilde__, P.s, P.a, P.W__plus__, P.G__minus__ ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFVVS8 ],
                couplings = {(0,0):C.GC_2121})

V_1445 = Vertex(name = 'V_1445',
                particles = [ P.b__tilde__, P.b, P.a, P.W__plus__, P.G__minus__ ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFVVS8 ],
                couplings = {(0,0):C.GC_1830})

V_1446 = Vertex(name = 'V_1446',
                particles = [ P.e__plus__, P.ve, P.W__minus__, P.G0 ],
                color = [ '1' ],
                lorentz = [ L.FFVS10 ],
                couplings = {(0,0):C.GC_1972})

V_1447 = Vertex(name = 'V_1447',
                particles = [ P.mu__plus__, P.vm, P.W__minus__, P.G0 ],
                color = [ '1' ],
                lorentz = [ L.FFVS10 ],
                couplings = {(0,0):C.GC_2044})

V_1448 = Vertex(name = 'V_1448',
                particles = [ P.ta__plus__, P.vt, P.W__minus__, P.G0 ],
                color = [ '1' ],
                lorentz = [ L.FFVS10 ],
                couplings = {(0,0):C.GC_2215})

V_1449 = Vertex(name = 'V_1449',
                particles = [ P.e__plus__, P.ve, P.a, P.W__minus__, P.G0 ],
                color = [ '1' ],
                lorentz = [ L.FFVVS8 ],
                couplings = {(0,0):C.GC_1978})

V_1450 = Vertex(name = 'V_1450',
                particles = [ P.mu__plus__, P.vm, P.a, P.W__minus__, P.G0 ],
                color = [ '1' ],
                lorentz = [ L.FFVVS8 ],
                couplings = {(0,0):C.GC_2051})

V_1451 = Vertex(name = 'V_1451',
                particles = [ P.ta__plus__, P.vt, P.a, P.W__minus__, P.G0 ],
                color = [ '1' ],
                lorentz = [ L.FFVVS8 ],
                couplings = {(0,0):C.GC_2221})

V_1452 = Vertex(name = 'V_1452',
                particles = [ P.e__plus__, P.ve, P.a, P.W__minus__, P.H ],
                color = [ '1' ],
                lorentz = [ L.FFVVS8 ],
                couplings = {(0,0):C.GC_1980})

V_1453 = Vertex(name = 'V_1453',
                particles = [ P.mu__plus__, P.vm, P.a, P.W__minus__, P.H ],
                color = [ '1' ],
                lorentz = [ L.FFVVS8 ],
                couplings = {(0,0):C.GC_2053})

V_1454 = Vertex(name = 'V_1454',
                particles = [ P.ta__plus__, P.vt, P.a, P.W__minus__, P.H ],
                color = [ '1' ],
                lorentz = [ L.FFVVS8 ],
                couplings = {(0,0):C.GC_2223})

V_1455 = Vertex(name = 'V_1455',
                particles = [ P.e__plus__, P.ve, P.a, P.W__minus__ ],
                color = [ '1' ],
                lorentz = [ L.FFVV7 ],
                couplings = {(0,0):C.GC_2007})

V_1456 = Vertex(name = 'V_1456',
                particles = [ P.mu__plus__, P.vm, P.a, P.W__minus__ ],
                color = [ '1' ],
                lorentz = [ L.FFVV7 ],
                couplings = {(0,0):C.GC_2083})

V_1457 = Vertex(name = 'V_1457',
                particles = [ P.ta__plus__, P.vt, P.a, P.W__minus__ ],
                color = [ '1' ],
                lorentz = [ L.FFVV7 ],
                couplings = {(0,0):C.GC_2250})

V_1458 = Vertex(name = 'V_1458',
                particles = [ P.e__plus__, P.e__minus__, P.a, P.W__plus__, P.G__minus__ ],
                color = [ '1' ],
                lorentz = [ L.FFVVS8 ],
                couplings = {(0,0):C.GC_1982})

V_1459 = Vertex(name = 'V_1459',
                particles = [ P.mu__plus__, P.mu__minus__, P.a, P.W__plus__, P.G__minus__ ],
                color = [ '1' ],
                lorentz = [ L.FFVVS8 ],
                couplings = {(0,0):C.GC_2055})

V_1460 = Vertex(name = 'V_1460',
                particles = [ P.ta__plus__, P.ta__minus__, P.a, P.W__plus__, P.G__minus__ ],
                color = [ '1' ],
                lorentz = [ L.FFVVS8 ],
                couplings = {(0,0):C.GC_2225})

V_1461 = Vertex(name = 'V_1461',
                particles = [ P.e__plus__, P.ve, P.W__minus__, P.W__plus__, P.G__minus__ ],
                color = [ '1' ],
                lorentz = [ L.FFVVS8 ],
                couplings = {(0,0):C.GC_1962})

V_1462 = Vertex(name = 'V_1462',
                particles = [ P.mu__plus__, P.vm, P.W__minus__, P.W__plus__, P.G__minus__ ],
                color = [ '1' ],
                lorentz = [ L.FFVVS8 ],
                couplings = {(0,0):C.GC_2034})

V_1463 = Vertex(name = 'V_1463',
                particles = [ P.ta__plus__, P.vt, P.W__minus__, P.W__plus__, P.G__minus__ ],
                color = [ '1' ],
                lorentz = [ L.FFVVS8 ],
                couplings = {(0,0):C.GC_2205})

V_1464 = Vertex(name = 'V_1464',
                particles = [ P.u__tilde__, P.u, P.a, P.W__minus__, P.G__plus__ ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFVVS8 ],
                couplings = {(0,0):C.GC_2286})

V_1465 = Vertex(name = 'V_1465',
                particles = [ P.c__tilde__, P.c, P.a, P.W__minus__, P.G__plus__ ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFVVS8 ],
                couplings = {(0,0):C.GC_1878})

V_1466 = Vertex(name = 'V_1466',
                particles = [ P.t__tilde__, P.t, P.a, P.W__minus__, P.G__plus__ ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFVVS8 ],
                couplings = {(0,0):C.GC_2170})

V_1467 = Vertex(name = 'V_1467',
                particles = [ P.d__tilde__, P.d, P.W__plus__, P.Z, P.G__minus__ ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFVVS8 ],
                couplings = {(0,0):C.GC_1919})

V_1468 = Vertex(name = 'V_1468',
                particles = [ P.s__tilde__, P.s, P.W__plus__, P.Z, P.G__minus__ ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFVVS8 ],
                couplings = {(0,0):C.GC_2117})

V_1469 = Vertex(name = 'V_1469',
                particles = [ P.b__tilde__, P.b, P.W__plus__, P.Z, P.G__minus__ ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFVVS8 ],
                couplings = {(0,0):C.GC_1827})

V_1470 = Vertex(name = 'V_1470',
                particles = [ P.e__plus__, P.ve, P.W__minus__, P.Z, P.G0 ],
                color = [ '1' ],
                lorentz = [ L.FFVVS8 ],
                couplings = {(0,0):C.GC_1969})

V_1471 = Vertex(name = 'V_1471',
                particles = [ P.mu__plus__, P.vm, P.W__minus__, P.Z, P.G0 ],
                color = [ '1' ],
                lorentz = [ L.FFVVS8 ],
                couplings = {(0,0):C.GC_2041})

V_1472 = Vertex(name = 'V_1472',
                particles = [ P.ta__plus__, P.vt, P.W__minus__, P.Z, P.G0 ],
                color = [ '1' ],
                lorentz = [ L.FFVVS8 ],
                couplings = {(0,0):C.GC_2212})

V_1473 = Vertex(name = 'V_1473',
                particles = [ P.e__plus__, P.ve, P.W__minus__, P.Z, P.H ],
                color = [ '1' ],
                lorentz = [ L.FFVVS8 ],
                couplings = {(0,0):C.GC_1967})

V_1474 = Vertex(name = 'V_1474',
                particles = [ P.mu__plus__, P.vm, P.W__minus__, P.Z, P.H ],
                color = [ '1' ],
                lorentz = [ L.FFVVS8 ],
                couplings = {(0,0):C.GC_2039})

V_1475 = Vertex(name = 'V_1475',
                particles = [ P.ta__plus__, P.vt, P.W__minus__, P.Z, P.H ],
                color = [ '1' ],
                lorentz = [ L.FFVVS8 ],
                couplings = {(0,0):C.GC_2210})

V_1476 = Vertex(name = 'V_1476',
                particles = [ P.e__plus__, P.ve, P.W__minus__, P.Z ],
                color = [ '1' ],
                lorentz = [ L.FFVV7 ],
                couplings = {(0,0):C.GC_2002})

V_1477 = Vertex(name = 'V_1477',
                particles = [ P.mu__plus__, P.vm, P.W__minus__, P.Z ],
                color = [ '1' ],
                lorentz = [ L.FFVV7 ],
                couplings = {(0,0):C.GC_2078})

V_1478 = Vertex(name = 'V_1478',
                particles = [ P.ta__plus__, P.vt, P.W__minus__, P.Z ],
                color = [ '1' ],
                lorentz = [ L.FFVV7 ],
                couplings = {(0,0):C.GC_2245})

V_1479 = Vertex(name = 'V_1479',
                particles = [ P.e__plus__, P.e__minus__, P.W__plus__, P.Z, P.G__minus__ ],
                color = [ '1' ],
                lorentz = [ L.FFVVS8 ],
                couplings = {(0,0):C.GC_1971})

V_1480 = Vertex(name = 'V_1480',
                particles = [ P.mu__plus__, P.mu__minus__, P.W__plus__, P.Z, P.G__minus__ ],
                color = [ '1' ],
                lorentz = [ L.FFVVS8 ],
                couplings = {(0,0):C.GC_2043})

V_1481 = Vertex(name = 'V_1481',
                particles = [ P.ta__plus__, P.ta__minus__, P.W__plus__, P.Z, P.G__minus__ ],
                color = [ '1' ],
                lorentz = [ L.FFVVS8 ],
                couplings = {(0,0):C.GC_2214})

V_1482 = Vertex(name = 'V_1482',
                particles = [ P.u__tilde__, P.u, P.W__minus__, P.Z, P.G__plus__ ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFVVS8 ],
                couplings = {(0,0):C.GC_2282})

V_1483 = Vertex(name = 'V_1483',
                particles = [ P.c__tilde__, P.c, P.W__minus__, P.Z, P.G__plus__ ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFVVS8 ],
                couplings = {(0,0):C.GC_1874})

V_1484 = Vertex(name = 'V_1484',
                particles = [ P.t__tilde__, P.t, P.W__minus__, P.Z, P.G__plus__ ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFVVS8 ],
                couplings = {(0,0):C.GC_2166})

