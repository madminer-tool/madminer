ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
c      written by the UFO converter
ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc

      SUBROUTINE COUP1()

      IMPLICIT NONE
      INCLUDE 'model_functions.inc'

      DOUBLE PRECISION PI, ZERO
      PARAMETER  (PI=3.141592653589793D0)
      PARAMETER  (ZERO=0D0)
      INCLUDE 'input.inc'
      INCLUDE 'coupl.inc'
      GC_1 = -(MDL_AH*MDL_COMPLEXI)
      GC_62 = (MDL_CKM1X1*MDL_EE*MDL_COMPLEXI)/(MDL_SW*MDL_SQRT__2)
      GC_91 = (MDL_EE__EXP__2*MDL_COMPLEXI*MDL_VEV)/(2.000000D+00
     $ *MDL_SW__EXP__2)
      GC_92 = -(MDL_CPHIWL2*MDL_EE__EXP__2*MDL_COMPLEXI*MDL_VEV)
     $ /(5.000000D+05*MDL_SW__EXP__2)
      GC_93 = -(MDL_CPWL2*MDL_EE__EXP__2*MDL_COMPLEXI*MDL_VEV)
     $ /(2.000000D+06*MDL_SW__EXP__2)
      GC_94 = (MDL_CWL2*MDL_EE__EXP__2*MDL_COMPLEXI*MDL_VEV)
     $ /(4.000000D+06*MDL_SW__EXP__2)
      GC_117 = -(MDL_CPHIDL2*MDL_EE__EXP__2*MDL_COMPLEXI
     $ *MDL_VEV__EXP__3)/(2.000000D+06*MDL_SW__EXP__2)
      GC_145 = (MDL_EE*MDL_COMPLEXI*MDL_CONJG__CKM1X1)/(MDL_SW
     $ *MDL_SQRT__2)
      END
