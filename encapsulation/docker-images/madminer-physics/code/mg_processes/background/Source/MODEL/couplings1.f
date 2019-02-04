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
      GC_3 = -(MDL_EE*MDL_COMPLEXI)/3.000000D+00
      GC_4 = (2.000000D+00*MDL_EE*MDL_COMPLEXI)/3.000000D+00
      GC_6 = MDL_EE*MDL_COMPLEXI
      GC_8 = MDL_EE__EXP__2*MDL_COMPLEXI
      GC_62 = (MDL_CKM1X1*MDL_EE*MDL_COMPLEXI)/(MDL_SW*MDL_SQRT__2)
      GC_71 = -(MDL_CW*MDL_EE*MDL_COMPLEXI)/(2.000000D+00*MDL_SW)
      GC_72 = (MDL_CW*MDL_EE*MDL_COMPLEXI)/(2.000000D+00*MDL_SW)
      GC_79 = -(MDL_EE*MDL_COMPLEXI*MDL_SW)/(6.000000D+00*MDL_CW)
      GC_145 = (MDL_EE*MDL_COMPLEXI*MDL_CONJG__CKM1X1)/(MDL_SW
     $ *MDL_SQRT__2)
      END
