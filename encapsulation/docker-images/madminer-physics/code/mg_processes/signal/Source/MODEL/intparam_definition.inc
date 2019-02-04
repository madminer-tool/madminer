ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
c      written by the UFO converter
ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc

C     Parameters that should not be recomputed event by event.
C     
      IF(READLHA) THEN

        G = 2 * DSQRT(AS*PI)  ! for the first init

        MDL_LAMWS__EXP__2 = MDL_LAMWS**2

        MDL_CKM1X1 = 1.000000D+00-MDL_LAMWS__EXP__2/2.000000D+00

        MDL_CKM1X2 = MDL_LAMWS

        MDL_COMPLEXI = DCMPLX(0.000000D+00,1.000000D+00)

        MDL_LAMWS__EXP__3 = MDL_LAMWS**3

        MDL_CKM1X3 = MDL_AWS*MDL_LAMWS__EXP__3*(-(MDL_ETAWS
     $   *MDL_COMPLEXI)+MDL_RHOWS)

        MDL_CKM2X1 = -MDL_LAMWS

        MDL_CKM2X2 = 1.000000D+00-MDL_LAMWS__EXP__2/2.000000D+00

        MDL_CKM2X3 = MDL_AWS*MDL_LAMWS__EXP__2

        MDL_CKM3X1 = MDL_AWS*MDL_LAMWS__EXP__3*(1.000000D+00-MDL_ETAWS
     $   *MDL_COMPLEXI-MDL_RHOWS)

        MDL_CKM3X2 = -(MDL_AWS*MDL_LAMWS__EXP__2)

        MDL_CKM3X3 = 1.000000D+00

        MDL_NB__2__EXP__0_25 = 2.000000D+00**2.500000D-01

        MDL_VEV = SQRT(DCMPLX(1.000000D+00/MDL_GF))
     $   /MDL_NB__2__EXP__0_25

        MDL_MZ__EXP__2 = MDL_MZ**2

        MDL_MZ__EXP__4 = MDL_MZ**4

        MDL_SQRT__2 = SQRT(DCMPLX(2.000000D+00))

        MDL_MH__EXP__6 = MDL_MH**6

        MDL_MT__EXP__6 = MDL_MT**6

        MDL_MH__EXP__4 = MDL_MH**4

        MDL_MT__EXP__4 = MDL_MT**4

        MDL_MH__EXP__2 = MDL_MH**2

        MDL_MT__EXP__2 = MDL_MT**2

        MDL_VEV__EXP__2 = MDL_VEV**2

        MDL_CPHIDL2__EXP__2 = MDL_CPHIDL2**2

        MDL_VEV__EXP__4 = MDL_VEV**4

        MDL_LAM = (4.000000D+00*MDL_MH__EXP__2-(MDL_CPHIDL2
     $   *MDL_MH__EXP__2*MDL_VEV__EXP__2)/2.500000D+05)/(8.000000D+00
     $   *MDL_VEV__EXP__2*(1.000000D+00-(3.000000D+00*MDL_CPHIDL2
     $   *MDL_VEV__EXP__2)/1.000000D+06+(MDL_CPHIDL2__EXP__2
     $   *MDL_VEV__EXP__4)/5.000000D+11))

        MDL_YB = (MDL_YMB*MDL_SQRT__2)/MDL_VEV

        MDL_YC = (MDL_YMC*MDL_SQRT__2)/MDL_VEV

        MDL_YM = (MDL_YMM*MDL_SQRT__2)/MDL_VEV

        MDL_YT = (MDL_YMT*MDL_SQRT__2)/MDL_VEV

        MDL_YTAU = (MDL_YMTAU*MDL_SQRT__2)/MDL_VEV

        MDL_MUH = SQRT(DCMPLX((-4.000000D+00*MDL_LAM*MDL_VEV__EXP__2
     $   +(MDL_CPHIDL2*MDL_LAM*MDL_VEV__EXP__4)/2.500000D+05)/(
     $   -1.000000D+00+(MDL_CPHIDL2*MDL_VEV__EXP__2)/1.000000D+06)))
     $   /2.000000D+00

        MDL_MH__EXP__12 = MDL_MH**12

        MDL_MH__EXP__10 = MDL_MH**10

        MDL_MH__EXP__8 = MDL_MH**8

        MDL_VEV__EXP__3 = MDL_VEV**3

        MDL_CONJG__CKM1X1 = CONJG(DCMPLX(MDL_CKM1X1))

        MDL_CONJG__CKM1X2 = CONJG(DCMPLX(MDL_CKM1X2))

        MDL_CONJG__CKM1X3 = CONJG(DCMPLX(MDL_CKM1X3))

        MDL_CONJG__CKM2X1 = CONJG(DCMPLX(MDL_CKM2X1))

        MDL_CONJG__CKM2X2 = CONJG(DCMPLX(MDL_CKM2X2))

        MDL_CONJG__CKM2X3 = CONJG(DCMPLX(MDL_CKM2X3))

        MDL_CONJG__CKM3X1 = CONJG(DCMPLX(MDL_CKM3X1))

        MDL_CONJG__CKM3X2 = CONJG(DCMPLX(MDL_CKM3X2))

        MDL_CONJG__CKM3X3 = CONJG(DCMPLX(MDL_CKM3X3))

        MDL_AEW = 1.000000D+00/AEWM1

        MDL_MW = SQRT(DCMPLX(MDL_MZ__EXP__2/2.000000D+00
     $   +SQRT(DCMPLX(MDL_MZ__EXP__4/4.000000D+00-(MDL_AEW*PI
     $   *MDL_MZ__EXP__2)/(MDL_GF*MDL_SQRT__2)))))

        MDL_SQRT__AEW = SQRT(DCMPLX(MDL_AEW))

        MDL_EE = 2.000000D+00*MDL_SQRT__AEW*SQRT(DCMPLX(PI))

        MDL_MW__EXP__2 = MDL_MW**2

        MDL_SW2 = 1.000000D+00-MDL_MW__EXP__2/MDL_MZ__EXP__2

        MDL_EE__EXP__2 = MDL_EE**2

        MDL_MW__EXP__12 = MDL_MW**12

        MDL_MW__EXP__10 = MDL_MW**10

        MDL_MW__EXP__8 = MDL_MW**8

        MDL_MW__EXP__6 = MDL_MW**6

        MDL_MW__EXP__4 = MDL_MW**4

        MDL_AH = (4.700000D+01*MDL_EE__EXP__2*(1.000000D+00-(2.000000D
     $   +00*MDL_MH__EXP__4)/(9.870000D+02*MDL_MT__EXP__4)-(1.400000D
     $   +01*MDL_MH__EXP__2)/(7.050000D+02*MDL_MT__EXP__2)+(2.130000D
     $   +02*MDL_MH__EXP__12)/(2.634632D+07*MDL_MW__EXP__12)
     $   +(5.000000D+00*MDL_MH__EXP__10)/(1.197560D+05*MDL_MW__EXP__10)
     $   +(4.100000D+01*MDL_MH__EXP__8)/(1.809500D+05*MDL_MW__EXP__8)
     $   +(8.700000D+01*MDL_MH__EXP__6)/(6.580000D+04*MDL_MW__EXP__6)
     $   +(5.700000D+01*MDL_MH__EXP__4)/(6.580000D+03*MDL_MW__EXP__4)
     $   +(3.300000D+01*MDL_MH__EXP__2)/(4.700000D+02*MDL_MW__EXP__2)))
     $   /(7.200000D+01*PI**2*MDL_VEV)

        MDL_CW = SQRT(DCMPLX(1.000000D+00-MDL_SW2))

        MDL_SQRT__SW2 = SQRT(DCMPLX(MDL_SW2))

        MDL_SW = MDL_SQRT__SW2

        MDL_G1 = MDL_EE/MDL_CW

        MDL_GW = MDL_EE/MDL_SW

        MDL_EE__EXP__4 = MDL_EE**4

        MDL_SW__EXP__4 = MDL_SW**4

        MDL_EE__EXP__3 = MDL_EE**3

        MDL_SW__EXP__3 = MDL_SW**3

        MDL_CW__EXP__2 = MDL_CW**2

        MDL_SW__EXP__2 = MDL_SW**2

        MDL_EE__EXP__6 = MDL_EE**6

        MDL_SW__EXP__6 = MDL_SW**6

        MDL_EE__EXP__5 = MDL_EE**5

        MDL_SW__EXP__5 = MDL_SW**5

        MDL_CW__EXP__3 = MDL_CW**3

      ENDIF
C     
C     Parameters that should be recomputed at an event by even basis.
C     
      AS = G**2/4/PI

      MDL_SQRT__AS = SQRT(DCMPLX(AS))

      MDL_G__EXP__2 = G**2

      MDL_GH = -(MDL_G__EXP__2*(1.000000D+00+(1.300000D+01
     $ *MDL_MH__EXP__6)/(1.680000D+04*MDL_MT__EXP__6)+MDL_MH__EXP__4
     $ /(1.680000D+02*MDL_MT__EXP__4)+(7.000000D+00*MDL_MH__EXP__2)
     $ /(1.200000D+02*MDL_MT__EXP__2)))/(1.200000D+01*PI**2*MDL_VEV)

      MDL_GPHI = -(MDL_G__EXP__2*(1.000000D+00+MDL_MH__EXP__6
     $ /(5.600000D+02*MDL_MT__EXP__6)+MDL_MH__EXP__4/(9.000000D+01
     $ *MDL_MT__EXP__4)+MDL_MH__EXP__2/(1.200000D+01*MDL_MT__EXP__2)))
     $ /(8.000000D+00*PI**2*MDL_VEV)

C     
C     Definition of the EW coupling used in the write out of aqed
C     
      GAL(1) = 3.5449077018110318D0 / DSQRT(AEWM1)
      GAL(2) = 1D0

