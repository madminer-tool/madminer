*
* $Id: dgauss.f,v 1.2 2007/01/07 00:14:07 madgraph Exp $
*
* $Log: dgauss.f,v $
* Revision 1.2  2007/01/07 00:14:07  madgraph
* Merged version 4.1 to main branch
*
* Revision 1.1.2.1  2006/09/29 23:35:23  madgraph
* Introducing MLM matching
*
* Revision 1.1.1.1  1996/04/01 15:02:13  mclareni
* Mathlib gen
*
*
      FUNCTION DGAUSS(F,A,B,EPS)
      IMPLICIT DOUBLE PRECISION (A-H,O-Z)

      CHARACTER NAME*(*)
      PARAMETER (NAME = 'DGAUSS')

*
* $Id: dgauss.f,v 1.2 2007/01/07 00:14:07 madgraph Exp $
*
* $Log: dgauss.f,v $
* Revision 1.2  2007/01/07 00:14:07  madgraph
* Merged version 4.1 to main branch
*
* Revision 1.1.2.1  2006/09/29 23:35:23  madgraph
* Introducing MLM matching
*
* Revision 1.1.1.1  1996/04/01 15:02:13  mclareni
* Mathlib gen
*
*
*
* gausscod.inc
*
      DIMENSION W(12),X(12)

      PARAMETER (Z1 = 1, HF = Z1/2, CST = 5*Z1/1000)

      DATA X( 1) /9.6028985649753623D-1/, W( 1) /1.0122853629037626D-1/
      DATA X( 2) /7.9666647741362674D-1/, W( 2) /2.2238103445337447D-1/
      DATA X( 3) /5.2553240991632899D-1/, W( 3) /3.1370664587788729D-1/
      DATA X( 4) /1.8343464249564980D-1/, W( 4) /3.6268378337836198D-1/
      DATA X( 5) /9.8940093499164993D-1/, W( 5) /2.7152459411754095D-2/
      DATA X( 6) /9.4457502307323258D-1/, W( 6) /6.2253523938647893D-2/
      DATA X( 7) /8.6563120238783174D-1/, W( 7) /9.5158511682492785D-2/
      DATA X( 8) /7.5540440835500303D-1/, W( 8) /1.2462897125553387D-1/
      DATA X( 9) /6.1787624440264375D-1/, W( 9) /1.4959598881657673D-1/
      DATA X(10) /4.5801677765722739D-1/, W(10) /1.6915651939500254D-1/
      DATA X(11) /2.8160355077925891D-1/, W(11) /1.8260341504492359D-1/
      DATA X(12) /9.5012509837637440D-2/, W(12) /1.8945061045506850D-1/

      H=0
      IF(B .EQ. A) GO TO 99
      CONST=CST/ABS(B-A)
      BB=A
    1 AA=BB
      BB=B
    2 C1=HF*(BB+AA)
      C2=HF*(BB-AA)
      S8=0
      DO 3 I = 1,4
      U=C2*X(I)
    3 S8=S8+W(I)*(F(C1+U)+F(C1-U))
      S16=0
      DO 4 I = 5,12
      U=C2*X(I)
    4 S16=S16+W(I)*(F(C1+U)+F(C1-U))
      S16=C2*S16
      IF(ABS(S16-C2*S8) .LE. EPS*(1+ABS(S16))) THEN
       H=H+S16
       IF(BB .NE. B) GO TO 1
      ELSE
       BB=C1
       IF(1+CONST*ABS(C2) .NE. 1) GO TO 2
       H=0
       WRITE(*,*) NAME,'ERROR: TOO HIGH ACCURACY REQUIRED'
       GO TO 99
      END IF
      

   99 DGAUSS=H
      RETURN
      END

