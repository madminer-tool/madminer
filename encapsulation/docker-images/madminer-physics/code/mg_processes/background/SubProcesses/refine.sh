#!/bin/bash

# For support of LHAPATH in cluster mode
if [ $CLUSTER_LHAPATH ]; then
  export LHAPATH=$CLUSTER_LHAPATH;
fi

if [[ -e MadLoop5_resources.tar.gz && ! -e MadLoop5_resources ]]; then
tar -xzf MadLoop5_resources.tar.gz
fi
keeplog=%(keeplog)s
if [ "$keeplog" = true ] ; then
    k=%(name)s_app.log
else
    k=/dev/null
fi
script=%(script_name)s                         

grid_directory=%(base_directory)s
j=%(directory)s
     if [[ ! -e $j ]]; then
          mkdir $j
          if [[ -e $grid_directory/ftn26 ]];then
             cp $grid_directory/ftn26 $j/ftn25
          fi 
          if [[ ! -e ../../SubProcesses ]];then
          	 if [[ -e ftn26 ]]; then
          	 	cp ./ftn26 $j/ftn25
          	 fi
          fi	
     fi
     cd $j
     if [ "$keeplog" = true ] ; then
	 rm -f $k
     fi
     rm -f moffset.dat >& /dev/null
      echo   %(offset)s  > moffset.dat
     if  [[ -e ftn26 ]]; then
          cp ftn26 ftn25
     fi
     # create the input file
         echo "    %(nevents)s       %(maxiter)s       %(miniter)s" >& input_sg.txt
         echo "    %(precision)s" >> input_sg.txt
     if [[ ! -e ftn25 ]]; then
         echo "2" >> input_sg.txt   # grid refinement
         echo "1" >> input_sg.txt   # suppress amplitude

     else
         echo "%(grid_refinment)s" >> input_sg.txt
         echo "1" >> input_sg.txt
     fi
     echo "%(nhel)s" >> input_sg.txt
     echo "%(channel)s" >> input_sg.txt

     # run the executable. The loop is design to avoid
     # filesystem problem (executable not found)
     for((try=1;try<=16;try+=1)); 
     do
	 if [ "$keeplog" = true ] ; then
             %(Ppath)s/madevent 2>&1 >> $k <input_sg.txt | tee -a $k;
	     status_code=${PIPESTATUS[0]};
             if [ -s $k ]
             then
		 break
             else
		 echo $try > fail.log 
             fi
	 else
	     %(Ppath)s/madevent 2>&1 >> log.txt <input_sg.txt | tee -a log.txt;
	     status_code=${PIPESTATUS[0]};
	     if [ -s log.txt ]
             then
		 rm log.txt
                 break
             else
                  echo $try > fail.log
             fi
	 fi
     done
     if [[ $status_code -ne 0 ]]; then 
	 rm results.dat
	 echo "ERROR DETECTED"
	 echo "end-code not correct $status_code" > results.dat
     fi     
     if [[ -e ftn26 ]]; then
         cp ftn26 ftn25
     fi

     if [ "$keeplog" = true ] ; then
	 echo "" >> $k; echo "ls status:" >> $k; ls >> $k	 
     else
	 rm ftn26 &> /dev/null
     fi





     cd ../

