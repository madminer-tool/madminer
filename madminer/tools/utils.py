from __future__ import absolute_import, division, print_function, unicode_literals

import logging
import six
import os
from subprocess import Popen, PIPE
import io
import numpy as np


def general_init(debug=False):
    logging.basicConfig(format='%(asctime)s  %(message)s', datefmt='%H:%M',
                        level=logging.DEBUG if debug else logging.INFO)

    logging.info('')
    logging.info('------------------------------------------------------------')
    logging.info('|                                                          |')
    logging.info('|  MadMiner                                                |')
    logging.info('|                                                          |')
    logging.info('|  Version from July 19, 2018                              |')
    logging.info('|                                                          |')
    logging.info('|           Johann Brehmer, Kyle Cranmer, and Felix Kling  |')
    logging.info('|                                                          |')
    logging.info('------------------------------------------------------------')
    logging.info('')

    logging.debug("""
    
                                                      @ @ @ @   @ @ @ @                                
                                                     @. . . . @ . . . . @                              
                                                     @. . . . . . . . . @                              
                   @@@@@@@@@@@@@@@                  @. . . . . . . . . . @                             
              @@@@@///////////////@@@@      @@@@    @////////////////////@    @@@@                     
         @@@@///////////@@@@@            @.......@ @//////////////////////@ @.......@                  
      @@//////@@@@@@@///@               @............@@@@@@@@@@@@@@@@@@...............@                
     @@@@@@@         @///@             @...............................................@               
                     @///@             @................******....****.................@               
                      @//@             @................-----**.._____.................@               
                      @///@             @............./       \*/     \....,..........@                
                      @///@              @...........|         |       |.............@                 
                       @//@                @.........|       * |\ *    |..........@                    
                       @///@                   @@@@...\       /..\ __ / @@@@@@                         
                       @///@                  @,,,,,,@  -----,%%%%%.     @.,,,,,@                      
                        @//@               @,,,,,,,,,@    %,,     ,,%    @.,,,,,,,,@                   
                        @///@            @,,,,,,,,,,@     %,,,   ,,,%    @,,,,,,,,,,@                  
                        @///@           @,,,,,,,,,,,,@@@@@ %,,,,,,,% @@@@@,,,,,,,,,,,@                 
                         @//@           @,,,,,,,,,,,,,,,,,,%%%%%%%,,,,,,,,,,,,,,,,,,,@                 
                         @///@          @,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,@                 
                         @///@_        @,,,,,,,,,,,,,,,----------------,,,,,,,,,,,,,,,,@       /\ ___  
                          @@@(  )      @,,,,,,,\,,/----,,,,,,,,,,,,,,,,----\,,/,,,,,,,,@      /  (___) 
                         (___) ) )    @**,,,,,,,\-,,,,,,,,,,,,,,,,,,,,,,,,,,-/,,,,,,,,,,@    (  \_(___)
                         (___)_)  )@@@@&********,\,,,,,,,,,,,,,,,,,,,,,,,,,,/,,*******&@@@@@@(    (___)
                         (____)@  )%%%%%&*******,,,,,,,,,,,,,,,,,,,,,,,,,,,,********&&%%%%%%%( ___(___)
                         (____)@__)@@@@%%%&*************,,,,,,,,,,,,,**************&%%%%@@@@@@@        
                           @///@      @%%%%&***************************************&%%%%@              
                            @//@      @%%%%%&&&&&&***************************&&&&&&%%%%%%@             
                            @///@    @%%%%%%%%%%%%&*************************&%%%%%%%%%%%%%@            
                            @@@@@    @%%%%%%%%%%%%%%&&&&&&&&&&&&&&&&&&&&&&&&%%%%%%%%%%%%%%@            
                                    @%%%%%%%%%%%%%%%%%%%%%%%%@%%%@%%%%%%%%%%%%%%%%%%%%%%%%%@           
                                    @%%%%%%%%%%%%%%%%%%%%%%%%@%%%@%%%%%%%%%%%%%%%%%%%%%%%%%@           
                                    @%%%%%%%%%%%%%%%%%%%%%%%%@%%%@%%%%%%%%%%%%%%%%%%%%%%%%%@           
                                    @%%%%%%%%%%%%%%%%%%%%%%%%@%%%@%%%%%%%%%%%%%%%%%%%%%%%%%@           
                                    @%%%%%%%%%%%%%%%%%%%%%-----------%%%%%%%%%%%%%%%%%%%%%%@           
                                    @###############%%%%/.............\%%%%################@           
                                    @##################|...............|###################@           
                                     @#################|...............|##################@            
                                      @((((#############\............./##############((((@             
                                       @((((((((((((((((#\___________/#(((((((((((((((((@              
                                          @((((((((((((((((##########((((((((((((((((@                 
                                             @((((((((((((((((((((((((((((((((((((@                    
                                                @@@@@@@@((((((((((((((((@@@@@@@@                       
                                                @##%%%%@                @%%%%##@                       
                                                @######@                @######@                       
                                             @@########@                @########@@                    
                                          @#######@@@@@@                @@@@@@#######@                 
                                  ........@@@@@@@@@..@@@................@@@..@@@@@@@@@........         
    
    """)

    # np.seterr(divide='ignore', invalid='ignore')
    # np.set_printoptions(formatter={'float_kind': lambda x: "%.2f" % x})


def call_command(cmd, log_file=None):
    if log_file is not None:
        with io.open(log_file, 'wb') as log:
            proc = Popen(cmd, stdout=log, stderr=log, shell=True)
            _ = proc.communicate()
            exitcode = proc.returncode

        if exitcode != 0:
            raise RuntimeError(
                'Calling command {} returned exit code {}. Output in file {}.'.format(
                    cmd, exitcode, log_file
                )
            )
    else:
        proc = Popen(cmd, stdout=PIPE, stderr=PIPE, shell=True)
        out, err = proc.communicate()
        exitcode = proc.returncode

        if exitcode != 0:
            raise RuntimeError(
                'Calling command {} returned exit code {}.\n\nStd output:\n\n{}Error output:\n\n{}'.format(
                    cmd, exitcode, out, err
                )
            )

    return exitcode


def create_missing_folders(folders):
    for folder in folders:
        if not os.path.exists(folder):
            os.makedirs(folder)

        elif not os.path.isdir(folder):
            raise OSError('Path {} exists, but is no directory!'.format(folder))


def format_benchmark(parameters, precision=2):
    output = ''

    for i, (key, value) in enumerate(six.iteritems(parameters)):
        if i > 0:
            output += ', '

        value = float(value)

        if value < 2. * 10. ** (- precision) or value > 100.:
            output += str(key) + (' = {0:.' + str(precision) + 'e}').format(value)
        else:
            output += str(key) + (' = {0:.' + str(precision) + 'f}').format(value)

    return output


def shuffle(*arrays):
    """ Shuffles multiple arrays simultaneously"""

    permutation = None
    n_samples = None
    shuffled_arrays = []

    for i, a in enumerate(arrays):
        if a is None:
            shuffled_arrays.append(a)
            continue

        if permutation is None:
            n_samples = a.shape[0]
            permutation = np.random.permutation(n_samples)

        assert a.shape[0] == n_samples
        shuffled_a = a[permutation]
        shuffled_arrays.append(shuffled_a)

    return shuffled_arrays
