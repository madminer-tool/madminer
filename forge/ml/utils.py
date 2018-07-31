from __future__ import absolute_import, division, print_function, unicode_literals

import logging
import six
import os
from subprocess import Popen, PIPE
import io
import numpy as np
import logging

import torch.nn.functional as F


def get_activation(activation):
    if activation == 'relu':
        return F.relu
    elif activation == 'tanh':
        return F.tanh
    elif activation == 'sigmoid':
        return F.sigmoid
    else:
        raise ValueError('Activation function %s unknown', activation)


def general_init(debug=False):
    logging.basicConfig(format='%(asctime)s  %(message)s', datefmt='%H:%M',
                        level=logging.DEBUG if debug else logging.INFO)

    logging.info('')
    logging.info('------------------------------------------------------------')
    logging.info('|                                                          |')
    logging.info('|  Forge                                                   |')
    logging.info('|                                                          |')
    logging.info('|  Version from July 31, 2018                              |')
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


def load_and_check(filename, warning_threshold=1.e9):

    if filename is None:
        return None

    data = np.load(filename)

    n_nans = np.sum(np.isnan(data))
    n_infs = np.sum(np.isinf(data))
    n_finite = np.sum(np.isfinite(data))

    if n_nans + n_infs > 0:
        logging.warning('Warning: file %s contains %s NaNs and %s Infs, compared to %s finite numbers!',
                        filename, n_nans, n_infs, n_finite)

    smallest = np.nanmin(data)
    largest = np.nanmax(data)

    if np.abs(smallest) > warning_threshold or np.abs(largest) > warning_threshold:
        logging.warning('Warning: file %s has some large numbers, rangin from %s to %s',
                        filename, smallest, largest)

    return data
