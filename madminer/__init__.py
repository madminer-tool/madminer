from madminer.__version__ import __version__

import logging

logger = logging.getLogger(__name__)

logger.info("")
logger.info("------------------------------------------------------------")
logger.info("|                                                          |")
logger.info("|  MadMiner v{}|".format(__version__.ljust(46)))
logger.info("|                                                          |")
logger.info("|           Johann Brehmer, Kyle Cranmer, and Felix Kling  |")
logger.info("|                                                          |")
logger.info("------------------------------------------------------------")
logger.info("")
