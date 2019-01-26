from madminer.__version__ import __version__

import logging

logging.getLogger("madminer").addHandler(logging.NullHandler())

logger = logging.getLogger(__name__)

logger.info("")
logger.info("------------------------------------------------------------------------")
logger.info("|                                                                      |")
logger.info("|  MadMiner v{}|".format(__version__.ljust(58)))
logger.info("|                                                                      |")
logger.info("|         Johann Brehmer, Felix Kling, Irina Espejo, and Kyle Cranmer  |")
logger.info("|                                                                      |")
logger.info("------------------------------------------------------------------------")
logger.info("")
