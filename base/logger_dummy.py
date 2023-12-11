import logging

log = logging.getLogger('dummy')
logging.getLogger('dummy').addHandler(logging.NullHandler())
log.propagate = False
log.error("error")
