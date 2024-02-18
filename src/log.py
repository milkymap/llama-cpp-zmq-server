import logging 

logging.basicConfig(
    format='%(asctime)s - %(filename)s - %(lineno)03d - %(levelname)s - %(message)s',
    level=logging.DEBUG
)

logger = logging.getLogger(name='[imagetopic]')

if __name__ == '__main__':
    logger.info('log initialized')