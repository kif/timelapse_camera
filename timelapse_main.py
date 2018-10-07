#!/usr/bin/env python3

"""Main programme to run moving timelaps """

from argparse import ArgumentParser
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

tl = None
if __name__ == "__main__":
    try: 
        from rfoo.utils import rconsole
    except:
        pass
    else:
        rconsole.spawn_server()

    parser = ArgumentParser("timelapse_main.py", 
                            description="TimeLapse photography over a trajectory")
    parser.add_argument("-j", "--json", help="config file", default="")
    parser.add_argument("-d", "--debug", help="debug", default=False, action="store_true")
    
    args = parser.parse_args()
    if args.debug:
        logging.root.setLevel(logging.DEBUG)
    
    from tlc.timelapse import TimeLapse
    if not args.json:
        logger.warning("Using default config ..., please provide a config")
    tl = TimeLapse(config_file=args.json)
    tl.run()
else:
    from tlc.timelapse import TimeLapse
