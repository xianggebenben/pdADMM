from concurrent import futures
from configparser import ConfigParser

import pickle
import codecs

import pyarrow.plasma as plasma
# from brain_plasma import Brain
import logging
import time
from tornado.iostream import StreamClosedError

from tornado.tcpserver import TCPServer
from tornado.ioloop import IOLoop
from tornado import gen
import numpy as np
import asyncio
from multiprocessing import Pool
import os

config = ConfigParser()
try:
    config.read(os.path.dirname(os.getcwd())+'/config.ini')
    layer = config.getint('currentLayer','layer')
except:
    config.read('config.ini')
    layer = config.getint('currentLayer','layer')
logger = logging.getLogger()
logger.setLevel(level=logging.INFO)
handler = logging.FileHandler(str(layer)+'.log')
formatter = logging.Formatter('%(asctime)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
sh = logging.StreamHandler()
sh.setFormatter(formatter)
logger.addHandler(sh)

# brain = Brain(path = config['common']['plasma_path'])
sentinel = b'---end---'
# message end with the name_iter|index, for example W01_123|02, W01: W of first layer, 123: iteration, 02: the second part
class MyServer(TCPServer):
    async def handle_stream(self, stream, address):
        # while True:
        try:
            msg = await stream.read_until(sentinel)
            # await asyncio.sleep(1)
    # p.apply_async(self.store_msg, args=(msg,))
            self.store_msg(msg)
        except StreamClosedError:
            logger.warning("Lost client at host %s", address[0])
            # break
        except Exception as e:
            print(e)

    def store_msg(self,msg):
        client = plasma.connect(config['common']['plasma_path'])
        data = pickle.loads(msg[:-19])
        name = msg[-19:-9].decode()
        #W01_123|02, W01: W of first layer, 123: iteration, 02: the second part
        id = plasma.ObjectID(10 * b'0'+msg[-19:-9]) 
        logger.info("Server logs: Received %s" , name)
        while(1):
            client.put(data,id)
            if client.contains(id):
                logger.info("Server logs: Store %s success!",name)
                break
            else:
                logger.info("Put fail, start again!")
        client.disconnect()


if __name__ == '__main__':
    # p = Pool()
    # client = plasma.connect(config['common']['plasma_path'])
    s = MyServer(max_buffer_size=int(1e12))
    s.bind(8888)
    if layer == 0:
        s.start(5)
    else:
        s.start(5)
    # s.listen(8888)
    IOLoop.current().start()
