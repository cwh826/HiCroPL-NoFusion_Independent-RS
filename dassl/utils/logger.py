import os
import sys
import time
import os.path as osp

from .tools import mkdir_if_missing

__all__ = ["Logger", "setup_logger"]


class Logger:
    """Write console output to external text file.

    Imported from `<https://github.com/Cysu/open-reid/blob/master/reid/utils/logging.py>`_

    Args:
        fpath (str): directory to save logging file.

    Examples::
       >>> import sys
       >>> import os.path as osp
       >>> save_dir = 'output/experiment-1'
       >>> log_name = 'train.log'
       >>> sys.stdout = Logger(osp.join(save_dir, log_name))
    """

    def __init__(self, fpath=None):
        self.console = sys.stdout
        self.file = None
        if fpath is not None:
            mkdir_if_missing(osp.dirname(fpath))
            self.file = open(fpath, "w")

    def __del__(self):
        self.close()

    def __enter__(self):
        pass

    def __exit__(self, *args):
        self.close()

    def write(self, msg):
        self.console.write(msg)
        if self.file is not None:
            self.file.write(msg)

    def flush(self):
        self.console.flush()
        if self.file is not None:
            self.file.flush()
            os.fsync(self.file.fileno())

    def close(self):
        self.console.close()
        if self.file is not None:
            self.file.close()


def setup_logger(output=None):  # 把程序里所有 print() 输出同时写到终端和文件。
    if output is None:
        return

    if output.endswith(".txt") or output.endswith(".log"):  # 确定日志文件路径，传的是目录就在下面生成log.txt，是文件就直接用。
        fpath = output
    else:
        fpath = osp.join(output, "log.txt")

    if osp.exists(fpath):   #  如果 log.txt 已经存在，给文件名追加时间戳（如 log.txt-2026-08-05-14-30-00），防止覆盖之前的日志。
        # make sure the existing log file is not over-written
        fpath += time.strftime("-%Y-%m-%d-%H-%M-%S")

    sys.stdout = Logger(fpath)  # 把 Python 的标准输出替换成自定义的 Logger 对象。sys.stdout 替换成了 Logger 对象。之后你代码里所有的 print() 底层全都会调用 Logger.write()
