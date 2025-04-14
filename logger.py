from collections import defaultdict, deque
import datetime

import torch
import torch.distributed as dist
import time
from loguru import logger
import sys
import os


class SmoothedValue(object):

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        return 

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)


class MetricLogger(object): 
    
    def __init__(self, delimiter, header, logger):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter
        self.header = header
        self.logger = logger

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq):
        i = 0
        
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        log_msg = [
            self.header,
            '[{0' + space_fmt + '}/{1}]',
            'eta: {eta}',
            '{meters}',
            'time: {time}',
            'data: {data}'
        ]
        if torch.cuda.is_available():
            log_msg.append('max mem: {memory:.0f}')
        log_msg = self.delimiter.join(log_msg)
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    message = log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time),
                        memory=torch.cuda.max_memory_allocated() / MB)
                    logger.info(message)
                else:
                    message = log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time))
                    logger.info(message)
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        logger.info('{} Total time: {} ({:.4f} s / it)'.format(
            self.header, total_time_str, total_time / len(iterable)))


import os
import sys
import datetime
from loguru import logger

class Logger:
    def __init__(self, log_dir="logs", prefix="logfile"):
        os.makedirs(log_dir, exist_ok=True)  # Tạo thư mục nếu chưa có

        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.log_file = f"{log_dir}/{prefix}_{timestamp}.log"

        logger.add(sys.stdout, format="{time} {level} {message}", level="INFO")  # Log ra console
        logger.add(self.log_file, format="{time} {level} {message}", level="INFO", rotation="10MB")  # Log ra file

        print(f"Logging to {self.log_file}")
        
    def write(self, message):
        """Ghi log thay thế print(), hỗ trợ live writing"""
        logger.info(message.strip())
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(message.strip() + "\n")  # Ghi vào file ngay lập tức
            f.flush()  # Đảm bảo ghi ngay

        # sys.__stdout__.write(message)  # Ghi ra console ngay lập tức

    def flush(self):
        """Flush dữ liệu (không cần thiết do đã gọi flush() trong write)"""
        pass  

    @staticmethod
    def info(msg):
        """Ghi log mức INFO"""
        logger.info(msg)

    @staticmethod
    def warning(msg):
        """Ghi log mức WARNING"""
        logger.warning(msg)

    @staticmethod
    def error(msg):
        """Ghi log mức ERROR"""
        logger.error(msg)


import os
import datetime
import torch
import sys
import json
from torch.utils.tensorboard import SummaryWriter
from torchinfo import summary
from torchviz import make_dot
import netron

class ModelLogger:
    def __init__(self, log_dir, prefix, model, input_size=(1, 3, 224, 224)):
   
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)

        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        log_file = f"{log_dir}/{prefix}_{timestamp}.log"

        self.log_file = open(log_file, "w", encoding="utf-8")
        self.writer = SummaryWriter(log_dir)

        self.model = model
        self.input_size = input_size

        if model:
            self.log_model_summary()
            sample = []
            for i in range(len(input_size)):
                sample.append(torch.randint(0, 5, (input_size[i])))
            # self.visualize_computational_graph(sample)

    def log_model_summary(self):
        if not self.model:
            return
        model_summary = summary(self.model, input_size=self.input_size, 
                                col_names=["input_size", "output_size", "num_params", "trainable", "mult_adds"],
                                row_settings=("var_names", "depth"),
                                dtypes=[torch.long, torch.long, torch.long, torch.bool], 
                                depth=10, verbose=1)
        with open(f"{self.log_dir}/model_summary.txt", "w", encoding="utf-8") as f:
            f.write(str(model_summary))

    def log_loss(self, loss, step):
        log_message = f"Step {step}: Loss = {loss:.6f}"
        self.write(log_message)
        self.writer.add_scalar("Loss", loss, step)

    def log_gradients(self, step):
        if not self.model:
            return
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                self.writer.add_histogram(f"Gradients/{name}", param.grad, step)

    def visualize_computational_graph(self, sample_input):
        if not self.model:
            return
        output = self.model(*sample_input)
        graph = make_dot(output, params=dict(self.model.named_parameters()))
        graph.render(f"{self.log_dir}/model_graph", format="png", cleanup=True)
        self.write("Computational graph đã được lưu tại 'model_graph.png'")

    def save_and_visualize_model(self, filename="model.onnx"):
        if not self.model:
            return
        dummy_input = torch.randn(*self.input_size)
        torch.onnx.export(self.model, dummy_input, filename, opset_version=11)
        netron.start(filename)

    def write(self, message):
        formatted_message = f"{datetime.datetime.now()} - {message}"
        # print(formatted_message)
        self.log_file.write(formatted_message + "\n")
        self.log_file.flush()

    def close(self):
        self.log_file.close()
        self.writer.close()
