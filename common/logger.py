r""" Logging during training/testing """
import datetime
import logging
import os

from tensorboardX import SummaryWriter
import torch

class AverageMeter:
    r""" Stores loss, evaluation results """
    def __init__(self, dataset):
        self.benchmark = dataset.benchmark
        self.nclass = 1
        self.f1_buf = []
        self.precision_buf = []
        self.recall_buf = []
        self.quality_buf = []
        self.cor_buf = []
        self.com_buf = []
        self.loss_buf = dict()
        self.ap_buf = []  # 新增AP缓冲区

    def update(self, f1, precision, recall, quality, cor, com, loss_dict, ap=None):  # 添加ap参数
        self.f1_buf.append(f1)
        self.precision_buf.append(precision)
        self.recall_buf.append(recall)
        self.quality_buf.append(quality)
        self.com_buf.append(com)
        self.cor_buf.append(cor)
        if ap is not None:  # 添加AP数据
            self.ap_buf.append(ap)
        if loss_dict is not None:
            for key in loss_dict.keys():
                if key not in self.loss_buf.keys():
                    self.loss_buf[key] = []
                loss = loss_dict[key].detach().clone()
                if loss is None:
                    loss = torch.tensor(0.0)
                self.loss_buf[key].append(loss)

    # 新增AP计算方法
    def compute_ap(self):
        if not self.ap_buf:
            return torch.tensor(0.0)
        ap = torch.stack(self.ap_buf)
        ap = ap.mean()
        return ap

    def compute_f1(self):
        f1 = torch.stack(self.f1_buf)
        f1 = f1.mean()
        return f1

    def compute_precision(self):
        precision = torch.stack(self.precision_buf)
        precision = precision.mean()
        return precision

    def compute_recall(self):
        recall = torch.stack(self.recall_buf)
        recall = recall.mean()
        return recall

    def compute_quality(self):
        quality = torch.stack(self.quality_buf)
        quality = quality.mean()
        return quality

    def compute_correctness(self):
        correctness = torch.stack(self.cor_buf)
        correctness = correctness.mean()
        return correctness

    def compute_completeness(self):
        completeness = torch.stack(self.com_buf)
        completeness = completeness.mean()
        return completeness

    def write_result(self, split, epoch):
        f1 = self.compute_f1()
        precision = self.compute_precision()
        recall = self.compute_recall()
        msg = '\n*** %s ' % split
        msg += '[@Epoch %02d] ' % epoch
        for key in self.loss_buf.keys():
            loss_buf = torch.stack(self.loss_buf[key])
            msg += 'Avg ' + str(key) + ' :  %6.5f  ' % loss_buf.mean()
        msg += 'F1: %5.2f   ' % f1
        msg += 'Pr: %5.2f   ' % precision
        msg += 'R: %5.2f   ' % recall
        msg += '***\n'
        Logger.info(msg)

    def write_process(self, batch_idx, datalen, epoch, write_batch_idx=20):
        if batch_idx % write_batch_idx == 0:
            dt_ms = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            msg = '[Time: ' + dt_ms + '] '
            msg += '[Epoch: %02d] ' % epoch if epoch != -1 else ''
            msg += '[Batch: %04d/%04d] ' % (batch_idx, datalen)
            f1 = self.compute_f1()
            if epoch != -1:
                for key in self.loss_buf.keys():
                    loss_buf = torch.stack(self.loss_buf[key])
                    msg += str(key) + ' :  %6.5f  ' % loss_buf[-1]
                    msg += 'Avg ' + str(key) + ' :  %6.5f  ' % loss_buf.mean()
            msg += 'F1: %5.2f  |  ' % f1
            Logger.info(msg)

class Logger:
    r""" Writes evaluation results of training/testing """
    @classmethod
    def initialize(cls, args, training):
        logtime = datetime.datetime.now().__format__('_%m%d_%H%M%S')
        logname = args.logname if training else '_TEST_' + args.weight.split('/')[-2].split('.')[0] #+ logtime
        if logname == '': logname = logtime

        cls.logpath = os.path.join('logs', logname + '.log')
        cls.benchmark = args.benchmark
        if not os.path.exists(cls.logpath):
            os.makedirs(cls.logpath)

        logging.basicConfig(filemode='w',
                            filename=os.path.join(cls.logpath, 'log.txt'),
                            level=logging.INFO,
                            format='%(message)s',
                            datefmt='%m-%d %H:%M:%S')

        # Console log config
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        formatter = logging.Formatter('%(message)s')
        console.setFormatter(formatter)
        logging.getLogger('').addHandler(console)

        # Tensorboard writer
        cls.tbd_writer = SummaryWriter(os.path.join(cls.logpath, 'tbd/runs'))

        # Log arguments
        logging.info('\n:=========== Curvilinear Segmentation. with cadet ===========')
        for arg_key in args.__dict__:
            logging.info('| %20s: %-24s' % (arg_key, str(args.__dict__[arg_key])))
        logging.info(':================================================\n')

    @classmethod
    def info(cls, msg):
        r""" Writes log message to log.txt """
        logging.info(msg)

    @classmethod
    def save_model_f1(cls, model, epoch, F1, optimizer):
        torch.save({'epoch': epoch, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}, os.path.join(cls.logpath, 'best_model.pt'))
        cls.info('Model saved @%d w/ val. F1: %5.2f.\n' % (epoch, F1))

    @classmethod
    def save_model_all(cls, model, epoch, F1, Pr, R, optimizer):
        torch.save({'epoch': epoch, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}, os.path.join(cls.logpath, 'best_model_all.pt'))
        cls.info('Model saved @%d w/ val. F1: %5.2f Pr: %5.2f R: %5.2f.\n' % (epoch, F1, Pr, R))

    @classmethod
    def log_params(cls, model):
        backbone_param = 0
        learner_param = 0
        for k in model.state_dict().keys():
            n_param = model.state_dict()[k].view(-1).size(0)
            if k.split('.')[0] in 'backbone':
                if k.split('.')[1] in ['classifier', 'fc']:  # as fc layers are not used in HSNet
                    continue
                backbone_param += n_param
            else:
                learner_param += n_param
        Logger.info('Backbone # param.: %d' % backbone_param)
        Logger.info('Learnable # param.: %d' % learner_param)
        Logger.info('Total # param.: %d' % (backbone_param + learner_param))

