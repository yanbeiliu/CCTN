import os
import copy

opt = dict()
opt['device'] = 'cuda:0'
opt['seed'] = '53'
opt['data'] = 'cora'
opt['epochs'] = 1200
opt['lr'] = 0.001
opt['input_dim'] = 1433
opt['momentum'] = 0.9
opt['alpha'] = 0.2
opt['beta'] = 0.8
opt['drop_edge'] = 0.0001
opt['drop_feat1'] = 0.0
opt['drop_feat2'] = 0.2


def command(opt):
    script = 'python traincora.py'
    for opt, val in opt.items():
        script += ' --' + opt + ' ' + str(val)
    return script


def run(opt):
    opt_ = copy.deepcopy(opt)
    os.system(command(opt_))


if __name__ == '__main__':
    run(opt)