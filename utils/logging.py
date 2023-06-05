"""
Based on rllab's logger.

https://github.com/rll/rllab
"""
from enum import Enum
from contextlib import contextmanager
import numpy as np
import os
import os.path as osp
import sys
import datetime
import dateutil.tz
import csv
import json
import pickle
import errno
import gzip

from utils.tabulate import tabulate

# dev
from utils.core import np_ify, torch_ify
import seaborn as sns
import matplotlib.pyplot as plt
from envs.point import PointEnv
from utils.env_utils import env_producer
from optimistic_exploration import get_optimistic_exploration_action
import pickle
# dev


class TerminalTablePrinter(object):
    def __init__(self):
        self.headers = None
        self.tabulars = []

    def print_tabular(self, new_tabular):
        if self.headers is None:
            self.headers = [x[0] for x in new_tabular]
        else:
            assert len(self.headers) == len(new_tabular)
        self.tabulars.append([x[1] for x in new_tabular])
        self.refresh()

    def refresh(self):
        import os
        rows, columns = os.popen('stty size', 'r').read().split()
        tabulars = self.tabulars[-(int(rows) - 3):]
        sys.stdout.write("\x1b[2J\x1b[H")
        sys.stdout.write(tabulate(tabulars, self.headers))
        sys.stdout.write("\n")


class MyEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, type):
            return {'$class': o.__module__ + "." + o.__name__}
        elif isinstance(o, Enum):
            return {
                '$enum': o.__module__ + "." + o.__class__.__name__ + '.' + o.name
            }
        elif callable(o):
            return {
                '$function': o.__module__ + "." + o.__name__
            }
        return json.JSONEncoder.default(self, o)


def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


class Logger(object):
    def __init__(self):
        self._prefixes = []
        self._prefix_str = ''

        self._tabular_prefixes = []
        self._tabular_prefix_str = ''

        self._tabular = []

        self._text_outputs = []
        self._tabular_outputs = []

        self._text_fds = {}
        self._tabular_fds = {}
        self._tabular_header_written = set()

        self._snapshot_dir = None
        self._snapshot_mode = 'all'
        self._snapshot_gap = 1
        self.keep_first = 0

        self._log_tabular_only = False
        self._header_printed = False
        self.table_printer = TerminalTablePrinter()
        self.prev_dict = None

    def reset(self):
        self.__init__()

    def _add_output(self, file_name, arr, fds, mode='a'):
        if file_name not in arr:
            mkdir_p(os.path.dirname(file_name))
            arr.append(file_name)
            fds[file_name] = open(file_name, mode)

    def _remove_output(self, file_name, arr, fds):
        if file_name in arr:
            fds[file_name].close()
            del fds[file_name]
            arr.remove(file_name)

    def push_prefix(self, prefix):
        self._prefixes.append(prefix)
        self._prefix_str = ''.join(self._prefixes)

    def add_text_output(self, file_name):
        self._add_output(file_name, self._text_outputs, self._text_fds,
                         mode='a')

    def remove_text_output(self, file_name):
        self._remove_output(file_name, self._text_outputs, self._text_fds)

    def add_tabular_output(self, file_name, relative_to_snapshot_dir=False):
        if relative_to_snapshot_dir:
            file_name = osp.join(self._snapshot_dir, file_name)
        self._add_output(file_name, self._tabular_outputs, self._tabular_fds,
                         mode='a')

    def remove_tabular_output(self, file_name, relative_to_snapshot_dir=False):
        if relative_to_snapshot_dir:
            file_name = osp.join(self._snapshot_dir, file_name)
        if self._tabular_fds[file_name] in self._tabular_header_written:
            self._tabular_header_written.remove(self._tabular_fds[file_name])
        self._remove_output(
            file_name, self._tabular_outputs, self._tabular_fds)

    def set_snapshot_dir(self, dir_name):
        self._snapshot_dir = dir_name

    def get_snapshot_dir(self, ):
        return self._snapshot_dir

    def get_snapshot_mode(self, ):
        return self._snapshot_mode

    def set_snapshot_mode(self, mode):
        self._snapshot_mode = mode

    def get_snapshot_gap(self, ):
        return self._snapshot_gap

    def set_snapshot_gap(self, gap):
        self._snapshot_gap = gap

    def get_keep_first(self, ):
        return self._keep_first

    def set_keep_first(self, gap):
        self._keep_first = gap

    def set_log_tabular_only(self, log_tabular_only):
        self._log_tabular_only = log_tabular_only

    def get_log_tabular_only(self, ):
        return self._log_tabular_only

    def set_alg(self, variant): # DEBUG: FIXME
        self.alg = variant['alg']

    def log(self, s, with_prefix=True, with_timestamp=True):
        out = s
        if with_prefix:
            out = self._prefix_str + out
        if with_timestamp:
            now = datetime.datetime.now(dateutil.tz.tzlocal())
            timestamp = now.strftime('%Y-%m-%d %H:%M:%S.%f %Z')
            out = "%s | %s" % (timestamp, out)
        if not self._log_tabular_only:
            # Also log to stdout
            out_str = out + '\n'
            print(out_str)
            for fd in list(self._text_fds.values()):
                fd.write(out_str)
                fd.flush()
            sys.stdout.flush()

    def record_tabular(self, key, val):
        self._tabular.append((self._tabular_prefix_str + str(key), str(val)))

    def record_dict(self, d, prefix=None):
        if prefix is not None:
            self.push_tabular_prefix(prefix)
        for k, v in d.items():
            self.record_tabular(k, v)
        if prefix is not None:
            self.pop_tabular_prefix()

    def push_tabular_prefix(self, key):
        self._tabular_prefixes.append(key)
        self._tabular_prefix_str = ''.join(self._tabular_prefixes)

    def pop_tabular_prefix(self, ):
        del self._tabular_prefixes[-1]
        self._tabular_prefix_str = ''.join(self._tabular_prefixes)

    def save_extra_data(self, data, file_name='extra_data.pkl', mode='joblib'):
        """
        Data saved here will always override the last entry

        :param data: Something pickle'able.
        """
        file_name = osp.join(self._snapshot_dir, file_name)
        if mode == 'joblib':
            import joblib
            joblib.dump(data, file_name, compress=3)
        elif mode == 'pickle':
            pickle.dump(data, open(file_name, "wb"))
        else:
            raise ValueError("Invalid mode: {}".format(mode))
        return file_name

    def get_table_dict(self, ):
        return dict(self._tabular)

    def get_table_key_set(self, ):
        return set(key for key, value in self._tabular)

    @contextmanager
    def prefix(self, key):
        self.push_prefix(key)
        try:
            yield
        finally:
            self.pop_prefix()

    @contextmanager
    def tabular_prefix(self, key):
        self.push_tabular_prefix(key)
        yield
        self.pop_tabular_prefix()

    def log_variant(self, log_file, variant_data):
        mkdir_p(os.path.dirname(log_file))
        with open(log_file, "w") as f:
            json.dump(variant_data, f, indent=2, sort_keys=True, cls=MyEncoder)

    def record_tabular_misc_stat(self, key, values, placement='back'):
        if placement == 'front':
            prefix = ""
            suffix = key
        else:
            prefix = key
            suffix = ""
        if len(values) > 0:
            self.record_tabular(prefix + "Average" +
                                suffix, np.average(values))
            self.record_tabular(prefix + "Std" + suffix, np.std(values))
            self.record_tabular(prefix + "Median" + suffix, np.median(values))
            self.record_tabular(prefix + "Min" + suffix, np.min(values))
            self.record_tabular(prefix + "Max" + suffix, np.max(values))
        else:
            self.record_tabular(prefix + "Average" + suffix, np.nan)
            self.record_tabular(prefix + "Std" + suffix, np.nan)
            self.record_tabular(prefix + "Median" + suffix, np.nan)
            self.record_tabular(prefix + "Min" + suffix, np.nan)
            self.record_tabular(prefix + "Max" + suffix, np.nan)

    def dump_tabular(self, *args, **kwargs):
        wh = kwargs.pop("write_header", None)
        if len(self._tabular) > 0:
            if self._log_tabular_only:
                self.table_printer.print_tabular(self._tabular)
            else:
                for line in tabulate(self._tabular).split('\n'):
                    self.log(line, *args, **kwargs)
            tabular_dict = dict(self._tabular)
            # Also write to the csv files
            # This assumes that the keys in each iteration won't change!
            # if self.prev_dict is not None:
            #     old_keys = self.prev_dict.keys()
            #     new_keys = tabular_dict.keys()
            #     difference = set(old_keys) ^ set(new_keys)
            #     print(difference)
            # self.prev_dict = tabular_dict
            for tabular_fd in list(self._tabular_fds.values()):
                writer = csv.DictWriter(tabular_fd,
                                        fieldnames=list(tabular_dict.keys()))
                if wh or (
                        wh is None and tabular_fd not in self._tabular_header_written):
                    writer.writeheader()
                    self._tabular_header_written.add(tabular_fd)
                writer.writerow(tabular_dict)
                tabular_fd.flush()
            del self._tabular[:]

    def pop_prefix(self, ):
        del self._prefixes[-1]
        self._prefix_str = ''.join(self._prefixes)

    def save_itr_params(self, itr, params, best=False):

        if self._snapshot_dir:
            if best:
                file_name = osp.join(self._snapshot_dir,
                                     'best.zip_pkl')
                with open(file_name, 'wb') as f:
                    pickle.dump(params, f)
            elif itr <= self._keep_first:
                file_name = osp.join(self._snapshot_dir,
                                     'itr_%d.zip_pkl' % itr)
                with open(file_name, 'wb') as f:
                    pickle.dump(params, f)
            elif self._snapshot_mode == 'all':
                file_name = osp.join(self._snapshot_dir,
                                     'itr_%d.zip_pkl' % itr)
                with open(file_name, 'wb') as f:
                    pickle.dump(params, f)
                #pickle.dump(params, gzip.open(file_name, "wb"))
            elif self._snapshot_mode == 'last':
                # override previous params
                file_name = osp.join(self._snapshot_dir, 'params.zip_pkl')
                with open(file_name, 'wb') as f:
                    pickle.dump(params, f)
                #pickle.dump(params, gzip.open(file_name, "wb"))
            elif self._snapshot_mode == "gap":
                if itr % self._snapshot_gap == 0:
                    file_name = osp.join(
                        self._snapshot_dir, 'itr_%d.zip_pkl' % itr)
                    with open(file_name, 'wb') as f:
                        pickle.dump(params, f)
                    #pickle.dump(params, gzip.open(file_name, "wb"))

                    # Also save as a genericly named file
                    # to make loading easier
                    file_name = osp.join(self._snapshot_dir, 'params.zip_pkl')
                    with open(file_name, 'wb') as f:
                        pickle.dump(params, f)
                    #pickle.dump(params, gzip.open(file_name, "wb"))

            # Save the params every snapshot_gap and override previously saved result
            elif self._snapshot_mode == "last_every_gap":
                if itr % self._snapshot_gap == 0:
                    file_name = osp.join(self._snapshot_dir, 'params.zip_pkl')
                    with open(file_name, 'wb') as f:
                        pickle.dump(params, f)
                    #pickle.dump(params, gzip.open(file_name, "wb"))

            elif self._snapshot_mode == "gap_and_last":
                if itr % self._snapshot_gap == 0:
                    file_name = osp.join(
                        self._snapshot_dir, 'itr_%d.zip_pkl' % itr)
                    with open(file_name, 'wb') as f:
                        pickle.dump(params, f)
                    #pickle.dump(params, gzip.open(file_name, "wb"))
                file_name = osp.join(self._snapshot_dir, 'params.zip_pkl')
                with open(file_name, 'wb') as f:
                    pickle.dump(params, f)
                #pickle.dump(params, gzip.open(file_name, "wb"))
            elif self._snapshot_mode == 'none':
                pass
            else:
                raise NotImplementedError

    def save_sampled_data(self, ob_sampled, ac_sampled):
        file_name = osp.join(
            self._snapshot_dir, 'sampled_states.csv')
        with open(file_name, 'a') as f:
            for i in range(ob_sampled.shape[0]):
                f.write(','.join(["%.2f" % i for i in ob_sampled[i].tolist()])+'\n')
                # pickle.dump(ac_sampled, f)
        file_name = osp.join(
            self._snapshot_dir, 'sampled_actions.csv')
        with open(file_name, 'a') as f:
            for i in range(ac_sampled.shape[0]):
                f.write(','.join(["%.2f" % i for i in ac_sampled[i].tolist()])+'\n')
            # pickle.dump(ob_sampled, f)

    def save_heatmap(self, trainer, domain, epoch, state_data, action_data): # dev
        if epoch <= 0:
            # self.algorithm = 'g-oac'
            self.log2 = False
            self.folder = os.path.join(self._snapshot_dir, 'heatmaps')
            if not os.path.exists(self.folder):
                os.makedirs(self.folder)
        #  self._snapshot_dir # dir
        if domain in ['point', 'ant_maze']:
            if epoch == 0:
                self.ori_min_state_x = 0
                self.ori_max_state_x = 25
                self.ori_min_state_y = -9.7
                self.ori_max_state_y = 9.7
                min_state_x, max_state_x = -1, 1
                min_state_y, max_state_y = -1, 1

                delta_state = 0.05
                self.xs = np.array([min_state_x + i * delta_state for i in range(int((max_state_x - min_state_x) / delta_state + 1))])
                self.ya = np.array([min_state_y + i * delta_state for i in range(int((max_state_y - min_state_y) / delta_state + 1))])
                self.cum_histo1 = np.zeros(shape=[41, 41], dtype=np.float64)
                self.cum_histo2 = np.zeros(shape=[41, 41], dtype=np.float64)
                self.unclipped = False

            variant = json.load(open(self._snapshot_dir + 'variant.json', 'r'))
            # env = PointEnv(difficulty=(variant['difficulty']))
            if self.unclipped:
                env_args = {
                    'clip_state': False,
                    'terminal': True,
                }
            else:
                env_args = {
                    'clip_state': True,
                    'terminal': True,
                }
            env = env_producer(domain, 0, difficulty=variant['difficulty'], **env_args)

            fig, ax = plt.subplots(2,2,figsize=(12, 9))
   
            n = 900
            obs_x = np.zeros([n], dtype=np.float)
            obs_y = np.zeros([n], dtype=np.float)
            for i in range(n):
                if i % 300 == 0:
                    ob = env.reset()
                # if self.alg = 'oac':
                #     get_optimistic_exploration_action(
                # o, **optimistic_exploration_kwargs)
                ac, *_ = trainer.policy(obs=torch_ify(ob))
                ac = np_ify(ac)
                
                ob, r, done, *_ = env.step(ac)
                obs_x[i] = ob[0]
                obs_y[i] = ob[1]
                if done:
                    print("Reached Goal!")
                    ob = env.reset()
            
            if self.unclipped:
                histog1 = np.histogram2d(obs_x, obs_y, bins=[len(self.xs),len(self.ya)], range=[[0, 25], [-9.7, 9.7]])[0]  
            else: 
                histog1 = np.histogram2d(obs_x, obs_y, bins=[len(self.xs),len(self.ya)], range=[[-1, 1], [-1, 1]])[0]  
            histog1T = histog1.T
            self.cum_histo1 += histog1T
            if self.log2:
                histog1T = np.log2(histog1T, out=np.zeros_like(histog1T), where=(histog1T!=0))
                plot_cum_histo1 = np.log2(self.cum_histo1, out=np.zeros_like(histog1T), where=(histog1T!=0))
            else:
                plot_cum_histo1 = self.cum_histo1

            tl = 10
            nyt = int((len(self.ya))/tl)+1
            nxt = int(len(self.xs)/tl)+1
            JG00 = sns.heatmap(data=histog1T, xticklabels = tl, yticklabels = tl, cmap="jet_r", ax=ax[0,0])
            JG00.set_xticklabels(np.linspace(self.ori_min_state_x,self.ori_max_state_x,nxt))
            JG00.set_yticklabels(np.linspace(self.ori_min_state_y,self.ori_max_state_y,nyt))
            JG00.set_title('expl policy')
            JG01 = sns.heatmap(data=plot_cum_histo1, xticklabels = tl, yticklabels = tl, vmax=1000, cmap="jet_r", ax=ax[0,1])
            JG01.set_xticklabels(np.linspace(self.ori_min_state_x,self.ori_max_state_x,nxt))
            JG01.set_yticklabels(np.linspace(self.ori_min_state_y,self.ori_max_state_y,nyt))
            JG01.set_title('expl policy cumulative polices')

            ob = env.reset()
            obs_x = np.zeros([300], dtype=np.float)
            obs_y = np.zeros([300], dtype=np.float)
            n = 300
            for i in range(n):
                if self.alg in ['g-oac', 'gs-oac']:
                    ac, *_ = trainer.target_policy.get_action(torch_ify(ob), deterministic=True)
                else:
                    ac, *_ = trainer.policy.get_action(torch_ify(ob), deterministic=True)

                ac = np_ify(ac)
                
                ob, r, done, *_ = env.step(ac)
                obs_x[i] = ob[0]
                obs_y[i] = ob[1]
                if done:
                    print("Reached Goal!")
                    ob = env.reset()
            if self.unclipped:
                histog2 = np.histogram2d(obs_x, obs_y, bins=[len(self.xs),len(self.ya)], range=[[0, 25], [-9.7, 9.7]])[0]  
            else: 
                histog2 = np.histogram2d(obs_x, obs_y, bins=[len(self.xs),len(self.ya)], range=[[-1, 1], [-1, 1]])[0]  
            histog2T = histog2.T
            self.cum_histo2 += histog2T
            if self.log2:
                histog2T = np.log2(histog2T, out=np.zeros_like(histog2T), where=(histog2T!=0))
                plot_cum_histo2 = np.log2(self.cum_histo2, out=np.zeros_like(histog1T), where=(histog1T!=0))
            else:
                plot_cum_histo2 = self.cum_histo2
            
            JG01 = sns.heatmap(data=histog2T, xticklabels = tl, yticklabels = tl, cmap="jet_r", ax=ax[1,0])
            JG01.set_xticklabels(np.linspace(self.ori_min_state_x,self.ori_max_state_x,nxt))
            JG01.set_yticklabels(np.linspace(self.ori_min_state_y,self.ori_max_state_y,nyt))
            JG01.set_title('target policy')
            JG11 = sns.heatmap(data=plot_cum_histo2, xticklabels = tl, vmax=1000, yticklabels = tl, cmap="jet_r", ax=ax[1,1])
            JG11.set_xticklabels(np.linspace(self.ori_min_state_x,self.ori_max_state_x,nxt))
            JG11.set_yticklabels(np.linspace(self.ori_min_state_y,self.ori_max_state_y,nyt))
            JG11.set_title('target policy cumulative policies')

            # # save self.cum_histo as pkl
            # comm_dir = os.path.join(self._snapshot_dir, 'ch')
            # if not os.path.exists(comm_dir):
            #     os.makedirs(comm_dir)

            # path1 = os.path.join(comm_dir, "cum_histo1_" + str(epoch) + ".npy")
            # path2 = os.path.join(comm_dir, "cum_histo2_" + str(epoch) + ".npy")
            # np.save(path1, self.cum_histo1)
            # np.save(path2, self.cum_histo2)

        if domain == 'point' and False: # replay buffer
            if epoch == 0:
                self.ori_min_state_x = 0
                self.ori_max_state_x = 25
                self.ori_min_state_y = -9.7
                self.ori_max_state_y = 9.7
                min_state_x, max_state_x = -1, 1
                min_state_y, max_state_y = -1, 1

                delta_state = 0.05
                self.xs = np.array([min_state_x + i * delta_state for i in range(int((max_state_x - min_state_x) / delta_state + 1))])
                self.ya = np.array([min_state_y + i * delta_state for i in range(int((max_state_y - min_state_y) / delta_state + 1))])

            if len(state_data):
                histog = np.histogram2d(state_data[:,0], state_data[:,1], bins=[41,41], range=[[-1, 1], [-1, 1]])[0] 

                histogT = histog.T
                if self.log2:
                    histogT = np.log2(histogT, out=np.zeros_like(histogT), where=(histogT!=0))

            fig = plt.figure(figsize=(12, 9))
            tl = 10
            nyt = int((len(self.ya))/tl)+1
            nxt = int(len(self.xs)/tl)+1
            if len(state_data):
                JG01 = sns.heatmap(data=histogT, xticklabels = tl, yticklabels = tl, vmax=2000, cmap="jet_r")
            JG01.set_xticklabels(np.linspace(self.ori_min_state_x,self.ori_max_state_x,nxt))
            JG01.set_yticklabels(np.linspace(self.ori_min_state_y,self.ori_max_state_y,nyt))
                

        if domain in ['riverswim', 'lqg', 'cliff_mono']:
            if epoch <= 0:
                self.vmax_std = -1

                env_args = {}
                if domain == 'riverswim':
                    env_args['dim'] = 25
                    self.ori_min_state = 0
                    self.ori_max_state = 25
                    self.ori_min_action = -1
                    self.ori_max_action = 1
                elif domain == 'lqg':
                    self.ori_min_state = -4 
                    self.ori_max_state = 4
                    self.ori_min_action = -4
                    self.ori_max_action = 4
                elif domain == 'cliff_mono':
                    self.ori_min_state = 0
                    self.ori_max_state = 12
                    self.ori_min_action = -1
                    self.ori_max_action = 1
                else:
                    raise ValueError('Not implemented')

                eval_env = env_producer(domain, 0, **env_args)

                min_state, max_state = -1, 1
                # min_state, max_state = 0, 25
                min_action, max_action = eval_env.action_space.low[0], eval_env.action_space.high[0]

                delta_state = 0.05
                # delta_state = 1
                delta_action = 0.05
                self.xs = np.array([min_state + i * delta_state for i in range(int((max_state - min_state) / delta_state + 1))])
                self.ya = np.array([min_action + i * delta_action for i in range(int((max_action - min_action) / delta_action + 1))])

                self.heatmap1 = np.zeros((len(self.xs), len(self.ya)), dtype=np.float)
                self.heatmap2 = np.zeros((len(self.xs), len(self.ya)), dtype=np.float)
                self.heatmap3 = np.zeros((len(self.xs), len(self.ya)), dtype=np.float)
                self.heatmap4 = np.zeros((len(self.xs), len(self.ya)), dtype=np.float)

                self.policy1w = np.zeros(len(self.xs), dtype=np.float) # opt policy current q estimate
                self.policy1b = np.zeros(len(self.xs), dtype=np.float)
                self.policy2w = np.zeros(len(self.xs), dtype=np.float)
                self.policy3b = np.zeros(len(self.xs), dtype=np.float)
                self.policy3w = np.zeros(len(self.xs), dtype=np.float)
                if domain == 'lqg':
                    self.opt_policy = np.zeros(len(self.xs), dtype=np.float)

                self.cum_histo = np.zeros(shape=[41, 41], dtype=np.float64)

                # plot
                self.num_ticks = 9
                yticks = np.linspace(0, len(self.ya) - 1, len(self.ya), dtype=int)
                xticks = np.linspace(0, len(self.xs) - 1, self.num_ticks, dtype=int)

                if domain == 'lqg':
                    for i in range(len(self.xs)): # states
                        self.opt_policy[i] = eval_env.get_opt_action(self.xs[i])

            for i in range(len(self.xs)): # states
                for j in range(len(self.ya)): # actions
                    o = np.array([self.xs[i]])
                    ob = np.array(o).reshape((1, 1))
                    a = np.array([self.ya[j]])
                    ac = np.array(a).reshape((1, a.shape[-1]))
                    if self.alg in ['g-oac', 'gs-oac']:
                        qs, upper_bound = trainer.predict(ob, ac, std=True)
                        std = np_ify(qs[1])[0]
                        mean = qs[0]
                    elif self.alg in ['sac', 'oac']:
                        mean = trainer.predict(ob, ac, upper_bound=False)
                        target_mean = trainer.predict(ob, ac, upper_bound=False, target=True)
                    elif self.alg in 'oac-w':
                        mean, std = trainer.predict(ob, ac, both_values=True)
                        upper_bound = trainer.predict(ob, ac, beta_UB=4.66)


                    ob_t = np.array(ob)
                    ob_t = torch_ify(ob_t)
                    ac_t = torch_ify(ac)
                    if hasattr(trainer, 'share_layers') and trainer.share_layers:
                        raise ValueError('Not implemented')
                        qs_t = trainer.q_target(ob_t, ac_t)
                        std_t = np_ify(qs_t[:, 1].unsqueeze(-1))
                    else:
                        if self.alg in ['g-oac', 'gs-oac']:
                            try:
                                titleJG21 = 'prv_std - std'
                                std_t = np_ify(trainer.prv_std(ob_t, ac_t))
                            except:
                                # titleJG21 = 'target_std - std'
                                # std_t = np_ify(trainer.std_target(ob_t, ac_t))
                                titleJG21 = 'Nothing'
                                std_t = np_ify(0)

                    self.heatmap1[i, j] = mean

                    if self.alg == 'oac':
                        Q1 = trainer.qf1(ob_t, ac_t)
                        Q2 = trainer.qf2(ob_t, ac_t)
                        self.heatmap2[i, j] = np.abs(np_ify(Q1 - Q2)) / 2.0
                            
                        # self.heatmap2[i, j] = std_oac
                    if self.alg in ['oac']:
                        self.heatmap3[i, j] = target_mean
                    elif self.alg in ['sac']:
                        self.heatmap3[i, j] = mean
                    elif self.alg in ['g-oac', 'gs-oac']:
                        self.heatmap2[i, j] = std
                        self.heatmap3[i, j] = upper_bound
                        self.heatmap4[i, j] = std_t - std
                    elif self.alg == 'oac-w':
                        self.heatmap2[i, j] = std
                        self.heatmap3[i, j] = upper_bound
                        self.heatmap4[i, j] = 0

                # optimal policy on this state
                self.policy1w[i] = np.argmax(self.heatmap1[i])
                self.policy2w[i] = np.argmax(self.heatmap2[i])
                self.policy3w[i] = np.argmax(self.heatmap3[i])
                if self.alg == 'sac':
                    ac, *_ = trainer.policy.forward(torch_ify(np.reshape(self.xs[i], (1,1))), deterministic=True)
                    self.policy1b[i] = np_ify(ac[0,0])
                    ac, *_ = trainer.policy.forward(torch_ify(np.reshape(self.xs[i], (1,1))), deterministic=False)
                    self.policy3b[i] = np_ify(ac[0,0])
                
                if self.alg in ['g-oac', 'gs-oac']:
                # max_diff?
                    ac, *_ = trainer.target_policy(torch_ify(np.reshape(self.xs[i], (1,1))), deterministic=True)
                    self.policy1b[i] = np_ify(ac[0,0])   
                    ac, *_ = trainer.policy.forward(torch_ify(np.reshape(self.xs[i], (1,1))), deterministic=True)
                    self.policy3b[i] = np_ify(ac[0,0])
                elif self.alg in ['oac', 'oac-w']:
                    from optimistic_exploration import my_o_expl_ac_det

                    hyper_params = {}
                    hyper_params['beta_UB'] = 4.66
                    hyper_params['delta'] = 10
                    hyper_params['share_layers'] = False

                    ac, _ = my_o_expl_ac_det(
                        np.reshape(self.xs[i], (1)), 
                        policy=trainer.policy,
                        trainer=trainer,
                        hyper_params=hyper_params)
                    self.policy3b[i] = ac[0]
                    
                    ac, *_ = trainer.policy.forward(torch_ify(np.reshape(self.xs[i], (1,1))), deterministic=True)
                    self.policy1b[i] = np_ify(ac[0,0])

                    
            #     self.init_save_heatmap()
            #save_heatmap
            if len(state_data):
                histog = np.histogram2d(state_data[:,0], action_data[:,0], bins=[41,41], range=[[-1, 1], [-1, 1]])[0]
                # histog = np.histogram2d(state_data[:,0], action_data[:,0], bins=[41,41], range=[[0, 25], [-1, 1]])[0]
                self.cum_histo += histog
                histogT = histog.T
                cum_histoT = self.cum_histo.T
                #histog_ori = histog
                #cum_histoT = cum_histoT.astype('float64')
                if self.log2:
                    histogT = np.log2(histogT, out=np.zeros_like(histogT), where=(histogT!=0))
                    cum_histoT = np.log2(cum_histoT, out=np.zeros_like(cum_histoT), where=(cum_histoT!=0))

                cum_max = min(np.max(cum_histoT), 1e4)
                his_max = min(np.max(histogT), 2e3)

            fig, ax = plt.subplots(3,2,figsize=(12, 9))

            fig.patch.set_facecolor('#E0E0E0') # background color
            xtl = 10
            nxt = int(len(self.xs)/xtl)+1
            ytl = 10
            nyt = int((len(self.ya))/ytl)+1
            # if self.vmax_std == 0:
            #     vmax_std = np.max(data_heatmap)
            # if vmax_std >= 0:
            #     JG10 = sns.heatmap(data=data_heatmap, xticklabels = xtl, yticklabels = ytl, vmin=0, vmax=vmax_std, cmap="jet_r", ax=ax[1,0])
            # else:

            JG00 = sns.heatmap(data=self.heatmap1.T, xticklabels = xtl, yticklabels = ytl, cmap="jet_r", ax=ax[0,0])
            JG10 = sns.heatmap(data=self.heatmap2.T, xticklabels = xtl, yticklabels = ytl, cmap="jet_r", ax=ax[1,0])
            # if self.algorithm == 'g-oac':
            JG20 = sns.heatmap(data=self.heatmap3.T, xticklabels = xtl, yticklabels = ytl, cmap="jet_r", ax=ax[2,0])
            # JG21 moved for cliff
            if len(state_data):
                JG11 = sns.heatmap(data=cum_histoT, xticklabels = xtl, yticklabels = ytl, vmax=cum_max, cmap="jet_r", ax=ax[1,1])
            # JG11 = sns.heatmap(data=histog_ori, xticklabels = xtl, yticklabels = 8, cmap="jet_r", ax=ax[1,1])

            # fig.suptitle(self._snapshot_dir + 'itr' + str(epoch))

            JG00.set_title('Mean')
            JG00.set_xticklabels(np.linspace(self.ori_min_state,self.ori_max_state,nxt))
            JG00.set_yticklabels(np.linspace(self.ori_min_action,self.ori_max_action,nyt))

            if self.alg in ['sac', 'oac']:
                JG20.set_title('Target Networks')
            if self.alg in ['g-oac', 'oac']:
                JG10.set_title('Std')
                JG20.set_title('Upper Bound')
            JG20.set_xticklabels(np.linspace(self.ori_min_state,self.ori_max_state,nxt))
            JG20.set_yticklabels(np.linspace(self.ori_min_action,self.ori_max_action,nyt))

            JG10.set_xticklabels(np.linspace(self.ori_min_state,self.ori_max_state,nxt))
            JG10.set_yticklabels(np.linspace(self.ori_min_action,self.ori_max_action,nyt))

            if self.alg in ['g-oac', 'gs-oac']: 
                JG21 = sns.heatmap(data=self.heatmap4.T, xticklabels = xtl, yticklabels = ytl, cmap="jet_r", ax=ax[2,1])
                JG21.set_title(titleJG21)
                JG21.set_xticklabels(np.linspace(self.ori_min_state,self.ori_max_state,nxt))
                JG21.set_yticklabels(np.linspace(self.ori_min_action,self.ori_max_action,nyt))
                JG21.set_title('WRONG')
            # elif domain == 'cliff_mono':
            #     # opt_qf = np.load('./data/cliff_mono/opt_qf.npy')
            #     # opt_qf = opt_qf.T
            #     policy_star = np.load('./data/cliff_mono/opt_ps_det.npy')

            #     max_action, min_action = +1, -1
            #     JG21b = sns.lineplot(x=np.linspace(0,61,61),y=policy_star*(opt_qf.shape[0]-1), color='white', ax=ax[2,1])

            #     xtl2 = int((opt_qf.shape[1] - 1) / self.ori_max_state - 0)
            #     d_y = 0.5
            #     ytl2 = int((opt_qf.shape[0] - 1) / ((max_action - min_action) / d_y))
            #     JG21 = sns.heatmap(opt_qf, xticklabels = xtl2, yticklabels = ytl2, cmap="jet_r", ax=ax[2,1])
                #JG21.set_xticklabels(np.arange(0,self.ori_max_state+1))
                # JG21.set_yticklabels(np.round(np.linspace(-10,10,int((max_action - min_action)/d_y+1)))/10)
            if len(state_data):
                JG01 = sns.heatmap(data=histogT, xticklabels = xtl, yticklabels = ytl, vmax=his_max, cmap="jet_r", ax=ax[0,1])
                #JG01.set_title('sampled [log_2]')
                JG01.set_title('sampled')
                # JG01.set_xticklabels(np.linspace(self.ori_min_state,self.ori_max_state,nxt))
                JG01.set_yticklabels(np.linspace(self.ori_min_action,self.ori_max_action,nyt))

                #JG11.set_title('Cum sampled [log_2] WARNING: of plotted heatmaps')
                JG11.set_title('Cum sampled')
                #JG11.set_title('sampled')
                #JG11.set_xticklabels(np.linspace(self.ori_min_state,self.ori_max_state,nxt))
                JG11.set_yticklabels(np.linspace(self.ori_min_action,self.ori_max_action,nyt))

            JG00b = sns.lineplot(x=np.linspace(0,len(self.xs),len(self.xs)),y=self.policy1w, color='white', ax=ax[0,0])
            if domain == 'lqg':
                JG00d = sns.lineplot(x=np.linspace(0,len(self.xs),len(self.xs)),y=((self.opt_policy+1)*(len(self.ya)-1)/2), color='gray', ax=ax[0,0])

            if self.alg in ['g-oac', 'oac', 'sac', 'gs-oac', 'oac-w']:
                JG00c = sns.lineplot(x=np.linspace(0,len(self.xs),len(self.xs)),y=((self.policy1b+1)*(len(self.ya)-1)/2), color='black', ax=ax[0,0])
                
            JG20b = sns.lineplot(x=np.linspace(0,len(self.xs),len(self.xs)),y=self.policy3w, color='white', ax=ax[2,0])

            JG10b = sns.lineplot(x=np.linspace(0,len(self.xs),len(self.xs)),y=self.policy2w, color='white', ax=ax[1,0])
            JG20c = sns.lineplot(x=np.linspace(0,len(self.xs),len(self.xs)),y=((self.policy3b+1)*(len(self.ya)-1)/2), color='black', ax=ax[2,0])

        fig.suptitle(self._snapshot_dir + 'itr' + str(epoch))
        output_name = os.path.join(self.folder, 'hm_' + str(epoch) + '.png')
        fig.savefig(output_name)
        plt.close()

    # def init_save_heatmap():


logger = Logger()
