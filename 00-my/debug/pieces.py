            for _ in range(self.num_train_loops_per_epoch):

                debug = True
                if debug:
                    with open('data/debug/straight.txt','a+') as f:
                        f.writelines('------------ \n')
                        f.writelines('before epoch: %d\n' % epoch)
                        f.writelines('first-weights:')
                        f.write(str(self.trainer.policy.fcs[0].weight.data) + '\n')
                        f.writelines('first-bias:')
                        f.write(str(self.trainer.policy.fcs[0].bias.data) + '\n')
                        f.writelines('last-weights:')
                        f.write(str(self.trainer.policy.last_fc.weight.data) + '\n')
                        f.writelines('last-bias:')
                        f.write(str(self.trainer.policy.last_fc.bias.data) + '\n')