class DataInput:
    def __init__(self, filename, conffile=None, batch_size=10, n_repeats=1000, one_hot_size=100):
        self.filename = filename
        self.one_hot_size = one_hot_size
        self.f = open(self.filename)
        self.conf = self._read_conf(conffile)
        self.memory = None
        
    def get_dataset(self):
        end_ping_num = 0 
        all_ping_num = 0 
        for n, line in enumerate(self.f):
            parts = line.strip().split('\t')
            if len(parts) != 4:
                continue
            fea = parts[0].split(' ')
            fea_n = parts[3].split(' ')
            if not parts[0][0].isdigit() and self.conf:
                fea = list(self.conf.get(i, '0') for i in fea)
                fea_n = list(self.conf.get(i, '0') for i in fea_n)
            state = list(map(int, fea))
            action = float(parts[1])
            reward = float(parts[2])
            all_ping_num += 1
            #if parts[3][0] == '0':
            if self.conf['LSSNum\1unknow'] in fea_n:
                next_state = None
                end_ping_num += 1
            else:
                next_state = list(map(int, fea_n))
            #print (state, action, reward, next_state)    
            yield self._one_hot(state), [action], reward, self._one_hot(next_state), parts[0]
        print ('end_ping_num=', end_ping_num)

    def _read_conf(self, conffile):
        if conffile is None:
            return {}
        conf = {}
        for line in open(conffile):
            parts = line.split('\t')
            conf[parts[0]] = int(parts[1])
        return conf

    def _one_hot(self, x): 
        if not x:
            return x
        return [1.0 if i in x else 0.0 for i in range(self.one_hot_size) ]


if __name__ == "__main__":
    data = DataInput('../data/corpus_test2', conffile='../hd_src/conf', one_hot_size=60)
    ds = data.get_dataset()
    for i in range(20):
        print(next(ds))

