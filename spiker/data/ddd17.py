"""DDD17 Data Utility.

Migrate DDD17 data reading utilities from
old utils to here.

It uses legacy codes from ddd17-utils
that read and write data to correct
form.

Source of the ddd17-utils:
https://code.ini.uzh.ch/jbinas/ddd17-utils

Author: Yuhuang Hu
Email : duguyue100@gmail.com
"""
from __future__ import print_function

import time
import struct
import Queue
import multiprocess as mp

import h5py
import numpy as np

from spiker import data

# data fields
EXPORT_DATA_VI = {
        'steering_wheel_angle',
        'brake_pedal_status',
        'accelerator_pedal_position',
        'engine_speed',
        'vehicle_speed',
        'windshield_wiper_status',
        'headlamp_status',
        'transmission_gear_position',
        'torque_at_transmission',
        'fuel_level',
        'high_beam_status',
        'ignition_status',
        # 'lateral_acceleration',
        'latitude',
        'longitude',
        # 'longitudinal_acceleration',
        'odometer',
        'parking_brake_status',
        # 'fine_odometer_since_restart',
        'fuel_consumed_since_restart',
    }

EXPORT_DATA_DAVIS = {
        'dvs_frame',
        'aps_frame',
    }

EXPORT_DATA = EXPORT_DATA_VI.union(EXPORT_DATA_DAVIS)

# datasets specs
SIZE_INC = 2048
CHUNK_SIZE = 128

# caer related
HEADER_FIELDS = (
        'etype',
        'esource',
        'esize',
        'eoffset',
        'eoverflow',
        'ecapacity',
        'enumber',
        'evalid',
        )

EVENT_TYPES = {
        'special_event': 0,
        'polarity_event': 1,
        'frame_event': 2,
        'imu6_event': 3,
        'imu9_event': 4,
        }

etype_by_id = {v: k for k, v in EVENT_TYPES.iteritems()}


def filter_frame(d):
    '''
    receives 16 bit frame,
    needs to return unsigned 8 bit img
    '''
    # add custom filters here...
    # d['data'] = my_filter(d['data'])
    frame8 = (d['data'] / 256).astype(np.uint8)
    return frame8


def raster_evts(data, data_shape=data.DAVIS346_SHAPE):
    _histrange = [(0, v) for v in data_shape]
    pol_on = data[:, 3] == 1
    pol_off = np.logical_not(pol_on)
    img_on, _, _ = np.histogram2d(
            data[pol_on, 2], data[pol_on, 1],
            bins=data_shape, range=_histrange)
    img_off, _, _ = np.histogram2d(
            data[pol_off, 2], data[pol_off, 1],
            bins=data_shape, range=_histrange)
    return (img_on - img_off).astype(np.int16)


def _flush_q(q):
    ''' flush queue '''
    while True:
        try:
            q.get(timeout=1e-3)
        except Queue.Empty:
            if q.empty():
                break


def unpack_events(p):
    '''
    Extract events from binary data,
    returns list of event tuples.
    '''
    if not p['etype'] == 'polarity_event':
        return False
    p_arr = np.fromstring(p['dvs_data'], dtype=np.uint32)
    p_arr = p_arr.reshape((p['ecapacity'], p['esize'] / 4))
    data, ts = p_arr[:, 0], p_arr[:, 1]
    pol = data >> 1 & 0b1
    y = data >> 2 & 0b111111111111111
    x = data >> 17
    return ts[0] * 1e-6, np.array([ts, x, y, pol]).T


def unpack_frame(p, data_shape=data.DAVIS346_SHAPE):
    '''
    Extract image from binary data, returns timestamp and 2d np.array.
    '''
    if not p['etype'] == 'frame_event':
        return False
    img_head = np.fromstring(p['dvs_data'][:36], dtype=np.uint32)
    img_data = np.fromstring(p['dvs_data'][36:], dtype=np.uint16)
    return img_head[2] * 1e-6, img_data.reshape(data_shape)


def unpack_special(p):
    '''
    Extract special event data (only return type id).
    '''
    if not p['etype'] == 'special_event':
        return False
    p_arr = np.fromstring(p['dvs_data'], dtype=np.uint32)
    p_arr = p_arr.reshape((p['ecapacity'], p['esize'] / 4))
    data, ts = p_arr[:, 0], p_arr[:, 1]
    typeid = data & 254
    # valid = data & 1
    # opt = data >> 8
    return ts[0] * 1e-6, typeid


def unpack_header(header_raw):
    '''
    Extract header info from binary data,
    returns dict object.
    '''
    vals = struct.unpack('hhiiiiii', header_raw)
    obj = dict(zip(HEADER_FIELDS, vals))
    obj['etype'] = etype_by_id.get(obj['etype'], obj['etype'])
    return obj


# caer utils related
unpack_func = {
        'polarity_event': unpack_events,
        'frame_event': unpack_frame,
        'special_event': unpack_special,
        }


def unpack_data(d):
    '''
    Unpack data for given caer packet,
    return False if event type does not exist.
    '''
    _get_data = unpack_func.get(d['etype'])
    if _get_data:
        d['timestamp'], d['data'] = _get_data(d)
        return d
    return False


def caer_event_from_row(row):
    '''
    Takes binary dvs data as input,
    returns unpacked event data or False if event type does not exist.
    '''
    sys_ts, head, body = (v.tobytes() for v in row)
    if not sys_ts:
        # rows with 0 timestamp do not contain any data
        return 0, False
    d = unpack_header(head)
    d['dvs_data'] = body
    return int(sys_ts) * 1e-6, unpack_data(d)


class HDF5Stream(mp.Process):
    def __init__(self, filename, tables, bufsize=64):
        super(HDF5Stream, self).__init__()
        self.f = h5py.File(filename, 'r')
        self.tables = tables
        self.q = {k: mp.Queue(bufsize) for k in self.tables}
        self.run_search = mp.Event()
        self.exit = mp.Event()
        self.done = mp.Event()
        self.skip_to = mp.Value('L', 0)
        self._init_count()
        self._init_time()
        self.daemon = True
        self.start()

    def run(self):
        while self.blocks_rem and not self.exit.is_set():
            blocks_read = 0
            for k in self.blocks_rem.keys():
                if self.q[k].full():
                    time.sleep(1e-6)
                    continue
                i = self.block_offset[k]
                self.q[k].put(
                    self.f[k]['data'][i*CHUNK_SIZE:(i+1)*CHUNK_SIZE])
                self.block_offset[k] += 1
                if self.blocks_rem[k].value:
                    self.blocks_rem[k].value -= 1
                else:
                    self.blocks_rem.pop(k)
                blocks_read += 1
            if not blocks_read:
                time.sleep(1e-6)
            if self.run_search.is_set():
                self._search()
        self.f.close()
        print('closed input file')
        while not self.exit.is_set():
            time.sleep(1e-3)
        # print('[DEBUG] flushing stream queues')
        for k in self.q:
            # print('[DEBUG] flushing', k)
            _flush_q(self.q[k])
            self.q[k].close()
            self.q[k].join_thread()
        # print('[DEBUG] flushed all stream queues')
        self.done.set()
        print('stream done')

    def get(self, k, block=True, timeout=None):
        return self.q[k].get(block, timeout)

    def _init_count(self, offset={}):
        self.block_offset = {k: offset.get(k, 0) / CHUNK_SIZE
                             for k in self.tables}
        self.size = {k: len(self.f[k]['data']) - v * CHUNK_SIZE
                     for k, v in self.block_offset.items()}
        self.blocks = {k: v / CHUNK_SIZE for k, v in self.size.items()}
        self.blocks_rem = {k: mp.Value('L', v)
                           for k, v in self.blocks.items() if v}

    def _init_time(self):
        self.ts_start = {}
        self.ts_stop = {}
        self.ind_stop = {}
        for k in self.tables:
            ts_start = self.f[k]['timestamp'][self.block_offset[k]*CHUNK_SIZE]
            self.ts_start[k] = mp.Value('L', ts_start)
            b = self.block_offset[k] + self.blocks_rem[k].value - 1
            while \
                b > self.block_offset[k] and \
                    self.f[k]['timestamp'][b*CHUNK_SIZE] == 0:
                    b -= 1
            print(k, 'final block:', b)
            self.ts_stop[k] = mp.Value(
                'L', self.f[k]['timestamp'][(b + 1) * CHUNK_SIZE - 1])
            self.ind_stop[k] = b

    def init_search(self, t):
        ''' start streaming from given time point '''
        if self.run_search.is_set():
            return
        self.skip_to.value = np.uint64(t)
        self.run_search.set()

    def _search(self):
        t = self.skip_to.value
        offset = {k: self._bsearch_by_timestamp(k, t) for k in self.tables}
        for k in self.tables:
            _flush_q(self.q[k])
        self._init_count(offset)
        # self._init_time()
        self.run_search.clear()

    def _bsearch_by_timestamp(self, k, t):
        '''performs binary search on timestamp, returns closest block index.'''
        l, r = 0, self.ind_stop[k]
        print('searching', k, t)
        while True:
            if r - l < 2:
                print('selecting block', l)
                return l * CHUNK_SIZE
            if self.f[k]['timestamp'][(l + (r - l) / 2) * CHUNK_SIZE] > t:
                r = l + (r - l) / 2
            else:
                l += (r - l) / 2


class MergedStream(mp.Process):
    ''' Unpacks and merges data from HDF5 stream '''
    def __init__(self, fbuf, bufsize=256):
        super(MergedStream, self).__init__()
        self.fbuf = fbuf
        self.ts_start = self.fbuf.ts_start
        self.ts_stop = self.fbuf.ts_stop
        self.q = mp.Queue(bufsize)
        self.run_search = mp.Event()
        self.skip_to = mp.Value('L', 0)
        self._init_state()
        self.done = mp.Event()
        self.fetched_all = mp.Event()
        self.exit = mp.Event()
        self.daemon = True
        self.start()

    def run(self):
        while self.blocks_rem and not self.exit.is_set():
            # find next event
            if self.q.full():
                time.sleep(1e-4)
                continue
            next_k = min(self.current_ts, key=self.current_ts.get)
            self.q.put((self.current_ts[next_k], self.current_dat[next_k]))
            self._inc_current(next_k)
            # get new blocks if necessary
            for k in {k for k in self.blocks_rem if self.i[k] == CHUNK_SIZE}:
                self.current_blk[k] = self.fbuf.get(k)
                self.i[k] = 0
                if self.blocks_rem[k]:
                    self.blocks_rem[k] -= 1
                else:
                    self.blocks_rem.pop(k)
                    self.current_ts.pop(k)
            if self.run_search.is_set():
                self._search()
        self.fetched_all.set()
        self.fbuf.exit.set()
        while not self.fbuf.done.is_set():
            time.sleep(1)
            # print('[DEBUG] waiting for stream process')
        while not self.exit.is_set():
            time.sleep(1)
            # print('[DEBUG] waiting for merger process')
        _flush_q(self.q)
        # print('[DEBUG] flushed merger q ->', self.q.qsize())
        self.q.close()
        self.q.join_thread()
        # print('[DEBUG] joined merger q')
        self.done.set()

    def close(self):
        self.exit.set()

    def _init_state(self):
        keys = self.fbuf.blocks_rem.keys()
        self.blocks_rem = {k: self.fbuf.blocks_rem[k].value for k in keys}
        self.current_blk = {k: self.fbuf.get(k) for k in keys}
        self.i = {k: 0 for k in keys}
        self.current_dat = {}
        self.current_ts = {}
        for k in keys:
            self._inc_current(k)

    def _inc_current(self, k):
        ''' get next event of given type and increment row pointer '''
        row = self.current_blk[k][self.i[k]]
        if k == 'dvs':
            ts, d = caer_event_from_row(row)
        else:
            # vi event
            ts = row[0] * 1e-6
            d = {'etype': k, 'timestamp': row[0], 'data': row[1]}
        if not ts and k in self.current_ts:
            self.current_ts.pop(k)
            self.blocks_rem.pop(k)
            return False
        self.current_ts[k], self.current_dat[k] = ts, d
        self.i[k] += 1

    def get(self, block=False):
        return self.q.get(block)

    @property
    def has_data(self):
        return not (self.fetched_all.is_set() and self.q.empty())

    @property
    def tmin(self):
        return self.ts_start['dvs'].value

    @property
    def tmax(self):
        return self.ts_stop['dvs'].value

    def search(self, t, block=True):
        if self.run_search.is_set():
            return
        self.skip_to.value = np.uint64(t)
        self.run_search.set()

    def _search(self):
        self.fbuf.init_search(self.skip_to.value)
        while self.fbuf.run_search.is_set():
            time.sleep(1e-6)
        _flush_q(self.q)
        self._init_state()
        self.q.put((0, {'etype': 'timestamp_reset'}))
        self.run_search.clear()


class HDF5(mp.Process):
    '''
    Creates a hdf5 file with datasets of specified types.
    Provides an append method.
    '''
    def __init__(self, filename='rec.hdf5', tables={},
                 bufsize=2048*16, chunksize=0, mode='w-', compression=None):
        super(HDF5, self).__init__()
        self.compression = compression
        self.fname = filename
        self.datasets = {}
        self.outbuffers = {}
        self.ndims = {}
        self.chunk_size = chunksize or CHUNK_SIZE
        self.tables = tables
        self.q = mp.Queue(bufsize)
        self.maxsize = self.q._maxsize
        self.exit = mp.Event()
        self.fmode = mode
        # self.daemon = True
        self.start()

    def init_ds(self):
        self.f = h5py.File(self.fname, self.fmode)
        self.create_datasets(self.tables, compression=self.compression)
        self.ptrs = {k: 0 for k in self.datasets}
        self.size = {k: SIZE_INC for k in self.datasets}

    def run(self):
        self.init_ds()
        f = file('datasets_ioerrors.txt', 'a')
        while not self.exit.is_set() or not self.q.empty():
            try:
                res = self.q.get(False, 1e-3)
                self._save(res)
            except Queue.Empty:
                pass
            except IOError:
                print('IOError, continuing')
                f.write(str(res))
                pass
            except KeyboardInterrupt:
                # print('datasets.run got interrupt')
                self.exit.set()
        f.close()
        self.close()

    def create_datasets(self, tables, compression=None):
        for tname, ttype in tables.iteritems():
            tname_split = tname.split('/')
            if len(tname_split) > 1:
                grpname, subtname = tname_split
                if grpname not in self.f:
                    rnode = self.f.create_group(grpname)
                else:
                    rnode = self.f[grpname]
            else:
                subtname = tname
                rnode = self.f
            tname = tname.replace('/', '_')
            extra_shape = ()
            self.ndims[tname] = 1
            if isinstance(ttype, (tuple, list)):
                extra_shape = ttype[1]
                ttype = ttype[0]
                self.ndims[tname] += 1
            print(tname)
            self.datasets[tname] = rnode.create_dataset(
                subtname,
                (SIZE_INC,) + extra_shape,
                maxshape=(None,) + extra_shape,
                chunks=(self.chunk_size,) + extra_shape,
                dtype=ttype,
                compression=compression)
            self.outbuffers[tname] = []

    def save(self, data):
        try:
            self.q.put_nowait(data)
        except Queue.Full:
            raise Queue.Full('dataset buffer overflow')

    def _save(self, data):
        for col, val in data.iteritems():
            self.outbuffers[col].append(val)
            if len(self.outbuffers[col]) == self.chunk_size:
                self[col][self.ptrs[col]:self.ptrs[col] + self.chunk_size] = \
                        self._get_outbuf(col)
                self.outbuffers[col] = []
                self.ptrs[col] += self.chunk_size
            if self.ptrs[col] == self.size[col]:
                self.size[col] += SIZE_INC
                self[col].resize(self.size[col], axis=0)

    def _get_outbuf(self, col):
        if self.ndims[col] > 1:
            return np.array(self.outbuffers[col])
        else:
            return self.outbuffers[col]

    def __getitem__(self, key):
        return self.datasets[key]

    def close(self):
        self.exit.set()
        self.f.flush()
        self.f.close()
        self.q.close()
        self.q.join_thread()
        print('\nclosed output file')
