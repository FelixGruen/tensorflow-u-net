# encoding=utf-8

import errno
import multiprocessing as mp
import os
import os.path
import pickle
import random
import threading

import numpy as np

pipeline = None


def _generator_func(source, transform):
    for datapoint in source:
        yield transform(datapoint)


def apply(source, transform):
    """
    Turns a normal function into a generator and registers it with the current pipeline.

    The function must accept whatever the given source yields and return a transformed datapoint based on its input.

    Parameters
    ----------
    source : iterable
        An iterable over a number of datapoints that transform accepts.
    transform : function
        A function that transforms each datapoint that source yields.

    Returns
    -------
    gen : generator
        A generator that yields the output of transform.
    """
    gen = _generator_func(source, transform)
    sign_up(gen)
    return gen


def current_pipeline():
    """
    Returns the current pipeline.
    """
    global pipeline
    return pipeline


def sign_up(generator):
    """
    Signs up a generator with the current pipeline.
    """
    global pipeline
    if pipeline is not None:
        pipeline.sign_up(generator)


class Pipeline(object):
    """
    A pipeline is not strictly needed, but offers a number of useful services. It should always be used in a with-block.
    """

    def __init__(self, storage_name="", initialization_generator=None):

        self.parent = None
        self.my_id = ""
        self.to_close = []
        self.to_initialize = []
        self.registered = []
        self.parameter_store = {}
        self.storage_name = storage_name
        self.folder_name = "temp"
        self.initialization_generator = initialization_generator
        self.current_generator = None

    def __enter__(self):

        global pipeline

        if pipeline is not None:
            self.parent = pipeline
            self.parent.register_close(self)
        pipeline = self

        self._load_parameter_store()

        return self

    def __exit__(self, exc_type, exc_value, traceback):

        self._initialize_all()
        self._save_parameter_store()

        global pipeline
        pipeline = self.parent

    def __iter__(self):
        return self

    def next(self):
        if self.current_generator is not None:
            return self.current_generator.next()
        raise StopIteration

    def _load_parameter_store(self):
        if self.storage_name != "":
            if self.parent is not None:
                self.my_id = self.parent.register(self.storage_name)
                if self.parent.has_value(self.my_id):
                    self.parameter_store = self.parent.load_value(self.my_id)
            elif os.path.isdir(self.folder_name):
                store = os.path.join(self.folder_name, self.storage_name)
                if os.path.isfile(store):
                    print 'Found dictionary file "{}". Loading dictionary...'.format(self.storage_name)
                    with open(store, 'rb') as infile:
                        self.parameter_store = pickle.load(infile)

    def _save_parameter_store(self):
        if self.storage_name != "":
            if self.parent is not None:
                if self.my_id == "":
                    self.my_id = self.parent.register(self.storage_name)
                self.parent.store_value(self.my_id, self.parameter_store)
            else:
                try:
                    os.makedirs(self.folder_name)
                except OSError as exception:
                    if exception.errno != errno.EEXIST:
                        raise
                store = os.path.join(self.folder_name, self.storage_name)
                with open(store, 'wb') as outfile:
                    pickle.dump(self.parameter_store, outfile)

    def _initialize_all(self):
        if self.initialization_generator is not None and len(self.to_initialize) > 0:

            for datapoint in self.initialization_generator:

                # Standard case, copy the arrays and the dictionary
                try:
                    if len(datapoint) >= 0:  # only test if type indexable

                        inputs = datapoint[0]
                        rest = datapoint[1:]

                        if len(inputs) >= 0:  # test if type indexable

                            for init_func in self.to_initialize:

                                cpy = [[np.copy(ary) if type(ary) is np.ndarray else ary for ary in inputs]]

                                for elem in rest:
                                    try:
                                        elem = elem.copy()
                                    except AttributeError:
                                        pass
                                    cpy.append(elem)

                                init_func(cpy)

                # Other cases, makes code reusable
                except TypeError, IndexError:
                    for init_func in self.to_initialize:
                        init_func(datapoint)

    def register_close(self, obj):
        """
        Register an object that has to be closed after use with the pipeline, so that close is called on this object,
        when the pipeline is closed.
        """
        self.to_close.append(obj)

    def close(self):
        """
        Close the pipeline and all objects that were registered to be closed.
        """
        for obj in self.to_close:
            obj.close()
        self.to_close = []

    def store_value(self, my_id, value):
        """
        Allows registered functions to store a value or object. This can be used to carry over values that were set during
        initialization into otherwise side-effect free transformation functions.
        """
        if my_id in self.registered:
            self.parameter_store[my_id] = value
        else:
            raise ValueError("The id {} is unknown. Have you used Pipeline.register(name) to obtain an id? Have you used the returned id for this call to Pipeline.store_value(id, value)?".format(my_id))

    def has_value(self, my_id):
        """
        Check if a value was stored for the given id.
        """
        return my_id in self.parameter_store

    def load_value(self, my_id):
        """
        Return a stored value for the given id or None if no value was stored.
        """
        if my_id in self.parameter_store:
            return self.parameter_store[my_id]
        return None

    def register(self, name):
        """
        Register a function with the pipeline. Returns an id that can be used to store values.
        """

        i = 1
        while name + '/' + str(i) in self.registered:
            i += 1

        _id = name + '/' + str(i)
        self.registered.append(_id)

        return _id

    def offers_initialization(self):
        """
        Check whether the pipeline was supplied with an iterable that supplies datapoints for initialization.
        """
        return self.initialization_generator is not None

    def initialize(self, init_func):
        """
        Register an initialization function. The function will be supplied with datapoints from the initialization iterable of
        the pipeline once the pipeline definition is completed and the with-block is exited.
        """
        self.to_initialize.append(init_func)

    def sign_up(self, generator):
        """
        Sign up a generator as the current generator. This is used, so the pipeline can itself be used as an iterable.
        """
        self.current_generator = generator


def initialize(name, init_func, obj, initialization_generator=None):

    global pipeline
    pipe = pipeline
    my_id = None

    # If possible load object from pipeline
    if pipe is not None:
        my_id = pipe.register(name)
        if pipe.has_value(my_id):
            return pipe.load_value(my_id), my_id

    # Else initialize object
    # Here the initialization_generator takes precedence over pipeline initialization
    if initialization_generator is not None:
        for datapoint in initialization_generator:
            init_func(datapoint)
    elif pipe is not None and pipe.offers_initialization():
        pipe.initialize(init_func)
    else:
        obj = None

    # If possible store the obj value
    if pipe is not None and obj is not None:
        pipe.store_value(my_id, obj)

    return obj, my_id


class ProcessManager(object):

    def __init__(self, source, processes, buffer_size):

        global pipeline
        self.pipe = pipeline
        self.started = False

        self.source = source
        self.processes = processes
        self.buffer_size = buffer_size

        self.queue = None
        self.process_list = []
        self.counter = None
        self.mutex = None

    def start(self):

        if not self.started:

            self.queue = mp.Queue(self.buffer_size)
            self.process_list = [mp.Process(target=self._execute, args=(self.source, self.queue)) for _ in xrange(self.processes)]
            self.counter = mp.Value('i', False)
            self.counter.value = 0
            self.mutex = mp.Lock()

            self._start_processes()
            if self.pipe is not None:
                self.pipe.register_close(self)

            self.started = True

    def _start_processes(self):
        for process in self.process_list:
            if not process.is_alive():
                process.start()
                with self.mutex:
                    self.counter.value += 1

    def close(self):
        with self.mutex:
            self.counter.value = 0
            self.queue.close()
            for process in self.process_list:
                if process.is_alive():
                    process.terminate()
                    process.join()

    def __iter__(self):
        return self

    def next(self):

        if not self.started:
            self.start()

        with self.mutex:
            if self.counter.value == 0:
                raise StopIteration

            datapoint = self.queue.get()

            if datapoint is None:
                self.counter.value -= 1

        return datapoint if datapoint is not None else self.next()

    def _execute(self, source, queue):

        # reseed random number generators to make sure every process generates different random numbers
        random.seed()
        np.random.seed()

        for datapoint in source:
            queue.put(datapoint)

        queue.put(None)


def run_on(source, processes=1, buffer_size=16):
    """
    Runs the pipeline from the last call to run_on up to source in parallel on the given number of processes.

    Parameters
    ----------
    source : iterable
        An iterable over a number of datapoints.
    ṕrocesses : int
        The number of processes to run in parallel.
    buffer_size: int
        The size of the buffer, where the output of the processes is stored.

    Returns
    -------
    gen : generator
        A generator that yields the output of the given source.
    """
    try:
        from IPython import get_ipython
        if get_ipython() is not None:
            cfg = get_ipython().config
            if "IPKernelApp" in cfg.keys():
                warn("Multiprocessing might crash in iPython Notebooks")
    except ImportError:
        pass

    return ProcessManager(source, processes, buffer_size)
