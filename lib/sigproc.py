"""
Simple Signal Processing Module

FEATURES
- Load/save signal in wav format
- Manipulate signals in both time and frequency domains
- Visualize signal in both time and frequency domains

AUTHOR

Chaiporn (Art) Jaikaeo
Intelligent Wireless Networking Group (IWING) -- http://iwing.cpe.ku.ac.th
Department of Computer Engineering
Kasetsart University
chaiporn.j@ku.ac.th

Last modified: 2017-08-28
"""

import numpy as np
from scipy.fftpack import fft, ifft
from scipy.io import wavfile
from pandas import DataFrame

DEFAULT_PLOT_SETTINGS = {
    "time_fig_options" : {
        "height" : 300,
        "x_axis_label" : "Time (ms)",
        "y_axis_label" : "Amplitude",
        "tools" : "pan,wheel_zoom,box_zoom,reset,save,crosshair",
    },
    "time_line_options" : {
        "color" : "blue",
        "line_width" : 2,
    },

    "freq_fig_options" : {
        "height" : 300,
        "x_axis_label" : "Frequency (Hz)",
        "y_axis_label" : "Magnitude",
        "y_range" : None,
        "tools" : "pan,wheel_zoom,box_zoom,reset,save,crosshair",
    },
    "freq_line_options" : {
        "color" : "green",
        "line_width" : 2,
    },

    "phase_fig_options" : {
        "height" : 300,
        "x_axis_label" : "Frequency (Hz)",
        "y_axis_label" : "Phase (degrees)",
        "y_range" : (-180,180),
        "tools" : "pan,wheel_zoom,box_zoom,reset,save,crosshair",
    },
    "phase_line_options" : {
        "color" : "orange",
        "line_width" : 1,
    },
    "phase_point_options" : {
        "line_color" : "red",
        "fill_color" : "red",
        "size" : 4,
    },

    "raw_fig_options" : {
        "height" : 300,
        "x_axis_label" : "Index",
        "y_axis_label" : "Raw Magnitude",
        "tools" : "pan,wheel_zoom,box_zoom,reset,save,crosshair",
    },
    "raw_line_options" : {
        "color" : "red",
        "line_width" : 1,
    },
}

NOTEBOOK_SESSION = False

###########################################
def start_notebook():
    """
    Initialize and start a sigproc session in Jupyter Notebook
    """
    global NOTEBOOK_SESSION
    global bkp
    try:
        import bokeh.plotting
        bkp = bokeh.plotting
        bkp.output_notebook()
        print("Notebook session ready.")
        NOTEBOOK_SESSION = True
        return bkp
    except:
        pass

###########################################
class Signal(object):
    """
    Define signal objects containing signal duration, sampling_rate, and
    amplitudes of all the signal elements in frequency domain.
    """

    def __init__(self, duration=1.0, sampling_rate=22050, func=None,
            wav_file=None, channel="left"):
        """
        Initialize a signal object with the specified duration (in seconds)
        and sampling rate (in Hz).  If func is provided, signal
        data will be initialized to values of this function for the entire
        duration.  If wav_file is provided, the signal will be created from
        the samples in the specified wave file and channel ("left" or "right")
        instead, and all other parameters will be ignored.
        """
        if wav_file is not None:
            self.read_wav(wav_file, channel=channel)
        else:
            self._sampling_rate = sampling_rate
            self.freqs = np.arange(int(duration*sampling_rate), dtype=complex)
            self.freqs[:] = 0j
            if func is not None:
                self.read_time_function(func)


    def read_wav(self, wav_file, channel="left"):
        """
        Read data from the specified wave file into the signal object.  For a
        stereo stream, only one channel ("left" or "right") can be extracted.
        """
        rate,data = wavfile.read(wav_file)
        n = data.shape[0]
        self._sampling_rate = rate

        if data.dtype == np.dtype("int16"):
            normalizer = 32768.0
        elif data.dtype == np.dtype("int8"):
            normalizer = 256.0
        else:
            raise(Exception("Unsupport data type"))

        if len(data.shape) == 2: # stereo stream
            if channel == "left":
                data = data[:,0]
            elif channel == "right":
                data = data[:,1]
            else:
                raise(Exception("Invalid channel choice '%s'" % channel))

        self.freqs = fft(data/normalizer)


    def write_wav(self, wav_file, start_time=None, end_time=None, normalize=True):
        """
        Write signal data into the specified wave file using int16 data type.
        If start_time or end_time are specified in seconds, only samples
        falling in the time interval [start_time, end_time) are written.

        If normalize is True and the signal level exceeds the valid range,
        the signal level will be scaled down to fit the range instead of being
        clipped.
        """
        x,y = self.time_domain.values.T
        if start_time is None:
            start_time = 0.0
        if end_time is None:
            end_time = x[-1]

        start_index = int(start_time*self._sampling_rate)
        end_index   = int(end_time*self._sampling_rate)

        data = y[start_index:end_index]

        if normalize:
            if data.max() > 1.0:
                data = data/data.max()
            if data.min() < -1.0:
                data = -data/data.min()
        wavfile.write(
                wav_file, 
                self._sampling_rate, 
                (data*32768).astype(np.dtype("int16"))
        )


    @property
    def sampling_rate(self):
        """
        Return the sampling rate associated with the signal in Hz.
        """
        return self._sampling_rate


    @property
    def duration(self):
        """
        Return the duration of the signal in seconds.
        """
        return float(len(self.freqs))/self._sampling_rate


    def add_pulse(self,value,time,duration):
        """
        Add a pulse of level value to the signal at the specified time for the
        specified duration
        """
        ts = self.time_domain
        ts.loc[(ts.t >= time) & (ts.t < time+duration),"a"] += value
        self.freqs = fft(ts.a)


    def amplify(self, factor):
        """
        Amplify the signal by the specified factor.
        """
        self.freqs *= factor
        return self


    def clear(self):
        """
        Set amplitudes of all frequencies to zero
        """
        self.freqs[:] = 0j


    def filter(self, range_or_func):
        """
        Apply the specified filter to the signal over the frequency domain.

        If the paramter range_or_func is a number or a pair of values, the
        signal will be filtered with all the frequencies in the specified
        range.
        
        If the parameter is a function, each frequency's amplitude is
        multiplied by the value returned from the function.

        """
        if isinstance(range_or_func,(int,float)):
            filter_func = lambda f: 1.0 if f<=range_or_func else 0.0
        elif hasattr(range_or_func,"__iter__"):  # a range is specified
            fmin,fmax = range_or_func
            filter_func = lambda f: 1.0 if f>=fmin and f<=fmax else 0.0
        elif hasattr(range_or_func,"__call__"): # a function is specified
            filter_func = range_or_func
        else:
            raise Exception("The parameter range_or_func is not a range nor a function")

        n = len(self.freqs)
        num_freqs = int(np.ceil((n-1)/2))
        for i in range(num_freqs):
            # convert index to corresponding frequency value
            f = float(i)/n*self._sampling_rate
            self.freqs[i] *= filter_func(f)

            # also apply filter to its negative frequency counterpart
            if i > 0:
                self.freqs[-i] *= filter_func(f)
        return self


    def filter_time(self, range_or_func):
        """
        Apply the specified filter to the signal over the time domain.

        If the paramter range_or_func is a number or a pair of values, the
        signal will be filtered with all the time points in the specified
        range.
        
        If the parameter is a function, each time's amplitude is
        multiplied by the value returned from the function.

        """
        if isinstance(range_or_func,int) or isinstance(range_or_func,float):
            filter_func = lambda t: (t<=range_or_func)
        elif hasattr(range_or_func,"__iter__"):  # a range is specified
            tmin,tmax = range_or_func
            filter_func = lambda t: (t>=fmin and t<=fmax)
        elif hasattr(range_or_func,"__call__"): # a function is specified
            filter_func = range_or_func
        else:
            raise Exception("The parameter range_or_func is not a range nor a function")

        ts = self.time_domain
        ts.a = ts.a*filter_func(ts.t)

        self.freqs = fft(ts.a)


    def __mul__(self,s):
        """
        Multiply the signal by another signal, s, time-wise, and return a new
        signal.  The signal to multiply must have exactly the same sampling
        rate and duration.
        
        If s is a scalar, the new signal is the current signal
        amplified by the factor s. 
        """
        if isinstance(s,(int,float)):
            newsig = self.copy()
            newsig.amplify(s)
            return newsig
        elif isinstance(s,Signal):
            # make sure both signals have the same sampling rate and duration
            if self.sampling_rate != s.sampling_rate:
                raise Exception("Both signals must have the same sampling rate")
            if self.duration != s.duration:
                raise Exception("Both signals must have the same duration")
            ts1 = self.time_domain
            ts2 = s.time_domain
            newsig = Signal()
            newsig.freqs = fft(ts1.a*ts2.a)
            return newsig
        else:
            raise Exception("Unsupported object type for signal multiplication")

    __rmul__ = __mul__


    def set_freq(self, freq, amplitude, phase=0):
        """
        Set a particular frequency component with the specified amplitude and
        phase-shift relative to cosine (in degrees) to the signal
        """
        n = len(self.freqs)

        # compute the index at which the specified frequency is located in the
        # array
        index = int(np.round(float(freq)*n/self._sampling_rate))

        # distribute the signal amplitude over the real and imaginary axes
        re = float(n)*amplitude*np.cos(phase*np.pi/180.0)
        im = float(n)*amplitude*np.sin(phase*np.pi/180.0)

        # distribute AC component evenly over positive and negative
        # frequencies
        if freq != 0: 
            re = re/2.0
            im = im/2.0

            # to ensure real-valued time-domain signal, the two parts need to
            # be complex conjugate of each other
            self.freqs[ index] = re + 1j*im
            self.freqs[-index] = re - 1j*im

        else:
            # DC component has only one part
            self.freqs[index] = re + 1j*im


    def read_time_function(self,func,clear=True,offset=0,duration=None):
        """
        Sample values from a vectorized time-domain, real-valued function,
        func(t), where t will be specified in second.  Samples are collected
        at the sampling rate associated with the Signal object.  The function
        values will be updated on the signal at the (optionally) specified
        time offset for the specified duration.  If duration is omitted, the
        update will be applied up to the end of the signal. 

        If the parameter clear is True, current signal values will be cleared
        before being set to the new values.

        """
        #n = len(self.freqs)
        #signal = np.arange(n, dtype=float)
        #for i in range(n):
        #    signal[i] = func(float(i)/self._sampling_rate)
        #self.freqs = fft(signal)

        ts = self.time_domain
        if duration is None:
            interval = (ts.t >= offset)
        else:
            interval = (ts.t >= offset) & (ts.t < offset+duration)
        if clear:
            ts.loc[interval,"a"] = func(ts[interval].t.as_matrix()-offset)
        else:
            ts.loc[interval,"a"] += func(ts[interval].t.as_matrix()-offset)

        self.freqs = fft(ts.a)


    @property
    def time_domain(self):
        """
        Return a DataFrame with two columns, t and a, representing
        the time sequence and the corresponding amplitude, respectively.
        """
        t_axis = np.linspace(0, self.duration, len(self.freqs))
        a_axis = ifft(self.freqs).real
        return DataFrame(np.array([t_axis,a_axis]).T,columns=["t","a"])


    @property
    def complex_freq_domain(self):
        """
        Return a DataFrame with two columns, f and
        c, representing the frequencies up to the Nyquist frequency, both
        negative and positive, and complex coefficients of all the
        frequencies.
        """
        n = len(self.freqs)
        num_freqs = int(np.ceil((n-1)/2))

        f_axis = (np.arange(n)-num_freqs)/float(n)*self._sampling_rate

        # align the coefficients with the frequencies (make 0 Hz center), then
        # scale the coefficients so that their magnitudes do not depend on the
        # length of the array
        c_axis = np.roll(self.freqs,num_freqs)/float(n)

        return DataFrame({"f":f_axis.real, "c":c_axis},columns=["f","c"])


    @property
    def freq_domain(self):
        """
        Return a DataFrame with three columns, f, a, and p, representing the
        frequencies up to the Nyquist frequency (excluding negative
        frequency), peak amplitudes, and phase-shifts in degrees of all the
        frequencies.

        """
        n = len(self.freqs)
        num_freqs = int(np.ceil((n-1)/2))

        f_axis = np.arange(num_freqs)/float(n)*self._sampling_rate

        # extract only positive frequencies and scale them so that the
        # magnitude does not depend on the length of the array
        a_axis = abs(self.freqs[:num_freqs])/float(n)
        p_axis = np.angle(self.freqs[:num_freqs],deg=True)

        # double amplitudes of the AC components (since we have thrown away
        # the negative frequencies)
        a_axis[1:] = a_axis[1:]*2

        return DataFrame(
                np.array([f_axis,a_axis,p_axis]).T,
                columns=["f","a","p"])


    def shift_freq(self, offset):
        """
        Shift signal in the frequency domain by the amount specified by offset
        (in Hz).  If offset is positive, the signal is shifted to the right
        along the frequency axis.  If offset is negative, the signal is
        shifted to the left along the frequency axis.  In case of negative
        shifting, all frequency components shifted pass the 0 Hz mark will be
        discarded.
        """
        n = len(self.freqs)
        nyquist = n//2

        # compute the array-based index from the specified offset in Hz
        offset = int(np.round(float(offset)*n/self._sampling_rate))
        if abs(offset) > nyquist:
            raise Exception(
            "Shifting offset cannot be greater than the Nyquist frequency")

        # treat freqs[0] (DC component) separately
        dc = self.freqs[0]
        self.freqs[0] = 0
        if offset > 0:
            self.freqs[offset:nyquist] = np.copy(self.freqs[:nyquist-offset])
            self.freqs[:offset] = 0

            self.freqs[-nyquist+1:-offset] = np.copy(self.freqs[-(nyquist-offset)+1:])
            self.freqs[-offset+1:] = 0

            # DC component gets split into positive and negative frequencies
            self.freqs[offset] = dc/2
            self.freqs[-offset] = dc/2
        else:
            offset = -offset
            self.freqs[:nyquist-offset] = np.copy(self.freqs[offset:nyquist])
            self.freqs[nyquist-offset:nyquist] = 0

            self.freqs[-(nyquist-offset)+1:] = np.copy(self.freqs[-nyquist+1:-offset])
            self.freqs[-nyquist+1:-nyquist+offset] = 0

            # DC component is the sum of both pos and neg freq
            self.freqs[0] *= 2


    def shift_time(self, offset):
        """
        Shift signal in the time domain by the amount specified by offset
        (in seconds).  If offset is positive, the signal is shifted to the
        right along the time axis.  If offset is negative, the signal is
        shifted to the left along the time axis.
        """
        noff = int(offset*self._sampling_rate)
        x,y = self.time_domain.values.T
        if noff > 0:
            y[noff:] = y[:len(x)-noff].copy()
            y[:noff] = 0.0
        elif noff < 0:
            noff = -noff
            y[:len(x)-noff] = y[noff:].copy()
            y[len(x)-noff:] = 0.0
        self.freqs = fft(y)


    def clear_time(self, cond=None):
        """
        Set amplitude of the signal all the time points satisfying cond to
        zero.  cond is a boolean fuction that takes one parameter, t,
        specifying a point in time (in seconds).  If not specified, the entire
        signal is cleared.
        """
        if cond is None:
            cond = lambda t: True
        x,y = self.time_domain.values.T
        for i in range(len(x)):
            if cond(x[i]):
                y[i] = 0.0
        self.freqs = fft(y)


    def copy(self, start_time=None, end_time=None):
        """
        Clone the signal object into another identical signal object.  If
        start_time and/or end_time are not None, only the specified signal
        portion is copied.
        """
        if start_time is None:
            start_time = 0.0
        if end_time is None:
            end_time = self.duration
        start_index = int(start_time*self._sampling_rate)
        end_index   = int(end_time*self._sampling_rate)
        x,y = self.time_domain.values.T

        s = Signal()
        s._sampling_rate = self._sampling_rate
        s.freqs = fft(y[start_index:end_index])

        return s


    @property
    def time_function(self):
        table = self.time_domain
        window = 1/self.sampling_rate

        def func(t):
            entries = table[ (table.t > t-window*2/3) & (table.t < t+window*2/3)]
            return entries.a.mean()

        return np.vectorize(func)


    def trim(self, start_time, end_time):
        """
        Trim the signal from the given start_time to the given end_time
        """
        start_index = start_time*self._sampling_rate
        end_index   = end_time*self._sampling_rate
        x,y = self.time_domain.values.T
        self.freqs = fft(y[start_index:end_index])


    def mix(self, signal, start_time=None, end_time=None, offset=None):
        """
        Mix the signal with another given signal.  Sampling rates of both
        signals must match.  If start_time and/or end_time are specified (in
        seconds), only the range of the given signal is used.  If offset is
        specified (in seconds), the mixing process will start at the given
        offset of the current signal.
        """
        if self._sampling_rate != signal._sampling_rate:
            raise Exception(
                "Signal to mix must have identical sampling rate")

        x_self,y_self = self.time_domain.values.T
        x_other,y_other = signal.time_domain.values.T

        if offset is None:
            offset = 0.0
        if start_time is None:
            start_time = 0.0
        if end_time is None:
            end_time = x_other[-1]

        offset_index = int(offset*self._sampling_rate)
        start_index  = int(start_time*self._sampling_rate)
        end_index    = int(end_time*self._sampling_rate)
        count = end_index - start_index

        # if necessary, extend the current array to make sufficient room for
        # signal mixing
        if offset_index + count >= len(y_self):
            y_self = np.append(y_self, np.zeros(offset_index+count-len(y_self)))

        y_self[offset_index:offset_index+count] += y_other[start_index:start_index+count]

        self.freqs = fft(y_self)


    def __add__(self, s):
        newSignal = self.copy()
        newSignal.mix(s)
        return newSignal


    def plot_raw_freq(self,fig_options={},line_options={}):
        """
        Generate a raw plot of the signal's frequency domain data.  The x-axis
        will show the raw indices instead of the corresponding frequency
        values.  Only the magnitude of each frequency is displayed.
        """
        if not NOTEBOOK_SESSION:
            raise Exception("This method is available in a notebook session only.")
        fig_cnf = {**DEFAULT_PLOT_SETTINGS["raw_fig_options"], **fig_options}
        line_cnf = {**DEFAULT_PLOT_SETTINGS["raw_line_options"], **line_options}
        fig = bkp.figure(**fig_cnf)
        indices = np.arange(len(self.freqs))
        fig.segment(indices,0,indices,np.abs(self.freqs),**line_cnf)
        bkp.show(fig)


    def plot_time(self,fig=None,fig_options={},line_options={}):
        """
        Generate a time vs. amplitude plot for the signal

        If an exist Bokeh's figure is not provided, a new figure will be
        created and the plot will be displayed immediately.

        The specified Bokeh figure's and line's options will be combined with
        the values from the dict DEFAULT_PLOT_SETTINGS.  See the contents of
        DEFAULT_PLOT_SETTINGS for the complete list of plot settings.
        """
        if not NOTEBOOK_SESSION:
            raise Exception("This method is available in a notebook session only.")
        fig_cnf = {**DEFAULT_PLOT_SETTINGS["time_fig_options"], **fig_options}
        line_cnf = {**DEFAULT_PLOT_SETTINGS["time_line_options"], **line_options}
        ts = self.time_domain
        if fig is not None:
            current_fig = fig
        else:
            current_fig = bkp.figure(**fig_cnf)
        current_fig.line(ts.t*1000,ts.a,**line_cnf)
        if fig is None:
            bkp.show(current_fig)


    def plot_freq(self,fig=None,min_amplitude=1e-4,fig_options={},line_options={}):
        """
        Generate a freq vs. magnitude plot of the signal.

        If an exist Bokeh's figure is not provided, a new figure will be
        created and the plot will be displayed immediately.

        The parameter min_amplitude is used for filtering only frequency
        components whose amplitudes are greater than the specified value.

        The specified Bokeh figure's and line's options will be combined with
        the values from the dict DEFAULT_PLOT_SETTINGS.  See the contents of
        DEFAULT_PLOT_SETTINGS for the complete list of plot settings.
        """
        if not NOTEBOOK_SESSION:
            raise Exception("This method is available in a notebook session only.")
        fig_cnf = {**DEFAULT_PLOT_SETTINGS["freq_fig_options"], **fig_options}
        line_cnf = {**DEFAULT_PLOT_SETTINGS["freq_line_options"], **line_options}
        fs = self.freq_domain
        fs = fs[fs.a >= min_amplitude]
        if fig is not None:
            current_fig = fig
        else:
            current_fig = bkp.figure(**fig_cnf)
        current_fig.segment(fs.f,0,fs.f,fs.a,**line_cnf)
        if fig is None:
            bkp.show(current_fig)


    def plot_phase(self,fig=None,min_amplitude=1e-4,fig_options={},line_options={},point_options={}):
        """
        Generate a freq vs. phase plot of the signal in degrees.

        If an exist Bokeh's figure is not provided, a new figure will be
        created and the plot will be displayed immediately.

        The parameter min_amplitude is used for filtering only frequency
        components whose amplitudes are greater than the specified value.

        The specified Bokeh figure's and line's options will be combined with
        the values from the dict DEFAULT_PLOT_SETTINGS.  See the contents of
        DEFAULT_PLOT_SETTINGS for the complete list of plot settings.
        """
        if not NOTEBOOK_SESSION:
            raise Exception("This method is available in a notebook session only.")
        fig_cnf = {**DEFAULT_PLOT_SETTINGS["phase_fig_options"], **fig_options}
        line_cnf = {**DEFAULT_PLOT_SETTINGS["phase_line_options"], **line_options}
        point_cnf = {**DEFAULT_PLOT_SETTINGS["phase_point_options"], **point_options}
        fs = self.freq_domain
        fs = fs[fs.a >= min_amplitude]
        if fig is not None:
            current_fig = fig
        else:
            current_fig = bkp.figure(**fig_cnf)
        current_fig.segment(fs.f,0,fs.f,fs.p,**line_cnf)
        current_fig.scatter(fs.f,fs.p,**point_cnf)
        if fig is None:
            bkp.show(current_fig)


    def play(self):
        """
        Create an Audio display that allows the signal to be played as audio data
        """
        if not NOTEBOOK_SESSION:
            raise Exception("This method is available in a notebook session only.")
        from IPython.display import Audio
        return Audio(self.time_domain.a,rate=self.sampling_rate)


#############################################
if __name__ == "__main__":
    print("The sigproc module is designed for Jupyter notebook")
