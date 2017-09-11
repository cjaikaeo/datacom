from math import ceil
import numpy as np
from sigproc import Signal,start_notebook
from digital import create_encoder as line_coder

#################################
class Qam:

    #################################
    def __init__(self,modulation,bitrate,fc):
        '''
        Create a modulator instance using the symbol mapping provided via the
        parameter modulation, whose entries are of the form:

          <data-bit(s)> : (<amplitude>,<phase-degrees>)

        The parameter bitrate is used to determine the symbol rate.  The
        modulation uses the frequence fc as the carrier frequency.
        '''
        keylens = [len(x) for x in modulation.keys()]
        if min(keylens) != max(keylens):
            raise Exception('All bit patterns must have the same length')
        self.modulation    = modulation
        self.bits_per_baud = keylens[0]
        self.baud_rate     = bitrate/self.bits_per_baud
        self.fc = fc

    #################################
    @property
    def bit_rate(self):
        return self.baud_rate*self.bits_per_baud

    #################################
    @staticmethod
    def generate_frequency(hz,duration,phase=0,amplitude=1):
        s = Signal(duration=duration)
        s.read_time_function(lambda t: np.sin(2*np.pi*hz*t + np.radians(phase)))
        return s

    #################################
    def generate_signal(self, data):
        '''
        Generate signal corresponding to the current modulation scheme to
        represent given binary string, data.
        '''
        data = data.replace(" ","")
        duration = float(len(data))/(self.baud_rate*self.bits_per_baud)
        s = Signal(duration=duration)
        sym_interval = 1/self.baud_rate

        # pad missing data in the last symbol with '0'
        padding = -len(data) % self.bits_per_baud
        data += '0'*padding

        for i in range(0,len(data),self.bits_per_baud):
            amp,phase = self.modulation[data[i:i+self.bits_per_baud]]
            phase = np.radians(phase)
            offset = sym_interval*i/self.bits_per_baud
            s.read_time_function(
                    lambda t: amp*np.sin(2*np.pi*self.fc*t + phase),
                    offset=offset,
                    duration=sym_interval)

        return s

    #################################
    def baseband_encoders(self):
        """
        Return baseband (digital) encoders for both I and Q components
        """
        symmap_i = {}
        symmap_q = {}
        for bits in self.modulation:
            a,p = self.modulation[bits]
            symmap_i[bits] = a*np.cos(np.radians(p))
            symmap_q[bits] = a*np.sin(np.radians(p))

        enci = line_coder(self.bit_rate,symmap_i)
        encq = line_coder(self.bit_rate,symmap_q)
        return enci,encq

    #################################
    def baseband_signals(self,data):
        """
        Return baseband (digital) signals for both I and Q components
        """
        enci,encq = self.baseband_encoders()
        data = data.replace(" ","")
        return enci(data),encq(data)

    #################################
    def carriers(self,duration):
        """
        Return carrier frequency signals for both I and Q components
        """
        carrier_i = Signal(duration=duration)
        carrier_q = Signal(duration=duration)
        # XXX need investigation why these two lines do not work
        #carrier_i.set_freq(self.fc,1,phase=-90)
        #carrier_q.set_freq(self.fc,1,phase=0)
        carrier_i.read_time_function(lambda t: np.sin(2*np.pi*self.fc*t))
        carrier_q.read_time_function(lambda t: np.sin(2*np.pi*self.fc*t + np.pi/2))
        return carrier_i, carrier_q

    #################################
    def demodulate(self,signal):
        '''
        Demodulate the provided signal using the carrier frequency, then
        return two signals for I and Q components
        '''
        i_sig = signal.copy()
        q_sig = signal.copy()

        # multiple receievd signal by the carrier frequency
        i_sig.filter_time(lambda t: 2*np.sin(2*np.pi*self.fc*t))
        q_sig.filter_time(lambda t: 2*np.sin(2*np.pi*self.fc*t + np.pi/2))

        # feed the results thru a low-pass filter
        i_sig.filter(self.fc)
        q_sig.filter(self.fc)

        return i_sig,q_sig

    #################################
    def closest_data_point(self,point):
        '''
        Find the data point whose signal components are closest to the
        provided complex value point
        '''
        # convert mag/phase form to rectangular form
        rect = np.array([mag*np.exp(1j*np.radians(phase)) 
                for mag,phase in self.modulation.values()])
        idx = np.argmin(np.abs(rect-point))

        return list(self.modulation.keys())[idx]

    #################################
    def plot_constellation(self,fig=None):
        '''
        Plot a constellation diagram representing the modulation scheme.
        '''
        from sigproc import NOTEBOOK_SESSION
        if not NOTEBOOK_SESSION:
            raise Exception("This function is only available in a notebook session")
        from sigproc import bkp
        from bokeh.models import ColumnDataSource,FixedTicker,LabelSet
        data = [(a*np.cos(np.radians(p)),a*np.sin(np.radians(p)),bits) 
                for bits,(a,p) in self.modulation.items()]
        x,y,bits = zip(*data)
        source = ColumnDataSource(data=dict(x=x,y=y,bits=bits))
        bound = ceil(max(abs(v) for v in x+y)) + 0.5
        fig_created = False
        if fig is None:
            fig_created = True
            fig = bkp.figure(
                    height=300,
                    width=400,
                    x_range=(-bound,bound),
                    y_range=(-bound,bound),
                    )
        fig.segment(-bound,0,bound,0,color="green",line_width=2)
        fig.segment(0,-bound,0,bound,color="green",line_width=2)
        fig.scatter(x="x",y="y",size=10,source=source,line_color="black",fill_color="red")
        set_aspect(fig,(-bound,bound),(-bound,bound))
        fig.xaxis[0].ticker = FixedTicker(ticks=np.arange(-bound,bound+0.5,0.5))
        fig.xgrid[0].ticker = FixedTicker(ticks=np.arange(-bound,bound+0.5,0.1))
        fig.yaxis[0].ticker = FixedTicker(ticks=np.arange(-bound,bound+0.5,0.5))
        fig.ygrid[0].ticker = FixedTicker(ticks=np.arange(-bound,bound+0.5,0.1))
        fig.add_layout(LabelSet(x="x",y="y",text="bits",x_offset=5,y_offset=5,text_font_size="8pt",source=source))
        if fig_created:
            bkp.show(fig)
        else:
            return fig
    
#################################
def ask_modulation(bitrate,fc):
    return Qam(modulation={
        '0':(0.5,0), 
        '1':(1.0,0)
        },
        bitrate=bitrate,
        fc=fc)

#################################
def ook_modulation(bitrate,fc):
    return Qam(modulation={
        '0':(0.0,0), 
        '1':(1.0,0)
        },
        bitrate=bitrate,
        fc=fc)

#################################
def psk_modulation(bitrate,fc):
    return Qam(modulation={
        '0':(1,0), 
        '1':(1,180)
        },
        bitrate=bitrate,
        fc=fc)

#################################
def qpsk_modulation(bitrate,fc):
    return Qam(modulation={
        '00':(1,-135), 
        '01':(1,135),
        '10':(1,-45),
        '11':(1,45),
        },
        bitrate=bitrate,
        fc=fc)

#################################
# Taken and modified from https://stackoverflow.com/a/36809908
def set_aspect(fig,xrange,yrange,aspect=1,margin=0.1):
    """Set the plot ranges to achieve a given aspect ratio.

    Args:
      fig (bokeh Figure): The figure object to modify.
      x (iterable): The x-coordinates of the displayed data.
      y (iterable): The y-coordinates of the displayed data.
      aspect (float, optional): The desired aspect ratio. Defaults to 1.
        Values larger than 1 mean the plot is squeezed horizontally.
      margin (float, optional): The margin to add for glyphs (as a fraction
        of the total plot range). Defaults to 0.1
    """
    from bokeh.models import Range1d
    xmin,xmax = xrange
    ymin,ymax = yrange
    width = (xmax - xmin)*(1+2*margin)
    if width <= 0:
        width = 1.0
    height = (ymax - ymin)*(1+2*margin)
    if height <= 0:
        height = 1.0
    xcenter = 0.5*(xmax + xmin)
    ycenter = 0.5*(ymax + ymin)
    r = aspect*(fig.plot_width/fig.plot_height)
    if width < r*height:
        width = r*height
    else:
        height = width/r
    fig.x_range = Range1d(xcenter-0.5*width, xcenter+0.5*width)
    fig.y_range = Range1d(ycenter-0.5*height, ycenter+0.5*height)
