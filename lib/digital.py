import sys
from sigproc import Signal

#################################
def create_encoder(bitrate,symmap):
    '''Create a line coder for digital data'''

    # determine the number of bits per signal from the first mapping entry
    bit_per_baud = len(next(iter(symmap)))
    baud = bitrate/bit_per_baud

    def line_coder(data):
        s = Signal(duration=float(len(data))/(bitrate))
        for i in range(0,len(data),bit_per_baud):
            symbol = symmap[data[i:i+bit_per_baud]]
            interval = 1/baud
            time = interval*i/bit_per_baud
            s.add_pulse(symbol,time,interval)
        return s

    return line_coder

##############################
def enc_ttl(data,bitrate):
    symmap = {'0':0.0,'1':5.0}
    encoder = create_encoder(bitrate,symmap)
    return encoder(data)

##############################
def enc_nrz(data,bitrate):
    symmap={'0':-.5,'1':.5}
    encoder = create_encoder(bitrate,symmap)
    return encoder(data)

##############################
def enc_rz(data,bitrate):
    data_mod = data.replace('0','-0').replace('1','+0')
    bitrate *=2
    symmap = { '+' : +0.5, '-' : -0.5, '0' :  0 }
    encoder = create_encoder(bitrate,symmap)
    return encoder(data_mod)

##############################
def enc_manchester(data,bitrate):
    data_mod = data.replace('0','-+').replace('1','+-')
    bitrate *=2
    symmap = { '+' : +0.5, '-' : -0.5 }
    encoder = create_encoder(bitrate,symmap)
    return encoder(data_mod)

##############################
def enc_2b1q(data,bitrate):
    symmap = {
        '00' : -0.5,
        '01' : -0.2,
        '11' : +0.2,
        '10' : +0.5,
    }
    encoder = create_encoder(bitrate,symmap)
    return encoder(data)

##############################
encodings = {
    'ttl' : enc_ttl,
    'nrz' : enc_nrz,
    'rz' : enc_rz,
    'manchester' : enc_manchester,
    'diff-manchester' : enc_manchester,
    '2b1q' : enc_2b1q,
}

##############################
def encode(encoder,bitrate,data):
    try:
        data = data.replace(" ","")
        return encodings[encoder](data,bitrate)
    except KeyError:
        print("Unknown encoder '%s'" % encoder)
        print("Available encodings are %s" % ",".join(encodings.keys()))
