import sys
from analog import *

OUTPUT = "example.wav"
BITRATE = 10
FC = [100,200,300]
BANDWIDTH=80
MODULATION = { 
    '00' : (0.3,  0),
    '01' : (0.3, 90),
    '10' : (0.3,180),
    '11' : (0.3,270),
}
bits = [
    "01001011",
    "11010010",
    "00010100",
]

qams = [Qam(bitrate=BITRATE,fc=fc,modulation=MODULATION) for fc in FC]
signals = [q.generate_signal(b) for q,b in zip(qams,bits)]

for sig,fc in zip(signals,FC):
    sig.filter((fc-BANDWIDTH/2,fc+BANDWIDTH/2))

combined = signals[0] + signals[1] + signals[2]
combined.write_wav(OUTPUT)
print("Output written to", OUTPUT)
