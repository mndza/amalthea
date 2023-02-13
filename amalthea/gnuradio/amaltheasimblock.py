#!/usr/bin/env python3

import os
import sys
import logging
import time

import numpy as np
from gnuradio import gr

from amaranth import *
from amaranth.sim import Simulator, Passive
from contextlib import nullcontext

SAMPWIDTH = 16
SCALE = (1 << (SAMPWIDTH-1)) - 1
MASK = (1 << SAMPWIDTH) - 1

def convert_to_fp(arr):
    scaled_block = np.round(arr * SCALE)
    if scaled_block.dtype == np.complex64:
        block  = scaled_block.real.astype(np.int32) & MASK
        block |= (scaled_block.imag.astype(np.int32) & MASK) << SAMPWIDTH
    else:
        block = scaled_block.astype(np.int16) & MASK
    return block

def convert_from_fp(arr, dtype):
    if dtype == np.complex64:
        real = (arr & MASK).astype(np.int16)
        imag = ((arr >> SAMPWIDTH) & MASK).astype(np.int16)
        re_f = real.astype(np.float32) / SCALE
        im_f = imag.astype(np.float32) / SCALE
        block = re_f + 1j*im_f
    else:
        block = arr.astype(np.int16).astype(np.float32) / SCALE
    return block

def get_input_process(stream, arr):
    # Create a fixed-point generator from input array
    input_items_fp = convert_to_fp(arr)
    gen = (int(x) for x in input_items_fp)
    # Return process to feed stream until generator is empty
    def f():
        while True:
            if (yield stream.ready | ~stream.valid):  # not stalled
                try:
                    yield stream.payload.eq(next(gen))
                    yield stream.valid.eq(1)
                except StopIteration:
                    yield stream.valid.eq(0)
                    return
            
            yield  # advance clock
    return f


class AmaltheaSimBlock(gr.sync_block):
    BLOCK_NAME='Amalthea Simulator'

    def __init__(self, sample_rate, freq, dut, in_sig=[np.float32], out_sig=[np.float32], vcd_file=None):
        gr.sync_block.__init__(
            self,
            name=self.BLOCK_NAME,
            in_sig=in_sig,
            out_sig=out_sig,
        )
        self._dut      = dut
        self._in_sig   = in_sig
        self._out_sig  = out_sig
        self._vcd_file = vcd_file

        self.sample_rate  = sample_rate
        self.freq         = freq

    def start(self):
        self.sim = sim = Simulator(self._dut)
        sim.add_clock(1/64e6)

        # Collect all outputs during the simulation calls
        outputs = self._dut._usb_outputs
        self.out_buffers = [ np.array([], dtype=np.int32) for _ in range(len(outputs)) ]
        def output_process():
            yield Passive()  # Never block
            for stream in outputs:
                yield stream.ready.eq(1)  # Never stall the outputs

            while True:
                for out_id, stream in enumerate(outputs):
                    if (yield stream.valid):
                        self.out_buffers[out_id] = \
                            np.append(self.out_buffers[out_id], (yield stream.payload))
                yield
        self.sim.add_sync_process(output_process)

        # Set up simulation context
        if self._vcd_file is not None:
            self.context = self.sim.write_vcd(vcd_file=self._vcd_file)
        else:
            self.context = nullcontext()
        # Use context manually instead of using "with" statement
        self.context.__enter__()

    def stop(self):
        self.context.__exit__(None, None, None)

    def set_freq(self, freq):
        self.freq = freq
        return True

    def work(self, input_items, output_items):

        # Every input stream is driven by its own separate sync process
        for i in range(len(input_items)):
            proc = get_input_process(self._dut._usb_inputs[i], input_items[i])
            self.sim.add_sync_process(proc)

        # Run the simulation for the current input block
        # Outputs are already handled in another background process
        self.sim.run()

        # At this point, output samples are already collected in the output buffers
        # Because we are using a sync_block, we can only output the same number
        # of items in every output buffer. Keep any difference in the internal
        # buffers for the next iteration.
        sample_count = min(len(buf) for buf in self.out_buffers)

        if sample_count == 0:
            return 0
        
        raw_outputs = [ buf[:sample_count] for buf in self.out_buffers ]
        for i in range(len(output_items)):
            buf = self.out_buffers[i]
            out = output_items[i]
            out[:sample_count] = convert_from_fp(buf[:sample_count], self._out_sig[i])
            self.out_buffers[i] = buf[sample_count:]

        return sample_count
