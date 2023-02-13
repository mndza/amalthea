#!/usr/bin/env python3
# pylint: disable=no-member

import os
import sys
import logging
import time
import usb1
import xdg.BaseDirectory

import numpy as np

from amaranth                        import *
from amaranth.lib.fifo               import AsyncFIFO
from amaranth.build.run              import LocalBuildProducts

from .stream                         import IQStream, SampleStream, StreamCombiner

from ..gnuradio.amaltheasimblock     import AmaltheaSimBlock


class Simulator(Elaboratable):
    """ Amalthea device. """

    def __init__(self, vcd_file=None):
        self._blocks = {}
        self._usb_outputs = []
        self._connections = []
        self._usb_connections = []
        # new input direction
        self._usb_inputs = []
        self._usb_connections_in = []
        self._vcd_file = vcd_file or None

    def get_rx(self):
        return None

    def add_block(self, block_id, block):
        self._blocks[block_id] = block
        return block

    def connect(self, source, sink):
        print(f"connect {source} {sink}")
        source_id, source_output = source
        sink_id, sink_input = sink

        source_block = self._blocks[source_id]
        sink_block   = self._blocks[sink_id]
        self._connections.append(sink_block.input.stream_eq(source_block.outputs[source_output]))

    def connect_usb(self, source, sink):
        print(f"connect_usb {source}")
        block_id, output_id = source
        self._usb_outputs.append(self._blocks[block_id].outputs[output_id])
        self._usb_connections.append((len(self._usb_outputs)-1, sink))

    def connect_usb_in(self, source, sink):
        print(f"connect_usb_in {source}")
        sink_id, sink_input = sink
        self._usb_inputs.append(self._blocks[sink_id].input)  # not supporting multiple inputs atm
        self._usb_connections_in.append((len(self._usb_inputs)-1, source))

    def flash(self):
        pass

    def finalize_usb_connections(self, tb):
        def to_np_type(output):
            if isinstance(output, IQStream):
                return np.complex64
            if isinstance(output, SampleStream):
                return np.float32
            raise TypeError("Unknown stream type")

        signature_in  = list(map(to_np_type, self._usb_inputs))
        signature_out = list(map(to_np_type, self._usb_outputs))
        self._sim_block = AmaltheaSimBlock(4e6, 2.45e9, self, signature_in, signature_out, self._vcd_file)
        for conn in self._usb_connections_in:
            tb.connect(conn[1], (self._sim_block, conn[0]))
        for conn in self._usb_connections:
            tb.connect((self._sim_block, conn[0]), conn[1])

    def elaborate(self, platform):
        m = Module()

        # Create radio clock domain
        m.domains.radio = ClockDomain()
        m.d.comb += [
            ClockSignal("radio").eq(ClockSignal()),
            ResetSignal("radio").eq(ResetSignal()),
        ]

        #
        m.submodules += map(DomainRenamer("radio"), self._blocks.values())
        m.d.comb += self._connections

        return m

