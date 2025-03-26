from pyClarion import (Agent, ChunkStore, FixedRules, 
    Family, Atoms, Atom)
from datetime import timedelta

import logging
import sys

logger = logging.getLogger("pyClarion.system")
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))

from pyClarion import Agent 
from pyClarion import Input 
from pyClarion import ChunkStore 
from pyClarion import (Chunks, Chunk)
from pyClarion import Layer, Train
from pyClarion import Choice
import numpy


class Words(Chunks):
    P: Chunk
    Penguin: Chunk
    Cold: Chunk
    Ice: Chunk
    Pizza: Chunk
    Cheese: Chunk
    Pepperoni: Chunk
    Pencil: Chunk
    Write: Chunk
    Paper: Chunk
    Puppy: Chunk
    Dog: Chunk
    Cute: Chunk
    Park: Chunk
    Trees: Chunk
    Play: Chunk
    Popcorn: Chunk
    Movie: Chunk
    Butter: Chunk
    Phone: Chunk
    Call: Chunk
    Ring: Chunk
    Piano: Chunk
    Music: Chunk
    Keys: Chunk

class SimData(Family):
    words: Words

d = SimData()
p = Family()

agent = Agent("agent", d=d, p=p)
with agent:
    ipt = Input("ipt", (d)) #input is d
    layer = Layer("layer", d, train=Train.NIL)
    choice = Choice("choice", p, d.words)
    layer.input = ipt.main
    choice.input = layer.main


# calculate W in numpy
#n is the number of words
#N is the number of iterations
a = (alpha * numpy.eye(n,n) + (1-alpha) * M)^N 

a = numpy.random.rand(3,3)
word_list = ["P", "Penguin", "Cold"]
for i, word_i in enumerate(word_list):
    for j, word_j in enumerate(word_list):
        with layer.weights[0].mutable() as weights:
            weights[~d.words[word_i] * ~d.words[word_j]] = a[i,j] #weights is a numdict, taks a[i,j] as value





    # chunks = ChunkStore("chunks", c=Words, d=Words, v=Words)
    # chunks.bu.input = ipt.main

