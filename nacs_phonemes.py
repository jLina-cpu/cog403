# %%
from pyClarion import (Agent, InputBL, ChoiceBL, ChunkStore, FixedRules, 
    Family, ChoiceTL, Atoms, Atom, InputTL, Key)
from datetime import timedelta

import logging
import sys

# %%
logger = logging.getLogger("pyClarion.system")
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))

# %%
class Position(Atoms):
    p1: Atom
    p2: Atom
    p3: Atom

# "cat mat hat pat put top {ch}at let met pal"

"cat pat pil pal"

class Phoneme(Atoms):
    a: Atom
    c: Atom
    i: Atom
    l: Atom
    p: Atom
    t: Atom


class Words(Family):
    pos: Position
    phon: Phoneme

# %%
p = Family()
d = Words()
with Agent("agent", p=p, d=d) as agent:
    data_in = InputBL("data_in", d=d, v=d)
    chunks = ChunkStore("chunks", c=d, d=d, v=d)
    retrieval = ChoiceTL("retrieval", p=p, t=chunks.chunks, sd=1e-4)
    chunks.bu.input = data_in.main
    retrieval.input = chunks.bu.main

# %%
pos = d.pos
phon = d.phon
chunk_defs = [
    + pos.p1 ** phon.c
    + pos.p2 ** phon.a
    + pos.p3 ** phon.t,

    + pos.p1 ** phon.p
    + pos.p2 ** phon.a
    + pos.p3 ** phon.t,

    + pos.p1 ** phon.p
    + pos.p2 ** phon.i
    + pos.p3 ** phon.l,

    + pos.p1 ** phon.p
    + pos.p2 ** phon.a
    + pos.p3 ** phon.l,
]
chunks.compile(*chunk_defs)

# %%

data_in.send(+ pos.p2 ** phon.a)
while agent.system.queue:
    event = agent.system.advance()
    if event.source == chunks.bu.update:
        retrieval.select()
print(retrieval.poll())
print(chunks.bu.input)
print(chunks.bu.main)

# %%


# %%
class Marker(Atoms):
    start: Atom

class Phoneme(Atoms):
    a: Atom
    c: Atom
    i: Atom
    l: Atom
    p: Atom
    t: Atom

class Words(Family):
    mrk: Marker
    phon: Phoneme

# %%
p = Family()
d = Words()
with Agent("agent", p=p, d=d) as agent:
    data_in = InputBL("data_in", d=d, v=d)
    chunks = ChunkStore("chunks", c=d, d=d, v=d)
    bias = InputTL("bias", chunks.chunks, reset=False)
    retrieval = ChoiceTL("retrieval", p=p, t=chunks.chunks, sd=1e-4)
    chunks.bu.input = data_in.main
    retrieval.input = chunks.bu.main
    retrieval.bias = bias.main

# %%
mrk = d.mrk
phon = d.phon
chunk_defs = [

    # Sc ca at
    + mrk.start ** phon.c
    + phon.c ** phon.a
    + phon.a ** phon.t,

    + mrk.start ** phon.p
    + phon.p ** phon.a
    + phon.a ** phon.t,

    + mrk.start ** phon.p
    + phon.p ** phon.i
    + phon.i ** phon.l,

    + mrk.start ** phon.p
    + phon.p ** phon.a
    + phon.a ** phon.l
]
chunks.compile(*chunk_defs)
while agent.system.queue:
    agent.system.advance()

# %%
bias.send({chunks.chunks["c1"]: .0005, chunks.chunks["c2"]: .0002, chunks.chunks["c3"]: .0004})
data_in.send(+ mrk.start ** phon.p)
while agent.system.queue:
    event = agent.system.advance()
    if event.source == chunks.bu.update:
        retrieval.select()
print(retrieval.poll())
print(chunks.bu.input)
print(chunks.bu.main)
print(retrieval.bias.sum(retrieval.input))

# %%
# pap    sp pa ap
# papa   sp pa ap pa 
# papapa sp pa ap pa ap pa

words = [
    + mrk.start ** phon.p 
    + phon.p ** phon.a 
    + phon.a ** phon.p,

    + mrk.start ** phon.p 
    + 2 * phon.p ** phon.a 
    + phon.a ** phon.p,

    + mrk.start ** phon.p 
    + 3 * phon.p ** phon.a 
    + 2 * phon.a ** phon.p,
]


words = [
    + mrk.start ** phon.p 
    + phon.p ** phon.a 
    + phon.a ** phon.p,

    + mrk.start ** phon.p 
    + 2 * phon.p ** phon.a 
    + phon.a ** phon.p,

    + mrk.start ** phon.p 
    + 3 * phon.p ** phon.a 
    + 2 * phon.a ** phon.p,
]


