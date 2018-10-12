from __future__ import absolute_import, division, print_function, unicode_literals

from skhep.math.vectors import LorentzVector


class MadMinerParticle(LorentzVector):
    def __init__(self):
        super(MadMinerParticle, self).__init__()

        self.charge = None
        self.pdgid = None

    def set_pdgid(self, pdgid):

        self.pdgid = int(pdgid)
        self.charge = 0.

        if self.pdgid in [11, 13, 15, -24]:
            self.charge = -1.
        elif self.pdgid in [-11, -13, -15, 24]:
            self.charge = 1.
        elif self.pdgid in [1, 3, 5]:
            self.charge = 2. / 3.
        elif self.pdgid in [-1, -3, -5]:
            self.charge = -2. / 3.
        elif self.pdgid in [2, 4, 6]:
            self.charge = -1. / 3.
        elif self.pdgid in [-2, -4, -6]:
            self.charge = 1. / 3.
