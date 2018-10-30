from __future__ import absolute_import, division, print_function, unicode_literals

from skhep.math.vectors import LorentzVector


class MadMinerParticle(LorentzVector):
    """ """

    def __init__(self, *args, **kwargs):
        super(MadMinerParticle, self).__init__(*args, **kwargs)

        self.charge = None
        self.pdgid = None

    def set_pdgid(self, pdgid):
        """

        Parameters
        ----------
        pdgid :
            

        Returns
        -------

        """

        self.pdgid = int(pdgid)
        self.charge = 0.0

        if self.pdgid in [11, 13, 15, -24]:
            self.charge = -1.0
        elif self.pdgid in [-11, -13, -15, 24]:
            self.charge = 1.0
        elif self.pdgid in [1, 3, 5]:
            self.charge = 2.0 / 3.0
        elif self.pdgid in [-1, -3, -5]:
            self.charge = -2.0 / 3.0
        elif self.pdgid in [2, 4, 6]:
            self.charge = -1.0 / 3.0
        elif self.pdgid in [-2, -4, -6]:
            self.charge = 1.0 / 3.0

    def __iadd__(self, other):
        super(MadMinerParticle, self).__iadd__(other)
        self.charge = None if self.charge is None or other.charge is None else self.charge + other.charge
        self.pdgid = None
        return self

    def __isub__(self, other):
        super(MadMinerParticle, self).__isub__(other)
        self.charge = None if self.charge is None or other.charge is None else self.charge - other.charge
        self.pdgid = None
        return self

    def __add__(self, other):
        vec = super(MadMinerParticle, self).__add__(other)
        vec.charge = None if self.charge is None or other.charge is None else self.charge + other.charge
        vec.pdgid = None
        return vec

    def __sub__(self, other):
        vec = super(MadMinerParticle, self).__sub__(other)
        vec.charge = None if self.charge is None or other.charge is None else self.charge - other.charge
        vec.pdgid = None
        return vec

    def boost(self, *args):
        vec = super(MadMinerParticle, self).boost(*args)

        particle = MadMinerParticle().from4vector(vec)
        particle.charge = self.charge
        particle.pdgid = self.pdgid

        return particle
