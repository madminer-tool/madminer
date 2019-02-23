from __future__ import absolute_import, division, print_function, unicode_literals

from skhep.math.vectors import LorentzVector
import logging

logger = logging.getLogger(__name__)


class MadMinerParticle(LorentzVector):
    """ """

    def __init__(self, *args, **kwargs):
        super(MadMinerParticle, self).__init__(*args, **kwargs)

        self.charge = None
        self.pdgid = None
        self.tau_tag = False
        self.b_tag = False
        self.t_tag = False

    def set_tags(self, tau_tag=False, b_tag=False, t_tag=False):
        self.tau_tag = tau_tag
        self.b_tag = b_tag
        self.t_tag = t_tag

    def set_pdgid(self, pdgid):
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

        if self.pdgid in [5, -5]:
            self.b_tag = True
        elif self.pdgid in [6, -6]:
            self.t_tag = True
        elif self.pdgid in [15, -15]:
            self.tau_tag = True

    def __iadd__(self, other):
        super(MadMinerParticle, self).__iadd__(other)
        self.charge = None if self.charge is None or other.charge is None else self.charge + other.charge
        self.pdgid = None
        self.tau_tag = self.tau_tag or other.tau_tag
        self.b_tag = self.b_tag or other.b_tag
        self.t_tag = self.t_tag or other.t_tag
        return self

    def __isub__(self, other):
        super(MadMinerParticle, self).__isub__(other)
        self.charge = None if self.charge is None or other.charge is None else self.charge - other.charge
        self.pdgid = None
        self.tau_tag = None
        self.b_tag = None
        self.t_tag = None
        self.tau_tag = self.tau_tag or other.tau_tag
        self.b_tag = self.b_tag or other.b_tag
        self.t_tag = self.t_tag or other.t_tag
        return self

    def __add__(self, other):
        vec = super(MadMinerParticle, self).__add__(other)
        vec.charge = None if self.charge is None or other.charge is None else self.charge + other.charge
        vec.pdgid = None
        vec.tau_tag = self.tau_tag or other.tau_tag
        vec.b_tag = self.b_tag or other.b_tag
        vec.t_tag = self.t_tag or other.t_tag
        return vec

    def __sub__(self, other):
        vec = super(MadMinerParticle, self).__sub__(other)
        vec.charge = None if self.charge is None or other.charge is None else self.charge - other.charge
        vec.pdgid = None
        vec.tau_tag = self.tau_tag or other.tau_tag
        vec.b_tag = self.b_tag or other.b_tag
        vec.t_tag = self.t_tag or other.t_tag
        return vec

    def boost(self, *args):
        vec = super(MadMinerParticle, self).boost(*args)

        particle = MadMinerParticle().from4vector(vec)
        particle.charge = self.charge
        particle.pdgid = self.pdgid
        particle.tau_tag = self.tau_tag
        particle.b_tag = self.b_tag
        particle.t_tag = self.t_tag

        return particle
