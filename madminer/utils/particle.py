import logging
import vector

from particle import Particle

logger = logging.getLogger(__name__)


B_PDG_IDS = {int(p.pdgid) for p in Particle.findall(pdg_name="b")}
T_PDG_IDS = {int(p.pdgid) for p in Particle.findall(pdg_name="t")}
TAU_PDG_IDS = {int(p.pdgid) for p in Particle.findall(pdg_name="tau")}


class MadMinerParticle(vector.MomentumObject4D):
    """ """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.charge = None
        self.pdgid = None
        self.spin = None
        self.tau_tag = False
        self.b_tag = False
        self.t_tag = False

    def set_tags(self, tau_tag=False, b_tag=False, t_tag=False):
        self.tau_tag = tau_tag
        self.b_tag = b_tag
        self.t_tag = t_tag

    def set_pdgid(self, pdgid):
        self.pdgid = int(pdgid)

        try:
            self.charge = Particle.from_pdgid(self.pdgid).charge
        except RuntimeError:
            self.charge = 0.0

        if self.pdgid in B_PDG_IDS:
            self.b_tag = True
        elif self.pdgid in T_PDG_IDS:
            self.t_tag = True
        elif self.pdgid in TAU_PDG_IDS:
            self.tau_tag = True

    def set_spin(self, spin):
        self.spin = spin

    def __iadd__(self, other):
        assert isinstance(other, self.__class__)
        super().__iadd__(other)
        self.charge = None if self.charge is None or other.charge is None else self.charge + other.charge
        self.pdgid = None
        self.spin = None
        self.tau_tag = self.tau_tag or other.tau_tag
        self.b_tag = self.b_tag or other.b_tag
        self.t_tag = self.t_tag or other.t_tag
        return self

    def __isub__(self, other):
        assert isinstance(other, self.__class__)
        super().__isub__(other)
        self.charge = None if self.charge is None or other.charge is None else self.charge - other.charge
        self.pdgid = None
        self.spin = None
        self.tau_tag = None
        self.b_tag = None
        self.t_tag = None
        self.tau_tag = self.tau_tag or other.tau_tag
        self.b_tag = self.b_tag or other.b_tag
        self.t_tag = self.t_tag or other.t_tag
        return self

    def __add__(self, other):
        assert isinstance(other, self.__class__)
        vec = super().__add__(other)
        vec.charge = None if self.charge is None or other.charge is None else self.charge + other.charge
        vec.pdgid = None
        vec.spin = None
        vec.tau_tag = self.tau_tag or other.tau_tag
        vec.b_tag = self.b_tag or other.b_tag
        vec.t_tag = self.t_tag or other.t_tag
        return vec

    def __sub__(self, other):
        assert isinstance(other, self.__class__)
        vec = super().__sub__(other)
        vec.charge = None if self.charge is None or other.charge is None else self.charge - other.charge
        vec.pdgid = None
        vec.spin = None
        vec.tau_tag = self.tau_tag or other.tau_tag
        vec.b_tag = self.b_tag or other.b_tag
        vec.t_tag = self.t_tag or other.t_tag
        return vec

    def boost(self, *args):
        vec = super().boost(*args)

        particle = MadMinerParticle.from_xyzt(vec.x, vec.y, vec.z, vec.t)
        particle.charge = self.charge
        particle.spin = self.spin
        particle.pdgid = self.pdgid
        particle.tau_tag = self.tau_tag
        particle.b_tag = self.b_tag
        particle.t_tag = self.t_tag

        return particle
