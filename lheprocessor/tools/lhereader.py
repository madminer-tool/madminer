import xml.etree.ElementTree as ET
import skhep.math

class Particle:
    def __init__(self,pdgid,spin,px=0,py=0,pz=0,energy=0,mass=0):
        self.pdgid=pdgid
        self.px=px
        self.py=py
        self.pz=pz
        self.energy=energy
        self.mass=mass
        self.spin=spin    
    
    @property
    def p4(self):
        vec = skhep.math.vectors.LorentzVector()
        vec.setpxpypze(self.px,self.py,self.pz,self.energy)
        return vec
    
    @p4.setter
    def p4(self,value):
        self.px=value.Px()
        self.py=value.Py()
        self.pz=value.Pz()
        self.energy=value.E()
        self.mass=value.M()
    
    @property
    def p(self):
        return self.p4.P()
    
    @property
    def eta(self):
        return self.p4.Eta()
    
    @property
    def pt(self):
        return self.p4.Pt()
    
    
class Event:
    def __init__(self,num_particles):
        self.num_particles=num_particles
        self.particles=[]
    
    def __addParticle__(self,particle):
        self.particles.append(particle)
        
    def getParticlesByIDs(self,idlist):
        partlist=[]
        for pdgid in idlist:
            for p in self.particles:
                if p.pdgid==pdgid:
                    partlist.append(p)
        return partlist

class LHEFData:
    def __init__(self,version):
        self.version=version
        self.events=[]
    
    def __addEvent__(self,event):
        self.events.append(event)
        
    def getParticlesByIDs(self,idlist):
        partlist=[]
        for event in self.events:
            partlist.extend(event.getParticlesByIDs(idlist))
        return partlist
        

def readLHEF(name):
    tree = ET.parse(name)
    root=tree.getroot()
    lhefdata=LHEFData(float(root.attrib['version']))
    for child in root:
        if(child.tag=='event'):
            lines=child.text.strip().split('\n')
            event_header=lines[0].strip()
            num_part=int(event_header.split()[0].strip())
            e=Event(num_part)
            for i in range(1,num_part+1):
                part_data=lines[i].strip().split()
                p=Particle(int(part_data[0]), float(part_data[12]), float(part_data[6]), float(part_data[7]), float(part_data[8]), float(part_data[9]), float(part_data[10]))
                e.__addParticle__(p)
            lhefdata.__addEvent__(e)
    
    return lhefdata
