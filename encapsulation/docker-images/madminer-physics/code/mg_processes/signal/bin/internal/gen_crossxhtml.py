################################################################################
#
# Copyright (c) 2011 The MadGraph5_aMC@NLO Development team and Contributors
#
# This file is a part of the MadGraph5_aMC@NLO project, an application which 
# automatically generates Feynman diagrams and matrix elements for arbitrary
# high-energy processes in the Standard Model and beyond.
#
# It is subject to the MadGraph5_aMC@NLO license which should accompany this 
# distribution.
#
# For more information, visit madgraph.phys.ucl.ac.be and amcatnlo.web.cern.ch
#
################################################################################
""" Create gen_crossxhtml """


import os
import math
import re
import pickle
import re
import glob
import logging

try:
    import madgraph
except ImportError:
    import internal.files as files
    import internal.save_load_object as save_load_object
    import internal.lhe_parser as lhe_parser
    import internal.misc as misc
    import internal.banner as bannerlib
else:
    import madgraph.iolibs.files as files
    import madgraph.iolibs.save_load_object as save_load_object
    import madgraph.various.lhe_parser as lhe_parser
    import madgraph.various.misc as misc
    import madgraph.various.banner as bannerlib

pjoin = os.path.join
exists = os.path.exists
logger = logging.getLogger('madgraph.stdout') # -> stdout



crossxhtml_template = """
<HTML> 
<HEAD> 
    %(refresh)s 
    <META HTTP-EQUIV="EXPIRES" CONTENT="20" > 
    <TITLE>Online Event Generation</TITLE>
    <link rel=stylesheet href="./HTML/mgstyle.css" type="text/css">
</HEAD>
<BODY>
<script type="text/javascript">
function UrlExists(url) {
  var http = new XMLHttpRequest();
  http.open('HEAD', url, false);
  try{
     http.send()
     }
  catch(err){
   return 1==2;
  }
  return http.status!=404;
}
function check_link(url,alt, id){
    var obj = document.getElementById(id);
    if ( ! UrlExists(url)){
       if ( ! UrlExists(alt)){
         obj.href = url;
         return 1==1;
        }
       obj.href = alt;
       return 1 == 2;
    }
    obj.href = url;
    return 1==1;
}
</script>    
    <H2 align=center> Results in the %(model)s for %(process)s </H2> 
    <HR>
    %(status)s
    <br>
    <br>
    <H2 align="center"> Available Results </H2>
        <TABLE BORDER=2 align="center">  
            <TR align="center">
                <TH>Run</TH> 
                <TH>Collider</TH> 
                <TH> Banner </TH>
                <TH> %(numerical_title)s </TH> 
                <TH> Events  </TH>
                <TH> Data </TH>  
                <TH>Output</TH>
                <TH>Action</TH> 
            </TR>      
            %(old_run)s
        </TABLE>
    <H3 align=center><A HREF="./index.html"> Main Page </A></H3>
</BODY> 
</HTML> 
"""

status_template = """
<H2 ALIGN=CENTER> Currently Running %(run_mode_string)s</H2>
<TABLE BORDER=2 ALIGN=CENTER>
    <TR ALIGN=CENTER>
        <TH nowrap ROWSPAN=2 font color="#0000FF"> Run Name </TH>
        <TH nowrap ROWSPAN=2 font color="#0000FF"> Tag Name </TH>
        <TH nowrap ROWSPAN=2 font color="#0000FF"> Cards </TH>   
        <TH nowrap ROWSPAN=2 font color="#0000FF"> Results </TH> 
        <TH nowrap ROWSPAN=1 COLSPAN=3 font color="#0000FF"> Status/Jobs </TH>
    </TR>
        <TR> 
            <TH>   Queued </TH>
            <TH>  Running </TH>
            <TH> Done  </TH>
        </TR>
    <TR ALIGN=CENTER> 
        <TD nowrap ROWSPAN=2> %(run_name)s </TD>
        <TD nowrap ROWSPAN=2> %(tag_name)s </TD>
        <TD nowrap ROWSPAN=2> <a href="./Cards/param_card.dat">param_card</a><BR>
                    <a href="./Cards/run_card.dat">run_card</a><BR>
                    %(plot_card)s
                    %(pythia_card)s
                    %(pgs_card)s
                    %(delphes_card)s
                    %(shower_card)s
                    %(fo_analyse_card)s
        </TD>
        <TD nowrap ROWSPAN=2> %(results)s </TD> 
        %(status)s
 </TR>
 <TR></TR>
   %(stop_form)s
 </TABLE>
"""

class AllResults(dict):
    """Store the results for all the run of a given directory"""
    
    web = False 
    
    _run_entries = ['cross', 'error','nb_event_pythia','run_mode','run_statistics',
                    'nb_event','cross_pythia','error_pythia',
                    'nb_event_pythia8','cross_pythia8','error_pythia8']

    def __init__(self, model, process, path, recreateold=True):
        
        dict.__init__(self)
        self.order = []
        self.lastrun = None
        self.process = ', '.join(process)
        if len(self.process) > 60:
            pos = self.process[50:].find(',')
            if pos != -1:
                self.process = self.process[:50+pos] + ', ...'
        self.path = path
        self.model = model
        self.status = ''
        self.unit = 'pb'
        self.current = None
        
        # Check if some directory already exists and if so add them
        runs = [d for d in os.listdir(pjoin(path, 'Events')) if 
                                      os.path.isdir(pjoin(path, 'Events', d))]

        if runs:
            if recreateold:
                for run in runs:
                    self.readd_old_run(run)
                if self.order:
                    self.current = self[self.order[-1]]
            else:
                logger.warning("Previous runs exists but they will not be present in the html output.")
    
    def readd_old_run(self, run_name):
        """ re-create the data-base from scratch if the db was remove """
        
        event_path = pjoin(self.path, "Events", run_name, "unweighted_events.lhe")
        
        try:
            import internal
        except ImportError:
            import madgraph.various.banner as bannerlib
        else:
            import internal.banner as bannerlib
        
        if os.path.exists("%s.gz" % event_path):
            misc.gunzip(event_path, keep=True)
        if not os.path.exists(event_path):
            return
        banner = bannerlib.Banner(event_path)
        
        # load the information to add a new Run:
        run_card = banner.charge_card("run_card")
        process = banner.get_detail("proc_card", "generate")
        #create the new object
        run = RunResults(run_name, run_card, process, self.path)
        run.recreate(banner)
        self[run_name] = run
        self.order.append(run_name)
        
            
    def def_current(self, run, tag=None):
        """define the name of the current run
            The first argument can be a OneTagResults
        """

        if isinstance(run, OneTagResults):
            self.current = run
            self.lastrun = run['run_name']
            return
        
        assert run in self or run == None
        self.lastrun = run
        if run:
            if not tag:
                self.current = self[run][-1]
            else:
                assert tag in self[run].tags
                index = self[run].tags.index(tag)
                self.current = self[run][index]
                
        else:
            self.current = None
    
    def delete_run(self, run_name, tag=None):
        """delete a run from the database"""

        assert run_name in self

        if not tag :
            if self.current and self.current['run_name'] == run_name:
                self.def_current(None)                    
            del self[run_name]
            self.order.remove(run_name)
            if self.lastrun == run_name:
                self.lastrun = None
        else:
            assert tag in [a['tag'] for a in self[run_name]]
            RUN = self[run_name]
            if len(RUN) == 1:
                self.delete_run(run_name)
                return
            RUN.remove(tag)

        #update the html
        self.output()
    
    def def_web_mode(self, web):
        """define if we are in web mode or not """
        if web is True:
            try:
                web = os.environ['SERVER_NAME']
            except Exception:
                web = 'my_computer'
        self['web'] = web
        self.web = web
        
    def add_run(self, name, run_card, current=True):
        """ Adding a run to this directory"""
        
        tag = run_card['run_tag']
        if name in self.order:
            #self.order.remove(name) # Reorder the run to put this one at the end 
            if  tag in self[name].tags:
                if self[name].return_tag(tag).parton and len(self[name]) > 1:
                    #move the parton information before the removr
                    self[name].return_tag(self[name][1]['tag']).parton = \
                                               self[name].return_tag(tag).parton
                if len(self[name]) > 1:        
                    self[name].remove(tag) # Remove previous tag if define 
                    self[name].add(OneTagResults(name, run_card, self.path))
            else:
                #add the new tag run    
                self[name].add(OneTagResults(name, run_card, self.path))
            new = self[name] 
        else:
            new = RunResults(name, run_card, self.process, self.path)
            self[name] = new  
            self.order.append(name)
        
        if current:
            self.def_current(name)        
        if new.info['unit'] == 'GeV':
            self.unit = 'GeV'
            
    def update(self, status, level, makehtml=True, error=False):
        """update the current run status"""
        if self.current:
            self.current.update_status(level)
        self.status = status
        if self.current and self.current.debug  and self.status and not error:
            self.current.debug = None

        if makehtml:
            self.output()

    def resetall(self, main_path=None):
        """check the output status of all run
           main_path redefines the path associated to the run (allowing to move 
           the directory)
        """
        
        self.path = main_path
        
        for key,run in self.items():
            if key == 'web':
                continue
            for i,subrun in enumerate(run):
                self.def_current(subrun)
                self.clean()
                self.current.event_path = pjoin(main_path,'Events')
                self.current.me_dir = main_path 
                if i==0:
                    self.current.update_status()
                else:
                    self.current.update_status(nolevel='parton')
        self.output()
                    
    def clean(self, levels = ['all'], run=None, tag=None):
        """clean the run for the levels"""

        if not run and not self.current:
            return
        to_clean = self.current
        if run and not tag:
            for tagrun in self[run]:
                self.clean(levels, run, tagrun['tag'])
            return

        if run:
            to_clean = self[run].return_tag(tag)
        else:
            run = to_clean['run_name']
        
        if 'all' in levels:
            levels = ['parton', 'pythia', 'pgs', 'delphes', 'channel']
        
        if 'parton' in levels:
            to_clean.parton = []
        if 'pythia' in levels:
            to_clean.pythia = []
        if 'pgs' in levels:
            to_clean.pgs = []
        if 'delphes' in levels:
            to_clean.delphes = []
        
        
    def save(self):
        """Save the results of this directory in a pickle file"""
        filename = pjoin(self.path, 'HTML', 'results.pkl')
        save_load_object.save_to_file(filename, self)

    def add_detail(self, name, value, run=None, tag=None):
        """ add information to current run (cross/error/event)"""
        assert name in AllResults._run_entries

        if not run and not self.current:
            return

        if not run:
            run = self.current
        else:
            run = self[run].return_tag(tag)
            
        if name in ['cross_pythia']:
            run[name] = float(value)
        elif name in ['nb_event']:
            run[name] = int(value)
        elif name in ['nb_event_pythia']:
            run[name] = int(value)
        elif name in ['run_mode','run_statistics']:
            run[name] = value
        else:    
            run[name] = float(value)    
    
    def get_detail(self, name, run=None, tag=None):
        """ add information to current run (cross/error/event)"""
        assert name in AllResults._run_entries

        if not run and not self.current:
            return None

        if not run:
            run = self.current
        else:
            run = self[run].return_tag(tag)
            
        return run[name]
    
    def output(self):
        """ write the output file """
        
        # 1) Create the text for the status directory        
        if self.status and self.current:
            if isinstance(self.status, str):
                status = '<td ROWSPAN=2 colspan=4>%s</td>' %  self.status
            else:                
                s = list(self.status)
                if s[0] == '$events':
                    if self.current['nb_event']:
                        nevent = self.current['nb_event']
                    else:
                        nevent = self[self.current['run_name']][0]['nb_event']
                    if nevent:
                        s[0] = nevent - int(s[1]) -int(s[2])
                    else:
                        s[0] = ''
                status ='''<td> %s </td> <td> %s </td> <td> %s </td>
                </tr><tr><td colspan=3><center> %s </center></td>''' % (s[0],s[1], s[2], s[3])
                
            
            status_dict = {'status': status,
                            'cross': self.current['cross'],
                            'error': self.current['error'],
                            'run_name': self.current['run_name'],
                            'tag_name': self.current['tag'],
                            'unit': self[self.current['run_name']].info['unit']}
            # add the run_mode_string for amcatnlo_run
            if 'run_mode' in self.current.keys():
                run_mode_string = {'aMC@NLO': '(aMC@NLO)',
                                   'aMC@LO': '(aMC@LO)',
                                   'noshower': '(aMC@NLO)',
                                   'noshowerLO': '(aMC@LO)',
                                   'NLO': '(NLO f.o.)',
                                   'LO': '(LO f.o.)',
                                   'madevent':''
                                   }
                status_dict['run_mode_string'] = run_mode_string[self.current['run_mode']]
            else:
                status_dict['run_mode_string'] = ''


            if exists(pjoin(self.path, 'HTML',self.current['run_name'], 
                        'results.html')):
                status_dict['results'] = """<A HREF="./HTML/%(run_name)s/results.html">%(cross).4g <font face=symbol>&#177;</font> %(error).4g (%(unit)s)</A>""" % status_dict
            else:
                status_dict['results'] = "No results yet"
            if exists(pjoin(self.path, 'Cards', 'plot_card.dat')):
                status_dict['plot_card'] = """ <a href="./Cards/plot_card.dat">plot_card</a><BR>"""
            else:
                status_dict['plot_card'] = ""
            if exists(pjoin(self.path, 'Cards', 'pythia_card.dat')):
                status_dict['pythia_card'] = """ <a href="./Cards/pythia_card.dat">pythia_card</a><BR>"""
            else:
                status_dict['pythia_card'] = ""
            if exists(pjoin(self.path, 'Cards', 'pgs_card.dat')):
                status_dict['pgs_card'] = """ <a href="./Cards/pgs_card.dat">pgs_card</a><BR>"""
            else:
                status_dict['pgs_card'] = ""
            if exists(pjoin(self.path, 'Cards', 'delphes_card.dat')):
                status_dict['delphes_card'] = """ <a href="./Cards/delphes_card.dat">delphes_card</a><BR>"""
            else:
                status_dict['delphes_card'] = ""
            if exists(pjoin(self.path, 'Cards', 'shower_card.dat')):
                status_dict['shower_card'] = """ <a href="./Cards/shower_card.dat">shower_card</a><BR>"""
            else:
                status_dict['shower_card'] = ""
            if exists(pjoin(self.path, 'Cards', 'FO_analyse_card.dat')):
                status_dict['fo_analyse_card'] = """ <a href="./Cards/FO_analyse_card.dat">FO_analyse_card</a><BR>"""
            else:
                status_dict['fo_analyse_card'] = ""                

            if self.web:
                status_dict['stop_form'] = """
                 <TR ALIGN=CENTER><TD COLSPAN=7 text-align=center>
<FORM ACTION="http://%(web)s/cgi-bin/RunProcess/handle_runs-pl"  ENCTYPE="multipart/form-data" METHOD="POST">
<INPUT TYPE=HIDDEN NAME=directory VALUE="%(me_dir)s">
<INPUT TYPE=HIDDEN NAME=whattodo VALUE="stop_job">
<INPUT TYPE=SUBMIT VALUE="Stop Current Job">
</FORM></TD></TR>""" % {'me_dir': self.path, 'web': self.web}
            else:
                status_dict['stop_form'] = ""
            
            
            status = status_template % status_dict
            refresh = "<META HTTP-EQUIV=\"Refresh\" CONTENT=\"10\">"
        else:
            status =''
            refresh = ''
        
        
        # See if we need to incorporate the button for submission
        if os.path.exists(pjoin(self.path, 'RunWeb')):       
            running  = True
        else:
            running = False
        
        # 2) Create the text for the old run:
        old_run = ''
        for key in self.order:
            old_run += self[key].get_html(self.path, web=self.web, running=running)
        
        text_dict = {'process': self.process,
                     'model': self.model,
                     'status': status,
                     'old_run': old_run,
                     'refresh': refresh,
                     'numerical_title': self.unit == 'pb' and 'Cross section (pb)'\
                                                          or 'Width (GeV)'}
        
        text = crossxhtml_template % text_dict
        open(pjoin(self.path,'crossx.html'),'w').write(text)
        

class AllResultsNLO(AllResults):
    """Store the results for a NLO run of a given directory"""
    
    def __init__(self,model, process, path, recreateold=False):
        return AllResults.__init__(self, model, process, path, recreateold=recreateold)
       

class RunResults(list):
    """The list of all OneTagResults"""        

    def __init__(self, run_name, run_card, process, path):
        """initialize the object"""
        
        self.info = {'run_name': run_name,'me_dir':path}
        self.tags = [run_card['run_tag']]
        
        # Set the collider information
        data = process.split('>',1)[0].split()
        if len(data) == 2:
            name1,name2 = data
            if run_card['lpp1'] == -1:
                name1 = ' p~'
            elif run_card['lpp1']  == 1:
                name1 = ' p'   
            elif run_card['lpp1'] in [2,3]:
                name1 = ' a'
            if run_card['lpp2'] == -1:
                name2 = 'p~'
            elif run_card['lpp2']  == 1:
                name2 = ' p' 
            elif run_card['lpp2'] == [2,3]:
                name2 = ' a'                
            self.info['collider'] = '''%s %s <br> %s x %s  GeV''' % \
                    (name1, name2, run_card['ebeam1'], run_card['ebeam2'])
            self.info['unit'] = 'pb'                       
        elif len(data) == 1:
            self.info['collider'] = 'decay'
            self.info['unit'] = 'GeV'
        else:
            self.info['collider'] = 'special mode'
            self.info['unit'] = ''            
        
        self.append(OneTagResults(run_name, run_card, path))
        
    
    def get_html(self, output_path, **opt):
        """WRITE HTML OUTPUT"""

        try:
            self.web = opt['web']
            self.info['web'] = self.web
        except Exception:
            self.web = False

        # check if more than one parton output except for tags corresponding
        # to different MA5 parton-level runs.
        parton = [r for r in self if (r.parton and 'lhe' in r.parton)]
        # clean wrong previous run link
        if len(parton)>1:
            for p in parton[:-1]:
                # Do not remove the MA5 parton level results.
                for res in p.parton:
                    if not res.startswith('ma5'):
                        p.parton.remove(res)

        dico = self.info
        dico['run_span'] = sum([tag.get_nb_line() for tag in self], 1) -1
        dico['tag_data'] = '\n'.join([tag.get_html(self) for tag in self])
        text = """
        <tr>
        <td rowspan=%(run_span)s>%(run_name)s</td> 
        <td rowspan=%(run_span)s><center> %(collider)s </center></td>
        %(tag_data)s
        </tr>
        """ % dico

        if self.web:
            
            text = text % self.info

        return text

    
    def return_tag(self, name):
        
        for data in self:
            if data['tag'] == name:
                return data
        
        if name is None:
            # return last entry
            return self[-1]
        
        raise Exception, '%s is not a valid tag' % name
    
    def recreate(self, banner):
        """Fully recreate the information due to a hard removal of the db
        Work for LO ONLY!"""
        
        run_name = self.info["run_name"]
        run_card = banner.get("run_card")
        path = self.info["me_dir"]
        # Recover the main information (cross-section/number of event)
        informations = banner['mggenerationinfo']
        #number of events
        nb_event = re.search(r"Number\s*of\s*Events\s*:\s*(\d*)", informations)
        if nb_event:
            nb_event = int(nb_event.group(1))
        else:
            nb_event = 0
            
        # cross-section
        cross = re.search(r"Integrated\s*weight\s*\(\s*pb\s*\)\s*:\s*([\+\-\d.e]+)", informations,
                          re.I)
        if cross:
            cross = float(cross.group(1))
        else:
            cross = 0

        # search pythia file for tag: tag_1_pythia.log
        path = pjoin(self.info['me_dir'],'Events', self.info['run_name'])
        files = [pjoin(path, f) for f in os.listdir(path) if
                 os.path.isfile(pjoin(path,f)) and f.endswith('pythia.log')]
        #order them by creation date.
        files.sort(key=lambda x: os.path.getmtime(x))
        tags = [os.path.basename(name[:-11]) for name in files]

     
        # No pythia only a single run:
        if not tags:
            self[-1]['nb_event'] = nb_event
            self[-1]['cross'] = cross
          
        #Loop over pythia run
        for tag in tags:
            if tag not in self.tags:
                tagresult = OneTagResults(run_name, run_card, path)
                tagresult['tag'] = tag
                self.add(tagresult)
            else:
                tagresult = self.return_tag(tag)
            tagresult['nb_event'] = nb_event
            tagresult['cross'] = cross
            if run_card['ickkw'] != 0:
                #parse the file to have back the information
                pythia_log = misc.BackRead(pjoin(path, '%s_pythia.log' % tag))
                pythiare = re.compile("\s*I\s+0 All included subprocesses\s+I\s+(?P<generated>\d+)\s+(?P<tried>\d+)\s+I\s+(?P<xsec>[\d\.D\-+]+)\s+I")            
                for line in pythia_log:
                    info = pythiare.search(line)
                    if not info:
                        continue
                    try:
                        # Pythia cross section in mb, we want pb
                        sigma_m = float(info.group('xsec').replace('D','E')) *1e9
                        Nacc = int(info.group('generated'))
                    except ValueError:
                        # xsec is not float - this should not happen
                        tagresult['cross_pythia'] = 0
                        tagresult['nb_event_pythia'] = 0
                        tagresult['error_pythia'] = 0
                    else:
                        tagresult['cross_pythia'] = sigma_m
                        tagresult['nb_event_pythia'] = Nacc
                        tagresult['error_pythia'] = 0
                    break                 
                pythia_log.close()   

    
    def is_empty(self):
        """Check if this run contains smtg else than html information"""

        if not self:
            return True
        if len(self) > 1:
            return False
        
        data = self[0]
        if data.parton or data.pythia or data.pgs or data.delphes:
            return False
        else:
            return True
        
    def add(self, obj):
        """ """
        
        assert isinstance(obj, OneTagResults)
        tag = obj['tag']
        assert tag not in self.tags
        self.tags.append(tag)
        self.append(obj)
        
    def get_last_pythia(self):
        for i in range(1, len(self)+1):
            if self[-i].pythia or self[-i].pythia8:
                return self[-i]['tag']

    def get_current_info(self):
        
        output = {}
        current = self[-1]
        # Check that cross/nb_event/error are define
        if current.pythia and not current['nb_event'] and len(self) > 1:
            output['nb_event'] = self[-2]['nb_event']
            output['cross'] = self[-2]['cross']
            output['error'] = self[-2]['error']
        elif (current.pgs or current.delphes) and not current['nb_event'] and len(self) > 1:
            if self[-2]['cross_pythia'] and self[-2]['nb_event_pythia']:
                output['cross'] = self[-2]['cross_pythia']
                output['nb_event'] = self[-2]['nb_event_pythia']
                output['error'] = self[-2]['error_pythia']
            else:
                output['nb_event'] = self[-2]['nb_event']
                output['cross'] = self[-2]['cross']
                output['error'] = self[-2]['error']
        elif current['cross']:
            return current
        elif len(self) > 1:
            output['nb_event'] = self[-2]['nb_event']
            output['cross'] = self[-2]['cross']
            output['error'] = self[-2]['error']
        else:
            output['nb_event'] = 0
            output['cross'] = 0
            output['error'] = 1e-99             
        return output
        
        
    def remove(self, tag):
        
        assert tag in self.tags
        
        obj = [o for o in self if o['tag']==tag][0]
        self.tags.remove(tag)
        list.remove(self, obj)
    
    
        
class OneTagResults(dict):
    """ Store the results of a specific run """
    
    def __init__(self, run_name, run_card, path):
        """initialize the object"""
        
        # define at run_result
        self['run_name'] = run_name
        self['tag'] = run_card['run_tag']
        self['event_norm'] = run_card['event_norm']
        self.event_path = pjoin(path,'Events')
        self.me_dir = path
        self.debug = None
        
        # Default value
        self['nb_event'] = 0
        self['cross'] = 0
        self['cross_pythia'] = ''
        self['nb_event_pythia'] = 0
        self['error'] = 0
        self['run_mode'] = 'madevent'
        self.parton = []
        self.reweight = [] 
        self.pythia = []
        self.pythia8 = []
        self.madanalysis5_hadron = []
        # This is just a container that contain 'done' when the parton level MA5
        # analysis is done, so that set_run_name knows when to update the tag
        self.madanalysis5_parton = []        
        self.pgs = []
        self.delphes = []
        self.shower = []
        
        self.level_modes = ['parton', 'pythia', 'pythia8',
                            'pgs', 'delphes','reweight','shower',
                            'madanalysis5_hadron','madanalysis5_parton']
        # data 
        self.status = ''

        # Dictionary with (Pdir,G) as keys and sum_html.RunStatistics instances
        # as values
        self['run_statistics'] = {}
    
    
    def update_status(self, level='all', nolevel=[]):
        """update the status of the current run """
        exists = os.path.exists
        run = self['run_name']
        tag =self['tag']
        
        path = pjoin(self.event_path, run)
        html_path = pjoin(self.event_path, os.pardir, 'HTML', run)
        
        # Check if the output of the last status exists
        if level in ['gridpack','all']:
            if 'gridpack' not in self.parton and \
                    exists(pjoin(path,os.pardir ,os.pardir,"%s_gridpack.tar.gz" % run)):
                self.parton.append('gridpack')
        # Check if the output of the last status exists
        if level in ['reweight','all']:
            if 'plot' not in self.reweight and \
                         exists(pjoin(html_path,"plots_%s.html" % tag)):
                self.reweight.append('plot')

        # We also trigger parton for madanalysis5_parton because its results
        # must be added to self.parton
        if level in ['parton','all'] and 'parton' not in nolevel:
            
            if 'lhe' not in self.parton and \
                        (exists(pjoin(path,"unweighted_events.lhe.gz")) or
                         exists(pjoin(path,"unweighted_events.lhe")) or
                         exists(pjoin(path,"events.lhe.gz")) or
                         exists(pjoin(path,"events.lhe"))):
                self.parton.append('lhe')
        
            if 'root' not in self.parton and \
                          exists(pjoin(path,"unweighted_events.root")):
                self.parton.append('root')
            
            if 'plot' not in self.parton and \
                                      exists(pjoin(html_path,"plots_parton.html")):
                self.parton.append('plot')

            if 'param_card' not in self.parton and \
                                    exists(pjoin(path, "param_card.dat")):
                self.parton.append('param_card')
            
            if 'syst' not in self.parton and \
                                    exists(pjoin(path, "parton_systematics.log")):
                self.parton.append('syst')

            for kind in ['top','HwU','pdf','ps']:
                if misc.glob("*.%s" % kind, path):
                    if self['run_mode'] in ['LO', 'NLO']:
                        self.parton.append('%s' % kind)
            if exists(pjoin(path,'summary.txt')):
                self.parton.append('summary.txt')
                            

        if level in ['madanalysis5_parton','all'] and 'madanalysis5_parton' not in nolevel:

            if 'ma5_plot' not in self.parton and \
               misc.glob("%s_MA5_parton_analysis_*.pdf"%self['tag'], path):
                self.parton.append('ma5_plot')                

            if 'ma5_html' not in self.parton and \
               misc.glob(pjoin('%s_MA5_PARTON_ANALYSIS_*'%self['tag'],'Output','HTML','MadAnalysis5job_0','index.html'),html_path):
                self.parton.append('ma5_html')                
            
            if 'ma5_card' not in self.parton and \
                misc.glob(pjoin('%s_MA5_PARTON_ANALYSIS_*'%self['tag'],'history.ma5'),html_path):
                self.parton.append('ma5_card')

            if 'done' not in self.madanalysis5_parton and \
              any(res in self.parton for res in ['ma5_plot','ma5_html','ma5_card']):
                self.madanalysis5_parton.append('done')

        if level in ['madanalysis5_hadron','all'] and 'madanalysis5_hadron' not in nolevel:

            if 'ma5_plot' not in self.madanalysis5_hadron and \
              misc.glob(pjoin("%s_MA5_hadron_analysis_*.pdf"%self['tag']),path):
                self.madanalysis5_hadron.append('ma5_plot')                

            if 'ma5_html' not in self.madanalysis5_hadron and \
               misc.glob(pjoin('%s_MA5_HADRON_ANALYSIS_*'%self['tag'],'Output','HTML','MadAnalysis5job_0','index.html'),html_path):
                self.madanalysis5_hadron.append('ma5_html')                

            if 'ma5_cls' not in self.madanalysis5_hadron and \
                       os.path.isfile(pjoin(path,"%s_MA5_CLs.dat"%self['tag'])):
                self.madanalysis5_hadron.append('ma5_cls') 
     
            if 'ma5_card' not in self.madanalysis5_hadron and \
               misc.glob(pjoin('%s_MA5_PARTON_ANALYSIS_*'%self['tag'],'history.ma5'),html_path):
                self.madanalysis5_hadron.append('ma5_card')

        if level in ['shower','all'] and 'shower' not in nolevel \
          and self['run_mode'] != 'madevent':
            # this is for hep/top/HwU files from amcatnlo
            if misc.glob("*.hep", path) + \
               misc.glob("*.hep.gz", path):
                self.shower.append('hep')

            if 'plot' not in self.shower and \
                          exists(pjoin(html_path,"plots_shower_%s.html" % tag)):
                self.shower.append('plot')                

            if misc.glob("*.hepmc", path) + \
               misc.glob("*.hepmc.gz", path):
                self.shower.append('hepmc')

            for kind in ['top','HwU','pdf','ps']:
                if misc.glob('*.' + kind, path):
                    if self['run_mode'] in ['LO', 'NLO']:
                        self.parton.append('%s' % kind)
                    else:
                        self.shower.append('%s' % kind)
        if level in ['pythia', 'all']:
            
            
            # Do not include the lhe in the html anymore
            #if 'lhe' not in self.pythia and \
            #                (exists(pjoin(path,"%s_pythia_events.lhe.gz" % tag)) or
            #                 exists(pjoin(path,"%s_pythia_events.lhe" % tag))):
            #    self.pythia.append('lhe')


            if 'hep' not in self.pythia and \
                            (exists(pjoin(path,"%s_pythia_events.hep.gz" % tag)) or
                             exists(pjoin(path,"%s_pythia_events.hep" % tag))):
                self.pythia.append('hep')
            if 'log' not in self.pythia and \
                          exists(pjoin(path,"%s_pythia.log" % tag)):
                self.pythia.append('log')  

            # pointless to check the following if not hep output
            if 'hep' in self.pythia:
                if 'plot' not in self.pythia and \
                              exists(pjoin(html_path,"plots_pythia_%s.html" % tag)):
                    self.pythia.append('plot')
                
                if 'rwt' not in self.pythia and \
                                (exists(pjoin(path,"%s_syscalc.dat.gz" % tag)) or
                                 exists(pjoin(path,"%s_syscalc.dat" % tag))):
                    self.pythia.append('rwt')
                
                if 'root' not in self.pythia and \
                                  exists(pjoin(path,"%s_pythia_events.root" % tag)):
                    self.pythia.append('root')
                    
                #if 'lheroot' not in self.pythia and \
                #              exists(pjoin(path,"%s_pythia_lhe_events.root" % tag)):
                #    self.pythia.append('lheroot')
            
   


        if level in ['pythia8', 'all']:
            
            if 'hepmc' not in self.pythia8 and \
                            (exists(pjoin(path,"%s_pythia8_events.hepmc.gz" % tag)) or
                             exists(pjoin(path,"%s_pythia8_events.hepmc" % tag))):
                self.pythia8.append('hepmc')

            if 'log' not in self.pythia8 and \
                          exists(pjoin(path,"%s_pythia8.log" % tag)):
                self.pythia8.append('log') 
                
            if 'hepmc' in self.pythia8:
                if 'plot' not in self.pythia8 and 'hepmc' in self.pythia8 and \
                              exists(pjoin(html_path,"plots_pythia_%s.html" % tag)):
                    self.pythia8.append('plot')
                  
                if 'merged_xsec' not in self.pythia8 and \
                               exists(pjoin(path,"%s_merged_xsecs.txt" % tag)):  
                    self.pythia8.append('merged_xsec')
                    
                if 'djr_plot' not  in self.pythia8 and \
                              exists(pjoin(html_path,'%s_PY8_plots'%tag,'index.html')):
                    self.pythia8.append('djr_plot') 

        if level in ['pgs', 'all']:
            
            if 'plot' not in self.pgs and \
                         exists(pjoin(html_path,"plots_pgs_%s.html" % tag)):
                self.pgs.append('plot')
            
            if 'lhco' not in self.pgs and \
                              (exists(pjoin(path,"%s_pgs_events.lhco.gz" % tag)) or
                              exists(pjoin(path,"%s_pgs_events.lhco." % tag))):
                self.pgs.append('lhco')
                
            if 'root' not in self.pgs and \
                                 exists(pjoin(path,"%s_pgs_events.root" % tag)):
                self.pgs.append('root')
            
            if 'log' not in self.pgs and \
                          exists(pjoin(path,"%s_pgs.log" % tag)):
                self.pgs.append('log') 
    
        if level in ['delphes', 'all']:
            
            if 'plot' not in self.delphes and \
                              exists(pjoin(html_path,"plots_delphes_%s.html" % tag)):
                self.delphes.append('plot')
            
            if 'lhco' not in self.delphes and \
                 (exists(pjoin(path,"%s_delphes_events.lhco.gz" % tag)) or
                 exists(pjoin(path,"%s_delphes_events.lhco" % tag))):
                self.delphes.append('lhco')
                
            if 'root' not in self.delphes and \
                             exists(pjoin(path,"%s_delphes_events.root" % tag)):
                self.delphes.append('root')     
            
            if 'log' not in self.delphes and \
                          exists(pjoin(path,"%s_delphes.log" % tag)):
                self.delphes.append('log') 

        if level in ['madanlysis5_hadron','all']:
            pass
    
    def special_link(self, link, level, name):
        
        id = '%s_%s_%s_%s' % (self['run_name'],self['tag'], level, name)
        
        return " <a  id='%(id)s' href='%(link)s.gz' onClick=\"check_link('%(link)s.gz','%(link)s','%(id)s')\">%(name)s</a>" \
              % {'link': link, 'id': id, 'name':name}
    
    def double_link(self, link1, link2, name, id):
        
         return " <a  id='%(id)s' href='%(link1)s' onClick=\"check_link('%(link1)s','%(link2)s','%(id)s')\">%(name)s</a>" \
              % {'link1': link1, 'link2':link2, 'id': id, 'name':name}       
        
    def get_links(self, level):
        """ Get the links for a given level"""
        
        out = ''
        if level == 'parton':
            if 'gridpack' in self.parton:
                out += self.special_link("./%(run_name)s_gridpack.tar",
                                                    'gridpack', 'gridpack')
            if 'lhe' in self.parton:
                if exists(pjoin(self.me_dir, 'Events', self['run_name'], 'unweighted_events.lhe')) or\
                  exists(pjoin(self.me_dir, 'Events', self['run_name'], 'unweighted_events.lhe.gz')):
                    link = './Events/%(run_name)s/unweighted_events.lhe'
                elif exists(pjoin(self.me_dir, 'Events', self['run_name'], 'events.lhe')) or\
                  exists(pjoin(self.me_dir, 'Events', self['run_name'], 'events.lhe.gz')):
                    link = './Events/%(run_name)s/events.lhe'
                else:
                    link = None
                if link:
                    level = 'parton'
                    name = 'LHE'
                    out += self.special_link(link, level, name) 
            if 'root' in self.parton:
                out += ' <a href="./Events/%(run_name)s/unweighted_events.root">rootfile</a>'
            if 'plot' in self.parton:
                out += ' <a href="./HTML/%(run_name)s/plots_parton.html">plots</a>'
            if 'param_card' in self.parton:
                out += ' <a href="./Events/%(run_name)s/param_card.dat">param_card</a>'
            for kind in ['top', 'pdf', 'ps']:
                if kind in self.parton:
            # fixed order plots
                    for f in \
                        misc.glob('*.' + kind, pjoin(self.me_dir, 'Events', self['run_name'])):
                        out += " <a href=\"%s\">%s</a> " % (f, '%s' % kind.upper())
            
            if 'ma5_html' in self.parton:
                for result in misc.glob(pjoin('%s_MA5_PARTON_ANALYSIS_*'%self['tag']),
                                        pjoin(self.me_dir,'HTML',self['run_name'])):
                    target    = pjoin(os.curdir,os.path.relpath(result,self.me_dir),'Output','HTML','MadAnalysis5job_0','index.html')
                    link_name = os.path.basename(result).split('PARTON_ANALYSIS')[-1]
                    out += """ <a href="%s">MA5_report%s</a> """%(target, link_name)
                        
            if 'HwU' in self.parton:
            # fixed order plots
                for f in \
                  misc.glob('*.HwU', pjoin(self.me_dir, 'Events', self['run_name'])):
                    out += " <a href=\"%s\">%s</a> " % (f, 'HwU data')
                    out += " <a href=\"%s\">%s</a> " % \
                                           (f.replace('.HwU','.gnuplot'), 'GnuPlot')
            if 'summary.txt' in self.parton:
                out += ' <a href="./Events/%(run_name)s/summary.txt">summary</a>'

            #if 'rwt' in self.parton:
            #    out += ' <a href="./Events/%(run_name)s/%(tag)s_parton_syscalc.log">systematic variation</a>'

            return out % self
        
        if level == 'reweight':
            if 'plot' in self.reweight:
                out += ' <a href="./HTML/%(run_name)s/plots_%(tag)s.html">plots</a>'           
            return out % self

        if level == 'pythia':          
            if 'log' in self.pythia:
                out += """ <a href="./Events/%(run_name)s/%(tag)s_pythia.log">LOG</a>"""
            if 'hep' in self.pythia:
                link = './Events/%(run_name)s/%(tag)s_pythia_events.hep'
                level = 'pythia'
                name = 'STDHEP'
                out += self.special_link(link, level, name)  
                               
            if 'lhe' in self.pythia:
                link = './Events/%(run_name)s/%(tag)s_pythia_events.lhe'
                level = 'pythia'
                name = 'LHE'                
                out += self.special_link(link, level, name) 
            if 'root' in self.pythia:
                out += """ <a href="./Events/%(run_name)s/%(tag)s_pythia_events.root">rootfile (LHE)</a>"""
            if 'lheroot' in self.pythia:
                out += """ <a href="./Events/%(run_name)s/%(tag)s_pythia_lhe_events.root">rootfile (LHE)</a>"""
            if 'rwt' in self.pythia:
                link = './Events/%(run_name)s/%(tag)s_syscalc.dat'
                level = 'pythia'
                name = 'systematics'
                out += self.special_link(link, level, name)                 
            if 'plot' in self.pythia:
                out += ' <a href="./HTML/%(run_name)s/plots_pythia_%(tag)s.html">plots</a>'
            return out % self

        if level == 'pythia8':          
            if 'log' in self.pythia8:
                out += """ <a href="./Events/%(run_name)s/%(tag)s_pythia8.log">LOG</a>"""
            if 'hep' in self.pythia8:
                link = './Events/%(run_name)s/%(tag)s_pythia8_events.hep'
                level = 'pythia8'
                name = 'STDHEP'
                
            if 'hepmc' in self.pythia8:
                link = './Events/%(run_name)s/%(tag)s_pythia8_events.hepmc'
                level = 'pythia8'
                name = 'HEPMC'                                 
                out += self.special_link(link, level, name)
                #if 'merged_xsec' in self.pythia8:  
                #    out += """ <a href="./Events/%(run_name)s/%(tag)s_merged_xsecs.txt">merged xsection</a> """
            #if 'plot' in self.pythia8:
            #    out += ' <a href="./HTML/%(run_name)s/plots_pythia_%(tag)s.html">plots</a>'
            if 'djr_plot' in self.pythia8:
                out += ' <a href="./HTML/%(run_name)s/%(tag)s_PY8_plots/index.html">Matching plots</a>'                      
                
            return out % self

        if level == 'pgs':
            if 'log' in self.pgs:
                out += """ <a href="./Events/%(run_name)s/%(tag)s_pgs.log">LOG</a>"""
            if 'lhco' in self.pgs:
                link = './Events/%(run_name)s/%(tag)s_pgs_events.lhco'
                level = 'pgs'
                name = 'LHCO'                
                out += self.special_link(link, level, name)  
            if 'root' in self.pgs:
                out += """ <a href="./Events/%(run_name)s/%(tag)s_pgs_events.root">rootfile</a>"""    
            if 'plot' in self.pgs:
                out += """ <a href="./HTML/%(run_name)s/plots_pgs_%(tag)s.html">plots</a>"""
            return out % self
        
        if level == 'delphes':
            if 'log' in self.delphes:
                out += """ <a href="./Events/%(run_name)s/%(tag)s_delphes.log">LOG</a>"""
            if 'lhco' in self.delphes:
                link = './Events/%(run_name)s/%(tag)s_delphes_events.lhco'
                level = 'delphes'
                name = 'LHCO'                
                out += self.special_link(link, level, name)
            if 'root' in self.delphes:
                out += """ <a href="./Events/%(run_name)s/%(tag)s_delphes_events.root">rootfile</a>"""    
            if 'plot' in self.delphes:
                out += """ <a href="./HTML/%(run_name)s/plots_delphes_%(tag)s.html">plots</a>"""            
            return out % self

        if level == 'madanalysis5_hadron':
            if 'ma5_cls' in self.madanalysis5_hadron:
                out += """ <a href="./Events/%(run_name)s/%(tag)s_MA5_CLs.dat">Recasting_CLs</a>"""
            if 'ma5_html' in self.madanalysis5_hadron:
                # First link analysis results
                linked_analysis = False
                for result in misc.glob(pjoin('%s_MA5_HADRON_ANALYSIS_*'%self['tag']),
                                        pjoin(self.me_dir,'HTML',self['run_name'])):
                    target    = pjoin(os.curdir,os.path.relpath(result,self.me_dir),'Output','HTML','MadAnalysis5job_0','index.html')
                    link_name = os.path.basename(result).split('HADRON_ANALYSIS')[-1]
                    if link_name.startswith('_reco_'):
                        continue
                    # Also, do not put a link to the Recasting as it does not
                    # have an HTML yet
                    if link_name=='_Recasting':
                        continue
                    linked_analysis = True
                    out += """ <a href="%s">%s</a> """%(target, link_name.strip('_'))
    
                # Also link reco results if no analysis was found
                if not linked_analysis:
                    for result in misc.glob(pjoin('%s_MA5_HADRON_ANALYSIS_*'%self['tag']),
                                            pjoin(self.me_dir,'HTML',self['run_name'])):
                        target    = pjoin(os.curdir,os.path.relpath(
                                        result,self.me_dir),'Output','HTML','MadAnalysis5job_0','index.html')
                        link_name = os.path.basename(result).split('HADRON_ANALYSIS')[-1]
                        if not link_name.startswith('_reco_'):
                            continue
                        out += """ <a href="%s">%s</a> """%(target, link_name.strip('_'))

            return out % self        

        if level == 'shower':
        # this is to add the link to the results after shower for amcatnlo
            for kind in ['hep', 'hepmc', 'top', 'HwU', 'pdf', 'ps']:
                if kind in self.shower:
                    for f in \
                      misc.glob('*.' + kind, pjoin(self.me_dir, 'Events', self['run_name'])) + \
                      misc.glob('*.%s.gz' % kind, pjoin(self.me_dir, 'Events', self['run_name'])):
                        if kind == 'HwU':
                            out += " <a href=\"%s\">%s</a> " % (f, 'HwU data')
                            out += " <a href=\"%s\">%s</a> " % (f.replace('.HwU','.gnuplot'), 'GnuPlot')
                        else:
                            out += " <a href=\"%s\">%s</a> " % (f, kind.upper())

            if 'plot' in self.shower:
                out += """ <a href="./HTML/%(run_name)s/plots_shower_%(tag)s.html">plots</a>"""
            
            return out % self
                
    
    
    def get_action(self, ttype, local_dico, runresults):
        # Fill the actions
        if ttype == 'parton':
            if runresults.web:
                action = """
<FORM ACTION="http://%(web)s/cgi-bin/RunProcess/handle_runs-pl"  ENCTYPE="multipart/form-data" METHOD="POST">
<INPUT TYPE=HIDDEN NAME=directory VALUE="%(me_dir)s">
<INPUT TYPE=HIDDEN NAME=whattodo VALUE="remove_level">
<INPUT TYPE=HIDDEN NAME=level VALUE="all">
<INPUT TYPE=HIDDEN NAME=tag VALUE=\"""" + self['tag'] + """\">
<INPUT TYPE=HIDDEN NAME=run VALUE="%(run_name)s">
<INPUT TYPE=SUBMIT VALUE="Remove run">
</FORM>
                    
<FORM ACTION="http://%(web)s/cgi-bin/RunProcess/handle_runs-pl"  ENCTYPE="multipart/form-data" METHOD="POST">
<INPUT TYPE=HIDDEN NAME=directory VALUE="%(me_dir)s">
<INPUT TYPE=HIDDEN NAME=whattodo VALUE="pythia">
<INPUT TYPE=HIDDEN NAME=run VALUE="%(run_name)s">
<INPUT TYPE=SUBMIT VALUE="Run Pythia">
</FORM>"""
            else:
                action = self.command_suggestion_html('remove %s parton --tag=%s' \
                                                                       % (self['run_name'], self['tag']))
                # this the detector simulation and pythia should be available only for madevent
                if self['run_mode'] == 'madevent':
                    action += self.command_suggestion_html('pythia %s ' % self['run_name'])
                else: 
                    pass

        elif ttype == 'shower':
            if runresults.web:
                action = """
<FORM ACTION="http://%(web)s/cgi-bin/RunProcess/handle_runs-pl"  ENCTYPE="multipart/form-data" METHOD="POST">
<INPUT TYPE=HIDDEN NAME=directory VALUE="%(me_dir)s">
<INPUT TYPE=HIDDEN NAME=whattodo VALUE="remove_level">
<INPUT TYPE=HIDDEN NAME=level VALUE="all">
<INPUT TYPE=HIDDEN NAME=tag VALUE=\"""" + self['tag'] + """\">
<INPUT TYPE=HIDDEN NAME=run VALUE="%(run_name)s">
<INPUT TYPE=SUBMIT VALUE="Remove run">
</FORM>
                
<FORM ACTION="http://%(web)s/cgi-bin/RunProcess/handle_runs-pl"  ENCTYPE="multipart/form-data" METHOD="POST">
<INPUT TYPE=HIDDEN NAME=directory VALUE="%(me_dir)s">
<INPUT TYPE=HIDDEN NAME=whattodo VALUE="pythia">
<INPUT TYPE=HIDDEN NAME=run VALUE="%(run_name)s">
<INPUT TYPE=SUBMIT VALUE="Run Pythia">
</FORM>"""
            else:
                action = self.command_suggestion_html('remove %s parton --tag=%s' \
                                                                   % (self['run_name'], self['tag']))
                # this the detector simulation and pythia should be available only for madevent
                if self['run_mode'] == 'madevent':
                    action += self.command_suggestion_html('pythia %s ' % self['run_name'])
                else: 
                    pass

        elif ttype in ['pythia', 'pythia8']:
            if self['tag'] == runresults.get_last_pythia():
                if runresults.web:
                    action = """
<FORM ACTION="http://%(web)s/cgi-bin/RunProcess/handle_runs-pl"  ENCTYPE="multipart/form-data" METHOD="POST">
<INPUT TYPE=HIDDEN NAME=directory VALUE="%(me_dir)s">
<INPUT TYPE=HIDDEN NAME=whattodo VALUE="remove_level">
<INPUT TYPE=HIDDEN NAME=level VALUE="pythia">
<INPUT TYPE=HIDDEN NAME=run VALUE="%(run_name)s">
<INPUT TYPE=HIDDEN NAME=tag VALUE=\"""" + self['tag'] + """\">
<INPUT TYPE=SUBMIT VALUE="Remove pythia">
</FORM>

<FORM ACTION="http://%(web)s/cgi-bin/RunProcess/handle_runs-pl"  ENCTYPE="multipart/form-data" METHOD="POST">
<INPUT TYPE=HIDDEN NAME=directory VALUE="%(me_dir)s">
<INPUT TYPE=HIDDEN NAME=whattodo VALUE="pgs">
<INPUT TYPE=HIDDEN NAME=run VALUE="%(run_name)s">
<INPUT TYPE=SUBMIT VALUE="Run Detector">
</FORM>"""
                else:
                    action = self.command_suggestion_html(
                                            'remove %s pythia --tag=%s' % \
                                            (self['run_name'], self['tag']))
                    action += self.command_suggestion_html(
                     'delphes %(1)s' % {'1': self['run_name']})
            else:
                if runresults.web:
                    action = ''
                else:
                    action = self.command_suggestion_html('remove %s  pythia --tag=%s'\
                                                                        % (self['run_name'], self['tag']))
        elif ttype in ['madanalysis5_hadron']:
            # For now, nothing special needs to be done since we don't
            # support actions for madanalysis5.
            action = ''
            
        else:
            if runresults.web:
                action = """
<FORM ACTION="http://%(web)s/cgi-bin/RunProcess/handle_runs-pl"  ENCTYPE="multipart/form-data" METHOD="POST">
<INPUT TYPE=HIDDEN NAME=directory VALUE="%(me_dir)s">
<INPUT TYPE=HIDDEN NAME=whattodo VALUE="remove_level">
<INPUT TYPE=HIDDEN NAME=level VALUE=\"""" + str(type) + """\">
<INPUT TYPE=HIDDEN NAME=tag VALUE=\"""" + self['tag'] + """\">
<INPUT TYPE=HIDDEN NAME=run VALUE="%(run_name)s">
<INPUT TYPE=SUBMIT VALUE="Remove """ + str(ttype) + """\">
</FORM>"""
            else:
                action = self.command_suggestion_html('remove %s %s --tag=%s' %\
                                                          (self['run_name'], ttype, self['tag']))
        return action


    def get_nb_line(self):
        
        nb_line = 0
        for key in ['parton', 'reweight', 'pythia', 'pythia8', 'pgs', 
                    'delphes', 'shower', 'madanalysis5_hadron']:
            if len(getattr(self, key)):
                nb_line += 1
        return max([nb_line,1])
    
    
    def get_html(self, runresults):
        """create the html output linked to the this tag
           RunResults is given in case of cross-section need to be taken
           from a previous run
        """
        
        tag_template = """
        <td rowspan=%(tag_span)s> <a href="./Events/%(run)s/%(run)s_%(tag)s_banner.txt">%(tag)s</a>%(debug)s</td>
        %(subruns)s"""
        
        # Compute the text for eachsubpart
        
        sub_part_template_parton = """
        <td rowspan=%(cross_span)s><center><a href="./HTML/%(run)s/results.html"> %(cross).4g <font face=symbol>&#177;</font> %(err).2g %(bias)s</a> %(syst)s </center></td>
        <td rowspan=%(cross_span)s><center> %(nb_event)s<center></td><td> %(type)s </td>
        <td> %(links)s</td>
        <td> %(action)s</td>
        </tr>"""

        sub_part_template_py8 = """
        <td rowspan=%(cross_span)s><center><a href="./Events/%(run)s/%(tag)s_merged_xsecs.txt"> merged xsection</a> %(syst)s </center></td>
        <td rowspan=%(cross_span)s><center> %(nb_event)s<center></td><td> %(type)s </td>
        <td> %(links)s</td>
        <td> %(action)s</td>
        </tr>"""
        
        sub_part_template_reweight = """
        <td rowspan=%(cross_span)s><center> %(cross).4g </center></td>
        <td rowspan=%(cross_span)s><center> %(nb_event)s<center></td><td> %(type)s </td>
        <td> %(links)s</td>
        <td> %(action)s</td>
        </tr>"""        
        
        sub_part_template_pgs = """
        <td> %(type)s </td>
        <td> %(links)s</td>
        <td> %(action)s</td> 
        </tr>"""        

        sub_part_template_shower = """
        <td> %(type)s %(run_mode)s </td>
        <td> %(links)s</td>
        <td> %(action)s</td>
        </tr>"""
        
        # Compute the HTMl output for subpart
        nb_line = self.get_nb_line()
        # Check that cross/nb_event/error are define
        if self.pythia and not self['nb_event']:
            try:
                self['nb_event'] = runresults[-2]['nb_event']
                self['cross'] = runresults[-2]['cross']
                self['error'] = runresults[-2]['error']
            except Exception:
                pass
        elif self.pythia8 and not self['nb_event']:
            try:
                self['nb_event'] = runresults[-2]['nb_event']
                self['cross'] = runresults[-2]['cross']
                self['error'] = runresults[-2]['error']
            except Exception:
                pass
                
        elif (self.pgs or self.delphes) and not self['nb_event'] and \
             len(runresults) > 1:
            if runresults[-2]['cross_pythia'] and runresults[-2]['cross']:
                self['cross'] = runresults[-2]['cross_pythia']
                self['error'] = runresults[-2]['error_pythia']
                self['nb_event'] = runresults[-2]['nb_event_pythia']                           
            else:
                self['nb_event'] = runresults[-2]['nb_event']
                self['cross'] = runresults[-2]['cross']
                self['error'] = runresults[-2]['error']

        
        first = None
        subresults_html = ''
        for ttype in self.level_modes:
            data = getattr(self, ttype)            
            if not data:
                continue
            
            if ttype == 'madanalysis5_parton':
                # The 'done' store in madanalysis5_parton is just a placeholder
                # it doesn't have a corresponding line
                continue
            local_dico = {'type': ttype, 'run': self['run_name'], 'syst': '',
                          'tag': self['tag']}
            if self['event_norm'].lower()=='bias':
                local_dico['bias']='(biased, do not use)'
            else:
                local_dico['bias']=''

            if 'run_mode' in self.keys():
                local_dico['run_mode'] = self['run_mode']
            else:
                local_dico['run_mode'] = ""
            if not first:
                if ttype == 'reweight':
                    template = sub_part_template_reweight
                elif ttype=='pythia8' and self['cross_pythia'] == -1:
                    template = sub_part_template_py8
                else:
                    template = sub_part_template_parton
                first = ttype
                if ttype=='parton' and self['cross_pythia']:
                    local_dico['cross_span'] = 1
                    local_dico['cross'] = self['cross']
                    local_dico['err'] = self['error']
                    local_dico['nb_event'] = self['nb_event']
                    if 'syst' in self.parton:
                        local_dico['syst'] = '<font face=symbol>&#177;</font> <a href="./Events/%(run_name)s/parton_systematics.log">systematics</a>' \
                                             % {'run_name':self['run_name']}
                
                elif self['cross_pythia']:
                    if self.parton:
                        local_dico['cross_span'] = nb_line -1
                    else:
                        local_dico['cross_span'] = nb_line
                    if self['nb_event_pythia']:
                        local_dico['nb_event'] = self['nb_event_pythia']
                    else:
                        local_dico['nb_event'] = 0
                    local_dico['cross'] = self['cross_pythia']
                    local_dico['err'] = self['error_pythia']
                    if 'rwt' in self.pythia:
                        local_dico['syst'] = '<font face=symbol>&#177;</font> <a href="./Events/%(run_name)s/%(tag)s_Pythia_syscalc.log">systematics</a>' \
                                             % {'run_name':self['run_name'], 'tag': self['tag']}
                else:
                    if 'lhe' not in self.parton and self.madanalysis5_parton:
                        local_dico['type'] += ' MA5'
                    elif ttype=='madanalysis5_hadron' and self.madanalysis5_hadron:
                        local_dico['type'] = 'hadron MA5'
                    else:
                        local_dico['type'] += ' %s' % self['run_mode']

                    local_dico['cross_span'] = nb_line
                    local_dico['cross'] = self['cross']
                    local_dico['err'] = self['error']
                    local_dico['nb_event'] = self['nb_event']
                    if 'syst' in self.parton:
                        local_dico['syst'] = '<font face=symbol>&#177;</font> <a href="./Events/%(run_name)s/parton_systematics.log">systematics</a>' \
                                             % {'run_name':self['run_name'], 'tag': self['tag']}
            elif ttype == 'pythia8' and self['cross_pythia'] ==-1 and 'merged_xsec' in self.pythia8:
                template = sub_part_template_py8
                if self.parton:           
                    local_dico['cross_span'] = nb_line - 1
                    local_dico['nb_event'] = self['nb_event_pythia']
                else:
                    local_dico['cross_span'] = nb_line
                    local_dico['nb_event'] = self['nb_event_pythia']                
            elif ttype in ['pythia','pythia8'] and self['cross_pythia']:
                template = sub_part_template_parton
                if self.parton:           
                    local_dico['cross_span'] = nb_line - 1
                    if self['nb_event_pythia']:
                        local_dico['nb_event'] = self['nb_event_pythia']
                    else:
                        local_dico['nb_event'] = 0
                else:
                    local_dico['cross_span'] = nb_line
                    local_dico['nb_event'] = self['nb_event']
                if 'rwt' in self.pythia:
                    local_dico['syst'] = '<font face=symbol>&#177;</font> <a href="./Events/%(run_name)s/%(tag)s_Pythia_syscalc.log">systematics</a>' \
                                             % {'run_name':self['run_name'], 'tag': self['tag']}
                local_dico['cross'] = self['cross_pythia']
                local_dico['err'] = self['error_pythia']

            elif ttype in ['madanalysis5_hadron']:
                # We can use the same template as pgs here
                template = sub_part_template_pgs
                local_dico['type'] = 'hadron MA5'
                # Nothing else needs to be done for now, since only type and
                # run_mode must be defined in local_dict and this has already
                # been done.

            elif ttype == 'shower':
                template = sub_part_template_shower
                if self.parton:           
                    local_dico['cross_span'] = nb_line - 1
                else:
                    local_dico['cross_span'] = nb_line
            else:
                template = sub_part_template_pgs             
            
            # Fill the links/actions
            local_dico['links'] = self.get_links(ttype)
            local_dico['action'] = self.get_action(ttype, local_dico, runresults)
            # create the text
            subresults_html += template % local_dico
            
        if subresults_html == '':
            if runresults.web:
                    action = """
<FORM ACTION="http://%(web)s/cgi-bin/RunProcess/handle_runs-pl"  ENCTYPE="multipart/form-data" METHOD="POST">
<INPUT TYPE=HIDDEN NAME=directory VALUE="%(me_dir)s">
<INPUT TYPE=HIDDEN NAME=whattodo VALUE="remove_level">
<INPUT TYPE=HIDDEN NAME=level VALUE="banner">
<INPUT TYPE=HIDDEN NAME=tag VALUE=\"""" + self['tag'] + """\">
<INPUT TYPE=HIDDEN NAME=run VALUE="%(run_name)s">
<INPUT TYPE=SUBMIT VALUE="Remove Banner">
</FORM>
                    
<FORM ACTION="http://%(web)s/cgi-bin/RunProcess/handle_runs-pl"  ENCTYPE="multipart/form-data" METHOD="POST">
<INPUT TYPE=HIDDEN NAME=directory VALUE="%(me_dir)s">
<INPUT TYPE=HIDDEN NAME=whattodo VALUE="banner">
<INPUT TYPE=HIDDEN NAME=run VALUE="%(run_name)s">
<INPUT TYPE=SUBMIT VALUE="Run the banner">
</FORM>"""
            else:
                    action = self.command_suggestion_html('remove %s banner --tag=%s' \
                                                                       % (self['run_name'], self['tag']))
                    action += self.command_suggestion_html('banner_run %s ' % self['run_name'])
            
            
            
            subresults_html = sub_part_template_parton % \
                          {'type': '', 
                           'run': self['run_name'],
                           'cross_span': 1,
                           'cross': self['cross'],
                           'err': self['error'],
                           'nb_event': self['nb_event'] and self['nb_event'] or 'No events yet',
                           'links': 'banner only',
                           'action': action,
                           'run_mode': '',
                           'syst':'',
                           'bias':''
                           }                                
                                  
        if self.debug is KeyboardInterrupt:
            debug = '<br><font color=red>Interrupted</font>'
        elif isinstance(self.debug, basestring):
            if not os.path.isabs(self.debug) and not self.debug.startswith('./'):
                self.debug = './' + self.debug
            elif os.path.isabs(self.debug):
                self.debug = os.path.relpath(self.debug, self.me_dir)
            debug = '<br> <a href=\'%s\'> <font color=red>ERROR</font></a>' \
                                               % (self.debug)
        elif self.debug:
            text = str(self.debug).replace('. ','.<br>')
            if 'http' in text:
                pat = re.compile('(http[\S]*)')
                text = pat.sub(r'<a href=\1> here </a>', text)
            debug = '<br><font color=red>%s<BR>%s</font>' % \
                                           (self.debug.__class__.__name__, text)
        else:
            debug = ''                                       
        text = tag_template % {'tag_span': nb_line,
                           'run': self['run_name'], 'tag': self['tag'],
                           'subruns' : subresults_html,
                           'debug':debug}

        return text
        

    def command_suggestion_html(self, command):
        """return html button with code suggestion"""
        
        if command.startswith('pythia'):
            button = 'launch pythia'
        if command.startswith('shower'):
            button = 'shower events'
        elif command.startswith('remove banner'):
            button = 'remove banner'
        elif command.startswith('remove'):
            button = 'remove run'
        elif command.startswith('banner_run'):
            button = 're-run from the banner'
        else:
            button = 'launch detector simulation'
        if self['run_mode'] == 'madevent':
            header = 'Launch ./bin/madevent in a shell, and run the following command: '
        else:
            header = 'Launch ./bin/aMCatNLO in a shell, and run the following command: '

        return "<INPUT TYPE=SUBMIT VALUE='%s' onClick=\"alert('%s')\">" % (button, header + command)


        return  + '<br>'




