import flybrainlab as fbl
import netpyne
from netpyne.specs import netParams, SimConfig
from netpyne import conversion, sim
import networkx as nx
import numpy as np
import os
import os.path
import tempfile
import typing as tp

def model_gen(client: fbl.Client,
              res: fbl.graph.NAqueryResult,
              filename: str,
              custom_mechs: tp.Dict[str, tp.Dict]=None,
              custom_cells: tp.Dict[tp.List[str], tp.Dict]=None,
              default_mech: tp.Dict=None,
              default_cell: tp.Dict=None,
              stim_sources: tp.Dict[str, tp.Dict]=None,
              stim_targets: tp.Dict[str, tp.Dict]=None,
              record_names: tp.List=None,
              sim_duration: int=1*1e3,
              dt: int=0.05,
              maintain_morphology: bool=False) -> tp.Tuple:
    ''' Generates a netpyne model and simulation parameters from a neuroNLP graph
    
    .. note::
        
        Only verified to work with Medulla queries / clients.
        
        See the NetPyNE package reference for more information on defining network components
        (http://netpyne.org/reference.html#function-string)
        
        Function strings can be defined using the following elements:
            - Numerical values
            - All python mathematical operators and functions
            - NEURON h.random() methods
              (https://www.neuron.yale.edu/neuron/static/py_doc/programming/math/random.html)
            - Single-valued numerical network parameters defined in the accompanying netparams
              dictionary.
            - Contextual variables such as cell location, segment position, distance between
              cells, etc. These are predefined and specific to where the function is used.
              (http://netpyne.org/reference.html#function-string)
    
    :param client: pointer to FBL client
    :param res: FBL NAqueryResult object, representing the system to be converted to netpyne
    :param filename: name of the file to save sim outputs to
    :param custom_mechs: dictionary of all custom synaptic mechanisms to be used in the simulation.
                         It should be noted that since all custom mechanisms are defined here
                         irrespective of type (synapses, cell mechs, etc.), this won't check
                         whether or not any defined mechanisms are actually valid for their
                         purpose.
    :param custom_cells: dictionary of cell names and custom cell definitions to assign to them.
    :param default_mech: default synaptic mechanism to be used for unassigned synaptic connections.
    :param default_cell: default cell model to be used for unassigned cells.
    :param stim_sources: dictionary of stimulation source names, and their accompanying parameters.
                         It is assumed that custom mechanisms are defined in custom_mechs.
    :param stim_targets: dictionary of stimulation targets, and their accompanying parameters.
                         It is assumed that any custom stimulation sources are defined in
                         stim_sources.
    :param record_names: list of neurons to record traces of
    :param sim_duration: duration of the simulation, in ms
    :param dt: internal integration timestep to use
    :param maintain_morphology: whether or not model morphology should be maintained in the
                                simulation (not recommended if neuron sections are unidentified
                                in swc definitions)
    '''
    
    # Define helper variables for easy access later
    G = res.graph
    neurons = res.neurons
    synapses = res.synapses
    
    # Generate network parameters and simulation configuration settings
    networkParams = generate_netparams(client=client,
                                       neurons=neurons,
                                       synapses=synapses,
                                       G=G,
                                       custom_mechs=custom_mechs,
                                       custom_cells=custom_cells,
                                       default_mech=default_mech,
                                       default_cell=default_cell,
                                       stim_sources=stim_sources,
                                       stim_targets=stim_targets,
                                       maintain_morphology=maintain_morphology)
    simConfig = generate_simconfig(duration=sim_duration,
                                   dt=dt,
                                   filename=filename,
                                   cells=record_names)
    
    return networkParams, simConfig

def simulate(networkParams: netpyne.specs.netParams.NetParams,
             simConfig: netpyne.specs.simConfig.SimConfig):
    ''' Create and run netpyne simulation
    
    :param networkParams: netpyne networkParams object to simulate
    :param simConfig: netpyne SimConfig object with desired simulation configuration 
    '''
    
    # Simulate and run
    sim.createSimulateAnalyze(netParams = networkParams, simConfig = simConfig)

def export_model(networkParams: netpyne.specs.netParams.NetParams,
                 simConfig: netpyne.specs.simConfig.SimConfig,
                 filename: str,
                 exp_type: str="python"):
    ''' Export netpyne model to a certain format
    
    .. note:
    
        python: use when exporting a model to run on netpyne somewhere else
        neuroml: use when exporting a model to a simulator compatable with the neuroml format. Make sure
        that you've set up the simulation before doing this! (Doesn't require you to import network
        parameters for some reason)
        
    :param networkParams: netpyne networkParams object to export
    :param simConfig: netpye SimConfig object to export
    :param filename: name of exported file
    :exp_type: 'python' or 'neuroml'
    '''
    
    if (exp_type != "python" and exp_type != "neuroml"):
        raise ValueError("exp_type must be 'python' or 'neuroml'")
        
    if (exp_type == "python"):
        conversion.pythonScript.createPythonScript(filename, networkParams, simConfig)
    else:
        conversion.neuromlFormat.exportNeuroML2(filename)
        
def run_with_brian2(file_path):
    ''' Runs a brian2 simulation from a provided neuroml2 file
    
        .. note:
        
            brian2 does not support simulations of network models with projections between
            cell populations. If the network model contains any synapses or stimulation sources,
            it will not run!
            
            Additionally, the following are required:
                - pyneuroml
                - graphviz
                - lxml
                - a java installation (java-jdk works for conda environments)
                
        :param file_path: path to a neuroml2 LEMS file (this gets generated with the neuroml option 
                          with export_model()
    '''
    
    os.system("pynml" + file_path + "-brian2")
    
    
def generate_netparams(client: fbl.Client,
                       neurons: tp.Dict,
                       synapses: tp.Dict,
                       G: nx.graph,
                       custom_mechs: tp.Dict[str, tp.Dict]=None,
                       custom_cells: tp.Dict[str, tp.Dict]=None,
                       default_mech: tp.Dict=None,
                       default_cell: tp.Dict=None,
                       stim_sources: tp.Dict[str, tp.Dict]=None,
                       stim_targets: tp.Dict[str, tp.Dict]=None,
                       maintain_morphology: bool=False) -> netpyne.specs.netParams.NetParams:
    ''' Generate a netpyne NetParams object from neurons and synapses.
    
    .. note::
    
        NetPyNE does no testing or validation of morphologies, make sure imported morphology
        is accurate and valid prior to use! 
    
    :param client: pointer to FBL client
    :param neurons: dictionary representing neurons
    :param synapses: dictionary representing synapses
    :param G: neuron graph
    :param custom_mechs: dictionary of all custom synaptic mechanisms to be used in the simulation.
                         It should be noted that since all custom mechanisms are defined here
                         irrespective of type (synapses, cell mechs, etc.), this won't check
                         whether or not any defined mechanisms are actually valid for their
                         purpose.
    :param custom_cells: dictionary of cell names and custom cell definitions to assign to them.
    :param default_mech: default synaptic mechanism to be used for unassigned synaptic connections.
    :param default_cell: default cell model to be used for unassigned cells.
    :param stim_sources: dictionary of stimulation source names, and their accompanying parameters.
                         It is assumed that custom mechanisms are defined in custom_mechs.
    :param stim_targets: dictionary of stimulation targets, and their accompanying stimulation targets.
                         It is assumed that any custom stimulation sources are defined in stim_sources.
    :param maintain_morphology: whether or not model morphology should be maintained in the
                                simulation (not recommended if neuron sections are unidentified
                                in swc definitions)
    '''
    
    # Set up our network parameters object
    networkParams = netParams.NetParams()
    
    # Define stimulation sources
    if (stim_sources == None):
        networkParams.addStimSourceParams('bkg', {'type': 'NetStim', 'rate': 10, 'noise': 0.5})
    else:
        for name, stim in stim_sources.items():
            networkParams.addStimSourceParams(name, stim)
        
    # Define synaptic mechanisms
    networkParams.addSynMechParams('exc', {'mod': 'Exp2Syn', 'tau1': 0.1, 'tau2': 5.0, 'e': 0})  # excitatory synaptic mechanism
    
    # Default mechanism
    if (default_mech == None):
        networkParams.addSynMechParams('default', {'mod': 'Exp2Syn', 'tau1': 0.1, 'tau2': 5.0, 'e': 0})
    else:
        networkParams.addSynMechParams('default', default_mech)
        
    # Custom-defined mechanisms
    if (custom_mechs != None):
        for name, mech in custom_mechs.items():
            networkParams.addSynMechParams(name, mech)
        
    #networkParams.addStimSourceParams('bkg', {'type': 'NetStim', 'rate': 10, 'noise': 0})
    
    rid_to_uname_morph = {rid: v['uname'] for rid, v in G.nodes(data=True)
                          if 'uname' in v and v.get('class', None) == 'MorphologyData'}
    rid_to_uname_neuron = {rid: v['uname'] for rid, v in G.nodes(data=True)
                           if 'uname' in v and v.get('class', None) != 'MorphologyData'}
    uname_to_rid = {v['uname']: rid for rid, v in G.nodes(data=True)
                    if 'uname' in v and v.get('class', None) != 'MorphologyData'}
    
    # Turn neurons into netpyne cells, turn synapses into netpyne connections
    for rid in rid_to_uname_morph.keys():
        # NEURONS
        
        # neuroml2 doesn't like dashes or slashes
        cellname = rid_to_uname_morph[rid]
        cellname_raw = cellname
        cellname = cellname.replace('-','_')
        cellname = cellname.replace('/','')
        
        # Useful for visualization, but shouldn't affect functionality significantly
        if(maintain_morphology):
            f = tempfile.NamedTemporaryFile(suffix='.swc')
            f.write('# SWC File for neuron ' + rid_to_uname_morph[rid] + '\n')
            f.write('#\n')
            
            # Turn morphology into a temporary .swc file. We need to to this because netpyne only
            # takes files as imputs for imported morphology
            for i in range(len(G[rid]['x'])):
                # Turns all unidentified components into somas
                # THIS IS BAD. WE ONLY DO THIS BECAUSE FBL DATA IS INCOMPLETE
                # WILL ALMOST CERTAINLY LEAD TO BAD SIMULATIONS
                if (G[rid]['identifier'][i] == 0):
                    G[rid]['identifier'][i] = 1
                    
                f.write(str(G[rid]['sample'][i]) + ' ' +
                    str(G[rid]['identifier'][i]) + ' ' +
                    str(G[rid]['x'][i]) + ' ' +
                    str(G[rid]['y'][i]) + ' ' +
                    str(G[rid]['z'][i]) + ' ' +
                    str(G[rid]['r'][i]) + ' ' +
                    str(G[rid]['parent'][i]) + '\n')
            
            cellRule = networkParams.importCellParams(label=cellname,
                                                      conds={'cellType': cellname, 'cellModel': 'HH3D'},
                                                      fileName=f.name,
                                                      cellName=cellname)
            
            # close our tempfile
            f.close()
    
            # rename imported section 'soma_0' to 'soma'
            networkParams.renameCellParamsSec(cellname + '_rule', 'soma_0', 'soma')
            
            # Define Cell Rules
            for secName in cellRule['secs']:
                cellRule['secs'][secName]['mechs']['pas'] = {'g': 0.0000357, 'e': -70}
                cellRule['secs'][secName]['geom']['cm'] = 1
                if secName.startswith('soma'):
                    cellRule['secs'][secName]['mechs']['hh'] = {'gnabar': 0.12, 'gkbar': 0.036, 'gl': 0.003, 'el': -70}
            
        # Set to default pyramidal cell if no default is provided
        elif (default_cell == None):
            PYRcell = {'secs': {}}
            PYRcell['secs']['soma'] = {'geom': {}, 'mechs': {}}
            PYRcell['secs']['soma']['geom'] = {'diam': 18.8, 'L': 18.8, 'Ra': 123.0}                           # soma geometry
            PYRcell['secs']['soma']['mechs']['hh'] = {'gnabar': 0.12, 'gkbar': 0.036, 'gl': 0.003, 'el': -70}  # soma hh mechanism
            networkParams.cellParams[cellname] = PYRcell

        # Is the cell part of custom_cells?
        elif (cellname_raw in custom_cells.keys()):
            networkParams.cellParams[cellname] = custom_cells[cellname_raw]
        
            
        # Create a cell population of 1
        networkParams.popParams[cellname] = {'cellType': cellname, 'numCells': 1}
        
        # SYNAPSES
        
        # Grab neuron info from the client because synaptic partners aren't easily accessible
        # from the graph for some reason
        info = client.getInfo(rid)
        
        connectivity = info['data']['connectivity']['pre']['details']
        
        # Loop through connections
        for con in connectivity:
            syn = con['syn_rid']
            pre, post = con['syn_uname'].split('--')
            
            # Only grab synapses to neurons that we actually care about
            if(pre in uname_to_rid.keys()):
                
                # neuroml2 doesn't like dashes or slashes
                con_uname_raw = con['syn_uname']
                con_uname = con['syn_uname'].replace('-','_')
                con_uname = con_uname.replace('/','')
                pre = pre.replace('-','_')
                pre = pre.replace('/','')
                post = post.replace('-','_')
                post = post.replace('/','')
                
                # Check to see if this is predefined synapse mechanism
                if (custom_mechs != None):
                    if (con_uname_raw in custom_mechs.keys()):
                        networkParams.addConnParams(con['syn_uname'], {'preConds': {'cellType': pre},
                                                                       'postConds': {'cellType': post},
                                                                        #'probability': 1,
                                                                       'weight': 0.1,
                                                                       'delay': 5,
                                                                       'synMech': custom_mechs[con_uname_raw]})
                    else:
                        networkParams.addConnParams(con['syn_uname'], {'preConds': {'cellType': pre},
                                                                       'postConds': {'cellType': post},
                                                                        #'probability': 1,
                                                                       'weight': 0.1,
                                                                       'delay': 5,
                                                                       'synMech': 'default'})
                # Set to default if not
                else:
                    networkParams.addConnParams(con['syn_uname'], {'preConds': {'cellType': pre},
                                                                   'postConds': {'cellType': post},
                                                                    #'probability': 1,
                                                                   'weight': 0.1,
                                                                   'delay': 5,
                                                                   'synMech': 'default'})
                
        # STIMULATION TARGETS
        if (stim_targets != None):
            for stim in stim_targets.keys():
                #networkParams.addStimTargetParams(stim + "_stim", stim_targets[stim])
                networkParams.addStimTargetParams(stim + "_stim",
                                                  {'source': stim_targets[stim]['source'],
                                                   'conds': {'pop': stim},
                                                   'weight': stim_targets[stim]['weight'],
                                                   'delay': stim_targets[stim]['delay'],
                                                   'synMech': stim_targets[stim]['mech']})
                
    return networkParams

# Currently only gives certain network analysis outputs, but more can be added
def generate_simconfig(duration: float,
                       dt: float,
                       filename: str,
                       cells: tp.List[str]=None,
                       recordStep: float=1,
                       verbose: bool=False) -> netpyne.specs.simConfig.SimConfig:
    ''' Generate a netpyne SimConfig object from provided specifications
    
    :param duration: duration of the simulation, in ms
    :param dt: internal integration timestep to use
    :param filename: file output name
    :param cells: list of cell names to record traces from
    :param recordstep: step size in ms to save data (e.g. V traces, LFP, etc)
    :param verbose: show detailed messages
    '''
    
    simConfig = SimConfig()
    
    simConfig.duration = duration          
    simConfig.dt = dt        
    simConfig.verbose = verbose
    simConfig.recordTraces = {'L2-C':{'sec':'soma','loc':0.5,'var':'v'}}  
    simConfig.recordStep = recordStep
    simConfig.filename = filename
    simConfig.savePickle = False
    
    traces = []
    
    for cell in cells:
        traces.append((cell,0))

    simConfig.analysis['plotRaster'] = {'orderBy': 'y', 'orderInverse': True, 'saveFig': True}
    simConfig.analysis['plotTraces'] = {'include': traces, 'saveFig': True}
    simConfig.analysis['plot2Dnet'] = {'saveFig': True}
    simConfig.analysis['plotConn'] = {'saveFig': True}
    
    return simConfig
