import flybrainlab as fbl
from brian2 import *
import networkx as nx
import numpy as np
import os
import os.path
import tempfile
import typing as tp

def model_gen(client: fbl.Client,
              res: fbl.graph.NAqueryResult,
              custom_mechs: tp.Dict[str, tp.Dict]=None,
              custom_cells: tp.Dict[str, tp.Dict]=None,
              default_mech: tp.Dict=None,
              default_cell: tp.Dict[str, tp.Dict]=None,
              stim_sources: tp.Dict[str, tp.Dict]=None,
              stim_targets: tp.Dict[str, str]=None,
              record_names: tp.List=None,
              sim_duration: int=1*1e3,
              dt: int=0.05,
              maintain_morphology: bool=False,
              **kwargs):
    ''' Generates a brian2 model from a neuroNLP graph
    
    .. note::
        
        Only verified to work with Medulla queries / clients.
        
        Kwargs should be a dict of custom variables used in differential equation definitions
        
        See the Brian2 package reference for more information on defining network components
        (https://brian2.readthedocs.io/en/stable/resources/tutorials/index.html)
    
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
    networkParams = generate_model(client=client,
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
    
    return networkParams

def simulate(networkParams,
             t: float):
    ''' Create and run netpyne simulation
    
    :param networkParams: netpyne networkParams object to simulate
    :param simConfig: netpyne SimConfig object with desired simulation configuration 
    '''
    
    # Simulate and run
    networkParams.run(t*ms)

    
def generate_model(client: fbl.Client,
                   neurons: tp.Dict,
                   synapses: tp.Dict,
                   G: nx.graph,
                   custom_mechs: tp.Dict[str, tp.Dict]=None,
                   custom_cells: tp.Dict[str, tp.Dict]=None,
                       default_mech: tp.Dict=None,
                       default_cell: tp.Dict[str, tp.Dict]=None,
                       stim_sources: tp.Dict[str, tp.Dict]=None,
                       stim_targets: tp.Dict[str, str]=None,
                       maintain_morphology: bool=False):
    ''' Generate a brian2 network from neurons and synapses.
    
    .. note::
    
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
    :param stim_targets: dictionary of stimulation names, and their accompanying stimulation targets.
                         It is assumed that any custom stimulation sources are defined in stim_sources.
    :param maintain_morphology: whether or not model morphology should be maintained in the
                                simulation (not recommended if neuron sections are unidentified
                                in swc definitions)
    '''
    
    # Define our network
    networkParams = Network()
    
    # Define stimulation sources
    if (stim_sources != None):
        for name, stim in stim_sources.items():
            locals()[name] = NeuronGroup(1, **stim)
            networkParams.add(locals()[name])
        
    #networkParams.addStimSourceParams('bkg', {'type': 'NetStim', 'rate': 10, 'noise': 0})
    
    rid_to_uname_morph = {rid: v['uname'] for rid, v in G.nodes(data=True)
                          if 'uname' in v and v.get('class', None) == 'MorphologyData'}
    rid_to_uname_neuron = {rid: v['uname'] for rid, v in G.nodes(data=True)
                           if 'uname' in v and v.get('class', None) != 'MorphologyData'}
    uname_to_rid = {v['uname']: rid for rid, v in G.nodes(data=True)
                    if 'uname' in v and v.get('class', None) != 'MorphologyData'}
    
    # Turn neurons into neuron groups
    for rid in rid_to_uname_morph.keys():
        # NEURONS
        
        # neuroml2 doesn't like dashes or slashes
        cellname = rid_to_uname_morph[rid]
        cellname_raw = cellname
        cellname = cellname.replace('-','_')
        cellname = cellname.replace('/','')

        # Is the cell part of custom_cells?
        if (custom_cells != None):
            if (cellname_raw in custom_cells.keys()):
                eqs = custom_cells[cellname_raw]
            else:
                eqs = default_cell
        else:
            eqs = default_cell
        
        # Create a cell population of 1
        locals()[cellname] = NeuronGroup(1, **eqs)
        
        networkParams.add(locals()[cellname])
    
    # Turn synapses into synapse groups                                         
    for rid in rid_to_uname_morph.keys():
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
                        locals()[con_uname] = Synapses(locals()[pre], locals()[post], **custom_mechs[con_uname_raw])
                    else:
                        locals()[con_uname] = Synapses(locals()[pre], locals()[post], **default_mech)
                else:
                    locals()[con_uname] = Synapses(locals()[pre], locals()[post], **default_mech)
                    
                locals()[con_uname].connect()
                    
                networkParams.add(locals()[con_uname])
                
        # STIMULATION TARGETS
        if (stim_targets != None):
            for stim in stim_targets.keys():
                #networkParams.addStimTargetParams(stim + "_stim", stim_targets[stim])
                locals()[stim + "_stim"] = Synapses(locals()[stim], locals()[stim_targets[stim]], on_pre=default_mech)
                locals()[stim + "_stim"].connect()
                
                networkParams.add(locals()[stim + "_stim"])
                
    return networkParams