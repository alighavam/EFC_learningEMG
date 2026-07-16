import os

import pandas as pd
import numpy as np
import globals as gl

experiment = 'emg'
sn = 106

baseDir = '/Volumes/diedrichsen_data$/data/Chord_exp/EFC_learningEMG/'

pinfo = pd.read_table(os.path.join(baseDir, 'participants.tsv'), sep='\t')
pinfo_row = pinfo[pinfo.sn == sn].reset_index(drop=True)
trained = np.array(pinfo_row['trained'][0].split('.'), dtype='int')
untrained = np.array(pinfo_row['untrained'][0].split('.'), dtype='int') 

template = pd.read_csv(os.path.join(baseDir, 'target', 'emg_pretraining_100_day0_run1.tgt'), sep='\t')

cols = ['subNum', 'chordID', 'planTime', 'success_holdTime', 'execMaxTime', 'feedbackTime', 'iti', 'startTime', 'endTime', 'session', 'day', 'week']
ntrials_perrun = 40
nruns = 10

# pretraining:
planTime = [1000]*ntrials_perrun
success_holdTime = [600]*ntrials_perrun
execMaxTime = [3500]*ntrials_perrun
feedbackTime = [750]*ntrials_perrun
iti = [250]*ntrials_perrun
startTime = template.startTime.tolist()
endTime = [0]*ntrials_perrun
session = ['pretraining']*ntrials_perrun
day = [0]*ntrials_perrun
week = [0]*ntrials_perrun


