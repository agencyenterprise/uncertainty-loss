from typing import Any, List, Optional, Union
from pathlib import Path

import torch 
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import git 
import mne 
import mne_nirs

mne.set_log_level(verbose=False)

PathOrStr = Union[Path, str]

class FingerTappingDataset(Dataset):
    """A pytorch dataset for the finger tapping task."""

    def __init__(self, x, y):
        self.x = torch.tensor(x, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.int64)
    
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]



class FingerTapping:
    """A wrapper for creating finger tapping datasets.

    Dataset:
        https://github.com/rob-luke/BIDS-NIRS-Tapping.git
    """
    
    def __init__(self, path: Optional[PathOrStr]=None):
        """Initialize a FingerTappingDataset instance.
        
        Args:
            path (Union[Path,str], optional): Path to the data. Defaults to Path(__file__).
        """
        if path is None:
            path = Path.home()
        else:
            path = Path(path)
        self.path = path / 'fnirs-finger-tapping' / 'raw'
        self.download()
        self._subjects = self._get_subjects()

    def dataset(self, subject_id):
        """Return a pytorch dataset for a given subject."""
        return self._subjects[subject_id].x, self._subjects[subject_id].y

    def times(self, subject_id):
        """Return the times for a given subject."""
        return self._subjects[subject_id].times
    
    def subject_ids(self)->List[str]:
        """Return a list of subjects."""
        return sorted([x.id for x in self._subjects.values()])
        
    def download(self):
        """Download the data."""
        if not self.path.exists():
            git.Repo.clone_from('https://github.com/rob-luke/BIDS-NIRS-Tapping.git', self.path, bare=False)
    
    def _get_subjects(self):
        """Return a dictionary mapping subject ids to Subjects."""
        subjects = {}
        subject_ids = [self._subject_id(path) for path in self.path.glob('**/*.snirf')]
        for id_ in subject_ids:
            metadata_file = self.path / id_ / 'nirs'/ f'{id_}_task-tapping_events.tsv'
            snirf_file = self.path / id_ / 'nirs' / f'{id_}_task-tapping_nirs.snirf'
            subjects[id_] = Subject(id_, snirf_file, metadata_file)
        return subjects 


    def _subject_id(self, path):
        """Return the subject id from a path."""
        return path.relative_to(self.path).parts[0]



class Subject:
    """A class holding the data for a single subject.
    """

    def __init__(self, id:Any, snirf_file:PathOrStr, metadata_file:PathOrStr):
        self.id = id 
        self.snirf_file = Path(snirf_file)
        self.metadata_file = Path(metadata_file)
        x,y,times = self.load_and_preprocess_data()
        self.x = x 
        self.y = y 
        self.times = times

    def get_dataset(self):
        """Return a pytorch dataset for a given subject."""
        raw_snirf, metadata_df = self.load_data()
        x,y = self.preprocess_data(raw_snirf, metadata_df)
        return x,y

    def load_data(self):
        """Load the data."""
        metadata = pd.read_csv(self.metadata_file, sep='\t')
        raw_snirf = mne.io.read_raw_snirf(self.snirf_file).load_data()
        return raw_snirf, metadata

    def preprocess_data(self, raw_snirf, metadata_df):
        """Preprocess the data."""
        return preprocess_data(raw_snirf, metadata_df)

    def load_and_preprocess_data(self):
        """Load and preprocess the data."""
        raw_snirf, metadata = self.load_data()
        return self.preprocess_data(raw_snirf, metadata)




# Functionality to load and preprocess data 
# -----------------------------------------
def load_snirf(file_path):
    """Loads the raw snirf data from a file."""
    return mne.io.read_raw_snirf(file_path).load_data()

def raw_to_corrected_long_channels(raw_snirf):
    """Convert raw snirf data into corrected and filtered hemoglobin concentration"""
    raw_snirf = raw_snirf.copy()
    raw_od = mne.preprocessing.nirs.optical_density(raw_snirf)
    corrected_od = mne_nirs.signal_enhancement.short_channel_regression(raw_od)
    haemo = mne.preprocessing.nirs.beer_lambert_law(corrected_od)
    long_channels = mne_nirs.channels.get_long_channels(haemo)
    filtered_long_channels = bandpass_filter_haemo(long_channels)
    return filtered_long_channels

def bandpass_filter_haemo(haemo):
    """Bandpass filter haemo"""
    haemo = haemo.copy()
    return haemo.filter(0.05, 0.7, h_trans_bandwidth=0.2,l_trans_bandwidth=0.02)


def load_event_info(metadata_df, delimiter="\t"):
    """Load event info from a metadata file
    
    Note this function leverages prior knowledge of the snirf annotations 
    used for this experiment and the metadata format.  In particular 
    the annotations are of the form `str(float(value))` where `value` 
    is an integer id associated to the annotation (a.k.a event). Because 
    of this format we return an `event_dict` which maps the given 
    annotations to a human readable semantic label.

    Additionally, there seems to be some inconsistency in the metadata
    that we address by manually injecting the ExperimentEnds metadata 
    if its not present in the metadata file.
    """
    df = metadata_df.copy()    
    df = df[['trial_type', 'value']].drop_duplicates() 
    
    # correcting event info - trial type "15.0" should 
    # be trial type 'ExperimentEnds'
    # values should be integers 1 less than provided except in 
    # the case of ExperimentEnds which should be 15
    df.loc[df['trial_type'] == '15.0', 'trial_type'] = 'ExperimentEnds'
    df.loc[:, 'value'] = df['value'].astype(int) - 1 
    df.loc[df['trial_type'] == 'ExperimentEnds', 'value'] = 15
    event_id = dict(zip(df.trial_type, df.value))
    event_dict = {str(float(v)):k for k,v in event_id.items()}

    if '15.0' not in event_dict:
        event_dict['15.0'] = 'ExperimentEnds'
    if '15.0' not in event_id:
        event_id['ExperimentEnds'] = 15
    
    return event_id, event_dict


def fix_annotations(haemo, event_dict):
    haemo.copy()
    haemo.annotations.description = np.array([event_dict[x] for x in haemo.annotations.description])
    return haemo

def make_events(haemo, event_id):
    haemo.copy()
    events, _ = mne.events_from_annotations(haemo, event_id=event_id)
    return events

def preprocess_data(raw_snirf, metadata_df):
    """Preprocess the data."""
    event_id, event_dict = load_event_info(metadata_df)
    haemo = raw_to_corrected_long_channels(raw_snirf)
    haemo = fix_annotations(haemo, event_dict) 
    events = make_events(haemo, event_id)
    trials = make_trials(haemo, events, event_id, tmin=0, tmax=10)
    x,y = trials_to_supervised(trials)
    return x,y, trials.times


def make_trials(haemo, events, event_id, tmin=0, tmax=10):
    return mne.Epochs(
        haemo, 
        events, 
        tmin=tmin, 
        tmax=tmax,
        baseline=(0,5),
        event_id=event_id
        )


def trials_to_supervised(trial):
    """Covert trials to example dictionary

    Args:
        trial: `mne.epochs.Epochs` object 
        subject_id: default=-1, the id of the subject this trial obj is associated to
        
    Returns:
        dict of example data
    """
    example = {}
    
    # get the different types of trials
    neg_examples = trial['Control'].copy().get_data()
    left_examples = trial['Tapping/Left'].copy().get_data()
    right_examples = trial['Tapping/Right'].copy().get_data()
    
    # get the different types of labels
    neg_labels = np.zeros((neg_examples.shape[0],1), dtype=int)
    left_labels = np.ones((left_examples.shape[0],1), dtype=int)
    right_labels = 2*np.ones((right_examples.shape[0],1), dtype=int)
    
    # create data and labels
    data = np.vstack((neg_examples, left_examples, right_examples)) # (trials, channels, time)
    labels = np.vstack((neg_labels, left_labels, right_labels )) # (trials, 1)
    labels = np.squeeze(labels) # (trials,)
    
    # channel indices for hbo
    hbo_idx = mne.channel_indices_by_type(trial.info, picks='hbo')['hbo']

    # make channels last to prepare for convolutions
    #data = permute_channels_to_last(data) #(trails, time, channels)
    
    # pack information into a dictionary
    x = data[:,hbo_idx,:] # (trials, channels, time)
    example['labels'] = labels
    
    return x, labels



def permute_channels_to_last(arr, channel_pos=1):
    return np.moveaxis(arr, channel_pos, -1)