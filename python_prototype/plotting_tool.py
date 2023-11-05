"""
This tool simply plots the waves in a given file
"""

from pyedflib import EdfReader
import matplotlib.pyplot as plt

EXPORT_FOLDER = '/Users/tristan/Desktop/psg/'

sleep_stage_annotation_to_int = { #Note: Stages 3 and 4 are combined and '0' is reserved to be a padding mask
                                    "Sleep stage 1": 1,
                                    "Sleep stage 2": 2,
                                    "Sleep stage 3": 3,
                                    "Sleep stage 4": 3,
                                    "Sleep stage R": 4,
                                    "Sleep stage W": 5,
                                    "Sleep stage ?": -1}

def main():
    sleep_stages = list(EdfReader('/Users/tristan/Desktop/engsci-thesis/python_prototype/01-03-0001 Base.edf').readAnnotations()[2])
    sleep_stages = [sleep_stage_annotation_to_int[item] for item in sleep_stages]
    sleep_stages = [item for item in sleep_stages for _ in range(7680)] #30sec*256Hz

    signal_reader = EdfReader('/Users/tristan/Desktop/engsci-thesis/python_prototype/01-03-0001 PSG.edf')
    channels = list(signal_reader.getSignalLabels())                                                        

    counter = 0
    for channel in signal_reader.getSignalLabels():
        plt.close()

        fig, ax1 = plt.subplots() 
        ax1.set_xlabel('Time (sec)') 
        plt.plot(signal_reader.readSignal(counter, digital=True))

        ax2 = ax1.twinx()
        ax2.plot(sleep_stages, color='red') 

        plt.title(channel)
        plt.savefig(EXPORT_FOLDER+channel)
        counter += 1

if __name__ == "__main__":
    main()