# Auxiliary-Synthesis-Framework
code for paper: An Auxiliary Synthesis Framework for Enhancing EEG-Based Classification with Limited Data

*******************************************************************************************************************************************************************
The codes are used for enhancing classifier performance mentioned in 'An Auxiliary Synthesis Framework for Enhancing EEG-Based Classification with Limited Data'
Other classifier can also used this framework by replacing the classifier
The length and channel of the data input into the framework need to be 1000 and 22, otherwise the structure of the generator and classifier need to be revised.
*******************************************************************************************************************************************************************

Requirements:
  pytorch 1.8.0
  numpy 1.19.2
  sklearn 0.23.2
  
file list:
  Pipeline.py: An example for using the Auxiliary Synthesis Framework to enhance classifier performance.
  Model.py: Classifier, generator.
  Train.py: Training method for classifier and generator, and synthetizing artificial samples.
  EarlyStopping.py: EarlyStopping strategy used in training stage
  A01_data_All.npy and A01_label_All.npy: Dataset for A01 subject.
