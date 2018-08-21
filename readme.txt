# THIS DIRECTORY CONTAINS THE IMPLEMENTATION CODE FOR TACOTRO2(TTS) USING SEQ2SEQ MODELS AND WAVENET BY FOLLOWING 
THE ARCHITECTURE SPECIFIED IN: https://arxiv.org/pdf/1712.05884.pdf

ALL THE CODE HAVE BEEN IMPLEMENTED FROM THE SCRATCH FOR LEARNING PURPOSES AND WILL BE IMPROVED FURTHER

AS THERE IS VERY FEW POSTS RELATED TO WAVENET AND IMPLEMENTING IT FROM SCRATCH IS DIFFICULT IT HAS NOT BEEN IMPLEMENTED BUT WORKING ON IT.

#DIRECTORY STRUCTURE
---data
-----dataset # Stores the tf records files after processing raw data theses file wil be used for training model.
-----files # Contains raw trainnig data and testing data.
-----tensorboard # Contains all visualization data related to dataset and training.
--------summary # Contains the visualization for dataset and training metrics.
----------graph # Conatains overall tensorflow graph of the session.
----------train # Conatains trainig metrics visuals.
----------test # Conatains testing metrics visulas.
--------variable_tensor # Contains checkpoint files of trained models variables which can be restored during training.

---src # Conatains python files for execution
-----main.py # It is the main file for execution pipoeline It executes and handles all the remainig files.

