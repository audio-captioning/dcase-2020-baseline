# Audio captioning DCASE 2020 baseline system

Welcome to the repository of the audio captioning baseline system
for the DCASE challenge of 2020. 

Here you can find the complete code of the baseline system, consisting
of:

  1. the caption evaluation part,
  2. the dataset pre-processing/feature extraction part,
  3. the data handling part for Pytorch library, and
  4. the deep neural network (DNN) method part
  
Parts 1, 2, and 3, also exist in separate repositories. Caption evaluation
tools for audio captioning can also be found
[here](https://github.com/audio-captioning/caption-evaluation-tools). 
Code for dataset pre-processing/feature extraction for Clotho dataset can
also be found [here](https://github.com/audio-captioning/clotho-dataloader).
Finally, code for handling the Clotho data (i.e. extracted features and 
one-hot encoded words) for PyTorch library (i.e. PyTorch DataLoader for
Clotho data) can also be found
[here](https://github.com/audio-captioning/clotho-dataloader).

This repository is maintained by [K. Drossos](https://github.com/dr-costas). 

----

## Table of contents

 1. [Too long - Didn't read (TL-DR)](#too-long---didnt-read-tl-dr)
 2. [Setting up the code](#setting-up-the-code)
    1. [Using conda](#using-conda-for-installing-dependencies)
    2. [Using PIP](#using-pip-for-installing-dependencies)  
 3. [Preparing the data](#preparing-the-data)
    1. [Getting the data from Zenodo](#getting-the-data-from-zenodo)
    2. [Settings up the data](#setting-up-the-data)
 4. [Use the baseline system](#use-the-baseline-system)
    1. [Create the dataset](#create-the-dataset)
    2. [Conduct an experiment](#conduct-an-experiment)
    3. [Use the pre-trained model](#use-the-pre-trained-model)
    4. [Evaluate predictions](#evaluate-predictions)
 5. [Explanation of settings](#explanation-of-settings)
    1. [Main settings](#main-settings)
    2. [Settings for directories and files](#settings-for-directories-and-files)
    3. [Settings for the creation of the dataset](#settings-for-the-creation-of-the-dataset)
    4. [Settings for the baseline model](#settings-for-the-baseline-model)
    5. [Settings for the baseline method](#settings-for-the-baseline-method)

----
  
## Too long - Didn't read (TL-DR)

If you are familiar with most of the stuff and you want to use this system
fast as possible, do the following: 

  1. Install all dependencies from the corresponding files.
  2. Make sure that your system has Java installed and enabled. 
  3. Download the data from Zenodo and place them in the data directory.
  4. Run the baseline system.  
  
If you want or need a bit more details, then read the following sections.

---- 

## Setting up the code

To start using the audio captioning DCASE 2020 baseline system, firstly you
have to set-up the code. Please **note bold** that the code in this repository
is tested with Python 3.7.

To set-up the code, you have to do the following: 

  1. Clone this repository.
  2. Use either pip or conda to install dependencies
  
Use the following command to clone this repository at your terminal:

````shell script
$ git clone git@github.com:audio-captioning/dacse-2020-baseline.git
````

The above command will create the directory `dacse-2020-baseline` and populate
it with the contents of this repository. The `dacse-2020-baseline` directory 
will be called root directory for the rest of this README file. 
  
For installing the dependencies, there are two ways. You can either use conda or
pip. 

### Using conda for installing dependencies

To use conda, you can issue the following command at your terminal (when you are 
in the root directory):

````shell script
$ conda create --name audio-captioning-baseline --file requirements_conda.yaml
````

The above command will create a new environment called `audio-captioning-baseline`, which 
will have set-up all the dependencies for the audio captioning DCASE 2020 baseline.  To 
activate the `audio-captioning-baseline` environment, you can issue the following command"

````shell script
$ conda activate audio-captioning-baseline
```` 

Now, you are ready to proceed to the following steps. 

### Using pip for installing dependencies

If you do not use anaconda/conda, you can use the default Python package manager to install
the dependencies of the audio captioning DCASE 2020 baseline. To do so, you have to issue
the following command at the terminal (when you are inside the root directory): 

````shell script
$ pip install -r requirements_pip.txt
````

The above command will install the required packages using pip. Now you are ready to go
to the following steps. 

----

## Preparing the data

After setting-up the code for the audio captioning DCASE 2020 baseline system, you have to
obtain the Clotho dataset, place it to the proper directory, and do the feature extraction.

### Getting the data from Zenodo

Clotho dataset is freely available online at the Zenodo platform. 
You can find Clotho at
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.3490684.svg)](https://doi.org/10.5281/zenodo.3490684)
 
You should download all `.7z` files and the `.csv` files with the captions. That is, you have
do download the following files from Zenodo: 

  1. `clotho_audio_development.7z`
  2. `clotho_audio_evaluation.7z`
  3. `clotho_captions_development.csv`
  4. `clotho_captions_evaluation.csv`
  
After downloading the files, you should place them in the `data` directory, in your root directory.

### Setting up the data

You should create two directories in your `data` directory:

  1. The first is called `clotho_audio_files`, and 
  2. second is called `clotho_csv_files`. 
  
Then, you have to expand the `7z` files. There are many options on how to do this. We do not want
to promote different software and/or packages, so you can just search on Google about how to expand
`7zip` files at your operating system. 

After you expand the `7z` files, you should have two directories created. The first is 
`development` and it will be created by teh `clotho_audio_development.7z` file. The second
is evaluation, and it will be created by the `clotho_audio_evaluation.7z` file. Finally, you should
have the following files and directories at your `root/data` directory: 

  1. `development` directory
  2. `evaluation` directory
  3. `clotho_captions_development.csv` file
  4. `clotho_captions_evaluation.csv` file
  
The `development` directory contains 2163 audio files and the `evaluation` directory 1045 audio
files. You should move the `development` and `evaluation` directories in the `data/clotho_audio_files`
directory, and the `clotho_captions_development.csv` and `clotho_captions_evaluation.csv` files in
the `data/clotho_csv_files` directory. Thus, there should be the following structure in your `data`
directory:


    data/
     | - clotho_audio_files/
     |   | - development/
     |   | - evaluation/
     | - clotho_csv_files/
     |   |- clotho_captions_development.csv
     |   |- clotho_captions_evaluation.csv 
 

Now, you can use the baseline system to extract the features and create the dataset.

----

## Use the baseline system

This baseline system implements all the necessary processes in order to use the Clotho data,
optimize a deep neural network (DNN), and predict and evaluate captions. Each process has\
some corresponding settings that can be modified in order to fine tune the baseline system. 

In the following subsection, the default settings will be used.

### Create the dataset

To create the dataset, you can either run the script `processes/dataset.py` using
the command:

````shell script
$ python processes/dataset.py
````

or run the baseline system using the `main.py` script. In any case, the dataset creation will
start. 

The dataset creation is a lengthy process, mainly due to the checking of the data. That is,
the dataset creation has two steps: 

  1. Firstly a split is created (e.g. development or evaluation), and then
  2. the data for the split are validated. 
  
You can select if you want to have the validation of the data by altering the `validate_dataset`
parameter at the `settings/dataset_creation.yaml` file. 

The result of the dataset creation process will be the creation of the directories: 

  1. `data/data_splits`, 
  2. `data/data_splits/development`, 
  3. `data/data_splits/evaluation`, and
  2. `data/pickles`
  
The directories in `data/data_splits` have the input and output examples for the optimization
and assessment of the baseline DNN. The `data/pickles` directory holds the `pickle` files that
have the frequencies of the words and characters (so one can use weights in the objective function)
and the correspondence of words and characters with indices.

**Note bold:** Once you have created the dataset, there is no need to create it every time. That is, after you create the dataset using the baseline system, then you can set 

````yaml
workflow:
  dataset_creation: No
````

at the `settings/main_settings.yaml` file. 

### Conduct an experiment

To conduct an experiment using the baseline DNN, you can use the `main.py` script. In case that you
do have previously created the input/output examples (using the above mentioned procedure), then you
can skip the dataset creation by altering the value of the `dataset_creation` in the
`settings/main_settings.yaml` file. To use the `main.py` script, you can issue the command:

````shell script
$ python main.py
```` 

Alternatively, you can use the `process/method.py` by: 

````shell script
$ python processes/method.py
````

The above commands will start the process of optimizing the baseline DNN, using the data
that were created in the [create the dataset](#create-the-dataset) section. 

### Use the pre-trained model

To use the pre-trained model, you have first to obtain the pre-trained weights. The
pre-trained weights are freely available online at Zenodo
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.3697687.svg)](https://doi.org/10.5281/zenodo.3697687)

### Evaluate predictions

**Note bold:** To use the caption evaluate tools you need to have Java installed and enabled. 

To evaluate the predictions, you have first to have a optimized (i.e. trained) model (i.e. a DNN). 
You can obtain this DNN directly from training process (i.e. you do first training and then
evaluation) or you can use some pre-trained weights. 

To use some pre-trained weights, you have to specify the name of the file having the weights
at the `settings/dirs_and_files.yaml` file. Also, you have to indicate that you will use a pre-trained
model (at the `settings/model_baseline.yaml` file) and indicate that you want to do evaluation
of the DNN (at the `settings/method_baseline.yaml` file). 

**Please note bold:** Before being able to run the code for the evaluation of the predictions, 
you have first to run the script `get_stanford_models.sh` in the `coco_caption` directory.

---- 

## Explanation of settings 

There are different settings for the baseline system, associated with the creation of the
dataset, the data, the outputs of the baseline system, and (of course) the optimization of
the baseline DNN. 

All these settings can be found in the `settings` directory. This directory has (by default)
the following files: 

  1. `main_settings.yaml`
  2. `dirs_and_files.yaml`
  3. `dataset_creation.yaml`
  4. `model_baseline.yaml`
  5. `method_baseline.yaml`
  
The parameters in the above `.yaml` files are explained in the following sections.

### Main settings

In the `settings/main_settings.yaml` file, you can find the following settings: 

    workflow:
      dataset_creation: Yes
      dnn_training: yes
      dnn_evaluation: yes
    dataset_creation_settings: !include dataset_creation.yaml
    feature_extraction_settings: !include feature_extraction.yaml
    dnn_training_settings: !include method_baseline.yaml
    dirs_and_files: !include dirs_and_files.yaml

The settings at the `workflow` block, correspond to the different processes that the baseline
system can do; the creation of the dataset (`dataset_creation`), the optimization of the DNN
(`dnn_training`), and the evaluation of captions (`dnn_evaluation`). By indicating
a `yes` or a `no`, you can switch on (with `yes`) or off (with `no`) the processes. For example,
the 

    workflow:
          dataset_creation: no
          dnn_training: yes
          dnn_evaluation: no
          
means that the baseline system will **not** create the dataset and will **not** evaluate captions,
but it will do the optimization of the DNN. 

The rest settings serve to indicate the files that hold the settings for each of the processes.
For example, the `dataset_creation.yaml` file holds the settings for the dataset creation. An
exception is the `dirs_and_files` field, which indicates which file holds the settings for inputs
and outputs of the baseline system. 

### Settings for directories and files

The `settings/dirs_and_files.yaml` file, holds the following settings: 

    root_dirs:
      outputs: 'outputs'
      data: 'data'
    dataset:
      development: &dev 'development'
      evaluation: &eva 'evaluation'
      features_dirs:
        output: 'data_splits'
        development: *dev
        evaluation: *eva
      audio_dirs:
        downloaded: 'clotho_audio_files'
        output: 'data_splits_audio'
        development: *dev
        evaluation: *eva
      annotations_dir: 'clotho_csv_files'
      pickle_files_dir: 'pickles'
      files:
        np_file_name_template: 'clotho_file_{audio_file_name}_{caption_index}.npy'
        words_list_file_name: 'words_list.p'
        words_counter_file_name: 'words_frequencies.p'
        characters_list_file_name: 'characters_list.p'
        characters_frequencies_file_name: 'characters_frequencies.p'
    model:
      model_dir: 'models'
      checkpoint_model_name: 'dcase_model_baseline.pt'
      pre_trained_model_name: 'dcase_model_pre_trained.pytorch'
    logging:
      logger_dir: 'logging'
      caption_logger_file: 'captions_baseline.txt'
      
These are the necessary settings, to specify the input and output directories for the
baseline system. Specifically, the `root_dirs` has the root directories for the outputs
(`outputs`) and the data (`data`). The root directory of the outputs will be used in order
to output the checkpoints of the baseline DNN and the logging files. The root directory for
the data will be used as the root directory where the data are, e.g. the `data` (at `root/data`)
directory mentioned in section [setting up the data](#setting-up-the-data). 

The `dataset` has the directory and file names used when creating and accessing the dataset
files. The `development` and `evaluation` are the name of the directories that will hold
the corresponding (in each case) development and evaluation (respectively) data. These
names can be overridden in the corresponding entries (for example, in the `dataset/audio_dirs`
for the audio directories). 

The `dataset/features_dirs` has the directory names that: 

  * parent directory for the ready-to-use features - `output`
  * the development features - `development`
  * the evaluation features - `evaluation`

The `dataset/audio` has the directory names that:

  * the downloaded audio will be - `downloaded`
  * the dataset files (i.e. the input/output examples) will be - `output`
  * the name of the directory that will hold the development data - `development`
  * the name of the directory that will hold the evaluation data - `evaluation`
  
The `annotations_dir` and `pickle_files_dir` hold the names of the directories where the
csv files will be (`annotations_dir`) and where the `pickle` files will be placed (`pickle_files_dir`).

The `files` entry at the `dataset` block, has:
 
  * the file names of the `pickle` files
    * `words_list_file_name`,
    * `words_counter_file_name`,
    * `characters_list_file_name`, and
    * `characters_frequencies_file_name`) 
  * and the template string of the file name of the input/output examples of the dataset
  (`np_file_name_template`). 

The `model` entry, has:

  * the name of the directory (in the `root_dirs/outputs`) where the checkpoints of the
  baseline DNN will be saved - `model_dir`,
  * the template file name of the checkpoint file - `checkpoint_model_name`, and
  * the file name that the baseline DNN will look as pre-trained model - `pre_trained_model_name`.
  
Finally, the `logging` entry has: 

  * the directory where the logging files will be placed - `logger_dir`, 
  * and the base file name of the logging - `caption_logger_file`. 

### Settings for the creation of the dataset

The file `settings/dataset_creation.yaml` has: 

    workflow:
      create_dataset: Yes
      validate_dataset: No
    annotations:
      development_file: 'clotho_captions_development.csv'
      evaluation_file: 'clotho_captions_evaluation.csv'
      audio_file_column: 'file_name'
      captions_fields_prefix: 'caption_{}'
      use_special_tokens: Yes
      nb_captions: 5
      keep_case: No
      remove_punctuation_words: Yes
      remove_punctuation_chars: Yes
      use_unique_words_per_caption: No
      use_unique_chars_per_caption: No
    audio:
      sr: 44100
      to_mono: Yes
      max_abs_value: 1.
      
The `workflow` block is to indicate the execution or not of the dataset creation and validation.
That is:

  * if the dataset will be created - `create_dataset`, and
  * if the data for each split will be validated (this is a lengthy process) - `validate_dataset`
  
The `annotations` block holds the settings needed for accessing and processing the annotations
(i.e. the csv files) of Clotho. That is: 

  * the name of the file holding the annotations of the development split - `development_file`
  * the name of the file holding the annotations of the evaluation split - `evaluation_file`
  * the name of the column in the annotations file that has the file name
  of the corresponding audio file - 'audio_file_column'
  * the prefix of the column name of the columns (at the csv files) that have the
  captions - `captions_fields_prefix`
  * indication of the special tokens (i.e. `<sos>` and `<eos>`) will be used - `use_special_tokens`
  * the amount of captions per each audio file - `nb_captions`
  * indication if the letter case. i.e. keep capital letters as capitals (`Yes`) or turn them to small
  case (`No`) will be kept - `keep_case`
  * if the punctuation will be removed from the word tokens - `remove_punctuation_words`
  * if the punctuation will be removed from the character tokens - `remove_punctuation_chars`
  * take into account unique words per audio file when counting the frequency of
  appearance of each word - `use_unique_words_per_caption`
  * take into account unique characters per audio file when counting the frequency of
  appearance of each character - `use_unique_chars_per_caption`
  
The `audio` block holds settings for processing the audio data: 

  * the sampling frequency that the audio data will be resampled (if needed) to - `sr`
  * indication if the audio files should be turned to mono - `to_mono`
  * maximum absolute value for normalizing the audio data - `max_abs_value`
  
### Settings for the baseline model

The file `settings/model_baseline.yaml` holds the settings for the baseline DNN:

    use_pre_trained_model: No
    encoder:
      input_dim_encoder: 64
      hidden_dim_encoder: 256
      output_dim_encoder: 256
      dropout_p_encoder: .25
    decoder:
      output_dim_h_decoder: 256
      nb_classes:  # Empty, to be filled automatically.
      dropout_p_decoder: .25
      max_out_t_steps: 22
      
The `use_pre_trained_model` flag indicates if a pre-trained model will be used. If
this flag is set to `Yes`, then the name of the file with the weights of the pre-trained
model has to be specified in the `settings/dirs_and_files.yaml` file. 
 
The `encoder` block has the settings for the encoder of the baseline DNN:

  * the input dimensionality to the first layer of the encoder - `input_dim_encoder`
  * the hidden output dimensionality of the first and second layers of the encoder -
  `hidden_dim_encoder`
  * the output dimensionality of the third layer of the encoder - `output_dim_encoder`
  * the dropout probability for the encoder - `dropout_p_encoder`
  
Similarly, the `decoder` block holds the settings for the decoder of the baseline DNN: 

  * the output dimensionality of the RNN of the decoder - `output_dim_h_decoder`
  * the amount of classes for the classifier (it is filled automatically by the
  baseline system) - `nb_classes`
  * the dropout probability for the decoder - `dropout_p_decoder`
  * the maximum output time-steps for the decoder - `max_out_t_steps`

### Settings for the baseline method

The file `settings/method_baseline.yaml` holds the settings for the training and
evaluation procedure of the baseline DNN: 

    model: !include model_baseline.yaml
    data:
      input_field_name: 'features'
      output_field_name: 'words_ind'
      load_into_memory: No
      batch_size: 16
      shuffle: Yes
      num_workers: 0
      drop_last: Yes
    training:
      nb_epochs: 300
      patience: 10
      loss_thr: !!float 1e-4
      optimizer:
        lr: !!float 1e-4
      grad_norm:
        value: !!float 1.
        norm: 2
      force_cpu: No
      text_output_every_nb_epochs: !!int 10
      nb_examples_to_sample: 100

The `model` specifies the file where the settings of the baseline DNN are (i.e. the
`settings/model_baseline.yaml` file). 

The `data` block has the settings for handling the data at the training/evaluation processes:

  * the name of the field of the numpy object that holds the input values - `input_field_name`
  * the name of the field of the numpy object that holds the output/target values - `output_field_name`
  * indication if the whole dataset should be loaded into the memory during training/evaluation
  processes - `load_into_memory`
  * the size of the batch - `batch_size`
  * indication for shuffling the training data - `shuffle`
  * the amount of workers that the PyTorch DataLoader will use - `num_workers`
  * indication if the last (incomplete) batch will be used or not - `drop_last`
  
The `training` block holds settings for the training process: 

  * the maximum amount of epochs - `nb_epochs`
  * the amount of epochs for patience - `patience`
  * the threshold at the loss for considering that the loss is the same or changed - `loss_thr`
  * the settings for the optimizer (just the learning rate) - `optimizer`
  * settings for clipping the gradient norm - `grad_norm`
  * indication if the training should necessarily be on the CPU (e.g. for debugging) - `force_cpu`
  * indication of every how many epochs there should be an output of predicted captions - `text_output_every_nb_epochs`
  * how many examples to use for outputting the captions - `nb_examples_to_sample`
  
