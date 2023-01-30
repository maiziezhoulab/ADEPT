# ADEPT: Autoencoder with Differentially Expressed Genes and Imputation for a Robust Spatial Transcriptomics Clustering

## Introduction 

Recent advancements in spatial transcriptomics (ST) have enabled an in-depth understanding of complex tissue by allowing the measurement of gene expression at spots of tissue along with their spatial information. Several notable clustering methods have been introduced to utilize both spatial and transcriptional information in analysis of ST datasets. However, data quality across different ST sequencing techniques and types of datasets appears as a crucial factor that influences the performance of different methods and influences benchmarks. To harness both spatial context and transcriptional profile in ST data, we develop a novel graph-based multi-stage framework for robust clustering, called ADEPT. To control and stabilize data quality, ADEPT relies on selection of differentially expressed genes (DEGs) and imputation of the multiple DEG-based matrices for the initial and final clustering of a graph autoencoder backbone that minimizes the variance of clustering results. We benchmarked ADEPT against five other popular methods on ST data generated by different ST platforms. ADEPT demonstrated its robustness and superiority in different analyses such as spatial domain identification, visualization, spatial trajectory inference, and data denoising.

The general workflow of ADEPT works as follows:

![image](https:****.png)

## Installation
### Dependencies
scanpy

pytorch

pyG

pandas

numpy

scipy

matplolib

### Install from github
1. git clone --recursive https://github.com/maiziezhoulab/AquilaDeepFilter.git
2. conda create -n [EnvName] python=3.8
3. source activate [EnvName]
4. pip install -r requirements.txt
5. conda install -c bioconda tabix

## Pipeline

**Part 1. Image construction**

a. create bed files for SV image construction (training data/evaluation data)

use **vcf2bed_training.py** and **vcf2bed_val.py** to generate .bed files before generating images

will have **3** .bed files as the output of **vcf2bed_training.py**

will have **1** .bed file as the output of **vcf2bed_val.py**

b. generate image from the .bed files (generated in last step) and .bam files (used for SV calling)

use **bed2image.py** to generate training/testing images. For training images, positive samples are generated from 2 .bed files corresponding to TP.vcf and FN.vcf while negative samples are generated from 1 .bed file corresponding to FP.vcf

c. augmentate the images generated in last step

use **augmentate.py** to augmentate the images

**Part 2. Model training**

a. split the training data with customized ratio

use **train_test_split.py** to split the data. Specifically, the input folder should have 2 subfolders for positive and negative samples respectively.

b. train the model with the split images in train and val folders

use **train** in **main.py** to train the model with saved checkpoints and loss

**Part 3. Truvari evaluation**

a. predict the evaluation data with the saved weights

(a-optional. ensemble prediction results from several models)

b. generate several filtered bed/vcf files with different thresholds

c. prepare (sort, zip and index) the vcf files for Truvari input format

d. perform evaluation and report metrics

## How to run the scripts

**1. file format conversion**
      This script is used to extract SV signals for image generation.

	python ./preprocess/vcf2bed/vcf2bed_training.py 

		--vcf_dir [path to folder of Truvari evaluation]
            --output_folder [path to output folder]
            --SV_type [DEL or INS]

    python ./preprocess/vcf2bed/vcf2bed_val.py 

		--path_to_vcf [path to vcf file of upstream caller]
            --path_to_output_folder [path to output folder]
            --SV_type [DEL or INS]

    python ./post/vcf2bed/bed2vcf.py 

		--path_to_vcf [path to raw vcf file for validation]
            --path_to_original_bed_file [path to raw bed file with index]
            --path_to_index_file [path to the index]
            --path_to_predicted_bed_file []
            --path_to_output_vcf_file_suffix []
            --path_to_header_vcf_file []
            --add_chr [index as chr1 or 1. True/chr1 of False/1]
            --confidence_threshold [minimum confidence threshold]
            --increment [intervals for threshold gradient]

**2. image generation and augmentate**
      These scripts are used for image generation and data augmentation.

	python ./preprocess/image_generation/bed2image.py 

		--sv_type [DEL or INS]
            --bam_path [path to .BAM file]
            --bed_path [path to .BED file generated in step 1]
            --output_imgs_dir [path to output folder for images]
            --patch_size [width, height]
    
    python ./preprocess/image_generation/augmentate.py 

		--output_imgs_dir [path to output folder for augmentated images]
            --image_path_file [path to file that includes all the images for augmentation]
            --patch_size [wdith, height]

**3. train**
      These scripts are used to train AquilaDeepFilter and split train/val set.  

	python ./AquilaDeepFilter/train_test_split.py

		--ratio 0.8 [training set ratio]
            --input_dir [path to folder with generated images]
	        --output_dir [path to output folder with split train/val folders]
    
    python ./AquilaDeepFilter/main.py train

		--model_arch [xception,densenet,efficientnet,vgg,resnet,mobilenet]
            --batch_size BATCH_SIZE [number of samples in one batch]
            --path_to_images [path to prediction images directory]
            --output_file [path where output file needs to be saved]
            --checkpoint_dir [path to directory from which checkpoints needs to be loaded]
            --num_classes [number of classes to pick prediction from]
            --height HEIGHT [height of input images]
            --width WIDTH [width of input images, default value is 224]
            --channel CHANNEL [channel of input images, default value is 3]

  
**4. predict**
      This script is used to make predictions for candidate SVs.  

	python ./AquilaDeepFilter/main.py predict

		--model_arch [xception,densenet,efficientnet,vgg,resnet,mobilenet]
            --batch_size BATCH_SIZE [number of samples in one batch]
            --path_to_images [path to prediction images directory]
            --output_file [path where output file needs to be saved]
            --checkpoint_dir [path to directory from which checkpoints needs to be loaded]
            --num_classes [number of classes to pick prediction from]
            --height HEIGHT [height of input images]
            --width WIDTH [width of input images, default value is 224]
            --channel CHANNEL [channel of input images, default value is 3]

**5. evaluate**
	This script performs Truvari evaluation on the training models.  

	python ./post/truvari/truvari_evaluation.py

	        --path_to_folder_with_gradiant_vcf
            --path_to_folder_with_gradiant_vcf [folder for storing converted vcf files]
            --path_to_output_folder [path to the folder for generated evaluation result]
            --vcf_bench [path to the benchmark giab vcf file]
            --fasta [path to the reference genome]
            --include_bed [path to the giab gold standard bed file]
            --minimum [the lower length bound for evaluating SV detection]
            --maximum [the upper length bound for evaluating SV detection]

**6. ensemble**
	This script is used to acquire majortiy voting results of all models.  

	python ./post/ensemble/ensemble_.py 

		--path_to_models_results [input folder of models predition results]
            --ensemble_output [path of output file after voting]

## Repo Structure and Output

1. The folder of AquilaDeepFilter, post and preprocess have corresponding scripts and codes for Running the AuilaDeepFilter software

2. The dependencies are documented in the requirements.txt.

3. The 'train' command in 'main.py' script will constantly store the weights for the training epoch with the best validation Acc. and stops after the convergence is reached.

4. The 'predict' command in 'main.py' script will generate output in the BED structure (but in .txt file format). It can be then converted back to vcf for evaluation.

5. The docker image of this project will also be uploaded later after all the configuration and testing work are done.

6. The uploaded weights for our model and the toy dataset could be found in zenodo: ...

Citation
--------
paper currently under review

