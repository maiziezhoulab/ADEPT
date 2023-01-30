# ADEPT: Autoencoder with Differentially Expressed Genes and Imputation for a Robust Spatial Transcriptomics Clustering

## Introduction 


<!-- 1. Run the upstream SV caller (such as Aquila, Delly) to generate the VCF file which includes structrual variants.

2. Preprocess VCF file to add header if needed (eg. make 1 -> chr1), and split it into INS and DEL for Truvari evaluation.

3. Get the Truvari evaluated VCF results as the training bed files and raw VCF results as the validation files.  

4. Converting to images and augmentation.

5. Training with CNNs.

6. Get the output BED file back to VCF files.

7. Ensemble strategy.

8. Evaluate with Truvari again to see the performance of our model. -->

Computational methods based on whole genome linked-reads and short reads have been successful in genome assembly and detection of structural variants (SVs). Numerous variant callers that rely on linked-reads and short reads can detect genetic variations, including SVs.  A shortcoming of existing tools is a propensity for overestimating SVs, especially for deletions. Optimizing the advantages of linked-read and short-read sequencing technologies would thus benefit from an additional step to effectively identify and eliminate false positive large deletions. Here, we introduce a novel tool, AquilaDeepFilter, aiming to automatically filter genome-wide false positive large deletions. Our approach relies on transforming sequencing data into an image and then relying on several convolutional neural networks to improve classification of candidate deletions as such. Input data take into account multiple alignment signals including read depth, split reads and discordant read pairs. AquilaDeepFilter is thus an effective SV refinement framework that can improve SV calling for both linked-reads and short-read data.

The general workflow of AquilaDeepFilter works as follows:

![image](https://github.com/maiziezhoulab/AquilaDeepFilter/blob/main/img.png)

## Installation
### Dependencies
truvari==3.0.0

tabix==1.11 [**bioconda**]

tensorflow==2.5.0

tensorboard==2.5.0

matplotlib==3.1.0

numpy==1.19.5

opencv-python

Pillow==7.2.0

pysam==0.15.4

scikit-learn

scipy==1.5.4

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
Y. Hu, S. V. Mangal, L. Zhang, X. Zhou. An ensemble deep learning framework to refine large deletions in linked-reads. The IEEE International Conference on Bioinformatics and Biomedicine (BIBM) (2021) 


