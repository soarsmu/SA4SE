# Introduction
In total, we run (5 + 4) * 6 = 54 experiments. Among them, we directly predict the labels on *Stanford CoreNLP*, *SentiStrength*, *SentiStrength-SE*, *Senti4SD* without re-training. While in *SentiCR* and pre-trained Transformer-based language models, we do supervised learning on each specific dataset.

Do remember to change your file name or location of the data into the scripts.
# Datasets
Six datasets have been used. The sources of these datasets are noted in the paper. Credit to the original authors. You can download the original datasets in the following sources.
- API Reviews (Downloaded from https://github.com/giasuddin/OpinionValueTSE/blob/master/BenchmarkUddinSO-ConsoliatedAspectSentiment.xls)
- APP Reviews (Downloaded from https://sentiment-se.github.io/replication.zip)
- Code Reviews (Download from https://github.com/senticr/SentiCR/blob/master/SentiCR/oracle.xlsx)
- GitHub Comments (Downloaded from https://doi.org/10.6084/m9.figshare.11604597)
- JIRA Issues (https://sentiment-se.github.io/replication.zip)
- StackOverflow (https://sentiment-se.github.io/replication.zip)

# Approaches
## SA4SE tools
### Stanford CoreNLP
Usage: https://github.com/smilli/py-corenlp
### SentiStrength
Download from: https://www.softpedia.com/get/Others/Home-Education/SentiStrength.shtml
You should download both the exe file and the SentiStrength Data zip file. After you extract the data file, you will see it contains many useful word lists.
### SentiStrength-SE
Download from: https://laser.cs.uno.edu/Projects/Projects.html
### SentiCR
Source code:https://github.com/senticr/SentiCR
### Senti4SD
Source code: https://github.com/collab-uniba/pySenti4SD

## Pre-trained Transformer-based Language Models
We used pre-trained BERT, XNLet, RoBERTa, ALBERT models. We use huggingface library: https://huggingface.co/transformers/

# Scripts
## Pre-trained Transformer-based language models
We used six python scripts, i.e., [api.py](./scripts/PTM/api.py), [app.py](./scripts/PTM/app.py), [cr.py](./scripts/PTM/cr.py), [github.py](./scripts/PTM/github.py), [jira.py](./scripts/PTM/jira.py) and [so.py](./scripts/PTM/so.py). For each script, the argument is the model. For example, to run BERT on api data, you can run the [api.py](./scripts/PTM/api.py) like follows: `$ python api.py -m 0` (Instead of tuning the hyper-parameters, we used fixed hyper-parameters as stated in our paper.)


You can also apply the early-stopping technique in the scripts, and the code in [this folder](./scripts/PTM/early-stopping). If you would like to train/fine-tune the BERT on dataset API, please run `$ python api.py -m 0 -r 1`; If you would like to evaluate the fine-tuned BERT on dataset API, please run `$ python api.py -m 0 -r 0`. The argument '-r' indicates whether you would like to re-train.

The data used by this group as located in [PTM folder](./data/PTM/)

## SentiCR
After cloning this [repo](https://github.com/senticr/SentiCR), you have to modify the training oracle and its corresponding test part. We also put the modified script in [SentiCR.py](./scripts/SentiCR/SentiCR.py) and [SenticrTest.py](./scripts/SentiCR/SenticrTest.py). You can replace our scripts in your cloned SentiCR repo to run the test. You should notice that you have to use the same training and test dataset. For example, use training oracle (GitHub dataset) and test file (GitHub dataset).

## Senti4SD
After your clone this [repo](https://github.com/collab-uniba/pySenti4SD), just run the following command without re-training:
```bash
sh classification.sh -i test_dataset.csv -o predictions.csv
```

After getting the predictions, please run [Senti4SD.py](./scripts/analyze-results/Senti4SD.py) to analyze the prediction performance.

## Stanford CoreNLP
After you download and start the Stanford CoreNLP server, you can import the library by referring the example in this [repo](https://github.com/smilli/py-corenlp). Our script is in [StanfordCoreNLP.py](./scripts/StanfordCoreNLP.py).

## SentiStrength
### Prepare data
As all the input should only be one line, we should convert our data into this format in case some sentences have multiple lines. Our used test data can be found in [SentiStrength folder](./data/SentiStrength/). If you want to run your own, you can directly use our [script](./scripts/prepare-data/convert_sentistrength.py)
### Prediction
Run SentiStrength2.3Free.exe. The process is as follows:
1. Select reporting options
Click on 'Reporting Options' -> Unchoose 'Report Classification Rationale' and 'Report Translation (From Abbreviations etc.)'. In other words, we only select 'Report Sentiment Strength CLassifications [don't uncheck this normally ever].
2.  Select input file
Click on 'Sentiment Strength Analysis' -> 'Analyse ALL Texts in File [each line separately]' -> Select the test file -> 'Echo header line to the results?', select 'Yes' -> 'Which column contains the text? Enter 1 for ...', enter 1 -> Choose your folder to save the output file



The input file is a txt file, which in each line is a test text. It will output a file, in each line, it has two values, which represent the negative and positive values, respectively. Our strategy is to calculate the sum of these two values.
### Evaluation
After getting the predictions, please run [SentiStrength.py](./scripts/analyze-results/SentiStrength.py) to analyze the prediction performance.

## SentiStrength-SE
Almost the same workflow as SentiStrength.
### Prepare data
The same as SentiStrength.
### Prediction
This application is almost the same as SentiStrength. It will output two integer values, and we assign a sentiment value based on the sum.
```bash
java -jar SentiStrength-SE_V1.5.jar
```
Input-> Select the test file


Detect Sentiments
### Evaluation
After getting the predictions, please run [SentiStrength-SE.py](./scripts/analyze-results/SentiStrength-SE.py) to analyze the predictions.

## Discussion part
We compared the predictions made by XLNet and SentiCR in Discussion part in our paper. The script used is [gh-xlnet-senticr.py](./scripts/analyze-results/gh-xlnet-senticr.py).

# Contact
If you have any problems, feel free to contact Ting Zhang (tingzhang.2019@phdcs.smu.edu.sg)

# Cite
If you find this repo useful, please consider to cite our work.

@inproceedings{zhang2020sentiment,
  title={Sentiment Analysis for Software Engineering: How Far Can Pre-trained Transformer Models Go?},
  author={Zhang, Ting and Xu, Bowen and Thung, Ferdian and Haryono, Stefanus Agus and Lo, David and Jiang, Lingxiao},
  booktitle={2020 IEEE International Conference on Software Maintenance and Evolution (ICSME)},
  pages={70--80},
  year={2020},
  organization={IEEE}
}
