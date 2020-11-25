# 2-state HMM for locating G-C Content

## Requirements to Download
```
pip install -r requirements.txt 
```

## Data
Genome of Methanocaldococcus jannaschii 
web URL: https://www.ncbi.nlm.nih.gov/genome/?term=Methanocaldococcus+jannaschii

## How to run
To run the training for HMM, run:
```
python src/train.py -fi data/GCF_000091665.1_ASM9166v1_genomic.fna -fo output/output.txt -gt data/GCF_000091665.1_ASM9166v1_genomic.gff 
```
