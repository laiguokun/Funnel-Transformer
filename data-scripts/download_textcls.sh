mkdir -p data/text_cls
cd data/text_cls

# **** IMDB ****
wget https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz
tar xzvf aclImdb_v1.tar.gz && rm aclImdb_v1.tar.gz
mv aclImdb imdb

# **** Other datasets ****

# Please refer to the google drive
# https://drive.google.com/drive/u/1/folders/0Bz8a_Dbh9Qhbfll6bVpmNUtUcFdjYmF2SEpmZUZUcVNiMUw1TWN6RDV3a0JHT3kxLVhVR2M

