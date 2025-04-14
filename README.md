<a id="readme-top"></a>

<h1 align="center">Yelp Review Classifier</h1>

### Name:Yuao Ai  
### SUID: 258527763




<!-- ABOUT THE PROJECT -->
## About The Project



#### Perform the following: 
• Identify all the unique words that appear in the “review/text” field of the reviews. Denote
the set of such words as L.  
• Remove from L all stopwords in “Long Stopword List” from http://www.ranks.nl/
stopwords. Denote the cleaned set as W.  
• Count the number of times each word in W appears among all reviews (“review/text” field)
and identify the top 500 words.  
• Vectorize all reviews (“review/text” field) using these 500 words (see an example of vector-
ization here: https://medium.com/data-science/understanding-nlp-word-embeddings-text-vectorization-1a23744f7223).  
• Cluster the vectorized reviews into 10 clusters using k-means. You are allowed to use any
program or code for k-means. This will give you 10 centroid vectors.  
• From each centroid, select the top 5 words that represent the centroid (i.e., the words with
the highest feature values)  


### Acknowledgements
#### Dataset
* [Yelp Dataset](https://www.yelp.com/dataset)

#### Preprocessing
* [Text Preprocessing Blog](https://thedatafrog.com/en/articles/text-preprocessing-machine-learning-yelp/)


<!-- GETTING STARTED -->
## Getting Started

This is a guide of how to set up my project locally.


### Prerequisites

These are the python libraries that you need to install first.
#### * PyTables
  ```sh
    pip install tables
  ```
#### * Downgrade numpy to a compatible version
  ```sh
    pip install numpy==1.23.5
  ```
#### * Pandas
  ```sh
    pip install pandas
  ```
  Assume you have already installed Python 3.6+ and pip. If not, please install them first.

### Installation

_Below is an example of how you can instruct your audience on installing and setting up your app. This template doesn't rely on any external dependencies or services._

1. Download dataset at [Yelp Dataset](https://www.yelp.com/dataset) and put it in `dataset/` 
2.  Enjoy!
<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Usage
### One step using a Makefile  
*Make sure you have installed make.*  

In this homework, I wrote a Makefile for running. Feel free to use it!
#### * Use make
  ```sh
    make
  ```
And then, you can find the answer in "results/"

#### * Clean the text
This task generates a lot of text files, which occupy significant space in your disk, 
but we can remove them all at once.
  ```sh
    make clean
  ```
It will delete all .txt and .npz files.  


### Execute step by step
Of course, you can also complete this task step by step.
##### 1. Get Tokenized dataset:
 ```sh
  python Tokenize.py -d "dataset" 'yelp_review.json' -l 1000000
 ```  
This takes only one input file, yelp_review.json, and reads 1000000 lines from this file in a single process.  
#### 2. Build the vocabulary
  ```sh
  python VocabularyBuilder.py -d "dataset" 'yelp_review_tok.json' -p 
  ```
#### 3. echo...
  ```sh
  echo "Building the vocabulary..."
  ```
## Submitted files
The files that are needed to submit are in "results/"
<table>
  <tr>
    <th colspan="2" align="center">Submitted files list</th>
  </tr>
  <tr>
    <th align="center"><strong>Content</strong></th>
    <th align="center"><strong>File Name</strong></th>
  </tr>
  <tr>
    <td align="center">Top 500 words + counts for these words</td>
    <td align="center">top_500_words_with_frequencies.txt</td>
  </tr>
  <tr>
    <td align="center">The top 5 words representing each cluster and their feature values</td>
    <td align="center">top_5_centroids.txt</td>
  </tr>
</table>


You can also use make to remove all intermediate files, leaving only the required files.
  ```sh
    make submit
  ```
## Contact

Aya -  yai104@syr.edu



