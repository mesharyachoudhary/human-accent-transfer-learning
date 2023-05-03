# Accent Transfer

## Team Name
SignalMinded

## Team Members
* Saket Kumar Singh 190101081 CSE bachelors
* Pradnesh Prasad Kalkar 190101103 CSE bachelors
* Mesharya M Choudhary 190101053 CSE bachelors
* Anirudh Phukan 190101104 CSE bachelors
* Korada Pavan Kumar 190102093 ECE bachelors

## Project Title
Accent Transfer Techniques for Speech Synthesis

## Objective
The objective of this project is to test the hypothesis that **accent transfer techniques**, such as deep neural networks, signal processing, or machine learning algorithms, can effectively extract the accent of one speaker and superimpose it on the content of some other speaker while preserving the voice signature (tone/style of speaking) of the latter.

The results of the study will provide insights into the effectiveness of accent transfer techniques while preserving the voice signature of the speaker. It has potential applications in speech synthesis, language learning, and accent modification.

## Background
Pronunciation of words (accent) of same language often differ across the region in which the language is spoken. This often creates a barrier in understanding words and hinders communication between people of different region despite knowing a common language. Converting accent of the speaker to the accent of listener while preserving the identity (voice signature) of the speaker would allow a conversation to be more meaningful and easily interpretable without loosing the identity of the speaker in the content being heard.

## Methodology
We will use a publicly available dataset _Speech Accent Archive dataset on Kaggle_ to train and test our model.
We shall use the following approaches to model the system (once the pre-processing of input data is done):

1. Fourier Transform
    * Take Fourier Transform of both input signals
    * Use a mathematical aggregation function to combine content of one and the style of other
    * Convert the aggregated signal back to time domain and then interpolate
2. Neural audio style transfer
    * The process of audio neural style transfer involves using deep convolutional neural networks to extract the complex features from the style clip and then apply them to the content clip
    * A frozen pre-trained CNN is used to extract features from the spectrogram of the style and content clip. The output spectrogram is iteratively updated to minimize style and content loss. Finally the spectrogram is converted back to an audio signal to get a new audio clip
3. Other ML Methods
    * Use of ICA (Independent Component Analysis) for feature extraction.
    * Using a pipelined model to perform following tasks:
        * NLP model to extract text (content) of speaker.
        * A model to extract voice signature of speaker.
        * A model to extract accent of listener.
        * A model to superimpose the accent of listener with content and voice signature of speaker.
    * Use of a model to split the voice into text, voice signature and accent. This model can be used to split the voice of speaker and sample input voice of listener. Following that the accent of listener can be superimposed with content and voice signature of speaker (this can be again done using ML method or simple signal processing aggregation methods)


## Deliverables
* A .ipynb notebook detailing the analysis and comparision between how various techniques can be used to validate (or confront) the stated hypothesis for development of the speech accent transfer system.
* A GitHub repository containing the source code and documentation of the project.
* An oral presentation in class.

## Dataset to be used
1. Speech Accent Archive dataset on Kaggle: Contains around 2140 speech samples, distinct speakers, each speaking in English. The speakers are from across 177 countries and 214 diverse native languages. This dataset will be used for training and testing our model.

## References
1. Audio Style Transfer. (n.d.). Audio Style Transfer | IEEE Conference Publication | IEEE Xplore. https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8461711&tag=1.
2. Verma, P. and Smith, J.O. (2018) Neural style transfer for audio spectograms, arXiv.org. Available at: https://arxiv.org/abs/1801.01589 (Accessed: April 5, 2023). Kurowski, M., SroczyÅ„ski, A., Bogdanis, G., & CzyÅ¼ewski, A. (2021). An automated method for biometric handwritten signature authentication employing neural networks. Electronics, 10(4), 456.
