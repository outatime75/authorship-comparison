## Authorship Comparison :books:

The objective of this task was to create a model to determine if two texts come from the same author. 

Link to notebook - https://www.kaggle.com/code/karlcini/1st-place-solution-tantusdata-summer-challenge 

**Context**

- Overview of the competition - https://www.kaggle.com/competitions/authorship-comparison/overview
- Data - https://www.kaggle.com/competitions/authorship-comparison/data

**Overview of the approach**

While different approaches were explored, using the cosine similarity and other distance metrics combined with emotion indices extraction from the embeddings and feeding these into an ensemble of machine learning models (CatBoost Classifier, RandomForest Classifier and LGBM Classifier) produced the most consistent results. 

**Details of the submission**

*Eliminating duplication*

It was noted at the early stages of the competition that the provided training dataset contained a number of duplicates, both within some of the authors as well as intra authors. While the organizers provided circa 10 text excerpts for each author, certain excerpts were almost identical (save for some punctuation). These texts were identified using Levenshtein distance and filtered out of the dataset. 

Subsequently it was also observed that some authors, while having different identification numbers, were actually the same person, having the same text excerpts. These were also filtered out.

*Additional data*

While it was possible to augment the provided data with additional blogs/writings from the provided corpus links, it was decided to retain the original set of excerpts to stay in line with the objective of the challenge, that of creating an efficient model with the least data possible. 

*Preprocessing*

The resulting dataset was preprocessed using regex and the genism library (Rehurek, R., & Sojka, P. (2011). Gensim–python framework for vector space modelling. NLP Centre, Faculty of Informatics, Masaryk University, Brno, Czech Republic, 3(2).) to remove stopwords, punctuation and numbers.

*Train/Evaluation sets*

The full dataset was split into a training set and an evaluation set, stratifying by ‘author’.

*Choice of model*

The model intfloat/e5-small-v2 [1] was obtained from Hugging Face. This is a model with 12 layers and an embedding size of 384.  The SentenceTransformer library was used to load this model.

*Creating training and evaluation pairs*

The main function provided by the organizers was used for this purpose, with some modifications to allow for the automatic setting of the number of positive and negative pairs and the introduction of an additional condition to ensure that no pair content is the same. 

A paired dataset was also created for the full dataset to capture the most amount of data. 

*Additional EDA*

In the analysis of the range of cosine similarities between the pairs, some work was conducted to identify all the possible combinations of matching and non matching pairs. This was deemed relevant within the context of the generation of the training/evaluation pairs, given that only a limited subset of these combinations were randomly selected for inclusion. 

For matching pairs it was calculated that there were circa 1665 possible combinations while for non-matching pairs this number grew to a staggering 68,265.  Matching cosine similarity scores were in the range of 0.71 to 0.96, while the scores for the non-matching pairs, after eliminating values outside of 3 standard deviations from the mean, were in the range of 0.69 to 0.84. This showed that there would be an overlap between the cosine similarity scores for matching and non-matching pairs, leading to possible misclassification. 

*Fine tuning of LLM model*

The chosen model was fine tuned using Contrastive Loss, 3 epochs, a learning rate of 1e-04, weight_decay of 1e-4, ‘warmupcosine’ as scheduler, batch_size of 4 and warmup_steps of 0.2 * n_train_steps where n_train_steps = len(train_examples) // batch_size

*Using emotions*

The idea behind this section is to augment the characteristics that can be extracted from the given texts, which characteristics could be used to identify whether the two given pieces of text are by the same author or not. For this section a language model based on DistilRoBERTa-base created by Jochen Hartmann and published on Huggingface https://huggingface.co/j-hartmann/emotion-english-distilroberta-base/  will be used to extracted Ekman's 6 basic emotions, plus a neutral class:

- anger
- disgust
- fear
- joy
- neutral
- sadness
- surprise

The scores for each pair in the training and evaluation datasets were extracted and saved into a dataframe. 
At the same time the cosine similarity, Euclidean distance, correlation distance and Manhattan distance were calculated using the embedded functions within SentenceTransformers for cosine similarity and sklearn’s pairwise_distances for the other distance metrics.  The results were added to the respective dataframes. 

*Preparing the test set*

A similar dataset was constructed for the test set, preprocessing the text, extracting the distance metrics and the emotion indices. 

**Setting machine learning model and making predicitons**

Three classifiers were chosen for this final task, CatBoost Classifier, RandomForestClassifier and LGBMClassifier. The various hyperparameters for each model were obtained after a series of experiments. 

The training set was split into 3 folds using StratifiedKFold on the basis of whether a pair was from a matching author or not. 

Two approaches can be used for generating predictions for the test set.  In one approach, the training set created earlier would be used for fitting the model and predictions made on the evaluation set. These would then be optimized for the best probability threshold for each of the models (or else combined using the average). The model would again have to be re fit with the predictions made on the provided test set for submission. The previously calculated probability thresholds for each model would be used to create the predictions made by each model.  A majority hard voting method would then be used to determine the final predictions.

In the second approach the training set comprising the full available dataset would be used to fit the model and predictions made directly on the test set.  The final predictions could either be calculated on the basis of hard majority voting of the three models or else have the probability predictions of the models averaged out and used for the final predictions. 

The majority voting approach resulted in the highest accuracy score. 







