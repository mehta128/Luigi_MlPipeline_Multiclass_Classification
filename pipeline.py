import luigi
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder

from sklearn import svm


class CleanDataTask(luigi.Task):
    """ Cleans the input CSV file by removing any rows without valid geo-coordinates.

        Output file should contain just the rows that have geo-coordinates and
        non-(0.0, 0.0) files.
    """

    tweet_file = luigi.Parameter()
    output_file = luigi.Parameter(default='clean_data.csv')

    def run(self):

        # Read the file with encoding and replace [0.0, 0.0] with None and drop all Na rows
        tweet_file = pd.read_csv(self.tweet_file, encoding='iso8859_1')
        tweet_file['tweet_coord'].replace('[0.0, 0.0]', np.nan, inplace=True)

        clean_data = tweet_file.dropna(subset=['tweet_coord'])
        clean_data.to_csv(self.output_file)

        print("CleanData.shape: {}".format(clean_data.shape))

    def output(self):
        return luigi.LocalTarget(self.output_file)



class TrainingDataTask(luigi.Task):
    """ Extracts features/outcome variable in preparation for training a model.

        Output file should have columns corresponding to the training data:
        - y = airline_sentiment (coded as 0=negative, 1=neutral, 2=positive)
        - X = a one-hot coded column for each city in "cities.csv"
    """
    tweet_file = luigi.Parameter()
    clean_data_file = luigi.Parameter(default='clean_data.csv')
    cities_file = luigi.Parameter(default='cities.csv')
    output_file = luigi.Parameter(default='features.csv')

    def requires(self):
        return CleanDataTask(self.tweet_file)

    def run(self):

        clean_data = pd.read_csv(self.clean_data_file,encoding='iso8859_1')
        city_file = pd.read_csv(self.cities_file, encoding='iso8859_1')

        sentiment = {"negative": 0, "neutral": 1, "positive": 2}
        clean_data = clean_data.replace({"airline_sentiment":sentiment})
        y = clean_data['airline_sentiment']

        # iterate over all cities rows with iterrows() and find minimum distance
        x = []


        for index, row in clean_data.iterrows():

            geo = str(row['tweet_coord']).split(',')
            latitude = float(geo[0].replace("[", ""))
            longitude = float(geo[1].replace("]", ""))

            #Sorts and finds Index faster!! WHY!?
            nearest_index = city_file.index[(((city_file['latitude'] - latitude).pow(2) + (
                    city_file['longitude'] - longitude).pow(2))**0.5).argsort()[0]]

            # nearest_distance = 9999999
            # nearest_index = 0
            #
            # # Find min euclidian distance for each record in clean_data
            # for c_index, city_row in city_file.iterrows():
            #     euclidian_distance = (pow((city_row['latitude']-latitude),2) + pow((city_row['longitude']-longitude),2))**.5
            #     if(euclidian_distance < nearest_distance):
            #         nearest_distance = euclidian_distance
            #         nearest_index = c_index
            x.append(city_file.loc[nearest_index,'name'])


        #One-hot encoding
        X = pd.get_dummies(pd.DataFrame(x))

        #Label Encoder
        # X = pd.DataFrame(x).apply(LabelEncoder().fit_transform)

        features = pd.concat([X, y], axis=1)
        features.to_csv(self.output_file)




    def output(self):
        return luigi.LocalTarget(self.output_file)


class TrainModelTask(luigi.Task):
    """ Trains a classifier to predict negative, neutral, positive
        based only on the input city.

        Output file should be the pickle'd model.
    """
    tweet_file = luigi.Parameter()
    feature_file = luigi.Parameter(default='features.csv')
    output_file = luigi.Parameter(default='model.pkl')

    def requires(self):
        return TrainingDataTask(self.tweet_file)

    def run(self):
        features = pd.read_csv(self.feature_file, encoding='iso8859_1')
        X= features.loc[:,~features.columns.isin(['airline_sentiment'])].values
        y = features.loc[:,'airline_sentiment'].values

        print("X1.shape: {} y1.shape: {}".format(X.shape, y.shape))


    def output(self):
        return luigi.LocalTarget(self.output_file)


class ScoreTask(luigi.Task):
    """ Uses the scored model to compute the sentiment for each city.

        Output file should be a four column CSV with columns:
        - city name
        - negative probability
        - neutral probability
        - positive probability
    """
    tweet_file = luigi.Parameter()
    output_file = luigi.Parameter(default='scores.csv')

    # TODO...


if __name__ == "__main__":
    luigi.run()
    # luigi.build([CleanDataTask()], workers=1, local_scheduler=True)
