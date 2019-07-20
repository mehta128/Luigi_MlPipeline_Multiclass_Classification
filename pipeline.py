import luigi
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn import linear_model
from sklearn.externals import joblib
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import resample
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import precision_score, recall_score, f1_score


class CleanDataTask(luigi.Task):

    """ Cleans the input CSV file by removing any rows without valid geo-coordinates.

        Output file should contain just the rows that have geo-coordinates and
        non-(0.0, 0.0) files.
    """

    tweet_file = luigi.Parameter()
    output_file = luigi.Parameter(default='clean_data.csv')

    def run(self):

        """Read the file with encoding and replace [0.0, 0.0] with None and drop all Na rows"""
        tweet_file = pd.read_csv(self.tweet_file, encoding='iso8859_1')
        tweet_file['tweet_coord'].replace('[0.0, 0.0]', np.nan, inplace=True)

        clean_data = tweet_file.dropna(subset=['tweet_coord'])
        clean_data['tweet_coord'].replace("[", "",inplace=True)
        clean_data['tweet_coord'].replace("]", "",inplace=True)


        clean_data.to_csv(self.output_file)

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

        # Iterate over all cities rows with iterrows() and find minimum euclidian distance for each row
        x = []
        R = 6371
        for index, row in clean_data.iterrows():

            geo = str(row['tweet_coord']).split(',')
            latitude = float(geo[0].replace("[", ""))
            longitude = float(geo[1].replace("]", ""))

            """
            Convert latitude and longitude to x,y,x coordinates using below formula:
            r = (x,y,z) = (R.cos(lat)cos(long),R.cos(lat)sin(long),R.sin(lat))
            And then calculate euclidean distance 
            
            Probably not required for data challenge but from physics point of view this the correct way to find the euclidian distance given two cordinates
            """
            convertToRadian = np.pi/180
            x1 = R*np.cos(latitude*convertToRadian)*np.cos(longitude*convertToRadian)
            y1 = R*np.cos(latitude*convertToRadian)*np.sin(longitude*convertToRadian)
            z1= R*np.sin(latitude*convertToRadian)

            euclidean_distance_list = ((((R*np.cos(city_file['latitude']*convertToRadian )*np.cos(city_file['longitude']*convertToRadian )) - x1).pow(2) +
                    ((R*np.cos(city_file['latitude']*convertToRadian)*np.sin(city_file['longitude']*convertToRadian )) - y1).pow(2)
                                        +((R*np.sin(city_file['latitude']*convertToRadian ))-z1).pow(2)) ** 0.5)

            nearest_index = euclidean_distance_list.index[euclidean_distance_list.argsort()[0]]

            x.append(city_file.loc[nearest_index,'name'])


        # One-hot encoding
        X = pd.get_dummies(pd.DataFrame(x),prefix='', prefix_sep='')

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
        X= features.iloc[:,1:len(features.columns)-1].values
        y = features.iloc[:,len(features.columns)-1].values

        def rf_cv(splits, X, Y, pipeline, average_method):
            kfold = StratifiedKFold(n_splits=splits, shuffle=True, random_state=777)
            accuracy = []
            precision = []
            recall = []
            f1 = []
            for train, test in kfold.split(X, Y):

                rf_fit = pipeline.fit(X[train], Y[train])

                prediction = rf_fit.predict(X[test])
                scores = rf_fit.score(X[test], Y[test])

                accuracy.append(scores * 100)
                precision.append(precision_score(Y[test], prediction, average=average_method) * 100)
                print('              negative    neutral     positive')
                print('precision:', precision_score(Y[test], prediction, average=None))
                recall.append(recall_score(Y[test], prediction, average=average_method) * 100)
                print('recall:   ', recall_score(Y[test], prediction, average=None))
                f1.append(f1_score(Y[test], prediction, average=average_method) * 100)
                print('f1 score: ', f1_score(Y[test], prediction, average=None))
                print('-' * 50)

            print("accuracy: %.2f%% (+/- %.2f%%)" % (np.mean(accuracy), np.std(accuracy)))
            print("precision: %.2f%% (+/- %.2f%%)" % (np.mean(precision), np.std(precision)))
            print("recall: %.2f%% (+/- %.2f%%)" % (np.mean(recall), np.std(recall)))
            print("f1 score: %.2f%% (+/- %.2f%%)" % (np.mean(f1), np.std(f1)))
            # Save the model
            joblib.dump(rf_fit, self.output_file)

        smote = SMOTE(ratio='minority')
        classifier = RandomForestClassifier(n_estimators=5,random_state=2)
        SMOTE_pipeline = Pipeline([('smt', smote), ('rf', classifier)])

        rf_cv(5, X, y, SMOTE_pipeline, 'macro')

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
    feature_file = luigi.Parameter(default='features.csv')
    model = luigi.Parameter(default='model.pkl')
    output_file = luigi.Parameter(default='scores.csv')

    def requires(self):
        return TrainModelTask(self.tweet_file)

    def run(self):

        features = pd.read_csv(self.feature_file, encoding='iso8859_1')
        loaded_model = joblib.load(open(self.model, 'rb'))
        features.drop(features.columns[0],inplace=True,axis=1)

        X = features.loc[:, ~features.columns.isin(['airline_sentiment'])]
        city_names = pd.DataFrame(X.idxmax(axis=1)).drop_duplicates(keep="first")

        Cities = features.drop("airline_sentiment",axis=1).drop_duplicates(keep="first").values
        sentiments = pd.DataFrame(loaded_model.predict_proba(Cities[:]))

        scores = pd.concat([city_names.reset_index(drop=True), sentiments.reset_index(drop=True)],axis=1)
        scores.columns = ['city', 'negative','neutral','positive']
        scores.sort_values(by='positive', ascending=False,inplace = True)
        scores.to_csv(self.output_file, index = False)

    def output(self):
        return luigi.LocalTarget(self.output_file)

if __name__ == "__main__":
    luigi.run()
