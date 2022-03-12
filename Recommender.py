class Recommender:

    def __init__(self):
        self.model = self.load_model()

    def load_model(self):
        with open('model/coliving_v1.model', 'rb') as model:
            return model

    def get_prediction(self, userID, activityID):
        return self.model.predict(userID, activityID)

    def get_recommendations(self, userID, n_recs):
        for i in range(1, n_recs):
            pass
        return None
