class Recommender:

    def __init__(self):
        self.model = self.load_model()

    def load_model(self):
        with open('model/coliving_v1.model', 'rb') as model:
            return model
