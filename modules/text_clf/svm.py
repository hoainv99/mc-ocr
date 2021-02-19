import pickle
class predict_svm(object):
    def __init__(self,path_svm='weights/svm_model_v3.pkl',path_tf_idf='weights/tfidf_vectorization_v3.pkl'):
        super().__init__()
        self.svm = pickle.load(open(path_svm, 'rb'))
        self.Tfidf_vect = pickle.load(open(path_tf_idf, 'rb'))
    def __call__(self, texts):
        inp = self.Tfidf_vect.transform(texts)
        out = self.svm.predict(inp)
        out_proba = self.svm.predict_proba(inp)
        return out,out_proba
