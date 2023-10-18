import pickle

# Load the models
with open('dv.bin', 'rb') as dv_file:
    dv = pickle.load(dv_file)
with open('model1.bin', 'rb') as model_file:
    model = pickle.load(model_file)

# Client data
client_data = {"job": "retired", "duration": 445, "poutcome": "success"}

# Transform the client data using the DictVectorizer
X = dv.transform([client_data])

# Predict the probability
probability = model.predict_proba(X)[0][1]

print(f"The probability that this client will get a credit is: {probability:.3f}")
