from joblib import dump, load


def model_save(model, file_name, dir='./models/'):
    dump(model, dir + file_name)

def model_load(file_name, dir='./models/'):
    return load(dir + file_name)
