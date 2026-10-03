import os
import pickle


def load_persisted_testcase(persist_path):
    if os.path.exists(persist_path):
        with open(persist_path, 'rb') as f:
            return pickle.load(f)

    print("No persisted test case list found!")
    return None
