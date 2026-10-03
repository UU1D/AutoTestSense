import pickle
import os

def load_persisted_testcase(PERSIST_PATH):
    if os.path.exists(PERSIST_PATH):
        with open(PERSIST_PATH, 'rb') as f:
            test_case_list = pickle.load(f)
        return test_case_list
    else:
        print("No persisted test case list found!")
        return None