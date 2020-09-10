#!/usr/bin/python3

# Unlike most of the code in this repository
# this program is licensed under 
# Affero General Public License 
# Copyright 2020 Forrest Sheng Bao
# FSB@iastate.edu, forrest.bao@gmail.com

import glob
import os, pickle, hashlib
import importlib.util

def compare_returns_md5(r1, r2):
    """compare the returns from two function calls using MD5sum. 

    r1, and r2 can be of any types, or tuples of any types. 

    # FIXME: Not sure how deep the comparison goes to 

    """
    return hashlib.md5(pickle.dumps(r1)).hexdigest()\
        == hashlib.md5(pickle.dumps(r2)).hexdigest()

def compare_returns(r1, r2):
    """compare the returns from two function calls.

    r1, and r2 can be of any types, or tuples of any types. 
    """
    # KEEP this function until the md5sum approach above is fully verified. 
    # TODO: the == operator is ambigous for types like numpy.ndarray. 
    # What are other types like this? 
    # TODO: This method will fail for highly hierarchical data structures, 
    # e.g., tuples of lists of dicts of {openCV camera instance:numpy.ndarray}. 

    if type(r1) != type(r2):
        return False 
    if isinstance(r1, tuple): # tuple compare 
        comparison = [compare_returns(r1[i], r2[i]) 
                     for i in range(len(r1))] 
        return 1 - comparison.count(False)

    elif isinstance(r1, numpy.ndarray):
        comparison = r1==r2
        return comparison.all() 

    else: # simple types 
        # TODO: it is unclear whether other types will act like numpy arrayes that == operator can be obigious 
        return r1==r2

def compare_cases(f1, f2, problem):
    """Compare the execution of two functions over cases

    problem: keys: number, points, import_cmd, cases,  grading_policy
    grading_policy: str, "partial" or "all"
    "partial" if the grade is propotional to the ratio of passed test cases. "all" if the student gets points only when all test cases are passed. 
    """
    for cmd in problem["import_cmd"]:
        exec(cmd)
    grading_policy = problem["grading_policy"]
    cases = problem["test_cases"]
    points = problem["points"]

    pass_no = 0 
    for i, args in enumerate(cases):
        returns1 = f1(*args)
        try :
           returns2 = f2(*args)
        except : 
            return 0 

        if compare_returns_md5(returns1, returns2): 
            pass_no += 1 
        else:
            if grading_policy == "all":
                return 0 # one error, return 0 point. 
    return pass_no/len(cases)*points # float            

def load_module_from_path(path):
    spec = importlib.util.spec_from_file_location("whatever", path)
    foo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(foo)
    return foo

def grade_a_student(teacher_module_path, student_module_path, hw):
    teacher_module = load_module_from_path(teacher_module_path)
    try: 
        student_module = load_module_from_path(student_module_path)
    except : 
        return 0 # if your submission cannot be imported, 0 for all problems. 
    grade = 0 

    for problem in hw: 
        function_name = problem["function_name"]
        teacher_function = getattr(teacher_module, function_name)
        student_function = getattr(student_module, function_name)
        # check all cases
        grade +=  compare_cases (teacher_function, student_function, problem)
    return grade 

def grade_all_students(teacher_module_path, student_submission_folder, hw):
    # TODO 3. Parallelize this. 
    for student_module_path in glob.glob(os.path.join(student_submission_folder, "*.py")):
        local_grade = grade_a_student(teacher_module_path, student_module_path, hw)
        print (student_module_path, local_grade)
        
if __name__ == "__main__": 
    import numpy
    import warnings
    warnings.filterwarnings("ignore")

    hw = [
        {
            "number": 1, 
            "points": 5, 
            "import_cmd": 
            ["import numpy", "import matplotlib.pyplot"], 
#            "function_name" : "learning_curve", 
            "function_name" : "f", 
            "test_cases":  # TODO: how to support both positional arguments and keyword arguments? 
            [   (numpy.array([1,2]), numpy.array([3,4]), "test.png"), # case 1
                (numpy.array([3,4]), numpy.array([5,6]), "test.pdf")  # case 2
            ], 
            "grading_policy": "partial"
        }
    ]

    # teacher_module_name = 'answer_test_hw1'
    # student_module_name = 'hw1'

#    grade = grade_a_student(teacher_module_name, student_module_name, hw)
#    print (grade)

    teacher_module_path = 'answer_test_hw1.py'
    # student_module_path = 'hw1.py'

    # grade = grade_a_student(teacher_module_path, student_module_path, hw)
    # print (grade)


    student_submission_folder = "grading_test"
    grade_all_students(teacher_module_path, student_submission_folder
    , hw )