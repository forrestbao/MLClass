#!/usr/bin/python3

# Unlike most of the code in this repository
# this program is licensed under 
# Affero General Public License 
# Copyright 2020 Forrest Sheng Bao
# FSB@iastate.edu, forrest.bao@gmail.com

import glob, os, operator, importlib.util
import itertools
import joblib


def ndarray_tuple_comparitor(t1, t2):
    """A comparitor to compare two tuples of numpy arrays
    t1, t2: tuples of numpy arrays
    """
    import numpy
    results = [*map(numpy.array_equal, t1, t2)] # a Boolean list
    return all(results)

def compare_returns_comparitor(r1, r2, comparitor=operator.eq):
    """Use a comparitor function to compare the two returns 
    """
    return comparitor(r1, r2)

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

    kwargs = {key:problem[key] for key in ['comparitor'] if key in problem}

    pass_no = 0 
    for i, args in enumerate(cases):
        returns1 = f1(*args)
        try :
            returns2 = f2(*args)
        except : 
            return 0 

        if compare_returns_comparitor(returns1, returns2, **kwargs): 
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
        return (student_module_path, 0) # if your submission cannot be imported, 0 for all problems. 
    grade = 0 

    for problem in hw: 
        function_name = problem["function_name"]
        teacher_function = getattr(teacher_module, function_name)
        try: 
            student_function = getattr(student_module, function_name)
        except: # if missing the function def 
            continue # to next problem

        # check all cases
        grade +=  compare_cases (teacher_function, student_function, problem)
    print (student_module_path, grade)
    return (student_module_path, grade)

def grade_all_students(teacher_module_path, student_submission_folder, hw):
    n_jobs = 5 

    student_modules = glob.glob(os.path.join(student_submission_folder, "*.py"))

    grades = joblib.Parallel(n_jobs=n_jobs,)(
        joblib.delayed(grade_a_student)
        (teacher_module_path, student_module, hw)
        for student_module in student_modules)

    return grades
    
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
            # TODO: lines to add before student and teacher .py
            [   (numpy.array([1,2]), numpy.array([3,4]), "test.png"), # case 1
                (numpy.array([3,4]), numpy.array([5,6]), "test.pdf")  # case 2
            ], 
            "grading_policy": "partial",
            "comparitor":ndarray_tuple_comparitor

        }
    ]

    teacher_module_path = 'alan_turing.py'
    # student_module_path = 'hw1.py'

    # grade = grade_a_student(teacher_module_path, student_module_path, hw)
    # print (grade)

    student_submission_folder = "grading_test"
    grades = grade_all_students(teacher_module_path, student_submission_folder, hw )

    # print (grades)
