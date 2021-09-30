import argparse  
from code_part2 import time



def setup():
    parser = argparse.ArgumentParser()  
    parser.add_argument("--X", default="", type=str, help = "Read content from the file")
    parser.add_argument("--Y", default="", type=str, help = "test file")
    return parser.parse_args()
    

if __name__ == '__main__':
    args = setup()
    time.solve(args)