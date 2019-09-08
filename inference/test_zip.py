import sys

def test_file(path):
    with open(path, 'r') as f:
        x = f.read().split()

    for a in x:
        a = float(a)
    print("success")

if __name__ == '__main__':
    test_file(sys.argv[1])