import sys

def test_file(path1, path2):
    with open(path1, 'r') as f:
        x = f.read().split()
    with open(path2, 'r') as f:
        y = f.read().split()

    for a in x:
        a = float(a)
    for a in y:
        a = float(a)
    val = 0
    for i in range(len(x)):
        a = float(x[i])
        b = float(y[i])
        val = val + (a-b)**2
    print(val/len(x))

if __name__ == '__main__':
    test_file(sys.argv[1], sys.argv[2])