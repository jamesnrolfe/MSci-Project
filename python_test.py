import numpy as np

arr = np.zeros[7]
arr[1,4,6] = 1

def zerotest(arr,k):
    test = False
    test2 = False
    length = len(arr)
    for i in arr:
        for j in k:
            if arr[i] == 0 and arr[(j+1)%length] == 0:
                test2 = True
    test = test2

    

    return test

def main():
    print(zerotest(arr,1))

main()


